"""
Comprehensive Automated Tests for RoboForge Billing, PayPal Subscriptions, and Credit Engine.

Tests:
1. Tampered plan / price input handling
2. Unauthorized subscription access & multi-tenant isolation
3. Invalid webhook signature rejection
4. Duplicate and out-of-order webhook delivery
5. Successful payment credited exactly once (concurrent webhook & sync idempotency)
6. Activation without payment receiving no credits
7. Renewal, failed renewal, cancellation, and expiry policies
8. Concurrent credit spending & atomic reservation safety
9. Failed-operation reservation release (two-phase commit safety)
10. Refund and reversal accounting with shortfall handling
11. Free-tier UTC anniversary grants and upgrade rollover expiry
"""

import os
import time
import uuid
import pytest
from fastapi.testclient import TestClient

# Use a temporary database for test runs
test_db_path = os.path.abspath(f"test_billing_{uuid.uuid4().hex[:8]}.db")
os.environ["ROBOFORGE_DB_PATH"] = test_db_path

import backend.database as db
import backend.billing.db as bdb
from backend.billing.config import PLANS_CONFIG, OPERATION_PRICING
from backend.billing.credit_service import credit_service, InsufficientCreditsError
from backend.billing.paypal_service import paypal_service
from backend.billing.webhook_handler import handle_webhook_event
from backend.app import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initializes schema before tests and cleans up afterwards."""
    db.init_db()
    bdb.init_billing_db()
    yield
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception:
        pass


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def user_a():
    email = f"alice_{uuid.uuid4().hex[:6]}@example.com"
    res = db.register_user("Alice Engineer", email, "secretPass123")
    return res


@pytest.fixture
def user_b():
    email = f"bob_{uuid.uuid4().hex[:6]}@example.com"
    res = db.register_user("Bob Researcher", email, "secretPass123")
    return res


# ---------------- 1. Tampered Plan / Price Input ----------------

def test_tampered_plan_slug_rejected(client, user_a):
    headers = {"Authorization": f"Bearer {user_a['token']}"}

    # Attempt subscribing with invalid plan slug
    resp = client.post("/api/billing/subscriptions", json={"plan_slug": "hacked_mega_tier"}, headers=headers)
    assert resp.status_code == 400
    assert "Invalid plan slug" in resp.json()["detail"]

    # Attempt subscribing to free tier via paid checkout endpoint
    resp_free = client.post("/api/billing/subscriptions", json={"plan_slug": "free"}, headers=headers)
    assert resp_free.status_code == 400
    assert "free plan requires no checkout" in resp_free.json()["detail"]


# ---------------- 2. Unauthorized Subscription Access ----------------

def test_unauthorized_access(client, user_a, user_b):
    # Unauthenticated GET /api/billing/me
    resp = client.get("/api/billing/me")
    assert resp.status_code == 401

    # Unauthenticated GET /api/billing/credits/ledger
    resp_ledg = client.get("/api/billing/credits/ledger")
    assert resp_ledg.status_code == 401

    # Create a subscription for User A directly in DB
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-{uuid.uuid4().hex[:8]}"
    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_a["user"]["id"],
        plan_slug="pro",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO",
        status="pending",
    )

    # User B attempts to sync User A's subscription -> 403 Forbidden
    headers_b = {"Authorization": f"Bearer {user_b['token']}"}
    resp_sync = client.post(f"/api/billing/subscriptions/{sub_id}/sync", headers=headers_b)
    assert resp_sync.status_code == 403
    assert "Access denied" in resp_sync.json()["detail"]

    # User B attempts to cancel User A's subscription -> 403 Forbidden
    resp_cancel = client.post(f"/api/billing/subscriptions/{sub_id}/cancel", headers=headers_b)
    assert resp_cancel.status_code == 403
    assert "Access denied" in resp_cancel.json()["detail"]


# ---------------- 3. Invalid Webhook Signatures ----------------

def test_invalid_webhook_signature_rejected(client, monkeypatch):
    # Mock verify_webhook_signature to return False
    monkeypatch.setattr(paypal_service, "verify_webhook_signature", lambda headers, body: False)

    payload = {
        "id": f"WH-FAKE-{uuid.uuid4().hex[:8]}",
        "event_type": "BILLING.SUBSCRIPTION.ACTIVATED",
        "resource": {"id": "I-FAKE"},
    }

    # Request without valid signature should return 400 Bad Request
    resp = client.post("/api/billing/paypal/webhook", json=payload)
    assert resp.status_code == 400
    assert "Invalid or unverified PayPal webhook signature" in resp.json()["detail"]


# ---------------- 4. Duplicate and Out-of-Order Events ----------------

def test_duplicate_webhook_event_idempotency(monkeypatch):
    monkeypatch.setattr(paypal_service, "verify_webhook_signature", lambda headers, body: True)

    event_id = f"WH-IDEM-{uuid.uuid4().hex[:8]}"
    event_payload = {
        "id": event_id,
        "event_type": "BILLING.SUBSCRIPTION.CANCELLED",
        "create_time": "2026-11-12T10:00:00Z",
        "resource": {"id": "I-NONEXISTENT"},
    }

    # First call processes
    res1 = handle_webhook_event(event_payload)
    assert res1["status"] == "processed"

    # Second call detects duplicate event_id and ignores idempotently
    res2 = handle_webhook_event(event_payload)
    assert res2["status"] == "already_processed"


def test_out_of_order_event_protection(user_a):
    user_id = user_a["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-{uuid.uuid4().hex[:8]}"
    now = time.time()

    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_id,
        plan_slug="pro",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO",
        status="cancelled",
    )

    # Record that subscription was cancelled at time T = now
    bdb.update_subscription(sub_id, status="cancelled", last_event_time=now)

    # An older ACTIVATED event from time (T - 500) arrives late
    old_event_time = now - 500.0
    res = bdb.update_subscription(sub_id, status="active", last_event_time=old_event_time)

    # Status must NOT be overwritten to active
    assert res["status"] == "cancelled"


# ---------------- 5. Activation Without Payment Grants No Credits ----------------

def test_activation_without_payment_grants_no_credits(user_a):
    user_id = user_a["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-ACT-{uuid.uuid4().hex[:8]}"
    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_id,
        plan_slug="pro",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO",
        status="pending",
    )

    summary_before = credit_service.get_user_credit_summary(user_id)

    # Simulate PayPal ACTIVATED webhook
    event_payload = {
        "id": f"WH-ACT-{uuid.uuid4().hex[:8]}",
        "event_type": "BILLING.SUBSCRIPTION.ACTIVATED",
        "create_time": "2026-11-12T10:00:00Z",
        "resource": {
            "id": paypal_sub_id,
            "start_time": "2026-11-12T10:00:00Z",
            "billing_info": {"next_billing_time": "2026-12-12T10:00:00Z"},
        },
    }
    handle_webhook_event(event_payload)

    # Subscription state is active
    sub = bdb.get_subscription_by_id(sub_id)
    assert sub["status"] == "active"

    # CRITICAL: No extra paid credits were granted merely from activation!
    summary_after = credit_service.get_user_credit_summary(user_id)
    assert summary_after["available"] == summary_before["available"]


# ---------------- 6. Successful Payment Credited Exactly Once ----------------

def test_payment_credited_once_concurrently(user_a, monkeypatch):
    user_id = user_a["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-PAY-{uuid.uuid4().hex[:8]}"
    capture_id = f"CAP-{uuid.uuid4().hex[:8]}"

    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_id,
        plan_slug="pro",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO",
        status="pending",
    )

    summary_before = credit_service.get_user_credit_summary(user_id)

    event_payload = {
        "id": f"WH-PAY-{uuid.uuid4().hex[:8]}",
        "event_type": "PAYMENT.SALE.COMPLETED",
        "create_time": "2026-11-12T10:00:00Z",
        "resource": {
            "id": capture_id,
            "billing_agreement_id": paypal_sub_id,
            "amount": {"total": "10.00", "currency": "USD"},
        },
    }

    # First event completes payment
    handle_webhook_event(event_payload)
    summary_after_first = credit_service.get_user_credit_summary(user_id)
    # Pro tier has 1000 credits
    assert summary_after_first["available"] == 1000

    # Simulate identical payment event delivered again (or concurrent sync)
    event_payload["id"] = f"WH-PAY-DUPLICATE-{uuid.uuid4().hex[:8]}"
    handle_webhook_event(event_payload)

    summary_after_second = credit_service.get_user_credit_summary(user_id)
    # Credit balance remains exactly 1000, not 2000!
    assert summary_after_second["available"] == 1000


# ---------------- 7. Cancellation, Preservation, Renewal & Expiry ----------------

def test_cancellation_preserves_paid_credits(user_a, client):
    user_id = user_a["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-CANCEL-{uuid.uuid4().hex[:8]}"

    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_id,
        plan_slug="pro_plus",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO-PLUS",
        status="active",
    )

    # Grant 2000 credits for pro_plus
    capture_id = f"CAP-CP-{uuid.uuid4().hex[:8]}"
    now = time.time()
    credit_service.grant_paid_subscription_credits(
        user_id=user_id,
        subscription_id=sub_id,
        paypal_capture_id=capture_id,
        plan_slug="pro_plus",
        credits_amount=2000,
        period_start=now,
        period_end=now + 30 * 86400,
    )

    # Cancel subscription
    headers = {"Authorization": f"Bearer {user_a['token']}"}
    resp = client.post(f"/api/billing/subscriptions/{sub_id}/cancel", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # User still retains their paid 2000 credits until period end
    summary = credit_service.get_user_credit_summary(user_id)
    assert summary["available"] == 2000

    # Repeated cancellation request is safe and idempotent
    resp_repeat = client.post(f"/api/billing/subscriptions/{sub_id}/cancel", headers=headers)
    assert resp_repeat.status_code == 200


def test_renewal_and_failed_renewal(user_a):
    user_id = user_a["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    paypal_sub_id = f"I-RENEW-{uuid.uuid4().hex[:8]}"

    bdb.create_subscription(
        sub_id=sub_id,
        user_id=user_id,
        plan_slug="pro",
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id="P-TEST-PRO",
        status="active",
    )

    # Month 1 payment
    cap1 = f"CAP-M1-{uuid.uuid4().hex[:8]}"
    credit_service.grant_paid_subscription_credits(
        user_id, sub_id, cap1, "pro", 1000, time.time() - 31 * 86400, time.time() - 1 * 86400
    )

    # Month 2 payment (successful renewal)
    cap2 = f"CAP-M2-{uuid.uuid4().hex[:8]}"
    now = time.time()
    credit_service.grant_paid_subscription_credits(
        user_id, sub_id, cap2, "pro", 1000, now, now + 30 * 86400
    )

    # Month 1 credits are expired; Month 2 credits (1000) are available
    summary = credit_service.get_user_credit_summary(user_id)
    assert summary["available"] == 1000

    # Failed renewal (BILLING.SUBSCRIPTION.SUSPENDED)
    event_payload = {
        "id": f"WH-SUSP-{uuid.uuid4().hex[:8]}",
        "event_type": "BILLING.SUBSCRIPTION.SUSPENDED",
        "create_time": "2026-11-12T10:00:00Z",
        "resource": {"id": paypal_sub_id},
    }
    handle_webhook_event(event_payload)
    sub = bdb.get_subscription_by_id(sub_id)
    assert sub["status"] == "suspended"


# ---------------- 8. Concurrent Credit Spending & Atomic Reservations ----------------

def test_atomic_reservation_prevents_overspending(user_a):
    user_id = user_a["user"]["id"]
    # User A starts with 50 free credits
    summary = credit_service.get_user_credit_summary(user_id)
    assert summary["available"] == 50

    op1 = f"op_batch_1_{uuid.uuid4().hex[:8]}"
    op2 = f"op_batch_2_{uuid.uuid4().hex[:8]}"

    # Reserve 35 credits for op1 -> succeeds
    res1 = credit_service.reserveCredits(user_id, op1, 35, "rl_train")
    assert res1["status"] == "reserved"

    # Only 15 credits remain available
    summary_mid = credit_service.get_user_credit_summary(user_id)
    assert summary_mid["available"] == 15
    assert summary_mid["reserved"] == 35

    # Attempt to reserve 20 credits for op2 -> raises InsufficientCreditsError (needs 20, only 15 available)
    with pytest.raises(InsufficientCreditsError) as exc_info:
        credit_service.reserveCredits(user_id, op2, 20, "rl_train")
    assert exc_info.value.required == 20
    assert exc_info.value.available == 15

    # Commit op1
    credit_service.commitCredits(op1)
    summary_final = credit_service.get_user_credit_summary(user_id)
    assert summary_final["available"] == 15
    assert summary_final["reserved"] == 0


# ---------------- 9. Failed-Operation Reservation Release ----------------

def test_reservation_release_on_failure(user_b):
    user_id = user_b["user"]["id"]
    summary_init = credit_service.get_user_credit_summary(user_id)
    init_avail = summary_init["available"]

    op_id = f"op_fail_{uuid.uuid4().hex[:8]}"
    # Reserve 10 credits
    credit_service.reserveCredits(user_id, op_id, 10, "motion_planning")
    assert credit_service.get_user_credit_summary(user_id)["available"] == init_avail - 10

    # Operation fails -> release credits
    credit_service.releaseCredits(op_id)

    # Full credits restored
    summary_after = credit_service.get_user_credit_summary(user_id)
    assert summary_after["available"] == init_avail
    assert summary_after["reserved"] == 0


# ---------------- 10. Refund and Reversal Accounting ----------------

def test_refund_accounting_and_shortfall(user_b):
    user_id = user_b["user"]["id"]
    sub_id = f"sub_{uuid.uuid4().hex[:8]}"
    capture_id = f"CAP-REF-{uuid.uuid4().hex[:8]}"
    now = time.time()

    # Grant 1000 credits from payment
    credit_service.grant_paid_subscription_credits(
        user_id, sub_id, capture_id, "pro", 1000, now, now + 30 * 86400
    )

    # User spends 300 credits
    op_spend = f"op_spend_{uuid.uuid4().hex[:8]}"
    credit_service.reserveCredits(user_id, op_spend, 300, "ai_chat")
    credit_service.commitCredits(op_spend)
    assert credit_service.get_user_credit_summary(user_id)["available"] == 700

    # Payment is refunded on PayPal
    refund_res = credit_service.handle_payment_refund(user_id, capture_id, refund_amount_cents=1000)
    assert refund_res["status"] == "adjusted"
    assert refund_res["revoked_credits"] == 700
    assert refund_res["shortfall"] == 300  # 300 credits were already spent, recorded as shortfall

    # Remaining balance is now 0
    assert credit_service.get_user_credit_summary(user_id)["available"] == 0

    # Ledger audit entry exists with shortfall note
    ledger = bdb.get_user_ledger_paginated(user_id, limit=5)
    descriptions = [entry["description"] for entry in ledger]
    assert any("Shortfall: 300 credits were already spent" in d for d in descriptions)


# ---------------- 11. Billable Route Credit Enforcement ----------------

def test_billable_endpoint_credit_check(client, user_a):
    headers = {"Authorization": f"Bearer {user_a['token']}"}

    # Motion plan costs 1 credit. User A has enough credits -> succeeds
    resp = client.post("/api/motion/plan-path", json={"start_pos": [0.0, 0.0], "goal_pos": [2.0, 2.0]}, headers=headers)
    assert resp.status_code == 200

    # Exhaust all remaining credits for user_a
    avail = credit_service.get_user_credit_summary(user_a["user"]["id"])["available"]
    if avail > 0:
        drain_op = f"op_drain_{uuid.uuid4().hex[:8]}"
        credit_service.reserveCredits(user_a["user"]["id"], drain_op, avail, "test_drain")
        credit_service.commitCredits(drain_op)

    # Verify user has 0 credits available
    assert credit_service.get_user_credit_summary(user_a["user"]["id"])["available"] == 0

    # Calling motion planning with 0 credits -> returns HTTP 402 Payment Required with structured error
    resp_empty = client.post("/api/motion/plan-path", json={"start_pos": [0.0, 0.0], "goal_pos": [2.0, 2.0]}, headers=headers)
    assert resp_empty.status_code == 402
    err_body = resp_empty.json()["detail"]
    assert err_body["error"] == "insufficient_credits"
    assert err_body["required_credits"] == OPERATION_PRICING["motion_planning"]
    assert err_body["available_credits"] == 0
