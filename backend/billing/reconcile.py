"""
RoboForge Subscription & Webhook Reconciliation Engine
Allows manual or scheduled verification of pending subscriptions, missed webhooks,
and authoritative payment settlement from PayPal Sandbox API.
"""

from __future__ import annotations
import json
import logging
import sys
import time
from typing import Dict, Any, List
import backend.billing.db as bdb
from backend.billing.config import PLANS_CONFIG
from backend.billing.paypal_service import paypal_service
from backend.billing.credit_service import credit_service
from backend.billing.webhook_handler import parse_iso_time, handle_webhook_event
from backend.database import get_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("roboforge.billing.reconcile")


def reconcile_subscription(sub_id_or_paypal_id: str) -> Dict[str, Any]:
    """
    Authoritatively queries PayPal API for a subscription's current state and reconciles
    it against local database records, crediting any settled payment not yet recorded.
    """
    sub = bdb.get_subscription_by_id(sub_id_or_paypal_id) or bdb.get_subscription_by_paypal_id(sub_id_or_paypal_id)
    if not sub:
        return {"status": "error", "message": f"Subscription '{sub_id_or_paypal_id}' not found locally."}

    paypal_sub_id = sub["paypal_subscription_id"]
    if not paypal_sub_id:
        return {"status": "error", "message": "Subscription has no associated PayPal subscription ID."}

    now = time.time()
    try:
        remote_data = paypal_service.get_subscription(paypal_sub_id)
    except Exception as e:
        logger.error("Failed to query PayPal for subscription %s: %s", paypal_sub_id, e)
        return {"status": "error", "message": str(e)}

    remote_status = remote_data.get("status", "").upper()
    billing_info = remote_data.get("billing_info", {})
    last_payment = billing_info.get("last_payment", {})
    next_billing_iso = billing_info.get("next_billing_time")

    period_start = parse_iso_time(remote_data.get("start_time")) or sub["current_period_start"] or now
    period_end = parse_iso_time(next_billing_iso) or (period_start + 30 * 86400.0)

    # Map PayPal status to local status
    status_map = {
        "ACTIVE": "active",
        "APPROVAL_PENDING": "pending",
        "APPROVED": "pending",
        "SUSPENDED": "suspended",
        "CANCELLED": "cancelled",
        "EXPIRED": "expired",
    }
    local_status = status_map.get(remote_status, "pending")

    # Update subscription record
    updated_sub = bdb.update_subscription(
        sub_id=sub["id"],
        status=local_status,
        current_period_start=period_start,
        current_period_end=period_end,
        metadata={"reconciled_at": now, "remote_status": remote_status},
    )

    # Check if a confirmed payment exists on PayPal that hasn't been credited yet
    credited_payment = False
    payment_capture_id = None
    if last_payment and remote_status == "ACTIVE":
        # In PayPal Subscriptions API, last_payment has amount and time
        # The transaction ID or payment ID can be extracted or derived from payment time
        last_pay_amount = float(last_payment.get("amount", {}).get("value", 0.0))
        last_pay_time_iso = last_payment.get("time")
        last_pay_time = parse_iso_time(last_pay_time_iso) or now

        # Create a deterministic payment capture ID if none explicitly provided
        payment_capture_id = last_payment.get("id") or f"pp_pay_{paypal_sub_id}_{int(last_pay_time)}"

        # Check if already recorded
        existing_payment = bdb.get_payment_by_capture_id(payment_capture_id)
        if not existing_payment:
            plan = PLANS_CONFIG.get(sub["plan_slug"])
            if plan:
                amount_cents = int(round(last_pay_amount * 100)) or plan["price_cents"]
                bdb.record_payment(
                    payment_id=f"pay_{payment_capture_id}",
                    user_id=sub["user_id"],
                    subscription_id=sub["id"],
                    paypal_capture_id=payment_capture_id,
                    amount_cents=amount_cents,
                    currency=plan["currency"],
                    status="completed",
                    payment_time=last_pay_time,
                    raw_data=last_payment,
                )
                credit_service.grant_paid_subscription_credits(
                    user_id=sub["user_id"],
                    subscription_id=sub["id"],
                    paypal_capture_id=payment_capture_id,
                    plan_slug=sub["plan_slug"],
                    credits_amount=plan["credits_per_period"],
                    period_start=period_start,
                    period_end=period_end,
                )
                credited_payment = True

    return {
        "status": "reconciled",
        "subscription": updated_sub,
        "remote_status": remote_status,
        "credited_payment": credited_payment,
        "capture_id": payment_capture_id,
    }


def reconcile_all_active_subscriptions() -> Dict[str, Any]:
    """Reconciles all pending and active subscriptions in the database."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, paypal_subscription_id, status FROM billing_subscriptions
            WHERE status IN ('pending', 'active')
        """).fetchall()

    results = []
    for row in rows:
        sub_id = row["id"]
        res = reconcile_subscription(sub_id)
        results.append({"sub_id": sub_id, "result": res})

    return {"reconciled_count": len(results), "details": results}


def reprocess_failed_webhooks() -> Dict[str, Any]:
    """Re-attempts processing for failed or received webhooks."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT event_id, payload_json FROM webhook_events
            WHERE status IN ('failed', 'received')
            ORDER BY created_at ASC
        """).fetchall()

    processed_count = 0
    failed_count = 0
    for row in rows:
        event_id = row["event_id"]
        try:
            payload = json.loads(row["payload_json"])
            handle_webhook_event(payload)
            processed_count += 1
        except Exception as e:
            logger.error("Failed re-processing event %s: %s", event_id, e)
            failed_count += 1

    return {"reprocessed": processed_count, "still_failed": failed_count}


if __name__ == "__main__":
    logger.info("Starting RoboForge billing reconciliation...")
    res_subs = reconcile_all_active_subscriptions()
    logger.info("Subscriptions reconciliation: %s", json.dumps(res_subs, indent=2))
    res_wh = reprocess_failed_webhooks()
    logger.info("Webhooks reconciliation: %s", json.dumps(res_wh, indent=2))

