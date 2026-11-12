"""
RoboForge PayPal Webhook Event Processor
Parses verified webhook events, handles out-of-order delivery, applies idempotency checks,
and coordinates subscription activation and payment credit grants.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Dict, Any, Optional
import backend.billing.db as bdb
from backend.billing.config import PLANS_CONFIG
from backend.billing.credit_service import credit_service

logger = logging.getLogger("roboforge.billing.webhook")


def parse_iso_time(iso_str: Optional[str]) -> Optional[float]:
    """Safely converts an ISO 8601 timestamp string to a unix float."""
    if not iso_str:
        return None
    try:
        from datetime import datetime, timezone
        # Handle trailing 'Z' or offset
        iso_clean = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_clean)
        return dt.timestamp()
    except Exception:
        return None


def handle_webhook_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Authoritatively dispatches and processes verified PayPal webhook notifications.
    Guaranteed idempotent through unique event ID tracking in `webhook_events`.
    """
    event_id = event.get("id")
    event_type = event.get("event_type", "UNKNOWN")
    resource = event.get("resource", {})
    create_time_iso = event.get("create_time")
    event_time = parse_iso_time(create_time_iso) or time.time()

    if not event_id:
        return {"status": "error", "message": "Missing event id"}

    # 1. Idempotency Check
    existing_event = bdb.get_webhook_event(event_id)
    if existing_event and existing_event["status"] == "processed":
        logger.info("Webhook event %s already processed. Skipping.", event_id)
        return {"status": "already_processed", "event_id": event_id}

    # Record event receipt
    bdb.record_webhook_event(
        event_id=event_id,
        event_type=event_type,
        status="received",
        resource_id=resource.get("id"),
        payload_json=json.dumps(event),
    )

    try:
        # 2. Dispatch by Event Type
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            _handle_subscription_activated(resource, event_time)

        elif event_type == "PAYMENT.SALE.COMPLETED":
            _handle_payment_sale_completed(resource, event_time)

        elif event_type in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.SUSPENDED", "BILLING.SUBSCRIPTION.EXPIRED"):
            _handle_subscription_state_change(resource, event_type, event_time)

        elif event_type in ("PAYMENT.SALE.REFUNDED", "PAYMENT.SALE.REVERSED"):
            _handle_payment_refunded_or_reversed(resource, event_type, event_time)

        else:
            logger.info("Acknowledged non-billing event: %s", event_type)

        bdb.update_webhook_event(event_id, status="processed")
        return {"status": "processed", "event_type": event_type}

    except Exception as e:
        logger.exception("Error processing webhook %s (%s): %s", event_id, event_type, e)
        bdb.update_webhook_event(event_id, status="failed", error_message=str(e))
        # Re-raise so the endpoint can return a 500/retryable code if appropriate
        raise


def _handle_subscription_activated(resource: Dict[str, Any], event_time: float) -> None:
    """
    Handles BILLING.SUBSCRIPTION.ACTIVATED.
    CRITICAL: Activation alone DOES NOT grant credits.
    Only updates local subscription state to 'active' or 'pending_payment'.
    """
    paypal_sub_id = resource.get("id")
    if not paypal_sub_id:
        return

    sub = bdb.get_subscription_by_paypal_id(paypal_sub_id)
    if not sub:
        logger.warning("Subscription %s activated on PayPal but not found in local DB", paypal_sub_id)
        return

    # Extract period boundaries if available
    billing_info = resource.get("billing_info", {})
    next_billing_iso = billing_info.get("next_billing_time")
    period_start = parse_iso_time(resource.get("start_time")) or event_time
    period_end = parse_iso_time(next_billing_iso) or (period_start + 30 * 86400.0)

    bdb.update_subscription(
        sub_id=sub["id"],
        status="active",
        current_period_start=period_start,
        current_period_end=period_end,
        last_event_time=event_time,
        metadata={"activated_at": event_time},
    )
    logger.info("Subscription %s activated (credits pending payment confirmation).", paypal_sub_id)


def _handle_payment_sale_completed(resource: Dict[str, Any], event_time: float) -> None:
    """
    Handles PAYMENT.SALE.COMPLETED.
    This is the authoritative trigger for granting paid subscription credits.
    """
    capture_id = resource.get("id")
    paypal_sub_id = resource.get("billing_agreement_id")
    amount_data = resource.get("amount", {})
    total_val = float(amount_data.get("total", 0.0))
    currency = amount_data.get("currency", "USD")

    if not capture_id or not paypal_sub_id:
        logger.warning("Payment sale completed event missing capture_id or billing_agreement_id")
        return

    sub = bdb.get_subscription_by_paypal_id(paypal_sub_id)
    if not sub:
        logger.warning("Payment received for unknown subscription %s", paypal_sub_id)
        return

    user_id = sub["user_id"]
    plan_slug = sub["plan_slug"]
    plan = PLANS_CONFIG.get(plan_slug)

    if not plan:
        logger.error("Subscription %s has unknown plan slug %s", sub["id"], plan_slug)
        return

    # Check if this payment capture was already completed
    existing_pay = bdb.get_payment_by_capture_id(capture_id)
    if existing_pay and existing_pay.get("status") == "completed":
        logger.info("Payment capture %s has already been recorded. Skipping duplicate processing.", capture_id)
        return

    # Idempotently record the payment
    amount_cents = int(round(total_val * 100))
    bdb.record_payment(
        payment_id=f"pay_{capture_id}",
        user_id=user_id,
        subscription_id=sub["id"],
        paypal_capture_id=capture_id,
        amount_cents=amount_cents,
        currency=currency,
        status="completed",
        payment_time=event_time,
        raw_data=resource,
    )

    # Calculate period boundaries for this cycle
    period_start = event_time
    period_end = period_start + (30 * 86400.0)

    # Grant the paid credits (guaranteed idempotent by unique source_id)
    credit_service.grant_paid_subscription_credits(
        user_id=user_id,
        subscription_id=sub["id"],
        paypal_capture_id=capture_id,
        plan_slug=plan_slug,
        credits_amount=plan["credits_per_period"],
        period_start=period_start,
        period_end=period_end,
    )

    # Ensure subscription is marked active
    bdb.update_subscription(
        sub_id=sub["id"],
        status="active",
        current_period_start=period_start,
        current_period_end=period_end,
        last_event_time=event_time,
    )
    logger.info("Successfully processed payment %s and granted %d credits to user %s", capture_id, plan["credits_per_period"], user_id)


def _handle_subscription_state_change(resource: Dict[str, Any], event_type: str, event_time: float) -> None:
    """Handles cancellation, suspension, or expiration of a subscription."""
    paypal_sub_id = resource.get("id")
    if not paypal_sub_id:
        return

    sub = bdb.get_subscription_by_paypal_id(paypal_sub_id)
    if not sub:
        return

    if event_type == "BILLING.SUBSCRIPTION.CANCELLED":
        bdb.update_subscription(
            sub_id=sub["id"],
            status="cancelled",
            cancel_at_period_end=True,
            cancelled_at=event_time,
            last_event_time=event_time,
        )
        logger.info("Subscription %s cancelled. Existing paid credits remain active until period end.", paypal_sub_id)

    elif event_type == "BILLING.SUBSCRIPTION.SUSPENDED":
        bdb.update_subscription(
            sub_id=sub["id"],
            status="suspended",
            last_event_time=event_time,
        )
        logger.info("Subscription %s suspended on PayPal.", paypal_sub_id)

    elif event_type == "BILLING.SUBSCRIPTION.EXPIRED":
        bdb.update_subscription(
            sub_id=sub["id"],
            status="expired",
            last_event_time=event_time,
        )
        logger.info("Subscription %s expired on PayPal.", paypal_sub_id)


def _handle_payment_refunded_or_reversed(resource: Dict[str, Any], event_type: str, event_time: float) -> None:
    """Handles payment refund or reversal, adjusting grants and ledger."""
    # PayPal refund event resource contains parent_payment or id
    parent_payment_id = resource.get("parent_payment") or resource.get("id")
    refund_amount_data = resource.get("amount", {})
    refund_val = float(refund_amount_data.get("total", 0.0))
    refund_cents = int(round(refund_val * 100))

    if not parent_payment_id:
        return

    # Check if we have this payment
    payment = bdb.get_payment_by_capture_id(parent_payment_id)
    if not payment:
        logger.warning("Refund received for untracked payment %s", parent_payment_id)
        return

    user_id = payment["user_id"]
    status_label = "refunded" if event_type == "PAYMENT.SALE.REFUNDED" else "reversed"

    bdb.update_payment_status(parent_payment_id, status=status_label)
    credit_service.handle_payment_refund(user_id=user_id, paypal_capture_id=parent_payment_id, refund_amount_cents=refund_cents)
    logger.info("Recorded %s for payment %s (user %s)", status_label, parent_payment_id, user_id)
