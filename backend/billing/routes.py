"""
RoboForge Billing API Routes
Implements all billing endpoints according to the API contract:
- GET  /api/billing/plans
- GET  /api/billing/me
- POST /api/billing/subscriptions
- POST /api/billing/subscriptions/{id}/sync
- POST /api/billing/subscriptions/{id}/cancel
- GET  /api/billing/credits/ledger
- POST /api/billing/paypal/webhook
"""

from __future__ import annotations
import json
import logging
import uuid
import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request, Header, Query, status
from pydantic import BaseModel, Field

import backend.database as db
import backend.billing.db as bdb
from backend.billing.config import (
    PAYPAL_ENV,
    FRONTEND_URL,
    APP_PUBLIC_URL,
    PLANS_CONFIG,
    get_plan_paypal_id,
)
from backend.billing.paypal_service import paypal_service, PayPalAPIError, PayPalConfigurationError
from backend.billing.credit_service import credit_service
from backend.billing.webhook_handler import handle_webhook_event
from backend.billing.reconcile import reconcile_subscription

logger = logging.getLogger("roboforge.billing.api")

router = APIRouter(prefix="/api/billing", tags=["billing"])


# ---------------- Authentication Helper ----------------

def _extract_token(request: Request, authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.cookies.get("roboforge_token") or request.query_params.get("token")


def get_authenticated_user(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency to extract and verify the current logged-in user."""
    token = _extract_token(request, authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )
    user = db.get_user_by_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please sign in again.",
        )
    return user


# ---------------- Request Models ----------------

class CreateSubscriptionRequest(BaseModel):
    plan_slug: str = Field(..., description="Local plan slug ('pro', 'pro_plus', 'pro_max')")


# ---------------- 1. GET /api/billing/plans ----------------

@router.get("/plans")
def get_plans_catalog():
    """Public plan catalog with prices, credit allowances, and sandbox indicator."""
    plans = bdb.list_plans()
    # Format plans for public consumption
    formatted_plans = []
    for p in plans:
        formatted_plans.append({
            "slug": p["slug"],
            "name": p["name"],
            "description": p["description"],
            "price_cents": p["price_cents"],
            "price_usd": round(p["price_cents"] / 100.0, 2),
            "currency": p["currency"],
            "credits_per_period": p["credits_per_period"],
            "billing_interval": p["billing_interval"],
            "interval_count": p["interval_count"],
            "paypal_plan_id": p["paypal_plan_id"],
        })
    return {
        "plans": formatted_plans,
        "sandbox": PAYPAL_ENV == "sandbox",
    }


# ---------------- 2. GET /api/billing/me ----------------

@router.get("/me")
def get_my_billing_profile(current_user: Dict[str, Any] = Depends(get_authenticated_user)):
    """Authenticated user's current plan, subscription status, and credit balance."""
    user_id = current_user["id"]
    now = time.time()

    # Credit summary
    credit_summary = credit_service.get_user_credit_summary(user_id, now)

    # Active subscription check
    active_sub = bdb.get_active_subscription_for_user(user_id)
    latest_sub = active_sub or bdb.get_latest_subscription_for_user(user_id)

    current_plan_slug = "free"
    subscription_data = None

    if latest_sub:
        subscription_data = {
            "id": latest_sub["id"],
            "paypal_subscription_id": latest_sub["paypal_subscription_id"],
            "plan_slug": latest_sub["plan_slug"],
            "status": latest_sub["status"],
            "current_period_start": latest_sub["current_period_start"],
            "current_period_end": latest_sub["current_period_end"],
            "cancel_at_period_end": bool(latest_sub["cancel_at_period_end"]),
            "approval_url": latest_sub["approval_url"],
        }
        if latest_sub["status"] in ("active", "cancelled"):
            # If cancelled but period hasn't ended, user still has paid tier access
            current_plan_slug = latest_sub["plan_slug"]

    plan_info = PLANS_CONFIG.get(current_plan_slug, PLANS_CONFIG["free"])

    return {
        "user_id": user_id,
        "email": current_user["email"],
        "plan": {
            "slug": plan_info["slug"],
            "name": plan_info["name"],
            "credits_per_period": plan_info["credits_per_period"],
            "price_cents": plan_info["price_cents"],
        },
        "subscription": subscription_data,
        "credits": {
            "available": credit_summary["available"],
            "reserved": credit_summary["reserved"],
            "period_end": credit_summary["period_end"],
        },
    }


# ---------------- 3. POST /api/billing/subscriptions ----------------

@router.post("/subscriptions")
def create_subscription_endpoint(
    req: CreateSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """
    Initiates a new recurring subscription on PayPal for the authenticated user.
    Server resolves the plan ID and price. Prevents duplicate active paid subscriptions.
    """
    user_id = current_user["id"]
    plan_slug = req.plan_slug.strip().lower()

    if plan_slug == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The free plan requires no checkout. All RoboForge accounts automatically receive the free allowance.",
        )

    plan = PLANS_CONFIG.get(plan_slug)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan slug '{plan_slug}'. Supported tiers: pro, pro_plus, pro_max.",
        )

    # Resolve PayPal plan ID from database or config
    db_plan = bdb.get_plan(plan_slug)
    paypal_plan_id = (db_plan.get("paypal_plan_id") if db_plan else None) or get_plan_paypal_id(plan_slug)

    if not paypal_plan_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PayPal plan for '{plan_slug}' is not yet configured. Please run 'python -m backend.billing.setup_paypal'.",
        )

    # Check for existing active or pending subscription
    existing_sub = bdb.get_active_subscription_for_user(user_id)
    if existing_sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You already have an existing {existing_sub['status']} subscription ({existing_sub['plan_slug']}). Please cancel your current plan or wait until its billing period ends before subscribing to a new tier.",
        )

    local_sub_id = f"sub_{uuid.uuid4().hex[:12]}"
    idempotency_key = f"idem_sub_{local_sub_id}"

    # Construct Return and Cancel URLs
    base_fe_url = FRONTEND_URL or APP_PUBLIC_URL
    return_url = f"{base_fe_url}/billing/return?subscription_id={local_sub_id}"
    cancel_url = f"{base_fe_url}/billing/cancel?subscription_id={local_sub_id}"

    try:
        remote_res = paypal_service.create_subscription(
            plan_id=paypal_plan_id,
            custom_id=user_id,
            return_url=return_url,
            cancel_url=cancel_url,
            idempotency_key=idempotency_key,
        )
    except PayPalConfigurationError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except PayPalAPIError as e:
        logger.error("PayPal subscription creation failed: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"PayPal API error: {e}")

    paypal_sub_id = remote_res.get("id")
    approval_url = remote_res.get("approval_url")

    # Save pending subscription to database
    sub = bdb.create_subscription(
        sub_id=local_sub_id,
        user_id=user_id,
        plan_slug=plan_slug,
        paypal_sub_id=paypal_sub_id,
        paypal_plan_id=paypal_plan_id,
        status="pending",
        approval_url=approval_url,
        metadata={"created_via": "api_checkout"},
    )

    return {
        "status": "pending",
        "subscription_id": local_sub_id,
        "paypal_subscription_id": paypal_sub_id,
        "approval_url": approval_url,
        "plan_slug": plan_slug,
    }


# ---------------- 4. POST /api/billing/subscriptions/{id}/sync ----------------

@router.post("/subscriptions/{sub_id}/sync")
def sync_subscription_endpoint(
    sub_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """
    Authoritatively queries PayPal for subscription and payment settlement.
    Enforces ownership and prevents granting credits without verified payment.
    """
    user_id = current_user["id"]
    sub = bdb.get_subscription_by_id(sub_id) or bdb.get_subscription_by_paypal_id(sub_id)

    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found.")

    if sub["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: unauthorized subscription.")

    res = reconcile_subscription(sub["id"])
    if res.get("status") == "error":
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=res.get("message"))

    # Return updated credit and subscription state
    credit_summary = credit_service.get_user_credit_summary(user_id)
    return {
        "status": "ok",
        "subscription": res.get("subscription"),
        "remote_status": res.get("remote_status"),
        "credited_payment": res.get("credited_payment"),
        "credits": credit_summary,
    }


# ---------------- 5. POST /api/billing/subscriptions/{id}/cancel ----------------

@router.post("/subscriptions/{sub_id}/cancel")
def cancel_subscription_endpoint(
    sub_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """
    Cancels recurring subscription renewal with PayPal.
    Preserves already paid credits until existing billing period end.
    Safe for repeated requests.
    """
    user_id = current_user["id"]
    sub = bdb.get_subscription_by_id(sub_id) or bdb.get_subscription_by_paypal_id(sub_id)

    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found.")

    if sub["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: unauthorized subscription.")

    paypal_sub_id = sub["paypal_subscription_id"]
    if paypal_sub_id:
        try:
            paypal_service.cancel_subscription(paypal_sub_id, reason="User requested cancellation in RoboForge.")
        except Exception as e:
            logger.warning("PayPal cancel returned note for %s: %s", paypal_sub_id, e)

    now = time.time()
    updated_sub = bdb.update_subscription(
        sub_id=sub["id"],
        status="cancelled",
        cancel_at_period_end=True,
        cancelled_at=now,
        last_event_time=now,
    )

    return {
        "status": "cancelled",
        "subscription_id": sub["id"],
        "message": "Subscription cancelled. Your purchased credits remain active until the end of your billing period.",
        "subscription": updated_sub,
    }


# ---------------- 6. GET /api/billing/credits/ledger ----------------

@router.get("/credits/ledger")
def get_credit_ledger_endpoint(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
):
    """Authenticated, paginated ledger history for the current user only."""
    user_id = current_user["id"]
    offset = (page - 1) * limit

    items = bdb.get_user_ledger_paginated(user_id, limit=limit, offset=offset)
    total_count = bdb.get_user_ledger_count(user_id)

    return {
        "items": items,
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total_count + limit - 1) // limit),
    }


# ---------------- 7. POST /api/billing/paypal/webhook ----------------

@router.post("/paypal/webhook")
async def paypal_webhook_endpoint(request: Request):
    """
    Public webhook endpoint authenticated by PayPal signature verification.
    Rejects unverified payloads before any state or credits are modified.
    """
    headers = dict(request.headers)
    raw_body = await request.body()

    # Webhook signature verification
    is_valid = paypal_service.verify_webhook_signature(headers, raw_body)
    if not is_valid:
        logger.warning("Rejected unverified PayPal webhook request from %s", request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unverified PayPal webhook signature.",
        )

    try:
        event = json.loads(raw_body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body.")

    try:
        res = handle_webhook_event(event)
        return {"status": "ok", "result": res}
    except Exception as e:
        logger.exception("Error during webhook processing: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook event.",
        )

