"""
RoboForge Billing Database Operations & Schema Migrations
Provides SQLite tables, transactions, and queries for plans, subscriptions,
payments, credit grants, append-only ledger, reservations, and webhooks.
"""

from __future__ import annotations
import sqlite3
import json
import time
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from backend.billing.config import PLANS_CONFIG, get_plan_paypal_id

logger = logging.getLogger("roboforge.billing.db")

DB_PATH = os.environ.get("ROBOFORGE_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "roboforge.db"))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_billing_db() -> None:
    """Creates billing tables and ensures default plans are populated."""
    with get_db() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1. Local Plan Catalog
        conn.execute("""
            CREATE TABLE IF NOT EXISTS billing_plans (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                price_cents INTEGER NOT NULL,
                currency TEXT NOT NULL DEFAULT 'USD',
                credits_per_period INTEGER NOT NULL,
                billing_interval TEXT NOT NULL DEFAULT 'month',
                interval_count INTEGER NOT NULL DEFAULT 1,
                paypal_plan_id TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # 2. PayPal Subscriptions Linked to Authenticated Users
        conn.execute("""
            CREATE TABLE IF NOT EXISTS billing_subscriptions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plan_slug TEXT NOT NULL,
                paypal_subscription_id TEXT UNIQUE,
                paypal_plan_id TEXT,
                status TEXT NOT NULL,
                current_period_start REAL,
                current_period_end REAL,
                cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
                cancelled_at REAL,
                approval_url TEXT,
                last_event_time REAL,
                metadata_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (plan_slug) REFERENCES billing_plans(slug)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subs_user_id ON billing_subscriptions(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subs_status ON billing_subscriptions(status);")

        # 3. Payment Records
        conn.execute("""
            CREATE TABLE IF NOT EXISTS billing_payments (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subscription_id TEXT,
                paypal_capture_id TEXT UNIQUE NOT NULL,
                amount_cents INTEGER NOT NULL,
                currency TEXT NOT NULL DEFAULT 'USD',
                status TEXT NOT NULL,
                payment_time REAL NOT NULL,
                raw_data_json TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (subscription_id) REFERENCES billing_subscriptions(id) ON DELETE SET NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_user_id ON billing_payments(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pay_sub_id ON billing_payments(subscription_id);")

        # 4. Credit Grants with Boundaries & Expiry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_grants (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subscription_id TEXT,
                source_type TEXT NOT NULL,
                source_id TEXT UNIQUE NOT NULL,
                amount INTEGER NOT NULL,
                used_amount INTEGER NOT NULL DEFAULT 0,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                expires_at REAL NOT NULL,
                is_expired INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (subscription_id) REFERENCES billing_subscriptions(id) ON DELETE SET NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_grants_user_exp ON credit_grants(user_id, is_expired);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_grants_expires_at ON credit_grants(expires_at);")

        # 5. Append-Only Credit Ledger
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_ledger (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                grant_id TEXT,
                amount INTEGER NOT NULL,
                balance_after INTEGER NOT NULL,
                entry_type TEXT NOT NULL,
                operation_id TEXT,
                description TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (grant_id) REFERENCES credit_grants(id) ON DELETE SET NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_user ON credit_ledger(user_id, created_at);")

        # 6. Credit Reservations for Asynchronous Operations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_reservations (
                operation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                status TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reservations_user ON credit_reservations(user_id, status);")

        # 7. Webhook Event Processing Records
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                resource_id TEXT,
                payload_json TEXT NOT NULL,
                error_message TEXT,
                created_at REAL NOT NULL,
                processed_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_webhook_status ON webhook_events(status);")

        # Seed / Upsert Local Plans from Config
        now = time.time()
        for slug, plan in PLANS_CONFIG.items():
            paypal_id = get_plan_paypal_id(slug)
            conn.execute("""
                INSERT INTO billing_plans (
                    slug, name, description, price_cents, currency,
                    credits_per_period, billing_interval, interval_count,
                    paypal_plan_id, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    price_cents = excluded.price_cents,
                    credits_per_period = excluded.credits_per_period,
                    paypal_plan_id = COALESCE(excluded.paypal_plan_id, billing_plans.paypal_plan_id),
                    updated_at = excluded.updated_at
            """, (
                slug,
                plan["name"],
                plan["description"],
                plan["price_cents"],
                plan["currency"],
                plan["credits_per_period"],
                plan["billing_interval"],
                plan["interval_count"],
                paypal_id,
                now,
                now,
            ))
        conn.commit()
    logger.info("RoboForge billing tables and plan catalog initialized.")


# ---------------- Plan Queries ----------------

def list_plans() -> List[Dict[str, Any]]:
    """Returns all active billing plans."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM billing_plans WHERE is_active = 1 ORDER BY price_cents ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_plan(slug: str) -> Optional[Dict[str, Any]]:
    """Finds a plan by its slug."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM billing_plans WHERE slug = ?", (slug,)).fetchone()
        return dict(row) if row else None


def update_plan_paypal_id(slug: str, paypal_plan_id: str) -> None:
    """Updates PayPal plan ID for a tier."""
    now = time.time()
    with get_db() as conn:
        conn.execute(
            "UPDATE billing_plans SET paypal_plan_id = ?, updated_at = ? WHERE slug = ?",
            (paypal_plan_id, now, slug),
        )
        conn.commit()


# ---------------- Subscription Queries ----------------

def create_subscription(
    sub_id: str,
    user_id: str,
    plan_slug: str,
    paypal_sub_id: Optional[str],
    paypal_plan_id: Optional[str],
    status: str,
    approval_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Creates a new local subscription record."""
    now = time.time()
    meta_str = json.dumps(metadata) if metadata else None
    with get_db() as conn:
        conn.execute("""
            INSERT INTO billing_subscriptions (
                id, user_id, plan_slug, paypal_subscription_id, paypal_plan_id,
                status, current_period_start, current_period_end, cancel_at_period_end,
                cancelled_at, approval_url, last_event_time, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL, ?, ?, ?, ?, ?)
        """, (
            sub_id, user_id, plan_slug, paypal_sub_id, paypal_plan_id,
            status, approval_url, now, meta_str, now, now
        ))
        conn.commit()
    return get_subscription_by_id(sub_id) or {}


def get_subscription_by_id(sub_id: str) -> Optional[Dict[str, Any]]:
    """Finds subscription by internal ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM billing_subscriptions WHERE id = ?", (sub_id,)).fetchone()
        return dict(row) if row else None


def get_subscription_by_paypal_id(paypal_sub_id: str) -> Optional[Dict[str, Any]]:
    """Finds subscription by PayPal subscription ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM billing_subscriptions WHERE paypal_subscription_id = ?",
            (paypal_sub_id,),
        ).fetchone()
        return dict(row) if row else None


def get_active_subscription_for_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Finds current active or pending paid subscription for a user."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM billing_subscriptions
            WHERE user_id = ? AND status IN ('active', 'pending')
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,)).fetchone()
        return dict(row) if row else None


def get_latest_subscription_for_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Finds latest subscription record for a user regardless of status."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM billing_subscriptions
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,)).fetchone()
        return dict(row) if row else None


def update_subscription(
    sub_id: str,
    status: Optional[str] = None,
    current_period_start: Optional[float] = None,
    current_period_end: Optional[float] = None,
    cancel_at_period_end: Optional[bool] = None,
    cancelled_at: Optional[float] = None,
    paypal_sub_id: Optional[str] = None,
    last_event_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Idempotently updates subscription details."""
    now = time.time()
    with get_db() as conn:
        existing = conn.execute("SELECT * FROM billing_subscriptions WHERE id = ?", (sub_id,)).fetchone()
        if not existing:
            return None

        # Out-of-order event check
        if last_event_time and existing["last_event_time"] and last_event_time < existing["last_event_time"]:
            logger.warning(
                "Ignoring out-of-order update for sub %s: incoming event_time %s < recorded %s",
                sub_id, last_event_time, existing["last_event_time"]
            )
            return dict(existing)

        new_status = status if status is not None else existing["status"]
        new_start = current_period_start if current_period_start is not None else existing["current_period_start"]
        new_end = current_period_end if current_period_end is not None else existing["current_period_end"]
        new_cancel_flag = int(cancel_at_period_end) if cancel_at_period_end is not None else existing["cancel_at_period_end"]
        new_cancelled_at = cancelled_at if cancelled_at is not None else existing["cancelled_at"]
        new_paypal_id = paypal_sub_id if paypal_sub_id is not None else existing["paypal_subscription_id"]
        new_event_time = max(last_event_time or 0.0, existing["last_event_time"] or 0.0)

        meta_json = existing["metadata_json"]
        if metadata is not None:
            prev_meta = json.loads(existing["metadata_json"]) if existing["metadata_json"] else {}
            prev_meta.update(metadata)
            meta_json = json.dumps(prev_meta)

        conn.execute("""
            UPDATE billing_subscriptions SET
                status = ?,
                current_period_start = ?,
                current_period_end = ?,
                cancel_at_period_end = ?,
                cancelled_at = ?,
                paypal_subscription_id = ?,
                last_event_time = ?,
                metadata_json = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            new_status, new_start, new_end, new_cancel_flag,
            new_cancelled_at, new_paypal_id, new_event_time, meta_json, now, sub_id
        ))
        conn.commit()
    return get_subscription_by_id(sub_id)


# ---------------- Payment Queries ----------------

def record_payment(
    payment_id: str,
    user_id: str,
    subscription_id: Optional[str],
    paypal_capture_id: str,
    amount_cents: int,
    currency: str,
    status: str,
    payment_time: float,
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Records a payment capture transaction idempotently."""
    existing = get_payment_by_capture_id(paypal_capture_id)
    if existing:
        return existing

    now = time.time()
    raw_str = json.dumps(raw_data) if raw_data else None
    with get_db() as conn:
        try:
            conn.execute("""
                INSERT INTO billing_payments (
                    id, user_id, subscription_id, paypal_capture_id,
                    amount_cents, currency, status, payment_time, raw_data_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                payment_id, user_id, subscription_id, paypal_capture_id,
                amount_cents, currency, status, payment_time, raw_str, now
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    return get_payment_by_capture_id(paypal_capture_id) or {}


def get_payment_by_capture_id(paypal_capture_id: str) -> Optional[Dict[str, Any]]:
    """Finds payment record by PayPal capture/transaction ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM billing_payments WHERE paypal_capture_id = ?",
            (paypal_capture_id,),
        ).fetchone()
        return dict(row) if row else None


def update_payment_status(paypal_capture_id: str, status: str) -> None:
    """Updates status of a payment (e.g. refunded or reversed)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE billing_payments SET status = ? WHERE paypal_capture_id = ?",
            (status, paypal_capture_id),
        )
        conn.commit()


# ---------------- Credit Grant Queries ----------------

def create_credit_grant(
    grant_id: str,
    user_id: str,
    subscription_id: Optional[str],
    source_type: str,
    source_id: str,
    amount: int,
    period_start: float,
    period_end: float,
    expires_at: float,
) -> Dict[str, Any]:
    """Inserts a credit grant. Unique constraint on source_id guarantees idempotency."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO credit_grants (
                id, user_id, subscription_id, source_type, source_id,
                amount, used_amount, period_start, period_end, expires_at, is_expired, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 0, ?)
        """, (
            grant_id, user_id, subscription_id, source_type, source_id,
            amount, period_start, period_end, expires_at, now
        ))
        conn.commit()
    row = get_credit_grant_by_id(grant_id)
    return row or {}


def get_credit_grant_by_id(grant_id: str) -> Optional[Dict[str, Any]]:
    """Finds grant by primary ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM credit_grants WHERE id = ?", (grant_id,)).fetchone()
        return dict(row) if row else None


def get_credit_grant_by_source_id(source_id: str) -> Optional[Dict[str, Any]]:
    """Finds grant by unique source identifier (idempotency check)."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM credit_grants WHERE source_id = ?", (source_id,)).fetchone()
        return dict(row) if row else None


def get_active_grants_for_user(user_id: str, now: float) -> List[Dict[str, Any]]:
    """Returns active, unexpired grants for user ordered FIFO by expires_at."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM credit_grants
            WHERE user_id = ? AND is_expired = 0 AND expires_at > ? AND used_amount < amount
            ORDER BY expires_at ASC, created_at ASC
        """, (user_id, now)).fetchall()
        return [dict(r) for r in rows]


def expire_user_free_grants(user_id: str) -> int:
    """Expires any existing free-plan grants when user upgrades to paid plan."""
    with get_db() as conn:
        cur = conn.execute("""
            UPDATE credit_grants SET is_expired = 1
            WHERE user_id = ? AND source_type = 'free_anniversary' AND is_expired = 0
        """, (user_id,))
        count = cur.rowcount
        conn.commit()
        return count


def expire_grants_past(now: float) -> int:
    """Marks grants past their expiration date as expired."""
    with get_db() as conn:
        cur = conn.execute("""
            UPDATE credit_grants SET is_expired = 1
            WHERE is_expired = 0 AND expires_at <= ?
        """, (now,))
        count = cur.rowcount
        conn.commit()
        return count


def update_grant_usage(grant_id: str, add_used: int, mark_expired: bool = False) -> None:
    """Updates used credit amount on a specific grant."""
    with get_db() as conn:
        conn.execute("""
            UPDATE credit_grants
            SET used_amount = used_amount + ?,
                is_expired = CASE WHEN ? THEN 1 ELSE is_expired END
            WHERE id = ?
        """, (add_used, 1 if mark_expired else 0, grant_id))
        conn.commit()


# ---------------- Credit Ledger Queries ----------------

def append_ledger_entry(
    entry_id: str,
    user_id: str,
    grant_id: Optional[str],
    amount: int,
    balance_after: int,
    entry_type: str,
    operation_id: Optional[str],
    description: str,
) -> Dict[str, Any]:
    """Appends an immutable record to the credit ledger."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO credit_ledger (
                id, user_id, grant_id, amount, balance_after,
                entry_type, operation_id, description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_id, user_id, grant_id, amount, balance_after,
            entry_type, operation_id, description, now
        ))
        conn.commit()
    return {
        "id": entry_id,
        "user_id": user_id,
        "grant_id": grant_id,
        "amount": amount,
        "balance_after": balance_after,
        "entry_type": entry_type,
        "operation_id": operation_id,
        "description": description,
        "created_at": now,
    }


def get_user_ledger_paginated(user_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """Returns paginated ledger entries for the authenticated user only."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM credit_ledger
            WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset)).fetchall()
        return [dict(r) for r in rows]


def get_user_ledger_count(user_id: str) -> int:
    """Returns total count of ledger entries for pagination."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM credit_ledger WHERE user_id = ?", (user_id,)).fetchone()
        return row["cnt"] if row else 0


# ---------------- Credit Reservation Queries ----------------

def create_reservation(
    operation_id: str,
    user_id: str,
    amount: int,
    operation_type: str,
    expires_at: float,
) -> Dict[str, Any]:
    """Creates a credit reservation to prevent concurrent overspending."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO credit_reservations (
                operation_id, user_id, amount, status, operation_type, expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, 'reserved', ?, ?, ?, ?)
        """, (operation_id, user_id, amount, operation_type, expires_at, now, now))
        conn.commit()
    return get_reservation(operation_id) or {}


def get_reservation(operation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves reservation by operation ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM credit_reservations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return dict(row) if row else None


def update_reservation_status(operation_id: str, status: str) -> None:
    """Updates reservation status ('committed', 'released', 'expired')."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            UPDATE credit_reservations
            SET status = ?, updated_at = ?
            WHERE operation_id = ?
        """, (status, now, operation_id))
        conn.commit()


def get_active_reserved_amount_for_user(user_id: str, now: float) -> int:
    """Returns total credits currently reserved by active pending operations."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(amount), 0) AS total_reserved
            FROM credit_reservations
            WHERE user_id = ? AND status = 'reserved' AND expires_at > ?
        """, (user_id, now)).fetchone()
        return int(row["total_reserved"]) if row else 0


def expire_stale_reservations(now: float) -> int:
    """Marks reservations past their TTL as expired, releasing hold."""
    with get_db() as conn:
        cur = conn.execute("""
            UPDATE credit_reservations
            SET status = 'expired', updated_at = ?
            WHERE status = 'reserved' AND expires_at <= ?
        """, (now, now))
        count = cur.rowcount
        conn.commit()
        return count


# ---------------- Webhook Event Queries ----------------

def record_webhook_event(
    event_id: str,
    event_type: str,
    status: str,
    resource_id: Optional[str],
    payload_json: str,
    error_message: Optional[str] = None,
) -> None:
    """Logs a webhook event."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO webhook_events (
                event_id, event_type, status, resource_id, payload_json, error_message, created_at, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(event_id) DO UPDATE SET
                status = excluded.status,
                error_message = excluded.error_message
        """, (event_id, event_type, status, resource_id, payload_json, error_message, now))
        conn.commit()


def get_webhook_event(event_id: str) -> Optional[Dict[str, Any]]:
    """Finds webhook event by PayPal event ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM webhook_events WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None


def update_webhook_event(event_id: str, status: str, error_message: Optional[str] = None) -> None:
    """Updates webhook processing status."""
    now = time.time()
    with get_db() as conn:
        conn.execute("""
            UPDATE webhook_events
            SET status = ?, error_message = ?, processed_at = ?
            WHERE event_id = ?
        """, (status, error_message, now, event_id))
        conn.commit()
