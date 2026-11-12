"""
RoboForge Centralized Credit Service
Implements atomic reservations, FIFO usage deductions, monthly UTC free anniversary grants,
paid subscription entitlement, expiry policies, and refund/reversal adjustments.
"""

from __future__ import annotations
import math
import time
import uuid
import logging
from typing import Dict, Any, Optional, Tuple, List
from backend.database import get_db, get_user_by_id
from backend.billing.config import OPERATION_PRICING, PLANS_CONFIG
import backend.billing.db as bdb

logger = logging.getLogger("roboforge.credit")


class InsufficientCreditsError(Exception):
    """Raised when user does not have enough available credits for an operation."""
    def __init__(self, required: int, available: int, message: Optional[str] = None):
        self.required = required
        self.available = available
        self.message = message or f"Insufficient credits: {required} required, but only {available} available."
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "insufficient_credits",
            "required_credits": self.required,
            "available_credits": self.available,
            "message": self.message,
        }


class CreditService:
    """Centralized credit allocation, reservation, and ledger service."""

    def __init__(self, reservation_ttl_seconds: int = 1800):
        # Default 30 minutes TTL for reservations
        self._reservation_ttl = reservation_ttl_seconds

    # ---------------- Anniversary Free Grants ----------------

    def ensure_free_anniversary_grant(self, user_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Idempotently creates the 50-credit free grant for the user's current UTC
        account-anniversary period, if the user is on the free plan.
        Does NOT grant free credits if an active paid subscription exists.
        """
        now = now or time.time()
        user = get_user_by_id(user_id)
        if not user:
            return None

        # Check if user has an active paid subscription or unexpired paid grant
        active_sub = bdb.get_active_subscription_for_user(user_id)
        if active_sub and active_sub["status"] in ("active", "pending") and active_sub["plan_slug"] != "free":
            # Paid users do not receive free credits while paid access is current
            return None

        latest_sub = bdb.get_latest_subscription_for_user(user_id)
        if latest_sub and latest_sub["plan_slug"] != "free":
            if latest_sub["status"] in ("active", "pending"):
                return None
            if latest_sub.get("current_period_end") and latest_sub["current_period_end"] > now:
                return None

        with get_db() as conn:
            paid_grant = conn.execute("""
                SELECT 1 FROM credit_grants
                WHERE user_id = ? AND source_type = 'subscription_payment'
                  AND period_end > ?
                LIMIT 1
            """, (user_id, now)).fetchone()
            if paid_grant:
                return None

        user_created_at = float(user.get("created_at") or now)
        # 30-day billing cycle calculation from user anniversary
        cycle_duration = 30 * 86400.0
        cycles_elapsed = math.floor(max(0.0, now - user_created_at) / cycle_duration)
        period_start = user_created_at + (cycles_elapsed * cycle_duration)
        period_end = period_start + cycle_duration

        # Source ID format: free:{user_id}:{period_start_int} enforces uniqueness
        source_id = f"free:{user_id}:{int(period_start)}"

        # Check if already granted
        existing = bdb.get_credit_grant_by_source_id(source_id)
        if existing:
            return existing

        grant_id = f"grnt_free_{uuid.uuid4().hex[:10]}"
        amount = 50

        try:
            with get_db() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                # Double check inside transaction
                row = conn.execute("SELECT * FROM credit_grants WHERE source_id = ?", (source_id,)).fetchone()
                if row:
                    return dict(row)

                conn.execute("""
                    INSERT INTO credit_grants (
                        id, user_id, subscription_id, source_type, source_id,
                        amount, used_amount, period_start, period_end, expires_at, is_expired, created_at
                    ) VALUES (?, ?, NULL, 'free_anniversary', ?, ?, 0, ?, ?, ?, 0, ?)
                """, (grant_id, user_id, source_id, amount, period_start, period_end, period_end, now))

                # Calculate new balance after this grant
                active_sum_row = conn.execute("""
                    SELECT COALESCE(SUM(amount - used_amount), 0) AS total
                    FROM credit_grants WHERE user_id = ? AND is_expired = 0 AND expires_at > ?
                """, (user_id, now)).fetchone()
                balance_after = int(active_sum_row["total"]) if active_sum_row else amount

                ledger_id = f"ledg_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO credit_ledger (
                        id, user_id, grant_id, amount, balance_after,
                        entry_type, operation_id, description, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'free_anniversary', NULL, 'Monthly free tier allowance (50 credits)', ?)
                """, (ledger_id, user_id, grant_id, amount, balance_after, now))
                conn.commit()

            logger.info("Granted 50 free credits to user %s for period ending %s", user_id, period_end)
            return bdb.get_credit_grant_by_id(grant_id)
        except Exception as e:
            logger.warning("Could not create free grant (possibly concurrent grant): %s", e)
            return bdb.get_credit_grant_by_source_id(source_id)

    # ---------------- Paid Subscription Grants ----------------

    def grant_paid_subscription_credits(
        self,
        user_id: str,
        subscription_id: str,
        paypal_capture_id: str,
        plan_slug: str,
        credits_amount: int,
        period_start: float,
        period_end: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Grants paid credits for a confirmed subscription payment.
        Guaranteed idempotent by unique constraint on source_id = payment:{paypal_capture_id}.
        On the first confirmed payment, expires remaining free-plan credits so balances do not stack.
        """
        now = time.time()
        source_id = f"payment:{paypal_capture_id}"

        existing = bdb.get_credit_grant_by_source_id(source_id)
        if existing:
            logger.info("Payment %s already credited. Skipping duplicate grant.", paypal_capture_id)
            return existing

        grant_id = f"grnt_sub_{uuid.uuid4().hex[:10]}"

        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Recheck inside lock
            row = conn.execute("SELECT * FROM credit_grants WHERE source_id = ?", (source_id,)).fetchone()
            if row:
                return dict(row)

            # Check if this is the first paid payment for this user
            prev_paid_grant = conn.execute("""
                SELECT 1 FROM credit_grants
                WHERE user_id = ? AND source_type = 'subscription_payment'
                LIMIT 1
            """, (user_id,)).fetchone()

            if not prev_paid_grant:
                # Expire any active free grants so free and paid allowances don't unintentionally stack
                cur = conn.execute("""
                    UPDATE credit_grants SET is_expired = 1
                    WHERE user_id = ? AND source_type = 'free_anniversary' AND is_expired = 0
                """, (user_id,))
                if cur.rowcount > 0:
                    ledger_exp_id = f"ledg_exp_{uuid.uuid4().hex[:10]}"
                    conn.execute("""
                        INSERT INTO credit_ledger (
                            id, user_id, grant_id, amount, balance_after,
                            entry_type, operation_id, description, created_at
                        ) VALUES (?, ?, NULL, 0, 0, 'expiry', NULL, 'Superseded free tier allowance on paid upgrade', ?)
                    """, (ledger_exp_id, user_id, now))

            # Insert the paid grant
            conn.execute("""
                INSERT INTO credit_grants (
                    id, user_id, subscription_id, source_type, source_id,
                    amount, used_amount, period_start, period_end, expires_at, is_expired, created_at
                ) VALUES (?, ?, ?, 'subscription_payment', ?, ?, 0, ?, ?, ?, 0, ?)
            """, (
                grant_id, user_id, subscription_id, source_id,
                credits_amount, period_start, period_end, period_end, now
            ))

            # Compute balance after
            active_sum_row = conn.execute("""
                SELECT COALESCE(SUM(amount - used_amount), 0) AS total
                FROM credit_grants WHERE user_id = ? AND is_expired = 0 AND expires_at > ?
            """, (user_id, now)).fetchone()
            balance_after = int(active_sum_row["total"]) if active_sum_row else credits_amount

            # Append to ledger
            ledger_id = f"ledg_pay_{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO credit_ledger (
                    id, user_id, grant_id, amount, balance_after,
                    entry_type, operation_id, description, created_at
                ) VALUES (?, ?, ?, ?, ?, 'grant', NULL, ?, ?)
            """, (
                ledger_id, user_id, grant_id, credits_amount, balance_after,
                f"Paid subscription allowance ({plan_slug}: {credits_amount} credits)", now
            ))
            conn.commit()

        logger.info(
            "Granted %d credits to user %s for paid subscription %s (payment %s)",
            credits_amount, user_id, subscription_id, paypal_capture_id
        )
        return bdb.get_credit_grant_by_id(grant_id)

    # ---------------- Credit Summary & Balances ----------------

    def get_user_credit_summary(self, user_id: str, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Returns full credit summary for authenticated user:
        - available credits
        - reserved credits
        - period_end timestamp
        - active plan details
        """
        now = now or time.time()

        # Maintenance: expire overdue grants and stale reservations
        bdb.expire_grants_past(now)
        bdb.expire_stale_reservations(now)

        # Ensure free grant if user is on free tier
        self.ensure_free_anniversary_grant(user_id, now)

        with get_db() as conn:
            # Active grants sum
            grant_row = conn.execute("""
                SELECT
                    COALESCE(SUM(amount - used_amount), 0) AS total_granted,
                    MAX(expires_at) AS latest_expiry
                FROM credit_grants
                WHERE user_id = ? AND is_expired = 0 AND expires_at > ? AND used_amount < amount
            """, (user_id, now)).fetchone()

            total_granted = int(grant_row["total_granted"]) if grant_row else 0
            latest_expiry = float(grant_row["latest_expiry"]) if grant_row and grant_row["latest_expiry"] else (now + 30 * 86400)

            # Active reservations sum
            res_row = conn.execute("""
                SELECT COALESCE(SUM(amount), 0) AS total_reserved
                FROM credit_reservations
                WHERE user_id = ? AND status = 'reserved' AND expires_at > ?
            """, (user_id, now)).fetchone()

            total_reserved = int(res_row["total_reserved"]) if res_row else 0

        available = max(0, total_granted - total_reserved)
        return {
            "available": available,
            "reserved": total_reserved,
            "period_end": latest_expiry,
        }

    # ---------------- Reservations (Two-Phase Spend) ----------------

    def reserveCredits(
        self,
        user_id: str,
        operation_id: str,
        amount: int,
        operation_type: str = "generic",
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Atomically checks available credits and creates a reservation.
        Prevents concurrent overspending across async operations.
        """
        now = now or time.time()
        if amount <= 0:
            raise ValueError("Reservation credit amount must be greater than zero.")

        # Maintenance
        bdb.expire_grants_past(now)
        bdb.expire_stale_reservations(now)
        self.ensure_free_anniversary_grant(user_id, now)

        expires_at = now + self._reservation_ttl

        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE;")

            # Check if this operation_id already has a reservation
            existing = conn.execute(
                "SELECT * FROM credit_reservations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()

            if existing:
                if existing["status"] == "reserved":
                    return dict(existing)
                elif existing["status"] == "committed":
                    raise ValueError(f"Operation {operation_id} was already committed.")
                elif existing["status"] in ("released", "expired"):
                    raise ValueError(f"Operation {operation_id} was already {existing['status']}.")

            # Compute current available balance
            grant_row = conn.execute("""
                SELECT COALESCE(SUM(amount - used_amount), 0) AS total_granted
                FROM credit_grants
                WHERE user_id = ? AND is_expired = 0 AND expires_at > ? AND used_amount < amount
            """, (user_id, now)).fetchone()
            total_granted = int(grant_row["total_granted"]) if grant_row else 0

            res_row = conn.execute("""
                SELECT COALESCE(SUM(amount), 0) AS total_reserved
                FROM credit_reservations
                WHERE user_id = ? AND status = 'reserved' AND expires_at > ?
            """, (user_id, now)).fetchone()
            total_reserved = int(res_row["total_reserved"]) if res_row else 0

            available = max(0, total_granted - total_reserved)
            if available < amount:
                raise InsufficientCreditsError(
                    required=amount,
                    available=available,
                    message=f"Insufficient credits for this operation. Required: {amount}, Available: {available}."
                )

            # Insert reservation atomically
            conn.execute("""
                INSERT INTO credit_reservations (
                    operation_id, user_id, amount, status, operation_type, expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'reserved', ?, ?, ?, ?)
            """, (operation_id, user_id, amount, operation_type, expires_at, now, now))
            conn.commit()

        logger.debug("Reserved %d credits for user %s on op %s", amount, user_id, operation_id)
        return {
            "operation_id": operation_id,
            "user_id": user_id,
            "amount": amount,
            "status": "reserved",
            "operation_type": operation_type,
            "expires_at": expires_at,
        }

    def commitCredits(self, operation_id: str, now: Optional[float] = None) -> bool:
        """
        Commits a reserved operation, deducting credits via FIFO from active grants
        and recording the debit in the append-only ledger.
        """
        now = now or time.time()
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            res = conn.execute(
                "SELECT * FROM credit_reservations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()

            if not res or res["status"] != "reserved":
                logger.info("Cannot commit op %s: not in reserved state", operation_id)
                return False

            user_id = res["user_id"]
            amount_needed = int(res["amount"])

            # Find active unexpired grants ordered FIFO by expires_at ASC
            grants = conn.execute("""
                SELECT * FROM credit_grants
                WHERE user_id = ? AND is_expired = 0 AND expires_at > ? AND used_amount < amount
                ORDER BY expires_at ASC, created_at ASC
            """, (user_id, now)).fetchall()

            remaining_to_deduct = amount_needed
            for g in grants:
                avail_on_grant = g["amount"] - g["used_amount"]
                deduct_here = min(remaining_to_deduct, avail_on_grant)
                new_used = g["used_amount"] + deduct_here
                is_now_expired = 1 if new_used >= g["amount"] else g["is_expired"]

                conn.execute("""
                    UPDATE credit_grants
                    SET used_amount = ?, is_expired = ?
                    WHERE id = ?
                """, (new_used, is_now_expired, g["id"]))

                remaining_to_deduct -= deduct_here
                if remaining_to_deduct <= 0:
                    break

            # Calculate new balance after deduction
            active_sum_row = conn.execute("""
                SELECT COALESCE(SUM(amount - used_amount), 0) AS total
                FROM credit_grants WHERE user_id = ? AND is_expired = 0 AND expires_at > ?
            """, (user_id, now)).fetchone()
            balance_after = int(active_sum_row["total"]) if active_sum_row else 0

            # Record in append-only ledger
            ledger_id = f"ledg_use_{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO credit_ledger (
                    id, user_id, grant_id, amount, balance_after,
                    entry_type, operation_id, description, created_at
                ) VALUES (?, ?, NULL, ?, ?, 'usage', ?, ?, ?)
            """, (
                ledger_id, user_id, -amount_needed, balance_after,
                operation_id, f"Executed {res['operation_type']} ({amount_needed} credits)", now
            ))

            # Mark reservation as committed
            conn.execute("""
                UPDATE credit_reservations
                SET status = 'committed', updated_at = ?
                WHERE operation_id = ?
            """, (now, operation_id))
            conn.commit()

        logger.info("Committed %d credits for operation %s (user %s)", amount_needed, operation_id, user_id)
        return True

    def releaseCredits(self, operation_id: str, now: Optional[float] = None) -> bool:
        """
        Releases a held reservation upon operation failure or cancellation,
        returning the credits to the user's available pool.
        """
        now = now or time.time()
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            res = conn.execute(
                "SELECT * FROM credit_reservations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()

            if not res or res["status"] != "reserved":
                return False

            conn.execute("""
                UPDATE credit_reservations
                SET status = 'released', updated_at = ?
                WHERE operation_id = ?
            """, (now, operation_id))
            conn.commit()

        logger.info("Released reservation for operation %s (user %s)", operation_id, res["user_id"])
        return True

    # ---------------- Refund & Reversal Adjustments ----------------

    def handle_payment_refund(
        self,
        user_id: str,
        paypal_capture_id: str,
        refund_amount_cents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Adjusts the corresponding credit grant when a payment is refunded or reversed.
        If the credits were already spent, records the shortfall in the ledger and flags for review.
        """
        now = time.time()
        source_id = f"payment:{paypal_capture_id}"
        grant = bdb.get_credit_grant_by_source_id(source_id)

        if not grant:
            logger.warning("No credit grant found for refunded payment %s", paypal_capture_id)
            return {"status": "grant_not_found"}

        grant_total = grant["amount"]
        grant_used = grant["used_amount"]
        unspent_on_grant = max(0, grant_total - grant_used)
        shortfall = grant_total - unspent_on_grant

        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Expire the grant completely
            conn.execute("""
                UPDATE credit_grants
                SET is_expired = 1, used_amount = amount
                WHERE id = ?
            """, (grant["id"],))

            # Recalculate balance
            active_sum_row = conn.execute("""
                SELECT COALESCE(SUM(amount - used_amount), 0) AS total
                FROM credit_grants WHERE user_id = ? AND is_expired = 0 AND expires_at > ?
            """, (user_id, now)).fetchone()
            balance_after = int(active_sum_row["total"]) if active_sum_row else 0

            # Append refund adjustment to ledger
            ledger_id = f"ledg_ref_{uuid.uuid4().hex[:12]}"
            desc = f"Refund adjustment for payment {paypal_capture_id}."
            if shortfall > 0:
                desc += f" Shortfall: {shortfall} credits were already spent (flagged for review)."

            conn.execute("""
                INSERT INTO credit_ledger (
                    id, user_id, grant_id, amount, balance_after,
                    entry_type, operation_id, description, created_at
                ) VALUES (?, ?, ?, ?, ?, 'refund_adjustment', NULL, ?, ?)
            """, (
                ledger_id, user_id, grant["id"], -unspent_on_grant, balance_after, desc, now
            ))
            conn.commit()

        logger.info(
            "Refund processed for payment %s (user %s): revoked %d unspent, shortfall %d",
            paypal_capture_id, user_id, unspent_on_grant, shortfall
        )
        return {
            "status": "adjusted",
            "revoked_credits": unspent_on_grant,
            "shortfall": shortfall,
        }


# Shared Singleton Instance
credit_service = CreditService()
