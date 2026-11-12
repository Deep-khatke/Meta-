"""
RoboForge PayPal Sandbox Setup CLI
Manually invoked command to create or reuse:
1. One RoboForge catalog product
2. Three monthly recurring billing plans: Pro ($10), Pro+ ($20), Pro Max ($40)

Run via:
    python -m backend.billing.setup_paypal
"""

from __future__ import annotations
import os
import sys
import logging
from backend.billing.config import (
    PAYPAL_ENV,
    PAYPAL_CLIENT_ID,
    PAYPAL_CLIENT_SECRET,
    PAYPAL_PRODUCT_ID,
    PAYPAL_PLAN_PRO_ID,
    PAYPAL_PLAN_PRO_PLUS_ID,
    PAYPAL_PLAN_PRO_MAX_ID,
    PLANS_CONFIG,
)
from backend.billing.paypal_service import paypal_service, PayPalConfigurationError, PayPalAPIError
import backend.billing.db as bdb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("roboforge.paypal.setup")


def run_setup() -> int:
    """Executes the idempotent PayPal Sandbox product and plans setup."""
    logger.info("=========================================================")
    logger.info("  RoboForge PayPal Sandbox Subscriptions Setup")
    logger.info("=========================================================")
    logger.info("Environment: %s", PAYPAL_ENV)
    logger.info("Client ID:   %s", PAYPAL_CLIENT_ID[:12] + "..." + PAYPAL_CLIENT_ID[-6:])

    # 1. Check credentials
    if not paypal_service.is_configured:
        logger.error("\n[!] PAYPAL_CLIENT_SECRET is missing or set to placeholder.")
        logger.error("Please add your secret to .env in the repository root:")
        logger.error("    PAYPAL_CLIENT_SECRET=<your_sandbox_secret>\n")
        logger.error("Sandbox setup is paused until credentials are provided.")
        return 1

    # Ensure DB tables are initialized
    bdb.init_billing_db()

    # 2. Test OAuth authentication
    try:
        logger.info("\nConnecting to PayPal Sandbox OAuth...")
        token = paypal_service.get_access_token()
        logger.info("[✓] OAuth Authentication Successful.")
    except Exception as e:
        logger.error("[✗] OAuth Authentication Failed: %s", e)
        return 1

    # 3. Create or reuse Product
    product_id = PAYPAL_PRODUCT_ID or None
    try:
        logger.info("\nChecking / Creating RoboForge Platform Product...")
        product = paypal_service.create_or_get_product(
            name="RoboForge Platform",
            description="Cloud physics simulation, RL compute, and robot kinematics platform",
            product_id_override=product_id,
        )
        product_id = product["id"]
        logger.info("[✓] Product ID: %s", product_id)
    except Exception as e:
        logger.error("[✗] Product Creation Failed: %s", e)
        return 1

    # 4. Create or reuse 3 monthly plans
    plan_overrides = {
        "pro": PAYPAL_PLAN_PRO_ID or None,
        "pro_plus": PAYPAL_PLAN_PRO_PLUS_ID or None,
        "pro_max": PAYPAL_PLAN_PRO_MAX_ID or None,
    }

    created_plan_ids = {}

    for slug in ("pro", "pro_plus", "pro_max"):
        cfg = PLANS_CONFIG[slug]
        logger.info("\nChecking / Creating Plan for %s ($%s/mo)...", cfg["name"], cfg["price_usd"])
        try:
            plan_res = paypal_service.create_or_get_billing_plan(
                product_id=product_id,
                name=cfg["name"],
                description=cfg["description"],
                price_usd=cfg["price_usd"],
                plan_id_override=plan_overrides.get(slug),
            )
            pid = plan_res["id"]
            created_plan_ids[slug] = pid
            # Update local DB plan catalog
            bdb.update_plan_paypal_id(slug, pid)
            logger.info("[✓] %s Plan ID: %s", cfg["name"], pid)
        except Exception as e:
            logger.error("[✗] Failed to create plan %s: %s", slug, e)
            return 1

    # 5. Output instructions
    logger.info("\n=========================================================")
    logger.info("  SETUP SUCCESSFUL!")
    logger.info("=========================================================")
    logger.info("Update your .env file with these IDs:\n")
    logger.info("PAYPAL_PRODUCT_ID=%s", product_id)
    logger.info("PAYPAL_PLAN_PRO_ID=%s", created_plan_ids.get("pro", ""))
    logger.info("PAYPAL_PLAN_PRO_PLUS_ID=%s", created_plan_ids.get("pro_plus", ""))
    logger.info("PAYPAL_PLAN_PRO_MAX_ID=%s", created_plan_ids.get("pro_max", ""))
    logger.info("\nDatabase plan catalog updated. Re-running this command is safe and idempotent.")
    return 0


if __name__ == "__main__":
    sys.exit(run_setup())

