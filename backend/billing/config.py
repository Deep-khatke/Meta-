"""
RoboForge Billing & PayPal Configuration
Defines plan specifications, operation credit pricing, and environment settings.
"""

from __future__ import annotations
import os
from typing import Dict, Any

# Environment & PayPal Credentials
PAYPAL_ENV: str = os.getenv("PAYPAL_ENV", "sandbox").strip().lower()
PAYPAL_CLIENT_ID: str = os.getenv(
    "PAYPAL_CLIENT_ID",
    "BAARXgyIwc813ICUv3wOEA_iF5T3ngyOGDVk-fEPz4im16dEZoYhKaYHMI8ZXG1g_VygwMmCP73XlPPe6U",
).strip()
PAYPAL_CLIENT_SECRET: str = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
PAYPAL_WEBHOOK_ID: str = os.getenv("PAYPAL_WEBHOOK_ID", "").strip()

PAYPAL_PRODUCT_ID: str = os.getenv("PAYPAL_PRODUCT_ID", "").strip()
PAYPAL_PLAN_PRO_ID: str = os.getenv("PAYPAL_PLAN_PRO_ID", "").strip()
PAYPAL_PLAN_PRO_PLUS_ID: str = os.getenv("PAYPAL_PLAN_PRO_PLUS_ID", "").strip()
PAYPAL_PLAN_PRO_MAX_ID: str = os.getenv("PAYPAL_PLAN_PRO_MAX_ID", "").strip()

APP_PUBLIC_URL: str = os.getenv("APP_PUBLIC_URL", "http://localhost:8000").rstrip("/")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")

# PayPal API Base URLs
PAYPAL_BASE_URLS = {
    "sandbox": "https://api-m.sandbox.paypal.com",
    "live": "https://api-m.paypal.com",
}


def get_paypal_base_url() -> str:
    return PAYPAL_BASE_URLS.get(PAYPAL_ENV, "https://api-m.sandbox.paypal.com")


# Plan Definitions (All amounts in USD, currency USD, exact credit amounts)
PLANS_CONFIG: Dict[str, Dict[str, Any]] = {
    "free": {
        "slug": "free",
        "name": "RoboForge Free",
        "description": "Essential simulation compute and basic robot development for students and hobbyists.",
        "price_cents": 0,
        "price_usd": 0.00,
        "currency": "USD",
        "credits_per_period": 50,
        "billing_interval": "month",
        "interval_count": 1,
        "paypal_plan_id": None,
    },
    "pro": {
        "slug": "pro",
        "name": "RoboForge Pro",
        "description": "Professional engineering compute: 1,000 credits/mo for live RL training, kinematics synthesis, and sensor simulation.",
        "price_cents": 1000,
        "price_usd": 10.00,
        "currency": "USD",
        "credits_per_period": 1000,
        "billing_interval": "month",
        "interval_count": 1,
        "paypal_plan_id_env": "PAYPAL_PLAN_PRO_ID",
    },
    "pro_plus": {
        "slug": "pro_plus",
        "name": "RoboForge Pro+",
        "description": "High-throughput robotics research: 2,000 credits/mo with accelerated multi-worker RL training and synthetic data pipelines.",
        "price_cents": 2000,
        "price_usd": 20.00,
        "currency": "USD",
        "credits_per_period": 2000,
        "billing_interval": "month",
        "interval_count": 1,
        "paypal_plan_id_env": "PAYPAL_PLAN_PRO_PLUS_ID",
    },
    "pro_max": {
        "slug": "pro_max",
        "name": "RoboForge Pro Max",
        "description": "Enterprise robotics suite: 5,000 credits/mo for large-scale Isaac Lab curricula, swarm orchestration, and advanced AI reasoning.",
        "price_cents": 4000,
        "price_usd": 40.00,
        "currency": "USD",
        "credits_per_period": 5000,
        "billing_interval": "month",
        "interval_count": 1,
        "paypal_plan_id_env": "PAYPAL_PLAN_PRO_MAX_ID",
    },
}


def get_plan_paypal_id(plan_slug: str) -> str | None:
    """Returns the configured PayPal Plan ID for a given paid tier."""
    if plan_slug == "pro":
        return os.getenv("PAYPAL_PLAN_PRO_ID", "").strip() or None
    elif plan_slug == "pro_plus":
        return os.getenv("PAYPAL_PLAN_PRO_PLUS_ID", "").strip() or None
    elif plan_slug == "pro_max":
        return os.getenv("PAYPAL_PLAN_PRO_MAX_ID", "").strip() or None
    return None


# Server-Side Operation Credit Pricing Map
# Configurable pricing for billable AI, simulation, and robotics compute tasks.
OPERATION_PRICING: Dict[str, int] = {
    "ai_chat": 2,                      # Natural language robot reasoning and mission suggestions
    "ai_interpret_prompt": 2,          # Autonomous mission code & scene generation
    "ai_refine_hypothesis": 3,         # Scientific hypothesis analysis
    "rl_train_live": 10,               # Live interactive Isaac Lab / MuJoCo RL training run
    "rl_train_background": 25,         # Full background multi-worker RL training job
    "motion_planning": 1,              # Collision-free path planning
    "synthetic_data_export": 5,        # Synthetic perception / COCO dataset rendering
    "experiment_multi_trial": 5,       # Multi-trial batch robotic experiment
}
