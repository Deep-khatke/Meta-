"""
RoboForge PayPal Subscriptions & Webhook Service
Communicates with PayPal Sandbox v1 APIs for OAuth token retrieval,
catalogs, billing plans, subscription lifecycle, and webhook verification.
"""

from __future__ import annotations
import base64
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any
import httpx
from backend.billing.config import (
    PAYPAL_ENV,
    PAYPAL_CLIENT_ID,
    PAYPAL_CLIENT_SECRET,
    PAYPAL_WEBHOOK_ID,
    get_paypal_base_url,
)

logger = logging.getLogger("roboforge.paypal")


class PayPalError(Exception):
    """Base exception for PayPal service errors."""
    pass


class PayPalConfigurationError(PayPalError):
    """Raised when PayPal credentials or configurations are missing."""
    pass


class PayPalAPIError(PayPalError):
    """Raised when PayPal returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class PayPalService:
    """Manages PayPal Subscriptions and Webhook verification in sandbox mode."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self._client_id = client_id or PAYPAL_CLIENT_ID
        self._client_secret = client_secret or PAYPAL_CLIENT_SECRET
        self._webhook_id = webhook_id or PAYPAL_WEBHOOK_ID
        self._base_url = (base_url or get_paypal_base_url()).rstrip("/")
        self._timeout = timeout

        # In-memory token cache: (access_token, expires_at_timestamp)
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def is_configured(self) -> bool:
        """Returns True if client credentials are provided."""
        return bool(self._client_id and self._client_secret and self._client_secret != "PASTE_YOUR_SANDBOX_SECRET_HERE")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Retrieves an OAuth2 Bearer token from PayPal, using expiry-aware caching."""
        now = time.time()
        # Return cached token if valid for at least another 60 seconds
        if not force_refresh and self._cached_token and now < (self._token_expires_at - 60):
            return self._cached_token

        if not self.is_configured:
            raise PayPalConfigurationError(
                "PayPal credentials are not configured. Please set PAYPAL_CLIENT_SECRET in your backend environment."
            )

        auth_str = f"{self._client_id}:{self._client_secret}"
        encoded_auth = base64.b64encode(auth_str.encode("utf-8")).decode("ascii")

        url = f"{self._base_url}/v1/oauth2/token"
        headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=headers, data=data)
        except Exception as e:
            logger.error("Network error requesting PayPal OAuth token: %s", type(e).__name__)
            raise PayPalAPIError("Failed to connect to PayPal sandbox API. Please check your network connection.")

        if resp.status_code != 200:
            logger.error("PayPal OAuth failed with status code %s", resp.status_code)
            raise PayPalAPIError(
                f"PayPal authentication failed with status {resp.status_code}.",
                status_code=resp.status_code,
            )

        try:
            payload = resp.json()
            token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
            self._cached_token = token
            self._token_expires_at = now + expires_in
            return token
        except Exception as e:
            logger.error("Failed to parse PayPal OAuth response: %s", e)
            raise PayPalAPIError("Invalid response received from PayPal OAuth endpoint.")

    def _get_auth_headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if idempotency_key:
            headers["PayPal-Request-Id"] = idempotency_key
        return headers

    # ---------------- Catalogs & Products ----------------

    def create_or_get_product(
        self,
        name: str = "RoboForge Platform",
        description: str = "Physics simulation, RL compute, and kinematics platform",
        product_id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates or reuses a RoboForge catalog product."""
        # 1. Check if existing product ID is valid
        if product_id_override:
            try:
                headers = self._get_auth_headers()
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.get(f"{self._base_url}/v1/catalogs/products/{product_id_override}", headers=headers)
                    if resp.status_code == 200:
                        logger.info("Reusing existing PayPal product: %s", product_id_override)
                        return resp.json()
            except Exception as e:
                logger.warning("Could not retrieve product %s: %s", product_id_override, e)

        # 2. Create product
        url = f"{self._base_url}/v1/catalogs/products"
        idem_key = f"prod-{name.lower().replace(' ', '-')}"
        headers = self._get_auth_headers(idempotency_key=idem_key)
        payload = {
            "name": name,
            "description": description,
            "type": "SERVICE",
            "category": "SOFTWARE",
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code not in (200, 201):
            logger.error("Failed to create PayPal product: status %s, body: %s", resp.status_code, resp.text)
            raise PayPalAPIError(f"Failed to create PayPal product: {resp.status_code}", status_code=resp.status_code)

        return resp.json()

    # ---------------- Billing Plans ----------------

    def create_or_get_billing_plan(
        self,
        product_id: str,
        name: str,
        description: str,
        price_usd: float,
        plan_id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates or reuses a monthly recurring subscription plan in PayPal Sandbox."""
        if plan_id_override:
            try:
                headers = self._get_auth_headers()
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.get(f"{self._base_url}/v1/billing/plans/{plan_id_override}", headers=headers)
                    if resp.status_code == 200:
                        logger.info("Reusing existing PayPal billing plan: %s", plan_id_override)
                        return resp.json()
            except Exception as e:
                logger.warning("Could not retrieve plan %s: %s", plan_id_override, e)

        url = f"{self._base_url}/v1/billing/plans"
        idem_key = f"plan-{name.lower().replace(' ', '-')}-{int(price_usd)}"
        headers = self._get_auth_headers(idempotency_key=idem_key)

        payload = {
            "product_id": product_id,
            "name": name[:127],
            "description": description[:127],
            "status": "ACTIVE",
            "billing_cycles": [
                {
                    "frequency": {
                        "interval_unit": "MONTH",
                        "interval_count": 1,
                    },
                    "tenure_type": "REGULAR",
                    "sequence": 1,
                    "total_cycles": 0,  # Infinite recurrence until cancelled
                    "pricing_scheme": {
                        "fixed_price": {
                            "value": f"{price_usd:.2f}",
                            "currency_code": "USD",
                        }
                    },
                }
            ],
            "payment_preferences": {
                "auto_bill_outstanding": False,
                "setup_fee": {
                    "value": "0.00",
                    "currency_code": "USD",
                },
                "setup_fee_failure_action": "CANCEL",
                "payment_failure_threshold": 3,
            },
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code not in (200, 201):
            logger.error("Failed to create PayPal billing plan: %s %s", resp.status_code, resp.text)
            raise PayPalAPIError(f"Failed to create PayPal plan: {resp.status_code}", status_code=resp.status_code)

        return resp.json()

    # ---------------- Subscriptions Lifecycle ----------------

    def create_subscription(
        self,
        plan_id: str,
        custom_id: str,
        return_url: str,
        cancel_url: str,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a pending PayPal recurring subscription and returns approval URL and ID."""
        url = f"{self._base_url}/v1/billing/subscriptions"
        idem_key = idempotency_key or str(uuid.uuid4())
        headers = self._get_auth_headers(idempotency_key=idem_key)

        payload = {
            "plan_id": plan_id,
            "custom_id": custom_id,
            "application_context": {
                "brand_name": "RoboForge",
                "locale": "en-US",
                "shipping_preference": "NO_SHIPPING",
                "user_action": "SUBSCRIBE_NOW",
                "return_url": return_url,
                "cancel_url": cancel_url,
            },
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code not in (200, 201):
            logger.error("PayPal create subscription failed: %s %s", resp.status_code, resp.text)
            raise PayPalAPIError(
                f"Failed to initiate PayPal subscription ({resp.status_code}).",
                status_code=resp.status_code,
                details=resp.text,
            )

        data = resp.json()
        approval_url = None
        for link in data.get("links", []):
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break

        return {
            "id": data.get("id"),
            "status": data.get("status", "APPROVAL_PENDING"),
            "approval_url": approval_url,
            "raw": data,
        }

    def get_subscription(self, paypal_sub_id: str) -> Dict[str, Any]:
        """Retrieves authoritative subscription status from PayPal."""
        url = f"{self._base_url}/v1/billing/subscriptions/{paypal_sub_id}"
        headers = self._get_auth_headers()

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, headers=headers)

        if resp.status_code != 200:
            logger.error("Failed to get PayPal subscription %s: %s", paypal_sub_id, resp.status_code)
            raise PayPalAPIError(
                f"Failed to fetch subscription from PayPal ({resp.status_code}).",
                status_code=resp.status_code,
            )

        return resp.json()

    def cancel_subscription(self, paypal_sub_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancels future billing with PayPal for a subscription."""
        url = f"{self._base_url}/v1/billing/subscriptions/{paypal_sub_id}/cancel"
        headers = self._get_auth_headers()
        payload = {"reason": reason}

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)

        # 204 No Content is the expected success response from PayPal
        if resp.status_code == 204:
            return True

        # If already cancelled or inactive, handle safely
        if resp.status_code in (400, 422):
            logger.info("Subscription %s was already inactive/cancelled on PayPal", paypal_sub_id)
            return True

        logger.error("Failed to cancel PayPal subscription %s: %s %s", paypal_sub_id, resp.status_code, resp.text)
        raise PayPalAPIError(f"Failed to cancel subscription on PayPal ({resp.status_code}).", status_code=resp.status_code)

    # ---------------- Webhook Signature Verification ----------------

    def verify_webhook_signature(
        self,
        headers: Dict[str, str],
        raw_body: str | bytes,
        webhook_id_override: Optional[str] = None,
    ) -> bool:
        """
        Validates the PayPal webhook signature against PayPal's verification endpoint.
        Uses exact header casing or case-insensitive lookup.
        """
        wh_id = webhook_id_override or self._webhook_id or PAYPAL_WEBHOOK_ID
        if not wh_id:
            logger.warning("PayPal Webhook ID not configured. Webhook signature cannot be verified.")
            return False

        # Extract headers safely (case-insensitive)
        headers_lower = {k.lower(): v for k, v in headers.items()}
        auth_algo = headers_lower.get("paypal-auth-algo")
        cert_url = headers_lower.get("paypal-cert-url")
        transmission_id = headers_lower.get("paypal-transmission-id")
        transmission_sig = headers_lower.get("paypal-transmission-sig")
        transmission_time = headers_lower.get("paypal-transmission-time")

        if not all([auth_algo, cert_url, transmission_id, transmission_sig, transmission_time]):
            logger.warning("Missing required PayPal webhook signature headers")
            return False

        if isinstance(raw_body, bytes):
            raw_body_str = raw_body.decode("utf-8", errors="replace")
        else:
            raw_body_str = raw_body

        try:
            webhook_event = json.loads(raw_body_str)
        except Exception:
            logger.warning("Invalid JSON payload in PayPal webhook")
            return False

        url = f"{self._base_url}/v1/notifications/verify-webhook-signature"
        auth_headers = self._get_auth_headers()

        payload = {
            "auth_algo": auth_algo,
            "cert_url": cert_url,
            "transmission_id": transmission_id,
            "transmission_sig": transmission_sig,
            "transmission_time": transmission_time,
            "webhook_id": wh_id,
            "webhook_event": webhook_event,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=auth_headers, json=payload)
        except Exception as e:
            logger.error("Network error during webhook verification: %s", e)
            return False

        if resp.status_code != 200:
            logger.warning("PayPal webhook verification endpoint returned %s: %s", resp.status_code, resp.text)
            return False

        result = resp.json()
        verification_status = result.get("verification_status", "").upper()
        if verification_status == "SUCCESS":
            return True

        logger.warning("PayPal webhook verification status: %s", verification_status)
        return False


# Shared Singleton Instance
paypal_service = PayPalService()

