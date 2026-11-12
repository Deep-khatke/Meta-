# 💳 RoboForge PayPal Subscriptions & Credit Engine Documentation

RoboForge features a built-in recurring subscription and credit system powered by official **PayPal Subscriptions (REST API v1)** in sandbox mode and an append-only SQLite credit ledger.

---

## 1. Plan Catalog & Pricing

All subscriptions are monthly recurring subscriptions in **USD**. Paid plans require explicit approval through PayPal's hosted checkout. The Free tier requires no checkout and is provisioned automatically.

| Plan Slug | Name | Monthly Price | Credits / Period | PayPal Checkout Required | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `free` | RoboForge Free | $0.00 / month | 50 credits | No | Baseline robotics simulation, URDF editing, and 50 free credits every UTC account month. |
| `pro` | RoboForge Pro | $10.00 / month | 1,000 credits | Yes (Recurring) | For active robotics engineers running interactive AI mission planning and PPO training. |
| `pro_plus` | RoboForge Pro+ | $20.00 / month | 2,000 credits | Yes (Recurring) | High-volume simulation, synthetic dataset exports, and extended RL swarm training. |
| `pro_max` | RoboForge Pro Max | $40.00 / month | 5,000 credits | Yes (Recurring) | Unlimited research compute, heavy synthetic vision generation, and multi-agent training. |

*Note: There are no setup fees, trial periods, or automatic overage charges.*

---

## 2. Configuration & Secrets

Configuration template is available in [.env.example](file:///c:/college/Meta--main/.env.example).

### Exact Secret Location
> [!IMPORTANT]
> Paste your PayPal sandbox client secret in:
> **`.env`** in the repository root (`c:\college\Meta--main\.env`) or as the environment variable **`PAYPAL_CLIENT_SECRET`** in your cloud deployment settings (AWS EC2 / systemd / Docker).
> 
> **Never commit this secret to git, log it, or expose it in frontend code.**

Required backend environment variables:
```dotenv
PAYPAL_ENV=sandbox
PAYPAL_CLIENT_ID=BAARXgyIwc813ICUv3wOEA_iF5T3ngyOGDVk-fEPz4im16dEZoYhKaYHMI8ZXG1g_VygwMmCP73XlPPe6U
PAYPAL_CLIENT_SECRET=PASTE_YOUR_SANDBOX_SECRET_HERE
PAYPAL_WEBHOOK_ID=PASTE_WEBHOOK_ID_HERE
PAYPAL_PRODUCT_ID=
PAYPAL_PLAN_PRO_ID=
PAYPAL_PLAN_PRO_PLUS_ID=
PAYPAL_PLAN_PRO_MAX_ID=
APP_PUBLIC_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

---

## 3. PayPal Webhook Configuration

### Webhook URL
Configure your PayPal sandbox webhook listener endpoint in the [PayPal Developer Dashboard](https://developer.paypal.com/dashboard/applications/sandbox):
```
https://<YOUR_DOMAIN>/api/billing/paypal/webhook
```
*(For local testing, use an HTTPS reverse tunnel like ngrok, cloudflared, or DuckDNS).*

### Webhook Events to Register
Register the following 8 subscription and payment events:
1. `BILLING.SUBSCRIPTION.ACTIVATED` — Subscription activated by user. *(Note: activation alone marks status active, but does NOT grant credits until payment completes).*
2. `PAYMENT.SALE.COMPLETED` — Authoritative payment captured. Grants the plan's credit allowance for the 30-day period.
3. `BILLING.SUBSCRIPTION.CANCELLED` — User cancelled future renewal. Existing paid credits remain active until `current_period_end`.
4. `BILLING.SUBSCRIPTION.SUSPENDED` — Subscription paused due to admin action or billing issue.
5. `BILLING.SUBSCRIPTION.EXPIRED` — Subscription reached maximum billing cycles.
6. `BILLING.SUBSCRIPTION.PAYMENT.FAILED` — Renewal payment failed. No new credits granted; logged in audit records.
7. `PAYMENT.SALE.REFUNDED` — Payment refunded. Revokes unspent credits from that grant. If already spent, logs a shortfall entry flagged for review.
8. `PAYMENT.SALE.REVERSED` — Payment reversed or disputed.

---

## 4. Exact Frontend API Contract

The frontend implements the following 6-step lifecycle:

### Step 1: Load Available Plans
`GET /api/billing/plans` (Public)
- **Response**: List of plans with `slug`, `name`, `price_cents`, `currency`, `credits_per_period`, `billing_interval`, and `is_sandbox`.
- **UI Behavior**: Displays the pricing cards for Free, Pro, Pro+, and Pro Max.

### Step 2: Request Subscription
`POST /api/billing/subscriptions` (Authenticated Bearer token)
- **Request Body**: `{"plan_slug": "pro"}` (Accepts only the slug; price and credits are resolved securely on the server).
- **Response `201 Created`**:
  ```json
  {
    "subscription_id": "sub_a1b2c3d4",
    "paypal_subscription_id": "I-XXXXXXXXXXXX",
    "approval_url": "https://www.sandbox.paypal.com/checkoutnow?token=...",
    "status": "pending",
    "plan_slug": "pro"
  }
  ```
- **Error `409 Conflict`**: Returned if user already has an active or pending paid subscription.

### Step 3: Redirect to PayPal Approval
- The frontend redirects the user's browser window to `approval_url`.
- User approves recurring billing in PayPal's sandbox interface.

### Step 4: Return to RoboForge
- PayPal redirects the user to the configured return URL (`FRONTEND_URL/billing?subscription_id=I-XXXXX`).

### Step 5: Sync Authoritative Status
`POST /api/billing/subscriptions/{subscription_id}/sync` (Authenticated)
- **Response**: Authoritative subscription status from PayPal.
- **Rule**: If payment is not yet confirmed by PayPal REST API, status remains `"pending"` and credits are **not** granted. The UI displays *"Processing payment confirmation with PayPal..."* until verified.

### Step 6: Refresh Credit Balance
`GET /api/billing/me` (Authenticated)
- **Response**:
  ```json
  {
    "user_id": "usr_...",
    "email": "user@example.com",
    "plan": {
      "slug": "pro",
      "name": "RoboForge Pro",
      "credits_per_period": 1000,
      "price_cents": 1000
    },
    "subscription": {
      "id": "sub_...",
      "paypal_subscription_id": "I-...",
      "status": "active",
      "current_period_start": 1794477600.0,
      "current_period_end": 1797069600.0,
      "cancel_at_period_end": false
    },
    "credits": {
      "available": 1000,
      "reserved": 0,
      "period_end": 1797069600.0
    }
  }
  ```

---

## 5. Centralized Credit Engine & Operation Pricing

Credit checks and reservations use a two-phase commit pattern to prevent concurrent overspending:
```
1. reserveCredits(userId, operationId, amount)  --> Atomically holds credits
2. Perform Billable Action (Simulation, AI, RL)
3. commitCredits(operationId)                   --> Deducts from earliest-expiring grant (FIFO)
   OR
   releaseCredits(operationId)                  --> Restores credits on error/cancellation
```

### Server-Side Provisional Operation Pricing Table

| Operation ID | Name | Credit Cost | Description |
| :--- | :--- | :--- | :--- |
| `motion_planning` | Collision-Free Path Planning | 1 credit | Fast A* obstacle-aware trajectory planning |
| `ai_chat` | AI Scene Co-Pilot Chat | 2 credits | Interactive simulation assistant prompt |
| `ai_interpret_prompt` | Natural Language Scene Execution | 3 credits | Prompt translation into physical scene objects |
| `ai_refine_hypothesis` | Autonomous Research Hypothesis | 3 credits | Iterative hypothesis testing and verification |
| `synthetic_data_export` | Synthetic Vision Dataset Export | 5 credits | Multi-camera bounding box & COCO dataset |
| `rl_train_background` | PPO Swarm Background Training | 10 credits | 100-episode background RL model checkpoint |
| `rl_train_live` | Live Swarm Interactive RL | 15 credits | Live Isaac-Lab style multi-rover visual PPO |

If a user does not have sufficient credits, billable endpoints return HTTP `402 Payment Required`:
```json
{
  "detail": {
    "error": "insufficient_credits",
    "message": "Operation 'rl_train_live' requires 15 credits, but you only have 4 available.",
    "required_credits": 15,
    "available_credits": 4,
    "plan_upgrade_url": "/billing"
  }
}
```

---

## 6. Credit Lifecycle & Fallback Policies

- **Free Tier**: 50 credits automatically allocated on the user's monthly account anniversary (UTC).
- **First Paid Payment**: Expired remaining free-plan credits immediately to prevent balance stacking.
- **Cancellation**: Cancelling a subscription stops future renewals with PayPal. Paid credits remain valid and usable until `current_period_end`.
- **Free-Plan Fallback Policy**: After paid access expires (`now >= current_period_end` and all paid grants expired), the system idempotently re-activates the 50 free credits monthly allowance.
- **Refund / Reversal Shortfalls**: When a payment is refunded, any unspent credits on that grant are revoked. If credits were already consumed before the refund, the shortfall is logged in `credit_ledger` as `refund_adjustment` with a negative entry and flagged for review rather than corrupting balances.

---

## 7. Setup & Reconciliation CLI Commands

### PayPal Sandbox Resource Setup
To create the RoboForge product and 3 billing plans in your PayPal Sandbox account:
```powershell
.venv\Scripts\python -m backend.billing.setup_paypal
```
*Note: This command is idempotent and will reuse existing product/plan IDs if already configured.*

### Manual / Scheduled Reconciliation
To reconcile pending subscriptions with PayPal and reprocess failed webhooks:
```powershell
.venv\Scripts\python -m backend.billing.reconcile
```
