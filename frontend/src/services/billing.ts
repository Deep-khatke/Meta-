/**
 * RoboForge Billing & Credit API Service
 * Interacts with PayPal Subscriptions and Centralized Credit Ledger backend.
 */

export interface BillingPlan {
  slug: string;
  name: string;
  description: string;
  price_cents: number;
  currency: string;
  credits_per_period: number;
  billing_interval: string;
  interval_count: number;
  is_sandbox: boolean;
}

export interface BillingSubscription {
  id: string;
  paypal_subscription_id: string;
  plan_slug: string;
  status: 'pending' | 'active' | 'cancelled' | 'suspended' | 'expired';
  current_period_start: number | null;
  current_period_end: number | null;
  cancel_at_period_end: boolean;
  approval_url?: string | null;
}

export interface BillingMeResponse {
  user_id: string;
  email: string;
  plan: {
    slug: string;
    name: string;
    credits_per_period: number;
    price_cents: number;
  };
  subscription: BillingSubscription | null;
  credits: {
    available: number;
    reserved: number;
    period_end: number;
  };
}

export interface CreateSubscriptionResponse {
  subscription_id: string;
  paypal_subscription_id: string;
  approval_url: string;
  status: string;
  plan_slug: string;
}

export interface CreditLedgerEntry {
  id: string;
  grant_id?: string | null;
  amount: number;
  balance_after: number;
  entry_type: 'free_anniversary' | 'subscription_payment' | 'spend' | 'reservation' | 'release' | 'refund_adjustment' | 'expiry';
  operation_id?: string | null;
  description: string;
  created_at: number;
}

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('roboforge_token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function fetchBillingPlans(): Promise<BillingPlan[]> {
  const res = await fetch('/api/billing/plans');
  if (!res.ok) {
    throw new Error('Failed to load billing plans');
  }
  return res.json();
}

export async function fetchBillingMe(): Promise<BillingMeResponse> {
  const res = await fetch('/api/billing/me', {
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to load user billing summary');
  }
  return res.json();
}

export async function createBillingSubscription(planSlug: string): Promise<CreateSubscriptionResponse> {
  const res = await fetch('/api/billing/subscriptions', {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ plan_slug: planSlug }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to initiate PayPal subscription');
  }
  return res.json();
}

export async function syncBillingSubscription(subscriptionId: string): Promise<any> {
  const res = await fetch(`/api/billing/subscriptions/${encodeURIComponent(subscriptionId)}/sync`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to sync subscription status');
  }
  return res.json();
}

export async function cancelBillingSubscription(subscriptionId: string): Promise<any> {
  const res = await fetch(`/api/billing/subscriptions/${encodeURIComponent(subscriptionId)}/cancel`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to cancel subscription');
  }
  return res.json();
}

export async function fetchCreditLedger(limit = 20, offset = 0): Promise<CreditLedgerEntry[]> {
  const res = await fetch(`/api/billing/credits/ledger?limit=${limit}&offset=${offset}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to load credit ledger');
  }
  const data = await res.json();
  return data.entries || [];
}
