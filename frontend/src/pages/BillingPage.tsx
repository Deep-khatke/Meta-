import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  fetchBillingPlans,
  fetchBillingMe,
  createBillingSubscription,
  syncBillingSubscription,
  cancelBillingSubscription,
  fetchCreditLedger,
  BillingPlan,
  BillingMeResponse,
  CreditLedgerEntry,
} from '../services/billing';
import { useAuth } from '../context/AuthContext';
import {
  CreditCard,
  Zap,
  CheckCircle2,
  AlertCircle,
  Clock,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  ExternalLink,
  ChevronLeft,
  X,
  History,
  Layers,
  Sparkles,
} from 'lucide-react';

const DEFAULT_PLANS: BillingPlan[] = [
  {
    slug: 'free',
    name: 'RoboForge Free',
    description: 'Baseline robotics simulation, URDF editing, and 50 free credits every UTC account month.',
    price_cents: 0,
    currency: 'USD',
    credits_per_period: 50,
    billing_interval: 'month',
    interval_count: 1,
    is_sandbox: true,
  },
  {
    slug: 'pro',
    name: 'RoboForge Pro',
    description: 'Professional engineering compute: 1,000 credits/mo for live RL training, kinematics, and AI co-pilot.',
    price_cents: 1000,
    currency: 'USD',
    credits_per_period: 1000,
    billing_interval: 'month',
    interval_count: 1,
    is_sandbox: true,
  },
  {
    slug: 'pro_plus',
    name: 'RoboForge Pro+',
    description: 'High-throughput robotics research: 2,000 credits/mo with accelerated multi-worker RL training.',
    price_cents: 2000,
    currency: 'USD',
    credits_per_period: 2000,
    billing_interval: 'month',
    interval_count: 1,
    is_sandbox: true,
  },
  {
    slug: 'pro_max',
    name: 'RoboForge Pro Max',
    description: 'Enterprise robotics suite: 5,000 credits/mo for large-scale curricula and swarm orchestration.',
    price_cents: 4000,
    currency: 'USD',
    credits_per_period: 5000,
    billing_interval: 'month',
    interval_count: 1,
    is_sandbox: true,
  },
];

export const BillingPage: React.FC = () => {
  const { isAuthenticated, user } = useAuth();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const [plans, setPlans] = useState<BillingPlan[]>(DEFAULT_PLANS);
  const [billingMe, setBillingMe] = useState<BillingMeResponse | null>(null);
  const [ledger, setLedger] = useState<CreditLedgerEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);
  const [showCancelModal, setShowCancelModal] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'plans' | 'ledger'>('plans');

  // Load plans & user billing state
  const loadData = useCallback(async () => {
    try {
      const plansData = await fetchBillingPlans();
      setPlans(plansData);

      if (isAuthenticated) {
        const meData = await fetchBillingMe();
        setBillingMe(meData);

        const ledgerData = await fetchCreditLedger(15, 0);
        setLedger(ledgerData);
      }
    } catch (err: any) {
      console.error('Failed to load billing details:', err);
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle PayPal return sync if subscription_id is in query params
  useEffect(() => {
    const subIdParam = searchParams.get('subscription_id') || searchParams.get('ba_token');
    if (subIdParam && isAuthenticated) {
      setSyncing(true);
      setNotification({
        type: 'info',
        message: 'Synchronizing with PayPal Sandbox to confirm your subscription and credit grant...',
      });

      // Clear params from URL
      setSearchParams({}, { replace: true });

      syncBillingSubscription(subIdParam)
        .then((res) => {
          if (res.status === 'active') {
            setNotification({
              type: 'success',
              message: `PayPal subscription confirmed! Your credits have been allocated.`,
            });
          } else {
            setNotification({
              type: 'info',
              message: `PayPal subscription status is '${res.status}'. Payment will be reconciled upon confirmation.`,
            });
          }
          return loadData();
        })
        .catch((err) => {
          setNotification({
            type: 'error',
            message: `Reconciliation notice: ${err.message}`,
          });
        })
        .finally(() => {
          setSyncing(false);
        });
    }
  }, [searchParams, isAuthenticated, setSearchParams, loadData]);

  // Handle plan upgrade click
  const handleUpgrade = async (planSlug: string) => {
    if (!isAuthenticated) {
      navigate('/login?redirect=/billing');
      return;
    }

    if (planSlug === 'free') {
      return;
    }

    setActionLoading(planSlug);
    setNotification(null);

    try {
      const res = await createBillingSubscription(planSlug);
      if (res && res.approval_url) {
        setNotification({
          type: 'info',
          message: 'Redirecting to PayPal Sandbox checkout approval...',
        });
        window.location.assign(res.approval_url);
      } else {
        throw new Error('PayPal did not return an approval URL');
      }
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.message || 'Unable to initiate subscription',
      });
      setActionLoading(null);
    }
  };

  // Handle manual sync
  const handleManualSync = async () => {
    if (!billingMe?.subscription?.id) return;
    setSyncing(true);
    setNotification(null);
    try {
      await syncBillingSubscription(billingMe.subscription.id);
      await loadData();
      setNotification({
        type: 'success',
        message: 'Subscription & credits state synced with PayPal.',
      });
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.message || 'Failed to sync with PayPal',
      });
    } finally {
      setSyncing(false);
    }
  };

  // Handle cancellation
  const handleCancelSubscription = async () => {
    if (!billingMe?.subscription?.id) return;
    setActionLoading('cancel');
    try {
      await cancelBillingSubscription(billingMe.subscription.id);
      setShowCancelModal(false);
      await loadData();
      setNotification({
        type: 'success',
        message: 'Subscription renewal cancelled. Your existing credits remain available until the end of your billing cycle.',
      });
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.message || 'Failed to cancel subscription',
      });
    } finally {
      setActionLoading(null);
    }
  };

  const currentSlug = billingMe?.plan?.slug || 'free';
  const currentSub = billingMe?.subscription;
  const isPaidActive = currentSub?.status === 'active';
  const isPaidCancelled = currentSub?.status === 'cancelled';

  const planFeatures: Record<string, string[]> = {
    free: [
      '50 credits refreshed monthly (UTC account anniversary)',
      'Full MuJoCo 3.17 physical simulation core',
      'Interactive 3D Robot Modeling & Joint Studio',
      'Basic A* collision-free path planning',
      'Community documentation & URDF importer',
    ],
    pro: [
      '1,000 credits per billing period',
      'NVIDIA Nemotron AI Scene Co-Pilot',
      'Autonomous One-Prompt mission generation',
      'Synthetic vision frame capture & COCO dataset export',
      'Single-agent PPO reinforcement learning training',
      'Direct ROS 2 Node & ONNX policy export',
    ],
    pro_plus: [
      '2,000 credits per billing period',
      'Everything in Pro tier',
      'Live 8-Rover Isaac-Lab style Swarm RL',
      'Parallel drone & articulated arm training',
      'Domain randomization & Sim-to-Real scorecard',
      'Priority WebSocket simulation throughput',
    ],
    pro_max: [
      '5,000 credits per billing period',
      'Everything in Pro+ tier',
      'Unlimited research hypothesis iterations',
      'High-throughput synthetic bounding box batches',
      'Long-run background PPO checkpoints',
      'Dedicated developer sandbox support',
    ],
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#121415',
      color: '#F1EEE7',
      fontFamily: 'Manrope, -apple-system, BlinkMacSystemFont, sans-serif',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* ── Top Navigation Bar ── */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 100,
        backgroundColor: 'rgba(18, 20, 21, 0.85)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        padding: '14px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button
            type="button"
            onClick={() => navigate('/studio')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              borderRadius: 8,
              padding: '6px 12px',
              color: '#F1EEE7',
              fontSize: 13,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.06)')}
          >
            <ChevronLeft size={16} />
            <span>Return to Studio</span>
          </button>

          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none' }}>
            <div style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              backgroundColor: '#C4472D',
              boxShadow: '0 0 10px rgba(196, 71, 45, 0.8)',
            }} />
            <span style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: 15,
              fontWeight: 700,
              letterSpacing: '0.12em',
              color: '#F1EEE7',
            }}>
              ROBOFORGE
            </span>
          </Link>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{
            fontFamily: 'IBM Plex Mono, monospace',
            fontSize: 11,
            letterSpacing: '0.08em',
            padding: '4px 10px',
            borderRadius: 100,
            backgroundColor: 'rgba(255, 170, 0, 0.12)',
            color: '#FFAA00',
            border: '1px solid rgba(255, 170, 0, 0.25)',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
          }}>
            <ShieldCheck size={12} />
            PAYPAL SANDBOX ENVIRONMENT
          </span>

          {isAuthenticated ? (
            <div style={{
              fontSize: 13,
              color: '#9A9A96',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <span>{user?.email}</span>
            </div>
          ) : (
            <Link
              to="/login?redirect=/billing"
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: '#121415',
                backgroundColor: '#F1EEE7',
                padding: '6px 14px',
                borderRadius: 100,
                textDecoration: 'none',
              }}
            >
              Sign In
            </Link>
          )}
        </div>
      </header>

      {/* ── Main Content Container ── */}
      <main style={{ maxWidth: 1180, width: '100%', margin: '0 auto', padding: '36px 20px', flex: 1 }}>
        {/* Banner notification */}
        {notification && (
          <div style={{
            marginBottom: 24,
            padding: '12px 18px',
            borderRadius: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            fontSize: 14,
            backgroundColor:
              notification.type === 'success'
                ? 'rgba(46, 204, 113, 0.12)'
                : notification.type === 'error'
                ? 'rgba(231, 76, 60, 0.12)'
                : 'rgba(52, 152, 219, 0.12)',
            border: `1px solid ${
              notification.type === 'success'
                ? 'rgba(46, 204, 113, 0.3)'
                : notification.type === 'error'
                ? 'rgba(231, 76, 60, 0.3)'
                : 'rgba(52, 152, 219, 0.3)'
            }`,
            color:
              notification.type === 'success'
                ? '#2ECC71'
                : notification.type === 'error'
                ? '#E74C3C'
                : '#3498DB',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {notification.type === 'success' && <CheckCircle2 size={18} />}
              {notification.type === 'error' && <AlertCircle size={18} />}
              {notification.type === 'info' && <RefreshCw size={18} className={syncing ? 'animate-spin' : ''} />}
              <span>{notification.message}</span>
            </div>
            <button
              type="button"
              onClick={() => setNotification(null)}
              style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>
        )}

        {/* ── Section 1: User Credit Status Card ── */}
        {isAuthenticated && billingMe ? (
          <div style={{
            backgroundColor: '#17191A',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 16,
            padding: '24px 28px',
            marginBottom: 36,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 24,
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)',
          }}>
            <div>
              <div style={{ fontSize: 12, color: '#9A9A96', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
                Current Subscription
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: 22, fontWeight: 700, color: '#F1EEE7' }}>
                  {billingMe.plan.name}
                </span>
                <span style={{
                  fontSize: 11,
                  fontFamily: 'IBM Plex Mono, monospace',
                  padding: '3px 8px',
                  borderRadius: 6,
                  backgroundColor: isPaidActive ? 'rgba(46, 204, 113, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                  color: isPaidActive ? '#2ECC71' : '#9A9A96',
                  border: isPaidActive ? '1px solid rgba(46, 204, 113, 0.3)' : '1px solid rgba(255, 255, 255, 0.15)',
                }}>
                  {currentSub ? currentSub.status.toUpperCase() : 'ACTIVE'}
                </span>
              </div>
              <div style={{ fontSize: 13, color: '#9A9A96', marginTop: 6 }}>
                ${(billingMe.plan.price_cents / 100).toFixed(2)} USD / month
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12, color: '#9A9A96', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
                Available Simulation Credits
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{
                  fontSize: 32,
                  fontWeight: 800,
                  fontFamily: 'IBM Plex Mono, monospace',
                  color: '#C4472D',
                  textShadow: '0 0 20px rgba(196, 71, 45, 0.4)',
                }}>
                  {billingMe.credits.available.toLocaleString()}
                </span>
                <span style={{ fontSize: 14, color: '#9A9A96' }}>credits</span>
              </div>
              <div style={{ fontSize: 12, color: '#6A6A66', marginTop: 4 }}>
                {billingMe.credits.reserved > 0 ? `${billingMe.credits.reserved} credits currently reserved` : '0 reserved'}
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12, color: '#9A9A96', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
                Billing Period End / Reset
              </div>
              <div style={{ fontSize: 16, fontWeight: 600, color: '#F1EEE7', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Clock size={16} color="#9A9A96" />
                <span>
                  {billingMe.credits.period_end
                    ? new Date(billingMe.credits.period_end * 1000).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })
                    : 'End of cycle'}
                </span>
              </div>
              <div style={{ fontSize: 12, color: '#9A9A96', marginTop: 4 }}>
                {isPaidCancelled
                  ? 'Access continues until period end (renewal cancelled)'
                  : isPaidActive
                  ? 'Auto-renews monthly via PayPal'
                  : 'Free tier refreshes on account anniversary'}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 8 }}>
              {currentSub && (
                <button
                  type="button"
                  onClick={handleManualSync}
                  disabled={syncing}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    color: '#F1EEE7',
                    borderRadius: 8,
                    padding: '8px 14px',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: syncing ? 'not-allowed' : 'pointer',
                  }}
                >
                  <RefreshCw size={14} className={syncing ? 'animate-spin' : ''} />
                  <span>{syncing ? 'Syncing...' : 'Sync with PayPal'}</span>
                </button>
              )}

              {isPaidActive && !isPaidCancelled && (
                <button
                  type="button"
                  onClick={() => setShowCancelModal(true)}
                  style={{
                    backgroundColor: 'transparent',
                    border: '1px solid rgba(231, 76, 60, 0.3)',
                    color: '#E74C3C',
                    borderRadius: 8,
                    padding: '8px 14px',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(231, 76, 60, 0.1)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  Cancel Renewal
                </button>
              )}
            </div>
          </div>
        ) : !isAuthenticated ? (
          <div style={{
            backgroundColor: '#17191A',
            border: '1px solid rgba(196, 71, 45, 0.3)',
            borderRadius: 16,
            padding: '24px 28px',
            marginBottom: 36,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 20,
            flexWrap: 'wrap',
          }}>
            <div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#F1EEE7', marginBottom: 4 }}>
                Ready to accelerate your robotics research?
              </div>
              <div style={{ fontSize: 14, color: '#9A9A96' }}>
                Sign in to manage your subscription, see your real-time credit balance, and upgrade to higher compute tiers.
              </div>
            </div>
            <Link
              to="/login?redirect=/billing"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                backgroundColor: '#C4472D',
                color: '#FFFFFF',
                padding: '10px 20px',
                borderRadius: 10,
                fontWeight: 700,
                fontSize: 14,
                textDecoration: 'none',
              }}
            >
              <span>Sign In to RoboForge</span>
              <ArrowRight size={16} />
            </Link>
          </div>
        ) : null}

        {/* ── Section 2: Tab Bar ── */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          marginBottom: 32,
        }}>
          <button
            type="button"
            onClick={() => setActiveTab('plans')}
            style={{
              padding: '12px 16px',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === 'plans' ? '2px solid #C4472D' : '2px solid transparent',
              color: activeTab === 'plans' ? '#F1EEE7' : '#9A9A96',
              fontWeight: 700,
              fontSize: 15,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <Layers size={18} />
            <span>Subscription Plans</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('ledger')}
            style={{
              padding: '12px 16px',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === 'ledger' ? '2px solid #C4472D' : '2px solid transparent',
              color: activeTab === 'ledger' ? '#F1EEE7' : '#9A9A96',
              fontWeight: 700,
              fontSize: 15,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <History size={18} />
            <span>Credits Ledger & Usage</span>
          </button>
        </div>

        {/* ── TAB 1: Plans Grid ── */}
        {activeTab === 'plans' && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: 36 }}>
              <h1 style={{ fontSize: 32, fontWeight: 800, color: '#F1EEE7', marginBottom: 10 }}>
                Predictable Compute for Robotics Engineering
              </h1>
              <p style={{ fontSize: 16, color: '#9A9A96', maxWidth: 640, margin: '0 auto' }}>
                All paid tiers are recurring monthly subscriptions billed in USD through PayPal Sandbox.
                Upgrade or cancel future renewal at any time.
              </p>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: 20,
              marginBottom: 48,
            }}>
              {(Array.isArray(plans) && plans.length > 0 ? plans : DEFAULT_PLANS).map((p) => {
                const isCurrent = currentSlug === p.slug;
                const isProPlus = p.slug === 'pro_plus';
                const isFree = p.slug === 'free';
                const features = planFeatures[p.slug] || [];

                return (
                  <div
                    key={p.slug}
                    style={{
                      backgroundColor: '#17191A',
                      borderRadius: 16,
                      border: isCurrent
                        ? '2px solid #C4472D'
                        : isProPlus
                        ? '1px solid rgba(196, 71, 45, 0.6)'
                        : '1px solid rgba(255, 255, 255, 0.1)',
                      padding: 24,
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      position: 'relative',
                      boxShadow: isProPlus ? '0 12px 36px rgba(196, 71, 45, 0.15)' : 'none',
                    }}
                  >
                    {isProPlus && (
                      <div style={{
                        position: 'absolute',
                        top: -11,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        backgroundColor: '#C4472D',
                        color: '#FFFFFF',
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: '0.08em',
                        padding: '3px 12px',
                        borderRadius: 100,
                        textTransform: 'uppercase',
                        fontFamily: 'IBM Plex Mono, monospace',
                      }}>
                        Most Popular
                      </div>
                    )}

                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <h3 style={{ fontSize: 18, fontWeight: 700, color: '#F1EEE7' }}>{p.name}</h3>
                        {isCurrent && (
                          <span style={{
                            fontSize: 11,
                            backgroundColor: 'rgba(196, 71, 45, 0.15)',
                            color: '#C4472D',
                            padding: '3px 8px',
                            borderRadius: 6,
                            fontWeight: 700,
                          }}>
                            CURRENT
                          </span>
                        )}
                      </div>

                      <p style={{ fontSize: 13, color: '#9A9A96', minHeight: 38, marginBottom: 16 }}>
                        {p.description}
                      </p>

                      <div style={{ marginBottom: 20 }}>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                          <span style={{ fontSize: 32, fontWeight: 800, color: '#F1EEE7' }}>
                            ${(p.price_cents / 100).toFixed(0)}
                          </span>
                          <span style={{ fontSize: 14, color: '#9A9A96' }}>/ month</span>
                        </div>
                        <div style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 6,
                          marginTop: 8,
                          backgroundColor: 'rgba(255, 255, 255, 0.06)',
                          padding: '4px 10px',
                          borderRadius: 8,
                          fontSize: 13,
                          fontWeight: 700,
                          color: '#F1EEE7',
                          fontFamily: 'IBM Plex Mono, monospace',
                        }}>
                          <Zap size={14} color="#C4472D" />
                          <span>{p.credits_per_period.toLocaleString()} credits / month</span>
                        </div>
                      </div>

                      <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: 16, marginBottom: 24 }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: '#F1EEE7', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
                          Included Capabilities
                        </div>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
                          {features.map((feat, idx) => (
                            <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 13, color: '#CCCCCC' }}>
                              <CheckCircle2 size={15} color="#2ECC71" style={{ flexShrink: 0, marginTop: 2 }} />
                              <span>{feat}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div>
                      {isCurrent ? (
                        <button
                          type="button"
                          disabled
                          style={{
                            width: '100%',
                            padding: '12px',
                            borderRadius: 10,
                            backgroundColor: 'rgba(255, 255, 255, 0.06)',
                            color: '#9A9A96',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: 'default',
                          }}
                        >
                          Current Active Plan
                        </button>
                      ) : isFree ? (
                        <button
                          type="button"
                          disabled
                          style={{
                            width: '100%',
                            padding: '12px',
                            borderRadius: 10,
                            backgroundColor: 'rgba(255, 255, 255, 0.06)',
                            color: '#9A9A96',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: 'default',
                          }}
                        >
                          Default Free Allowance
                        </button>
                      ) : !isAuthenticated ? (
                        <button
                          type="button"
                          onClick={() => navigate('/login?redirect=/billing')}
                          style={{
                            width: '100%',
                            padding: '12px',
                            borderRadius: 10,
                            backgroundColor: '#C4472D',
                            color: '#FFFFFF',
                            border: 'none',
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 8,
                            boxShadow: '0 4px 14px rgba(196, 71, 45, 0.4)',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#d35237')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#C4472D')}
                        >
                          <CreditCard size={16} />
                          <span>Sign In to Subscribe</span>
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => handleUpgrade(p.slug)}
                          disabled={Boolean(actionLoading)}
                          style={{
                            width: '100%',
                            padding: '12px',
                            borderRadius: 10,
                            backgroundColor: '#C4472D',
                            color: '#FFFFFF',
                            border: 'none',
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: actionLoading ? 'wait' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 8,
                            boxShadow: '0 4px 14px rgba(196, 71, 45, 0.4)',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#d35237')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#C4472D')}
                        >
                          {actionLoading === p.slug ? (
                            <>
                              <RefreshCw size={16} className="animate-spin" />
                              <span>Connecting to PayPal...</span>
                            </>
                          ) : (
                            <>
                              <CreditCard size={16} />
                              <span>Upgrade with PayPal</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* ── Operation Pricing Table ── */}
            <div style={{
              backgroundColor: '#17191A',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: 16,
              padding: '24px 28px',
              marginTop: 36,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <Zap size={20} color="#C4472D" />
                <h3 style={{ fontSize: 18, fontWeight: 700, color: '#F1EEE7' }}>
                  Server-Side Operation Pricing Guide
                </h3>
              </div>
              <p style={{ fontSize: 13, color: '#9A9A96', marginBottom: 20 }}>
                Operations use atomic two-phase reservations (`reserve` before compute, `commit` on success, `release` on error). No unexpected overage charges.
              </p>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#9A9A96' }}>
                      <th style={{ padding: '10px 14px' }}>Operation</th>
                      <th style={{ padding: '10px 14px' }}>Credit Cost</th>
                      <th style={{ padding: '10px 14px' }}>Billable Route</th>
                      <th style={{ padding: '10px 14px' }}>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>Collision-Free Path Planning</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>1 credit</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/motion/plan-path</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Fast A* obstacle-aware trajectory planning</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>AI Scene Co-Pilot Chat</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>2 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/ai/chat</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Telemetry-grounded multimodal conversational agent</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>Natural Language Scene Execution</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>3 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/ai/interpret-prompt</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Prompt translation into physical MuJoCo scene objects</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>Autonomous Research Hypothesis</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>3 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/ai/refine-hypothesis</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Iterative hypothesis generation, evaluation, and refinement</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>Synthetic Vision Dataset Export</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>5 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/synthetic/export-coco</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Multi-camera bounding box & COCO dataset export</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>PPO Swarm Background Training</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>10 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/rl/train</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>100-episode headless PPO training checkpoint</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#F1EEE7' }}>Live Swarm Interactive RL</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#C4472D', fontWeight: 700 }}>15 credits</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>POST /api/rl/train-rover-live</td>
                      <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>Live 8-rover Isaac-Lab style visual policy learning</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 2: Credits Ledger ── */}
        {activeTab === 'ledger' && (
          <div style={{
            backgroundColor: '#17191A',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 16,
            padding: '24px 28px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
              <div>
                <h3 style={{ fontSize: 18, fontWeight: 700, color: '#F1EEE7' }}>
                  Append-Only Credit Activity Ledger
                </h3>
                <p style={{ fontSize: 13, color: '#9A9A96', marginTop: 4 }}>
                  Audited historical log of credit grants, burns, reservations, and refunds for your account.
                </p>
              </div>
              <button
                type="button"
                onClick={loadData}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  backgroundColor: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  borderRadius: 8,
                  padding: '6px 12px',
                  color: '#F1EEE7',
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                <RefreshCw size={13} />
                <span>Refresh</span>
              </button>
            </div>

            {ledger.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '48px 20px', color: '#9A9A96' }}>
                <History size={32} style={{ opacity: 0.4, marginBottom: 12 }} />
                <div style={{ fontSize: 14 }}>No transactions recorded yet.</div>
                <div style={{ fontSize: 12, marginTop: 4 }}>
                  As you run AI co-pilot chats or RL simulations, transactions will be audited here.
                </div>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#9A9A96' }}>
                      <th style={{ padding: '10px 14px' }}>Timestamp</th>
                      <th style={{ padding: '10px 14px' }}>Type</th>
                      <th style={{ padding: '10px 14px' }}>Change</th>
                      <th style={{ padding: '10px 14px' }}>Balance After</th>
                      <th style={{ padding: '10px 14px' }}>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ledger.map((entry) => {
                      const isPositive = entry.amount > 0;
                      return (
                        <tr key={entry.id} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                          <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#9A9A96' }}>
                            {new Date(entry.created_at * 1000).toLocaleString()}
                          </td>
                          <td style={{ padding: '12px 14px' }}>
                            <span style={{
                              fontFamily: 'IBM Plex Mono, monospace',
                              fontSize: 11,
                              padding: '2px 8px',
                              borderRadius: 4,
                              backgroundColor: 'rgba(255, 255, 255, 0.06)',
                              color: '#F1EEE7',
                            }}>
                              {entry.entry_type}
                            </span>
                          </td>
                          <td style={{
                            padding: '12px 14px',
                            fontFamily: 'IBM Plex Mono, monospace',
                            fontWeight: 700,
                            color: isPositive ? '#2ECC71' : entry.amount < 0 ? '#E74C3C' : '#9A9A96',
                          }}>
                            {isPositive ? `+${entry.amount}` : entry.amount}
                          </td>
                          <td style={{ padding: '12px 14px', fontFamily: 'IBM Plex Mono, monospace', color: '#F1EEE7' }}>
                            {entry.balance_after}
                          </td>
                          <td style={{ padding: '12px 14px', color: '#CCCCCC' }}>
                            {entry.description}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Cancel Renewal Confirmation Modal ── */}
      {showCancelModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: 20,
        }}>
          <div style={{
            backgroundColor: '#17191A',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: 16,
            padding: 28,
            maxWidth: 460,
            width: '100%',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.9)',
          }}>
            <h3 style={{ fontSize: 18, fontWeight: 700, color: '#F1EEE7', marginBottom: 12 }}>
              Cancel Future Renewal?
            </h3>
            <p style={{ fontSize: 14, color: '#CCCCCC', lineHeight: 1.5, marginBottom: 20 }}>
              Your future recurring payments with PayPal will be stopped. You will keep all your remaining paid credits, and your full plan entitlements will remain active until the end of your current billing period.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
              <button
                type="button"
                onClick={() => setShowCancelModal(false)}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  color: '#F1EEE7',
                  borderRadius: 8,
                  padding: '8px 16px',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Keep Renewal
              </button>
              <button
                type="button"
                onClick={handleCancelSubscription}
                disabled={actionLoading === 'cancel'}
                style={{
                  backgroundColor: '#E74C3C',
                  border: 'none',
                  color: '#FFFFFF',
                  borderRadius: 8,
                  padding: '8px 16px',
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: actionLoading === 'cancel' ? 'wait' : 'pointer',
                }}
              >
                {actionLoading === 'cancel' ? 'Cancelling...' : 'Confirm Cancellation'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BillingPage;

