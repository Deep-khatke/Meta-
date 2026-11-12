import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Zap,
  CreditCard,
  ArrowRight,
  RefreshCw,
  Layers,
  ShieldCheck,
  TrendingUp,
  AlertCircle
} from 'lucide-react';
import { AIAssistantChat } from './AIAssistantChat';
import { fetchBillingMe, BillingMeResponse } from '../services/billing';
import { useAuth } from '../context/AuthContext';

interface LeftAIChatPanelProps {
  onShowToast: (msg: string) => void;
  targetPosition?: [number, number, number] | null;
  onProjectLoaded?: () => void;
  onSelectTarget?: (pos: [number, number, number] | null) => void;
  onMissionWaypoints?: (waypoints: Array<[number, number]>, loop: boolean) => void;
  onSwitchToScene: () => void;
}

const DEFAULT_BILLING_ME: BillingMeResponse = {
  user_id: 'guest',
  email: 'guest@roboforge.io',
  plan: {
    slug: 'free',
    name: 'RoboForge Free',
    credits_per_period: 50,
    price_cents: 0,
  },
  subscription: null,
  credits: {
    available: 50,
    reserved: 0,
    period_end: Math.floor(Date.now() / 1000) + 30 * 86400,
  },
};

export const LeftAIChatPanel: React.FC<LeftAIChatPanelProps> = ({
  onShowToast,
  targetPosition,
  onProjectLoaded,
  onSelectTarget,
  onMissionWaypoints,
  onSwitchToScene,
}) => {
  const { isAuthenticated } = useAuth();
  const [billingMe, setBillingMe] = useState<BillingMeResponse>(DEFAULT_BILLING_ME);
  const [loadingBilling, setLoadingBilling] = useState<boolean>(false);

  const loadCredits = useCallback(async () => {
    if (!isAuthenticated) {
      setBillingMe(DEFAULT_BILLING_ME);
      return;
    }
    setLoadingBilling(true);
    try {
      const data = await fetchBillingMe();
      setBillingMe(data);
    } catch (err: any) {
      console.warn('Could not load user credits:', err);
    } finally {
      setLoadingBilling(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    loadCredits();
  }, [loadCredits]);

  const totalCredits = billingMe.plan?.credits_per_period || 50;
  const availableCredits = billingMe.credits?.available ?? 50;
  const reservedCredits = billingMe.credits?.reserved ?? 0;
  const usedCredits = Math.max(0, totalCredits - availableCredits);
  const percentUsed = Math.min(100, Math.max(0, Math.round((usedCredits / totalCredits) * 100)));
  const planName = billingMe.plan?.name || 'RoboForge Free';
  const isPaid = billingMe.plan?.slug !== 'free' && billingMe.subscription?.status === 'active';

  return (
    <aside
      className="rf-left-sidebar"
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border-subtle)',
        overflow: 'hidden',
      }}
    >
      {/* ── Top Header & Tab Pill Switcher ── */}
      <div
        className="rf-panel-header"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
          borderBottom: '1px solid var(--border-subtle)',
          backgroundColor: '#17191A',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Segmented control to switch back to Scene or stay in AI Chat */}
          <div className="rf-segmented" style={{ padding: 2, gap: 2 }}>
            <button
              className="rf-segmented-item"
              onClick={onSwitchToScene}
              title="Switch to Scene Hierarchy & Obstacles"
              style={{ padding: '3px 9px', fontSize: 10.5 }}
            >
              <Layers size={11} />
              <span>Scene</span>
            </button>
            <button
              className="rf-segmented-item active"
              title="RoboForge Autonomous AI Co-Pilot"
              style={{
                padding: '3px 10px',
                fontSize: 10.5,
                backgroundColor: '#C4472D',
                color: '#FFFFFF',
                boxShadow: '0 2px 10px rgba(196, 71, 45, 0.45)',
              }}
            >
              <Sparkles size={11} />
              <span>Autonomous AI</span>
            </button>
          </div>
        </div>

        <button
          className="rf-btn"
          onClick={loadCredits}
          disabled={loadingBilling}
          title="Refresh credit balance"
          style={{ padding: '3px 6px', fontSize: 10, borderRadius: 100 }}
        >
          <RefreshCw size={11} className={loadingBilling ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* ── Real-Time Credit Usage & Plan Upgrade Card ── */}
      <div
        style={{
          padding: '12px 14px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div
          style={{
            background: 'rgba(23, 25, 28, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 12,
            padding: '12px 14px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.35)',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          {/* Plan badge & status */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: 6,
                  backgroundColor: 'rgba(196, 71, 45, 0.16)',
                  border: '1px solid rgba(196, 71, 45, 0.35)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Zap size={11} color="#C4472D" />
              </div>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.06em',
                  color: '#F1EEE7',
                }}
              >
                {planName.toUpperCase()}
              </span>
            </div>

            <span
              style={{
                fontSize: 9.5,
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 100,
                fontFamily: 'var(--font-mono)',
                backgroundColor: isPaid ? 'rgba(52, 211, 153, 0.15)' : 'rgba(255, 255, 255, 0.06)',
                color: isPaid ? '#34D399' : '#9E9C96',
                border: isPaid ? '1px solid rgba(52, 211, 153, 0.3)' : '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              {isPaid ? 'ACTIVE' : 'FREE TIER'}
            </span>
          </div>

          {/* Credits remaining & used count */}
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
            <div>
              <span
                style={{
                  fontSize: 22,
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  color: '#F1EEE7',
                  letterSpacing: '-0.02em',
                }}
              >
                {availableCredits.toLocaleString()}
              </span>
              <span style={{ fontSize: 11, color: '#9E9C96', marginLeft: 6 }}>
                credits left
              </span>
            </div>

            <span
              style={{
                fontSize: 10.5,
                fontFamily: 'var(--font-mono)',
                color: '#9E9C96',
              }}
            >
              {usedCredits} / {totalCredits} used
            </span>
          </div>

          {/* Tactile Progress bar */}
          <div
            style={{
              width: '100%',
              height: 5,
              borderRadius: 100,
              backgroundColor: 'rgba(255, 255, 255, 0.06)',
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <div
              style={{
                width: `${percentUsed}%`,
                height: '100%',
                backgroundColor: percentUsed > 85 ? '#FB7185' : '#C4472D',
                borderRadius: 100,
                transition: 'width 0.3s ease',
              }}
            />
          </div>

          {/* Upgrade Plan CTA Button */}
          <Link
            to="/billing"
            className="rf-btn rf-btn-primary"
            style={{
              width: '100%',
              padding: '7px 12px',
              borderRadius: 100,
              backgroundColor: '#C4472D',
              color: '#FFFFFF',
              textDecoration: 'none',
              fontSize: 11.5,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 7,
              boxShadow: '0 4px 14px rgba(196, 71, 45, 0.4)',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#D35237')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#C4472D')}
            title="Open Billing Page to upgrade plan with PayPal Subscriptions"
          >
            <CreditCard size={13} />
            <span>Upgrade Plan</span>
            <ArrowRight size={12} style={{ marginLeft: 'auto' }} />
          </Link>
        </div>
      </div>

      {/* ── Direct Autonomous AI Chat Co-Pilot ── */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <AIAssistantChat
          onShowToast={onShowToast}
          selectedScenarioId="scen_slalom_beta"
          targetPosition={targetPosition}
          onProjectLoaded={onProjectLoaded}
          onSelectTarget={onSelectTarget}
          onMissionWaypoints={onMissionWaypoints}
        />
      </div>
    </aside>
  );
};
