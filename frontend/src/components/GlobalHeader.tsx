import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Wrench,
  Save,
  Cpu,
  Zap,
  Check,
  ChevronRight,
  Circle,
  FolderOpen
} from 'lucide-react';
import { Project, SimulationState } from '../types';
import { UserProfileBadge } from './UserProfileBadge';
import { fetchBillingMe } from '../services/billing';

export type ActiveWorkspace = 'ai' | 'engineering';

interface GlobalHeaderProps {
  activeWorkspace: ActiveWorkspace;
  onSwitchWorkspace: (workspace: ActiveWorkspace) => void;
  project: Project;
  simState: SimulationState;
  isConnected: boolean;
  onSaveProject: () => void;
  onOpenURDFModal?: () => void;
  onShowBilling?: () => void;
  isEducatorMode?: boolean;
  onToggleEducatorMode?: () => void;
}

export const GlobalHeader: React.FC<GlobalHeaderProps> = ({
  activeWorkspace,
  onSwitchWorkspace,
  project,
  simState,
  isConnected,
  onSaveProject,
  onOpenURDFModal,
  onShowBilling,
  isEducatorMode = false,
  onToggleEducatorMode,
}) => {
  const [availableCredits, setAvailableCredits] = useState<number | null>(null);
  const [justSaved, setJustSaved] = useState<boolean>(false);

  useEffect(() => {
    fetchBillingMe()
      .then((data) => setAvailableCredits(data?.credits?.available ?? null))
      .catch(() => setAvailableCredits(null));
  }, []);

  const handleSave = () => {
    onSaveProject();
    setJustSaved(true);
    setTimeout(() => setJustSaved(false), 2000);
  };

  // Keyboard shortcut listeners: Ctrl+1 (AI Workspace), Ctrl+2 (Engineering Studio), Ctrl+S (Save)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger when typing in inputs/textareas
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === '1') {
        e.preventDefault();
        onSwitchWorkspace('ai');
      } else if ((e.ctrlKey || e.metaKey) && e.key === '2') {
        e.preventDefault();
        onSwitchWorkspace('engineering');
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        handleSave();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'e') {
        e.preventDefault();
        onToggleEducatorMode?.();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSwitchWorkspace, onSaveProject, onToggleEducatorMode]);

  return (
    <header
      className="rf-global-header"
      style={{
        height: 48,
        minHeight: 48,
        padding: '0 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#101113',
        borderBottom: '1px solid #232428',
        zIndex: 50,
        userSelect: 'none',
      }}
    >
      {/* ── Left: Branding & Active Project ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
        {/* Brand Mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <div
            style={{
              width: 26,
              height: 26,
              borderRadius: 7,
              background: 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 10px rgba(211, 84, 50, 0.45)',
            }}
          >
            <Cpu size={14} color="#FFFFFF" />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <span style={{ fontWeight: 800, fontSize: 13.5, letterSpacing: '0.5px', color: '#F5F5F5' }}>
              ROBO<span style={{ color: '#D35432' }}>FORGE</span>
            </span>
            <span
              style={{
                fontSize: 9.5,
                fontWeight: 700,
                color: activeWorkspace === 'ai' ? '#D35432' : '#A1A1AA',
                fontFamily: 'var(--font-mono, monospace)',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                padding: '1px 5px',
                borderRadius: 4,
                backgroundColor: activeWorkspace === 'ai' ? 'rgba(211, 84, 50, 0.12)' : 'rgba(255, 255, 255, 0.05)',
                border: activeWorkspace === 'ai' ? '1px solid rgba(211, 84, 50, 0.3)' : '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              {activeWorkspace === 'ai' ? 'AI' : 'STUDIO'}
            </span>
          </div>
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: '#303136' }} />

        {/* Project Name Breadcrumb */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 11.5,
            color: '#71717A',
          }}
          title={project?.name || 'New Robotics Workspace'}
        >
          <span>/</span>
          <span
            style={{
              fontWeight: 600,
              color: '#D4D4D8',
              maxWidth: 160,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {project?.name || 'New Robotics Workspace'}
          </span>
        </div>
      </div>

      {/* ── Centre: Global Workspace Switcher (Prominent & Seamless) ── */}
      <div
        className="rf-global-workspace-switcher"
        style={{
          display: 'flex',
          alignItems: 'center',
          background: '#151619',
          border: '1px solid #303136',
          borderRadius: 8,
          padding: 3,
          gap: 3,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.35)',
        }}
      >
        <button
          onClick={() => onSwitchWorkspace('ai')}
          title="Switch to AI Workspace (Ctrl+1) — Natural language robot building, simulation & training"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            padding: '5px 14px',
            borderRadius: 6,
            fontSize: 11.5,
            fontWeight: activeWorkspace === 'ai' ? 700 : 500,
            color: activeWorkspace === 'ai' ? '#FFFFFF' : '#A1A1AA',
            background: activeWorkspace === 'ai' ? 'linear-gradient(135deg, #D35432 0%, #B84323 100%)' : 'transparent',
            border: activeWorkspace === 'ai' ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid transparent',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeWorkspace === 'ai' ? '0 2px 8px rgba(211, 84, 50, 0.45)' : 'none',
          }}
        >
          <Sparkles size={13} color={activeWorkspace === 'ai' ? '#FFFFFF' : '#D35432'} />
          <span>AI Workspace</span>
          <span
            style={{
              fontSize: 9,
              opacity: activeWorkspace === 'ai' ? 0.8 : 0.4,
              fontFamily: 'var(--font-mono, monospace)',
              padding: '0 3px',
              borderRadius: 3,
              background: 'rgba(0, 0, 0, 0.25)',
            }}
          >
            ^1
          </span>
        </button>

        <button
          onClick={() => onSwitchWorkspace('engineering')}
          title="Switch to Engineering Studio (Ctrl+2) — Precision multibody modeling, kinematics, sensors & physics controls"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            padding: '5px 14px',
            borderRadius: 6,
            fontSize: 11.5,
            fontWeight: activeWorkspace === 'engineering' ? 700 : 500,
            color: activeWorkspace === 'engineering' ? '#FFFFFF' : '#A1A1AA',
            background: activeWorkspace === 'engineering' ? '#242529' : 'transparent',
            border: activeWorkspace === 'engineering' ? '1px solid #3F4046' : '1px solid transparent',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            boxShadow: activeWorkspace === 'engineering' ? '0 2px 8px rgba(0, 0, 0, 0.4)' : 'none',
          }}
        >
          <Wrench size={13} color={activeWorkspace === 'engineering' ? '#38BDF8' : '#71717A'} />
          <span>Engineering Studio</span>
          <span
            style={{
              fontSize: 9,
              opacity: activeWorkspace === 'engineering' ? 0.8 : 0.4,
              fontFamily: 'var(--font-mono, monospace)',
              padding: '0 3px',
              borderRadius: 3,
              background: 'rgba(0, 0, 0, 0.25)',
            }}
          >
            ^2
          </span>
        </button>

        <div style={{ height: 16, width: 1, backgroundColor: 'rgba(255, 255, 255, 0.1)', margin: '0 2px' }} />

        {/* Educator Mode Switcher */}
        <button
          onClick={onToggleEducatorMode}
          title="Toggle Educator Mode (Ctrl+E) — Teaches robotics step-by-step for beginners using intuitive analogies & autonomous simulation demonstrations"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '5px 12px',
            borderRadius: 6,
            fontSize: 11.5,
            fontWeight: 700,
            color: isEducatorMode ? '#FFFFFF' : '#94A3B8',
            background: isEducatorMode
              ? 'linear-gradient(135deg, #0284C7 0%, #0369A1 100%)'
              : 'transparent',
            border: isEducatorMode
              ? '1px solid rgba(56, 189, 248, 0.45)'
              : '1px solid transparent',
            cursor: 'pointer',
            transition: 'all 0.18s ease',
            boxShadow: isEducatorMode ? '0 2px 10px rgba(2, 132, 199, 0.45)' : 'none',
          }}
        >
          <span style={{ fontSize: 13 }}>🎓</span>
          <span>Educator Mode</span>
          <span
            style={{
              fontSize: 9,
              padding: '1px 5px',
              borderRadius: 100,
              backgroundColor: isEducatorMode ? 'rgba(255, 255, 255, 0.25)' : 'rgba(255, 255, 255, 0.08)',
              color: isEducatorMode ? '#FFFFFF' : '#64748B',
              fontWeight: 700,
            }}
          >
            {isEducatorMode ? 'ON' : 'OFF'}
          </span>
        </button>
      </div>

      {/* ── Right: Save, Status, Credits & Profile ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        {/* Save Project Button */}
        <button
          onClick={handleSave}
          title="Save project changes (Ctrl+S)"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            borderRadius: 6,
            fontSize: 11,
            fontWeight: 600,
            color: justSaved ? '#4ADE80' : '#D4D4D8',
            background: justSaved ? 'rgba(74, 222, 128, 0.1)' : '#1C1D20',
            border: justSaved ? '1px solid rgba(74, 222, 128, 0.4)' : '1px solid #303136',
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          {justSaved ? <Check size={12} color="#4ADE80" /> : <Save size={12} color="#A1A1AA" />}
          <span>{justSaved ? 'Saved' : 'Save'}</span>
        </button>

        {/* Live Simulation Connection Telemetry Pill */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '4px 8px',
            borderRadius: 6,
            backgroundColor: '#151619',
            border: '1px solid #303136',
            fontSize: 10.5,
            fontFamily: 'var(--font-mono, monospace)',
            color: isConnected ? '#4ADE80' : '#EF4444',
          }}
          title={isConnected ? 'Connected to MuJoCo 3.17 Physics Server (60Hz)' : 'Simulation Server Disconnected'}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: isConnected ? '#4ADE80' : '#EF4444',
              boxShadow: isConnected ? '0 0 6px #4ADE80' : '0 0 6px #EF4444',
            }}
          />
          <span>{isConnected ? `${simState?.status === 'running' ? 'RUNNING' : 'ONLINE'}` : 'OFFLINE'}</span>
        </div>

        {/* User Credits Pill */}
        {availableCredits !== null && (
          <div
            onClick={onShowBilling}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 9px',
              borderRadius: 6,
              backgroundColor: '#1C1D20',
              border: '1px solid #303136',
              fontSize: 11,
              fontWeight: 600,
              color: '#F59E0B',
              cursor: onShowBilling ? 'pointer' : 'default',
            }}
            title="Available AI Credits (Click to upgrade)"
          >
            <Zap size={11} fill="#F59E0B" color="#F59E0B" />
            <span>{availableCredits}</span>
            <span style={{ fontSize: 9.5, color: '#71717A' }}>cr</span>
          </div>
        )}

        {/* User Profile Badge */}
        <UserProfileBadge />
      </div>
    </header>
  );
};

