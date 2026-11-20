import React, { useState } from 'react';
import {
  Plus,
  Search,
  MessageSquare,
  Wrench,
  TrendingUp,
  FolderOpen,
  FileCode,
  Settings,
  ChevronLeft,
  ChevronRight,
  Trash2,
  MoreVertical,
  Compass,
  Cpu
} from 'lucide-react';
import { ArtifactTabType } from './AIArtifactPanel';

export interface ChatSessionMeta {
  id: string;
  title: string;
  timestamp: number;
}

interface AIWorkspaceSidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onNewChat: () => void;
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onDeleteSession?: (id: string) => void;
  onOpenArtifactTab: (tab: ArtifactTabType) => void;
  sessions?: ChatSessionMeta[];
}

export const AIWorkspaceSidebar: React.FC<AIWorkspaceSidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
  onNewChat,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  onOpenArtifactTab,
  sessions = [],
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Default sessions if none passed
  const displaySessions: ChatSessionMeta[] = sessions.length > 0 ? sessions : [
    { id: 'sess_1', title: 'Train PPO Rover on Obstacles', timestamp: Date.now() - 3600 * 1000 },
    { id: 'sess_2', title: '6-DOF Articulated Arm IK Planner', timestamp: Date.now() - 12 * 3600 * 1000 },
    { id: 'sess_3', title: 'Quadcopter Aerodynamic Rotor Flight', timestamp: Date.now() - 28 * 3600 * 1000 },
    { id: 'sess_4', title: 'Warehouse A* Navigation Loop', timestamp: Date.now() - 3 * 86400 * 1000 },
    { id: 'sess_5', title: 'Sim-to-Real Robustness Benchmark', timestamp: Date.now() - 5 * 86400 * 1000 },
  ];

  const filtered = displaySessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Group by date
  const now = Date.now();
  const oneDay = 86400 * 1000;
  const todaySessions = filtered.filter((s) => now - s.timestamp < oneDay);
  const yesterdaySessions = filtered.filter((s) => now - s.timestamp >= oneDay && now - s.timestamp < 2 * oneDay);
  const weekSessions = filtered.filter((s) => now - s.timestamp >= 2 * oneDay);

  if (isCollapsed) {
    return (
      <aside
        style={{
          width: 52,
          minWidth: 52,
          height: '100%',
          backgroundColor: '#151619',
          borderRight: '1px solid #28292E',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '12px 0',
          gap: 12,
          zIndex: 20,
          userSelect: 'none',
        }}
      >
        <button
          onClick={onToggleCollapse}
          title="Expand Sidebar (Ctrl+B)"
          style={{
            width: 32,
            height: 32,
            borderRadius: 7,
            background: '#202227',
            border: '1px solid #303136',
            color: '#A1A1AA',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}
        >
          <ChevronRight size={14} />
        </button>

        <button
          onClick={onNewChat}
          title="New Chat (Ctrl+N)"
          style={{
            width: 32,
            height: 32,
            borderRadius: 7,
            background: 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
            border: 'none',
            color: '#FFFFFF',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(211, 84, 50, 0.4)',
          }}
        >
          <Plus size={16} />
        </button>

        <div style={{ height: 1, width: 28, backgroundColor: '#28292E', margin: '4px 0' }} />

        {/* Quick Artifact Shortcuts */}
        <button
          onClick={() => onOpenArtifactTab('simulation')}
          title="Open Simulation Artifact"
          style={{ width: 32, height: 32, borderRadius: 6, background: 'transparent', border: 'none', color: '#A1A1AA', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
        >
          <Compass size={15} />
        </button>

        <button
          onClick={() => onOpenArtifactTab('robot')}
          title="Open Robot Library"
          style={{ width: 32, height: 32, borderRadius: 6, background: 'transparent', border: 'none', color: '#A1A1AA', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
        >
          <Wrench size={15} />
        </button>

        <button
          onClick={() => onOpenArtifactTab('training')}
          title="Open Training Sessions"
          style={{ width: 32, height: 32, borderRadius: 6, background: 'transparent', border: 'none', color: '#A1A1AA', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
        >
          <TrendingUp size={15} />
        </button>

        <button
          onClick={() => onOpenArtifactTab('files')}
          title="Open Project Files"
          style={{ width: 32, height: 32, borderRadius: 6, background: 'transparent', border: 'none', color: '#A1A1AA', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
        >
          <FileCode size={15} />
        </button>
      </aside>
    );
  }

  return (
    <aside
      className="rf-ai-sidebar"
      style={{
        width: 250,
        minWidth: 250,
        height: '100%',
        backgroundColor: '#151619',
        borderRight: '1px solid #28292E',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 20,
        userSelect: 'none',
      }}
    >
      {/* ── Top Bar: New Chat & Collapse Toggle ── */}
      <div style={{ padding: '12px 14px 8px 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={onNewChat}
          title="Start a new AI conversation (Ctrl+N)"
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 7,
            padding: '7px 12px',
            borderRadius: 8,
            background: 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
            border: 'none',
            color: '#FFFFFF',
            fontSize: 12,
            fontWeight: 700,
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(211, 84, 50, 0.35)',
            transition: 'all 0.15s ease',
          }}
        >
          <Plus size={14} />
          <span>New Chat</span>
        </button>

        <button
          onClick={onToggleCollapse}
          title="Collapse Sidebar"
          style={{
            width: 30,
            height: 30,
            borderRadius: 7,
            background: '#1F2025',
            border: '1px solid #303136',
            color: '#A1A1AA',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}
        >
          <ChevronLeft size={14} />
        </button>
      </div>

      {/* ── Search Bar ── */}
      <div style={{ padding: '4px 14px 10px 14px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            padding: '5px 10px',
            borderRadius: 7,
            background: '#1B1C20',
            border: '1px solid #2B2C32',
          }}
        >
          <Search size={12} color="#71717A" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search conversations…"
            style={{
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: '#F4F4F5',
              fontSize: 11.5,
              width: '100%',
            }}
          />
        </div>
      </div>

      {/* ── Quick Navigation Shortcuts ── */}
      <div style={{ padding: '0 10px 10px 10px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ fontSize: 9.5, fontWeight: 700, color: '#71717A', textTransform: 'uppercase', padding: '0 8px 4px 8px' }}>
          Workspace Views
        </div>

        {[
          { label: 'Simulation Viewport', tab: 'simulation' as ArtifactTabType, icon: <Compass size={13} color="#38BDF8" /> },
          { label: 'Robot Library', tab: 'robot' as ArtifactTabType, icon: <Wrench size={13} color="#D35432" /> },
          { label: 'Training Sessions', tab: 'training' as ArtifactTabType, icon: <TrendingUp size={13} color="#A855F7" /> },
          { label: 'Experiments & Benchmarks', tab: 'results' as ArtifactTabType, icon: <Cpu size={13} color="#4ADE80" /> },
          { label: 'Project Files & Code', tab: 'files' as ArtifactTabType, icon: <FileCode size={13} color="#F59E0B" /> },
        ].map((item) => (
          <button
            key={item.tab}
            onClick={() => onOpenArtifactTab(item.tab)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 9px',
              borderRadius: 6,
              background: 'transparent',
              border: 'none',
              color: '#D4D4D8',
              fontSize: 11.5,
              fontWeight: 500,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'background 0.1s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#222329')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </div>

      <div style={{ height: 1, backgroundColor: '#28292E', margin: '0 14px 10px 14px' }} />

      {/* ── Conversation History Groups ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 10px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {todaySessions.length > 0 && (
          <div>
            <div style={{ fontSize: 9.5, fontWeight: 700, color: '#71717A', textTransform: 'uppercase', padding: '0 8px 4px 8px' }}>
              Today
            </div>
            {todaySessions.map((s) => (
              <ConversationItem
                key={s.id}
                session={s}
                isActive={activeSessionId === s.id}
                onSelect={() => onSelectSession(s.id)}
                onDelete={onDeleteSession ? () => onDeleteSession(s.id) : undefined}
              />
            ))}
          </div>
        )}

        {yesterdaySessions.length > 0 && (
          <div>
            <div style={{ fontSize: 9.5, fontWeight: 700, color: '#71717A', textTransform: 'uppercase', padding: '0 8px 4px 8px' }}>
              Yesterday
            </div>
            {yesterdaySessions.map((s) => (
              <ConversationItem
                key={s.id}
                session={s}
                isActive={activeSessionId === s.id}
                onSelect={() => onSelectSession(s.id)}
                onDelete={onDeleteSession ? () => onDeleteSession(s.id) : undefined}
              />
            ))}
          </div>
        )}

        {weekSessions.length > 0 && (
          <div>
            <div style={{ fontSize: 9.5, fontWeight: 700, color: '#71717A', textTransform: 'uppercase', padding: '0 8px 4px 8px' }}>
              Previous 7 Days
            </div>
            {weekSessions.map((s) => (
              <ConversationItem
                key={s.id}
                session={s}
                isActive={activeSessionId === s.id}
                onSelect={() => onSelectSession(s.id)}
                onDelete={onDeleteSession ? () => onDeleteSession(s.id) : undefined}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Bottom Status ── */}
      <div
        style={{
          padding: '10px 14px',
          borderTop: '1px solid #28292E',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 11,
          color: '#71717A',
        }}
      >
        <span>RoboForge AI v2.5</span>
        <span style={{ fontFamily: 'monospace', color: '#4ADE80' }}>Ready</span>
      </div>
    </aside>
  );
};

const ConversationItem: React.FC<{
  session: ChatSessionMeta;
  isActive: boolean;
  onSelect: () => void;
  onDelete?: () => void;
}> = ({ session, isActive, onSelect, onDelete }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 8px',
        borderRadius: 6,
        background: isActive ? '#26272E' : hovered ? '#1D1E23' : 'transparent',
        cursor: 'pointer',
        transition: 'background 0.1s ease',
      }}
      onClick={onSelect}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, overflow: 'hidden' }}>
        <MessageSquare size={13} color={isActive ? '#D35432' : '#71717A'} style={{ flexShrink: 0 }} />
        <span
          style={{
            fontSize: 11.5,
            fontWeight: isActive ? 600 : 400,
            color: isActive ? '#FFFFFF' : '#D4D4D8',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {session.title}
        </span>
      </div>

      {hovered && onDelete && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          title="Delete conversation"
          style={{ background: 'transparent', border: 'none', color: '#71717A', cursor: 'pointer', padding: 2 }}
        >
          <Trash2 size={12} />
        </button>
      )}
    </div>
  );
};

