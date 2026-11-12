import React, { useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Play, Pause, RotateCcw, StepForward, Save, FolderOpen,
  FilePlus, RefreshCw, Cpu, Activity, Clock, BrainCircuit,
  Undo2, Redo2, Copy, Download, Upload, Video, Disc,
  Box, Bot, Package, Radio, Compass, Layers, Plus, Sparkles, GitFork, Zap
} from 'lucide-react';
import { Project, SimulationState, WorkspaceSection } from '../types';
import { UserProfileBadge } from './UserProfileBadge';

interface ToolbarProps {
  project: Project;
  simState: SimulationState;
  isConnected: boolean;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onChangeSpeed: (multiplier: number) => void;
  onChangeTimestep: (dt: number) => void;
  onNewProject: () => void;
  onSaveProject: () => void;
  onLoadProject: (jsonContent: string) => void;
  onResetToSaved: () => void;
  onDuplicateProject?: () => void;
  onExportScene?: () => void;
  onOpenURDFModal?: () => void;
  onTrainRobot?: () => void;
  isRecordingTrajectory?: boolean;
  onToggleRecordTrajectory?: () => void;
  activeSection?: WorkspaceSection;
  onSelectSection?: (section: WorkspaceSection) => void;
  onOpenAssetLibrary?: () => void;
  onOpenRobotCreator?: () => void;
  onOpenEnvironmentCreator?: () => void;
  onOpenRobotStudio?: () => void;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  project,
  simState,
  isConnected,
  canUndo = false,
  canRedo = false,
  onUndo,
  onRedo,
  onPlay,
  onPause,
  onStep,
  onReset,
  onChangeSpeed,
  onChangeTimestep,
  onNewProject,
  onSaveProject,
  onLoadProject,
  onResetToSaved,
  onDuplicateProject,
  onExportScene,
  onOpenURDFModal,
  onTrainRobot,
  isRecordingTrajectory = false,
  onToggleRecordTrajectory,
  activeSection = 'scene',
  onSelectSection,
  onOpenAssetLibrary,
  onOpenRobotCreator,
  onOpenEnvironmentCreator,
  onOpenRobotStudio,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content) {
        onLoadProject(content);
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const isRunning = simState.status === 'running';

  return (
    <header className="rf-toolbar" style={{ height: 46, padding: '0 14px', gap: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-subtle)' }}>
      {/* Brand & Project File Quick Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        {/* Sleek Brand Mark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, userSelect: 'none' }}>
          <div style={{
            width: 24,
            height: 24,
            borderRadius: 7,
            background: '#C4472D',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 10px rgba(196, 71, 45, 0.45)'
          }}>
            <Cpu size={13} color="#FFFFFF" />
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
            <span style={{ fontWeight: 800, fontSize: 13, letterSpacing: '0.4px', color: '#F1EEE7' }}>
              ROBO<span style={{ color: '#C4472D' }}>FORGE</span>
            </span>
            <span style={{ fontSize: 9.5, fontWeight: 700, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.12em' }}>
              STUDIO
            </span>
          </div>
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: 'var(--border-subtle)' }} />

        {/* Undo / Redo */}
        <div className="rf-segmented" style={{ padding: 2, gap: 1 }}>
          <button
            className="rf-btn-icon"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo Scene Edit (Ctrl+Z)"
            style={{ width: 22, height: 22, padding: 3, opacity: canUndo ? 0.9 : 0.35, background: 'transparent', border: 'none', cursor: canUndo ? 'pointer' : 'default', color: 'var(--text-primary)' }}
          >
            <Undo2 size={12} />
          </button>
          <button
            className="rf-btn-icon"
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo Scene Edit (Ctrl+Y)"
            style={{ width: 22, height: 22, padding: 3, opacity: canRedo ? 0.9 : 0.35, background: 'transparent', border: 'none', cursor: canRedo ? 'pointer' : 'default', color: 'var(--text-primary)' }}
          >
            <Redo2 size={12} />
          </button>
        </div>

        {/* Project File Actions */}
        <div className="rf-segmented" style={{ padding: '2px 4px', gap: 2 }}>
          <button
            className="rf-segmented-item"
            onClick={onNewProject}
            title="Create New Simulation Project"
            style={{ padding: '3px 8px', fontSize: 10.5 }}
          >
            <FilePlus size={11} />
            <span>New</span>
          </button>
          <button
            className="rf-segmented-item"
            onClick={onSaveProject}
            title="Save Project to Disk"
            style={{ padding: '3px 8px', fontSize: 10.5 }}
          >
            <Save size={11} />
            <span>Save</span>
          </button>
          <button
            className="rf-segmented-item"
            onClick={() => fileInputRef.current?.click()}
            title="Open Project (.rfproj.json)"
            style={{ padding: '3px 8px', fontSize: 10.5 }}
          >
            <FolderOpen size={11} />
            <span>Open</span>
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".json,.rfproj.json"
            style={{ display: 'none' }}
          />
          {onOpenURDFModal && (
            <button
              className="rf-segmented-item"
              onClick={onOpenURDFModal}
              title="Import Robot from URDF XML"
              style={{ padding: '3px 8px', fontSize: 10.5 }}
            >
              <Upload size={11} />
              <span>URDF</span>
            </button>
          )}
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: 'var(--border-subtle)' }} />

        {/* Breadcrumb / Project Name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }} title={project.name}>
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>/</span>
          <span style={{ fontWeight: 500, color: 'var(--text-secondary)', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {project.name}
          </span>
        </div>
      </div>

      {/* Central Workspace Priority Tabs (Segmented CAD Mode Switcher) */}
      <div className="rf-segmented" style={{ padding: '2px 3px', gap: 2, flexShrink: 0 }}>
        <button
          className={`rf-segmented-item ${activeSection === 'scene' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('scene')}
          title="Scene Hierarchy & Physics Viewport"
        >
          <Box size={12} />
          <span>Scene</span>
        </button>

        <button
          className={`rf-segmented-item ${activeSection === 'robots' ? 'active' : ''}`}
          onClick={() => {
            onSelectSection?.('robots');
            onOpenRobotStudio?.();
          }}
          title="Robot Modeling & Control Studio (Links, Joints, FK, IK, Actuators)"
        >
          <GitFork size={12} />
          <span>Robot Studio</span>
        </button>

        <button
          className={`rf-segmented-item ${activeSection === 'assets' ? 'active' : ''}`}
          onClick={() => {
            if (onOpenAssetLibrary) onOpenAssetLibrary();
            else onSelectSection?.('assets');
          }}
          title="Reusable Asset Library (Robots, Obstacles, Sensors, Scene Templates)"
        >
          <Package size={12} />
          <span>Assets</span>
        </button>

        <button
          className={`rf-segmented-item ${activeSection === 'sensors' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('sensors')}
          title="Sensors & Actuators (LiDAR, Camera, IMU Telemetry)"
        >
          <Radio size={12} />
          <span>Sensors</span>
        </button>

        <button
          className={`rf-segmented-item ${activeSection === 'experiments' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('experiments')}
          title="Simulation Scenarios, Waypoint Targets & RL Training"
        >
          <Compass size={12} />
          <span>Experiments</span>
        </button>

        <button
          className={`rf-segmented-item ${activeSection === 'results' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('results')}
          title="Real-time Telemetry, Trajectories & Evaluation Results"
        >
          <Activity size={12} />
          <span>Results</span>
        </button>
      </div>

      {/* Physics Simulation Controls & Status Transport Dock */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        {/* Precision Playback Transport Dock */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            background: 'rgba(23, 25, 28, 0.9)',
            backdropFilter: 'blur(16px)',
            padding: '3px 5px',
            borderRadius: 100,
            border: '1px solid rgba(255, 255, 255, 0.08)',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.25)',
          }}
        >
          {isRunning ? (
            <button
              className="rf-btn"
              onClick={onPause}
              title="Pause Simulation (Space)"
              style={{
                padding: '4px 13px',
                fontSize: 11,
                fontWeight: 700,
                borderRadius: 100,
                backgroundColor: 'rgba(245, 158, 11, 0.16)',
                borderColor: 'rgba(245, 158, 11, 0.4)',
                color: '#fbbf24',
              }}
            >
              <Pause size={12} />
              <span>Pause</span>
            </button>
          ) : (
            <button
              className="rf-btn rf-btn-primary"
              onClick={onPlay}
              title="Start Simulation (Space)"
              style={{
                padding: '4px 13px',
                fontSize: 11,
                fontWeight: 700,
                borderRadius: 100,
                backgroundColor: '#C4472D',
                boxShadow: '0 4px 14px rgba(196, 71, 45, 0.45)',
              }}
            >
              <Play size={12} fill="currentColor" />
              <span>Run</span>
            </button>
          )}

          <button
            className="rf-btn"
            onClick={onStep}
            disabled={isRunning}
            title="Single Step Physics (1 frame)"
            style={{ padding: '3px 8px', fontSize: 11, borderRadius: 100 }}
          >
            <StepForward size={12} />
            <span>Step</span>
          </button>

          <button
            className="rf-btn"
            onClick={onReset}
            title="Reset Simulation to Initial Conditions (R)"
            style={{ padding: '3px 8px', fontSize: 11, borderRadius: 100 }}
          >
            <RotateCcw size={11} />
            <span>Reset</span>
          </button>

          {onToggleRecordTrajectory && (
            <button
              className={`rf-btn ${isRecordingTrajectory ? 'rf-btn-danger' : ''}`}
              onClick={onToggleRecordTrajectory}
              title={isRecordingTrajectory ? 'Stop Trajectory Recording' : 'Record Robot Trajectory Path'}
              style={{ padding: '3px 8px', fontSize: 10, gap: 3, borderRadius: 100 }}
            >
              <Disc size={11} color={isRecordingTrajectory ? '#ef4444' : 'currentColor'} />
              <span>{isRecordingTrajectory ? 'REC' : 'Rec'}</span>
            </button>
          )}

          <div style={{ height: 14, width: 1, backgroundColor: 'rgba(255, 255, 255, 0.08)', margin: '0 2px' }} />

          {/* Speed Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '0 4px' }}>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>SPEED</span>
            <select
              className="rf-select"
              value={project.settings.speed_multiplier}
              onChange={(e) => onChangeSpeed(parseFloat(e.target.value))}
              style={{ padding: '2px 6px', fontSize: 10, height: 22, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 100, color: '#F1EEE7' }}
            >
              <option value="0.2">0.2x</option>
              <option value="0.5">0.5x</option>
              <option value="1.0">1.0x</option>
              <option value="2.0">2.0x</option>
              <option value="5.0">5.0x</option>
            </select>
          </div>
        </div>

        {/* State Badge */}
        <div className={`rf-badge rf-badge-${simState.status}`} style={{ fontSize: 9.5, padding: '3px 8px', letterSpacing: '0.6px', borderRadius: 100 }}>
          <span style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            backgroundColor: isRunning ? '#34D399' : simState.status === 'paused' ? '#FBBF24' : simState.status === 'error' ? '#FB7185' : 'var(--text-muted)'
          }} />
          {simState.status}
        </div>

        {/* Telemetry Clock */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '3px 8px',
            borderRadius: 100,
            background: 'rgba(23, 25, 28, 0.85)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: 10.5,
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)'
          }}
          title="Simulated physics time"
        >
          <Clock size={10} style={{ marginRight: 4, opacity: 0.6 }} />
          <span style={{ color: '#F1EEE7', fontWeight: 600 }}>{simState.sim_time.toFixed(1)}s</span>
        </div>

        {/* WebSocket Connection */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 10,
            padding: '3px 8px',
            borderRadius: 100,
            fontFamily: 'var(--font-mono)',
            background: isConnected ? 'rgba(52, 211, 153, 0.12)' : 'rgba(225, 29, 72, 0.12)',
            border: isConnected ? '1px solid rgba(52, 211, 153, 0.3)' : '1px solid rgba(225, 29, 72, 0.3)',
            color: isConnected ? '#34D399' : '#FB7185',
            fontWeight: 700,
            letterSpacing: '0.04em'
          }}
          title={isConnected ? 'Backend WebSocket Connected' : 'Disconnected'}
        >
          <span style={{ width: 5, height: 5, borderRadius: '50%', backgroundColor: isConnected ? '#34D399' : '#FB7185' }} />
          <span>{isConnected ? 'LIVE' : 'OFFLINE'}</span>
        </div>

        {/* Billing & Credits quick pill */}
        <Link
          to="/billing"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '4px 10px',
            borderRadius: 100,
            backgroundColor: 'rgba(196, 71, 45, 0.12)',
            border: '1px solid rgba(196, 71, 45, 0.25)',
            color: '#F1EEE7',
            textDecoration: 'none',
            fontSize: 11,
            fontFamily: 'IBM Plex Mono, monospace',
            fontWeight: 700,
            letterSpacing: '0.04em',
            transition: 'all 0.15s ease',
          }}
          title="Billing, Plans & Credit Ledger"
        >
          <Zap size={11} color="#C4472D" />
          <span>Credits</span>
        </Link>

        {/* User Profile Badge */}
        <div style={{ marginLeft: 2, display: 'flex', alignItems: 'center' }}>
          <UserProfileBadge variant="dark" />
        </div>
      </div>
    </header>
  );
};
