import React, { useRef } from 'react';
import {
  Play, Pause, RotateCcw, StepForward, Save, FolderOpen,
  FilePlus, RefreshCw, Cpu, Activity, Clock, BrainCircuit,
  Undo2, Redo2, Copy, Download, Upload, Video, Disc,
  Box, Bot, Package, Radio, Compass, Layers, Plus, Sparkles, GitFork
} from 'lucide-react';
import { Project, SimulationState, WorkspaceSection } from '../types';

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
    <header className="rf-toolbar" style={{ height: 44, padding: '0 10px', gap: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      {/* Brand & Project File Quick Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, letterSpacing: '0.6px', fontSize: 13 }}>
          <Cpu size={16} color="var(--accent-blue)" />
          <span>ROBO<span style={{ color: 'var(--accent-blue)' }}>FORGE</span></span>
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: 'var(--border-subtle)' }} />

        {/* Undo / Redo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <button
            className="rf-btn rf-btn-icon"
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo Scene Edit (Ctrl+Z)"
            style={{ width: 24, height: 24, padding: 3, opacity: canUndo ? 1.0 : 0.4 }}
          >
            <Undo2 size={12} />
          </button>
          <button
            className="rf-btn rf-btn-icon"
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo Scene Edit (Ctrl+Y)"
            style={{ width: 24, height: 24, padding: 3, opacity: canRedo ? 1.0 : 0.4 }}
          >
            <Redo2 size={12} />
          </button>
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: 'var(--border-subtle)' }} />

        {/* Project File Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <button className="rf-btn" onClick={onNewProject} title="Create New Simulation Project" style={{ padding: '3px 7px', fontSize: 11 }}>
            <FilePlus size={11} />
            <span>New</span>
          </button>
          <button className="rf-btn" onClick={onSaveProject} title="Save Project to Disk" style={{ padding: '3px 7px', fontSize: 11 }}>
            <Save size={11} />
            <span>Save</span>
          </button>
          <button
            className="rf-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Open Project (.rfproj.json)"
            style={{ padding: '3px 7px', fontSize: 11 }}
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
              className="rf-btn"
              onClick={onOpenURDFModal}
              title="Import Robot from URDF XML"
              style={{ padding: '3px 8px', fontSize: 11, color: '#38bdf8', borderColor: 'rgba(56, 189, 248, 0.4)' }}
            >
              <Upload size={11} color="#38bdf8" />
              <span>URDF</span>
            </button>
          )}
        </div>

        <div style={{ height: 16, width: 1, backgroundColor: 'var(--border-subtle)' }} />

        {/* Project Name Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={project.name}>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{project.name}</span>
        </div>
      </div>

      {/* Central Workspace Priority Tabs: Intelligent space allocation based on user workflow */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          background: 'rgba(15, 17, 23, 0.85)',
          padding: '2px 4px',
          borderRadius: 6,
          border: '1px solid var(--border-subtle)',
          flexShrink: 0,
        }}
      >
        <button
          className={`rf-workspace-tab ${activeSection === 'scene' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('scene')}
          title="Scene Hierarchy & Physics Viewport"
          style={{ padding: '4px 9px', fontSize: 11, gap: 5 }}
        >
          <Box size={12} />
          <span>Scene</span>
        </button>

        <button
          className={`rf-workspace-tab ${activeSection === 'robots' ? 'active' : ''}`}
          onClick={() => {
            onSelectSection?.('robots');
            onOpenRobotStudio?.();
          }}
          title="Robot Modeling & Control Studio (Links, Joints, FK, IK, Actuators)"
          style={{
            padding: '4px 10px',
            fontSize: 11,
            gap: 5,
            color: activeSection === 'robots' ? '#60a5fa' : undefined,
            borderColor: activeSection === 'robots' ? 'rgba(96, 165, 250, 0.4)' : undefined,
            background: activeSection === 'robots' ? 'rgba(96, 165, 250, 0.15)' : undefined,
          }}
        >
          <GitFork size={12} color={activeSection === 'robots' ? '#60a5fa' : 'currentColor'} />
          <span>Robot Studio</span>
        </button>

        <button
          className={`rf-workspace-tab ${activeSection === 'assets' ? 'active' : ''}`}
          onClick={() => {
            if (onOpenAssetLibrary) onOpenAssetLibrary();
            else onSelectSection?.('assets');
          }}
          title="Reusable Asset Library (Robots, Obstacles, Sensors, Scene Templates)"
          style={{ padding: '4px 9px', fontSize: 11, gap: 5 }}
        >
          <Package size={12} />
          <span>Assets</span>
        </button>

        <button
          className={`rf-workspace-tab ${activeSection === 'sensors' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('sensors')}
          title="Sensors & Actuators (LiDAR, Camera, IMU Telemetry)"
          style={{ padding: '4px 9px', fontSize: 11, gap: 5 }}
        >
          <Radio size={12} />
          <span>Sensors</span>
        </button>

        <button
          className={`rf-workspace-tab ${activeSection === 'experiments' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('experiments')}
          title="Simulation Scenarios, Waypoint Targets & RL Training"
          style={{ padding: '4px 9px', fontSize: 11, gap: 5 }}
        >
          <Compass size={12} />
          <span>Experiments</span>
        </button>

        <button
          className={`rf-workspace-tab ${activeSection === 'results' ? 'active' : ''}`}
          onClick={() => onSelectSection?.('results')}
          title="Real-time Telemetry, Trajectories & Evaluation Results"
          style={{ padding: '4px 9px', fontSize: 11, gap: 5 }}
        >
          <Activity size={12} />
          <span>Results</span>
        </button>
      </div>

      {/* Physics Simulation Controls & Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        {/* Playback Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, background: 'var(--bg-panel)', padding: '2px 4px', borderRadius: 5, border: '1px solid var(--border-subtle)' }}>
          {isRunning ? (
            <button className="rf-btn rf-btn-primary" onClick={onPause} title="Pause Simulation (Space)" style={{ padding: '3px 8px', fontSize: 11 }}>
              <Pause size={12} />
              <span>Pause</span>
            </button>
          ) : (
            <button className="rf-btn rf-btn-primary" onClick={onPlay} title="Start Simulation (Space)" style={{ padding: '3px 8px', fontSize: 11 }}>
              <Play size={12} fill="currentColor" />
              <span>Run</span>
            </button>
          )}

          <button className="rf-btn" onClick={onStep} disabled={isRunning} title="Single Step Physics (1 frame)" style={{ padding: '3px 7px', fontSize: 11 }}>
            <StepForward size={12} />
            <span>Step</span>
          </button>

          <button className="rf-btn" onClick={onReset} title="Reset Simulation to Initial Conditions (R)" style={{ padding: '3px 7px', fontSize: 11 }}>
            <RotateCcw size={11} />
            <span>Reset</span>
          </button>

          {onToggleRecordTrajectory && (
            <button
              className={`rf-btn ${isRecordingTrajectory ? 'rf-btn-danger' : ''}`}
              onClick={onToggleRecordTrajectory}
              title={isRecordingTrajectory ? 'Stop Trajectory Recording' : 'Record Robot Trajectory Path'}
              style={{ padding: '3px 6px', fontSize: 10, gap: 3 }}
            >
              <Disc size={11} color={isRecordingTrajectory ? '#ef4444' : 'currentColor'} />
              <span>{isRecordingTrajectory ? 'REC' : 'Rec'}</span>
            </button>
          )}
        </div>

        {/* Speed Multiplier */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 10, color: 'var(--text-secondary)' }}>
          <span>Speed:</span>
          <select
            className="rf-select"
            value={project.settings.speed_multiplier}
            onChange={(e) => onChangeSpeed(parseFloat(e.target.value))}
            style={{ padding: '2px 4px', fontSize: 10 }}
          >
            <option value="0.2">0.2x</option>
            <option value="0.5">0.5x</option>
            <option value="1.0">1.0x</option>
            <option value="2.0">2.0x</option>
            <option value="5.0">5.0x</option>
          </select>
        </div>

        {/* Status Badge */}
        <div className={`rf-badge rf-badge-${simState.status}`} style={{ fontSize: 10, padding: '2px 5px' }}>
          <span style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            backgroundColor: isRunning ? 'var(--accent-green)' : simState.status === 'paused' ? 'var(--accent-yellow)' : simState.status === 'error' ? 'var(--accent-red)' : 'var(--text-muted)'
          }} />
          {simState.status}
        </div>

        {/* Clocks */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
          <span title="Physics time"><strong style={{ color: 'var(--text-primary)' }}>{simState.sim_time.toFixed(1)}s</strong></span>
        </div>

        {/* Connection */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 10,
            color: isConnected ? 'var(--accent-green)' : 'var(--accent-red)',
          }}
          title={isConnected ? 'Backend WebSocket Connected' : 'Disconnected'}
        >
          <Activity size={11} />
          <span>{isConnected ? 'ON' : 'OFF'}</span>
        </div>
      </div>
    </header>
  );
};
