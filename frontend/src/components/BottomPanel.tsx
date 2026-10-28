import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Terminal, ChevronUp, ChevronDown, Trash2,
  Copy, Check, AlertTriangle, Info, AlertCircle, FlaskConical,
  ShieldAlert, Play, Pause, RotateCcw, CircleDot, Brain, Sparkles
} from 'lucide-react';
import { SimulationState, LogMessage, Project, ExperimentRunResult } from '../types';
import { ExperimentPanel } from './ExperimentPanel';
import { ResearchDashboard } from './ResearchDashboard';
import { AIAssistantChat } from './AIAssistantChat';

export interface TrajectoryPoint {
  time: number;
  position: [number, number, number];
  orientation: [number, number, number, number];
  linear_speed: number;
}

interface BottomPanelProps {
  simState: SimulationState;
  logs: LogMessage[];
  onClearLogs: () => void;
  currentProject?: Project | null;
  onShowToast?: (msg: string) => void;
  onProjectLoaded?: () => void;
  onSelectTarget?: (pos: [number, number, number] | null) => void;
  onSetTrail?: (trail: Array<[number, number, number]> | null) => void;
  activeLiveRun?: ExperimentRunResult | null;
  targetPosition?: [number, number, number] | null;
  recordedTrajectory?: TrajectoryPoint[];
  isRecordingTrajectory?: boolean;
  onToggleTrajectoryRecording?: () => void;
  replayIndex?: number | null;
  onSeekReplay?: (index: number | null) => void;
  forcedTab?: 'telemetry' | 'contacts' | 'console' | 'experiments' | null;
}

interface TelemetryPoint {
  time: number;
  linearSpeed: number;
  angularSpeed: number;
}

export const BottomPanel: React.FC<BottomPanelProps> = ({
  simState,
  logs,
  onClearLogs,
  currentProject = null,
  onShowToast = () => {},
  onProjectLoaded,
  onSelectTarget,
  onSetTrail,
  activeLiveRun,
  targetPosition,
  recordedTrajectory = [],
  isRecordingTrajectory = false,
  onToggleTrajectoryRecording,
  replayIndex = null,
  onSeekReplay,
  forcedTab = null,
}) => {
  const [isExpanded, setIsExpanded] = useState(Boolean(forcedTab));
  const [activeTab, setActiveTab] = useState<'telemetry' | 'contacts' | 'console' | 'experiments' | 'research' | 'ai_studio'>('experiments');
  const [copied, setCopied] = useState(false);
  const [logFilter, setLogFilter] = useState<'all' | 'info' | 'warning' | 'error'>('all');
  const [panelHeight, setPanelHeight] = useState<number>(240);
  const [isReplaying, setIsReplaying] = useState(false);

  useEffect(() => {
    if (forcedTab) {
      setActiveTab(forcedTab);
      setIsExpanded(true);
    }
  }, [forcedTab]);

  // Drag resizer refs
  const isDraggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHeightRef = useRef(310);

  const handleResizePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    startYRef.current = e.clientY;
    startHeightRef.current = panelHeight;

    const onPointerMove = (ev: PointerEvent) => {
      if (!isDraggingRef.current) return;
      const dy = startYRef.current - ev.clientY;
      const newHeight = Math.max(150, Math.min(650, startHeightRef.current + dy));
      setPanelHeight(newHeight);
    };

    const onPointerUp = () => {
      isDraggingRef.current = false;
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  };

  // History buffer for real-time telemetry graph
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryPoint[]>([]);
  const lastStateTime = useRef<number>(0);
  const totalDistance = useRef<number>(0);
  const lastPos = useRef<[number, number, number]>([0, 0, 0]);

  useEffect(() => {
    if (!simState || !simState.robot) return;

    const r = simState.robot;
    const curTime = simState.sim_time;

    // Calculate distance traveled
    if (lastPos.current) {
      const dx = r.position[0] - lastPos.current[0];
      const dy = r.position[1] - lastPos.current[1];
      const distStep = Math.sqrt(dx * dx + dy * dy);
      if (distStep < 0.5) { // ignore teleport jumps on reset
        totalDistance.current += distStep;
      }
    }
    lastPos.current = [...r.position];

    // Sample telemetry at ~10-20Hz
    if (Math.abs(curTime - lastStateTime.current) > 0.05) {
      lastStateTime.current = curTime;
      setTelemetryHistory((prev) => {
        const next = [...prev, {
          time: curTime,
          linearSpeed: r.linear_speed,
          angularSpeed: r.angular_speed,
        }];
        return next.length > 80 ? next.slice(next.length - 80) : next;
      });
    }
  }, [simState]);

  const handleCopyLogs = () => {
    const text = logs
      .map((l) => `[${new Date(l.timestamp * 1000).toLocaleTimeString()}] [${l.level.toUpperCase()}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const filteredLogs = logs.filter((l) => logFilter === 'all' || l.level === logFilter);

  // Compute SVG Plot geometry
  const maxSpeed = Math.max(2.0, ...telemetryHistory.map((d) => d.linearSpeed));
  const svgWidth = 420;
  const svgHeight = 90;
  const padding = 15;

  const pointsString = telemetryHistory
    .map((pt, idx) => {
      const x = padding + (idx / Math.max(1, telemetryHistory.length - 1)) * (svgWidth - padding * 2);
      const y = svgHeight - padding - (pt.linearSpeed / maxSpeed) * (svgHeight - padding * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <div
      className="rf-bottom-panel"
      style={{ height: isExpanded ? panelHeight : 34, position: 'relative', transition: isDraggingRef.current ? 'none' : 'height 0.15s ease' }}
    >
      {/* Top Drag Resize Handle */}
      {isExpanded && (
        <div
          onPointerDown={handleResizePointerDown}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            cursor: 'ns-resize',
            zIndex: 10,
            backgroundColor: 'transparent',
          }}
          title="Drag to resize bottom panel height"
        />
      )}

      {/* Panel Tab Header */}
      <div
        className="rf-panel-header"
        style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Tab Selectors */}
          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('telemetry');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'telemetry' ? 'var(--bg-panel-hover)' : 'transparent',
              color: activeTab === 'telemetry' ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            <Activity size={13} />
            <span>Telemetry</span>
          </div>

          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('console');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'console' ? 'var(--bg-panel-hover)' : 'transparent',
              color: activeTab === 'console' ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            <Terminal size={13} />
            <span>Console</span>
            {logs.some((l) => l.level === 'error') && (
              <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'var(--accent-red)' }} />
            )}
          </div>

          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('contacts');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'contacts' ? 'var(--bg-panel-hover)' : 'transparent',
              color: activeTab === 'contacts' ? 'var(--accent-orange)' : 'var(--text-secondary)',
            }}
          >
            <ShieldAlert size={13} />
            <span>Contacts</span>
            {simState.contacts.length > 0 && (
              <span style={{
                fontSize: 10,
                padding: '0 4px',
                borderRadius: 8,
                backgroundColor: 'var(--accent-orange)',
                color: '#fff',
                fontWeight: 600
              }}>
                {simState.contacts.length}
              </span>
            )}
          </div>

          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('experiments');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'experiments' ? 'var(--bg-panel-hover)' : 'transparent',
              color: activeTab === 'experiments' ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            <FlaskConical size={13} />
            <span>Experiments</span>
          </div>

          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('research');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'research' ? 'var(--bg-panel-hover)' : 'transparent',
              color: activeTab === 'research' ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            <Brain size={13} />
            <span>Research Agent</span>
          </div>

          <div
            onClick={(e) => {
              e.stopPropagation();
              setActiveTab('ai_studio');
              setIsExpanded(true);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: activeTab === 'ai_studio' ? 'rgba(168, 85, 247, 0.2)' : 'transparent',
              color: activeTab === 'ai_studio' ? '#c084fc' : 'var(--text-secondary)',
              cursor: 'pointer',
            }}
            title="RoboForge Autonomous AI Studio (Natural Language & Self-Driving Experimentation)"
          >
            <Sparkles size={13} color="#c084fc" />
            <span>Autonomous AI</span>
          </div>
        </div>

        {/* Trajectory Playback Scrubber & Quick Control */}
        {recordedTrajectory.length > 0 && (
          <div
            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 8px' }}
            onClick={(e) => e.stopPropagation()}
          >
            <span style={{ fontSize: 10, color: 'var(--accent-orange)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <CircleDot size={10} className={isRecordingTrajectory ? 'animate-pulse' : ''} />
              {isRecordingTrajectory ? 'REC' : 'TRAJ'} ({recordedTrajectory.length} pts)
            </span>
            <input
              type="range"
              min={0}
              max={recordedTrajectory.length - 1}
              value={replayIndex ?? recordedTrajectory.length - 1}
              onChange={(e) => onSeekReplay?.(parseInt(e.target.value, 10))}
              style={{ width: 140, accentColor: 'var(--accent-orange)', height: 4, cursor: 'pointer' }}
              title="Trajectory scrub position"
            />
            {replayIndex !== null && (
              <button
                className="rf-btn"
                style={{ fontSize: 10, padding: '1px 5px' }}
                onClick={() => onSeekReplay?.(null)}
                title="Exit replay & return to live simulation"
              >
                Live
              </button>
            )}
          </div>
        )}

        {/* Panel Height Preset Controls & Expand/Collapse */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isExpanded && (
            <div
              style={{ display: 'flex', alignItems: 'center', gap: 4 }}
              onClick={(e) => e.stopPropagation()}
            >
              <button
                className="rf-btn"
                onClick={() => setPanelHeight(190)}
                style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  backgroundColor: panelHeight <= 210 ? 'var(--accent-blue)' : 'transparent',
                  color: panelHeight <= 210 ? '#fff' : 'var(--text-muted)'
                }}
                title="Compact Height (190px)"
              >
                Compact
              </button>
              <button
                className="rf-btn"
                onClick={() => setPanelHeight(310)}
                style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  backgroundColor: panelHeight > 210 && panelHeight <= 380 ? 'var(--accent-blue)' : 'transparent',
                  color: panelHeight > 210 && panelHeight <= 380 ? '#fff' : 'var(--text-muted)'
                }}
                title="Comfortable Height (310px)"
              >
                Comfort
              </button>
              <button
                className="rf-btn"
                onClick={() => setPanelHeight(480)}
                style={{
                  fontSize: 10,
                  padding: '1px 6px',
                  backgroundColor: panelHeight > 380 ? 'var(--accent-blue)' : 'transparent',
                  color: panelHeight > 380 ? '#fff' : 'var(--text-muted)'
                }}
                title="Expanded Height (480px)"
              >
                Expanded
              </button>
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)' }}>
            {isExpanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </div>
        </div>
      </div>

      {/* Expanded Content Body */}
      {isExpanded && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', padding: '6px 12px' }}>
          {activeTab === 'ai_studio' ? (
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <AIAssistantChat
                onShowToast={onShowToast}
                selectedScenarioId="scen_slalom_beta"
                targetPosition={targetPosition}
                onProjectLoaded={onProjectLoaded}
                onSelectTarget={onSelectTarget}
                onSwitchView={(v) => {
                  if (v === 'hypotheses') setActiveTab('research');
                  else setActiveTab('experiments');
                }}
              />
            </div>
          ) : activeTab === 'research' ? (
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <ResearchDashboard onShowToast={onShowToast} />
            </div>
          ) : activeTab === 'experiments' ? (
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <ExperimentPanel
                currentProject={currentProject}
                onShowToast={onShowToast}
                onProjectLoaded={onProjectLoaded}
                onSelectTarget={onSelectTarget}
                onSetTrail={onSetTrail}
                activeLiveRun={activeLiveRun}
                targetPosition={targetPosition}
              />
            </div>
          ) : activeTab === 'telemetry' ? (
            <div style={{ flex: 1, display: 'flex', gap: 16, alignItems: 'center', overflow: 'hidden' }}>
              {/* Telemetry Real-time SVG Plot */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>LINEAR VELOCITY PROFILE (m/s)</span>
                  <span>Max: {maxSpeed.toFixed(1)} m/s</span>
                </div>
                <div
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    borderRadius: 4,
                    border: '1px solid var(--border-subtle)',
                    width: svgWidth,
                    height: svgHeight,
                  }}
                >
                  <svg width={svgWidth} height={svgHeight}>
                    {/* Background Grid Lines */}
                    <line x1={padding} y1={svgHeight - padding} x2={svgWidth - padding} y2={svgHeight - padding} stroke="var(--border-subtle)" strokeWidth="1" />
                    <line x1={padding} y1={svgHeight / 2} x2={svgWidth - padding} y2={svgHeight / 2} stroke="var(--border-subtle)" strokeDasharray="3 3" strokeWidth="1" />
                    <line x1={padding} y1={padding} x2={svgWidth - padding} y2={padding} stroke="var(--border-subtle)" strokeWidth="1" />

                    {/* Speed Polyline */}
                    {telemetryHistory.length > 1 && (
                      <polyline
                        fill="none"
                        stroke="var(--accent-blue)"
                        strokeWidth="2"
                        points={pointsString}
                      />
                    )}
                  </svg>
                </div>
              </div>

              {/* Numerical Metrics Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, flex: 1 }}>
                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Current Speed</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)' }}>
                    {simState.robot.linear_speed.toFixed(2)} <span style={{ fontSize: 11, fontWeight: 400 }}>m/s</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Yaw Rate</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>
                    {simState.robot.angular_speed.toFixed(2)} <span style={{ fontSize: 11, fontWeight: 400 }}>rad/s</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Distance</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    {totalDistance.current.toFixed(2)} <span style={{ fontSize: 11, fontWeight: 400 }}>m</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Active Contacts</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: simState.contacts.length > 0 ? 'var(--accent-orange)' : 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    {simState.contacts.length}
                  </div>
                </div>

                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Physics Steps</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    {simState.step_count}
                  </div>
                </div>

                <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Sim Time</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    {simState.sim_time.toFixed(2)} <span style={{ fontSize: 11, fontWeight: 400 }}>s</span>
                  </div>
                </div>
                {/* Sensor & Telemetry Metrics */}
                {(simState.imu_linear_acceleration || simState.lidar_readings) && (
                  <div style={{ background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 4, border: '1px solid var(--border-subtle)', gridColumn: 'span 3' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>Live Sensor Data (Authoritative MuJoCo)</div>
                    <div style={{ display: 'flex', gap: 16, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                      {simState.imu_linear_acceleration && (
                        <div>
                          <span style={{ color: 'var(--text-secondary)' }}>IMU Accel: </span>
                          <span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>
                            [{simState.imu_linear_acceleration.map((v: number) => v.toFixed(2)).join(', ')}] m/s²
                          </span>
                        </div>
                      )}
                      {simState.imu_angular_velocity && (
                        <div>
                          <span style={{ color: 'var(--text-secondary)' }}>IMU Gyro: </span>
                          <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
                            [{simState.imu_angular_velocity.map((v: number) => v.toFixed(2)).join(', ')}] rad/s
                          </span>
                        </div>
                      )}
                      {simState.lidar_readings && simState.lidar_readings.length > 0 && (
                        <div>
                          <span style={{ color: 'var(--text-secondary)' }}>LiDAR: </span>
                          <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>
                            {simState.lidar_readings.length} beams (min: {Math.min(...simState.lidar_readings).toFixed(2)}m)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : activeTab === 'contacts' ? (
            /* Collision & Contacts Inspector */
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 6 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <ShieldAlert size={14} color={simState.contacts.length > 0 ? 'var(--accent-red)' : 'var(--accent-green)'} />
                  <span>Authoritative MuJoCo Contact Pairs ({simState.contacts.length} Active)</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Visualized as red contact spheres in 3D viewport
                </div>
              </div>

              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  backgroundColor: 'var(--bg-primary)',
                  borderRadius: 4,
                  border: '1px solid var(--border-subtle)',
                  padding: 8,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                }}
              >
                {simState.contacts.length === 0 ? (
                  <div style={{ padding: '16px 8px', color: 'var(--text-muted)', textAlign: 'center' }}>
                    No active collisions or contacts detected. Physics manifold is clear.
                  </div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: 10 }}>
                        <th style={{ padding: '4px 8px' }}>#</th>
                        <th style={{ padding: '4px 8px' }}>GEOM 1</th>
                        <th style={{ padding: '4px 8px' }}>GEOM 2</th>
                        <th style={{ padding: '4px 8px' }}>PENETRATION / DIST</th>
                        <th style={{ padding: '4px 8px' }}>CONTACT POSITION [X, Y, Z]</th>
                      </tr>
                    </thead>
                    <tbody>
                      {simState.contacts.map((c, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', color: 'var(--text-primary)' }}>
                          <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>{i + 1}</td>
                          <td style={{ padding: '6px 8px', color: 'var(--accent-blue)', fontWeight: 600 }}>{c.geom1}</td>
                          <td style={{ padding: '6px 8px', color: 'var(--accent-orange)', fontWeight: 600 }}>{c.geom2}</td>
                          <td style={{ padding: '6px 8px', color: c.distance < 0 ? 'var(--accent-red)' : 'var(--accent-yellow)' }}>
                            {c.distance < 0 ? `Penetration: ${(-c.distance * 1000).toFixed(1)} mm` : `${(c.distance).toFixed(3)} m`}
                          </td>
                          <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                            [{c.position.map(v => v.toFixed(3)).join(', ')}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          ) : (
            /* Console View */
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {/* Console Toolbar */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: 6 }}>
                <div style={{ display: 'flex', gap: 6 }}>
                  {(['all', 'info', 'warning', 'error'] as const).map((lvl) => (
                    <button
                      key={lvl}
                      className="rf-btn"
                      style={{
                        padding: '2px 8px',
                        fontSize: 10,
                        textTransform: 'uppercase',
                        backgroundColor: logFilter === lvl ? 'var(--accent-blue)' : 'var(--bg-panel)',
                        color: logFilter === lvl ? '#fff' : 'var(--text-secondary)',
                      }}
                      onClick={() => setLogFilter(lvl)}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="rf-btn" onClick={handleCopyLogs} title="Copy Log Output">
                    {copied ? <Check size={12} color="var(--accent-green)" /> : <Copy size={12} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                  <button className="rf-btn rf-btn-danger" onClick={onClearLogs} title="Clear Console">
                    <Trash2 size={12} />
                    <span>Clear</span>
                  </button>
                </div>
              </div>

              {/* Log Messages Output */}
              <div
                style={{
                  flex: 1,
                  overflowY: 'auto',
                  backgroundColor: 'var(--bg-primary)',
                  borderRadius: 4,
                  border: '1px solid var(--border-subtle)',
                  padding: 8,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 3,
                }}
              >
                {filteredLogs.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)' }}>No logs to display.</div>
                ) : (
                  filteredLogs.map((log, idx) => {
                    const color =
                      log.level === 'error'
                        ? 'var(--accent-red)'
                        : log.level === 'warning'
                        ? 'var(--accent-yellow)'
                        : 'var(--text-secondary)';

                    return (
                      <div key={idx} style={{ display: 'flex', gap: 8, color }}>
                        <span style={{ color: 'var(--text-muted)' }}>
                          {new Date(log.timestamp * 1000).toLocaleTimeString()}
                        </span>
                        <span style={{ fontWeight: 600 }}>[{log.level.toUpperCase()}]</span>
                        <span style={{ color: 'var(--text-primary)' }}>{log.message}</span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
