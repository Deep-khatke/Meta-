import React, { useRef, useState } from 'react';
import {
  Rocket, FolderOpen, Bot, Boxes, Upload, FileCode, ArrowRight, ArrowLeft, Cpu, Sparkles,
} from 'lucide-react';

export type WorkspaceChoice =
  | { kind: 'continue' }
  | { kind: 'empty'; name: string }
  | { kind: 'diff_drive'; name: string }
  | { kind: 'open'; json: string }
  | { kind: 'import_urdf'; name: string; urdf: string };

interface LandingScreenProps {
  hasLastSession: boolean;
  onChoose: (choice: WorkspaceChoice) => void;
}

type StartRow = {
  key: string;
  title: string;
  desc: string;
  Icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
  tag?: string;
  onClick: () => void;
};

/**
 * First-run / new-workspace chooser. Opens before the main workspace so the user starts from a
 * deliberate project (a starter rover, an empty scene to import a robot into, a saved file, or
 * the last session) instead of always landing on a pre-loaded robot.
 */
export const LandingScreen: React.FC<LandingScreenProps> = ({ hasLastSession, onChoose }) => {
  const [mode, setMode] = useState<'home' | 'empty' | 'import'>('home');
  const [projectName, setProjectName] = useState('My Robotics Workspace');
  const [urdfText, setUrdfText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const urdfFileRef = useRef<HTMLInputElement>(null);

  const handleOpenFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => onChoose({ kind: 'open', json: String(reader.result || '') });
    reader.readAsText(file);
  };

  const handleUrdfFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => setUrdfText(String(reader.result || ''));
    reader.readAsText(file);
  };

  const rows: StartRow[] = [];
  if (hasLastSession) {
    rows.push({
      key: 'continue', title: 'Continue last session', Icon: Rocket,
      desc: 'Pick up where you left off with the last autosaved workspace.',
      onClick: () => onChoose({ kind: 'continue' }),
    });
  }
  rows.push(
    {
      key: 'rover', title: 'Differential-drive starter', Icon: Bot, tag: 'Quick start',
      desc: 'A ready-to-drive rover with LiDAR, IMU and encoders in an empty arena. The fastest way to try navigation and RL.',
      onClick: () => onChoose({ kind: 'diff_drive', name: projectName || 'Rover Workspace' }),
    },
    {
      key: 'empty', title: 'New empty workspace', Icon: Boxes,
      desc: 'Start from an empty scene and import your own robot as URDF or MJCF.',
      onClick: () => setMode('empty'),
    },
    {
      key: 'open', title: 'Open project file', Icon: FolderOpen,
      desc: 'Load a saved .rfproj.json workspace and restore the robot, scene and sensors exactly.',
      onClick: () => fileInputRef.current?.click(),
    },
  );

  const capabilities = ['MuJoCo physics', 'PPO training', 'ONNX + ROS 2 export'];

  return (
    <div
      className="rf-start"
      style={{
        position: 'fixed', inset: 0, zIndex: 10000, overflowY: 'auto',
        background: 'var(--bg-app)', color: 'var(--text-primary)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 28px',
      }}
    >
      <div className="rf-start-grid" aria-hidden="true" />

      <div
        style={{
          position: 'relative', width: '100%', maxWidth: 1040,
          display: 'flex', flexWrap: 'wrap', gap: '48px 72px', alignItems: 'center',
        }}
      >
        {/* Left: brand + promise */}
        <div style={{ flex: '1 1 380px', minWidth: 300 }}>
          <div className="rf-rise" style={{ ['--i' as string]: 0, display: 'flex', alignItems: 'center', gap: 12, marginBottom: 36 } as React.CSSProperties}>
            <div
              style={{
                width: 38, height: 38, borderRadius: 10, background: 'var(--bg-raised)',
                border: '1px solid var(--border-strong)', color: 'var(--accent)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <Cpu size={20} strokeWidth={1.75} />
            </div>
            <span style={{ fontSize: 17, fontWeight: 800, letterSpacing: '0.02em' }}>RoboForge</span>
          </div>

          <h1
            className="rf-rise"
            style={{
              ['--i' as string]: 1,
              fontSize: 'clamp(30px, 4vw, 44px)', lineHeight: 1.1, fontWeight: 700,
              letterSpacing: '-0.025em', marginBottom: 18,
            } as React.CSSProperties}
          >
            Build, train and test robots in your browser.
          </h1>
          <p className="rf-rise" style={{ ['--i' as string]: 2, fontSize: 15, lineHeight: 1.6, color: 'var(--text-secondary)', maxWidth: 440, marginBottom: 28 } as React.CSSProperties}>
            Simulate on real physics, watch reinforcement learning improve a policy live, and
            export the result to a real ROS 2 robot.
          </p>

          <div className="rf-rise" style={{ ['--i' as string]: 3, display: 'flex', flexWrap: 'wrap', gap: 8 } as React.CSSProperties}>
            {capabilities.map((c) => (
              <span
                key={c}
                className="rf-mono"
                style={{
                  fontSize: 11.5, padding: '5px 10px', borderRadius: 999,
                  border: '1px solid var(--border-strong)', color: 'var(--text-secondary)',
                }}
              >
                {c}
              </span>
            ))}
          </div>
        </div>

        {/* Right: actions */}
        <div style={{ flex: '1 1 440px', minWidth: 300, maxWidth: 520 }}>
          <div className="rf-float" style={{ padding: 8 }}>
            {mode === 'home' && (
              <>
                <div style={{ padding: '12px 14px 8px', fontSize: 12, fontWeight: 700, color: 'var(--text-muted)' }}>
                  Start a workspace
                </div>
                {rows.map(({ key, title, desc, Icon, tag, onClick }, idx) => (
                  <button
                    key={key}
                    type="button"
                    className="rf-start-row"
                    style={{ ['--i' as string]: idx } as React.CSSProperties}
                    onClick={onClick}
                  >
                    <span className="rf-start-icon"><Icon size={18} strokeWidth={1.75} /></span>
                    <span style={{ flex: 1, minWidth: 0, textAlign: 'left' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, fontWeight: 700 }}>
                        {title}
                        {tag && (
                          <span
                            style={{
                              fontSize: 10.5, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
                              color: 'var(--accent)', background: 'var(--accent-soft)',
                              border: '1px solid var(--accent-line)',
                            }}
                          >
                            {tag}
                          </span>
                        )}
                      </span>
                      <span style={{ display: 'block', marginTop: 3, fontSize: 12.5, lineHeight: 1.5, color: 'var(--text-secondary)', fontWeight: 500 }}>
                        {desc}
                      </span>
                    </span>
                    <ArrowRight size={16} className="rf-start-arrow" />
                  </button>
                ))}
                <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }}
                       onChange={(e) => e.target.files?.[0] && handleOpenFile(e.target.files[0])} />
              </>
            )}

            {mode === 'empty' && (
              <div style={{ padding: 16 }}>
                <button type="button" onClick={() => setMode('home')} className="rf-start-back">
                  <ArrowLeft size={14} /> Back
                </button>
                <div style={{ fontSize: 18, fontWeight: 700, margin: '14px 0 6px' }}>New empty workspace</div>
                <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginBottom: 18, lineHeight: 1.5 }}>
                  Name your workspace and optionally import a robot now. You can also import later from the toolbar.
                </div>
                <label htmlFor="rf-ws-name" style={fieldLabel}>Workspace name</label>
                <input id="rf-ws-name" className="rf-input" value={projectName}
                       onChange={(e) => setProjectName(e.target.value)} style={{ width: '100%', height: 36 }} />

                <div style={{ display: 'flex', gap: 10, marginTop: 20 }}>
                  <button
                    className="rf-btn rf-btn-primary"
                    style={{ flex: 1, height: 36 }}
                    onClick={() => onChoose({ kind: 'empty', name: projectName || 'Empty Workspace' })}
                  >
                    <Boxes size={15} /> Create empty scene
                  </button>
                  <button className="rf-btn" style={{ flex: 1, height: 36 }} onClick={() => setMode('import')}>
                    <FileCode size={15} /> Import robot now
                  </button>
                </div>
              </div>
            )}

            {mode === 'import' && (
              <div style={{ padding: 16 }}>
                <button type="button" onClick={() => setMode('empty')} className="rf-start-back">
                  <ArrowLeft size={14} /> Back
                </button>
                <div style={{ fontSize: 18, fontWeight: 700, margin: '14px 0 6px' }}>Import your robot (URDF)</div>
                <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginBottom: 14, lineHeight: 1.5 }}>
                  Paste URDF XML or load a <code>.urdf</code> file. Links, joints, limits and inertials are parsed
                  into an empty MuJoCo scene with per-joint controls ready.
                </div>
                <button className="rf-btn" onClick={() => urdfFileRef.current?.click()} style={{ marginBottom: 10 }}>
                  <Upload size={13} /> Load .urdf file
                </button>
                <input ref={urdfFileRef} type="file" accept=".urdf,.xml,.txt" style={{ display: 'none' }}
                       onChange={(e) => e.target.files?.[0] && handleUrdfFile(e.target.files[0])} />
                <textarea
                  className="rf-input"
                  aria-label="URDF XML"
                  value={urdfText}
                  onChange={(e) => setUrdfText(e.target.value)}
                  placeholder="<robot name='my_arm'> ... </robot>"
                  style={{ width: '100%', height: 200, padding: 10, fontFamily: 'var(--font-mono)', fontSize: 11.5, resize: 'vertical' }}
                />
                <button
                  className="rf-btn rf-btn-primary"
                  style={{ width: '100%', height: 36, marginTop: 12 }}
                  disabled={!urdfText.trim()}
                  onClick={() => onChoose({ kind: 'import_urdf', name: projectName || 'Imported Robot', urdf: urdfText })}
                >
                  <FileCode size={15} /> Create workspace and import robot
                </button>
              </div>
            )}
          </div>

          <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 12 }}>
            <Sparkles size={13} />
            <span>Once a robot is in the scene, ask the assistant to write a reward and train it for you.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const fieldLabel: React.CSSProperties = {
  display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6,
};
