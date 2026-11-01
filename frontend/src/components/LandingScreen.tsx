import React, { useRef, useState } from 'react';
import {
  Rocket, FolderOpen, Bot, Boxes, Upload, FileCode, ArrowRight, Cpu, Sparkles,
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

/**
 * First-run / new-workspace chooser. Opens before the main workspace so the user starts from a
 * deliberate project (empty + import your own robot, a starter rover, open a saved file, or
 * continue the last session) instead of always landing on a pre-loaded robot.
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

  const card: React.CSSProperties = {
    background: 'linear-gradient(145deg, rgba(30,34,45,0.9), rgba(20,23,31,0.95))',
    border: '1px solid rgba(99,102,241,0.18)',
    borderRadius: 14,
    padding: '22px 22px',
    cursor: 'pointer',
    transition: 'transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  };

  const hover = (e: React.MouseEvent<HTMLDivElement>, on: boolean) => {
    const el = e.currentTarget;
    el.style.transform = on ? 'translateY(-3px)' : 'none';
    el.style.borderColor = on ? 'rgba(99,102,241,0.6)' : 'rgba(99,102,241,0.18)';
    el.style.boxShadow = on ? '0 12px 30px rgba(49,46,129,0.35)' : 'none';
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 10000,
        background:
          'radial-gradient(1200px 600px at 20% -10%, rgba(56,189,248,0.10), transparent), radial-gradient(1000px 500px at 100% 110%, rgba(168,85,247,0.12), transparent), #0b0d12',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24, overflowY: 'auto',
      }}
    >
      <div style={{ width: '100%', maxWidth: 920 }}>
        {/* Brand header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 6 }}>
          <div
            style={{
              width: 44, height: 44, borderRadius: 12,
              background: 'linear-gradient(135deg, #6366f1, #38bdf8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 6px 20px rgba(56,189,248,0.4)',
            }}
          >
            <Cpu size={24} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#f8fafc', letterSpacing: -0.5 }}>
              RoboForge
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>
              Robotics simulation &amp; RL training studio — MuJoCo physics, AI training co-pilot
            </div>
          </div>
        </div>

        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '18px 0 24px' }} />

        {mode === 'home' && (
          <>
            <div style={{ fontSize: 15, color: '#cbd5e1', marginBottom: 16, fontWeight: 600 }}>
              Start a workspace
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              {/* Empty workspace -> import your own robot */}
              <div style={card} onClick={() => setMode('empty')}
                   onMouseEnter={(e) => hover(e, true)} onMouseLeave={(e) => hover(e, false)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Boxes size={22} color="#38bdf8" />
                  <span style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>New Empty Workspace</span>
                </div>
                <div style={{ fontSize: 12.5, color: '#94a3b8', lineHeight: 1.5 }}>
                  Start with an empty scene and import your own robot (URDF / MJCF). Arm, rover,
                  gripper, or custom kinematic tree — per-joint controls are generated automatically.
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#38bdf8', fontSize: 12, fontWeight: 600, marginTop: 4 }}>
                  <span>Choose &amp; import</span><ArrowRight size={14} />
                </div>
              </div>

              {/* Diff-drive starter */}
              <div style={card} onClick={() => onChoose({ kind: 'diff_drive', name: projectName || 'Rover Workspace' })}
                   onMouseEnter={(e) => hover(e, true)} onMouseLeave={(e) => hover(e, false)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Bot size={22} color="#4ade80" />
                  <span style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>Differential-Drive Starter</span>
                </div>
                <div style={{ fontSize: 12.5, color: '#94a3b8', lineHeight: 1.5 }}>
                  A ready-to-drive mobile rover (LiDAR, IMU, encoders) in an empty arena. Great for
                  navigation RL and obstacle-avoidance experiments out of the box.
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#4ade80', fontSize: 12, fontWeight: 600, marginTop: 4 }}>
                  <span>Create &amp; open</span><ArrowRight size={14} />
                </div>
              </div>

              {/* Open saved project */}
              <div style={card} onClick={() => fileInputRef.current?.click()}
                   onMouseEnter={(e) => hover(e, true)} onMouseLeave={(e) => hover(e, false)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <FolderOpen size={22} color="#fbbf24" />
                  <span style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>Open Project File</span>
                </div>
                <div style={{ fontSize: 12.5, color: '#94a3b8', lineHeight: 1.5 }}>
                  Load a saved <code style={{ color: '#fbbf24' }}>.rfproj.json</code> workspace from disk,
                  restoring the robot, scene, sensors, and settings exactly.
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#fbbf24', fontSize: 12, fontWeight: 600, marginTop: 4 }}>
                  <span>Browse file</span><ArrowRight size={14} />
                </div>
                <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }}
                       onChange={(e) => e.target.files?.[0] && handleOpenFile(e.target.files[0])} />
              </div>

              {/* Continue last session */}
              <div
                style={{ ...card, opacity: hasLastSession ? 1 : 0.5, cursor: hasLastSession ? 'pointer' : 'not-allowed' }}
                onClick={() => hasLastSession && onChoose({ kind: 'continue' })}
                onMouseEnter={(e) => hasLastSession && hover(e, true)} onMouseLeave={(e) => hover(e, false)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Rocket size={22} color="#a78bfa" />
                  <span style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9' }}>Continue Last Session</span>
                </div>
                <div style={{ fontSize: 12.5, color: '#94a3b8', lineHeight: 1.5 }}>
                  {hasLastSession
                    ? 'Pick up exactly where you left off — the last autosaved workspace.'
                    : 'No previous session found yet. Create a workspace to begin.'}
                </div>
                {hasLastSession && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#a78bfa', fontSize: 12, fontWeight: 600, marginTop: 4 }}>
                    <span>Resume</span><ArrowRight size={14} />
                  </div>
                )}
              </div>
            </div>

            <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 8, color: '#64748b', fontSize: 11.5 }}>
              <Sparkles size={13} color="#6366f1" />
              <span>Tip: once your robot is in the scene, ask the AI co-pilot to write a reward and train it for you.</span>
            </div>
          </>
        )}

        {mode === 'empty' && (
          <div style={{ maxWidth: 560 }}>
            <button onClick={() => setMode('home')} style={backBtn}>← Back</button>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#f1f5f9', margin: '12px 0 6px' }}>
              New Empty Workspace
            </div>
            <div style={{ fontSize: 12.5, color: '#94a3b8', marginBottom: 16 }}>
              Name your workspace and (optionally) import a robot now. You can also import later from the toolbar.
            </div>
            <label style={label}>Workspace name</label>
            <input value={projectName} onChange={(e) => setProjectName(e.target.value)} style={input} />

            <div style={{ display: 'flex', gap: 10, marginTop: 18 }}>
              <button
                className="rf-btn rf-btn-primary"
                style={{ flex: 1, justifyContent: 'center', gap: 8, padding: '10px' }}
                onClick={() => onChoose({ kind: 'empty', name: projectName || 'Empty Workspace' })}
              >
                <Boxes size={15} /> Create empty scene
              </button>
              <button
                className="rf-btn"
                style={{ flex: 1, justifyContent: 'center', gap: 8, padding: '10px' }}
                onClick={() => setMode('import')}
              >
                <FileCode size={15} /> Import robot now
              </button>
            </div>
          </div>
        )}

        {mode === 'import' && (
          <div style={{ maxWidth: 620 }}>
            <button onClick={() => setMode('empty')} style={backBtn}>← Back</button>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#f1f5f9', margin: '12px 0 6px' }}>
              Import your robot (URDF)
            </div>
            <div style={{ fontSize: 12.5, color: '#94a3b8', marginBottom: 14 }}>
              Paste URDF XML or load a <code style={{ color: '#38bdf8' }}>.urdf</code> file. We'll parse the links,
              joints, limits, and inertials into an empty MuJoCo scene with per-joint controls ready.
            </div>
            <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
              <button className="rf-btn" style={{ gap: 6 }} onClick={() => urdfFileRef.current?.click()}>
                <Upload size={13} /> Load .urdf file
              </button>
              <input ref={urdfFileRef} type="file" accept=".urdf,.xml,.txt" style={{ display: 'none' }}
                     onChange={(e) => e.target.files?.[0] && handleUrdfFile(e.target.files[0])} />
            </div>
            <textarea
              value={urdfText}
              onChange={(e) => setUrdfText(e.target.value)}
              placeholder="<robot name='my_arm'> ... </robot>"
              style={{ ...input, height: 220, fontFamily: 'var(--font-mono, monospace)', fontSize: 11.5, resize: 'vertical' }}
            />
            <button
              className="rf-btn rf-btn-primary"
              style={{ width: '100%', justifyContent: 'center', gap: 8, padding: '10px', marginTop: 12 }}
              disabled={!urdfText.trim()}
              onClick={() => onChoose({ kind: 'import_urdf', name: projectName || 'Imported Robot', urdf: urdfText })}
            >
              <FileCode size={15} /> Create workspace &amp; import robot
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const backBtn: React.CSSProperties = {
  background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 12, padding: 0,
};
const label: React.CSSProperties = { display: 'block', fontSize: 11, color: '#94a3b8', marginBottom: 6 };
const input: React.CSSProperties = {
  width: '100%', background: '#0d0f14', border: '1px solid #2b3140', borderRadius: 8,
  padding: '10px 12px', color: '#f0f6fc', fontSize: 13, outline: 'none',
};
