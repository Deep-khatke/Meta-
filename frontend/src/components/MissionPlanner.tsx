import React, { useEffect, useState } from 'react';
import { MapPin, Box, Cylinder, Circle, Trash2, Play, Brain, Square, Save, FolderOpen, X, ArrowUp, ArrowDown, Crosshair, MousePointer } from 'lucide-react';
import type { Project, SimulationState } from '../types';
import { missionRun, missionTrain, missionSave, missionList, missionLoad, stopMission } from '../services/api';

export type PlacementMode = 'waypoint' | 'box' | 'cylinder' | 'sphere' | 'delete' | null;

interface Props {
  project: Project;
  simState: SimulationState | null;
  waypoints: Array<[number, number]>;
  setWaypoints: (w: Array<[number, number]>) => void;
  mode: PlacementMode;
  setMode: (m: PlacementMode) => void;
  loop: boolean;
  setLoop: (v: boolean) => void;
  onShowToast: (msg: string) => void;
  onClose: () => void;
  onDeleteObstacle: (id: string) => void;
}

const btn = (active: boolean, color = '#3b82f6'): React.CSSProperties => ({
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
  padding: '7px 8px', fontSize: 11.5, fontWeight: 600, borderRadius: 7, cursor: 'pointer',
  border: `1px solid ${active ? color : 'rgba(255,255,255,0.12)'}`,
  background: active ? `${color}33` : 'rgba(255,255,255,0.04)',
  color: active ? '#fff' : '#cbd5e1',
});
const section: React.CSSProperties = { borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, marginTop: 10 };
const label: React.CSSProperties = { fontSize: 10.5, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.4, marginBottom: 6, display: 'block' };

export const MissionPlanner: React.FC<Props> = ({
  project, simState, waypoints, setWaypoints, mode, setMode, loop, setLoop, onShowToast, onClose, onDeleteObstacle,
}) => {
  const [speed, setSpeed] = useState(1.0);
  const [laps, setLaps] = useState(3);
  const [name, setName] = useState('my_mission');
  const [domainRandomize, setDomainRandomize] = useState(false);
  const [altitude, setAltitude] = useState(1.2);
  const [saved, setSaved] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  // Detect whether the mission should fly a drone: a drone is in the scene and no active rover.
  const hasDrone = (project.objects || []).some(
    (o: any) => (o.name || '').toLowerCase().includes('drone') || (o.id || '').startsWith('drone')
  );
  const hasRover = project.robot != null && project.robot_active !== false;
  const isDrone = hasDrone && !hasRover;

  // A drone is the controlled vehicle, not an obstacle — keep it out of the obstacle list.
  const isDroneObj = (o: any) =>
    (o.name || '').toLowerCase().includes('drone') ||
    (o.id || '').startsWith('drone') ||
    (o.name || '').toLowerCase().includes('uav') ||
    (o.name || '').toLowerCase().includes('quad');
  const obstacleObjs = (project.objects || []).filter((o: any) => !isDroneObj(o));

  useEffect(() => { missionList().then(setSaved).catch(() => {}); }, []);

  const payload = () => ({
    name, waypoints, speed, loop, laps, domain_randomize: domainRandomize,
    altitude, vehicle: (isDrone ? 'drone' : 'rover') as 'drone' | 'rover',
  });
  const run = async (kind: 'run' | 'train') => {
    if (waypoints.length < 2) { onShowToast('Place at least 2 waypoints first (click the ground).'); return; }
    setMode(null);
    setBusy(kind);
    try {
      if (kind === 'run') {
        await missionRun(payload());
        onShowToast(isDrone ? 'Drone is flying your path.' : 'Rover is driving your path.');
      } else {
        await missionTrain(payload());
        onShowToast('Training started — watch it learn, then follow your path.');
      }
    } catch (e: any) { onShowToast(e.message || 'Request failed'); }
    finally { setBusy(null); }
  };
  const save = async () => {
    try { const r = await missionSave(payload()); onShowToast(`Saved mission "${r.name}" with scene.`); setSaved(await missionList()); }
    catch (e: any) { onShowToast(e.message || 'Save failed'); }
  };
  const load = async (n: string) => {
    if (!n) return;
    try {
      const m = await missionLoad(n);
      setWaypoints((m.waypoints || []).map((p: any) => [p[0], p[1]] as [number, number]));
      setSpeed(m.speed ?? 1); setLoop(m.loop ?? true); setLaps(m.laps ?? 3); setName(n);
      onShowToast(`Loaded mission "${n}".`);
    } catch (e: any) { onShowToast(e.message || 'Load failed'); }
  };
  const move = (i: number, d: number) => {
    const j = i + d; if (j < 0 || j >= waypoints.length) return;
    const w = [...waypoints]; [w[i], w[j]] = [w[j], w[i]]; setWaypoints(w);
  };
  const addRobotPos = () => {
    const p = simState?.robot?.position;
    if (p) setWaypoints([...waypoints, [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]]);
  };

  const training = !!simState?.rl_training_active;
  const status = simState?.rl_training_status;
  const hint: Record<string, string> = {
    waypoint: 'Click on the ground to drop waypoints in order.',
    box: 'Click on the ground to place a box obstacle.',
    cylinder: 'Click on the ground to place a cylinder.',
    sphere: 'Click on the ground to place a sphere.',
    delete: 'Click near a waypoint or obstacle to remove it.',
  };

  return (
    <div style={{
      position: 'absolute', top: 56, right: 14, width: 320, maxHeight: 'calc(100% - 70px)', overflowY: 'auto',
      zIndex: 30, background: 'rgba(23, 25, 28, 0.95)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 14,
      padding: 16, boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6)', color: '#F1EEE7', fontSize: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <span style={{ fontWeight: 800, fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 7 }}>
          <Crosshair size={15} color="#C4472D" /> <span style={{ color: '#F1EEE7' }}>Mission Planner</span>
          <span style={{ fontSize: 9.5, fontWeight: 700, padding: '2px 7px', borderRadius: 100, background: isDrone ? 'rgba(196, 71, 45, 0.15)' : 'rgba(52, 211, 153, 0.15)', color: isDrone ? '#C4472D' : '#34D399', fontFamily: 'var(--font-mono)' }}>
            {isDrone ? 'DRONE' : 'ROVER'}
          </span>
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#9E9C96', cursor: 'pointer' }} aria-label="Close mission planner"><X size={16} /></button>
      </div>

      {/* 1. Place */}
      <span style={label}>1 · PLACE (click in the 3D view)</span>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
        <button style={btn(mode === 'waypoint', '#f59e0b')} onClick={() => setMode(mode === 'waypoint' ? null : 'waypoint')}><MapPin size={13} />Waypoint</button>
        <button style={btn(mode === 'box', '#ef4444')} onClick={() => setMode(mode === 'box' ? null : 'box')}><Box size={13} />Box</button>
        <button style={btn(mode === 'cylinder', '#ef4444')} onClick={() => setMode(mode === 'cylinder' ? null : 'cylinder')}><Cylinder size={13} />Cylinder</button>
        <button style={btn(mode === 'sphere', '#ef4444')} onClick={() => setMode(mode === 'sphere' ? null : 'sphere')}><Circle size={13} />Sphere</button>
        <button style={btn(mode === 'delete', '#f43f5e')} onClick={() => setMode(mode === 'delete' ? null : 'delete')}><Trash2 size={13} />Delete</button>
        <button style={btn(mode === null, '#64748b')} onClick={() => setMode(null)}><MousePointer size={13} />Select</button>
      </div>
      <div style={{ marginTop: 6, fontSize: 11, color: mode ? '#fbbf24' : '#64748b', minHeight: 15 }}>
        {mode ? hint[mode] : 'Pick a tool, then click the ground.'}
      </div>

      {/* 2. Waypoints */}
      <div style={section}>
        <span style={label}>2 · PATH ({waypoints.length} waypoints)</span>
        {waypoints.length === 0 && <div style={{ color: '#64748b', fontSize: 11 }}>No waypoints yet. Choose Waypoint and click the ground.</div>}
        {waypoints.map(([x, y], i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 0' }}>
            <span style={{ width: 20, height: 20, borderRadius: 10, background: i === 0 ? '#22c55e' : '#f59e0b', color: '#000', fontWeight: 800, fontSize: 11, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{i + 1}</span>
            <span style={{ flex: 1, fontFamily: 'monospace', fontSize: 11 }}>({x.toFixed(2)}, {y.toFixed(2)})</span>
            <button onClick={() => move(i, -1)} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }} aria-label="Move up"><ArrowUp size={13} /></button>
            <button onClick={() => move(i, 1)} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }} aria-label="Move down"><ArrowDown size={13} /></button>
            <button onClick={() => setWaypoints(waypoints.filter((_, k) => k !== i))} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer' }} aria-label="Remove waypoint"><X size={13} /></button>
          </div>
        ))}
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          <button style={{ ...btn(false), flex: 1 }} onClick={addRobotPos}>+ Robot position</button>
          <button style={{ ...btn(false), flex: 1 }} onClick={() => setWaypoints([])}>Clear path</button>
        </div>
      </div>

      {/* 3. Settings */}
      <div style={section}>
        <span style={label}>3 · SETTINGS</span>
        <label style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}><span>Speed</span><b>{speed.toFixed(1)} m/s</b></label>
        <input type="range" min={0.2} max={2.5} step={0.1} value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} style={{ width: '100%' }} aria-label="Speed" />
        {isDrone && (
          <>
            <label style={{ display: 'flex', justifyContent: 'space-between', margin: '8px 0 4px' }}><span>Flight altitude</span><b>{altitude.toFixed(1)} m</b></label>
            <input type="range" min={0.4} max={3.0} step={0.1} value={altitude} onChange={(e) => setAltitude(parseFloat(e.target.value))} style={{ width: '100%' }} aria-label="Flight altitude" />
          </>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
            <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> Loop path
          </label>
          {loop && (<label style={{ display: 'flex', alignItems: 'center', gap: 5 }}>Laps
            <input type="number" min={1} max={50} value={laps} onChange={(e) => setLaps(Math.max(1, parseInt(e.target.value) || 1))}
              style={{ width: 48, background: '#0f172a', color: '#fff', border: '1px solid #334155', borderRadius: 5, padding: '2px 4px' }} />
          </label>)}
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, cursor: 'pointer' }} title="Randomize friction, mass, gravity & add sensor noise each episode so the trained policy survives real-world variation (sim-to-real).">
          <input type="checkbox" checked={domainRandomize} onChange={(e) => setDomainRandomize(e.target.checked)} />
          <span>Domain randomize <span style={{ color: '#64748b' }}>(robust sim-to-real training)</span></span>
        </label>
      </div>

      {/* Obstacles */}
      <div style={section}>
        <span style={label}>OBSTACLES ({obstacleObjs.length})</span>
        {obstacleObjs.length === 0 && <div style={{ color: '#64748b', fontSize: 11 }}>None. Use Box / Cylinder / Sphere above.</div>}
        {obstacleObjs.map((o) => (
          <div key={o.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0' }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: o.color }} />
            <span style={{ flex: 1, fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{o.name} <span style={{ color: '#64748b' }}>({o.position[0].toFixed(1)}, {o.position[1].toFixed(1)})</span></span>
            <button onClick={() => onDeleteObstacle(o.id)} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer' }} aria-label={`Delete ${o.name}`}><Trash2 size={12} /></button>
          </div>
        ))}
      </div>

      {/* 4. Save / Load */}
      <div style={section}>
        <span style={label}>4 · SAVE SCENE + PATH</span>
        <div style={{ display: 'flex', gap: 6 }}>
          <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Mission name"
            style={{ flex: 1, background: '#0f172a', color: '#fff', border: '1px solid #334155', borderRadius: 6, padding: '5px 7px', fontSize: 11.5 }} />
          <button style={btn(false, '#22c55e')} onClick={save}><Save size={13} />Save</button>
        </div>
        {saved.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
            <FolderOpen size={13} color="#94a3b8" />
            <select defaultValue="" onChange={(e) => load(e.target.value)} aria-label="Load mission"
              style={{ flex: 1, background: '#0f172a', color: '#fff', border: '1px solid #334155', borderRadius: 6, padding: '4px', fontSize: 11.5 }}>
              <option value="" disabled>Load saved mission…</option>
              {saved.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* 5. Go */}
      <div style={section}>
        <span style={label}>5 · RUN</span>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          <button disabled={!!busy || training} style={{ ...btn(true, '#C4472D'), opacity: busy || training ? 0.5 : 1, borderRadius: 100 }} onClick={() => run('run')}><Play size={13} />Run Path</button>
          <button disabled={!!busy || training} style={{ ...btn(true, '#A855F7'), opacity: busy || training ? 0.5 : 1, borderRadius: 100 }} onClick={() => run('train')}><Brain size={13} />Train &amp; Run</button>
        </div>
        <button style={{ ...btn(true, '#E11D48'), width: '100%', marginTop: 6, borderRadius: 100 }} onClick={() => stopMission().then(() => onShowToast('Stopped.'))}><Square size={12} />Stop</button>
        <div style={{ marginTop: 8, fontSize: 10.5, color: '#94a3b8', lineHeight: 1.4 }}>
          <b>Run Path</b> drives it now. <b>Train &amp; Run</b> trains a PPO policy live (saved to the PPO tab), then follows your path.
        </div>
        {status && <div style={{ marginTop: 6, fontSize: 10.5, color: '#c084fc', fontFamily: 'monospace' }}>{status}</div>}
      </div>
    </div>
  );
};
