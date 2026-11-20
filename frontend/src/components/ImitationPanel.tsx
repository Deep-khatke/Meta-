import React, { useEffect, useState } from 'react';
import { X, Circle, Square, Play, Trash2, Crosshair, Bot } from 'lucide-react';

interface Props {
  onClose: () => void;
  onShowToast: (msg: string) => void;
}

const box: React.CSSProperties = {
  border: '1px solid var(--border-subtle, #2b2f31)', borderRadius: 10, padding: 10, marginTop: 8,
  background: 'var(--bg-input, #101213)', fontSize: 12,
};
const muted: React.CSSProperties = { color: 'var(--text-muted, #8a8d89)' };
const row: React.CSSProperties = { display: 'flex', gap: 6, marginTop: 6 };
const full: React.CSSProperties = { flex: 1, justifyContent: 'center' };

async function call(path: string, method = 'POST') {
  const res = await fetch(`/api/imitation/${path}`, { method });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || res.statusText);
  return j;
}

const pct = (v?: number) => (v == null ? '-' : `${Math.round(v * 100)}%`);

export const ImitationPanel: React.FC<Props> = ({ onClose, onShowToast }) => {
  const [st, setSt] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [n, setN] = useState(200);
  const [ft, setFt] = useState(0);

  useEffect(() => {
    let alive = true;
    const tick = () => call('status', 'GET').then((s) => alive && setSt(s)).catch(() => {});
    tick();
    const id = setInterval(tick, 1200);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const run = async (fn: () => Promise<any>) => {
    setBusy(true);
    try { const r = await fn(); onShowToast(r.details || 'Done'); } catch (e: any) { onShowToast(e.message); }
    setBusy(false);
  };

  const tr = st?.training || {};
  const training = tr.phase === 'training';
  const res = tr.results || {};
  const stages: Array<[string, string]> = [
    ['untrained', 'Untrained'], ['behaviour_cloning', 'Behaviour cloning'], ['dagger', 'BC + DAgger'], ['ppo_finetune', 'PPO fine-tune'],
  ];

  return (
    <div className="rf-glass-hud" style={{
      position: 'absolute', bottom: 64, right: 16, width: 300, zIndex: 32, padding: 14,
      color: 'var(--text-primary, #ecebe6)', maxHeight: '72vh', overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong style={{ fontSize: 13 }}>Imitation learning (arm)</strong>
        <button className="rf-btn" onClick={onClose} style={{ padding: 4 }}><X size={13} /></button>
      </div>

      <div style={box}>
        <div style={{ fontWeight: 700 }}>1. Demonstrations</div>
        <div style={muted}>{st ? `${st.samples} samples: ${st.demos.expert} expert, ${st.demos.user} yours` : '...'}
          {st?.expert_success != null && ` (expert success ${pct(st.expert_success)})`}</div>
        <div style={row}>
          <input type="number" min={10} max={500} value={n} onChange={(e) => setN(Number(e.target.value))}
            style={{ width: 64, background: 'transparent', color: 'inherit', border: '1px solid #2b2f31', borderRadius: 6, padding: '4px 6px' }} />
          <button className="rf-btn" style={full} disabled={busy || training} onClick={() => run(() => call(`expert-demos?n=${n}`))}>
            <Bot size={12} />Expert demos</button>
        </div>
        <div style={row}>
          {st?.recording
            ? <button className="rf-btn rf-btn-danger" style={full} onClick={() => run(() => call('record/stop'))}>
                <Square size={11} fill="currentColor" />Stop and save my demo</button>
            : <button className="rf-btn" style={full} disabled={busy || training} onClick={() => run(() => call('record/start'))}>
                <Circle size={11} fill="#ef4444" color="#ef4444" />Record me</button>}
          <button className="rf-btn" disabled={busy || training} title="Delete all demos" onClick={() => run(() => call('demos', 'DELETE'))}>
            <Trash2 size={12} /></button>
        </div>
        {st?.recording && <div style={{ ...muted, marginTop: 6 }}>Move the arm's tip with the joint sliders or gizmo, then stop.</div>}
      </div>

      <div style={box}>
        <div style={{ fontWeight: 700 }}>2. Learn</div>
        <div style={{ ...row, alignItems: 'center' }}>
          <span style={muted}>PPO fine-tune steps</span>
          <input type="number" min={0} step={50000} value={ft} onChange={(e) => setFt(Number(e.target.value))}
            style={{ width: 80, background: 'transparent', color: 'inherit', border: '1px solid #2b2f31', borderRadius: 6, padding: '4px 6px' }} />
        </div>
        <button className="rf-btn rf-btn-primary" style={{ ...full, width: '100%', marginTop: 6 }} disabled={busy || training || !st?.samples}
          onClick={() => run(() => call(`train?dagger_rounds=8&finetune_steps=${ft}`))}>
          <Play size={11} fill="currentColor" />{training ? tr.stage : 'Train from demos'}</button>
        {(training || tr.phase === 'done' || tr.phase === 'error') && (
          <div style={{ marginTop: 6 }}>
            <div style={{ height: 5, background: '#2b2f31', borderRadius: 3 }}>
              <div style={{ height: 5, width: `${Math.round((tr.progress || 0) * 100)}%`, background: '#7aa2ff', borderRadius: 3 }} />
            </div>
            <div style={{ ...muted, marginTop: 4 }}>{tr.message}</div>
            {stages.filter(([k]) => res[k]).map(([k, label]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>{label}{k === 'ppo_finetune' && res[k].kept === false ? ' (not kept)' : ''}</span>
                <span>{pct(res[k].success_rate)} | {res[k].final_distance_m.toFixed(2)} m</span>
              </div>
            ))}
            {Object.keys(res).length > 0 && <div style={{ ...muted, marginTop: 4 }}>Success = tip within 3 cm of a new random goal at the end of a 3 s trial (40 trials).</div>}
          </div>
        )}
      </div>

      <div style={box}>
        <div style={{ fontWeight: 700 }}>3. Use it</div>
        <button className="rf-btn" style={{ ...full, width: '100%', marginTop: 6 }} disabled={busy || training}
          onClick={() => run(() => call('goal'))}><Crosshair size={12} />Reach a new random point</button>
        {st?.goal && <div style={{ ...muted, marginTop: 4 }}>Goal: {st.goal.map((v: number) => v.toFixed(2)).join(', ')} m (green marker){st.tip_distance_m != null && ` | tip is ${(st.tip_distance_m * 100).toFixed(1)} cm away`}</div>}
      </div>
    </div>
  );
};
