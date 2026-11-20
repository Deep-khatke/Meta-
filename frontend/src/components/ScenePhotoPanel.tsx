import React, { useEffect, useRef, useState } from 'react';
import { X, Upload, Play, Square } from 'lucide-react';

interface Props {
  onClose: () => void;
  onShowToast: (msg: string) => void;
}

const box: React.CSSProperties = {
  border: '1px solid var(--border-subtle, #2b2f31)', borderRadius: 10, padding: 10, marginTop: 8,
  background: 'var(--bg-input, #101213)', fontSize: 12,
};
const input: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '6px 8px', borderRadius: 8, fontSize: 12,
  background: 'var(--bg-input, #101213)', color: 'inherit', border: '1px solid var(--border-subtle, #2b2f31)',
};

async function call(path: string, method = 'GET', body?: unknown) {
  const res = await fetch(`/api/scene/${path}`, {
    method, headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined,
  });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || res.statusText);
  return j;
}

export const ScenePhotoPanel: React.FC<Props> = ({ onClose, onShowToast }) => {
  const [ctx, setCtx] = useState<any>(null);
  const [st, setSt] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [hint, setHint] = useState('');
  const [task, setTask] = useState('walk');
  const [target, setTarget] = useState('');
  const [steps, setSteps] = useState(2000000);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => {
      call('context').then((c) => alive && setCtx(c)).catch(() => {});
      call('task/status').then((s) => alive && setSt(s)).catch(() => {});
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const upload = (f: File) => {
    const r = new FileReader();
    r.onload = async () => {
      const url = String(r.result);
      setPreview(url);
      setBusy(true);
      try {
        const res = await call('from-image', 'POST', { image: url, mime: f.type, hint, name: f.name });
        onShowToast(`Scene built: ${res.object_count} objects. Ask the assistant to train the robot.`);
      } catch (e: any) { onShowToast(e.message || 'Image analysis failed'); }
      setBusy(false);
    };
    r.readAsDataURL(f);
  };

  const train = async () => {
    setBusy(true);
    try {
      const res = await call('task/train', 'POST', { task, target: task === 'walk' ? null : target || null, timesteps: steps });
      onShowToast(res.status === 'started' ? res.details : res.details || 'Could not start');
    } catch (e: any) { onShowToast(e.message); }
    setBusy(false);
  };

  const training = st?.phase === 'training' || st?.phase === 'evaluating';
  const labels: string[] = ctx?.active ? Object.keys(ctx.targets || {}) : [];

  return (
    <div style={{
      position: 'absolute',
      bottom: 58,
      right: 18,
      width: 340,
      maxWidth: 'calc(100% - 36px)',
      maxHeight: 'calc(100% - 72px)',
      zIndex: 35,
      padding: '14px 16px',
      color: 'var(--text-primary, #ecebe6)',
      background: 'rgba(21, 23, 26, 0.95)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(255, 255, 255, 0.12)',
      borderRadius: 10,
      boxShadow: '0 12px 36px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.05)',
      overflowY: 'auto',
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        position: 'sticky',
        top: 0,
        backgroundColor: '#15171A',
        paddingBottom: 8,
        marginBottom: 6,
        zIndex: 5,
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      }}>
        <strong style={{ fontSize: 13.5 }}>Train a skill</strong>
        <button className="rf-btn" onClick={onClose} style={{ padding: '3px 6px', borderRadius: 6 }} title="Close"><X size={14} /></button>
      </div>
      <input style={{ ...input, marginTop: 8 }} placeholder="Optional measurements, e.g. each step is 15 cm high"
        value={hint} onChange={(e) => setHint(e.target.value)} />
      <input ref={file} type="file" accept="image/*" style={{ display: 'none' }}
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
      <button className="rf-btn rf-btn-primary" style={{ width: '100%', marginTop: 8, justifyContent: 'center' }}
        disabled={busy || training} onClick={() => file.current?.click()}>
        <Upload size={12} />{busy ? 'Working...' : 'Upload photo of the environment (optional)'}
      </button>
      {preview && <img src={preview} alt="uploaded" style={{ width: '100%', borderRadius: 8, marginTop: 8 }} />}

      {ctx?.active && (
        <div style={box}>
          <div style={{ marginBottom: 6 }}>{ctx.description}</div>
          {Object.entries(ctx.targets as Record<string, number[]>).map(([k, p]) => (
            <div key={k} style={{ color: 'var(--text-muted, #8a8d89)' }}>{k}: {p.map((v) => v.toFixed(2)).join(', ')} m</div>
          ))}
          {ctx.stairs && <div style={{ color: 'var(--text-muted, #8a8d89)' }}>
            stairs: {ctx.stairs.steps} x {(ctx.stairs.step_height * 100).toFixed(0)} cm</div>}
          <div style={{ marginTop: 6, color: 'var(--text-muted, #8a8d89)' }}>The chat assistant knows this scene.</div>
        </div>
      )}

      {(
        <div style={box}>
          <div style={{ display: 'flex', gap: 6 }}>
            <select style={input} value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="walk">Walk forward (no photo needed)</option>
              <option value="climb">Walk / climb to</option>
              <option value="reach">Arm reach / pick</option>
            </select>
            <select style={input} value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">default</option>
              {labels.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <input style={{ ...input, marginTop: 6 }} type="number" min={20000} step={50000} value={steps}
            onChange={(e) => setSteps(Number(e.target.value))} />
          {training
            ? <button className="rf-btn rf-btn-danger" style={{ width: '100%', marginTop: 6, justifyContent: 'center' }}
                onClick={() => call('task/stop', 'POST')}><Square size={11} fill="currentColor" />Stop</button>
            : <button className="rf-btn rf-btn-primary" style={{ width: '100%', marginTop: 6, justifyContent: 'center' }}
                disabled={busy} onClick={train}><Play size={11} fill="currentColor" />Train</button>}
        </div>
      )}

      {st && st.phase !== 'idle' && (
        <div style={box}>
          <div>{st.task} {st.target ? `-> ${st.target}` : ''} ({st.robot})</div>
          <div style={{ height: 5, background: '#2b2f31', borderRadius: 3, margin: '6px 0' }}>
            <div style={{ height: 5, width: `${Math.round((st.progress || 0) * 100)}%`, background: '#7aa2ff', borderRadius: 3 }} />
          </div>
          <div style={{ color: 'var(--text-muted, #8a8d89)' }}>{st.message}{st.mean_reward != null ? ` | reward ${st.mean_reward}` : ''}</div>
          {st.eval && (
            <div style={{ marginTop: 6 }}>
              Success rate: {Math.round(st.eval.untrained.success_rate * 100)}% untrained, {Math.round(st.eval.trained.success_rate * 100)}% trained.
              Mean final distance {st.eval.untrained.final_distance_m.toFixed(2)} m to {st.eval.trained.final_distance_m.toFixed(2)} m.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
