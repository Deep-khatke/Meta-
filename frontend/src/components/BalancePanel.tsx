import React, { useEffect, useState } from 'react';
import { X, Play, Square, RotateCcw, Zap, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Shuffle } from 'lucide-react';
import {
  balanceStatus, balanceSpawn, balancePush, balanceTrain, balanceStop, balanceReset, balanceUntrain,
} from '../services/api';

interface Props {
  onClose: () => void;
  onShowToast: (msg: string) => void;
}

const card: React.CSSProperties = {
  border: '1px solid var(--border-subtle, #2b2f31)', borderRadius: 10, padding: 10, marginTop: 8,
  background: 'var(--bg-input, #101213)',
};
const label: React.CSSProperties = { fontSize: 11, fontWeight: 700, color: 'var(--text-muted, #8a8d89)', marginBottom: 6, display: 'block' };

function Sparkline({ points }: { points: Array<[number, number]> }) {
  if (points.length < 2) return <div style={{ height: 34 }} />;
  const xs = points.map((p) => p[0]); const ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const d = points.map(([x, y]) => `${(((x - x0) / (x1 - x0 || 1)) * 236 + 2).toFixed(1)},${(30 - ((y - y0) / (y1 - y0 || 1)) * 26).toFixed(1)}`).join(' ');
  return (
    <svg width="240" height="34" aria-label="Mean episode reward while training">
      <polyline points={d} fill="none" stroke="var(--accent, #e8622c)" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}

export const BalancePanel: React.FC<Props> = ({ onClose, onShowToast }) => {
  const [st, setSt] = useState<any>(null);
  const [kind, setKind] = useState<'dog' | 'humanoid'>('dog');
  const [steps, setSteps] = useState(250000);
  const [strength, setStrength] = useState(220);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = () => balanceStatus().then((s) => alive && setSt(s)).catch(() => {});
    tick();
    const id = window.setInterval(tick, 1000);
    return () => { alive = false; window.clearInterval(id); };
  }, []);

  const run = async (fn: () => Promise<any>, ok?: string) => {
    setBusy(true);
    try { const r = await fn(); if (r?.status === 'error' || r?.status === 'busy') onShowToast(r.details || 'Request failed'); else if (ok) onShowToast(ok); }
    catch (e: any) { onShowToast(e.message || 'Request failed'); }
    finally { setBusy(false); }
  };

  const phase = st?.phase || 'idle';
  const training = phase === 'training' || phase === 'evaluating';
  const standing = !!st?.free_standing;
  const trained = !!st?.controller_active;
  const badge = training ? `Training ${Math.round((st?.progress || 0) * 100)}%` : trained ? 'Trained' : standing ? 'Untrained' : 'No robot';
  const badgeColor = training ? '#dca23a' : trained ? '#3fb27f' : '#a9aca7';
  const btn = (primary = false): React.CSSProperties => ({ flex: 1, justifyContent: 'center', ...(primary ? {} : {}) });

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
      fontSize: 12,
      overflowY: 'auto',
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        backgroundColor: '#15171A',
        paddingBottom: 8,
        marginBottom: 6,
        zIndex: 5,
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      }}>
        <span style={{ fontWeight: 800, fontSize: 13.5, display: 'flex', alignItems: 'center', gap: 8 }}>
          Balance lab
          <span style={{ fontSize: 10.5, fontWeight: 700, padding: '2px 8px', borderRadius: 4, color: badgeColor, border: `1px solid ${badgeColor}55`, background: `${badgeColor}15` }}>{badge}</span>
        </span>
        <button onClick={onClose} aria-label="Close balance lab" style={{ background: 'none', border: 'none', color: 'var(--text-muted, #8a8d89)', cursor: 'pointer', padding: '3px 6px' }}><X size={16} /></button>
      </div>
      <div style={{ color: 'var(--text-muted, #8a8d89)', marginTop: 4, lineHeight: 1.45 }}>
        Spawn a free-standing robot, push it, then train it with PPO to stay upright.
      </div>

      <div style={card}>
        <span style={label}>1 · Robot</span>
        <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
          {(['dog', 'humanoid'] as const).map((k) => (
            <button key={k} className={`rf-btn ${kind === k ? 'rf-btn-active' : ''}`} style={btn()} onClick={() => setKind(k)}>
              {k === 'dog' ? 'Robot dog' : 'Humanoid'}
            </button>
          ))}
        </div>
        <button className="rf-btn" style={{ width: '100%', justifyContent: 'center' }} disabled={busy || training}
          onClick={() => run(() => balanceSpawn(kind), 'Spawned an untrained robot. Try pushing it.')}>
          Spawn untrained {kind === 'dog' ? 'dog' : 'humanoid'}
        </button>
      </div>

      <div style={card}>
        <span style={label}>2 · Push ({strength} N for 0.1 s)</span>
        <input type="range" min={60} max={400} step={10} value={strength} onChange={(e) => setStrength(parseInt(e.target.value, 10))}
          aria-label="Push strength" style={{ width: '100%' }} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, marginTop: 6 }}>
          {[
            { a: 180, i: <ArrowLeft size={13} />, t: 'Push backward' }, { a: 0, i: <ArrowRight size={13} />, t: 'Push forward' },
            { a: 90, i: <ArrowUp size={13} />, t: 'Push left' }, { a: 270, i: <ArrowDown size={13} />, t: 'Push right' },
          ].map((b) => (
            <button key={b.t} className="rf-btn" title={b.t} aria-label={b.t} disabled={!standing || busy}
              style={{ justifyContent: 'center', padding: 0 }} onClick={() => run(() => balancePush(b.a, strength))}>{b.i}</button>
          ))}
          <button className="rf-btn" title="Push in a random direction" aria-label="Random push" disabled={!standing || busy}
            style={{ justifyContent: 'center', padding: 0 }} onClick={() => run(() => balancePush(undefined, strength))}><Shuffle size={13} /></button>
        </div>
      </div>

      <div style={card}>
        <span style={label}>3 · Train to stay upright</span>
        <div style={{ display: 'flex', gap: 6 }}>
          <select className="rf-select" aria-label="Training length" value={steps} onChange={(e) => setSteps(parseInt(e.target.value, 10))} disabled={training} style={{ flex: 1 }}>
            <option value={100000}>Quick · 100k steps (~40 s)</option>
            <option value={250000}>Standard · 250k (~100 s)</option>
            <option value={600000}>Long · 600k (~4 min)</option>
          </select>
          {training
            ? <button className="rf-btn rf-btn-danger" onClick={() => run(() => balanceStop())}><Square size={11} fill="currentColor" />Stop</button>
            : <button className="rf-btn rf-btn-primary" disabled={!standing || busy} onClick={() => run(() => balanceTrain(kind, steps), 'Training started.')}><Play size={11} fill="currentColor" />Train</button>}
        </div>
        {(training || st?.history?.length > 0) && (
          <div style={{ marginTop: 8 }}>
            <div style={{ height: 6, borderRadius: 999, background: 'var(--border-subtle, #2b2f31)', overflow: 'hidden' }}>
              <div style={{ width: `${Math.round((st?.progress || 0) * 100)}%`, height: '100%', background: 'var(--accent, #e8622c)', transition: 'width .4s' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5, color: 'var(--text-secondary, #a9aca7)', fontFamily: 'var(--font-mono, monospace)', fontSize: 11 }}>
              <span>{(st?.timesteps || 0).toLocaleString()} steps</span>
              <span>reward {st?.mean_reward ?? '–'}</span>
            </div>
            <Sparkline points={st?.history || []} />
            <div style={{ color: 'var(--text-muted, #8a8d89)', fontSize: 11 }}>{st?.message}</div>
          </div>
        )}
      </div>

      {st?.eval && (
        <div style={{ ...card, borderColor: '#3fb27f66' }}>
          <span style={label}>Result · survival under a {Math.round(st.eval.push_newtons)} N push ({st.eval.trials} directions)</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, fontFamily: 'var(--font-mono, monospace)' }}>
            <span style={{ fontSize: 18, color: 'var(--text-secondary, #a9aca7)' }}>{Math.round(st.eval.untrained * 100)}%</span>
            <span>→</span>
            <span style={{ fontSize: 22, fontWeight: 800, color: '#3fb27f' }}>{Math.round(st.eval.trained * 100)}%</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted, #8a8d89)' }}>untrained → trained</span>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
        <button className="rf-btn" style={btn()} disabled={!standing || busy} onClick={() => run(() => balanceReset(), 'Robot reset to standing.')}><RotateCcw size={12} />Reset stance</button>
        <button className="rf-btn" style={btn()} disabled={!trained || busy} onClick={() => run(() => balanceUntrain(), 'Policy removed: the robot is untrained again.')}><Zap size={12} />Remove policy</button>
      </div>
    </div>
  );
};
