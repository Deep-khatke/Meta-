import React, { useEffect, useState, useCallback } from 'react';
import { Radio, Cpu, Camera, Zap, Plus, Trash2, RefreshCw } from 'lucide-react';
import { SimulationState } from '../types';
import { listSensors, addSensor, updateSensor, removeSensor } from '../services/api';

interface SensorsPanelProps {
  simState: SimulationState;
  onShowToast?: (msg: string) => void;
}

const TYPE_META: Record<string, { icon: React.ReactNode; label: string }> = {
  imu: { icon: <Cpu size={13} color="#4ade80" />, label: 'IMU (6-DOF)' },
  lidar: { icon: <Radio size={13} color="#38bdf8" />, label: 'LiDAR' },
  rangefinder: { icon: <Radio size={13} color="#38bdf8" />, label: 'Rangefinder' },
  camera: { icon: <Camera size={13} color="#f59e0b" />, label: 'Camera' },
  contact: { icon: <Zap size={13} color="#f472b6" />, label: 'Contact' },
};

export const SensorsPanel: React.FC<SensorsPanelProps> = ({ simState, onShowToast }) => {
  const [sensors, setSensors] = useState<any[]>([]);
  const [bodies, setBodies] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [newType, setNewType] = useState('imu');
  const [newParent, setNewParent] = useState('robot');

  const refresh = useCallback(async () => {
    try {
      const res = await listSensors();
      setSensors(res.sensors || []);
      setBodies(res.attachable_bodies || []);
      if (res.attachable_bodies && res.attachable_bodies.length > 0 && !res.attachable_bodies.includes(newParent)) {
        setNewParent(res.attachable_bodies[0]);
      }
    } catch { /* quiet */ }
  }, [newParent]);

  useEffect(() => { refresh(); }, [refresh]);

  const readings = (simState as any).sensor_readings || {};

  const handleAdd = async () => {
    setLoading(true);
    try {
      await addSensor({ type: newType, parent_body: newParent });
      onShowToast?.(`Attached ${newType} to ${newParent}`);
      await refresh();
    } catch (e: any) {
      onShowToast?.(e.message || 'Failed to add sensor');
    } finally { setLoading(false); }
  };

  const handleRemove = async (id: string) => {
    try { await removeSensor(id); onShowToast?.('Sensor removed'); await refresh(); }
    catch (e: any) { onShowToast?.(e.message); }
  };

  const handleToggle = async (s: any) => {
    try { await updateSensor(s.id, { enabled: !s.enabled }); await refresh(); }
    catch (e: any) { onShowToast?.(e.message); }
  };

  const handleParam = async (id: string, patch: Record<string, any>) => {
    try { await updateSensor(id, patch); await refresh(); }
    catch (e: any) { onShowToast?.(e.message); }
  };

  const renderReading = (s: any) => {
    const r = readings[s.id];
    if (!r) return <span style={{ color: 'var(--text-muted)' }}>no data</span>;
    if (r.attached === false) return <span style={{ color: '#fb923c' }}>detached ({r.parent})</span>;
    if (r.type === 'imu') {
      const a = r.linear_acceleration || [0, 0, 0];
      const g = r.angular_velocity || [0, 0, 0];
      return <span>accel [{a.map((v: number) => v.toFixed(2)).join(', ')}] · gyro [{g.map((v: number) => v.toFixed(2)).join(', ')}]</span>;
    }
    if (r.type === 'lidar' || r.type === 'rangefinder') {
      return <span>closest {r.closest != null ? `${r.closest}m` : '—'} · {r.num_beams} beams</span>;
    }
    if (r.type === 'contact') {
      return <span style={{ color: r.in_contact ? '#f472b6' : 'var(--text-muted)' }}>{r.in_contact ? `IN CONTACT (${r.contacts})` : 'clear'}</span>;
    }
    if (r.type === 'camera') return <span>{(r.resolution || []).join('×')} @ {r.fov_deg}°</span>;
    return <span>{r.type}</span>;
  };

  return (
    <div className="rf-prop-group">
      <div className="rf-prop-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span>Attached Sensors ({sensors.length})</span>
        <button className="rf-btn" style={{ fontSize: 10, padding: '2px 6px' }} onClick={refresh} title="Refresh">
          <RefreshCw size={11} />
        </button>
      </div>

      {/* Add sensor row */}
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 8 }}>
        <select className="rf-select" value={newType} onChange={(e) => setNewType(e.target.value)} style={{ fontSize: 10, flex: 1 }}>
          <option value="imu">IMU</option>
          <option value="lidar">LiDAR</option>
          <option value="camera">Camera</option>
          <option value="contact">Contact</option>
        </select>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>on</span>
        <select className="rf-select" value={newParent} onChange={(e) => setNewParent(e.target.value)} style={{ fontSize: 10, flex: 1 }}>
          {bodies.length === 0 && <option value="robot">robot</option>}
          {bodies.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
        <button className="rf-btn rf-btn-primary" style={{ fontSize: 10, padding: '3px 10px', gap: 3, borderRadius: 100 }} onClick={handleAdd} disabled={loading}>
          <Plus size={11} /><span>Add</span>
        </button>
      </div>

      {sensors.length === 0 && (
        <div style={{ padding: '14px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 11,
          border: '1px dashed rgba(255, 255, 255, 0.1)', borderRadius: 10, background: 'rgba(255, 255, 255, 0.02)', lineHeight: 1.5 }}>
          No sensors attached. Pick a type and a body above and click <strong>Add</strong>,
          or ask the assistant to <em>"attach an IMU to the drone"</em>.
        </div>
      )}

      {sensors.map((s) => (
        <div key={s.id} style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-subtle)', borderRadius: 5, padding: 8, marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {TYPE_META[s.type]?.icon}
              <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)' }}>{s.name}</span>
              <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>→ {s.parent_body}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 3, cursor: 'pointer' }}>
                <input type="checkbox" checked={s.enabled} onChange={() => handleToggle(s)} />
                <span style={{ fontSize: 9 }}>On</span>
              </label>
              <button className="rf-btn rf-btn-danger" style={{ padding: 2 }} onClick={() => handleRemove(s.id)} title="Remove">
                <Trash2 size={11} />
              </button>
            </div>
          </div>

          {/* Live reading */}
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)', marginTop: 5 }}>
            {renderReading(s)}
          </div>

          {/* Type-specific editable params */}
          {(s.type === 'lidar' || s.type === 'rangefinder') && (
            <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
              <label style={{ fontSize: 9, color: 'var(--text-muted)', display: 'flex', flexDirection: 'column' }}>
                Beams
                <input type="number" min={4} max={128} step={4} className="rf-input" style={{ width: 60 }}
                  value={s.num_beams} onChange={(e) => handleParam(s.id, { num_beams: parseInt(e.target.value) || 16 })} />
              </label>
              <label style={{ fontSize: 9, color: 'var(--text-muted)', display: 'flex', flexDirection: 'column' }}>
                Range (m)
                <input type="number" min={0.5} max={50} step={0.5} className="rf-input" style={{ width: 60 }}
                  value={s.max_range} onChange={(e) => handleParam(s.id, { max_range: parseFloat(e.target.value) || 8 })} />
              </label>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};
