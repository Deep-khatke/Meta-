import React, { useState, useEffect, useRef } from 'react';
import {
  Compass,
  Maximize2,
  Minimize2,
  Plane,
  X,
  Gauge,
  Zap,
  Radio,
  Sliders,
  ShieldAlert,
  ArrowUp,
  ArrowDown
} from 'lucide-react';
import { ObjectPhysicsState } from '../types';

interface DroneFpvHUDProps {
  droneState: ObjectPhysicsState | null;
  activeKeys?: { [key: string]: boolean };
  flightMode: 'stabilize' | 'acro' | 'alt_hold';
  onChangeFlightMode: (mode: 'stabilize' | 'acro' | 'alt_hold') => void;
  onTakeoff: () => void;
  onLand: () => void;
  onHover: () => void;
  onManualFlightCommand: (pitch: number, roll: number, yaw: number, climb: number) => void;
  isFullScreen: boolean;
  onToggleFullScreen: () => void;
  onClose?: () => void;
}

export const DroneFpvHUD: React.FC<DroneFpvHUDProps> = ({
  droneState,
  activeKeys = {},
  flightMode,
  onChangeFlightMode,
  onTakeoff,
  onLand,
  onHover,
  onManualFlightCommand,
  isFullScreen,
  onToggleFullScreen,
  onClose,
}) => {
  // Extract real physics telemetry
  const pos = droneState?.position || [0, 0, 0];
  const vel = droneState?.linear_velocity || [0, 0, 0];
  const euler = droneState?.attitude_euler || [0, 0, 0]; // [roll, pitch, yaw] in deg
  const thrusts = droneState?.thrusts || [0, 0, 0, 0];
  const batteryPct = droneState?.battery_pct !== undefined ? droneState.battery_pct : 98;
  const isArmed = droneState?.rotors_active ?? false;
  const rawFlightMode = droneState?.flight_mode || 'grounded';

  const rollDeg = Math.max(-85, Math.min(85, euler[0] || 0));
  const pitchDeg = Math.max(-85, Math.min(85, euler[1] || 0));
  let yawDeg = (euler[2] || 0) % 360;
  if (yawDeg < 0) yawDeg += 360;

  const altitude = Math.max(0, pos[2] || 0);
  const climbRate = vel[2] || 0;
  const horizontalSpeed = Math.hypot(vel[0], vel[1]);
  const groundSpeed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2);

  // Virtual flight stick drag handlers for mouse/touch
  const [leftStick, setLeftStick] = useState<{ x: number; y: number }>({ x: 0, y: 0 }); // Yaw (x), Climb (y)
  const [rightStick, setRightStick] = useState<{ x: number; y: number }>({ x: 0, y: 0 }); // Roll (x), Pitch (y)
  const leftStickDragging = useRef(false);
  const rightStickDragging = useRef(false);

  // Send virtual stick telemetry when sticks are held
  useEffect(() => {
    if (leftStick.x !== 0 || leftStick.y !== 0 || rightStick.x !== 0 || rightStick.y !== 0) {
      const interval = setInterval(() => {
        onManualFlightCommand(rightStick.y, rightStick.x, leftStick.x, leftStick.y);
      }, 40);
      return () => clearInterval(interval);
    }
  }, [leftStick, rightStick, onManualFlightCommand]);

  // Compass heading points
  const headings = [
    { label: 'N', deg: 0 },
    { label: '030', deg: 30 },
    { label: '060', deg: 60 },
    { label: 'E', deg: 90 },
    { label: '120', deg: 120 },
    { label: '150', deg: 150 },
    { label: 'S', deg: 180 },
    { label: '210', deg: 210 },
    { label: '240', deg: 240 },
    { label: 'W', deg: 270 },
    { label: '300', deg: 300 },
    { label: '330', deg: 330 },
  ];

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        zIndex: 10,
        color: '#06b6d4',
        fontFamily: 'var(--font-mono, monospace)',
        userSelect: 'none',
      }}
    >
      {/* 1. TOP AVIONICS TAPE: COMPASS HEADING & FLIGHT MODE BAR */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 18px',
          background: 'linear-gradient(180deg, rgba(10, 15, 22, 0.85) 0%, rgba(10, 15, 22, 0) 100%)',
          pointerEvents: 'auto',
        }}
      >
        {/* Left Status: Flight Mode & Arming */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: isArmed ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${isArmed ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'}`,
              padding: '4px 10px',
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 700,
              color: isArmed ? '#4ade80' : '#f87171',
              boxShadow: isArmed ? '0 0 10px rgba(34, 197, 94, 0.25)' : 'none',
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: isArmed ? '#22c55e' : '#ef4444',
                boxShadow: isArmed ? '0 0 8px #22c55e' : 'none',
              }}
            />
            <span>{isArmed ? `ARMED // ${flightMode.toUpperCase()}` : 'STANDBY // DISARMED'}</span>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              padding: '4px 10px',
              borderRadius: 6,
              fontSize: 11,
              color: '#38bdf8',
            }}
          >
            <Zap size={12} color="#38bdf8" />
            <span>3S {(11.1 + (batteryPct / 100) * 1.5).toFixed(1)}V ({Math.round(batteryPct)}%)</span>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
              color: '#94a3b8',
              background: 'rgba(15, 23, 42, 0.7)',
              padding: '4px 8px',
              borderRadius: 6,
              border: '1px solid rgba(255, 255, 255, 0.08)',
            }}
          >
            <span>STATUS:</span>
            <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{rawFlightMode.toUpperCase()}</span>
          </div>
        </div>

        {/* Center: Rotating Compass Heading Ribbon */}
        <div
          style={{
            position: 'relative',
            width: 280,
            height: 38,
            overflow: 'hidden',
            background: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid rgba(6, 182, 212, 0.4)',
            borderRadius: 8,
            boxShadow: '0 4px 15px rgba(0, 0, 0, 0.4)',
          }}
        >
          {/* Top Center Amber Pointer */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '5px solid transparent',
              borderRight: '5px solid transparent',
              borderTop: '6px solid #f59e0b',
              zIndex: 3,
            }}
          />

          {/* Heading Number Badge */}
          <div
            style={{
              position: 'absolute',
              bottom: 2,
              left: '50%',
              transform: 'translateX(-50%)',
              fontSize: 10,
              fontWeight: 800,
              color: '#f59e0b',
              zIndex: 3,
            }}
          >
            {Math.round(yawDeg).toString().padStart(3, '0')}°
          </div>

          {/* Sliding Tape */}
          <div
            style={{
              display: 'flex',
              position: 'absolute',
              top: 8,
              left: '50%',
              transform: `translateX(-${(yawDeg / 360) * 800}px)`,
              width: 1600,
              transition: 'transform 0.05s linear',
            }}
          >
            {/* Repeated twice for continuous wrap */}
            {[...headings, ...headings, ...headings].map((h, idx) => (
              <div
                key={idx}
                style={{
                  width: 66.6,
                  textAlign: 'center',
                  fontSize: 10,
                  fontWeight: h.label.length === 1 ? 800 : 500,
                  color: h.label.length === 1 ? '#38bdf8' : '#64748b',
                }}
              >
                | {h.label}
              </div>
            ))}
          </div>
        </div>

        {/* Right Controls: Fullscreen & Mode Selectors */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div
            className="rf-segmented"
            style={{
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              padding: 2,
              borderRadius: 6,
            }}
          >
            {(['stabilize', 'alt_hold', 'acro'] as const).map((m) => (
              <button
                key={m}
                onClick={() => onChangeFlightMode(m)}
                style={{
                  padding: '3px 8px',
                  fontSize: 10,
                  fontWeight: flightMode === m ? 700 : 500,
                  color: flightMode === m ? '#fff' : '#94a3b8',
                  background: flightMode === m ? 'rgba(6, 182, 212, 0.4)' : 'transparent',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                }}
              >
                {m.replace('_', '-')}
              </button>
            ))}
          </div>

          <button
            onClick={onToggleFullScreen}
            title={isFullScreen ? 'Exit Fullscreen FPV' : 'Enter Fullscreen FPV (F)'}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 30,
              height: 30,
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              borderRadius: 6,
              color: '#38bdf8',
              cursor: 'pointer',
            }}
          >
            {isFullScreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>

          {onClose && (
            <button
              onClick={onClose}
              title="Close FPV Mode"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 30,
                height: 30,
                background: 'rgba(239, 68, 68, 0.2)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                borderRadius: 6,
                color: '#f87171',
                cursor: 'pointer',
              }}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* 2. CENTER RETICLE, ARTIFICIAL HORIZON & PITCH LADDER */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 320,
          height: 320,
          pointerEvents: 'none',
        }}
      >
        {/* Fixed Boresight Crosshair */}
        <svg
          viewBox="-160 -160 320 320"
          style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}
        >
          {/* Center Dot */}
          <circle cx="0" cy="0" r="2.5" fill="#f59e0b" />
          {/* Inverted T reticle */}
          <line x1="-24" y1="0" x2="-8" y2="0" stroke="#f59e0b" strokeWidth="1.8" />
          <line x1="8" y1="0" x2="24" y2="0" stroke="#f59e0b" strokeWidth="1.8" />
          <line x1="0" y1="8" x2="0" y2="18" stroke="#f59e0b" strokeWidth="1.8" />
        </svg>

        {/* Rotating & Tilting Pitch Ladder & Artificial Horizon */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            transform: `rotate(${-rollDeg}deg) translateY(${pitchDeg * 3.5}px)`,
            transformOrigin: 'center center',
            transition: 'transform 0.04s linear',
          }}
        >
          <svg viewBox="-160 -160 320 320" style={{ width: '100%', height: '100%' }}>
            {/* Horizon Line */}
            <line
              x1="-130"
              y1="0"
              x2="130"
              y2="0"
              stroke="#06b6d4"
              strokeWidth="2"
              strokeDasharray="18 8"
              opacity="0.85"
            />
            {/* Horizon Center Gap */}
            <circle cx="0" cy="0" r="16" fill="none" stroke="#06b6d4" strokeWidth="1.2" opacity="0.5" />

            {/* Pitch Rungs at +30, +20, +10, -10, -20, -30 */}
            {[
              { deg: 30, text: '+30' },
              { deg: 20, text: '+20' },
              { deg: 10, text: '+10' },
              { deg: -10, text: '-10' },
              { deg: -20, text: '-20' },
              { deg: -30, text: '-30' },
            ].map(({ deg, text }) => {
              const y = -deg * 3.5;
              const isPositive = deg > 0;
              return (
                <g key={deg} opacity="0.75">
                  {/* Left Rung */}
                  <line
                    x1="-50"
                    y1={y}
                    x2="-18"
                    y2={y}
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                    strokeDasharray={isPositive ? 'none' : '4 3'}
                  />
                  <line
                    x1="-50"
                    y1={y}
                    x2="-50"
                    y2={y + (isPositive ? 6 : -6)}
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                  />
                  {/* Right Rung */}
                  <line
                    x1="18"
                    y1={y}
                    x2="50"
                    y2={y}
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                    strokeDasharray={isPositive ? 'none' : '4 3'}
                  />
                  <line
                    x1="50"
                    y1={y}
                    x2="50"
                    y2={y + (isPositive ? 6 : -6)}
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                  />
                  {/* Degrees Label */}
                  <text
                    x="-55"
                    y={y + 3}
                    fill="#38bdf8"
                    fontSize="9"
                    textAnchor="end"
                    fontFamily="monospace"
                  >
                    {text}
                  </text>
                  <text
                    x="55"
                    y={y + 3}
                    fill="#38bdf8"
                    fontSize="9"
                    textAnchor="start"
                    fontFamily="monospace"
                  >
                    {text}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* 3. FLIGHT SCALES: LEFT AIRSPEED TAPE & RIGHT ALTITUDE TAPE */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          transform: 'translateY(-50%)',
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          padding: '0 24px',
          pointerEvents: 'none',
        }}
      >
        {/* Left Tape: Airspeed & Groundspeed */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid rgba(6, 182, 212, 0.35)',
            borderRadius: 8,
            padding: '8px 12px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
            minWidth: 90,
          }}
        >
          <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.5px' }}>SPEED</div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 800,
              color: '#38bdf8',
              lineHeight: 1.2,
              margin: '3px 0',
            }}
          >
            {groundSpeed.toFixed(1)}
          </div>
          <div style={{ fontSize: 9, color: '#64748b' }}>M/S</div>
          <div
            style={{
              width: '100%',
              height: 1,
              background: 'rgba(255, 255, 255, 0.08)',
              margin: '6px 0',
            }}
          />
          <div style={{ fontSize: 9, color: '#94a3b8' }}>
            HORIZ: <span style={{ color: '#f1f5f9' }}>{horizontalSpeed.toFixed(1)}</span>
          </div>
        </div>

        {/* Right Tape: Altitude & Vario */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid rgba(6, 182, 212, 0.35)',
            borderRadius: 8,
            padding: '8px 12px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
            minWidth: 90,
          }}
        >
          <div style={{ fontSize: 9, color: '#94a3b8', letterSpacing: '0.5px' }}>ALTITUDE</div>
          <div
            style={{
              fontSize: 20,
              fontWeight: 800,
              color: '#38bdf8',
              lineHeight: 1.2,
              margin: '3px 0',
            }}
          >
            {altitude.toFixed(2)}
          </div>
          <div style={{ fontSize: 9, color: '#64748b' }}>METERS</div>
          <div
            style={{
              width: '100%',
              height: 1,
              background: 'rgba(255, 255, 255, 0.08)',
              margin: '6px 0',
            }}
          />
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              fontSize: 9,
              color: climbRate > 0.05 ? '#4ade80' : climbRate < -0.05 ? '#f87171' : '#94a3b8',
            }}
          >
            {climbRate > 0.05 ? <ArrowUp size={10} /> : climbRate < -0.05 ? <ArrowDown size={10} /> : null}
            <span>VARIO: {climbRate > 0 ? `+${climbRate.toFixed(1)}` : climbRate.toFixed(1)} m/s</span>
          </div>
        </div>
      </div>

      {/* 4. BOTTOM DECK: 4-ROTOR TELEMETRY & FLIGHT CONTROLS */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          padding: '16px 20px',
          background: 'linear-gradient(0deg, rgba(10, 15, 22, 0.92) 0%, rgba(10, 15, 22, 0) 100%)',
          pointerEvents: 'auto',
        }}
      >
        {/* Left: Real-time 4-Rotor Aerodynamic Thrust Telemetry Bars */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.88)',
            border: '1px solid rgba(6, 182, 212, 0.3)',
            borderRadius: 10,
            padding: '10px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            minWidth: 230,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: 10,
              fontWeight: 700,
              color: '#38bdf8',
            }}
          >
            <span>MUJOCO PROPULSION // 4-ROTOR THRUST</span>
            <span style={{ fontSize: 9, color: '#94a3b8' }}>TOTAL: {thrusts.reduce((a, b) => a + b, 0).toFixed(1)}N</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {[
              { id: 'M1 (FL)', val: thrusts[0] || 0, dir: 'CCW' },
              { id: 'M2 (FR)', val: thrusts[1] || 0, dir: 'CW' },
              { id: 'M3 (RL)', val: thrusts[2] || 0, dir: 'CW' },
              { id: 'M4 (RR)', val: thrusts[3] || 0, dir: 'CCW' },
            ].map((m, idx) => {
              const maxThrust = 7.5; // Newtons
              const pct = Math.min(100, Math.max(0, (m.val / maxThrust) * 100));
              return (
                <div key={idx} style={{ fontSize: 9.5 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', marginBottom: 2 }}>
                    <span>{m.id} [{m.dir}]</span>
                    <span style={{ fontWeight: 700, color: '#38bdf8' }}>{m.val.toFixed(2)}N</span>
                  </div>
                  <div style={{ height: 5, background: 'rgba(255, 255, 255, 0.1)', borderRadius: 3, overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${pct}%`,
                        background: pct > 85 ? '#ef4444' : pct > 60 ? '#f59e0b' : '#06b6d4',
                        transition: 'width 0.05s ease',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Center: Live Keyboard Teleop Helper Pill & Quick Action Triggers */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 8,
          }}
        >
          {/* Quick Action Buttons: Takeoff, Hover, Land */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={onTakeoff}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '6px 14px',
                background: 'linear-gradient(90deg, #0284c7, #0369a1)',
                border: '1px solid rgba(56, 189, 248, 0.6)',
                borderRadius: 6,
                color: '#fff',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 0 12px rgba(2, 132, 199, 0.4)',
              }}
            >
              <ArrowUp size={12} />
              <span>TAKEOFF (T)</span>
            </button>

            <button
              onClick={onHover}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '6px 14px',
                background: 'rgba(15, 23, 42, 0.85)',
                border: '1px solid rgba(6, 182, 212, 0.4)',
                borderRadius: 6,
                color: '#38bdf8',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              <span>HOVER (H)</span>
            </button>

            <button
              onClick={onLand}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '6px 14px',
                background: 'rgba(239, 68, 68, 0.2)',
                border: '1px solid rgba(239, 68, 68, 0.5)',
                borderRadius: 6,
                color: '#f87171',
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              <ArrowDown size={12} />
              <span>LAND (L)</span>
            </button>
          </div>

          {/* Keyboard Controls Guide */}
          <div
            style={{
              display: 'flex',
              gap: 12,
              padding: '5px 14px',
              background: 'rgba(15, 23, 42, 0.75)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: 20,
              fontSize: 10,
              color: '#94a3b8',
            }}
          >
            <span>
              <b style={{ color: activeKeys['w'] || activeKeys['s'] ? '#38bdf8' : '#e2e8f0' }}>W/S</b>: Pitch
            </span>
            <span>
              <b style={{ color: activeKeys['a'] || activeKeys['d'] ? '#38bdf8' : '#e2e8f0' }}>A/D</b>: Roll
            </span>
            <span>
              <b style={{ color: activeKeys['q'] || activeKeys['e'] || activeKeys['arrowleft'] || activeKeys['arrowright'] ? '#38bdf8' : '#e2e8f0' }}>
                Q/E
              </b>: Yaw
            </span>
            <span>
              <b style={{ color: activeKeys[' '] || activeKeys['shift'] || activeKeys['c'] ? '#38bdf8' : '#e2e8f0' }}>
                Space/Shift
              </b>: Altitude
            </span>
            <span>
              <b style={{ color: '#e2e8f0' }}>F</b>: FPV
            </span>
          </div>
        </div>

        {/* Right: Attitude Roll / Pitch / Yaw Readout Card */}
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.88)',
            border: '1px solid rgba(6, 182, 212, 0.3)',
            borderRadius: 10,
            padding: '10px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            minWidth: 170,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
            fontSize: 11,
          }}
        >
          <div style={{ fontSize: 10, fontWeight: 700, color: '#38bdf8', marginBottom: 2 }}>
            ATTITUDE ESTIMATE
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
            <span>ROLL:</span>
            <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{rollDeg.toFixed(1)}°</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
            <span>PITCH:</span>
            <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{pitchDeg.toFixed(1)}°</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
            <span>YAW:</span>
            <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{yawDeg.toFixed(1)}°</span>
          </div>
        </div>
      </div>
    </div>
  );
};
