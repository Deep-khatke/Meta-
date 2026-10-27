import React, { useState } from 'react';
import {
  Sliders, Bot, Box, Trash2, Gauge, Move, Rotate3d, Compass,
  Target, Plus, Layers, Sparkles, BrainCircuit, Activity,
  Radio, Cpu, GitFork, ShieldCheck, Video
} from 'lucide-react';
import { Project, SceneObject, SimulationState, SensorSettings } from '../types';

interface PropertiesPanelProps {
  project: Project;
  simState: SimulationState;
  selectedObjectId: string | null;
  onUpdateObject: (updated: SceneObject) => void;
  onDeleteObject: (id: string) => void;
  onUpdateRobotLimits: (maxLin: number, maxAng: number) => void;
  onUpdateGravity: (gravity: [number, number, number]) => void;
  targetPosition?: [number, number, number] | null;
  onUpdateTarget?: (target: [number, number, number]) => void;
  onSelectObject?: (id: string | null) => void;
  onAddObstacle?: (type: 'box' | 'sphere' | 'cylinder') => void;
  onTrainRobot?: () => void;
  onUpdateSensors?: (sensors: SensorSettings) => void;
  onUpdateBoundaries?: (enable: boolean, bounds: [number, number, number, number, number, number]) => void;
  onJointControl?: (jointName: string, target: number) => void;
  onClearGenericRobot?: () => void;
  onOpenRobotStudio?: () => void;
}

export const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  project,
  simState,
  selectedObjectId,
  onUpdateObject,
  onDeleteObject,
  onUpdateRobotLimits,
  onUpdateGravity,
  targetPosition,
  onUpdateTarget,
  onSelectObject,
  onAddObstacle,
  onTrainRobot,
  onUpdateSensors,
  onUpdateBoundaries,
  onJointControl,
  onClearGenericRobot,
  onOpenRobotStudio,
}) => {
  const selectedObstacle = project.objects.find((o) => o.id === selectedObjectId);
  const isRobotSelected = selectedObjectId === 'robot';
  const isTargetSelected = selectedObjectId === 'target';
  const isGenericRobotSelected = selectedObjectId === 'generic_robot';

  // State for interactive joint control sliders
  const [jointSliderVals, setJointSliderVals] = useState<Record<string, number>>({});

  // 0. Target Waypoint Inspector View
  if (isTargetSelected) {
    const tx = targetPosition ? targetPosition[0] : 2.5;
    const ty = targetPosition ? targetPosition[1] : 0.5;
    const tz = targetPosition ? targetPosition[2] : 0.25;
    const rPos = simState.robot.position;
    const distToTarget = Math.hypot(tx - rPos[0], ty - rPos[1]);

    return (
      <aside className="rf-right-sidebar">
        <div className="rf-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Target size={14} color="var(--accent-green)" />
            <span>Target Waypoint Inspector</span>
          </div>
        </div>

        <div className="rf-panel-body">
          <div className="rf-prop-group">
            <div className="rf-prop-title">Coordinates (meters)</div>
            <div className="rf-coord-input-row">
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>X (m)</span>
                <input
                  type="number"
                  step="0.2"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={tx}
                  onChange={(e) => onUpdateTarget?.([parseFloat(e.target.value) || 0, ty, tz])}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Y (m)</span>
                <input
                  type="number"
                  step="0.2"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={ty}
                  onChange={(e) => onUpdateTarget?.([tx, parseFloat(e.target.value) || 0, tz])}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Z (m)</span>
                <input
                  type="number"
                  step="0.1"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={tz}
                  onChange={(e) => onUpdateTarget?.([tx, ty, parseFloat(e.target.value) || 0.25])}
                />
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', padding: '8px 10px', borderRadius: 4, marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Distance to Robot:</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>
                {distToTarget.toFixed(2)}m
              </span>
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Quick Target Presets</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              <button
                className="rf-btn"
                onClick={() => onUpdateTarget?.([2.0, 0.0, 0.25])}
                style={{ fontSize: 11, padding: '5px 8px', justifyContent: 'center' }}
              >
                Ahead (+2.0m)
              </button>
              <button
                className="rf-btn"
                onClick={() => onUpdateTarget?.([3.2, 0.0, 0.25])}
                style={{ fontSize: 11, padding: '5px 8px', justifyContent: 'center' }}
              >
                Far (+3.2m)
              </button>
              <button
                className="rf-btn"
                onClick={() => onUpdateTarget?.([2.5, 0.8, 0.25])}
                style={{ fontSize: 11, padding: '5px 8px', justifyContent: 'center' }}
              >
                Slalom (+2.5, +0.8)
              </button>
              <button
                className="rf-btn"
                onClick={() => onUpdateTarget?.([2.0, -1.2, 0.25])}
                style={{ fontSize: 11, padding: '5px 8px', justifyContent: 'center' }}
              >
                Left (+2.0, -1.2)
              </button>
            </div>
          </div>
        </div>
      </aside>
    );
  }

  // 1. Generic Kinematic Robot Inspector View
  if (isGenericRobotSelected && project.generic_robot) {
    const gr = project.generic_robot;
    const jPositions = simState.joint_positions || {};
    const jVelocities = simState.joint_velocities || {};
    const jTorques = simState.joint_torques || {};

    return (
      <aside className="rf-right-sidebar">
        <div className="rf-panel-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <GitFork size={14} color="var(--accent-green)" />
            <span>Kinematic Robot & Joints</span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {onOpenRobotStudio && (
              <button
                className="rf-btn rf-btn-primary"
                style={{ fontSize: 10, padding: '2px 8px', gap: 4 }}
                onClick={onOpenRobotStudio}
                title="Open comprehensive Robot Modeling & Control Studio"
              >
                <GitFork size={11} />
                <span>Studio</span>
              </button>
            )}
            <button
              className="rf-btn rf-btn-danger"
              style={{ fontSize: 10, padding: '2px 8px' }}
              onClick={() => onClearGenericRobot?.()}
              title="Remove imported robot from scene"
            >
              <Trash2 size={11} />
              <span>Remove</span>
            </button>
          </div>
        </div>

        <div className="rf-panel-body">
          {onOpenRobotStudio && (
            <button
              className="rf-btn"
              onClick={onOpenRobotStudio}
              style={{
                width: '100%',
                marginBottom: 10,
                padding: '6px 10px',
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.25) 100%)',
                border: '1px solid rgba(59, 130, 246, 0.4)',
                color: '#60a5fa',
                fontWeight: 600,
                fontSize: 11,
                gap: 6,
                justifyContent: 'center',
              }}
            >
              <GitFork size={13} color="#60a5fa" />
              <span>Launch Robot Modeling & Control Studio</span>
            </button>
          )}
          <div className="rf-prop-group">
            <div className="rf-prop-title">Robot Architecture</div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Model:</span>
              <span style={{ fontWeight: 600 }}>{gr.name}</span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Links / Joints:</span>
              <span>{gr.links.length} Links / {gr.joints.length} Joints</span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Base Type:</span>
              <span style={{ color: gr.is_fixed_base ? 'var(--accent-blue)' : 'var(--accent-orange)' }}>
                {gr.is_fixed_base ? 'Fixed Base (Arm/Manipulator)' : 'Mobile Kinematic Base'}
              </span>
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Actuated Joint Controls</div>
            {gr.joints.map((joint) => {
              const mode = joint.actuator_mode || 'position';
              const curPos = jPositions[joint.name] ?? 0.0;
              const curVel = jVelocities[joint.name] ?? 0.0;
              const curTorque = jTorques[joint.name] ?? 0.0;

              let sliderMin = -3.14;
              let sliderMax = 3.14;
              let step = 0.02;
              let unitLabel = joint.type === 'prismatic' ? 'm' : 'rad';

              if (mode === 'position') {
                sliderMin = joint.limit?.lower ?? (joint.type === 'prismatic' ? -0.5 : -3.14);
                sliderMax = joint.limit?.upper ?? (joint.type === 'prismatic' ? 0.5 : 3.14);
                step = joint.type === 'prismatic' ? 0.01 : 0.02;
                unitLabel = joint.type === 'prismatic' ? 'm' : 'rad';
              } else if (mode === 'velocity') {
                const vMax = joint.limit?.velocity && joint.limit.velocity > 0 ? joint.limit.velocity : 5.0;
                sliderMin = -vMax;
                sliderMax = vMax;
                step = 0.05;
                unitLabel = joint.type === 'prismatic' ? 'm/s' : 'rad/s';
              } else if (mode === 'torque') {
                const effMax = joint.limit?.effort && joint.limit.effort > 0 ? joint.limit.effort : 50.0;
                sliderMin = -effMax;
                sliderMax = effMax;
                step = 0.5;
                unitLabel = joint.type === 'prismatic' ? 'N' : 'Nm';
              }

              const targetVal = jointSliderVals[joint.name] ?? (mode === 'position' ? curPos : 0.0);

              return (
                <div key={joint.name} style={{ background: 'var(--bg-panel)', padding: 8, borderRadius: 5, marginBottom: 8, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontWeight: 600, fontSize: 11, color: 'var(--text-primary)' }}>{joint.name}</span>
                      <span style={{ fontSize: 9, padding: '1px 5px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 3, textTransform: 'capitalize', color: 'var(--text-muted)' }}>
                        {joint.type} ({mode})
                      </span>
                    </div>
                    <span style={{ fontSize: 10, color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)' }}>
                      {mode === 'position'
                        ? (joint.type === 'prismatic' ? `${curPos.toFixed(3)} m` : `${curPos.toFixed(3)} rad (${(curPos * 180 / Math.PI).toFixed(1)}°)`)
                        : (mode === 'velocity' ? `${curVel.toFixed(2)} ${unitLabel}` : `${curTorque.toFixed(1)} ${unitLabel}`)}
                    </span>
                  </div>

                  {joint.type !== 'fixed' && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                      <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{sliderMin.toFixed(1)}</span>
                      <input
                        type="range"
                        min={sliderMin}
                        max={sliderMax}
                        step={step}
                        value={targetVal}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setJointSliderVals((prev) => ({ ...prev, [joint.name]: val }));
                          onJointControl?.(joint.name, val);
                        }}
                        style={{ flex: 1, cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{sliderMax.toFixed(1)}</span>
                    </div>
                  )}

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                    <span>Pos: {curPos.toFixed(2)} {joint.type === 'prismatic' ? 'm' : 'rad'}</span>
                    <span>Vel: {curVel.toFixed(2)} {joint.type === 'prismatic' ? 'm/s' : 'rad/s'}</span>
                    <span>Torque: {curTorque.toFixed(1)} {joint.type === 'prismatic' ? 'N' : 'Nm'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </aside>
    );
  }

  // 2. Robot Properties View
  if (isRobotSelected) {
    const rConfig = project.robot;
    const rState = simState.robot;

    return (
      <aside className="rf-right-sidebar">
        <div className="rf-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Bot size={14} color="var(--accent-blue)" />
            <span>Robot Inspector (Diff-Drive)</span>
          </div>
        </div>

        <div className="rf-panel-body">
          <div className="rf-prop-group">
            <div className="rf-prop-title">Robot Identity & Geometry</div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Model:</span>
              <span style={{ fontWeight: 600 }}>{rConfig.name}</span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Wheel Base:</span>
              <span>{rConfig.wheel_base}m</span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Wheel Radius:</span>
              <span>{rConfig.wheel_radius}m</span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Chassis Mass:</span>
              <span>{rConfig.chassis_mass} kg</span>
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Velocity Limits (MuJoCo Clamped)</div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Max Linear (m/s):</span>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="5.0"
                className="rf-input"
                style={{ width: 75 }}
                value={rConfig.max_linear_velocity}
                onChange={(e) => onUpdateRobotLimits(parseFloat(e.target.value) || 1.5, rConfig.max_angular_velocity)}
              />
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Max Angular (rad/s):</span>
              <input
                type="number"
                step="0.2"
                min="0.2"
                max="10.0"
                className="rf-input"
                style={{ width: 75 }}
                value={rConfig.max_angular_velocity}
                onChange={(e) => onUpdateRobotLimits(rConfig.max_linear_velocity, parseFloat(e.target.value) || 3.0)}
              />
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Live Physics State (MuJoCo)</div>
            <div style={{ background: 'var(--bg-panel)', padding: 8, borderRadius: 4, marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Position (m):</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>X: {rState.position[0].toFixed(3)}</span>
                <span>Y: {rState.position[1].toFixed(3)}</span>
                <span>Z: {rState.position[2].toFixed(3)}</span>
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', padding: 8, borderRadius: 4, marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Speeds:</div>
              <div style={{ fontSize: 11, color: 'var(--accent-blue)', display: 'flex', justifyContent: 'space-between' }}>
                <span>Linear: {rState.linear_speed.toFixed(2)} m/s</span>
                <span>Angular: {rState.angular_speed.toFixed(2)} rad/s</span>
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', padding: 8, borderRadius: 4 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Wheel Speeds (rad/s):</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>Left: {rState.left_wheel_velocity.toFixed(2)}</span>
                <span>Right: {rState.right_wheel_velocity.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    );
  }

  // 3. Obstacle Properties View
  if (selectedObstacle) {
    const liveObj = simState.objects[selectedObstacle.id];
    const friction = selectedObstacle.friction || [1.0, 0.005, 0.0001];

    return (
      <aside className="rf-right-sidebar">
        <div className="rf-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Box size={13} color="var(--accent-orange)" />
            <span>Object Inspector</span>
          </div>
          <button
            className="rf-btn rf-btn-icon rf-btn-danger"
            onClick={() => onDeleteObject(selectedObstacle.id)}
            title="Delete Selected Object"
          >
            <Trash2 size={13} />
          </button>
        </div>

        <div className="rf-panel-body">
          <div className="rf-prop-group">
            <div className="rf-prop-title">Object Identity</div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Name:</span>
              <input
                type="text"
                className="rf-input"
                style={{ width: 140 }}
                value={selectedObstacle.name}
                onChange={(e) => onUpdateObject({ ...selectedObstacle, name: e.target.value })}
              />
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Geometry:</span>
              <span style={{ textTransform: 'uppercase', fontWeight: 600, fontSize: 11 }}>
                {selectedObstacle.type}
              </span>
            </div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Color:</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input
                  type="color"
                  value={selectedObstacle.color}
                  onChange={(e) => onUpdateObject({ ...selectedObstacle, color: e.target.value })}
                  style={{ width: 26, height: 22, border: 'none', background: 'transparent', cursor: 'pointer' }}
                />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{selectedObstacle.color}</span>
              </div>
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Physics Properties</div>
            <div className="rf-prop-row">
              <span className="rf-prop-label">Body Mode:</span>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={selectedObstacle.is_static}
                  onChange={(e) => onUpdateObject({ ...selectedObstacle, is_static: e.target.checked })}
                />
                <span style={{ fontSize: 12, fontWeight: 600, color: selectedObstacle.is_static ? '#94a3b8' : '#fb923c' }}>
                  {selectedObstacle.is_static ? 'Static (Fixed in space)' : 'Dynamic (Free Physics)'}
                </span>
              </label>
            </div>
            {!selectedObstacle.is_static && (
              <div className="rf-prop-row">
                <span className="rf-prop-label">Mass (kg):</span>
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  max="500"
                  className="rf-input"
                  style={{ width: 80 }}
                  value={selectedObstacle.mass}
                  onChange={(e) => {
                    const m = parseFloat(e.target.value);
                    if (!isNaN(m) && m > 0) onUpdateObject({ ...selectedObstacle, mass: m });
                  }}
                />
              </div>
            )}
            <div className="rf-prop-row">
              <span className="rf-prop-label">Sliding Friction:</span>
              <input
                type="number"
                step="0.1"
                min="0.0"
                max="5.0"
                className="rf-input"
                style={{ width: 75 }}
                value={friction[0]}
                onChange={(e) => {
                  const f0 = parseFloat(e.target.value) || 0.0;
                  onUpdateObject({ ...selectedObstacle, friction: [f0, friction[1], friction[2]] });
                }}
              />
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Position (meters)</div>
            <div className="rf-coord-input-row">
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>X</span>
                <input
                  type="number"
                  step="0.1"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.position[0]}
                  onChange={(e) => {
                    const x = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, position: [x, selectedObstacle.position[1], selectedObstacle.position[2]] });
                  }}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Y</span>
                <input
                  type="number"
                  step="0.1"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.position[1]}
                  onChange={(e) => {
                    const y = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, position: [selectedObstacle.position[0], y, selectedObstacle.position[2]] });
                  }}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Z</span>
                <input
                  type="number"
                  step="0.1"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.position[2]}
                  onChange={(e) => {
                    const z = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, position: [selectedObstacle.position[0], selectedObstacle.position[1], z] });
                  }}
                />
              </div>
            </div>
            {liveObj && (
              <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Live: [{liveObj.position.map((v) => v.toFixed(2)).join(', ')}]
              </div>
            )}
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Rotation (Euler Deg)</div>
            <div className="rf-coord-input-row">
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Roll</span>
                <input
                  type="number"
                  step="5"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.rotation[0]}
                  onChange={(e) => {
                    const r = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, rotation: [r, selectedObstacle.rotation[1], selectedObstacle.rotation[2]] });
                  }}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Pitch</span>
                <input
                  type="number"
                  step="5"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.rotation[1]}
                  onChange={(e) => {
                    const p = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, rotation: [selectedObstacle.rotation[0], p, selectedObstacle.rotation[2]] });
                  }}
                />
              </div>
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Yaw</span>
                <input
                  type="number"
                  step="5"
                  className="rf-input"
                  style={{ width: '100%' }}
                  value={selectedObstacle.rotation[2]}
                  onChange={(e) => {
                    const y = parseFloat(e.target.value) || 0;
                    onUpdateObject({ ...selectedObstacle, rotation: [selectedObstacle.rotation[0], selectedObstacle.rotation[1], y] });
                  }}
                />
              </div>
            </div>
          </div>

          <div className="rf-prop-group">
            <div className="rf-prop-title">Dimensions (meters)</div>
            {selectedObstacle.type === 'box' && (
              <div className="rf-coord-input-row">
                <div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half X</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.05"
                    className="rf-input"
                    style={{ width: '100%' }}
                    value={selectedObstacle.size[0] || 0.2}
                    onChange={(e) => {
                      const sx = Math.max(0.05, parseFloat(e.target.value) || 0.2);
                      onUpdateObject({ ...selectedObstacle, size: [sx, selectedObstacle.size[1] || 0.2, selectedObstacle.size[2] || 0.2] });
                    }}
                  />
                </div>
                <div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Y</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.05"
                    className="rf-input"
                    style={{ width: '100%' }}
                    value={selectedObstacle.size[1] || 0.2}
                    onChange={(e) => {
                      const sy = Math.max(0.05, parseFloat(e.target.value) || 0.2);
                      onUpdateObject({ ...selectedObstacle, size: [selectedObstacle.size[0] || 0.2, sy, selectedObstacle.size[2] || 0.2] });
                    }}
                  />
                </div>
                <div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Z</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.05"
                    className="rf-input"
                    style={{ width: '100%' }}
                    value={selectedObstacle.size[2] || 0.2}
                    onChange={(e) => {
                      const sz = Math.max(0.05, parseFloat(e.target.value) || 0.2);
                      onUpdateObject({ ...selectedObstacle, size: [selectedObstacle.size[0] || 0.2, selectedObstacle.size[1] || 0.2, sz] });
                    }}
                  />
                </div>
              </div>
            )}

            {selectedObstacle.type === 'sphere' && (
              <div className="rf-prop-row">
                <span className="rf-prop-label">Radius:</span>
                <input
                  type="number"
                  step="0.05"
                  min="0.05"
                  className="rf-input"
                  style={{ width: 80 }}
                  value={selectedObstacle.size[0] || 0.2}
                  onChange={(e) => {
                    const r = Math.max(0.05, parseFloat(e.target.value) || 0.2);
                    onUpdateObject({ ...selectedObstacle, size: [r] });
                  }}
                />
              </div>
            )}

            {selectedObstacle.type === 'cylinder' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                <div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Radius (m)</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.05"
                    className="rf-input"
                    style={{ width: '100%' }}
                    value={selectedObstacle.size[0] || 0.2}
                    onChange={(e) => {
                      const r = Math.max(0.05, parseFloat(e.target.value) || 0.2);
                      onUpdateObject({ ...selectedObstacle, size: [r, selectedObstacle.size[1] || 0.3] });
                    }}
                  />
                </div>
                <div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Height (m)</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.05"
                    className="rf-input"
                    style={{ width: '100%' }}
                    value={selectedObstacle.size[1] || 0.3}
                    onChange={(e) => {
                      const hh = Math.max(0.05, parseFloat(e.target.value) || 0.3);
                      onUpdateObject({ ...selectedObstacle, size: [selectedObstacle.size[0] || 0.2, hh] });
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>
    );
  }

  // 4. World Physics, Sensors & Environment View (Default)
  const sensors = project.sensors || {
    lidar: { enabled: true, mount_pos: [0.1, 0, 0.09], num_beams: 16, fov_deg: 360, max_range: 8, min_range: 0.05 },
    imu: { enabled: true, mount_pos: [0, 0, 0.05] }
  };
  const b = project.settings.world_boundaries || [-15, 15, -15, 15, 0, 8];
  const boundariesOn = project.settings.enable_boundaries ?? true;

  return (
    <aside className="rf-right-sidebar">
      <div className="rf-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sliders size={13} />
          <span>Simulation & Sensor Studio</span>
        </div>
      </div>

      <div className="rf-panel-body">
        {/* World Physics & Gravity */}
        <div className="rf-prop-group">
          <div className="rf-prop-title">World Gravity & Environment</div>
          <div className="rf-prop-row">
            <span className="rf-prop-label">Gravity Presets:</span>
            <select
              className="rf-select"
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                onUpdateGravity([0, 0, val]);
              }}
              value={project.settings.gravity[2].toString()}
            >
              <option value="-9.81">Earth (-9.81 m/s²)</option>
              <option value="-1.62">Moon (-1.62 m/s²)</option>
              <option value="-3.71">Mars (-3.71 m/s²)</option>
              <option value="0.0">Zero-G (0.0 m/s²)</option>
            </select>
          </div>

          <div className="rf-prop-row">
            <span className="rf-prop-label">Gravity Z (m/s²):</span>
            <input
              type="number"
              step="0.5"
              className="rf-input"
              style={{ width: 80 }}
              value={project.settings.gravity[2]}
              onChange={(e) => {
                const gz = parseFloat(e.target.value);
                if (!isNaN(gz)) onUpdateGravity([0, 0, gz]);
              }}
            />
          </div>
        </div>

        {/* World Boundaries */}
        <div className="rf-prop-group">
          <div className="rf-prop-title">World Containment Boundaries</div>
          <div className="rf-prop-row">
            <span className="rf-prop-label">Containment Walls:</span>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={boundariesOn}
                onChange={(e) => onUpdateBoundaries?.(e.target.checked, b as any)}
              />
              <span style={{ fontSize: 11 }}>{boundariesOn ? 'Enabled (Physical Walls)' : 'Disabled (Open Floor)'}</span>
            </label>
          </div>
          {boundariesOn && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              Boundary Span: [{b[0]}m to {b[1]}m] × [{b[2]}m to {b[3]}m]
            </div>
          )}
        </div>

        {/* Simulated LiDAR Configuration & Live Data */}
        <div className="rf-prop-group">
          <div className="rf-prop-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <Radio size={13} color="var(--accent-blue)" />
              <span>Simulated LiDAR</span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={sensors.lidar.enabled}
                onChange={(e) => {
                  onUpdateSensors?.({
                    ...sensors,
                    lidar: { ...sensors.lidar, enabled: e.target.checked }
                  });
                }}
              />
              <span style={{ fontSize: 10 }}>Active</span>
            </label>
          </div>

          {sensors.lidar.enabled && (
            <>
              <div className="rf-prop-row">
                <span className="rf-prop-label">Beams:</span>
                <input
                  type="number"
                  step="4"
                  min="4"
                  max="64"
                  className="rf-input"
                  style={{ width: 70 }}
                  value={sensors.lidar.num_beams}
                  onChange={(e) => {
                    const nb = parseInt(e.target.value) || 16;
                    onUpdateSensors?.({
                      ...sensors,
                      lidar: { ...sensors.lidar, num_beams: nb }
                    });
                  }}
                />
              </div>
              <div className="rf-prop-row">
                <span className="rf-prop-label">FOV:</span>
                <span>{sensors.lidar.fov_deg}°</span>
              </div>
              <div className="rf-prop-row">
                <span className="rf-prop-label">Max Range:</span>
                <span>{sensors.lidar.max_range}m</span>
              </div>

              {simState.lidar_readings && simState.lidar_readings.length > 0 && (
                <div style={{ background: 'var(--bg-panel)', padding: 6, borderRadius: 4, marginTop: 6 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>
                    Live LiDAR Rays ({simState.lidar_readings.length}):
                  </div>
                  <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>
                    Closest Hit: {Math.min(...simState.lidar_readings).toFixed(2)}m
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Simulated IMU */}
        <div className="rf-prop-group">
          <div className="rf-prop-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <Cpu size={13} color="var(--accent-green)" />
              <span>Simulated IMU (6-DOF)</span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={sensors.imu.enabled}
                onChange={(e) => {
                  onUpdateSensors?.({
                    ...sensors,
                    imu: { ...sensors.imu, enabled: e.target.checked }
                  });
                }}
              />
              <span style={{ fontSize: 10 }}>Active</span>
            </label>
          </div>

          {sensors.imu.enabled && simState.imu_linear_acceleration && (
            <div style={{ background: 'var(--bg-panel)', padding: 6, borderRadius: 4 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>MuJoCo Accelerometer (m/s²):</div>
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                [{simState.imu_linear_acceleration.map((v) => v.toFixed(2)).join(', ')}]
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>MuJoCo Gyro (rad/s):</div>
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
                [{simState.imu_angular_velocity?.map((v) => v.toFixed(2)).join(', ')}]
              </div>
            </div>
          )}
        </div>

        {/* Simulated Vision Camera (RGB) */}
        <div className="rf-prop-group">
          <div className="rf-prop-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <Video size={13} color="#38bdf8" />
              <span>Vision Camera (RGB Sensor)</span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={sensors.camera?.enabled !== false}
                onChange={(e) => {
                  onUpdateSensors?.({
                    ...sensors,
                    camera: {
                      ...(sensors.camera || { mount_pos: [0.18, 0, 0.08], mount_rpy: [0, 0, 0], width: 320, height: 240, fov_deg: 65 }),
                      enabled: e.target.checked
                    }
                  });
                }}
              />
              <span style={{ fontSize: 10 }}>Active</span>
            </label>
          </div>

          {sensors.camera?.enabled !== false && (
            <>
              <div className="rf-prop-row">
                <span className="rf-prop-label">Field of View (FOV):</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="range"
                    min="30"
                    max="120"
                    step="5"
                    value={sensors.camera?.fov_deg || 65}
                    onChange={(e) => {
                      const fov = parseInt(e.target.value) || 65;
                      onUpdateSensors?.({
                        ...sensors,
                        camera: {
                          ...(sensors.camera || { mount_pos: [0.18, 0, 0.08], mount_rpy: [0, 0, 0], width: 320, height: 240 }),
                          enabled: true,
                          fov_deg: fov
                        }
                      });
                    }}
                    style={{ width: 80, cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                    {sensors.camera?.fov_deg || 65}°
                  </span>
                </div>
              </div>

              <div className="rf-prop-row">
                <span className="rf-prop-label">Mount Position:</span>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Front Bumper [0.18, 0, 0.08]m</span>
              </div>
            </>
          )}
        </div>

        {/* Imported Robot Model Management */}
        {project.generic_robot && (
          <div className="rf-prop-group" style={{ borderColor: 'rgba(59, 130, 246, 0.3)' }}>
            <div className="rf-prop-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>Imported Robot Model</span>
              <button
                className="rf-btn"
                style={{ fontSize: 10, padding: '2px 8px', color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.4)' }}
                onClick={() => onClearGenericRobot?.()}
                title="Remove imported URDF kinematic robot"
              >
                Clear Model
              </button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
              {project.generic_robot.name} ({project.generic_robot.joints.length} actuated joints)
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
