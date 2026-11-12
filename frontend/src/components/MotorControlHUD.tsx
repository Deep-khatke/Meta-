import React, { useState, useEffect, useRef } from 'react';
import {
  Sliders, ChevronDown, ChevronUp, RotateCcw, Play, Pause,
  Layers, Zap, Compass, Move, Shield, Activity, HelpCircle
} from 'lucide-react';
import { GenericRobotModel, JointModel } from '../types';
import { controlRobotJoint } from '../services/api';

interface MotorControlHUDProps {
  genericRobot?: GenericRobotModel | null;
  onShowToast: (msg: string) => void;
  position?: 'bottom-left' | 'bottom-right' | 'top-right';
}

export const MotorControlHUD: React.FC<MotorControlHUDProps> = ({
  genericRobot,
  onShowToast,
  position = 'bottom-left',
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [jointValues, setJointValues] = useState<Record<string, number>>({});
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const demoIntervalRef = useRef<any>(null);

  // Initialize joint angles
  useEffect(() => {
    if (genericRobot?.joints) {
      const initVals: Record<string, number> = {};
      genericRobot.joints.forEach((j) => {
        initVals[j.name] = 0.0;
      });
      setJointValues(initVals);
    }
  }, [genericRobot]);

  // Clean up demo timer on unmount
  useEffect(() => {
    return () => {
      if (demoIntervalRef.current) clearInterval(demoIntervalRef.current);
    };
  }, []);

  if (!genericRobot || !genericRobot.joints || genericRobot.joints.length === 0) {
    return null;
  }

  const controllableJoints = genericRobot.joints.filter((j) => j.type !== 'fixed');

  const handleSliderChange = async (jointName: string, valRad: number) => {
    setJointValues((prev) => ({ ...prev, [jointName]: valRad }));
    try {
      await controlRobotJoint(jointName, valRad);
    } catch (e: any) {
      console.warn(`Error setting joint ${jointName}:`, e);
    }
  };

  const handleStepJoint = (joint: JointModel, deltaRad: number) => {
    const cur = jointValues[joint.name] || 0.0;
    const lower = joint.limit?.lower ?? -3.14159;
    const upper = joint.limit?.upper ?? 3.14159;
    const next = Math.max(lower, Math.min(upper, cur + deltaRad));
    handleSliderChange(joint.name, next);
  };

  const handleZeroJoint = (jointName: string) => {
    handleSliderChange(jointName, 0.0);
  };

  const handleZeroAll = async () => {
    const resetVals: Record<string, number> = {};
    for (const j of controllableJoints) {
      resetVals[j.name] = 0.0;
      await controlRobotJoint(j.name, 0.0);
    }
    setJointValues(resetVals);
    onShowToast('All motor actuators returned to zero home position.');
  };

  const handleParkPose = async () => {
    // Standard tucked industrial park pose
    const parkAngles: Record<string, number> = {};
    controllableJoints.forEach((j, idx) => {
      if (idx === 0) parkAngles[j.name] = 0.0; // yaw
      else if (idx === 1) parkAngles[j.name] = -0.5; // shoulder
      else if (idx === 2) parkAngles[j.name] = 1.0; // elbow
      else if (idx === 3) parkAngles[j.name] = 0.0;
      else if (idx === 4) parkAngles[j.name] = -0.5;
      else parkAngles[j.name] = 0.0;
    });

    for (const [jname, val] of Object.entries(parkAngles)) {
      await controlRobotJoint(jname, val);
    }
    setJointValues((prev) => ({ ...prev, ...parkAngles }));
    onShowToast('Actuators driven to standard Park Pose.');
  };

  const toggleDemoMotion = () => {
    if (isDemoRunning) {
      if (demoIntervalRef.current) clearInterval(demoIntervalRef.current);
      setIsDemoRunning(false);
      onShowToast('Motion demo stopped.');
    } else {
      setIsDemoRunning(true);
      onShowToast('Running kinematic sweep demonstration...');
      let t = 0.0;
      demoIntervalRef.current = setInterval(async () => {
        t += 0.06;
        const newVals: Record<string, number> = {};
        for (let i = 0; i < controllableJoints.length; i++) {
          const j = controllableJoints[i];
          const span = (j.limit?.upper ?? 2.0) - (j.limit?.lower ?? -2.0);
          const amp = Math.min(span * 0.35, 0.8);
          const angle = Math.sin(t * 1.5 + i * 0.7) * amp;
          newVals[j.name] = angle;
          controlRobotJoint(j.name, angle).catch(() => {});
        }
        setJointValues((prev) => ({ ...prev, ...newVals }));
      }, 50);
    }
  };

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 14,
        left: 14,
        zIndex: 10,
        backgroundColor: 'rgba(15, 18, 26, 0.94)',
        border: '1px solid #232b3e',
        borderRadius: 10,
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(232, 98, 44, 0.15)',
        width: 320,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        color: '#f4f3ef',
        fontFamily: 'var(--font-sans)',
      }}
    >
      {/* HUD Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: 'var(--bg-panel)',
          borderBottom: isCollapsed ? 'none' : '1px solid #232b3e',
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div
            style={{
              width: 20,
              height: 20,
              borderRadius: 5,
              background: 'var(--bg-panel-hover)',
              border: '1px solid var(--border-strong)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Sliders size={12} color="var(--accent)" />
          </div>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.4px', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>MOTOR &amp; JOINT HUD</span>
              <span style={{ fontSize: 9, background: '#c4472d', color: '#fff', padding: '1px 5px', borderRadius: 10, fontWeight: 600 }}>
                {controllableJoints.length} ACTUATORS
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            className="rf-btn rf-btn-icon"
            style={{ width: 20, height: 20, padding: 2, background: 'transparent', border: 'none', color: '#a9aca7' }}
            onClick={(e) => {
              e.stopPropagation();
              setIsCollapsed(!isCollapsed);
            }}
          >
            {isCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expanded Controls List */}
      {!isCollapsed && (
        <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 340, overflowY: 'auto' }}>
          {/* Quick Preset Actions Bar */}
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              className="rf-btn"
              onClick={handleZeroAll}
              style={{ fontSize: 10, padding: '3px 8px', flex: 1, justifyContent: 'center' }}
              title="Drive all joints to 0.0 rad"
            >
              <RotateCcw size={10} />
              <span>Zero All</span>
            </button>
            <button
              className="rf-btn"
              onClick={handleParkPose}
              style={{ fontSize: 10, padding: '3px 8px', flex: 1, justifyContent: 'center' }}
              title="Drive joints into safe compact tucked pose"
            >
              <Shield size={10} />
              <span>Park Pose</span>
            </button>
            <button
              className={`rf-btn ${isDemoRunning ? 'rf-btn-active' : ''}`}
              onClick={toggleDemoMotion}
              style={{
                fontSize: 10,
                padding: '3px 8px',
                flex: 1,
                justifyContent: 'center',
                backgroundColor: isDemoRunning ? '#a9aca7' : undefined,
                borderColor: isDemoRunning ? '#a9aca7' : undefined,
                color: isDemoRunning ? '#fff' : undefined,
              }}
              title="Continuous animated sine wave motion across all actuators"
            >
              {isDemoRunning ? <Pause size={10} /> : <Play size={10} />}
              <span>{isDemoRunning ? 'Stop Demo' : 'Demo Motion'}</span>
            </button>
          </div>

          <div style={{ height: 1, backgroundColor: '#25292b', margin: '2px 0' }} />

          {/* Joint Sliders */}
          {controllableJoints.map((j, idx) => {
            const curVal = jointValues[j.name] || 0.0;
            const deg = (curVal * 180.0 / Math.PI).toFixed(1);
            const lower = j.limit?.lower ?? -3.14159;
            const upper = j.limit?.upper ?? 3.14159;
            const isPrismatic = j.type === 'prismatic';
            const displayVal = isPrismatic ? `${(curVal * 1000).toFixed(0)} mm` : `${deg}°`;

            return (
              <div
                key={j.name}
                style={{
                  background: 'rgba(26, 31, 46, 0.6)',
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: '1px solid #25292b',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 3,
                }}
              >
                {/* Joint Header */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ color: '#e8622c', fontWeight: 600, fontSize: 10 }}>J{idx + 1}</span>
                    <span style={{ fontWeight: 600, color: '#ecebe6', fontSize: 11 }}>{j.name}</span>
                    <span
                      style={{
                        fontSize: 8,
                        color: '#a9aca7',
                        textTransform: 'uppercase',
                        padding: '1px 4px',
                        background: '#151923',
                        borderRadius: 3,
                      }}
                    >
                      {j.type}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 11,
                        color: '#5cc596',
                        fontWeight: 600,
                        minWidth: 48,
                        textAlign: 'right',
                      }}
                    >
                      {displayVal}
                    </span>
                    <button
                      className="rf-btn"
                      onClick={() => handleZeroJoint(j.name)}
                      style={{ fontSize: 9, padding: '1px 4px', minWidth: 18, height: 16, lineHeight: 1 }}
                      title="Set Joint to 0"
                    >
                      0
                    </button>
                  </div>
                </div>

                {/* Slider + Step Buttons */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <button
                    className="rf-btn rf-btn-icon"
                    onClick={() => handleStepJoint(j, -0.05)}
                    style={{ width: 18, height: 18, padding: 0, fontSize: 11, borderRadius: 3 }}
                    title="Fine Step Decrement (-0.05 rad)"
                  >
                    -
                  </button>

                  <input
                    type="range"
                    min={lower}
                    max={upper}
                    step={0.01}
                    value={curVal}
                    onChange={(e) => handleSliderChange(j.name, parseFloat(e.target.value))}
                    style={{
                      flex: 1,
                      accentColor: '#e8622c',
                      cursor: 'pointer',
                      height: 4,
                    }}
                  />

                  <button
                    className="rf-btn rf-btn-icon"
                    onClick={() => handleStepJoint(j, 0.05)}
                    style={{ width: 18, height: 18, padding: 0, fontSize: 11, borderRadius: 3 }}
                    title="Fine Step Increment (+0.05 rad)"
                  >
                    +
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

