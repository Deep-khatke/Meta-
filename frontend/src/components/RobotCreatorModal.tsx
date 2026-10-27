import React, { useState } from 'react';
import {
  RobotConfig, GenericRobotModel, LinkModel, JointModel, JointLimit, LinkGeom,
  ValidationResult
} from '../types';
import {
  validateRobotModel, createCustomAsset
} from '../services/api';
import {
  Bot, Cpu, CheckCircle2, AlertTriangle, Plus, Trash2, ShieldCheck,
  X, Layers, Play, Save
} from 'lucide-react';

interface RobotCreatorModalProps {
  onClose: () => void;
  onAddToScene: (robotType: 'mobile' | 'articulated', config: any) => void;
  onShowToast: (msg: string) => void;
}

export const RobotCreatorModal: React.FC<RobotCreatorModalProps> = ({
  onClose,
  onAddToScene,
  onShowToast,
}) => {
  const [robotType, setRobotType] = useState<'mobile' | 'articulated'>('mobile');

  // Mobile Robot State
  const [mobileConfig, setMobileConfig] = useState<RobotConfig>({
    id: `custom_bot_${Date.now().toString().slice(-4)}`,
    name: 'Custom Mobile Scout',
    initial_position: [0.0, 0.0, 0.07],
    initial_yaw: 0.0,
    chassis_size: [0.20, 0.15, 0.05],
    wheel_radius: 0.06,
    wheel_width: 0.025,
    wheel_base: 0.28,
    caster_radius: 0.02,
    chassis_mass: 3.5,
    wheel_mass: 0.3,
    max_linear_velocity: 1.5,
    max_angular_velocity: 3.0,
  });

  // Articulated Robot State
  const [articulatedConfig, setArticulatedConfig] = useState<GenericRobotModel>({
    name: 'Custom Articulated Arm',
    root_link: 'base_link',
    base_position: [2.5, 2.5, 0.05],
    base_rpy: [0.0, 0.0, 0.0],
    is_fixed_base: true,
    links: [
      {
        name: 'base_link',
        visual: { type: 'cylinder', size: [0.18, 0.05], origin_pos: [0, 0, 0.05], origin_rpy: [0, 0, 0], color: '#334155' },
        collision: { type: 'cylinder', size: [0.18, 0.05], origin_pos: [0, 0, 0.05], origin_rpy: [0, 0, 0], color: '#334155' },
        inertial: { mass: 4.0, origin_pos: [0, 0, 0.05], ixx: 0.05, iyy: 0.05, izz: 0.05 }
      },
      {
        name: 'arm_link_1',
        visual: { type: 'box', size: [0.22, 0.04, 0.04], origin_pos: [0.22, 0, 0], origin_rpy: [0, 0, 0], color: '#2563eb' },
        collision: { type: 'box', size: [0.22, 0.04, 0.04], origin_pos: [0.22, 0, 0], origin_rpy: [0, 0, 0], color: '#2563eb' },
        inertial: { mass: 1.5, origin_pos: [0.22, 0, 0], ixx: 0.02, iyy: 0.02, izz: 0.02 }
      }
    ],
    joints: [
      {
        name: 'shoulder_joint',
        type: 'revolute',
        parent_link: 'base_link',
        child_link: 'arm_link_1',
        origin_pos: [0.0, 0.0, 0.12],
        origin_rpy: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        limit: { lower: -3.14, upper: 3.14, effort: 60.0, velocity: 5.0 },
        damping: 0.2,
        armature: 0.01,
        actuator_mode: 'position',
        kp: 80.0,
        kv: 8.0
      }
    ]
  });

  // Validation State
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      const config = robotType === 'mobile' ? mobileConfig : articulatedConfig;
      const res = await validateRobotModel(config, robotType);
      setValidationResult(res);
      if (res.valid) {
        onShowToast('Physical parameters verified via MuJoCo compilation!');
      } else {
        onShowToast(`Validation failed: ${res.errors[0]}`);
      }
    } catch (err: any) {
      setValidationResult({
        valid: false,
        errors: [err.message || 'Validation request failed'],
        warnings: []
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleAddLink = () => {
    const idx = articulatedConfig.links.length + 1;
    const newLinkName = `link_${idx}`;
    const newJointName = `joint_${idx}`;
    const parentName = articulatedConfig.links[articulatedConfig.links.length - 1]?.name || 'base_link';

    const newLink: LinkModel = {
      name: newLinkName,
      visual: { type: 'box', size: [0.18, 0.035, 0.03], origin_pos: [0.18, 0, 0], origin_rpy: [0, 0, 0], color: '#f97316' },
      collision: { type: 'box', size: [0.18, 0.035, 0.03], origin_pos: [0.18, 0, 0], origin_rpy: [0, 0, 0], color: '#f97316' },
      inertial: { mass: 1.0, origin_pos: [0.18, 0, 0], ixx: 0.01, iyy: 0.01, izz: 0.01 }
    };

    const newJoint: JointModel = {
      name: newJointName,
      type: 'revolute',
      parent_link: parentName,
      child_link: newLinkName,
      origin_pos: [0.36, 0.0, 0.0],
      origin_rpy: [0.0, 0.0, 0.0],
      axis: [0.0, 0.0, 1.0],
      limit: { lower: -2.8, upper: 2.8, effort: 40.0, velocity: 5.0 },
      damping: 0.15,
      armature: 0.01,
      actuator_mode: 'position',
      kp: 60.0,
      kv: 6.0
    };

    setArticulatedConfig({
      ...articulatedConfig,
      links: [...articulatedConfig.links, newLink],
      joints: [...articulatedConfig.joints, newJoint]
    });
    setValidationResult(null);
  };

  const handleRemoveLink = (linkIndex: number) => {
    if (articulatedConfig.links.length <= 1) {
      onShowToast('Robot must have at least one link.');
      return;
    }
    const linkToRemove = articulatedConfig.links[linkIndex].name;
    const newLinks = articulatedConfig.links.filter((_, i) => i !== linkIndex);
    const newJoints = articulatedConfig.joints.filter((j) => j.child_link !== linkToRemove && j.parent_link !== linkToRemove);

    setArticulatedConfig({
      ...articulatedConfig,
      links: newLinks,
      joints: newJoints
    });
    setValidationResult(null);
  };

  const handleApplyToScene = async () => {
    // Validate first
    const config = robotType === 'mobile' ? mobileConfig : articulatedConfig;
    const res = await validateRobotModel(config, robotType);
    if (!res.valid) {
      setValidationResult(res);
      onShowToast(`Cannot add invalid robot: ${res.errors[0]}`);
      return;
    }

    onAddToScene(robotType, config);
    onShowToast(`Added '${config.name}' to active simulation scene.`);
    onClose();
  };

  const handleSaveToLibrary = async () => {
    const config = robotType === 'mobile' ? mobileConfig : articulatedConfig;
    const res = await validateRobotModel(config, robotType);
    if (!res.valid) {
      setValidationResult(res);
      onShowToast(`Cannot save invalid robot: ${res.errors[0]}`);
      return;
    }

    try {
      await createCustomAsset({
        name: config.name,
        category: 'robot',
        description: `Custom ${robotType} robot designed in Robot Creator.`,
        tags: [robotType, 'custom', 'robot-creator'],
        data: { type: robotType, config },
        thumbnail_hint: robotType === 'articulated' ? 'arm' : 'bot'
      });
      onShowToast(`Saved '${config.name}' to Asset Library.`);
      onClose();
    } catch (err: any) {
      onShowToast(err.message || 'Failed to save to library');
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.25)',
      backdropFilter: 'blur(2px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 8,
        width: 680,
        maxWidth: '92vw',
        maxHeight: '88vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 48px rgba(0,0,0,0.7)'
      }}>
        {/* Header */}
        <div style={{
          padding: '12px 18px',
          borderBottom: '1px solid var(--border-subtle)',
          backgroundColor: 'var(--bg-panel)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ padding: 6, borderRadius: 6, backgroundColor: 'rgba(59, 130, 246, 0.2)' }}>
              <Bot size={16} color="var(--accent-blue)" />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                ROBOT CREATOR & KINEMATICS DESIGNER
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                Authoritative physical geometry, joint kinematics & MuJoCo validation
              </div>
            </div>
          </div>

          <button className="rf-btn rf-btn-icon" onClick={onClose}>
            <X size={14} />
          </button>
        </div>

        {/* Robot Type Switcher */}
        <div style={{
          padding: '8px 18px',
          borderBottom: '1px solid var(--border-subtle)',
          backgroundColor: 'var(--bg-primary)',
          display: 'flex',
          gap: 10
        }}>
          <button
            className={`rf-btn ${robotType === 'mobile' ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 11, padding: '4px 14px', gap: 6 }}
            onClick={() => {
              setRobotType('mobile');
              setValidationResult(null);
            }}
          >
            <Bot size={13} />
            <span>Mobile Differential-Drive</span>
          </button>

          <button
            className={`rf-btn ${robotType === 'articulated' ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 11, padding: '4px 14px', gap: 6 }}
            onClick={() => {
              setRobotType('articulated');
              setValidationResult(null);
            }}
          >
            <Cpu size={13} />
            <span>Articulated Robotic Manipulator</span>
          </button>
        </div>

        {/* Modal Body Form */}
        <div style={{ padding: 18, overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Mobile Robot Form */}
          {robotType === 'mobile' ? (
            <>
              <div className="rf-prop-group">
                <div className="rf-prop-title">Robot Identity</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Robot Model Name</span>
                    <input
                      type="text"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.name}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, name: e.target.value })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Chassis Mass (kg)</span>
                    <input
                      type="number"
                      step="0.5"
                      min="0.5"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.chassis_mass}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, chassis_mass: parseFloat(e.target.value) || 1.0 })}
                    />
                  </div>
                </div>
              </div>

              <div className="rf-prop-group">
                <div className="rf-prop-title">Chassis Box Half-Dimensions (meters)</div>
                <div className="rf-coord-input-row">
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Length (X)</span>
                    <input
                      type="number"
                      step="0.02"
                      min="0.04"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.chassis_size[0]}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, chassis_size: [parseFloat(e.target.value) || 0.1, mobileConfig.chassis_size[1], mobileConfig.chassis_size[2]] })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Width (Y)</span>
                    <input
                      type="number"
                      step="0.02"
                      min="0.04"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.chassis_size[1]}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, chassis_size: [mobileConfig.chassis_size[0], parseFloat(e.target.value) || 0.1, mobileConfig.chassis_size[2]] })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half Height (Z)</span>
                    <input
                      type="number"
                      step="0.01"
                      min="0.02"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.chassis_size[2]}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, chassis_size: [mobileConfig.chassis_size[0], mobileConfig.chassis_size[1], parseFloat(e.target.value) || 0.05] })}
                    />
                  </div>
                </div>
              </div>

              <div className="rf-prop-group">
                <div className="rf-prop-title">Drive Wheel Kinematics</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Wheel Base (m)</span>
                    <input
                      type="number"
                      step="0.02"
                      min="0.1"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.wheel_base}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, wheel_base: parseFloat(e.target.value) || 0.2 })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Wheel Radius (m)</span>
                    <input
                      type="number"
                      step="0.01"
                      min="0.02"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.wheel_radius}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, wheel_radius: parseFloat(e.target.value) || 0.05 })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Wheel Mass (kg)</span>
                    <input
                      type="number"
                      step="0.05"
                      min="0.05"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.wheel_mass}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, wheel_mass: parseFloat(e.target.value) || 0.2 })}
                    />
                  </div>
                </div>
              </div>

              <div className="rf-prop-group">
                <div className="rf-prop-title">Velocity Limits (MuJoCo Actuator Constraints)</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max Linear Speed (m/s)</span>
                    <input
                      type="number"
                      step="0.2"
                      min="0.2"
                      max="8.0"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.max_linear_velocity}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, max_linear_velocity: parseFloat(e.target.value) || 1.5 })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max Angular Speed (rad/s)</span>
                    <input
                      type="number"
                      step="0.2"
                      min="0.5"
                      max="10.0"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={mobileConfig.max_angular_velocity}
                      onChange={(e) => setMobileConfig({ ...mobileConfig, max_angular_velocity: parseFloat(e.target.value) || 3.0 })}
                    />
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Articulated Robot Form */}
              <div className="rf-prop-group">
                <div className="rf-prop-title">Robot Identity & Base</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Model Name</span>
                    <input
                      type="text"
                      className="rf-input"
                      style={{ width: '100%' }}
                      value={articulatedConfig.name}
                      onChange={(e) => setArticulatedConfig({ ...articulatedConfig, name: e.target.value })}
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Base Type</span>
                    <select
                      className="rf-select"
                      style={{ width: '100%' }}
                      value={articulatedConfig.is_fixed_base ? 'fixed' : 'mobile'}
                      onChange={(e) => setArticulatedConfig({ ...articulatedConfig, is_fixed_base: e.target.value === 'fixed' })}
                    >
                      <option value="fixed">Fixed Base (Workbench Arm)</option>
                      <option value="mobile">Floating Free-Base</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Kinematic Links Manager */}
              <div className="rf-prop-group">
                <div className="rf-prop-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>Kinematic Links ({articulatedConfig.links.length})</span>
                  <button
                    className="rf-btn"
                    style={{ fontSize: 10, padding: '2px 8px', gap: 4 }}
                    onClick={handleAddLink}
                  >
                    <Plus size={11} />
                    <span>Add Link</span>
                  </button>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {articulatedConfig.links.map((link, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--bg-panel)',
                        padding: 10,
                        borderRadius: 6,
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 6
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 600, fontSize: 11, color: 'var(--text-primary)' }}>
                          Link #{idx + 1}: <code style={{ color: 'var(--accent-blue)' }}>{link.name}</code>
                        </span>
                        {idx > 0 && (
                          <button
                            className="rf-btn rf-btn-icon rf-btn-danger"
                            style={{ width: 20, height: 20, padding: 2 }}
                            onClick={() => handleRemoveLink(idx)}
                            title="Remove link"
                          >
                            <Trash2 size={11} />
                          </button>
                        )}
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Geometry</span>
                          <select
                            className="rf-select"
                            style={{ width: '100%', fontSize: 10 }}
                            value={link.visual?.type || 'box'}
                            onChange={(e) => {
                              const t = e.target.value as any;
                              const updatedLinks = [...articulatedConfig.links];
                              if (updatedLinks[idx].visual) {
                                updatedLinks[idx].visual!.type = t;
                              }
                              setArticulatedConfig({ ...articulatedConfig, links: updatedLinks });
                            }}
                          >
                            <option value="box">Box</option>
                            <option value="cylinder">Cylinder</option>
                            <option value="sphere">Sphere</option>
                          </select>
                        </div>

                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Mass (kg)</span>
                          <input
                            type="number"
                            step="0.2"
                            min="0.1"
                            className="rf-input"
                            style={{ width: '100%', fontSize: 10 }}
                            value={link.inertial?.mass || 1.0}
                            onChange={(e) => {
                              const m = parseFloat(e.target.value) || 1.0;
                              const updatedLinks = [...articulatedConfig.links];
                              if (updatedLinks[idx].inertial) {
                                updatedLinks[idx].inertial!.mass = m;
                              }
                              setArticulatedConfig({ ...articulatedConfig, links: updatedLinks });
                            }}
                          />
                        </div>

                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Color</span>
                          <input
                            type="color"
                            className="rf-input"
                            style={{ width: '100%', height: 26, padding: 2 }}
                            value={link.visual?.color || '#3b82f6'}
                            onChange={(e) => {
                              const updatedLinks = [...articulatedConfig.links];
                              if (updatedLinks[idx].visual) {
                                updatedLinks[idx].visual!.color = e.target.value;
                              }
                              setArticulatedConfig({ ...articulatedConfig, links: updatedLinks });
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actuated Joints Manager */}
              <div className="rf-prop-group">
                <div className="rf-prop-title">
                  Actuated Joints ({articulatedConfig.joints.length})
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {articulatedConfig.joints.map((joint, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--bg-panel)',
                        padding: 10,
                        borderRadius: 6,
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 6
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600, fontSize: 11 }}>
                          Joint #{idx + 1}: <code style={{ color: 'var(--accent-green)' }}>{joint.name}</code>
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                          {joint.parent_link} ➔ {joint.child_link}
                        </span>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Type</span>
                          <select
                            className="rf-select"
                            style={{ width: '100%', fontSize: 10 }}
                            value={joint.type}
                            onChange={(e) => {
                              const updatedJoints = [...articulatedConfig.joints];
                              updatedJoints[idx].type = e.target.value as any;
                              setArticulatedConfig({ ...articulatedConfig, joints: updatedJoints });
                            }}
                          >
                            <option value="revolute">Revolute (Hinge)</option>
                            <option value="prismatic">Prismatic (Slide)</option>
                            <option value="continuous">Continuous</option>
                            <option value="fixed">Fixed</option>
                          </select>
                        </div>

                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Lower Limit</span>
                          <input
                            type="number"
                            step="0.2"
                            className="rf-input"
                            style={{ width: '100%', fontSize: 10 }}
                            value={joint.limit?.lower ?? -3.14}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              const updatedJoints = [...articulatedConfig.joints];
                              if (updatedJoints[idx].limit) {
                                updatedJoints[idx].limit!.lower = val;
                              }
                              setArticulatedConfig({ ...articulatedConfig, joints: updatedJoints });
                            }}
                          />
                        </div>

                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>Upper Limit</span>
                          <input
                            type="number"
                            step="0.2"
                            className="rf-input"
                            style={{ width: '100%', fontSize: 10 }}
                            value={joint.limit?.upper ?? 3.14}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              const updatedJoints = [...articulatedConfig.joints];
                              if (updatedJoints[idx].limit) {
                                updatedJoints[idx].limit!.upper = val;
                              }
                              setArticulatedConfig({ ...articulatedConfig, joints: updatedJoints });
                            }}
                          />
                        </div>

                        <div>
                          <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>KP Stiffness</span>
                          <input
                            type="number"
                            step="10"
                            className="rf-input"
                            style={{ width: '100%', fontSize: 10 }}
                            value={joint.kp}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 50;
                              const updatedJoints = [...articulatedConfig.joints];
                              updatedJoints[idx].kp = val;
                              setArticulatedConfig({ ...articulatedConfig, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Validation Status Banner */}
          {validationResult && (
            <div style={{
              padding: 10,
              borderRadius: 6,
              backgroundColor: validationResult.valid ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              border: validationResult.valid ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              gap: 4
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 12,
                fontWeight: 600,
                color: validationResult.valid ? 'var(--accent-green)' : 'var(--accent-red)'
              }}>
                {validationResult.valid ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                <span>{validationResult.valid ? 'MuJoCo Physics Model Verified' : 'Physics Model Validation Errors'}</span>
              </div>

              {!validationResult.valid && (
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11, color: 'var(--text-secondary)' }}>
                  {validationResult.errors.map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              )}

              {validationResult.warnings.length > 0 && (
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11, color: 'var(--accent-yellow)' }}>
                  {validationResult.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer Actions */}
        <div style={{
          padding: '12px 18px',
          borderTop: '1px solid var(--border-subtle)',
          backgroundColor: 'var(--bg-panel)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '5px 12px', gap: 5 }}
            onClick={handleValidate}
            disabled={isValidating}
          >
            <ShieldCheck size={13} color="var(--accent-blue)" />
            <span>{isValidating ? 'Validating...' : 'Validate MuJoCo Model'}</span>
          </button>

          <div style={{ display: 'flex', gap: 8 }}>
            <button className="rf-btn" onClick={onClose}>
              Cancel
            </button>

            <button
              className="rf-btn"
              style={{ fontSize: 11, padding: '5px 12px', gap: 5 }}
              onClick={handleSaveToLibrary}
            >
              <Save size={13} color="var(--accent-green)" />
              <span>Save to Asset Library</span>
            </button>

            <button
              className="rf-btn rf-btn-primary"
              style={{ fontSize: 11, padding: '5px 14px', gap: 6, fontWeight: 600 }}
              onClick={handleApplyToScene}
            >
              <Play size={13} />
              <span>Instantiate in Scene</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
