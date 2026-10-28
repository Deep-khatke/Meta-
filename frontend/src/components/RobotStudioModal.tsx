import React, { useState, useEffect } from 'react';
import {
  GenericRobotModel, LinkModel, JointModel, JointLimit, LinkGeom,
  RobotValidationResult, RobotDiagnostic, FKResult, IKResult, SimulationState, Project, AssetModel
} from '../types';
import {
  validateGenericRobot, autoCalculateInertials, computeFK, solveIK,
  fetchRobotPresets, exportRobotURDF, setGenericRobotModel, controlRobotJoint,
  controlMultipleJoints, saveRobotModelAsset, fetchAssets
} from '../services/api';
import {
  Cpu, GitFork, CheckCircle2, AlertTriangle, AlertCircle, Plus, Trash2,
  ShieldCheck, X, Layers, Play, Save, Download, Copy, RefreshCw,
  Sliders, Target, Compass, Sparkles, Box, Wrench, Check, Upload, FileText,
  RotateCcw, Info, Maximize2, Minimize2, Magnet
} from 'lucide-react';

interface RobotStudioModalProps {
  project: Project;
  simState: SimulationState;
  onClose: () => void;
  onApplyToScene: (robot: GenericRobotModel) => void;
  onShowToast: (msg: string) => void;
  initialRobot?: GenericRobotModel | null;
  targetPosition?: [number, number, number] | null;
  isDocked?: boolean;
  dockWidth?: number;
  onSetDockWidth?: (w: number) => void;
}

export const RobotStudioModal: React.FC<RobotStudioModalProps> = ({
  project,
  simState,
  onClose,
  onApplyToScene,
  onShowToast,
  initialRobot,
  targetPosition,
  isDocked = false,
  dockWidth = 540,
  onSetDockWidth,
}) => {
  // Active Studio Tab: 'builder' | 'control' | 'kinematics' | 'validation' | 'export'
  const [activeTab, setActiveTab] = useState<'builder' | 'control' | 'kinematics' | 'validation' | 'export'>('builder');

  // Working Robot Model
  const [robot, setRobot] = useState<GenericRobotModel>(() => {
    if (initialRobot && initialRobot.links && initialRobot.links.length > 0) {
      return JSON.parse(JSON.stringify(initialRobot));
    }
    if (project.generic_robot && project.generic_robot.links && project.generic_robot.links.length > 0) {
      return JSON.parse(JSON.stringify(project.generic_robot));
    }
    // Default 2-Link Arm
    return {
      name: 'Custom Robotic Arm',
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
          visual: { type: 'box', size: [0.20, 0.04, 0.04], origin_pos: [0.20, 0, 0], origin_rpy: [0, 0, 0], color: '#2563eb' },
          collision: { type: 'box', size: [0.20, 0.04, 0.04], origin_pos: [0.20, 0, 0], origin_rpy: [0, 0, 0], color: '#2563eb' },
          inertial: { mass: 1.5, origin_pos: [0.20, 0, 0], ixx: 0.02, iyy: 0.02, izz: 0.02 }
        },
        {
          name: 'arm_link_2',
          visual: { type: 'box', size: [0.15, 0.03, 0.03], origin_pos: [0.15, 0, 0], origin_rpy: [0, 0, 0], color: '#f97316' },
          collision: { type: 'box', size: [0.15, 0.03, 0.03], origin_pos: [0.15, 0, 0], origin_rpy: [0, 0, 0], color: '#f97316' },
          inertial: { mass: 0.8, origin_pos: [0.15, 0, 0], ixx: 0.01, iyy: 0.01, izz: 0.01 }
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
          limit: { lower: -3.14, upper: 3.14, effort: 50.0, velocity: 5.0 },
          damping: 0.2,
          armature: 0.01,
          actuator_mode: 'position',
          kp: 80.0,
          kv: 8.0
        },
        {
          name: 'elbow_joint',
          type: 'revolute',
          parent_link: 'arm_link_1',
          child_link: 'arm_link_2',
          origin_pos: [0.40, 0.0, 0.0],
          origin_rpy: [0.0, 0.0, 0.0],
          axis: [0.0, 0.0, 1.0],
          limit: { lower: -2.6, upper: 2.6, effort: 30.0, velocity: 5.0 },
          damping: 0.15,
          armature: 0.01,
          actuator_mode: 'position',
          kp: 60.0,
          kv: 6.0
        }
      ]
    };
  });

  // Selected Item in Builder: { type: 'link' | 'joint', name: string }
  const [selectedItem, setSelectedItem] = useState<{ type: 'link' | 'joint'; name: string } | null>({
    type: 'link',
    name: robot.links[0]?.name || 'base_link'
  });

  // Validation State
  const [validationResult, setValidationResult] = useState<RobotValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);

  // Kinematics State
  const [fkResult, setFkResult] = useState<FKResult | null>(null);
  const [ikTargetPos, setIkTargetPos] = useState<[number, number, number]>([
    robot.base_position[0] + 0.35,
    robot.base_position[1] + 0.20,
    robot.base_position[2] + 0.12
  ]);
  const [ikTolerance, setIkTolerance] = useState<number>(0.005);
  const [ikMaxIter, setIkMaxIter] = useState<number>(100);
  const [ikResult, setIkResult] = useState<IKResult | null>(null);
  const [isSolvingIK, setIsSolvingIK] = useState<boolean>(false);

  // Joint Controls State
  const [jointSliderVals, setJointSliderVals] = useState<Record<string, number>>({});

  // Presets, Asset Library & URDF Export State
  const [presets, setPresets] = useState<GenericRobotModel[]>([]);
  const [customRobotAssets, setCustomRobotAssets] = useState<AssetModel[]>([]);
  const [exportedURDF, setExportedURDF] = useState<string>('');
  const [copiedURDF, setCopiedURDF] = useState<boolean>(false);
  const [assetSaveName, setAssetSaveName] = useState<string>(robot.name);
  const [assetSaveDesc, setAssetSaveDesc] = useState<string>(`Articulated robot with ${robot.links.length} links and ${robot.joints.length} joints`);
  const [assetSaveTags, setAssetSaveTags] = useState<string>('articulated, manipulator, robot-studio');
  const [isSavingAsset, setIsSavingAsset] = useState<boolean>(false);

  // JSON Import/Export State
  const [showJsonModal, setShowJsonModal] = useState<boolean>(false);
  const [jsonText, setJsonText] = useState<string>('');

  // Sync initialRobot prop changes
  useEffect(() => {
    if (initialRobot && initialRobot.links && initialRobot.links.length > 0) {
      const cloned = JSON.parse(JSON.stringify(initialRobot));
      setRobot(cloned);
      setSelectedItem({ type: 'link', name: cloned.links[0]?.name || 'base_link' });
      runValidation(cloned);
      runFK(cloned);
    }
  }, [initialRobot]);

  // Sync targetPosition prop changes
  useEffect(() => {
    if (targetPosition) {
      setIkTargetPos([targetPosition[0], targetPosition[1], targetPosition[2]]);
    }
  }, [targetPosition]);

  // Run initial validation and FK when robot changes
  useEffect(() => {
    runValidation(robot);
    runFK(robot);
  }, []);

  // Real-time FK update whenever joint positions stream in or sliders move
  useEffect(() => {
    runFK(robot);
  }, [simState.joint_positions, jointSliderVals, robot]);

  // Load Presets and Saved Robot Assets
  useEffect(() => {
    fetchRobotPresets().then((res) => {
      if (res.presets) setPresets(res.presets);
    }).catch(() => {});

    loadCustomRobotAssets();
  }, []);

  const loadCustomRobotAssets = async () => {
    try {
      const res = await fetchAssets('robot');
      const filtered = (res.assets || []).filter((a: AssetModel) => a.data?.type === 'articulated');
      setCustomRobotAssets(filtered);
    } catch {
      // Ignore background fetch error
    }
  };

  const runValidation = async (rModel: GenericRobotModel) => {
    setIsValidating(true);
    try {
      const res = await validateGenericRobot(rModel);
      setValidationResult(res);
    } catch (err: any) {
      setValidationResult({
        valid: false,
        errors: [{ code: 'VALIDATION_EXCEPTION', severity: 'error', target: 'robot', message: err.message || 'Validation request failed' }],
        warnings: [],
        summary: 'Validation error: ' + (err.message || 'Server error')
      });
    } finally {
      setIsValidating(false);
    }
  };

  const runFK = async (rModel: GenericRobotModel) => {
    try {
      const livePositions = simState.joint_positions || {};
      const positions = { ...livePositions, ...jointSliderVals };
      const res = await computeFK(rModel, positions);
      setFkResult(res);
    } catch (e) {
      // Ignore background FK error
    }
  };

  // Handle IK Solve
  const handleSolveIK = async () => {
    setIsSolvingIK(true);
    try {
      const livePositions = simState.joint_positions || {};
      const currentPos = { ...livePositions, ...jointSliderVals };
      const res = await solveIK({
        robot,
        target_position: ikTargetPos,
        initial_joint_positions: currentPos,
        tolerance: ikTolerance,
        max_iterations: ikMaxIter
      });
      setIkResult(res);
      if (res.success) {
        onShowToast(`IK Solved successfully! Residual error: ${(res.error_distance * 1000).toFixed(1)}mm`);
      } else {
        onShowToast(`IK: ${res.message}`);
      }
    } catch (err: any) {
      setIkResult({
        success: false,
        status: 'ERROR',
        iterations: 0,
        error_distance: 999,
        joint_solution: {},
        achieved_position: [0, 0, 0],
        achieved_rpy_deg: [0, 0, 0],
        message: err.message || 'Failed to solve IK'
      });
      onShowToast(err.message || 'IK Solver error');
    } finally {
      setIsSolvingIK(false);
    }
  };

  const handleApplyIKSolution = async () => {
    if (!ikResult || !ikResult.joint_solution) return;
    try {
      await controlMultipleJoints(ikResult.joint_solution);
      setJointSliderVals((prev) => ({ ...prev, ...ikResult.joint_solution }));
      onShowToast('Applied IK joint solution to active robot!');
    } catch (err: any) {
      onShowToast(`Failed to apply IK solution: ${err.message}`);
    }
  };

  // Auto-Calculate Inertials
  const handleAutoInertials = async () => {
    try {
      const updated = await autoCalculateInertials(robot);
      setRobot(updated);
      await runValidation(updated);
      onShowToast('Physically consistent inertial tensors computed for all links!');
    } catch (err: any) {
      onShowToast(`Failed to auto-calculate inertials: ${err.message}`);
    }
  };

  // Load Preset
  const handleLoadPreset = (preset: GenericRobotModel) => {
    const clone = JSON.parse(JSON.stringify(preset));
    setRobot(clone);
    setSelectedItem({ type: 'link', name: clone.links[0]?.name || 'base_link' });
    setAssetSaveName(clone.name);
    setAssetSaveDesc(`Robot model '${clone.name}' with ${clone.links.length} links and ${clone.joints.length} joints`);
    runValidation(clone);
    runFK(clone);
    onShowToast(`Loaded preset '${preset.name}'`);
  };

  // Load Asset from Asset Library
  const handleLoadAsset = (asset: AssetModel) => {
    try {
      const robotConfig = asset.data?.config;
      if (!robotConfig) throw new Error('Asset does not contain valid robot configuration.');
      const clone = JSON.parse(JSON.stringify(robotConfig));
      setRobot(clone);
      setSelectedItem({ type: 'link', name: clone.links[0]?.name || 'base_link' });
      setAssetSaveName(clone.name || asset.name);
      runValidation(clone);
      runFK(clone);
      onShowToast(`Loaded '${asset.name}' from Asset Library!`);
    } catch (err: any) {
      onShowToast(`Failed to load asset: ${err.message}`);
    }
  };

  // Save Robot to Asset Library
  const handleSaveToAssetLibrary = async () => {
    setIsSavingAsset(true);
    try {
      const tags = assetSaveTags.split(',').map((t) => t.trim()).filter(Boolean);
      await saveRobotModelAsset(assetSaveName, assetSaveDesc, robot, tags);
      onShowToast(`Saved '${assetSaveName}' into RoboForge Asset Library!`);
      await loadCustomRobotAssets();
    } catch (err: any) {
      onShowToast(`Failed to save to Asset Library: ${err.message}`);
    } finally {
      setIsSavingAsset(false);
    }
  };

  // Export JSON
  const handleExportJSON = () => {
    const jsonStr = JSON.stringify(robot, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${robot.name.toLowerCase().replace(/\s+/g, '_')}.robot.json`;
    a.click();
    URL.revokeObjectURL(url);
    onShowToast(`Exported ${robot.name} model JSON!`);
  };

  // Import JSON
  const handleImportJSON = (text: string) => {
    try {
      const parsed = JSON.parse(text);
      if (!parsed.links || !Array.isArray(parsed.links)) {
        throw new Error("Invalid robot JSON: missing 'links' array.");
      }
      setRobot(parsed);
      setSelectedItem({ type: 'link', name: parsed.links[0]?.name || 'base_link' });
      runValidation(parsed);
      runFK(parsed);
      setShowJsonModal(false);
      onShowToast(`Imported robot '${parsed.name || 'Custom'}' successfully!`);
    } catch (e: any) {
      onShowToast(`JSON Import Error: ${e.message}`);
    }
  };

  // Add Link
  const handleAddLink = () => {
    const newIndex = robot.links.length + 1;
    const newName = `link_${newIndex}`;
    const newLink: LinkModel = {
      name: newName,
      visual: { type: 'box', size: [0.15, 0.03, 0.03], origin_pos: [0.15, 0, 0], origin_rpy: [0, 0, 0], color: '#38bdf8' },
      collision: { type: 'box', size: [0.15, 0.03, 0.03], origin_pos: [0.15, 0, 0], origin_rpy: [0, 0, 0], color: '#38bdf8' },
      inertial: { mass: 1.0, origin_pos: [0.15, 0, 0], ixx: 0.01, iyy: 0.01, izz: 0.01 }
    };
    const updated = { ...robot, links: [...robot.links, newLink] };
    setRobot(updated);
    setSelectedItem({ type: 'link', name: newName });
    runValidation(updated);
  };

  // Add Joint
  const handleAddJoint = () => {
    if (robot.links.length < 2) {
      onShowToast('At least two links are required before creating a joint.');
      return;
    }
    const newIndex = robot.joints.length + 1;
    const newName = `joint_${newIndex}`;
    const parentLink = robot.links[robot.links.length - 2].name;
    const childLink = robot.links[robot.links.length - 1].name;

    const newJoint: JointModel = {
      name: newName,
      type: 'revolute',
      parent_link: parentLink,
      child_link: childLink,
      origin_pos: [0.30, 0.0, 0.0],
      origin_rpy: [0.0, 0.0, 0.0],
      axis: [0.0, 0.0, 1.0],
      limit: { lower: -3.14, upper: 3.14, effort: 40.0, velocity: 5.0 },
      damping: 0.15,
      armature: 0.01,
      actuator_mode: 'position',
      kp: 70.0,
      kv: 7.0
    };
    const updated = { ...robot, joints: [...robot.joints, newJoint] };
    setRobot(updated);
    setSelectedItem({ type: 'joint', name: newName });
    runValidation(updated);
  };

  // Delete Link
  const handleDeleteLink = (linkName: string) => {
    if (robot.links.length <= 1) {
      onShowToast('Robot must retain at least one link.');
      return;
    }
    const newLinks = robot.links.filter((l) => l.name !== linkName);
    const newJoints = robot.joints.filter((j) => j.parent_link !== linkName && j.child_link !== linkName);
    let newRoot = robot.root_link;
    if (robot.root_link === linkName) {
      newRoot = newLinks[0]?.name || 'base_link';
    }
    const updated = { ...robot, root_link: newRoot, links: newLinks, joints: newJoints };
    setRobot(updated);
    setSelectedItem(newLinks.length > 0 ? { type: 'link', name: newLinks[0].name } : null);
    runValidation(updated);
  };

  // Delete Joint
  const handleDeleteJoint = (jointName: string) => {
    const newJoints = robot.joints.filter((j) => j.name !== jointName);
    const updated = { ...robot, joints: newJoints };
    setRobot(updated);
    setSelectedItem(robot.links.length > 0 ? { type: 'link', name: robot.links[0].name } : null);
    runValidation(updated);
  };

  // Apply to Active Simulation Scene
  const handleApplyToScene = async () => {
    try {
      const res = await setGenericRobotModel(robot);
      onApplyToScene(res.robot);
      onShowToast(`Robot '${robot.name}' successfully loaded into active simulation!`);
      onClose();
    } catch (err: any) {
      onShowToast(`Failed to apply robot to scene: ${err.message}`);
    }
  };

  // Prepare URDF Export
  const handlePrepareURDF = async () => {
    try {
      const res = await exportRobotURDF(robot);
      setExportedURDF(res.urdf);
      setActiveTab('export');
    } catch (err: any) {
      onShowToast(`Failed to export URDF: ${err.message}`);
    }
  };

  // Quick Action Buttons for Joint Control
  const handleZeroAllJoints = async () => {
    const targets: Record<string, number> = {};
    robot.joints.forEach((j) => {
      targets[j.name] = 0.0;
    });
    setJointSliderVals(targets);
    await controlMultipleJoints(targets);
    onShowToast('All joints commanded to 0.0 (Zero pose)');
  };

  const handleCenterLimits = async () => {
    const targets: Record<string, number> = {};
    robot.joints.forEach((j) => {
      if (j.limit && j.type !== 'continuous') {
        targets[j.name] = (j.limit.lower + j.limit.upper) * 0.5;
      } else {
        targets[j.name] = 0.0;
      }
    });
    setJointSliderVals(targets);
    await controlMultipleJoints(targets);
    onShowToast('All joints commanded to center of physical limits');
  };

  const handleHoldCurrentPosition = async () => {
    const targets: Record<string, number> = {};
    const live = simState.joint_positions || {};
    robot.joints.forEach((j) => {
      targets[j.name] = live[j.name] ?? 0.0;
    });
    setJointSliderVals(targets);
    await controlMultipleJoints(targets);
    onShowToast('Holding currently measured joint positions');
  };

  const handleRelaxTorques = async () => {
    const targets: Record<string, number> = {};
    robot.joints.forEach((j) => {
      if (j.actuator_mode === 'torque') {
        targets[j.name] = 0.0;
      }
    });
    setJointSliderVals((prev) => ({ ...prev, ...targets }));
    if (Object.keys(targets).length > 0) {
      await controlMultipleJoints(targets);
      onShowToast('Torque actuators set to 0.0 Nm (Relaxed)');
    } else {
      onShowToast('No torque actuators configured');
    }
  };

  const curLink = selectedItem?.type === 'link' ? robot.links.find((l) => l.name === selectedItem.name) : null;
  const curJoint = selectedItem?.type === 'joint' ? robot.joints.find((j) => j.name === selectedItem.name) : null;

  const studioInner = (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: 0,
        overflow: 'hidden',
        backgroundColor: '#101216',
      }}
    >
      {/* Top Header Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: isDocked ? '8px 12px' : '12px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        backgroundColor: '#161920',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: isDocked ? 8 : 12 }}>
          <div style={{
            width: isDocked ? 28 : 34,
            height: isDocked ? 28 : 34,
            borderRadius: 6,
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-blue)',
            flexShrink: 0,
          }}>
            <GitFork size={isDocked ? 15 : 18} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
              <span style={{ margin: 0, fontSize: isDocked ? 13 : 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                Robot Studio
              </span>
              <span style={{
                fontSize: 10,
                padding: '1px 6px',
                borderRadius: 4,
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                color: 'var(--accent-blue)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                fontWeight: 600,
                fontFamily: 'var(--font-mono)'
              }}>
                {robot.name}
              </span>
              <span style={{
                fontSize: 9,
                padding: '1px 5px',
                borderRadius: 3,
                backgroundColor: validationResult?.valid ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                color: validationResult?.valid ? 'var(--accent-green)' : '#ef4444',
                fontWeight: 600,
              }}>
                {validationResult?.valid ? 'Valid Tree' : 'Audit Errors'}
              </span>
            </div>
            {!isDocked && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Kinematics, inertial dynamics, actuator control modes, and physics compilation for MuJoCo
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isDocked && onSetDockWidth && (
            <button
              className="rf-btn"
              onClick={() => onSetDockWidth((dockWidth || 540) > 650 ? 540 : 720)}
              style={{ fontSize: 10, padding: '3px 7px', gap: 4 }}
              title={(dockWidth || 540) > 650 ? "Compact Studio (give 3D Viewport more room)" : "Expand Studio to Wide View"}
            >
              {(dockWidth || 540) > 650 ? <Minimize2 size={11} /> : <Maximize2 size={11} />}
              <span>{(dockWidth || 540) > 650 ? 'Compact' : 'Wide'}</span>
            </button>
          )}

          <button
            className="rf-btn rf-btn-primary"
            onClick={handleApplyToScene}
            style={{ fontSize: 10, padding: isDocked ? '3px 8px' : '6px 14px', display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600 }}
            title="Compile and load this articulated robot into the active 3D MuJoCo viewport"
          >
            <Play size={11} fill="currentColor" />
            <span>Apply to Scene</span>
          </button>

          <button
            className="rf-btn"
            onClick={handlePrepareURDF}
            style={{ fontSize: 10, padding: isDocked ? '3px 7px' : '6px 10px', display: 'flex', alignItems: 'center', gap: 4 }}
            title="Export and view URDF XML representation"
          >
            <Save size={11} />
            <span>URDF</span>
          </button>

          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 4,
            }}
            title="Close Robot Studio"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Tab Navigation Ribbon */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border-subtle)',
        backgroundColor: '#14161c',
        padding: isDocked ? '0 8px' : '0 20px',
        gap: 2,
        flexShrink: 0,
        overflowX: 'auto',
      }}>
        {[
          { id: 'builder', label: isDocked ? 'Builder' : '1. Robot Builder', icon: Box },
          { id: 'control', label: isDocked ? 'Joint Controls' : '2. Joint Control Studio', icon: Sliders },
          { id: 'kinematics', label: isDocked ? 'Kinematics' : '3. Kinematics Tools (FK / IK)', icon: Compass },
          { id: 'validation', label: isDocked ? 'Diagnostics' : '4. Physical Diagnostics', icon: ShieldCheck },
          { id: 'export', label: isDocked ? 'Presets' : '5. Presets & Persistence', icon: Save },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id as any);
                if (tab.id === 'validation') runValidation(robot);
                if (tab.id === 'kinematics') runFK(robot);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: isDocked ? '7px 9px' : '10px 14px',
                fontSize: 11,
                fontWeight: isActive ? 600 : 500,
                color: isActive ? 'var(--accent-blue)' : 'var(--text-secondary)',
                borderBottom: isActive ? '2px solid var(--accent-blue)' : '2px solid transparent',
                background: 'none',
                borderTop: 'none',
                borderLeft: 'none',
                borderRight: 'none',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={13} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Studio Body Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: isDocked ? 12 : 20 }}>
        {/* TAB 1: ROBOT BUILDER */}
        {activeTab === 'builder' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: isDocked ? ((dockWidth || 540) > 650 ? '260px 1fr' : '200px 1fr') : '320px 1fr',
            gap: isDocked ? 12 : 20,
            height: '100%'
          }}>
              {/* Left Column: Link & Joint Kinematic Hierarchy */}
              <div style={{
                backgroundColor: 'var(--bg-panel)',
                borderRadius: 8,
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden'
              }}>
                <div style={{
                  padding: '10px 14px',
                  borderBottom: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  backgroundColor: 'rgba(255,255,255,0.02)'
                }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Kinematic Tree ({robot.links.length} Links, {robot.joints.length} Joints)
                  </span>
                  <div style={{ display: 'flex', gap: 4 }}>
                    <button
                      className="rf-btn"
                      onClick={handleAddLink}
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      title="Add child link"
                    >
                      <Plus size={11} /> Link
                    </button>
                    <button
                      className="rf-btn"
                      onClick={handleAddJoint}
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      title="Add connecting joint"
                    >
                      <Plus size={11} /> Joint
                    </button>
                  </div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: 12 }}>
                  {/* Robot Global Config Card */}
                  <div style={{
                    padding: 10,
                    borderRadius: 6,
                    backgroundColor: 'rgba(56, 189, 248, 0.05)',
                    border: '1px solid rgba(56, 189, 248, 0.2)',
                    marginBottom: 12,
                    fontSize: 11
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{ fontWeight: 700, color: 'var(--accent-blue)' }}>Robot Base Configuration</span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>Model Name:</span>
                        <input
                          type="text"
                          className="rf-input"
                          style={{ width: '100%', marginTop: 2, fontSize: 11 }}
                          value={robot.name}
                          onChange={(e) => setRobot({ ...robot, name: e.target.value })}
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 2 }}>
                        <span style={{ color: 'var(--text-muted)' }}>Root Link:</span>
                        <select
                          className="rf-select"
                          style={{ fontSize: 10 }}
                          value={robot.root_link}
                          onChange={(e) => setRobot({ ...robot, root_link: e.target.value })}
                        >
                          {robot.links.map((l) => (
                            <option key={l.name} value={l.name}>{l.name}</option>
                          ))}
                        </select>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Base Type:</span>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: 10 }}>
                          <input
                            type="checkbox"
                            checked={robot.is_fixed_base}
                            onChange={(e) => setRobot({ ...robot, is_fixed_base: e.target.checked })}
                          />
                          Fixed Base
                        </label>
                      </div>
                    </div>
                  </div>

                  {/* Links Section */}
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Layers size={12} color="var(--accent-blue)" />
                    <span>Links ({robot.links.length})</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 14 }}>
                    {robot.links.map((link) => {
                      const isSelected = selectedItem?.type === 'link' && selectedItem.name === link.name;
                      const isRoot = robot.root_link === link.name;
                      return (
                        <div
                          key={link.name}
                          onClick={() => setSelectedItem({ type: 'link', name: link.name })}
                          style={{
                            padding: '6px 10px',
                            borderRadius: 5,
                            backgroundColor: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255,255,255,0.02)',
                            border: isSelected ? '1px solid var(--accent-blue)' : '1px solid transparent',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            fontSize: 11,
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <Box size={13} color={link.visual?.color || '#38bdf8'} />
                            <span style={{ fontWeight: isSelected ? 600 : 400, color: isSelected ? '#fff' : 'var(--text-primary)' }}>
                              {link.name}
                            </span>
                            {isRoot && (
                              <span style={{ fontSize: 9, padding: '1px 5px', backgroundColor: 'rgba(56, 189, 248, 0.2)', color: 'var(--accent-blue)', borderRadius: 3 }}>
                                ROOT
                              </span>
                            )}
                          </div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                            {link.inertial?.mass ?? 1} kg
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Joints Section */}
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <GitFork size={12} color="var(--accent-orange)" />
                    <span>Joints ({robot.joints.length})</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {robot.joints.map((joint) => {
                      const isSelected = selectedItem?.type === 'joint' && selectedItem.name === joint.name;
                      return (
                        <div
                          key={joint.name}
                          onClick={() => setSelectedItem({ type: 'joint', name: joint.name })}
                          style={{
                            padding: '6px 10px',
                            borderRadius: 5,
                            backgroundColor: isSelected ? 'rgba(249, 115, 22, 0.15)' : 'rgba(255,255,255,0.02)',
                            border: isSelected ? '1px solid var(--accent-orange)' : '1px solid transparent',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            fontSize: 11,
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <Sliders size={13} color="var(--accent-orange)" />
                            <span style={{ fontWeight: isSelected ? 600 : 400, color: isSelected ? '#fff' : 'var(--text-primary)' }}>
                              {joint.name}
                            </span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: 9, padding: '1px 5px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 3, textTransform: 'capitalize' }}>
                              {joint.type}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Right Column: Selected Entity Inspector */}
              <div style={{
                backgroundColor: 'var(--bg-panel)',
                borderRadius: 8,
                border: '1px solid var(--border-subtle)',
                padding: 18,
                overflowY: 'auto'
              }}>
                {curLink && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                      <div>
                        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                          Link Inspector: <span style={{ color: 'var(--accent-blue)' }}>{curLink.name}</span>
                        </h3>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          Configure visual shape, collision geometry, and inertial tensor
                        </div>
                      </div>
                      <button
                        className="rf-btn rf-btn-danger"
                        style={{ fontSize: 11, padding: '4px 10px' }}
                        onClick={() => handleDeleteLink(curLink.name)}
                      >
                        <Trash2 size={12} /> Delete Link
                      </button>
                    </div>

                    {/* Link Name */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Link Identifier</div>
                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Name:</span>
                        <input
                          type="text"
                          className="rf-input"
                          value={curLink.name}
                          onChange={(e) => {
                            const newName = e.target.value;
                            const oldName = curLink.name;
                            const updatedLinks = robot.links.map(l => l.name === oldName ? { ...l, name: newName } : l);
                            const updatedJoints = robot.joints.map(j => ({
                              ...j,
                              parent_link: j.parent_link === oldName ? newName : j.parent_link,
                              child_link: j.child_link === oldName ? newName : j.child_link,
                            }));
                            let newRoot = robot.root_link === oldName ? newName : robot.root_link;
                            setRobot({ ...robot, root_link: newRoot, links: updatedLinks, joints: updatedJoints });
                            setSelectedItem({ type: 'link', name: newName });
                          }}
                        />
                      </div>
                    </div>

                    {/* Visual Geometry */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Visual Geometry</div>
                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Shape Type:</span>
                        <select
                          className="rf-select"
                          value={curLink.visual?.type || 'box'}
                          onChange={(e) => {
                            const t = e.target.value as any;
                            const defSize = t === 'sphere' ? [0.1] : (t === 'cylinder' ? [0.05, 0.15] : [0.2, 0.04, 0.04]);
                            const newVisual: LinkGeom = {
                              type: t,
                              size: defSize,
                              origin_pos: curLink.visual?.origin_pos || [0, 0, 0],
                              origin_rpy: curLink.visual?.origin_rpy || [0, 0, 0],
                              color: curLink.visual?.color || '#38bdf8'
                            };
                            const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: newVisual } : l);
                            setRobot({ ...robot, links: updatedLinks });
                          }}
                        >
                          <option value="box">Box (Rectangular Prism)</option>
                          <option value="cylinder">Cylinder</option>
                          <option value="sphere">Sphere</option>
                        </select>
                      </div>

                      {/* Dimensions based on shape */}
                      {curLink.visual?.type === 'box' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 8 }}>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-X (m)</span>
                            <input
                              type="number"
                              step="0.02"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[0] || 0.1}
                              onChange={(e) => {
                                const val = Math.max(0.005, parseFloat(e.target.value) || 0.1);
                                const newSize = [val, curLink.visual!.size[1] || 0.05, curLink.visual!.size[2] || 0.05];
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: newSize } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Y (m)</span>
                            <input
                              type="number"
                              step="0.02"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[1] || 0.05}
                              onChange={(e) => {
                                const val = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                const newSize = [curLink.visual!.size[0] || 0.1, val, curLink.visual!.size[2] || 0.05];
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: newSize } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Z (m)</span>
                            <input
                              type="number"
                              step="0.02"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[2] || 0.05}
                              onChange={(e) => {
                                const val = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                const newSize = [curLink.visual!.size[0] || 0.1, curLink.visual!.size[1] || 0.05, val];
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: newSize } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                        </div>
                      )}

                      {curLink.visual?.type === 'cylinder' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 8 }}>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Radius (m)</span>
                            <input
                              type="number"
                              step="0.01"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[0] || 0.05}
                              onChange={(e) => {
                                const r = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                const newSize = [r, curLink.visual!.size[1] || 0.15];
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: newSize } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Length (m)</span>
                            <input
                              type="number"
                              step="0.02"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[1] || 0.15}
                              onChange={(e) => {
                                const l_val = Math.max(0.005, parseFloat(e.target.value) || 0.15);
                                const newSize = [curLink.visual!.size[0] || 0.05, l_val];
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: newSize } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                        </div>
                      )}

                      {curLink.visual?.type === 'sphere' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, marginTop: 8 }}>
                          <div>
                            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Radius (m)</span>
                            <input
                              type="number"
                              step="0.01"
                              min="0.005"
                              className="rf-input"
                              value={curLink.visual.size[0] || 0.1}
                              onChange={(e) => {
                                const r = Math.max(0.005, parseFloat(e.target.value) || 0.1);
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, size: [r] } } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Visual Color */}
                      <div className="rf-prop-row" style={{ marginTop: 8 }}>
                        <span className="rf-prop-label">Visual Color:</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <input
                            type="color"
                            value={curLink.visual?.color || '#38bdf8'}
                            onChange={(e) => {
                              const col = e.target.value;
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, visual: { ...l.visual!, color: col } } : l);
                              setRobot({ ...robot, links: updatedLinks });
                            }}
                            style={{ width: 30, height: 24, border: 'none', background: 'transparent', cursor: 'pointer' }}
                          />
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{curLink.visual?.color || '#38bdf8'}</span>
                        </div>
                      </div>
                    </div>

                    {/* Collision Geometry */}
                    <div className="rf-prop-group">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <div className="rf-prop-title" style={{ margin: 0 }}>Collision Geometry (Physics Contacts)</div>
                        <button
                          className="rf-btn"
                          onClick={() => {
                            if (curLink.visual) {
                              const syncedCollision: LinkGeom = {
                                type: curLink.visual.type,
                                size: [...curLink.visual.size],
                                origin_pos: [...curLink.visual.origin_pos],
                                origin_rpy: [...curLink.visual.origin_rpy],
                                color: curLink.visual.color
                              };
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: syncedCollision } : l);
                              setRobot({ ...robot, links: updatedLinks });
                              onShowToast(`Collision geometry synchronized from visual shape for '${curLink.name}'.`);
                            }
                          }}
                          style={{ fontSize: 10, padding: '2px 8px' }}
                          title="Copy visual geometry shape and dimensions into collision shape"
                        >
                          Sync from Visual
                        </button>
                      </div>

                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Collision Status:</span>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11 }}>
                          <input
                            type="checkbox"
                            checked={!!curLink.collision}
                            onChange={(e) => {
                              const enabled = e.target.checked;
                              const newCollision = enabled
                                ? (curLink.visual ? { ...curLink.visual } : { type: 'box' as const, size: [0.15, 0.03, 0.03], origin_pos: [0, 0, 0] as [number, number, number], origin_rpy: [0, 0, 0] as [number, number, number], color: '#38bdf8' })
                                : null;
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: newCollision } : l);
                              setRobot({ ...robot, links: updatedLinks });
                            }}
                          />
                          <span>{curLink.collision ? 'Enabled (Generates Contact Forces)' : 'Disabled (Intangible)'}</span>
                        </label>
                      </div>

                      {curLink.collision && (
                        <div>
                          <div className="rf-prop-row" style={{ marginTop: 6 }}>
                            <span className="rf-prop-label">Collision Shape:</span>
                            <select
                              className="rf-select"
                              value={curLink.collision.type}
                              onChange={(e) => {
                                const t = e.target.value as any;
                                const defSize = t === 'sphere' ? [0.1] : (t === 'cylinder' ? [0.05, 0.15] : [0.15, 0.03, 0.03]);
                                const updatedLinks = robot.links.map(l => l.name === curLink.name ? {
                                  ...l,
                                  collision: { ...l.collision!, type: t, size: defSize }
                                } : l);
                                setRobot({ ...robot, links: updatedLinks });
                              }}
                            >
                              <option value="box">Box (Rectangular Prism)</option>
                              <option value="cylinder">Cylinder</option>
                              <option value="sphere">Sphere</option>
                            </select>
                          </div>

                          {/* Collision Dimensions */}
                          {curLink.collision.type === 'box' && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 8 }}>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-X (m)</span>
                                <input
                                  type="number"
                                  step="0.02"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[0] || 0.1}
                                  onChange={(e) => {
                                    const val = Math.max(0.005, parseFloat(e.target.value) || 0.1);
                                    const newSize = [val, curLink.collision!.size[1] || 0.05, curLink.collision!.size[2] || 0.05];
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: newSize } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Y (m)</span>
                                <input
                                  type="number"
                                  step="0.02"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[1] || 0.05}
                                  onChange={(e) => {
                                    const val = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                    const newSize = [curLink.collision!.size[0] || 0.1, val, curLink.collision!.size[2] || 0.05];
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: newSize } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Z (m)</span>
                                <input
                                  type="number"
                                  step="0.02"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[2] || 0.05}
                                  onChange={(e) => {
                                    const val = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                    const newSize = [curLink.collision!.size[0] || 0.1, curLink.collision!.size[1] || 0.05, val];
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: newSize } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                            </div>
                          )}

                          {curLink.collision.type === 'cylinder' && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 8 }}>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Radius (m)</span>
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[0] || 0.05}
                                  onChange={(e) => {
                                    const r = Math.max(0.005, parseFloat(e.target.value) || 0.05);
                                    const newSize = [r, curLink.collision!.size[1] || 0.15];
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: newSize } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Half-Length (m)</span>
                                <input
                                  type="number"
                                  step="0.02"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[1] || 0.15}
                                  onChange={(e) => {
                                    const l_val = Math.max(0.005, parseFloat(e.target.value) || 0.15);
                                    const newSize = [curLink.collision!.size[0] || 0.05, l_val];
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: newSize } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                            </div>
                          )}

                          {curLink.collision.type === 'sphere' && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, marginTop: 8 }}>
                              <div>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Radius (m)</span>
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0.005"
                                  className="rf-input"
                                  value={curLink.collision.size[0] || 0.1}
                                  onChange={(e) => {
                                    const r = Math.max(0.005, parseFloat(e.target.value) || 0.1);
                                    const updatedLinks = robot.links.map(l => l.name === curLink.name ? { ...l, collision: { ...l.collision!, size: [r] } } : l);
                                    setRobot({ ...robot, links: updatedLinks });
                                  }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Inertial Properties */}
                    <div className="rf-prop-group">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <div className="rf-prop-title" style={{ margin: 0 }}>Inertial Properties (MuJoCo Physics)</div>
                        <button
                          className="rf-btn"
                          onClick={handleAutoInertials}
                          style={{ fontSize: 10, padding: '2px 8px' }}
                        >
                          Auto-Calculate Mass Tensor
                        </button>
                      </div>
                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Mass (kg):</span>
                        <input
                          type="number"
                          step="0.1"
                          min="0.01"
                          className="rf-input"
                          style={{ width: 90 }}
                          value={curLink.inertial?.mass || 1.0}
                          onChange={(e) => {
                            const m = Math.max(0.01, parseFloat(e.target.value) || 1.0);
                            const updatedLinks = robot.links.map(l => l.name === curLink.name ? {
                              ...l,
                              inertial: { ...(l.inertial || { origin_pos: [0, 0, 0] as [number, number, number], ixx: 0.01, iyy: 0.01, izz: 0.01 }), mass: m }
                            } : l);
                            setRobot({ ...robot, links: updatedLinks });
                          }}
                        />
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginTop: 8 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Ixx (kg·m²)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="rf-input"
                            value={curLink.inertial?.ixx || 0.01}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0.01;
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? {
                                ...l,
                                inertial: { ...l.inertial!, ixx: val }
                              } : l);
                              setRobot({ ...robot, links: updatedLinks });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Iyy (kg·m²)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="rf-input"
                            value={curLink.inertial?.iyy || 0.01}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0.01;
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? {
                                ...l,
                                inertial: { ...l.inertial!, iyy: val }
                              } : l);
                              setRobot({ ...robot, links: updatedLinks });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Izz (kg·m²)</span>
                          <input
                            type="number"
                            step="0.005"
                            className="rf-input"
                            value={curLink.inertial?.izz || 0.01}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0.01;
                              const updatedLinks = robot.links.map(l => l.name === curLink.name ? {
                                ...l,
                                inertial: { ...l.inertial!, izz: val }
                              } : l);
                              setRobot({ ...robot, links: updatedLinks });
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {curJoint && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                      <div>
                        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                          Joint Inspector: <span style={{ color: 'var(--accent-orange)' }}>{curJoint.name}</span>
                        </h3>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          Configure motion type, parent/child connectivity, limits, and actuator gains
                        </div>
                      </div>
                      <button
                        className="rf-btn rf-btn-danger"
                        style={{ fontSize: 11, padding: '4px 10px' }}
                        onClick={() => handleDeleteJoint(curJoint.name)}
                      >
                        <Trash2 size={12} /> Delete Joint
                      </button>
                    </div>

                    {/* Joint Type & Connectivity */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Kinematic Connectivity</div>
                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Joint Type:</span>
                        <select
                          className="rf-select"
                          value={curJoint.type}
                          onChange={(e) => {
                            const t = e.target.value as any;
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, type: t } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          <option value="revolute">Revolute (Limited Hinge)</option>
                          <option value="continuous">Continuous (Unbounded Hinge)</option>
                          <option value="prismatic">Prismatic (Linear Slider)</option>
                          <option value="fixed">Fixed (Rigid Weld)</option>
                        </select>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 8 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Parent Link</span>
                          <select
                            className="rf-select"
                            style={{ width: '100%' }}
                            value={curJoint.parent_link}
                            onChange={(e) => {
                              const pl = e.target.value;
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, parent_link: pl } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          >
                            {robot.links.map(l => (
                              <option key={l.name} value={l.name}>{l.name}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Child Link</span>
                          <select
                            className="rf-select"
                            style={{ width: '100%' }}
                            value={curJoint.child_link}
                            onChange={(e) => {
                              const cl = e.target.value;
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, child_link: cl } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          >
                            {robot.links.map(l => (
                              <option key={l.name} value={l.name}>{l.name}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Joint Origin Frame & Link Assembly Snapping */}
                    <div className="rf-prop-group">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <div className="rf-prop-title" style={{ margin: 0 }}>Joint Origin Frame & Snapping</div>
                        <span style={{ fontSize: 10, color: 'var(--accent-blue)', fontWeight: 600 }}>Parent Link Relative</span>
                      </div>
                      
                      {/* Assembly Snapping Presets */}
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          title="Snap joint origin to parent link length extremity (+X end of parent visual geometry)"
                          onClick={() => {
                            const parentL = robot.links.find(l => l.name === curJoint.parent_link);
                            if (!parentL || !parentL.visual) return;
                            let snapX = 0.3;
                            if (parentL.visual.type === 'box') snapX = parentL.visual.size[0] * 2.0;
                            else if (parentL.visual.type === 'cylinder') snapX = parentL.visual.size[1] * 2.0;
                            else if (parentL.visual.type === 'sphere') snapX = parentL.visual.size[0] * 2.0;
                            const snappedPos: [number, number, number] = [parseFloat(snapX.toFixed(3)), 0, 0];
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: snappedPos } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          <Magnet size={11} style={{ marginRight: 4 }} /> Snap Parent Tip (+X)
                        </button>

                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          title="Snap joint origin to parent link top surface (+Z height of parent visual geometry)"
                          onClick={() => {
                            const parentL = robot.links.find(l => l.name === curJoint.parent_link);
                            if (!parentL || !parentL.visual) return;
                            let snapZ = 0.15;
                            if (parentL.visual.type === 'box') snapZ = parentL.visual.size[2] * 2.0;
                            else if (parentL.visual.type === 'cylinder') snapZ = parentL.visual.size[1] * 2.0;
                            else if (parentL.visual.type === 'sphere') snapZ = parentL.visual.size[0] * 2.0;
                            const snappedPos: [number, number, number] = [0, 0, parseFloat(snapZ.toFixed(3))];
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: snappedPos } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          <Magnet size={11} style={{ marginRight: 4 }} /> Snap Parent Top (+Z)
                        </button>

                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          title="Snap joint origin to parent link coordinate center (0, 0, 0)"
                          onClick={() => {
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: [0, 0, 0] as [number, number, number] } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          Snap Origin (0,0,0)
                        </button>
                      </div>

                      {/* Origin Position (X, Y, Z in meters) */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 8 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin X (m)</span>
                          <input
                            type="number"
                            step="0.05"
                            className="rf-input"
                            value={curJoint.origin_pos[0]}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              const newPos: [number, number, number] = [val, curJoint.origin_pos[1], curJoint.origin_pos[2]];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: newPos } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin Y (m)</span>
                          <input
                            type="number"
                            step="0.05"
                            className="rf-input"
                            value={curJoint.origin_pos[1]}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              const newPos: [number, number, number] = [curJoint.origin_pos[0], val, curJoint.origin_pos[2]];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: newPos } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin Z (m)</span>
                          <input
                            type="number"
                            step="0.05"
                            className="rf-input"
                            value={curJoint.origin_pos[2]}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              const newPos: [number, number, number] = [curJoint.origin_pos[0], curJoint.origin_pos[1], val];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_pos: newPos } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>

                      {/* Origin RPY (Roll, Pitch, Yaw in degrees) */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin Roll (deg)</span>
                          <input
                            type="number"
                            step="15"
                            className="rf-input"
                            value={Math.round((curJoint.origin_rpy[0] * 180) / Math.PI)}
                            onChange={(e) => {
                              const deg = parseFloat(e.target.value) || 0;
                              const rad = (deg * Math.PI) / 180.0;
                              const newRpy: [number, number, number] = [rad, curJoint.origin_rpy[1], curJoint.origin_rpy[2]];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_rpy: newRpy } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin Pitch (deg)</span>
                          <input
                            type="number"
                            step="15"
                            className="rf-input"
                            value={Math.round((curJoint.origin_rpy[1] * 180) / Math.PI)}
                            onChange={(e) => {
                              const deg = parseFloat(e.target.value) || 0;
                              const rad = (deg * Math.PI) / 180.0;
                              const newRpy: [number, number, number] = [curJoint.origin_rpy[0], rad, curJoint.origin_rpy[2]];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_rpy: newRpy } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Origin Yaw (deg)</span>
                          <input
                            type="number"
                            step="15"
                            className="rf-input"
                            value={Math.round((curJoint.origin_rpy[2] * 180) / Math.PI)}
                            onChange={(e) => {
                              const deg = parseFloat(e.target.value) || 0;
                              const rad = (deg * Math.PI) / 180.0;
                              const newRpy: [number, number, number] = [curJoint.origin_rpy[0], curJoint.origin_rpy[1], rad];
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, origin_rpy: newRpy } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Joint Axis */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Motion Axis Vector</div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          onClick={() => {
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [0, 0, 1] as [number, number, number] } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          +Z (Yaw)
                        </button>
                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          onClick={() => {
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [0, 1, 0] as [number, number, number] } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          +Y (Pitch)
                        </button>
                        <button
                          className="rf-btn"
                          style={{ fontSize: 10, padding: '3px 8px' }}
                          onClick={() => {
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [1, 0, 0] as [number, number, number] } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          +X (Roll)
                        </button>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Axis X</span>
                          <input
                            type="number"
                            step="0.1"
                            className="rf-input"
                            value={curJoint.axis[0]}
                            onChange={(e) => {
                              const ax = parseFloat(e.target.value) || 0;
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [ax, j.axis[1], j.axis[2]] as [number, number, number] } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Axis Y</span>
                          <input
                            type="number"
                            step="0.1"
                            className="rf-input"
                            value={curJoint.axis[1]}
                            onChange={(e) => {
                              const ay = parseFloat(e.target.value) || 0;
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [j.axis[0], ay, j.axis[2]] as [number, number, number] } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Axis Z</span>
                          <input
                            type="number"
                            step="0.1"
                            className="rf-input"
                            value={curJoint.axis[2]}
                            onChange={(e) => {
                              const az = parseFloat(e.target.value) || 0;
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, axis: [j.axis[0], j.axis[1], az] as [number, number, number] } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Joint Limits */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Physical Limits & Motion Bounds</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Lower Limit ({curJoint.type === 'prismatic' ? 'm' : 'rad'})</span>
                          <input
                            type="number"
                            step="0.1"
                            className="rf-input"
                            value={curJoint.limit?.lower ?? -3.14}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || -3.14;
                              const newLimit = { ...(curJoint.limit || { upper: 3.14, effort: 50.0, velocity: 5.0 }), lower: val };
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, limit: newLimit } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Upper Limit ({curJoint.type === 'prismatic' ? 'm' : 'rad'})</span>
                          <input
                            type="number"
                            step="0.1"
                            className="rf-input"
                            value={curJoint.limit?.upper ?? 3.14}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 3.14;
                              const newLimit = { ...(curJoint.limit || { lower: -3.14, effort: 50.0, velocity: 5.0 }), upper: val };
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, limit: newLimit } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max Effort ({curJoint.type === 'prismatic' ? 'N' : 'Nm'})</span>
                          <input
                            type="number"
                            step="5"
                            min="1"
                            className="rf-input"
                            value={curJoint.limit?.effort ?? 50.0}
                            onChange={(e) => {
                              const val = Math.max(1, parseFloat(e.target.value) || 50.0);
                              const newLimit = { ...(curJoint.limit || { lower: -3.14, upper: 3.14, velocity: 5.0 }), effort: val };
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, limit: newLimit } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max Velocity ({curJoint.type === 'prismatic' ? 'm/s' : 'rad/s'})</span>
                          <input
                            type="number"
                            step="0.5"
                            min="0.1"
                            className="rf-input"
                            value={curJoint.limit?.velocity ?? 5.0}
                            onChange={(e) => {
                              const val = Math.max(0.1, parseFloat(e.target.value) || 5.0);
                              const newLimit = { ...(curJoint.limit || { lower: -3.14, upper: 3.14, effort: 50.0 }), velocity: val };
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, limit: newLimit } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Actuator Configuration */}
                    <div className="rf-prop-group">
                      <div className="rf-prop-title">Actuator Configuration (MuJoCo Physics)</div>
                      <div className="rf-prop-row">
                        <span className="rf-prop-label">Control Mode:</span>
                        <select
                          className="rf-select"
                          value={curJoint.actuator_mode || 'position'}
                          onChange={(e) => {
                            const m = e.target.value as any;
                            const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, actuator_mode: m } : j);
                            setRobot({ ...robot, joints: updatedJoints });
                          }}
                        >
                          <option value="position">Position Control (&lt;position&gt; servo)</option>
                          <option value="velocity">Velocity Control (&lt;velocity&gt; drive)</option>
                          <option value="torque">Torque / Direct Motor (&lt;motor&gt;)</option>
                        </select>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 8 }}>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Position Gain (kp)</span>
                          <input
                            type="number"
                            step="10"
                            min="0"
                            className="rf-input"
                            value={curJoint.kp}
                            onChange={(e) => {
                              const val = Math.max(0, parseFloat(e.target.value) || 0);
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, kp: val } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Velocity Gain (kv)</span>
                          <input
                            type="number"
                            step="1"
                            min="0"
                            className="rf-input"
                            value={curJoint.kv}
                            onChange={(e) => {
                              const val = Math.max(0, parseFloat(e.target.value) || 0);
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, kv: val } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Damping</span>
                          <input
                            type="number"
                            step="0.05"
                            min="0"
                            className="rf-input"
                            value={curJoint.damping}
                            onChange={(e) => {
                              const val = Math.max(0, parseFloat(e.target.value) || 0);
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, damping: val } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                        <div>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Armature</span>
                          <input
                            type="number"
                            step="0.005"
                            min="0"
                            className="rf-input"
                            value={curJoint.armature}
                            onChange={(e) => {
                              const val = Math.max(0, parseFloat(e.target.value) || 0);
                              const updatedJoints = robot.joints.map(j => j.name === curJoint.name ? { ...j, armature: val } : j);
                              setRobot({ ...robot, joints: updatedJoints });
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: JOINT CONTROL STUDIO */}
          {activeTab === 'control' && (
            <div style={{ maxWidth: 880, margin: '0 auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Interactive Joint Control Studio
                  </h3>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Command Position, Velocity, or Torque actuators with real-time limit enforcement and telemetry
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="rf-btn" onClick={handleZeroAllJoints} style={{ fontSize: 11, padding: '4px 10px' }} title="Set all joint targets to 0">
                    Zero All
                  </button>
                  <button className="rf-btn" onClick={handleCenterLimits} style={{ fontSize: 11, padding: '4px 10px' }} title="Set joint positions to middle of limit range">
                    Center Limits
                  </button>
                  <button className="rf-btn" onClick={handleHoldCurrentPosition} style={{ fontSize: 11, padding: '4px 10px' }} title="Hold current measured positions">
                    Hold Position
                  </button>
                  <button className="rf-btn" onClick={handleRelaxTorques} style={{ fontSize: 11, padding: '4px 10px' }} title="Zero motor torques">
                    Relax Torques
                  </button>
                </div>
              </div>

              {robot.joints.length === 0 ? (
                <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)', backgroundColor: 'var(--bg-panel)', borderRadius: 8 }}>
                  No actuated joints found in robot. Add joints in the Robot Builder tab to control them here.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {robot.joints.map((joint) => {
                    const isFixed = joint.type === 'fixed';
                    const mode = joint.actuator_mode || 'position';
                    const curPos = simState.joint_positions?.[joint.name] ?? 0.0;
                    const curVel = simState.joint_velocities?.[joint.name] ?? 0.0;
                    const curTorque = simState.joint_torques?.[joint.name] ?? 0.0;

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

                    // Limit proximity detection
                    const isAtLimit = !isFixed && joint.limit && mode === 'position' && (
                      Math.abs(curPos - sliderMin) < 0.02 || Math.abs(curPos - sliderMax) < 0.02
                    );
                    const isNearLimit = !isFixed && joint.limit && mode === 'position' && !isAtLimit && (
                      Math.abs(curPos - sliderMin) < 0.12 || Math.abs(curPos - sliderMax) < 0.12
                    );

                    return (
                      <div
                        key={joint.name}
                        style={{
                          backgroundColor: 'var(--bg-panel)',
                          border: isAtLimit ? '1px solid #ef4444' : (isNearLimit ? '1px solid var(--accent-orange)' : '1px solid var(--border-subtle)'),
                          borderRadius: 8,
                          padding: 14,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>{joint.name}</span>
                            <span style={{ fontSize: 10, padding: '1px 6px', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 3, textTransform: 'capitalize' }}>
                              {joint.type}
                            </span>

                            {/* Mode switcher pills */}
                            <div style={{ display: 'flex', gap: 2, backgroundColor: 'rgba(0,0,0,0.3)', padding: 2, borderRadius: 4 }}>
                              {(['position', 'velocity', 'torque'] as const).map((m) => (
                                <button
                                  key={m}
                                  onClick={() => {
                                    const updatedJoints = robot.joints.map((j) => j.name === joint.name ? { ...j, actuator_mode: m } : j);
                                    setRobot({ ...robot, joints: updatedJoints });
                                    const initialVal = m === 'position' ? curPos : 0.0;
                                    setJointSliderVals((prev) => ({ ...prev, [joint.name]: initialVal }));
                                    controlRobotJoint(joint.name, initialVal);
                                  }}
                                  style={{
                                    fontSize: 9,
                                    padding: '1px 6px',
                                    borderRadius: 3,
                                    border: 'none',
                                    cursor: 'pointer',
                                    fontWeight: mode === m ? 700 : 400,
                                    backgroundColor: mode === m ? 'var(--accent-blue)' : 'transparent',
                                    color: mode === m ? '#fff' : 'var(--text-muted)'
                                  }}
                                >
                                  {m.toUpperCase()}
                                </button>
                              ))}
                            </div>

                            {isAtLimit && (
                              <span style={{ fontSize: 9, padding: '1px 6px', backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', borderRadius: 3, fontWeight: 700 }}>
                                AT LIMIT
                              </span>
                            )}
                            {isNearLimit && (
                              <span style={{ fontSize: 9, padding: '1px 6px', backgroundColor: 'rgba(249, 115, 22, 0.2)', color: 'var(--accent-orange)', borderRadius: 3 }}>
                                NEAR LIMIT
                              </span>
                            )}
                          </div>

                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-blue)' }}>
                            {mode === 'position'
                              ? (joint.type === 'prismatic'
                                ? `${curPos.toFixed(3)} m`
                                : `${curPos.toFixed(3)} rad (${(curPos * 180 / Math.PI).toFixed(1)}°)`)
                              : (mode === 'velocity'
                                ? `${curVel.toFixed(2)} ${unitLabel}`
                                : `${curTorque.toFixed(1)} ${unitLabel}`)}
                          </div>
                        </div>

                        {!isFixed && (
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <span style={{ fontSize: 10, color: 'var(--text-muted)', width: 44, textAlign: 'right' }}>
                                {sliderMin.toFixed(2)}
                              </span>
                              <input
                                type="range"
                                min={sliderMin}
                                max={sliderMax}
                                step={step}
                                value={targetVal}
                                onChange={(e) => {
                                  const val = parseFloat(e.target.value);
                                  setJointSliderVals((prev) => ({ ...prev, [joint.name]: val }));
                                  controlRobotJoint(joint.name, val);
                                }}
                                style={{ flex: 1, cursor: 'pointer' }}
                              />
                              <span style={{ fontSize: 10, color: 'var(--text-muted)', width: 44 }}>
                                {sliderMax.toFixed(2)}
                              </span>
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', marginTop: 8 }}>
                              <span>Measured Pos: <span style={{ fontFamily: 'var(--font-mono)' }}>{curPos.toFixed(3)} {joint.type === 'prismatic' ? 'm' : 'rad'}</span></span>
                              <span>Velocity: <span style={{ fontFamily: 'var(--font-mono)' }}>{curVel.toFixed(2)} {joint.type === 'prismatic' ? 'm/s' : 'rad/s'}</span></span>
                              <span>Torque: <span style={{ fontFamily: 'var(--font-mono)' }}>{curTorque.toFixed(1)} {joint.type === 'prismatic' ? 'N' : 'Nm'}</span></span>
                              <span>Target: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-green)', fontWeight: 600 }}>{targetVal.toFixed(3)} {unitLabel}</span></span>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: KINEMATICS TOOLS (FK / IK) */}
          {activeTab === 'kinematics' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              {/* Forward Kinematics Readout */}
              <div style={{ backgroundColor: 'var(--bg-panel)', borderRadius: 8, border: '1px solid var(--border-subtle)', padding: 18 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Compass size={16} color="var(--accent-blue)" />
                    <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                      Forward Kinematics (FK)
                    </h3>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    Live Auto-Updating
                  </span>
                </div>

                {fkResult ? (
                  <div>
                    <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.05)', border: '1px solid rgba(56, 189, 248, 0.2)', borderRadius: 6, padding: 12, marginBottom: 14 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-blue)', marginBottom: 6 }}>
                        End-Effector Pose (Link: '{fkResult.end_effector_link}')
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 12, fontFamily: 'var(--font-mono)', marginBottom: 8 }}>
                        <div>X: <span style={{ color: '#fff' }}>{fkResult.end_effector_position[0].toFixed(3)}m</span></div>
                        <div>Y: <span style={{ color: '#fff' }}>{fkResult.end_effector_position[1].toFixed(3)}m</span></div>
                        <div>Z: <span style={{ color: '#fff' }}>{fkResult.end_effector_position[2].toFixed(3)}m</span></div>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 11, color: 'var(--text-secondary)' }}>
                        <div>Roll: {fkResult.end_effector_rpy_deg[0].toFixed(1)}°</div>
                        <div>Pitch: {fkResult.end_effector_rpy_deg[1].toFixed(1)}°</div>
                        <div>Yaw: {fkResult.end_effector_rpy_deg[2].toFixed(1)}°</div>
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8 }}>
                        Total Maximum Theoretical Reach: <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{fkResult.total_reach.toFixed(2)}m</span>
                      </div>
                    </div>

                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
                      Kinematic Link World Coordinates:
                    </div>
                    <div style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {Object.values(fkResult.link_poses).map((lp) => (
                        <div key={lp.link_name} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 8px', borderRadius: 4, backgroundColor: 'rgba(255,255,255,0.02)', fontSize: 11 }}>
                          <span style={{ fontWeight: 600 }}>{lp.link_name}</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            [{lp.position[0].toFixed(2)}, {lp.position[1].toFixed(2)}, {lp.position[2].toFixed(2)}]
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
                    Computing Forward Kinematics...
                  </div>
                )}
              </div>

              {/* Inverse Kinematics Solver */}
              <div style={{ backgroundColor: 'var(--bg-panel)', borderRadius: 8, border: '1px solid var(--border-subtle)', padding: 18 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                  <Target size={16} color="var(--accent-green)" />
                  <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Inverse Kinematics (IK) Solver
                  </h3>
                </div>

                <div className="rf-prop-group">
                  <div className="rf-prop-title">Target End-Effector Coordinates (meters)</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                    <div>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Target X</span>
                      <input
                        type="number"
                        step="0.05"
                        className="rf-input"
                        value={ikTargetPos[0]}
                        onChange={(e) => setIkTargetPos([parseFloat(e.target.value) || 0, ikTargetPos[1], ikTargetPos[2]])}
                      />
                    </div>
                    <div>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Target Y</span>
                      <input
                        type="number"
                        step="0.05"
                        className="rf-input"
                        value={ikTargetPos[1]}
                        onChange={(e) => setIkTargetPos([ikTargetPos[0], parseFloat(e.target.value) || 0, ikTargetPos[2]])}
                      />
                    </div>
                    <div>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Target Z</span>
                      <input
                        type="number"
                        step="0.05"
                        className="rf-input"
                        value={ikTargetPos[2]}
                        onChange={(e) => setIkTargetPos([ikTargetPos[0], ikTargetPos[1], parseFloat(e.target.value) || 0])}
                      />
                    </div>
                  </div>

                  {/* Reach Presets */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
                    <button
                      className="rf-btn"
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      onClick={() => {
                        const bx = robot.base_position[0];
                        const by = robot.base_position[1];
                        const bz = robot.base_position[2];
                        setIkTargetPos([bx + 0.40, by, bz + 0.15]);
                      }}
                    >
                      Reach +X
                    </button>
                    <button
                      className="rf-btn"
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      onClick={() => {
                        const bx = robot.base_position[0];
                        const by = robot.base_position[1];
                        const bz = robot.base_position[2];
                        setIkTargetPos([bx, by + 0.40, bz + 0.15]);
                      }}
                    >
                      Reach +Y
                    </button>
                    <button
                      className="rf-btn"
                      style={{ fontSize: 10, padding: '3px 8px' }}
                      onClick={() => {
                        const bx = robot.base_position[0];
                        const by = robot.base_position[1];
                        const bz = robot.base_position[2];
                        setIkTargetPos([bx, by, bz + 0.45]);
                      }}
                    >
                      Reach High (+Z)
                    </button>
                    {fkResult && (
                      <button
                        className="rf-btn"
                        style={{ fontSize: 10, padding: '3px 8px' }}
                        onClick={() => setIkTargetPos([...fkResult.end_effector_position])}
                      >
                        Copy Current Pose
                      </button>
                    )}
                    {targetPosition && (
                      <button
                        className="rf-btn"
                        style={{ fontSize: 10, padding: '3px 8px' }}
                        onClick={() => setIkTargetPos([targetPosition[0], targetPosition[1], targetPosition[2]])}
                      >
                        Copy Scene Target
                      </button>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 10, marginBottom: 14 }}>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Tolerance (m)</span>
                    <input
                      type="number"
                      step="0.001"
                      min="0.001"
                      className="rf-input"
                      value={ikTolerance}
                      onChange={(e) => setIkTolerance(Math.max(0.001, parseFloat(e.target.value) || 0.005))}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max Iterations</span>
                    <input
                      type="number"
                      step="20"
                      min="20"
                      className="rf-input"
                      value={ikMaxIter}
                      onChange={(e) => setIkMaxIter(Math.max(20, parseInt(e.target.value) || 100))}
                    />
                  </div>
                </div>

                <button
                  className="rf-btn rf-btn-primary"
                  onClick={handleSolveIK}
                  disabled={isSolvingIK}
                  style={{ width: '100%', padding: '8px 14px', fontSize: 12, display: 'flex', justifyContent: 'center', gap: 6 }}
                >
                  <Compass size={14} />
                  <span>{isSolvingIK ? 'Solving Kinematics...' : 'Solve Inverse Kinematics'}</span>
                </button>

                {/* Honest IK Outcome Report */}
                {ikResult && (
                  <div style={{
                    marginTop: 14,
                    padding: 12,
                    borderRadius: 6,
                    backgroundColor: ikResult.success
                      ? 'rgba(34, 197, 94, 0.1)'
                      : (ikResult.status === 'JOINT_LIMITS_BLOCKED' ? 'rgba(249, 115, 22, 0.1)' : 'rgba(239, 68, 68, 0.1)'),
                    border: ikResult.success
                      ? '1px solid rgba(34, 197, 94, 0.3)'
                      : (ikResult.status === 'JOINT_LIMITS_BLOCKED' ? '1px solid rgba(249, 115, 22, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)'),
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: ikResult.success
                          ? 'var(--accent-green)'
                          : (ikResult.status === 'JOINT_LIMITS_BLOCKED' ? 'var(--accent-orange)' : '#ef4444')
                      }}>
                        Status: {ikResult.status}
                      </span>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {ikResult.iterations} iterations · Err: {(ikResult.error_distance * 1000).toFixed(1)}mm
                      </span>
                    </div>

                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
                      {ikResult.message}
                    </div>

                    {ikResult.success && (
                      <button
                        className="rf-btn"
                        onClick={handleApplyIKSolution}
                        style={{
                          width: '100%',
                          backgroundColor: 'var(--accent-green)',
                          color: '#fff',
                          border: 'none',
                          fontSize: 11,
                          padding: '6px 10px',
                          display: 'flex',
                          justifyContent: 'center',
                          gap: 6,
                          fontWeight: 600
                        }}
                      >
                        <Check size={13} /> Apply Solution to Robot Joints
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 4: PHYSICAL DIAGNOSTICS & VALIDATION */}
          {activeTab === 'validation' && (
            <div style={{ maxWidth: 840, margin: '0 auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Physical Robot Validation Audit
                  </h3>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Diagnostics for kinematic cycles, disconnected links, collision geometry, inertial tensors, and joint limits
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    className="rf-btn"
                    onClick={handleAutoInertials}
                    style={{ fontSize: 11, padding: '5px 10px' }}
                  >
                    Auto-Calculate Inertials
                  </button>
                  <button
                    className="rf-btn"
                    onClick={() => runValidation(robot)}
                    disabled={isValidating}
                    style={{ fontSize: 11, padding: '5px 12px', display: 'flex', alignItems: 'center', gap: 6 }}
                  >
                    <RefreshCw size={12} className={isValidating ? 'spin' : ''} />
                    <span>Re-run Audit</span>
                  </button>
                </div>
              </div>

              {validationResult ? (
                <div>
                  <div style={{
                    padding: 14,
                    borderRadius: 8,
                    backgroundColor: validationResult.valid ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    border: validationResult.valid ? '1px solid rgba(34, 197, 94, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
                    marginBottom: 16,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12
                  }}>
                    {validationResult.valid ? (
                      <CheckCircle2 size={24} color="var(--accent-green)" />
                    ) : (
                      <AlertCircle size={24} color="#ef4444" />
                    )}
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: validationResult.valid ? 'var(--accent-green)' : '#ef4444' }}>
                        {validationResult.valid ? 'MuJoCo Simulation Ready' : 'Physics Validation Errors Detected'}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {validationResult.summary}
                      </div>
                    </div>
                  </div>

                  {/* Errors List */}
                  {validationResult.errors.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#ef4444', marginBottom: 8, textTransform: 'uppercase' }}>
                        Critical Blocking Errors ({validationResult.errors.length}):
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {validationResult.errors.map((diag, i) => (
                          <div key={i} style={{ padding: 12, borderRadius: 6, backgroundColor: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                              <span style={{ fontWeight: 600, fontSize: 12, color: '#f87171' }}>[{diag.code}] {diag.target}</span>
                              <span style={{ fontSize: 10, color: '#ef4444', textTransform: 'uppercase', fontWeight: 700 }}>Error</span>
                            </div>
                            <div style={{ fontSize: 12, color: 'var(--text-primary)', marginBottom: 4 }}>{diag.message}</div>
                            {diag.suggestion && (
                              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                Action: {diag.suggestion}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Warnings List */}
                  {validationResult.warnings.length > 0 && (
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-orange)', marginBottom: 8, textTransform: 'uppercase' }}>
                        Non-Blocking Warnings ({validationResult.warnings.length}):
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {validationResult.warnings.map((diag, i) => (
                          <div key={i} style={{ padding: 12, borderRadius: 6, backgroundColor: 'rgba(249, 115, 22, 0.08)', border: '1px solid rgba(249, 115, 22, 0.25)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                              <span style={{ fontWeight: 600, fontSize: 12, color: '#fb923c' }}>[{diag.code}] {diag.target}</span>
                              <span style={{ fontSize: 10, color: 'var(--accent-orange)', textTransform: 'uppercase', fontWeight: 700 }}>Warning</span>
                            </div>
                            <div style={{ fontSize: 12, color: 'var(--text-primary)', marginBottom: 4 }}>{diag.message}</div>
                            {diag.suggestion && (
                              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                Recommendation: {diag.suggestion}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ padding: 30, textAlign: 'center', color: 'var(--text-muted)' }}>
                  Running validation checks...
                </div>
              )}
            </div>
          )}

          {/* TAB 5: PRESETS & PERSISTENCE */}
          {activeTab === 'export' && (
            <div style={{ maxWidth: 880, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
              {/* Save to Asset Library Section */}
              <div style={{
                backgroundColor: 'var(--bg-panel)',
                borderRadius: 8,
                border: '1px solid var(--border-subtle)',
                padding: 16
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <Save size={16} color="var(--accent-green)" />
                  <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Save Robot to Asset Library
                  </h3>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr 1fr auto', gap: 10, alignItems: 'flex-end' }}>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Asset Name</span>
                    <input
                      type="text"
                      className="rf-input"
                      value={assetSaveName}
                      onChange={(e) => setAssetSaveName(e.target.value)}
                      placeholder="e.g. 3-DOF Arm"
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Description</span>
                    <input
                      type="text"
                      className="rf-input"
                      value={assetSaveDesc}
                      onChange={(e) => setAssetSaveDesc(e.target.value)}
                      placeholder="Brief functional description"
                    />
                  </div>
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Tags (comma-separated)</span>
                    <input
                      type="text"
                      className="rf-input"
                      value={assetSaveTags}
                      onChange={(e) => setAssetSaveTags(e.target.value)}
                      placeholder="manipulator, scara"
                    />
                  </div>
                  <button
                    className="rf-btn rf-btn-primary"
                    onClick={handleSaveToAssetLibrary}
                    disabled={isSavingAsset || !assetSaveName.trim()}
                    style={{ padding: '7px 14px', fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 5 }}
                  >
                    <Save size={12} />
                    <span>{isSavingAsset ? 'Saving...' : 'Save to Library'}</span>
                  </button>
                </div>
              </div>

              {/* Saved User Robot Assets from Library */}
              {customRobotAssets.length > 0 && (
                <div style={{
                  backgroundColor: 'var(--bg-panel)',
                  borderRadius: 8,
                  border: '1px solid var(--border-subtle)',
                  padding: 16
                }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10 }}>
                    Saved Custom Robots in Asset Library ({customRobotAssets.length})
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
                    {customRobotAssets.map((asset) => (
                      <div
                        key={asset.id}
                        style={{
                          backgroundColor: 'rgba(255,255,255,0.02)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 6,
                          padding: 12,
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'space-between',
                          gap: 8
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)' }}>{asset.name}</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{asset.description || 'Custom robot model'}</div>
                          <div style={{ fontSize: 10, color: 'var(--accent-blue)', marginTop: 4 }}>
                            {asset.data?.config?.links?.length || 0} Links · {asset.data?.config?.joints?.length || 0} Joints
                          </div>
                        </div>
                        <button
                          className="rf-btn"
                          onClick={() => handleLoadAsset(asset)}
                          style={{ fontSize: 10, padding: '4px 8px', alignSelf: 'flex-start' }}
                        >
                          Load into Studio
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Built-In Presets Grid */}
              <div style={{
                backgroundColor: 'var(--bg-panel)',
                borderRadius: 8,
                border: '1px solid var(--border-subtle)',
                padding: 16
              }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10 }}>
                  Built-In Engineering Presets ({presets.length})
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
                  {presets.map((preset) => (
                    <div
                      key={preset.name}
                      style={{
                        backgroundColor: 'rgba(255,255,255,0.02)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 6,
                        padding: 12,
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        gap: 8
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--accent-blue)' }}>{preset.name}</div>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                          {preset.links.length} Links · {preset.joints.length} Joints ({preset.joints.map(j => j.type).join(', ')})
                        </div>
                      </div>
                      <button
                        className="rf-btn"
                        onClick={() => handleLoadPreset(preset)}
                        style={{ fontSize: 10, padding: '4px 8px', alignSelf: 'flex-start' }}
                      >
                        Load Preset
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Model Interchange: JSON Export/Import & URDF XML */}
              <div style={{
                backgroundColor: 'var(--bg-panel)',
                borderRadius: 8,
                border: '1px solid var(--border-subtle)',
                padding: 16
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Model Interchange (URDF & JSON)
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="rf-btn"
                      onClick={handleExportJSON}
                      style={{ fontSize: 10, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 5 }}
                      title="Download complete robot definition as JSON"
                    >
                      <Download size={11} /> JSON
                    </button>
                    <button
                      className="rf-btn"
                      onClick={() => setShowJsonModal(true)}
                      style={{ fontSize: 10, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 5 }}
                      title="Import robot definition from JSON"
                    >
                      <Upload size={11} /> Import JSON
                    </button>
                    <button
                      className="rf-btn"
                      onClick={() => {
                        navigator.clipboard.writeText(exportedURDF);
                        setCopiedURDF(true);
                        setTimeout(() => setCopiedURDF(false), 2000);
                        onShowToast('URDF copied to clipboard!');
                      }}
                      style={{ fontSize: 10, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 5 }}
                    >
                      {copiedURDF ? <Check size={11} color="var(--accent-green)" /> : <Copy size={11} />}
                      <span>{copiedURDF ? 'Copied' : 'Copy URDF'}</span>
                    </button>
                    <button
                      className="rf-btn"
                      onClick={() => {
                        const blob = new Blob([exportedURDF], { type: 'text/xml' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${robot.name.toLowerCase().replace(/\s+/g, '_')}.urdf`;
                        a.click();
                        URL.revokeObjectURL(url);
                        onShowToast('URDF file downloaded!');
                      }}
                      style={{ fontSize: 10, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 5 }}
                    >
                      <Download size={11} /> .urdf
                    </button>
                  </div>
                </div>

                <div style={{
                  backgroundColor: '#0a0b0e',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  padding: 12,
                  maxHeight: 260,
                  overflowY: 'auto'
                }}>
                  <pre style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: 11, color: '#38bdf8', whiteSpace: 'pre-wrap' }}>
                    {exportedURDF || 'Click "URDF" in top header or wait for generation...'}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );

  if (isDocked) {
    return (
      <div
        className="rf-studio-docked-panel"
        style={{
          width: dockWidth || 540,
          minWidth: 460,
          maxWidth: '65vw',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#111317',
          borderLeft: '1px solid var(--border-subtle)',
          position: 'relative',
          overflow: 'hidden',
          zIndex: 15,
        }}
      >
        {studioInner}

        {/* JSON Import Dialog */}
        {showJsonModal && (
          <div
            style={{
              position: 'absolute',
              top: 40,
              left: 12,
              right: 12,
              zIndex: 100,
              backgroundColor: '#161822',
              border: '1px solid var(--border-strong)',
              boxShadow: '0 16px 40px rgba(0,0,0,0.85)',
              padding: 16,
              borderRadius: 8
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                Import Robot from JSON
              </h3>
              <button
                onClick={() => setShowJsonModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={15} />
              </button>
            </div>
            <textarea
              className="rf-input"
              style={{ width: '100%', height: 180, fontFamily: 'var(--font-mono)', fontSize: 11, resize: 'vertical' }}
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              placeholder="Paste GenericRobotModel JSON here..."
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
              <button className="rf-btn" onClick={() => setShowJsonModal(false)}>
                Cancel
              </button>
              <button
                className="rf-btn rf-btn-primary"
                onClick={() => handleImportJSON(jsonText)}
                disabled={!jsonText.trim()}
              >
                Import Model
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="rf-modal-overlay" onClick={onClose} style={{ zIndex: 120 }}>
      <div
        className="rf-modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '92vw',
          maxWidth: 1280,
          height: '88vh',
          display: 'flex',
          flexDirection: 'column',
          padding: 0,
          overflow: 'hidden',
          backgroundColor: '#101216',
          border: '1px solid var(--border-subtle)',
          boxShadow: '0 16px 48px rgba(0,0,0,0.8)',
          borderRadius: 10,
        }}
      >
        {studioInner}
      </div>

      {showJsonModal && (
        <div
          className="rf-modal-overlay"
          onClick={() => setShowJsonModal(false)}
          style={{ zIndex: 140 }}
        >
          <div
            className="rf-modal-content"
            onClick={(e) => e.stopPropagation()}
            style={{ width: 560, backgroundColor: '#14161c', padding: 20, borderRadius: 8 }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                Import Robot from JSON
              </h3>
              <button
                onClick={() => setShowJsonModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={16} />
              </button>
            </div>
            <textarea
              className="rf-input"
              style={{ width: '100%', height: 220, fontFamily: 'var(--font-mono)', fontSize: 11, resize: 'vertical' }}
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              placeholder="Paste GenericRobotModel JSON here..."
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 14 }}>
              <button className="rf-btn" onClick={() => setShowJsonModal(false)}>
                Cancel
              </button>
              <button
                className="rf-btn rf-btn-primary"
                onClick={() => handleImportJSON(jsonText)}
                disabled={!jsonText.trim()}
              >
                Import Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
