import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Project, SimulationState, SceneObject, LogMessage, ExperimentRunResult,
  PolicyTrainingResult, SensorSettings, URDFImportResult, WorkspaceSection,
  RobotConfig, GenericRobotModel
} from './types';
import {
  fetchProject, saveProjectToBackend, createNewProject, loadProjectFromJSON,
  resetProjectToSaved, controlSim, addSceneObject, updateSceneObject,
  deleteSceneObject, clearServerLogs, SimulationSocket, trainAvoidancePolicy, runLiveExperiment,
  duplicateSceneObject, duplicateProject, exportScene, importScene, importURDF, controlRobotJoint,
  clearGenericRobot, setAutoDrive, setGenericRobotModel
} from './services/api';
import { Toolbar } from './components/Toolbar';
import { ScenePanel } from './components/ScenePanel';
import { Viewport3D } from './components/Viewport3D';
import { PropertiesPanel } from './components/PropertiesPanel';
import { BottomPanel, TrajectoryPoint } from './components/BottomPanel';
import { AssetLibraryPanel } from './components/AssetLibraryPanel';
import { RobotCreatorModal } from './components/RobotCreatorModal';
import { EnvironmentCreatorModal } from './components/EnvironmentCreatorModal';
import { RobotStudioModal } from './components/RobotStudioModal';
import { useKeyboardTeleop } from './hooks/useKeyboardTeleop';
import { BrainCircuit, Sparkles, CheckCircle2, XCircle, X, Play, ShieldAlert, Upload, FileCode, AlertTriangle, FileText } from 'lucide-react';

const initialSimState: SimulationState = {
  status: 'stopped',
  sim_time: 0.0,
  wall_time: 0.0,
  step_count: 0,
  robot: {
    position: [0, 0, 0.07],
    quaternion: [1, 0, 0, 0],
    linear_velocity: [0, 0, 0],
    angular_velocity: [0, 0, 0],
    linear_speed: 0,
    angular_speed: 0,
    left_wheel_velocity: 0,
    right_wheel_velocity: 0,
    forward_vector: [1, 0, 0],
  },
  objects: {},
  contacts: [],
  errors: [],
};

const SAMPLE_URDF_TEXT = `<robot name="two_link_manipulator">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
  </link>

  <link name="arm_link1">
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="2.0"/>
  </joint>

  <link name="arm_link2">
    <inertial>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.4"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="elbow_joint" type="revolute">
    <parent link="arm_link1"/>
    <child link="arm_link2"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="5.0" velocity="3.0"/>
  </joint>
</robot>`;

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error: any }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }
  componentDidCatch(error: any, info: any) {
    console.error('RoboForge UI Error caught by boundary:', error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#111215', color: '#fff', gap: 14 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-orange)' }}>Simulation Interface Recovered</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 500, textAlign: 'center' }}>
            {this.state.error?.message || 'An interface synchronization error occurred.'}
          </div>
          <button
            className="rf-btn rf-btn-primary"
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
          >
            Reload Workspace
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export const AppContent: React.FC = () => {
  const [project, setProject] = useState<Project | null>(null);
  const [simState, setSimState] = useState<SimulationState>(initialSimState);
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Undo / Redo stacks
  const [undoStack, setUndoStack] = useState<SceneObject[][]>([]);
  const [redoStack, setRedoStack] = useState<SceneObject[][]>([]);

  // Trajectory recording & replay
  const [isRecordingTrajectory, setIsRecordingTrajectory] = useState<boolean>(false);
  const [recordedTrajectory, setRecordedTrajectory] = useState<TrajectoryPoint[]>([]);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);

  // URDF Import Modal
  const [showURDFModal, setShowURDFModal] = useState<boolean>(false);
  const [urdfInputText, setUrdfInputText] = useState<string>(SAMPLE_URDF_TEXT);
  const [urdfImportResult, setUrdfImportResult] = useState<URDFImportResult | null>(null);
  const [isImportingURDF, setIsImportingURDF] = useState<boolean>(false);

  // 3D Viewport Experiment Target & Trajectory
  const [targetPosition, setTargetPosition] = useState<[number, number, number] | null>(null);
  const [trajectoryTrail, setTrajectoryTrail] = useState<Array<[number, number, number]> | null>(null);
  const [latestLiveRun, setLatestLiveRun] = useState<ExperimentRunResult | null>(null);

  // Global Robot AI Training & Optimizer State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingResult, setTrainingResult] = useState<PolicyTrainingResult | null>(null);
  const [showGlobalTrainingModal, setShowGlobalTrainingModal] = useState<boolean>(false);

  // Spacious Workspace: Collapsible Sidebars & Auto-Drive Mode
  const [leftSidebarOpen, setLeftSidebarOpen] = useState<boolean>(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState<boolean>(true);
  const [autoDriveEnabled, setAutoDriveEnabled] = useState<boolean>(false);

  // Unified Simulation Workspace & Reusable Asset Library State
  const [workspaceSection, setWorkspaceSection] = useState<WorkspaceSection>('scene');
  const [showAssetLibrary, setShowAssetLibrary] = useState<boolean>(false);
  const [showRobotCreator, setShowRobotCreator] = useState<boolean>(false);
  const [showEnvironmentCreator, setShowEnvironmentCreator] = useState<boolean>(false);
  const [showRobotStudio, setShowRobotStudio] = useState<boolean>(false);
  const [studioDockWidth, setStudioDockWidth] = useState<number>(540);
  const [studioInitialRobot, setStudioInitialRobot] = useState<GenericRobotModel | null>(null);

  const handleOpenRobotStudio = (customRobot?: GenericRobotModel) => {
    if (customRobot) {
      setStudioInitialRobot(customRobot);
    } else if (project?.generic_robot) {
      setStudioInitialRobot(project.generic_robot);
    } else {
      setStudioInitialRobot(null);
    }
    setShowRobotStudio(true);
    setWorkspaceSection('robots');
    setRightSidebarOpen(true);
  };

  const handleApplyRobotFromStudio = async (robotModel: GenericRobotModel) => {
    try {
      const res = await setGenericRobotModel(robotModel);
      if (res && res.robot) {
        setProject((prev) => (prev ? { ...prev, generic_robot: res.robot } : prev));
        showToast(`Loaded '${robotModel.name}' into simulation scene.`);
      }
    } catch (e: any) {
      showToast(e.message || 'Failed to apply robot model to scene');
    }
  };

  const handleSelectWorkspaceSection = (sec: WorkspaceSection) => {
    setWorkspaceSection(sec);
    if (sec === 'assets') {
      setShowAssetLibrary(true);
      setShowRobotStudio(false);
      setLeftSidebarOpen(true);
    } else if (sec === 'robots') {
      setShowAssetLibrary(false);
      handleOpenRobotStudio();
    } else if (sec === 'scene') {
      setShowRobotStudio(false);
      setShowAssetLibrary(false);
      setLeftSidebarOpen(true);
      setRightSidebarOpen(true);
    } else if (sec === 'sensors') {
      setSelectedObjectId('robot');
      setRightSidebarOpen(true);
    } else if (sec === 'experiments') {
      // Focus experiments in bottom panel
    } else if (sec === 'results') {
      // Focus telemetry & results in bottom panel
    }
  };

  const handleApplyEnvironment = async (envData: {
    gravity: [number, number, number];
    world_boundaries: [number, number, number, number, number, number];
    objects: SceneObject[];
  }) => {
    if (!project) return;
    const updatedProject: Project = {
      ...project,
      objects: envData.objects,
      settings: {
        ...project.settings,
        gravity: envData.gravity,
        world_boundaries: envData.world_boundaries,
        enable_boundaries: true,
      },
    };
    setProject(updatedProject);
    try {
      await saveProjectToBackend(updatedProject);
      await controlSim('reset');
      showToast('Environment successfully applied to simulation scene');
    } catch (err: any) {
      showToast(`Failed to update environment: ${err.message}`);
    }
  };

  const handleRobotCreated = async (type: 'mobile' | 'articulated', config: RobotConfig | GenericRobotModel) => {
    if (!project) return;
    let updatedProject: Project;
    if (type === 'mobile') {
      updatedProject = {
        ...project,
        robot: config as RobotConfig,
        generic_robot: undefined,
      };
    } else {
      updatedProject = {
        ...project,
        generic_robot: config as GenericRobotModel,
      };
    }
    setProject(updatedProject);
    try {
      await saveProjectToBackend(updatedProject);
      await controlSim('reset');
      showToast(`New ${type} robot applied to simulation scene`);
    } catch (err: any) {
      showToast(`Failed to update robot: ${err.message}`);
    }
  };

  const handleAssetApplied = async (updatedProject: Project) => {
    setProject(updatedProject);
    try {
      await controlSim('reset');
      showToast('Asset successfully loaded into simulation scene');
    } catch (err: any) {
      showToast(`Failed to reset simulation: ${err.message}`);
    }
  };

  const [socket, setSocket] = useState<SimulationSocket | null>(null);
  const socketRef = useRef<SimulationSocket | null>(null);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleToggleAutoDrive = async () => {
    const next = !autoDriveEnabled;
    setAutoDriveEnabled(next);
    try {
      await setAutoDrive(next);
      socketRef.current?.sendAutoDrive(next);
      showToast(next ? 'Auto-Drive ON: Robot navigating towards target' : 'Auto-Drive OFF: Manual control');
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleClearGenericRobot = async () => {
    try {
      const updated = await clearGenericRobot();
      setProject(updated);
      setSelectedObjectId(null);
      showToast('Imported URDF robot model removed');
    } catch (e: any) {
      showToast(e.message);
    }
  };

  // Push snapshot to undo stack
  const pushHistory = useCallback((currentObjects: SceneObject[]) => {
    setUndoStack((prev) => [...prev.slice(-30), currentObjects.map((o) => ({ ...o }))]);
    setRedoStack([]);
  }, []);

  // Initial Project Load
  useEffect(() => {
    fetchProject()
      .then((p) => {
        setProject(p);
      })
      .catch((err) => {
        console.error('Failed to load project:', err);
        showToast('Failed to connect to RoboForge backend');
      });
  }, []);

  // Initialize WebSocket stream
  useEffect(() => {
    const sock = new SimulationSocket(
      (newState) => {
        setSimState(newState);

        // Record live trajectory points if active
        if (isRecordingTrajectory && newState.robot) {
          setRecordedTrajectory((prev) => [
            ...prev,
            {
              time: newState.sim_time,
              position: [...newState.robot.position],
              orientation: [...newState.robot.quaternion],
              linear_speed: newState.robot.linear_speed,
            },
          ]);
        }
      },
      (newLog) => {
        setLogs((prev) => [...prev.slice(-150), newLog]);
      },
      (connected) => {
        setIsConnected(connected);
      },
      (updatedProject) => {
        setProject(updatedProject);
      },
      (type, data) => {
        if (type === 'live_experiment_started') {
          if (data && data.target_position) {
            setTargetPosition(data.target_position);
          }
          setTrajectoryTrail([]);
          showToast(`Live 3D experiment '${data.name || ''}' running in viewport...`);
        } else if (type === 'live_experiment_completed') {
          const runResult = data as ExperimentRunResult;
          setLatestLiveRun(runResult);
          if (runResult.measurements && runResult.measurements.length > 0) {
            setTrajectoryTrail(runResult.measurements.map((m) => m.position));
          }
          showToast(`Experiment completed: ${runResult.metrics?.status_reason || runResult.status}`);
        }
      }
    );

    socketRef.current = sock;
    setSocket(sock);
    return () => {
      sock.destroy();
      setSocket(null);
    };
  }, [isRecordingTrajectory]);

  // Simulation Controls
  const handlePlay = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.sendAction('play');
    } else {
      controlSim('play');
    }
  }, []);

  const handlePause = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.sendAction('pause');
    } else {
      controlSim('pause');
    }
  }, []);

  const handleStep = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.sendAction('step');
    } else {
      controlSim('step');
    }
  }, []);

  const handleReset = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.sendAction('reset');
    } else {
      controlSim('reset');
    }
    showToast('Simulation reset to initial conditions');
  }, []);

  const handleTogglePlay = useCallback(() => {
    if (simState.status === 'running') {
      handlePause();
    } else {
      handlePlay();
    }
  }, [simState.status, handlePause, handlePlay]);

  // Manual teleop for on-screen D-pad buttons
  const handleManualTeleop = useCallback((linear: number, angular: number) => {
    socketRef.current?.sendTeleop(linear, angular);
  }, []);

  // Hook keyboard teleoperation (WASD, Arrows, Space, R)
  const activeKeys = useKeyboardTeleop(socketRef.current, handleTogglePlay, handleReset);

  // Speed and Timestep adjustments
  const handleChangeSpeed = async (multiplier: number) => {
    try {
      await controlSim('set_speed', multiplier);
      if (project) {
        setProject({
          ...project,
          settings: { ...project.settings, speed_multiplier: multiplier },
        });
      }
      showToast(`Simulation speed set to ${multiplier}x`);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleChangeTimestep = async (dt: number) => {
    try {
      await controlSim('set_timestep', dt);
      if (project) {
        setProject({
          ...project,
          settings: { ...project.settings, timestep: dt },
        });
      }
      showToast(`Physics timestep set to ${dt}s`);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  // Global Robot AI Optimization & Training Handlers
  const handleGlobalTrainRobot = async () => {
    setShowGlobalTrainingModal(true);
    setIsTraining(true);
    showToast('AI Optimizer: Training robot obstacle avoidance policy...');
    try {
      const res = await trainAvoidancePolicy('');
      setTrainingResult(res);
      showToast(res.message);
      // Refresh current project if updated
      const p = await fetchProject();
      setProject(p);
    } catch (e: any) {
      showToast(`Policy training failed: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleDeployAndRun3D = async () => {
    if (!trainingResult) return;
    setShowGlobalTrainingModal(false);
    showToast('Deploying optimized policy & launching live 3D run in viewport...');
    try {
      await runLiveExperiment({
        scenario_id: trainingResult.scenario_id,
        duration: 12.0,
      });
      const p = await fetchProject();
      setProject(p);
    } catch (e: any) {
      showToast(`Failed to launch 3D run: ${e.message}`);
    }
  };

  // Project Management Actions
  const handleNewProject = async () => {
    const name = window.prompt('Enter new simulation project name:', 'New Robotics Scene');
    if (!name) return;
    try {
      const newProj = await createNewProject(name);
      setProject(newProj);
      setSelectedObjectId(null);
      showToast(`Created new project '${newProj.name}'`);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleSaveProject = async () => {
    if (!project) return;
    try {
      const res = await saveProjectToBackend(project);
      showToast(`Project saved successfully to ${res.path}`);

      // Also trigger a browser JSON download for user portability
      const blob = new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project.name.toLowerCase().replace(/\s+/g, '_')}.rfproj.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleLoadProject = async (jsonContent: string) => {
    try {
      const loaded = await loadProjectFromJSON(jsonContent);
      setProject(loaded);
      setSelectedObjectId(null);
      showToast(`Loaded project '${loaded.name}'`);
    } catch (e: any) {
      showToast(`Failed to load project: ${e.message}`);
    }
  };

  const handleResetToSaved = async () => {
    try {
      const reverted = await resetProjectToSaved();
      setProject(reverted);
      setSelectedObjectId(null);
      showToast('Project reverted to last saved state');
    } catch (e: any) {
      showToast(e.message);
    }
  };

  // Scene Object Actions
  const handleAddObstacle = async (type: 'box' | 'sphere' | 'cylinder') => {
    const count = (project?.objects.length || 0) + 1;
    let size: number[] = [0.25, 0.25, 0.25];
    let color = '#e06c75';
    let name = `Box ${count}`;

    if (type === 'sphere') {
      size = [0.25];
      color = '#d19a66';
      name = `Sphere ${count}`;
    } else if (type === 'cylinder') {
      size = [0.2, 0.35];
      color = '#98c379';
      name = `Cylinder ${count}`;
    }

    const posX = 1.0 + ((count % 3) * 0.7);
    const posY = ((count % 4) - 1.5) * 0.7;
    const posZ = type === 'cylinder' ? 0.35 : (type === 'sphere' ? 0.25 : 0.25);

    try {
      const newObj = await addSceneObject({
        name,
        type,
        position: [posX, posY, posZ],
        rotation: [0, 0, 0],
        size,
        color,
        is_static: false,
        mass: 1.5,
      });

      if (project) {
        setProject({
          ...project,
          objects: [...project.objects, newObj],
        });
      }
      setSelectedObjectId(newObj.id);
      showToast(`Added ${newObj.name} to scene`);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  // Undo & Redo Handlers
  const handleUndo = useCallback(async () => {
    if (undoStack.length === 0 || !project) return;
    const previousObjects = undoStack[undoStack.length - 1];
    const nextUndo = undoStack.slice(0, undoStack.length - 1);
    setRedoStack((prev) => [...prev, project.objects.map((o) => ({ ...o }))]);
    setUndoStack(nextUndo);

    const updatedProject = { ...project, objects: previousObjects };
    setProject(updatedProject);
    await saveProjectToBackend(updatedProject).catch(() => {});
    showToast('Undo action executed');
  }, [undoStack, project]);

  const handleRedo = useCallback(async () => {
    if (redoStack.length === 0 || !project) return;
    const nextObjects = redoStack[redoStack.length - 1];
    const nextRedo = redoStack.slice(0, redoStack.length - 1);
    setUndoStack((prev) => [...prev, project.objects.map((o) => ({ ...o }))]);
    setRedoStack(nextRedo);

    const updatedProject = { ...project, objects: nextObjects };
    setProject(updatedProject);
    await saveProjectToBackend(updatedProject).catch(() => {});
    showToast('Redo action executed');
  }, [redoStack, project]);

  // Global Keyboard Shortcuts (Ctrl+Z / Ctrl+Y)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        if (e.shiftKey) {
          e.preventDefault();
          handleRedo();
        } else {
          e.preventDefault();
          handleUndo();
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        handleRedo();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo]);

  const handleUpdateObject = async (updated: SceneObject) => {
    try {
      if (project) {
        pushHistory(project.objects);
        setProject({
          ...project,
          objects: project.objects.map((o) => (o.id === updated.id ? updated : o)),
        });
      }
      await updateSceneObject(updated);
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleDeleteObject = async (id: string) => {
    try {
      if (project) {
        pushHistory(project.objects);
        await deleteSceneObject(id);
        setProject({
          ...project,
          objects: project.objects.filter((o) => o.id !== id),
        });
      }
      if (selectedObjectId === id) setSelectedObjectId(null);
      showToast('Object deleted');
    } catch (e: any) {
      showToast(e.message);
    }
  };

  const handleDuplicateObject = async (id: string) => {
    if (!project) return;
    try {
      pushHistory(project.objects);
      const newObj = await duplicateSceneObject(id);
      setProject({
        ...project,
        objects: [...project.objects, newObj],
      });
      setSelectedObjectId(newObj.id);
      showToast(`Duplicated object as '${newObj.name}'`);
    } catch (e: any) {
      showToast(`Duplicate failed: ${e.message}`);
    }
  };

  const handleDuplicateProject = async () => {
    try {
      const dup = await duplicateProject();
      setProject(dup);
      setSelectedObjectId(null);
      showToast(`Project cloned as '${dup.name}'`);
    } catch (e: any) {
      showToast(`Project clone failed: ${e.message}`);
    }
  };

  const handleExportScene = async () => {
    try {
      const sceneData = await exportScene();
      const blob = new Blob([JSON.stringify(sceneData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${(project?.name || 'scene').toLowerCase().replace(/\s+/g, '_')}_scene.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Scene exported successfully');
    } catch (e: any) {
      showToast(`Export failed: ${e.message}`);
    }
  };

  const handleImportURDFSubmit = async () => {
    if (!urdfInputText.trim()) return;
    setIsImportingURDF(true);
    try {
      const res = await importURDF(urdfInputText);
      setUrdfImportResult(res);
      const p = await fetchProject();
      setProject(p);
      showToast(res.robot ? `Imported robot '${res.robot.name}'` : 'URDF parsed');
    } catch (e: any) {
      showToast(`URDF import failed: ${e.message}`);
    } finally {
      setIsImportingURDF(false);
    }
  };

  const handleJointControl = async (jointName: string, targetPosition: number) => {
    try {
      await controlRobotJoint(jointName, targetPosition);
    } catch (e: any) {
      showToast(`Joint control failed: ${e.message}`);
    }
  };

  const handleUpdateSensors = async (sensors: SensorSettings) => {
    if (!project) return;
    const updated = {
      ...project,
      sensors,
    };
    setProject(updated);
    await saveProjectToBackend(updated).catch(() => {});
    showToast('Sensor configuration updated in MuJoCo model');
  };

  const handleUpdateBoundaries = async (enable: boolean, bounds: [number, number, number, number, number, number]) => {
    if (!project) return;
    const updated = {
      ...project,
      settings: {
        ...project.settings,
        enable_boundaries: enable,
        world_boundaries: bounds,
      },
    };
    setProject(updated);
    await saveProjectToBackend(updated).catch(() => {});
    showToast(`World boundaries ${enable ? 'enabled' : 'disabled'}`);
  };

  const handleToggleTrajectoryRecording = () => {
    if (isRecordingTrajectory) {
      setIsRecordingTrajectory(false);
      showToast(`Trajectory recording stopped (${recordedTrajectory.length} points)`);
    } else {
      setRecordedTrajectory([]);
      setReplayIndex(null);
      setIsRecordingTrajectory(true);
      showToast('Trajectory recording started...');
    }
  };

  const handleSeekReplay = (index: number | null) => {
    setReplayIndex(index);
    if (index !== null && recordedTrajectory[index]) {
      const pt = recordedTrajectory[index];
      // Display replayed robot position in simState for 3D inspection
      setSimState((prev) => ({
        ...prev,
        robot: {
          ...prev.robot,
          position: pt.position,
          quaternion: pt.orientation,
          linear_speed: pt.linear_speed,
        },
      }));
    }
  };

  const handleToggleVisibility = (id: string) => {
    if (!project) return;
    const target = project.objects.find((o) => o.id === id);
    if (!target) return;
    const updated = { ...target, visible: target.visible === false ? true : false };
    handleUpdateObject(updated);
  };

  const handleUpdateRobotLimits = async (maxLin: number, maxAng: number) => {
    if (!project) return;
    const updatedRobot = {
      ...project.robot,
      max_linear_velocity: maxLin,
      max_angular_velocity: maxAng,
    };
    const updatedProject = { ...project, robot: updatedRobot };
    setProject(updatedProject);
    await saveProjectToBackend(updatedProject).catch(() => {});
  };

  const handleUpdateGravity = async (gravity: [number, number, number]) => {
    if (!project) return;
    const updatedProject = {
      ...project,
      settings: { ...project.settings, gravity },
    };
    setProject(updatedProject);
    await saveProjectToBackend(updatedProject).catch(() => {});
  };

  const handleClearLogs = async () => {
    await clearServerLogs();
    setLogs([]);
  };

  if (!project) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)', color: 'var(--text-secondary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 28, height: 28, border: '3px solid var(--border-strong)', borderTopColor: 'var(--accent-blue)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
          <span>Connecting to RoboForge Simulation Backend...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rf-app-container">
      {/* Toast Notification */}
      {toastMessage && (
        <div
          style={{
            position: 'fixed',
            bottom: 200,
            right: 20,
            backgroundColor: 'rgba(30, 32, 38, 0.95)',
            border: '1px solid var(--accent-blue)',
            color: 'var(--text-primary)',
            padding: '8px 14px',
            borderRadius: 6,
            fontSize: 12,
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          }}
        >
          {toastMessage}
        </div>
      )}

      {/* Top Application Toolbar */}
      <Toolbar
        project={project}
        simState={simState}
        isConnected={isConnected}
        onPlay={handlePlay}
        onPause={handlePause}
        onStep={handleStep}
        onReset={handleReset}
        onChangeSpeed={handleChangeSpeed}
        onChangeTimestep={handleChangeTimestep}
        onNewProject={handleNewProject}
        onSaveProject={handleSaveProject}
        onLoadProject={handleLoadProject}
        onResetToSaved={handleResetToSaved}
        onTrainRobot={handleGlobalTrainRobot}
        onUndo={handleUndo}
        onRedo={handleRedo}
        canUndo={undoStack.length > 0}
        canRedo={redoStack.length > 0}
        onDuplicateProject={handleDuplicateProject}
        onExportScene={handleExportScene}
        onOpenURDFModal={() => setShowURDFModal(true)}
        isRecordingTrajectory={isRecordingTrajectory}
        onToggleRecordTrajectory={handleToggleTrajectoryRecording}
        activeSection={workspaceSection}
        onSelectSection={handleSelectWorkspaceSection}
        onOpenAssetLibrary={() => setShowAssetLibrary(true)}
        onOpenRobotCreator={() => setShowRobotCreator(true)}
        onOpenEnvironmentCreator={() => setShowEnvironmentCreator(true)}
        onOpenRobotStudio={() => handleOpenRobotStudio()}
      />

      {/* Main Workspace Layout */}
      <div className="rf-main-content">
        {/* Left Priority Panel: Asset Library when active, otherwise Scene Hierarchy */}
        <div className={`rf-sidebar-container ${!leftSidebarOpen ? 'collapsed' : ''}`} style={{ width: (workspaceSection === 'assets' || showAssetLibrary) ? 380 : undefined }}>
          {(workspaceSection === 'assets' || showAssetLibrary) && project ? (
            <AssetLibraryPanel
              isDocked={true}
              project={project}
              onAssetApplied={handleAssetApplied}
              onOpenRobotCreator={() => {
                setShowRobotCreator(true);
              }}
              onOpenEnvironmentCreator={() => {
                setShowEnvironmentCreator(true);
              }}
              onOpenRobotStudio={(model) => {
                handleOpenRobotStudio(model);
              }}
              onShowToast={showToast}
              onClose={() => {
                setShowAssetLibrary(false);
                setWorkspaceSection('scene');
              }}
            />
          ) : leftSidebarOpen && (
            <ScenePanel
              project={project}
              simState={simState}
              selectedObjectId={selectedObjectId}
              onSelectObject={setSelectedObjectId}
              onAddObstacle={handleAddObstacle}
              onToggleVisibility={handleToggleVisibility}
              onDeleteObject={handleDeleteObject}
              onDuplicateObject={handleDuplicateObject}
              onOpenURDFModal={() => setShowURDFModal(true)}
              onClearGenericRobot={handleClearGenericRobot}
            />
          )}
          <button
            className="rf-sidebar-toggle-btn rf-sidebar-toggle-left"
            onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
            title={leftSidebarOpen ? 'Collapse Left Panel (give 3D viewport more room)' : 'Expand Left Panel'}
          >
            {leftSidebarOpen ? '◀' : '▶'}
          </button>
        </div>

        {/* Central 3D Viewport: ALWAYS VISIBLE AND ALIVE */}
        <div className="rf-viewport-wrapper">
          <Viewport3D
            project={project}
            simState={simState}
            selectedObjectId={selectedObjectId}
            onSelectObject={setSelectedObjectId}
            activeKeys={activeKeys}
            onManualTeleop={handleManualTeleop}
            targetPosition={targetPosition}
            trajectoryTrail={trajectoryTrail}
            onUpdateObject={handleUpdateObject}
            onUpdateTarget={setTargetPosition}
            autoDriveEnabled={autoDriveEnabled}
            onToggleAutoDrive={handleToggleAutoDrive}
          />
        </div>

        {/* Right Priority Panel: Robot Studio Workbench when active, otherwise Object/Robot Inspector */}
        <div className={`rf-sidebar-container ${!rightSidebarOpen ? 'collapsed' : ''}`} style={{ width: (showRobotStudio && project) ? studioDockWidth : undefined }}>
          <button
            className="rf-sidebar-toggle-btn rf-sidebar-toggle-right"
            onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
            title={rightSidebarOpen ? 'Collapse Right Panel (give 3D viewport more room)' : 'Expand Right Panel'}
          >
            {rightSidebarOpen ? '▶' : '◀'}
          </button>
          {showRobotStudio && project ? (
            <RobotStudioModal
              project={project}
              simState={simState}
              initialRobot={studioInitialRobot}
              targetPosition={targetPosition}
              onClose={() => {
                setShowRobotStudio(false);
                if (workspaceSection === 'robots') setWorkspaceSection('scene');
              }}
              onApplyToScene={handleApplyRobotFromStudio}
              onShowToast={showToast}
              isDocked={true}
              dockWidth={studioDockWidth}
              onSetDockWidth={setStudioDockWidth}
            />
          ) : rightSidebarOpen && (
            <PropertiesPanel
              project={project}
              simState={simState}
              selectedObjectId={selectedObjectId}
              onUpdateObject={handleUpdateObject}
              onDeleteObject={handleDeleteObject}
              onUpdateRobotLimits={handleUpdateRobotLimits}
              onUpdateGravity={handleUpdateGravity}
              targetPosition={targetPosition}
              onUpdateTarget={setTargetPosition}
              onSelectObject={setSelectedObjectId}
              onAddObstacle={handleAddObstacle}
              onTrainRobot={handleGlobalTrainRobot}
              onUpdateSensors={handleUpdateSensors}
              onUpdateBoundaries={handleUpdateBoundaries}
              onJointControl={handleJointControl}
              onClearGenericRobot={handleClearGenericRobot}
              onOpenRobotStudio={() => handleOpenRobotStudio()}
            />
          )}
        </div>
      </div>

      {/* Collapsible Bottom Panel (Telemetry Plot + Console + Experiments) */}
      <BottomPanel
        simState={simState}
        logs={logs}
        onClearLogs={handleClearLogs}
        currentProject={project}
        onShowToast={showToast}
        onProjectLoaded={async () => {
          try {
            const p = await fetchProject();
            setProject(p);
          } catch {}
        }}
        onSelectTarget={setTargetPosition}
        onSetTrail={setTrajectoryTrail}
        activeLiveRun={latestLiveRun}
        targetPosition={targetPosition}
        recordedTrajectory={recordedTrajectory}
        isRecordingTrajectory={isRecordingTrajectory}
        onToggleTrajectoryRecording={handleToggleTrajectoryRecording}
        replayIndex={replayIndex}
        onSeekReplay={handleSeekReplay}
        forcedTab={workspaceSection === 'experiments' ? 'experiments' : workspaceSection === 'results' ? 'telemetry' : null}
      />

      {/* URDF Import Modal */}
      {showURDFModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.25)',
            backdropFilter: 'blur(2px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget && !isImportingURDF) {
              setShowURDFModal(false);
            }
          }}
        >
          <div
            style={{
              background: '#18191f',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              width: 640,
              maxWidth: '92vw',
              maxHeight: '85vh',
              overflowY: 'auto',
              boxShadow: '0 20px 45px rgba(0, 0, 0, 0.8)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 18px',
                borderBottom: '1px solid var(--border-subtle)',
                background: 'rgba(56, 189, 248, 0.08)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ background: 'rgba(56, 189, 248, 0.2)', padding: 6, borderRadius: 6 }}>
                  <FileCode size={18} color="var(--accent-blue)" />
                </div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                    URDF Robot Model Importer
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Authoritative MuJoCo link, joint, mass & collision kinematics
                  </div>
                </div>
              </div>

              {!isImportingURDF && (
                <button
                  className="rf-btn"
                  onClick={() => setShowURDFModal(false)}
                  style={{ padding: 4 }}
                >
                  <X size={16} />
                </button>
              )}
            </div>

            {/* Modal Body */}
            <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  Paste standard URDF XML definition or load sample:
                </span>
                <button
                  className="rf-btn"
                  style={{ fontSize: 11, padding: '2px 8px' }}
                  onClick={() => setUrdfInputText(SAMPLE_URDF_TEXT)}
                >
                  Load 2-DOF Sample Arm
                </button>
              </div>

              <textarea
                value={urdfInputText}
                onChange={(e) => setUrdfInputText(e.target.value)}
                placeholder="<robot name='...'>...</robot>"
                style={{
                  width: '100%',
                  height: 220,
                  backgroundColor: 'var(--bg-primary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  padding: 10,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  resize: 'vertical',
                }}
              />

              {/* Import Result / Warnings Banner */}
              {urdfImportResult && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      background: 'rgba(34, 197, 94, 0.1)',
                      border: '1px solid rgba(34, 197, 94, 0.3)',
                      padding: 10,
                      borderRadius: 6,
                      fontSize: 11,
                      color: 'var(--accent-green)',
                    }}
                  >
                    <CheckCircle2 size={16} />
                    <span>
                      Successfully parsed robot '{urdfImportResult.robot?.name || 'robot'}': {urdfImportResult.total_links} links, {urdfImportResult.total_joints} joints.
                    </span>
                  </div>

                  {urdfImportResult.warnings.length > 0 && (
                    <div
                      style={{
                        background: 'rgba(234, 179, 8, 0.1)',
                        border: '1px solid rgba(234, 179, 8, 0.3)',
                        padding: 10,
                        borderRadius: 6,
                        fontSize: 11,
                        color: 'var(--accent-yellow)',
                      }}
                    >
                      <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                        <AlertTriangle size={14} />
                        <span>Unsupported constructs reported:</span>
                      </div>
                      <ul style={{ margin: 0, paddingLeft: 18, fontSize: 10 }}>
                        {urdfImportResult.warnings.map((w, idx) => (
                          <li key={idx}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Actions */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 4 }}>
                <button
                  className="rf-btn"
                  onClick={() => setShowURDFModal(false)}
                >
                  Cancel
                </button>
                <button
                  className="rf-btn rf-btn-primary"
                  onClick={handleImportURDFSubmit}
                  disabled={isImportingURDF || !urdfInputText.trim()}
                  style={{ gap: 6 }}
                >
                  <Upload size={13} />
                  <span>{isImportingURDF ? 'Parsing & Compiling...' : 'Import into MuJoCo'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Global Robot Training & Policy Optimizer Modal */}
      {showGlobalTrainingModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.25)',
            backdropFilter: 'blur(2px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget && !isTraining) {
              setShowGlobalTrainingModal(false);
            }
          }}
        >
          <div
            style={{
              background: '#18191f',
              border: '1px solid rgba(168, 85, 247, 0.4)',
              borderRadius: 8,
              width: 580,
              maxWidth: '92vw',
              maxHeight: '85vh',
              overflowY: 'auto',
              boxShadow: '0 20px 45px rgba(0, 0, 0, 0.8)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 18px',
                borderBottom: '1px solid var(--border-subtle)',
                background: 'rgba(147, 51, 234, 0.08)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ background: 'rgba(168, 85, 247, 0.2)', padding: 6, borderRadius: 6 }}>
                  <BrainCircuit size={18} color="#c084fc" />
                </div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Robot AI Policy Optimizer & Trainer
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    MuJoCo Physical Evaluation & Collision Minimization
                  </div>
                </div>
              </div>

              {!isTraining && (
                <button
                  className="rf-btn"
                  onClick={() => setShowGlobalTrainingModal(false)}
                  style={{ padding: 4 }}
                >
                  <X size={16} />
                </button>
              )}
            </div>

            {/* Modal Body */}
            <div style={{ padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
              {isTraining ? (
                <div style={{ padding: '36px 20px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      border: '3px solid rgba(168, 85, 247, 0.2)',
                      borderTopColor: '#c084fc',
                      borderRadius: '50%',
                      animation: 'spin 0.8s linear infinite',
                    }}
                  />
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#e9d5ff' }}>
                    Simulating Multi-Candidate Physical Steering Policies...
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 420 }}>
                    Evaluating differential-drive kinematic profiles, tangential vortex repulsive fields, and safety envelopes across MuJoCo physics steps.
                  </div>
                </div>
              ) : trainingResult ? (
                <>
                  {/* Results Banner */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 10,
                      background: 'rgba(34, 197, 94, 0.1)',
                      border: '1px solid rgba(34, 197, 94, 0.3)',
                      padding: 12,
                      borderRadius: 6,
                    }}
                  >
                    <CheckCircle2 size={18} color="var(--accent-green)" style={{ marginTop: 2, flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent-green)' }}>
                        Policy Optimization Complete
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
                        {trainingResult.message}
                      </div>
                    </div>
                  </div>

                  {/* Collision Comparison Scorecard */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                    <div style={{ background: 'var(--bg-panel)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Baseline Collisions</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: trainingResult.baseline_collisions > 0 ? 'var(--accent-orange)' : 'var(--text-primary)', marginTop: 4 }}>
                        {trainingResult.baseline_collisions}
                      </div>
                    </div>
                    <div style={{ background: 'var(--bg-panel)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Trained Collisions</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent-green)', marginTop: 4 }}>
                        {trainingResult.trained_collisions}
                      </div>
                    </div>
                    <div style={{ background: 'var(--bg-panel)', padding: 12, borderRadius: 6, border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Policy Status</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: trainingResult.trained_collisions === 0 ? 'var(--accent-green)' : 'var(--accent-blue)', marginTop: 6 }}>
                        {trainingResult.trained_collisions === 0 ? 'Zero Collision' : trainingResult.improved ? 'Collision Reduced' : 'Optimal Policy'}
                      </div>
                    </div>
                  </div>

                  {/* Best Policy Parameters Details */}
                  <div style={{ background: 'var(--bg-panel)', borderRadius: 6, border: '1px solid var(--border-subtle)', padding: 12 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Sparkles size={12} color="#c084fc" />
                      <span>Optimized Steering & Avoidance Parameters</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Safety Margin:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{trainingResult.best_policy.safety_margin}m</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Repulsion Gain:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{trainingResult.best_policy.repulsion_gain}x</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Swerve / Tangential:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{trainingResult.best_policy.tangential_gain}x</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Max Turn Rate:</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{trainingResult.best_policy.max_turn_rate} rad/s</span>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 4 }}>
                    <button
                      className="rf-btn"
                      onClick={() => handleGlobalTrainRobot()}
                      style={{ gap: 6 }}
                    >
                      <BrainCircuit size={13} />
                      <span>Re-train</span>
                    </button>
                    <button
                      className="rf-btn rf-btn-primary"
                      onClick={handleDeployAndRun3D}
                      style={{
                        background: 'linear-gradient(135deg, #7928ca 0%, #4c1d95 100%)',
                        border: '1px solid #9333ea',
                        color: '#fff',
                        fontWeight: 600,
                        gap: 6,
                        padding: '6px 14px',
                      }}
                    >
                      <Play size={13} />
                      <span>Deploy & Run in 3D Viewport</span>
                    </button>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {/* Robot Creator Modal */}
      {showRobotCreator && (
        <RobotCreatorModal
          onClose={() => setShowRobotCreator(false)}
          onAddToScene={handleRobotCreated}
          onShowToast={(msg) => showToast(msg)}
        />
      )}

      {/* Environment Creator Modal */}
      {showEnvironmentCreator && (
        <EnvironmentCreatorModal
          onClose={() => setShowEnvironmentCreator(false)}
          onApplyEnvironment={handleApplyEnvironment}
          onShowToast={(msg) => showToast(msg)}
        />
      )}
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
};
