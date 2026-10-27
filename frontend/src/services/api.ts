import {
  Project, SceneObject, SimulationState, LogMessage, TeleopCommand,
  ExperimentConfig, ExperimentRunResult, RunSummary, RunComparison,
  PolicyTrainingResult, RLTrainingConfig, RLTrainingStatus, RLModelMetadata,
  EvaluationSuiteResult, AIAssistantConfigResponse, HypothesisRefinementResult,
  AssetModel, ValidationResult, CreateAssetRequest, AssetCategory,
  GenericRobotModel, RobotValidationResult, FKResult, IKRequest, IKResult
} from '../types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; engine_status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to reach backend health endpoint');
  return res.json();
}

export async function fetchProject(): Promise<Project> {
  const res = await fetch(`${API_BASE}/project`);
  if (!res.ok) throw new Error('Failed to fetch project');
  return res.json();
}

export async function saveProjectToBackend(project?: Project): Promise<{ status: string; path: string }> {
  if (project) {
    await fetch(`${API_BASE}/project`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(project),
    });
  }
  const res = await fetch(`${API_BASE}/project/save`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to save project');
  return res.json();
}

export async function createNewProject(name: string = 'New Scene'): Promise<Project> {
  const res = await fetch(`${API_BASE}/project/new?name=${encodeURIComponent(name)}`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to create new project');
  return res.json();
}

export async function loadProjectFromJSON(jsonString: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/project/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_json: jsonString }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to load project' }));
    throw new Error(err.detail || 'Failed to load project');
  }
  return res.json();
}

export async function resetProjectToSaved(): Promise<Project> {
  const res = await fetch(`${API_BASE}/project/reset-to-saved`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to reset project to saved state');
  return res.json();
}

export async function controlSim(action: 'play' | 'pause' | 'step' | 'reset' | 'set_speed' | 'set_timestep', value?: number): Promise<any> {
  const res = await fetch(`${API_BASE}/simulation/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation control failed' }));
    throw new Error(err.detail || 'Simulation control failed');
  }
  return res.json();
}

export async function postTeleop(cmd: TeleopCommand): Promise<any> {
  const res = await fetch(`${API_BASE}/simulation/teleop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cmd),
  });
  return res.json();
}

export async function addSceneObject(obj: Omit<SceneObject, 'id'>): Promise<SceneObject> {
  const res = await fetch(`${API_BASE}/scene/objects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj),
  });
  if (!res.ok) throw new Error('Failed to add scene object');
  return res.json();
}

export async function updateSceneObject(obj: SceneObject): Promise<SceneObject> {
  const res = await fetch(`${API_BASE}/scene/objects/${obj.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj),
  });
  if (!res.ok) throw new Error('Failed to update scene object');
  return res.json();
}

export async function deleteSceneObject(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/scene/objects/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete scene object');
}

export async function duplicateSceneObject(id: string): Promise<SceneObject> {
  const res = await fetch(`${API_BASE}/scene/objects/${id}/duplicate`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to duplicate scene object');
  return res.json();
}

export async function duplicateProject(newName?: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/project/duplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_name: newName }),
  });
  if (!res.ok) throw new Error('Failed to duplicate project');
  return res.json();
}

export async function exportScene(): Promise<Project> {
  const res = await fetch(`${API_BASE}/scene/export`);
  if (!res.ok) throw new Error('Failed to export scene');
  return res.json();
}

export async function importScene(project: Project): Promise<any> {
  const res = await fetch(`${API_BASE}/scene/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to import scene' }));
    throw new Error(err.detail || 'Failed to import scene');
  }
  return res.json();
}

export async function importURDF(urdfXml: string, setAsActive: boolean = true): Promise<any> {
  const res = await fetch(`${API_BASE}/robot/import-urdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ urdf_xml: urdfXml, set_as_active_robot: setAsActive }),
  });
  if (!res.ok) throw new Error('Failed to import URDF');
  return res.json();
}

export async function controlRobotJoint(jointName: string, target: number): Promise<any> {
  const res = await fetch(`${API_BASE}/robot/joint-control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ joint_name: jointName, target }),
  });
  return res.json();
}

export async function controlMultipleJoints(targets: Record<string, number>): Promise<any> {
  const res = await fetch(`${API_BASE}/robot/joint-control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ targets }),
  });
  return res.json();
}

export async function validateGenericRobot(robot: GenericRobotModel): Promise<RobotValidationResult> {
  const res = await fetch(`${API_BASE}/robot/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(robot),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to validate robot model' }));
    throw new Error(err.detail || 'Failed to validate robot model');
  }
  return res.json();
}

export async function autoCalculateInertials(robot: GenericRobotModel): Promise<GenericRobotModel> {
  const res = await fetch(`${API_BASE}/robot/auto-inertials`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(robot),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to auto-calculate inertials' }));
    throw new Error(err.detail || 'Failed to auto-calculate inertials');
  }
  return res.json();
}

export async function computeFK(robot: GenericRobotModel, jointPositions: Record<string, number> = {}): Promise<FKResult> {
  const res = await fetch(`${API_BASE}/robot/kinematics/fk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ robot, joint_positions: jointPositions }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to compute Forward Kinematics' }));
    throw new Error(err.detail || 'Failed to compute Forward Kinematics');
  }
  return res.json();
}

export async function solveIK(req: IKRequest): Promise<IKResult> {
  const res = await fetch(`${API_BASE}/robot/kinematics/ik`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to solve Inverse Kinematics' }));
    throw new Error(err.detail || 'Failed to solve Inverse Kinematics');
  }
  return res.json();
}

export async function fetchRobotPresets(): Promise<{ presets: GenericRobotModel[] }> {
  const res = await fetch(`${API_BASE}/robot/presets`);
  if (!res.ok) throw new Error('Failed to fetch robot presets');
  return res.json();
}

export async function exportRobotURDF(robot: GenericRobotModel): Promise<{ urdf: string; filename: string }> {
  const res = await fetch(`${API_BASE}/robot/export-urdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(robot),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to export URDF' }));
    throw new Error(err.detail || 'Failed to export URDF');
  }
  return res.json();
}

export async function setGenericRobotModel(robot: GenericRobotModel): Promise<{ status: string; robot: GenericRobotModel }> {
  const res = await fetch(`${API_BASE}/robot/generic`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(robot),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to update generic robot in scene' }));
    throw new Error(err.detail || 'Failed to update generic robot in scene');
  }
  return res.json();
}

export async function fetchGenericRobotModel(): Promise<{ robot: GenericRobotModel | null }> {
  const res = await fetch(`${API_BASE}/robot/generic`);
  if (!res.ok) throw new Error('Failed to fetch generic robot model');
  return res.json();
}

export async function clearServerLogs(): Promise<void> {
  await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
}

// Experiment & Scenario APIs
export async function fetchScenarios(): Promise<ExperimentConfig[]> {
  const res = await fetch(`${API_BASE}/experiments/scenarios`);
  if (!res.ok) throw new Error('Failed to fetch scenarios');
  const data = await res.json();
  return data.scenarios || [];
}

export async function fetchScenario(id: string): Promise<ExperimentConfig> {
  const res = await fetch(`${API_BASE}/experiments/scenarios/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error('Failed to fetch scenario');
  return res.json();
}

export async function saveScenario(config: ExperimentConfig): Promise<{ status: string; path: string }> {
  const res = await fetch(`${API_BASE}/experiments/scenarios`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to save scenario');
  return res.json();
}

export async function duplicateScenario(id: string, newName?: string): Promise<ExperimentConfig> {
  const res = await fetch(`${API_BASE}/experiments/scenarios/${encodeURIComponent(id)}/duplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_name: newName }),
  });
  if (!res.ok) throw new Error('Failed to duplicate scenario');
  return res.json();
}

export async function deleteScenario(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/experiments/scenarios/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete scenario');
}

export async function loadScenarioToScene(id: string): Promise<{ status: string; project: Project }> {
  const res = await fetch(`${API_BASE}/experiments/scenarios/${encodeURIComponent(id)}/load-to-scene`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to load scenario to scene');
  return res.json();
}

export async function saveScenarioFromActive(data: {
  name: string;
  description?: string;
  seed?: number;
  target_position?: [number, number, number];
  distance_threshold?: number;
  timeout?: number;
}): Promise<ExperimentConfig> {
  const res = await fetch(`${API_BASE}/experiments/save-from-active`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to save active scene as scenario');
  return res.json();
}

export async function updateScenario(scenarioId: string, config: ExperimentConfig): Promise<{ status: string; scenario: ExperimentConfig }> {
  const res = await fetch(`${API_BASE}/experiments/scenarios/${encodeURIComponent(scenarioId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to update scenario');
  return res.json();
}

export async function trainAvoidancePolicy(scenarioId: string): Promise<PolicyTrainingResult> {
  const res = await fetch(`${API_BASE}/experiments/train-policy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_id: scenarioId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to train policy' }));
    throw new Error(err.detail || 'Failed to train policy');
  }
  return res.json();
}

export async function runExperiment(req: {
  scenario_id?: string;
  config?: ExperimentConfig;
  seed?: number;
  duration?: number;
}): Promise<ExperimentRunResult> {
  const res = await fetch(`${API_BASE}/experiments/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Experiment failed to run' }));
    throw new Error(err.detail || 'Experiment failed to run');
  }
  return res.json();
}

export async function runLiveExperiment(req: {
  scenario_id?: string;
  config?: ExperimentConfig;
  seed?: number;
  duration?: number;
}): Promise<{ status: string; experiment_id: string; target_position: [number, number, number] }> {
  const res = await fetch(`${API_BASE}/experiments/run-live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start live experiment' }));
    throw new Error(err.detail || 'Failed to start live experiment');
  }
  return res.json();
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/experiments/runs`);
  if (!res.ok) throw new Error('Failed to fetch runs');
  const data = await res.json();
  return data.runs || [];
}

export async function fetchRun(runId: string): Promise<ExperimentRunResult> {
  const res = await fetch(`${API_BASE}/experiments/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) throw new Error('Failed to fetch run details');
  return res.json();
}

export async function compareRuns(runAId: string, runBId: string): Promise<RunComparison> {
  const res = await fetch(`${API_BASE}/experiments/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_a_id: runAId, run_b_id: runBId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to compare runs' }));
    throw new Error(err.detail || 'Failed to compare runs');
  }
  return res.json();
}

// ==========================================
// RL & AI Assistant Services
// ==========================================

export async function startRLTraining(config: RLTrainingConfig): Promise<RLTrainingStatus> {
  const res = await fetch(`${API_BASE}/rl/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start RL training' }));
    throw new Error(err.detail || 'Failed to start RL training');
  }
  return res.json();
}

export async function getRLTrainingStatus(jobId: string): Promise<RLTrainingStatus> {
  const res = await fetch(`${API_BASE}/rl/train/${jobId}/status`);
  if (!res.ok) throw new Error('Failed to fetch RL training status');
  return res.json();
}

export async function cancelRLTraining(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/rl/train/${jobId}/cancel`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to cancel RL training');
  return res.json();
}

export async function fetchRLModels(): Promise<RLModelMetadata[]> {
  const res = await fetch(`${API_BASE}/rl/models`);
  if (!res.ok) throw new Error('Failed to fetch RL models');
  const data = await res.json();
  return data.models || [];
}

export async function evaluateRLPolicy(modelId: string, seed: number = 101): Promise<EvaluationSuiteResult> {
  const res = await fetch(`${API_BASE}/rl/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, seed }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to evaluate RL policy' }));
    throw new Error(err.detail || 'Failed to evaluate RL policy');
  }
  return res.json();
}

export async function executeRLPolicyLive(
  modelId: string,
  scenarioId?: string,
  targetPosition?: [number, number, number],
  duration: number = 14.0
): Promise<any> {
  const res = await fetch(`${API_BASE}/rl/execute-live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: modelId,
      scenario_id: scenarioId,
      target_position: targetPosition,
      duration,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to launch live RL execution' }));
    throw new Error(err.detail || 'Failed to launch live RL execution');
  }
  return res.json();
}

export async function interpretAIPrompt(prompt: string, baseScenarioId?: string): Promise<AIAssistantConfigResponse> {
  const res = await fetch(`${API_BASE}/ai/interpret-prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, base_scenario_id: baseScenarioId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'AI assistant interpretation failed' }));
    throw new Error(err.detail || 'AI assistant interpretation failed');
  }
  return res.json();
}

export async function applyAIConfig(config: any): Promise<any> {
  const res = await fetch(`${API_BASE}/ai/apply-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to apply AI configuration' }));
    throw new Error(err.detail || 'Failed to apply AI configuration');
  }
  return res.json();
}

export async function refineHypothesis(scenarioId?: string, focusMetric: string = 'collision_rate'): Promise<HypothesisRefinementResult> {
  const res = await fetch(`${API_BASE}/ai/refine-hypothesis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_id: scenarioId, focus_metric: focusMetric }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Hypothesis refinement failed' }));
    throw new Error(err.detail || 'Hypothesis refinement failed');
  }
  return res.json();
}

export async function clearGenericRobot(): Promise<Project> {
  const res = await fetch(`${API_BASE}/robot/generic`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to clear generic robot' }));
    throw new Error(err.detail || 'Failed to clear generic robot');
  }
  return res.json();
}

export async function setAutoDrive(enabled: boolean): Promise<{ success: boolean; auto_drive: boolean }> {
  const res = await fetch(`${API_BASE}/simulation/auto-drive`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to update auto drive' }));
    throw new Error(err.detail || 'Failed to update auto drive');
  }
  return res.json();
}

// -------------------------------------------------------------
// Asset Library API
// -------------------------------------------------------------
export async function fetchAssets(category?: string, search?: string, tag?: string): Promise<{ assets: AssetModel[]; total: number }> {
  const params = new URLSearchParams();
  if (category && category !== 'all') params.append('category', category);
  if (search) params.append('search', search);
  if (tag) params.append('tag', tag);

  const res = await fetch(`${API_BASE}/assets?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch asset library');
  return res.json();
}

export async function fetchAsset(assetId: string): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets/${assetId}`);
  if (!res.ok) throw new Error(`Asset '${assetId}' not found`);
  return res.json();
}

export async function createCustomAsset(req: CreateAssetRequest): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to create asset' }));
    throw new Error(err.detail || 'Failed to create asset');
  }
  return res.json();
}

export async function deleteCustomAsset(assetId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/assets/${assetId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to delete asset' }));
    throw new Error(err.detail || 'Failed to delete asset');
  }
  return res.json();
}

export async function duplicateAsset(assetId: string, newName?: string): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets/${assetId}/duplicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: newName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to duplicate asset' }));
    throw new Error(err.detail || 'Failed to duplicate asset');
  }
  return res.json();
}

export async function applyAsset(assetId: string, positionOffset?: [number, number, number]): Promise<{ status: string; asset_id: string; project: Project }> {
  const res = await fetch(`${API_BASE}/assets/${assetId}/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position_offset: positionOffset }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to apply asset to scene' }));
    throw new Error(err.detail || 'Failed to apply asset to scene');
  }
  return res.json();
}

export async function validateRobotModel(robotData: any, robotType: 'mobile' | 'articulated'): Promise<ValidationResult> {
  const res = await fetch(`${API_BASE}/assets/validate-robot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ robot: robotData, type: robotType }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Robot validation request failed' }));
    throw new Error(err.detail || 'Robot validation request failed');
  }
  return res.json();
}

export async function validateEnvironmentModel(envData: any): Promise<ValidationResult> {
  const res = await fetch(`${API_BASE}/assets/validate-environment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ environment: envData }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Environment validation request failed' }));
    throw new Error(err.detail || 'Environment validation request failed');
  }
  return res.json();
}

export async function saveCurrentRobotAsset(name: string, description: string = ''): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets/save-current-robot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save current robot as asset' }));
    throw new Error(err.detail || 'Failed to save current robot as asset');
  }
  return res.json();
}

export async function saveRobotModelAsset(name: string, description: string = '', robot: GenericRobotModel, tags: string[] = ['custom', 'articulated', 'robot-studio']): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets/save-robot-model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, robot, tags }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save robot model to Asset Library' }));
    throw new Error(err.detail || 'Failed to save robot model to Asset Library');
  }
  return res.json();
}

export async function saveCurrentSceneTemplate(name: string, description: string = ''): Promise<AssetModel> {
  const res = await fetch(`${API_BASE}/assets/save-current-scene`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to save current scene as template' }));
    throw new Error(err.detail || 'Failed to save current scene as template');
  }
  return res.json();
}

// Real-time WebSocket Client
export class SimulationSocket {
  private ws: WebSocket | null = null;
  private onStateCallback: (state: SimulationState) => void;
  private onLogCallback: (log: LogMessage) => void;
  private onConnectionChange: (connected: boolean) => void;
  private onProjectCallback?: (project: Project) => void;
  private onExperimentEventCallback?: (type: string, data: any) => void;
  private isDestroyed = false;
  private reconnectTimer: number | null = null;

  constructor(
    onState: (state: SimulationState) => void,
    onLog: (log: LogMessage) => void,
    onConnectionChange: (connected: boolean) => void,
    onProject?: (project: Project) => void,
    onExperimentEvent?: (type: string, data: any) => void
  ) {
    this.onStateCallback = onState;
    this.onLogCallback = onLog;
    this.onConnectionChange = onConnectionChange;
    this.onProjectCallback = onProject;
    this.onExperimentEventCallback = onExperimentEvent;
    this.connect();
  }

  private connect() {
    if (this.isDestroyed) return;

    if (this.ws) {
      try {
        this.ws.onopen = null;
        this.ws.onmessage = null;
        this.ws.onclose = null;
        this.ws.onerror = null;
        this.ws.close();
      } catch {}
      this.ws = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    // In dev proxy, /ws will proxy to ws://localhost:8000/ws
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.onConnectionChange(true);
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'state' && msg.data) {
            this.onStateCallback(msg.data);
          } else if (msg.type === 'log' && msg.data) {
            this.onLogCallback(msg.data);
          } else if (msg.type === 'project_updated' && msg.data) {
            if (this.onProjectCallback) this.onProjectCallback(msg.data);
          } else if (msg.type === 'live_experiment_started' || msg.type === 'live_experiment_completed') {
            if (this.onExperimentEventCallback) this.onExperimentEventCallback(msg.type, msg.data);
          } else if (msg.type === 'init') {
            if (msg.state) this.onStateCallback(msg.state);
            if (msg.project && this.onProjectCallback) this.onProjectCallback(msg.project);
            if (msg.logs && Array.isArray(msg.logs)) {
              msg.logs.forEach(this.onLogCallback);
            }
          }
        } catch (e) {
          console.error('Error parsing WebSocket message', e);
        }
      };

      this.ws.onclose = () => {
        this.onConnectionChange(false);
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.onConnectionChange(false);
      };
    } catch (e) {
      this.onConnectionChange(false);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.isDestroyed || this.reconnectTimer) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  public sendTeleop(linear: number, angular: number) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'teleop', linear, angular }));
    } else {
      // Fallback to REST
      postTeleop({ linear, angular }).catch(() => {});
    }
  }

  public sendAutoDrive(enabled: boolean) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'auto_drive', enabled }));
    } else {
      setAutoDrive(enabled).catch(() => {});
    }
  }

  public sendAction(action: 'play' | 'pause' | 'step' | 'reset') {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: action }));
    } else {
      controlSim(action).catch(() => {});
    }
  }

  public destroy() {
    this.isDestroyed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
