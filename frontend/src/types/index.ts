export type ObjectType = 'box' | 'sphere' | 'cylinder';

export interface SceneObject {
  id: string;
  name: string;
  type: ObjectType;
  position: [number, number, number];
  rotation: [number, number, number]; // roll, pitch, yaw in degrees
  size: number[]; // box: [hx, hy, hz], sphere: [r], cylinder: [r, hh]
  color: string;
  is_static: boolean;
  mass: number;
  friction?: [number, number, number]; // [sliding, torsional, rolling]
  visible?: boolean;
}

export interface RobotConfig {
  id: string;
  name: string;
  initial_position: [number, number, number];
  initial_yaw: number;
  chassis_size: [number, number, number];
  wheel_radius: number;
  wheel_width: number;
  wheel_base: number;
  caster_radius: number;
  chassis_mass: number;
  wheel_mass: number;
  max_linear_velocity: number;
  max_angular_velocity: number;
}

export interface LidarConfig {
  enabled: boolean;
  mount_pos: [number, number, number];
  num_beams: number;
  fov_deg: number;
  max_range: number;
  min_range: number;
}

export interface ImuConfig {
  enabled: boolean;
  mount_pos: [number, number, number];
}

export interface CameraConfig {
  enabled: boolean;
  mount_pos: [number, number, number];
  mount_rpy: [number, number, number];
  width: number;
  height: number;
  fov_deg: number;
}

export interface EncoderConfig {
  enabled: boolean;
  ticks_per_rev: number;
  noise_std: number;
}

export interface SensorSettings {
  lidar: LidarConfig;
  imu: ImuConfig;
  camera?: CameraConfig;
  encoders?: EncoderConfig;
}

export interface JointLimit {
  lower: number;
  upper: number;
  effort: number;
  velocity: number;
}

export interface JointModel {
  name: string;
  type: 'revolute' | 'continuous' | 'prismatic' | 'fixed';
  parent_link: string;
  child_link: string;
  origin_pos: [number, number, number];
  origin_rpy: [number, number, number];
  axis: [number, number, number];
  limit?: JointLimit | null;
  damping: number;
  armature: number;
  actuator_mode?: 'position' | 'velocity' | 'torque' | null;
  kp: number;
  kv: number;
}

export interface LinkGeom {
  type: 'box' | 'sphere' | 'cylinder';
  size: number[];
  origin_pos: [number, number, number];
  origin_rpy: [number, number, number];
  color: string;
}

export interface LinkModel {
  name: string;
  visual?: LinkGeom | null;
  collision?: LinkGeom | null;
  inertial?: {
    mass: number;
    origin_pos: [number, number, number];
    ixx: number;
    iyy: number;
    izz: number;
  } | null;
}

export interface GenericRobotModel {
  name: string;
  root_link: string;
  links: LinkModel[];
  joints: JointModel[];
  base_position: [number, number, number];
  base_rpy: [number, number, number];
  is_fixed_base: boolean;
}

export interface URDFImportResult {
  status: 'success' | 'warning' | 'error';
  robot?: GenericRobotModel | null;
  warnings: string[];
  errors: string[];
  total_links: number;
  total_joints: number;
}

export interface RobotDiagnostic {
  code: string;
  severity: 'error' | 'warning' | 'info';
  target: string;
  message: string;
  suggestion?: string | null;
}

export interface RobotValidationResult {
  valid: boolean;
  errors: RobotDiagnostic[];
  warnings: RobotDiagnostic[];
  summary: string;
}

export interface LinkPose {
  link_name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  rpy_deg: [number, number, number];
  parent_link?: string | null;
}

export interface JointFrame {
  joint_name: string;
  origin_position: [number, number, number];
  axis_vector: [number, number, number];
  joint_type: string;
  current_position: number;
  limits?: JointLimit | null;
}

export interface FKResult {
  success: boolean;
  end_effector_link: string;
  end_effector_position: [number, number, number];
  end_effector_quaternion: [number, number, number, number];
  end_effector_rpy_deg: [number, number, number];
  link_poses: Record<string, LinkPose>;
  joint_frames: Record<string, JointFrame>;
  total_reach: number;
}

export interface IKRequest {
  robot: GenericRobotModel;
  target_position: [number, number, number];
  target_rpy_deg?: [number, number, number] | null;
  initial_joint_positions?: Record<string, number> | null;
  tolerance?: number;
  max_iterations?: number;
  end_effector_link?: string | null;
}

export interface IKResult {
  success: boolean;
  status: 'SUCCESS' | 'TARGET_OUT_OF_REACH' | 'JOINT_LIMITS_BLOCKED' | 'CONVERGENCE_FAILED' | 'ERROR';
  iterations: number;
  error_distance: number;
  joint_solution: Record<string, number>;
  achieved_position: [number, number, number];
  achieved_rpy_deg: [number, number, number];
  message: string;
}

export interface SimulationSettings {
  timestep: number;
  speed_multiplier: number;
  gravity: [number, number, number];
  world_boundaries?: [number, number, number, number, number, number];
  enable_boundaries?: boolean;
}

export interface Project {
  schema_version: string;
  name: string;
  description?: string;
  robot: RobotConfig;
  generic_robot?: GenericRobotModel | null;
  sensors?: SensorSettings;
  objects: SceneObject[];
  settings: SimulationSettings;
}

export interface RobotState {
  position: [number, number, number];
  quaternion: [number, number, number, number]; // [w, x, y, z]
  linear_velocity: [number, number, number];
  angular_velocity: [number, number, number];
  linear_speed: number;
  angular_speed: number;
  left_wheel_velocity: number;
  right_wheel_velocity: number;
  forward_vector: [number, number, number];
}

export interface ObjectPhysicsState {
  id: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  linear_velocity?: [number, number, number];
}

export interface ContactInfo {
  geom1: string;
  geom2: string;
  position: [number, number, number];
  distance: number;
}

export type SimStatus = 'running' | 'paused' | 'stopped' | 'error';

export interface SimulationState {
  status: SimStatus;
  sim_time: number;
  wall_time: number;
  step_count: number;
  robot: RobotState;
  objects: Record<string, ObjectPhysicsState>;
  contacts: ContactInfo[];
  errors: string[];
  lidar_readings?: number[];
  lidar_angles?: number[];
  imu_linear_acceleration?: [number, number, number];
  imu_angular_velocity?: [number, number, number];
  joint_positions?: Record<string, number>;
  joint_velocities?: Record<string, number>;
  joint_torques?: Record<string, number>;
  encoder_ticks?: Record<string, number>;
}

export interface LogMessage {
  timestamp: number;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  source: string;
}

export interface TeleopCommand {
  linear: number;
  angular: number;
}

export interface AvoidancePolicy {
  enabled: boolean;
  safety_margin: number;
  repulsion_gain: number;
  tangential_gain: number;
  max_turn_rate: number;
  slowdown_distance: number;
}

export interface TaskObjective {
  target_position: [number, number, number];
  distance_threshold: number;
  max_allowed_collisions: number;
  timeout: number;
  control_mode: 'goal_seeking' | 'scripted';
  policy?: AvoidancePolicy;
  scripted_commands?: Array<{ time: number; linear: number; angular: number }>;
}

export interface ExperimentConfig {
  schema_version: string;
  id: string;
  name: string;
  description?: string;
  seed: number;
  robot: RobotConfig;
  environment: SceneObject[];
  settings: SimulationSettings;
  task: TaskObjective;
  duration: number;
  sample_rate_hz: number;
  created_at?: number;
}

export interface SimulatedMeasurement {
  sim_time: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  linear_velocity: [number, number, number];
  angular_velocity: [number, number, number];
  linear_speed: number;
  angular_speed: number;
  contacts_count: number;
  active_contacts: string[];
}

export interface DerivedMetrics {
  total_duration: number;
  total_distance_traveled: number;
  initial_distance_to_target: number;
  final_distance_to_target: number;
  min_distance_to_target: number;
  max_linear_speed: number;
  mean_linear_speed: number;
  max_angular_speed: number;
  total_collisions: number;
  success: boolean;
  time_to_goal?: number | null;
  status_reason: string;
}

export interface ExperimentRunResult {
  run_id: string;
  experiment_id: string;
  experiment_name: string;
  config_version: string;
  seed: number;
  status: 'completed' | 'failed' | 'timeout' | 'running';
  created_at: number;
  completed_at?: number;
  measurements: SimulatedMeasurement[];
  metrics: DerivedMetrics;
  errors: string[];
}

export interface RunSummary {
  run_id: string;
  experiment_id: string;
  experiment_name: string;
  seed: number;
  status: string;
  created_at: number;
  metrics: DerivedMetrics;
}

export interface RunComparison {
  run_a_id: string;
  run_b_id: string;
  run_a_name: string;
  run_b_name: string;
  run_a_seed: number;
  run_b_seed: number;
  metric_deltas: Record<string, number>;
  run_a_metrics: DerivedMetrics;
  run_b_metrics: DerivedMetrics;
  summary: string;
}

export interface PolicyTrainingIteration {
  iteration: number;
  policy: AvoidancePolicy;
  success: boolean;
  collisions: number;
  time_to_goal: number | null;
  final_distance: number;
  score: number;
}

export interface PolicyTrainingResult {
  scenario_id: string;
  iterations: PolicyTrainingIteration[];
  best_policy: AvoidancePolicy;
  baseline_collisions: number;
  trained_collisions: number;
  improved: boolean;
  message: string;
}

// ==========================================
// Reinforcement Learning & AI Assistant Types
// ==========================================

export interface RLHyperparameters {
  learning_rate: number;
  n_steps: number;
  batch_size: number;
  n_epochs: number;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  ent_coef: number;
}

export interface RLTrainingConfig {
  job_id?: string;
  model_name: string;
  total_timesteps: number;
  seed: number;
  randomize_scenarios: boolean;
  target_position: [number, number, number];
  scenario_id?: string;
  hyperparameters?: Partial<RLHyperparameters>;
}

export interface RLTrainingMetrics {
  timestep: number;
  total_timesteps: number;
  percent_complete: number;
  episodes: number;
  mean_reward: number;
  success_rate: number;
  collision_rate: number;
  elapsed_seconds: number;
  fps: number;
}

export interface RLTrainingStatus {
  job_id: string;
  model_name: string;
  status: 'queued' | 'training' | 'completed' | 'cancelled' | 'failed';
  metrics: RLTrainingMetrics;
  error?: string | null;
  checkpoint_path?: string | null;
  created_at: number;
  updated_at: number;
}

export interface RLModelMetadata {
  model_id: string;
  name: string;
  checkpoint_path: string;
  timesteps_trained: number;
  training_duration_seconds: number;
  success_rate: number;
  collision_rate: number;
  mean_reward: number;
  created_at: number;
  hyperparameters: Record<string, any>;
  scenario_id?: string;
}

export interface PolicyBenchmarkComparison {
  policy_type: 'PPO' | 'APF' | 'Baseline';
  name: string;
  model_id?: string | null;
  episodes: number;
  success_rate: number;
  collision_rate: number;
  avg_episode_time: number;
  avg_path_length: number;
  avg_reward: number;
  confidence_interval_95: {
    success_rate_margin?: number;
    collision_rate_margin?: number;
    duration_margin_sec?: number;
  };
  training_cost_timesteps?: number | null;
  training_cost_seconds?: number | null;
  summary: string;
}

export interface EvaluationSuiteResult {
  eval_id: string;
  timestamp: number;
  model_id: string;
  benchmark_results: PolicyBenchmarkComparison[];
  scenario_ids: string[];
  verdict: string;
}

export interface AIAssistantConfigResponse {
  interpreted_goal: string;
  suggested_configuration: any;
  validation_status: 'valid' | 'warning' | 'invalid';
  validation_messages: string[];
  safety_summary: string;
  suggested_actions: string[];
}

export interface HypothesisRefinementResult {
  hypothesis_id: string;
  timestamp: number;
  target_scenario_id: string;
  observed_failure: string;
  hypothesis: string;
  controlled_parameter: string;
  baseline_value: any;
  proposed_value: any;
  baseline_run_id: string;
  refinement_run_id: string;
  baseline_collisions: number;
  refinement_collisions: number;
  baseline_success: boolean;
  refinement_success: boolean;
  baseline_time: number;
  refinement_time: number;
  outcome_verdict: 'SUPPORTED' | 'REFUTED' | 'INCONCLUSIVE';
  empirical_evidence: string;
}

export type AssetCategory = 'robot' | 'obstacle' | 'sensor' | 'template';

export interface AssetModel {
  id: string;
  name: string;
  category: AssetCategory;
  description: string;
  tags: string[];
  is_built_in: boolean;
  created_at: number;
  thumbnail_hint: string;
  version: string;
  data: any;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  details?: any;
}

export interface CreateAssetRequest {
  name: string;
  category: AssetCategory;
  description?: string;
  tags?: string[];
  data: any;
  thumbnail_hint?: string;
}

export type WorkspaceSection = 'scene' | 'robots' | 'assets' | 'sensors' | 'experiments' | 'results';
