"""Pydantic schemas and metadata contracts for Reinforcement Learning in RoboForge."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import time


class RLHyperparameters(BaseModel):
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


class RLTrainingConfig(BaseModel):
    job_id: Optional[str] = None
    model_name: str = "PPO-DiffDrive-Agent"
    total_timesteps: int = 10000
    seed: int = 42
    randomize_scenarios: bool = True
    target_position: List[float] = Field(default_factory=lambda: [2.5, 0.5, 0.25])
    scenario_id: Optional[str] = None
    hyperparameters: RLHyperparameters = Field(default_factory=RLHyperparameters)
    custom_reward_code: Optional[str] = None
    max_episode_steps: int = 1000


class RLTrainingMetrics(BaseModel):
    timestep: int = 0
    total_timesteps: int = 10000
    percent_complete: float = 0.0
    episodes: int = 0
    mean_reward: float = 0.0
    recent_rewards: List[float] = Field(default_factory=list)
    success_rate: float = 0.0
    collision_rate: float = 0.0
    elapsed_seconds: float = 0.0
    fps: float = 0.0


class RLTrainingStatus(BaseModel):
    job_id: str
    model_name: str
    status: str = "queued"  # "queued" | "training" | "completed" | "cancelled" | "failed"
    metrics: RLTrainingMetrics = Field(default_factory=RLTrainingMetrics)
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class RLModelMetadata(BaseModel):
    model_id: str
    name: str
    checkpoint_path: str
    timesteps_trained: int
    training_duration_seconds: float
    success_rate: float = 0.0
    collision_rate: float = 0.0
    mean_reward: float = 0.0
    created_at: float = Field(default_factory=time.time)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    scenario_id: Optional[str] = None


class RLEvaluationRequest(BaseModel):
    model_id: str
    eval_episodes: int = 5
    scenario_ids: Optional[List[str]] = None
    seed: int = 101


class PolicyBenchmarkComparison(BaseModel):
    policy_type: str  # "PPO", "APF", "Baseline"
    name: str
    model_id: Optional[str] = None
    episodes: int
    success_rate: float
    collision_rate: float
    avg_episode_time: float
    avg_path_length: float
    avg_reward: float
    confidence_interval_95: Dict[str, float]
    training_cost_timesteps: Optional[int] = None
    training_cost_seconds: Optional[float] = None
    summary: str


class EvaluationSuiteResult(BaseModel):
    eval_id: str
    timestamp: float = Field(default_factory=time.time)
    model_id: str
    benchmark_results: List[PolicyBenchmarkComparison]
    scenario_ids: List[str]
    verdict: str


class AIAssistantPromptRequest(BaseModel):
    prompt: str
    base_scenario_id: Optional[str] = None


class AIAssistantConfigResponse(BaseModel):
    interpreted_goal: str
    suggested_configuration: Dict[str, Any]
    validation_status: str  # "valid" | "warning" | "invalid"
    validation_messages: List[str]
    safety_summary: str
    suggested_actions: List[str]
    assistant_reply: str = ""
    provider: str = "deterministic_rule_engine"
    is_model_inference: bool = False
    is_fallback: bool = True
    active_simulation_context: Optional[Dict[str, Any]] = None
    tools_invoked: List[str] = Field(default_factory=list)
    structured_mission: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentThoughtStep(BaseModel):
    step: str
    title: str
    detail: str
    status: Literal["success", "running", "warning", "info"] = "success"


class AIChatRequest(BaseModel):
    prompt: str
    mode: Literal["supervised", "autonomous"] = "supervised"
    session_id: Optional[str] = "default_session"
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    mission_id: Optional[str] = None
    selected_scenario_id: Optional[str] = None
    max_iterations: int = 5
    max_experiments: int = 8


class AIChatResponse(BaseModel):
    reply: str
    provider: str
    is_model_inference: bool
    is_fallback: bool
    mode: str
    thought_steps: List[AgentThoughtStep] = Field(default_factory=list)
    observed_facts: Dict[str, Any] = Field(default_factory=dict)
    tools_invoked: List[str] = Field(default_factory=list)
    mission: Optional[Dict[str, Any]] = None
    awaiting_approval: bool = False
    approval_plan: Optional[Dict[str, Any]] = None
    suggested_configuration: Optional[Dict[str, Any]] = None
    actions_executed: List[Dict[str, Any]] = Field(default_factory=list)
    thought_process: Optional[str] = None
    error_message: Optional[str] = None


class AIApprovalRequest(BaseModel):
    mission_id: str
    approved: bool
    reason: Optional[str] = None


class HypothesisRefinementRequest(BaseModel):
    eval_id: Optional[str] = None
    scenario_id: Optional[str] = None
    focus_metric: str = "collision_rate"  # "collision_rate", "time_to_goal", "success_rate"


class HypothesisRefinementResult(BaseModel):
    hypothesis_id: str
    timestamp: float = Field(default_factory=time.time)
    target_scenario_id: str
    observed_failure: str
    hypothesis: str
    controlled_parameter: str
    baseline_value: Any
    proposed_value: Any
    baseline_run_id: str
    refinement_run_id: str
    baseline_collisions: int
    refinement_collisions: int
    baseline_success: bool
    refinement_success: bool
    baseline_time: float
    refinement_time: float
    outcome_verdict: str  # "SUPPORTED", "REFUTED", "INCONCLUSIVE"
    empirical_evidence: str


class ExecuteRLPolicyRequest(BaseModel):
    model_id: str
    scenario_id: Optional[str] = None
    target_position: Optional[List[float]] = None
    duration: float = 0.0  # 0.0 indicates continuous live execution without arbitrary timeout
    seed: int = 42
    run_until_goal: bool = True
