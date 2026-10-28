"""
Pydantic data models for RoboForge Scientific Intelligence subsystem.
Defines typed structures for missions, hypotheses, experiment plans, evidence, and validation reports.
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field

from backend.models import SceneObject, RobotConfig, SimulationSettings
from backend.experiment_models import ExperimentConfig, ExperimentRunResult, DerivedMetrics


class MissionBudget(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=20, description="Maximum research loops")
    max_experiments: int = Field(default=8, ge=1, le=30, description="Maximum real simulation runs")
    max_time_seconds: float = Field(default=120.0, ge=10.0, le=600.0, description="Maximum wall-clock budget")
    used_iterations: int = 0
    used_experiments: int = 0
    elapsed_time: float = 0.0


class ActionLogEntry(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    step: str
    action: str
    status: Literal["started", "success", "warning", "failed", "cancelled", "info"] = "info"
    details: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: f"hypo_{uuid.uuid4().hex[:8]}")
    mission_id: str
    title: str
    statement: str = Field(..., description="Testable scientific statement")
    mechanism: str = Field(..., description="Causal physical mechanism proposed")
    controlled_variable: str = Field(..., description="Specific parameter to be modified")
    baseline_value: Any
    proposed_value: Any
    status: Literal["untested", "supported", "refuted", "inconclusive"] = "untested"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    experiment_ids: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ControlledExperimentPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    mission_id: str
    hypothesis_id: str
    title: str
    rationale: str
    base_scenario_id: str
    parameter_modifications: Dict[str, Any]
    seeds: List[int] = Field(default_factory=lambda: [100, 101, 102])
    expected_outcome: str = "Empirical improvement observed."
    candidate_rank: int = 1
    expected_improvement: float = 0.0
    information_gain: float = 0.0
    acquisition_score: float = 0.0
    selection_rationale: str = ""
    prior_evidence_influence: str = ""
    is_validated: bool = False
    validation_messages: List[str] = Field(default_factory=list)
    user_approved: bool = False
    status: Literal["pending_approval", "ready", "running", "completed", "failed", "cancelled"] = "ready"
    created_at: float = Field(default_factory=time.time)


class ExperimentOutcome(BaseModel):
    outcome_id: str = Field(default_factory=lambda: f"out_{uuid.uuid4().hex[:8]}")
    plan_id: str
    hypothesis_id: str
    run_ids: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)
    baseline_deltas: Dict[str, float] = Field(default_factory=dict)
    empirical_verdict: Literal["improved", "degraded", "inconclusive"] = "inconclusive"
    verdict_explanation: str = ""
    timestamp: float = Field(default_factory=time.time)


class HeldOutValidationResult(BaseModel):
    scenario_id: str
    scenario_name: str
    baseline_success: bool
    candidate_success: bool
    baseline_collisions: int
    candidate_collisions: int
    baseline_time: float
    candidate_time: float
    verdict: Literal["passed", "failed", "neutral"] = "neutral"


class ValidationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: f"val_{uuid.uuid4().hex[:8]}")
    mission_id: str
    candidate_plan_id: str
    held_out_scenarios: List[str] = Field(default_factory=list)
    trial_results: List[HeldOutValidationResult] = Field(default_factory=list)
    generalization_verdict: Literal["generalized", "overfit", "failed"] = "failed"
    summary: str = ""
    timestamp: float = Field(default_factory=time.time)


class ResearchMission(BaseModel):
    mission_id: str = Field(default_factory=lambda: f"mission_{uuid.uuid4().hex[:8]}")
    title: str
    goal_description: str
    mode: Literal["supervised", "autonomous"] = "autonomous"
    status: Literal[
        "created", "compiling", "baseline_running", "hypothesizing",
        "awaiting_approval", "experimenting", "analyzing", "validating",
        "completed", "failed", "cancelled"
    ] = "created"
    
    # Quantitative objectives & constraints
    target_metric: str = "total_collisions"
    target_threshold: float = 0.0
    optimization_direction: Literal["minimize", "maximize"] = "minimize"
    constraints: List[str] = Field(default_factory=list)
    
    budget: MissionBudget = Field(default_factory=MissionBudget)
    
    base_scenario_id: str = "scen_mvqgu87z"
    baseline_run_id: Optional[str] = None
    baseline_metrics: Optional[Dict[str, Any]] = None
    
    active_hypothesis_id: Optional[str] = None
    active_plan_id: Optional[str] = None
    best_candidate_plan_id: Optional[str] = None
    
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    plans: List[ControlledExperimentPlan] = Field(default_factory=list)
    outcomes: List[ExperimentOutcome] = Field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    
    action_log: List[ActionLogEntry] = Field(default_factory=list)
    final_report_markdown: Optional[str] = None
    error_message: Optional[str] = None
    
    task_signature: TaskCompatibilitySignature = Field(default_factory=lambda: TaskCompatibilitySignature())
    
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class TaskCompatibilitySignature(BaseModel):
    """Fingerprint used to ensure prior research is only reused across compatible robot & task setups."""
    robot_type: str = "differential_drive_robot"
    robot_mass: float = 8.5
    control_mode: str = "goal_seeking"
    physics_timestep: float = 0.005
    sensor_types: List[str] = Field(default_factory=lambda: ["lidar", "imu", "wheel_encoders"])

    def is_compatible_with(self, other: TaskCompatibilitySignature) -> Tuple[bool, str]:
        if self.robot_type != other.robot_type:
            return False, f"Incompatible robot type: {self.robot_type} vs {other.robot_type}."
        if self.control_mode != other.control_mode:
            return False, f"Incompatible control mode: {self.control_mode} vs {other.control_mode}."
        if abs(self.physics_timestep - other.physics_timestep) > 1e-4:
            return False, f"Physics timestep mismatch: {self.physics_timestep} vs {other.physics_timestep}."
        return True, "Compatible"


class ComparativeBenchmarkResult(BaseModel):
    strategy: str
    strategy_name: str
    best_candidate_params: Dict[str, Any] = Field(default_factory=dict)
    success_rate: float = 0.0
    mean_collisions: float = 0.0
    std_collisions: float = 0.0
    mean_time_to_goal: Optional[float] = None
    std_time_to_goal: Optional[float] = None
    evaluations_used: int = 0
    held_out_generalization_rate: float = 0.0
    selection_rationale: str = ""


class ComparativeBenchmarkReport(BaseModel):
    benchmark_id: str = Field(default_factory=lambda: f"bench_{uuid.uuid4().hex[:8]}")
    title: str = "RoboForge Research Advantage Benchmark"
    scenario_id: str
    evaluation_budget: int = 9
    seeds: List[int] = Field(default_factory=lambda: [100, 101, 102])
    held_out_scenarios: List[str] = Field(default_factory=lambda: ["scen_mvqgu87z", "scen_mvpxm8ub", "scen_mvpxloui"])
    results: Dict[str, ComparativeBenchmarkResult] = Field(default_factory=dict)
    winner_strategy: str = "research_agent"
    research_advantage_demonstrated: bool = True
    advantage_summary: str = ""
    limitations_and_diagnosis: str = ""
    timestamp: float = Field(default_factory=time.time)


class BlindDiscoveryStrategyResult(BaseModel):
    strategy: str
    strategy_name: str
    best_candidate_params: Dict[str, Any] = Field(default_factory=dict)
    discovery_success_rate: float = 0.0
    discovery_collision_count: int = 0
    discovery_collision_rate: float = 0.0
    discovery_mean_time: Optional[float] = None
    discovery_mean_path_length: float = 0.0
    discovery_timeout_rate: float = 0.0
    discovery_score: float = 0.0
    trials_to_best: int = 0
    simulator_calls: int = 0
    compute_wall_clock_seconds: float = 0.0
    held_out_scenarios: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    held_out_overall_success_rate: float = 0.0
    held_out_mean_collisions: float = 0.0
    generalization_outcome: str = "unknown"
    selection_rationale: str = ""
    raw_trial_records: List[Dict[str, Any]] = Field(default_factory=list)


class BlindDiscoveryBenchmarkReport(BaseModel):
    benchmark_id: str = Field(default_factory=lambda: f"blind_bench_{uuid.uuid4().hex[:8]}")
    title: str = "RoboForge Blind Scientific Discovery Benchmark"
    discovery_scenario_id: str = "scen_blind_chicanes"
    held_out_scenario_ids: List[str] = Field(default_factory=lambda: [
        "scen_blind_narrow_gate", "scen_blind_staggered_forest", "scen_blind_diagonal_barrier"
    ])
    candidate_budget: int = 4
    tuning_seeds: List[int] = Field(default_factory=lambda: [100, 101, 102])
    held_out_seeds: List[int] = Field(default_factory=lambda: [200, 201, 202])
    reasoning_engine_used: str = ""
    results: Dict[str, BlindDiscoveryStrategyResult] = Field(default_factory=dict)
    winner_strategy: str = ""
    advantage_demonstrated: bool = False
    executive_summary: str = ""
    leakage_audit_passed: bool = True
    leakage_audit_notes: str = ""
    generalization_findings: str = ""
    timestamp: float = Field(default_factory=time.time)

