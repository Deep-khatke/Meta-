from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import uuid
import time

from backend.models import RobotConfig, SceneObject, SimulationSettings, SCHEMA_VERSION

class AvoidancePolicy(BaseModel):
    enabled: bool = Field(default=True, description="Enable active obstacle avoidance")
    safety_margin: float = Field(default=0.65, ge=0.1, le=3.0, description="Obstacle avoidance trigger distance in meters")
    repulsion_gain: float = Field(default=1.6, ge=0.0, le=10.0, description="Repulsive potential force multiplier")
    tangential_gain: float = Field(default=1.3, ge=0.0, le=10.0, description="Tangential vortex circulation gain to slide around obstacles")
    max_turn_rate: float = Field(default=2.5, ge=0.5, le=6.0, description="Maximum turning rate rad/s")
    slowdown_distance: float = Field(default=0.45, ge=0.1, le=2.0, description="Distance at which robot reduces speed to maneuver")

class ScriptedCommand(BaseModel):
    time: float = Field(..., ge=0.0, description="Simulation timestamp in seconds")
    linear: float = Field(default=0.0, ge=-10.0, le=10.0)
    angular: float = Field(default=0.0, ge=-15.0, le=15.0)

class TaskObjective(BaseModel):
    target_position: List[float] = Field(default=[2.5, 0.0, 0.25])
    distance_threshold: float = Field(default=0.25, gt=0.01, le=5.0, description="Tolerance distance in meters to declare success")
    max_allowed_collisions: int = Field(default=0, ge=0, description="Maximum permitted collisions before task failure")
    timeout: float = Field(default=10.0, gt=0.5, le=120.0, description="Maximum simulation time in seconds")
    control_mode: Literal["goal_seeking", "scripted"] = Field(default="goal_seeking")
    policy: AvoidancePolicy = Field(default_factory=AvoidancePolicy, description="Tunable obstacle avoidance steering policy")
    scripted_commands: Optional[List[ScriptedCommand]] = Field(default=None)

    @field_validator("target_position")
    @classmethod
    def validate_target(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("target_position must have 3 coordinates [x, y, z]")
        return v

class ExperimentConfig(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = "RoboForge simulation experiment configuration"
    seed: int = Field(default=42, ge=0, le=2147483647)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    environment: List[SceneObject] = Field(default_factory=list)
    settings: SimulationSettings = Field(default_factory=SimulationSettings)
    task: TaskObjective = Field(default_factory=TaskObjective)
    duration: float = Field(default=8.0, gt=0.5, le=120.0)
    sample_rate_hz: float = Field(default=20.0, ge=1.0, le=100.0)
    created_at: float = Field(default_factory=time.time)

class SimulatedMeasurement(BaseModel):
    """Raw, un-derived physical measurements recorded directly from MuJoCo state."""
    sim_time: float
    position: List[float]
    quaternion: List[float]
    linear_velocity: List[float]
    angular_velocity: List[float]
    linear_speed: float
    angular_speed: float
    contacts_count: int
    active_contacts: List[str] = Field(default_factory=list)

class DerivedMetrics(BaseModel):
    """Post-processed evaluation metrics computed from simulated measurements and task goals."""
    total_duration: float
    total_distance_traveled: float
    initial_distance_to_target: float
    final_distance_to_target: float
    min_distance_to_target: float
    max_linear_speed: float
    mean_linear_speed: float
    max_angular_speed: float
    total_collisions: int
    success: bool
    time_to_goal: Optional[float] = None
    status_reason: str

class ExperimentRunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    experiment_id: str
    experiment_name: str
    config_version: str = SCHEMA_VERSION
    seed: int
    status: Literal["completed", "failed", "timeout", "running"] = "completed"
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    measurements: List[SimulatedMeasurement] = Field(default_factory=list)
    metrics: DerivedMetrics
    errors: List[str] = Field(default_factory=list)

class RunComparison(BaseModel):
    run_a_id: str
    run_b_id: str
    run_a_name: str
    run_b_name: str
    run_a_seed: int
    run_b_seed: int
    metric_deltas: Dict[str, float]
    run_a_metrics: DerivedMetrics
    run_b_metrics: DerivedMetrics
    summary: str

class PolicyTrainingIteration(BaseModel):
    iteration: int
    policy: AvoidancePolicy
    success: bool
    collisions: int
    time_to_goal: Optional[float]
    final_distance: float
    score: float

class PolicyTrainingResult(BaseModel):
    scenario_id: str
    iterations: List[PolicyTrainingIteration]
    best_policy: AvoidancePolicy
    baseline_collisions: int
    trained_collisions: int
    improved: bool
    message: str
