"""
RoboForge Milestone 3D: Blind Scientific Discovery Benchmark Scenarios.

Defines deterministic, reproducible navigation environments:
1. scen_blind_chicanes: Discovery Arena used for parameter tuning.
2. scen_blind_narrow_gate: Held-out Scenario A (Constricted gap).
3. scen_blind_staggered_forest: Held-out Scenario B (Multi-column forest).
4. scen_blind_diagonal_barrier: Held-out Scenario C (Diagonal wall with offset pass).

Evaluation scenarios and their optimal parameters are strictly kept out of the agent's
accessible research memory and prompt context to prevent answer leakage.
"""

from typing import Dict
from backend.experiment_models import (
    ExperimentConfig, SimulationSettings, TaskObjective, AvoidancePolicy
)
from backend.models import SceneObject
from backend.project_manager import get_default_project

BENCHMARK_SIMULATION_SETTINGS = SimulationSettings(
    timestep=0.005,
    speed_multiplier=1.0,
    gravity=[0.0, 0.0, -9.81]
)


def create_blind_chicanes_scenario() -> ExperimentConfig:
    """
    Discovery / Tuning Scenario: Double Chicane Pylons.
    Requires balanced safety margin and tangential vortex circulation to navigate.
    Target: [2.8, 0.4, 0.25], Timeout: 11.0s.
    """
    base_proj = get_default_project()
    return ExperimentConfig(
        id="scen_blind_chicanes",
        name="Blind Discovery: Double Chicane",
        description="Double offset chicane pylons forcing curved S-maneuvers towards target beacon (2.8, 0.4).",
        seed=100,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="chicane_pylon_1",
                name="Chicane Pylon 1",
                type="cylinder",
                position=[1.1, 0.15, 0.3],
                size=[0.15, 0.3],
                color="#e06c75",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="chicane_pylon_2",
                name="Chicane Pylon 2",
                type="cylinder",
                position=[1.9, -0.20, 0.3],
                size=[0.15, 0.3],
                color="#98c379",
                is_static=True,
                mass=10.0
            )
        ],
        settings=BENCHMARK_SIMULATION_SETTINGS.model_copy(deep=True),
        task=TaskObjective(
            target_position=[2.8, 0.4, 0.25],
            distance_threshold=0.30,
            max_allowed_collisions=0,
            timeout=11.0,
            control_mode="goal_seeking",
            policy=AvoidancePolicy()  # Unbiased default APF
        ),
        duration=11.0,
        sample_rate_hz=20.0
    )


def create_blind_narrow_gate_scenario() -> ExperimentConfig:
    """
    Held-Out Scenario A: Constricted Slit / Narrow Gate.
    Unseen during parameter search; tests whether candidate squeezes through narrow clearance.
    Target: [2.6, 0.0, 0.25], Timeout: 10.0s.
    """
    base_proj = get_default_project()
    return ExperimentConfig(
        id="scen_blind_narrow_gate",
        name="Blind Held-Out: Narrow Gate",
        description="Constricted gateway at x=1.4 requiring precise clearance without repulsive stall.",
        seed=200,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="gate_pillar_left",
                name="Left Gate Post",
                type="cylinder",
                position=[1.4, 0.60, 0.3],
                size=[0.15, 0.3],
                color="#61afef",
                is_static=True,
                mass=15.0
            ),
            SceneObject(
                id="gate_pillar_right",
                name="Right Gate Post",
                type="cylinder",
                position=[1.4, -0.60, 0.3],
                size=[0.15, 0.3],
                color="#61afef",
                is_static=True,
                mass=15.0
            )
        ],
        settings=BENCHMARK_SIMULATION_SETTINGS.model_copy(deep=True),
        task=TaskObjective(
            target_position=[2.6, 0.0, 0.25],
            distance_threshold=0.30,
            max_allowed_collisions=0,
            timeout=10.0,
            control_mode="goal_seeking",
            policy=AvoidancePolicy()  # Unbiased default APF
        ),
        duration=10.0,
        sample_rate_hz=20.0
    )


def create_blind_staggered_forest_scenario() -> ExperimentConfig:
    """
    Held-Out Scenario B: Staggered Column Forest.
    Unseen during parameter search; tests multi-obstacle continuous circulation.
    Target: [2.8, 0.0, 0.25], Timeout: 15.0s.
    """
    base_proj = get_default_project()
    return ExperimentConfig(
        id="scen_blind_staggered_forest",
        name="Blind Held-Out: Staggered Forest",
        description="3 staggered columns requiring continuous multi-obstacle avoidance.",
        seed=201,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="forest_p1",
                name="Forest Column 1",
                type="cylinder",
                position=[0.9, -0.30, 0.3],
                size=[0.15, 0.3],
                color="#c678dd",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="forest_p2",
                name="Forest Column 2",
                type="cylinder",
                position=[1.5, 0.35, 0.3],
                size=[0.15, 0.3],
                color="#c678dd",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="forest_p3",
                name="Forest Column 3",
                type="cylinder",
                position=[2.1, -0.35, 0.3],
                size=[0.15, 0.3],
                color="#c678dd",
                is_static=True,
                mass=10.0
            ),
        ],
        settings=BENCHMARK_SIMULATION_SETTINGS.model_copy(deep=True),
        task=TaskObjective(
            target_position=[2.8, 0.0, 0.25],
            distance_threshold=0.30,
            max_allowed_collisions=0,
            timeout=15.0,
            control_mode="goal_seeking",
            policy=AvoidancePolicy()  # Unbiased default APF
        ),
        duration=15.0,
        sample_rate_hz=20.0
    )


def create_blind_diagonal_barrier_scenario() -> ExperimentConfig:
    """
    Held-Out Scenario C: Diagonal Barrier with offset pass.
    Unseen during parameter search; tests lateral circulation and recovery angle.
    Target: [2.6, 0.0, 0.25], Timeout: 12.0s.
    """
    base_proj = get_default_project()
    return ExperimentConfig(
        id="scen_blind_diagonal_barrier",
        name="Blind Held-Out: Diagonal Barrier",
        description="Diagonal obstacle barrier with offset pass-through.",
        seed=202,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="diag_obs_1",
                name="Diagonal Wall 1",
                type="box",
                position=[1.0, -0.4, 0.25],
                size=[0.2, 0.2, 0.25],
                color="#d19a66",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="diag_obs_2",
                name="Diagonal Wall 2",
                type="box",
                position=[1.5, 0.1, 0.25],
                size=[0.2, 0.2, 0.25],
                color="#d19a66",
                is_static=True,
                mass=10.0
            ),
        ],
        settings=BENCHMARK_SIMULATION_SETTINGS.model_copy(deep=True),
        task=TaskObjective(
            target_position=[2.6, 0.0, 0.25],
            distance_threshold=0.30,
            max_allowed_collisions=0,
            timeout=12.0,
            control_mode="goal_seeking",
            policy=AvoidancePolicy()  # Unbiased default APF
        ),
        duration=12.0,
        sample_rate_hz=20.0
    )


def get_all_blind_benchmark_scenarios() -> Dict[str, ExperimentConfig]:
    """Returns dictionary mapping scenario ID to its definitive ExperimentConfig."""
    scenarios = [
        create_blind_chicanes_scenario(),
        create_blind_narrow_gate_scenario(),
        create_blind_staggered_forest_scenario(),
        create_blind_diagonal_barrier_scenario()
    ]
    return {s.id: s for s in scenarios}
