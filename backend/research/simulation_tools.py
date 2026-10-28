"""
Typed simulation tools interface for RoboForge Scientific Intelligence.
Provides safe, validated execution of scenarios, parameter modifications, and multi-trial runs in the real MuJoCo simulator.
"""

from typing import Any, Dict, List, Optional
import copy
import logging

from backend.models import Project, SimulationSnapshot
from backend.experiment_models import ExperimentConfig, ExperimentRunResult
from backend.experiment_manager import experiment_manager, ExperimentManager
from backend.project_manager import ProjectManager
from backend.agent_interface import AgentSimulationInterface
from backend.scene_validator import validate_scene

logger = logging.getLogger("roboforge.research.tools")


class SimulationTools:
    """Safe, typed, validated wrapper around RoboForge physics and experiment systems."""

    def __init__(
        self,
        exp_manager: Optional[ExperimentManager] = None,
        proj_manager: Optional[ProjectManager] = None
    ):
        self.em = exp_manager or experiment_manager
        self.pm = proj_manager or ProjectManager()

    def inspect_current_environment(self) -> Dict[str, Any]:
        """Inspects current active project, robot configuration, obstacles, and boundary limits."""
        proj = self.pm.get_current()
        val = validate_scene(proj)
        return {
            "project_name": proj.name,
            "robot_name": proj.robot.name,
            "initial_position": proj.robot.initial_position,
            "max_linear_velocity": proj.robot.max_linear_velocity,
            "max_angular_velocity": proj.robot.max_angular_velocity,
            "total_obstacles": len(proj.objects),
            "obstacles": [o.model_dump() for o in proj.objects],
            "physics_settings": proj.settings.model_dump(),
            "is_valid": val.is_valid,
            "validation_errors": [e.message for e in val.errors]
        }

    def get_base_scenario(self, scenario_id: str) -> Optional[ExperimentConfig]:
        """Retrieves a baseline scenario by ID."""
        return self.em.get_scenario(scenario_id)

    def run_scenario(
        self,
        scenario_id: str,
        seed: int = 100,
        parameter_overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentRunResult:
        """
        Executes a real simulation run in MuJoCo.
        Applies validated parameter modifications if provided, without modifying base scenario permanently.
        """
        base_scen = self.em.get_scenario(scenario_id)
        if not base_scen:
            scens = self.em.list_scenarios()
            if scens:
                base_scen = scens[0]
            else:
                raise FileNotFoundError(f"Scenario '{scenario_id}' not found.")

        scen_to_run = base_scen.model_copy(deep=True)
        scen_to_run.seed = seed

        # Apply parameter modifications (e.g. "task.policy.safety_margin", "robot.max_linear_velocity")
        if parameter_overrides:
            self._apply_parameter_overrides(scen_to_run, parameter_overrides)

        logger.info(
            "Executing real MuJoCo experiment run on scenario '%s' (seed=%d, overrides=%s)",
            scen_to_run.id, seed, parameter_overrides
        )
        return self.em.run_experiment(scen_to_run, seed_override=seed)

    def run_multi_trial(
        self,
        scenario_id: str,
        seeds: List[int],
        parameter_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs repeated trials over multiple deterministic seeds to gather statistically sound metrics.
        """
        results: List[ExperimentRunResult] = []
        for s in seeds:
            res = self.run_scenario(scenario_id, seed=s, parameter_overrides=parameter_overrides)
            results.append(res)

        # Aggregate metrics
        durations = [r.metrics.total_duration for r in results]
        collisions = [r.metrics.total_collisions for r in results]
        successes = [1 if r.metrics.success else 0 for r in results]
        times_to_goal = [r.metrics.time_to_goal for r in results if r.metrics.time_to_goal is not None]

        mean_duration = float(sum(durations) / len(durations)) if durations else 0.0
        mean_collisions = float(sum(collisions) / len(collisions)) if collisions else 0.0
        success_rate = float(sum(successes) / len(successes)) if successes else 0.0
        mean_time_to_goal = float(sum(times_to_goal) / len(times_to_goal)) if times_to_goal else mean_duration

        return {
            "scenario_id": scenario_id,
            "seeds": seeds,
            "trials_count": len(results),
            "success_rate": round(success_rate, 3),
            "mean_collisions": round(mean_collisions, 2),
            "mean_duration": round(mean_duration, 3),
            "mean_time_to_goal": round(mean_time_to_goal, 3),
            "run_ids": [r.run_id for r in results],
            "runs": results
        }

    def evaluate_held_out(
        self,
        held_out_scenario_ids: List[str],
        parameter_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Evaluates candidate parameter configuration against a held-out set of scenarios to verify generalizability.
        """
        eval_reports = []
        for hid in held_out_scenario_ids:
            # 1. Run baseline on this held-out scenario (no overrides)
            base_res = self.run_scenario(hid, seed=seed, parameter_overrides=None)
            # 2. Run candidate on this held-out scenario (with overrides)
            cand_res = self.run_scenario(hid, seed=seed, parameter_overrides=parameter_overrides)

            passed = cand_res.metrics.success and (cand_res.metrics.total_collisions <= base_res.metrics.total_collisions)
            eval_reports.append({
                "scenario_id": hid,
                "baseline_run_id": base_res.run_id,
                "candidate_run_id": cand_res.run_id,
                "baseline_success": base_res.metrics.success,
                "candidate_success": cand_res.metrics.success,
                "baseline_collisions": base_res.metrics.total_collisions,
                "candidate_collisions": cand_res.metrics.total_collisions,
                "baseline_time": round(base_res.metrics.time_to_goal or base_res.metrics.total_duration, 2),
                "candidate_time": round(cand_res.metrics.time_to_goal or cand_res.metrics.total_duration, 2),
                "passed": passed
            })
        return eval_reports

    def _apply_parameter_overrides(self, config: ExperimentConfig, overrides: Dict[str, Any]):
        """Safely modifies nested attributes specified as dot-separated keys."""
        for path, val in overrides.items():
            parts = path.split(".")
            target = config
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                elif isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    raise KeyError(f"Invalid parameter path: {path}")

            last = parts[-1]
            if hasattr(target, last):
                # Type validation
                orig_type = type(getattr(target, last))
                setattr(target, last, orig_type(val) if orig_type is not type(None) else val)
            elif isinstance(target, dict):
                target[last] = val
            else:
                raise KeyError(f"Cannot set parameter: {path}")


# Global default instance
simulation_tools = SimulationTools()
