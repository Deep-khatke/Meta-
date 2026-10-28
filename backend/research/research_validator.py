"""
Research Validator for RoboForge Scientific Intelligence.
Independently verifies acceptance criteria and held-out generalizability, preventing benchmark overfitting.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.research.models import (
    ResearchMission, ControlledExperimentPlan, ValidationReport, HeldOutValidationResult, ActionLogEntry
)
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory
from backend.experiment_manager import experiment_manager

logger = logging.getLogger("roboforge.research.validator")


class ResearchValidator:
    """Evaluates candidate improvements against held-out benchmark scenarios to detect overfitting."""

    def __init__(self, tools=None, memory=None):
        self.tools = tools or simulation_tools
        self.memory = memory or research_memory

    def validate_candidate(
        self,
        mission: ResearchMission,
        candidate_plan: ControlledExperimentPlan,
        held_out_scenario_ids: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Executes real MuJoCo evaluation on held-out test scenarios.
        """
        logger.info(
            "Validating candidate plan '%s' on held-out scenarios for mission '%s'",
            candidate_plan.plan_id, mission.mission_id
        )

        # Select held-out scenarios distinct from base_scenario_id
        if not held_out_scenario_ids:
            all_scens = experiment_manager.list_scenarios()
            held_out_scenario_ids = [
                s.id for s in all_scens if s.id != mission.base_scenario_id
            ][:2]

        if not held_out_scenario_ids:
            logger.warning("No separate held-out scenarios found; reusing standard benchmark suite.")
            held_out_scenario_ids = [mission.base_scenario_id]

        # Evaluate candidate parameter modifications on held-out suite
        eval_raw = self.tools.evaluate_held_out(
            held_out_scenario_ids=held_out_scenario_ids,
            parameter_overrides=candidate_plan.parameter_modifications,
            seed=303
        )

        trial_results: List[HeldOutValidationResult] = []
        passed_count = 0

        for r in eval_raw:
            scen = experiment_manager.get_scenario(r["scenario_id"])
            s_name = scen.name if scen else r["scenario_id"]
            verdict = "passed" if r["passed"] else "failed"
            if r["passed"]:
                passed_count += 1

            trial_results.append(HeldOutValidationResult(
                scenario_id=r["scenario_id"],
                scenario_name=s_name,
                baseline_success=r["baseline_success"],
                candidate_success=r["candidate_success"],
                baseline_collisions=r["baseline_collisions"],
                candidate_collisions=r["candidate_collisions"],
                baseline_time=r["baseline_time"],
                candidate_time=r["candidate_time"],
                verdict=verdict
            ))

        total_trials = len(trial_results)
        pass_ratio = (passed_count / total_trials) if total_trials > 0 else 0.0

        if pass_ratio >= 0.66:
            gen_verdict = "generalized"
            summary = (
                f"Candidate successfully generalized across {passed_count}/{total_trials} held-out scenarios "
                f"without introducing collision regressions."
            )
        elif pass_ratio > 0.0:
            gen_verdict = "overfit"
            summary = (
                f"Candidate passed {passed_count}/{total_trials} held-out scenarios; improvements on the baseline "
                f"scenario did not fully transfer to novel geometry."
            )
        else:
            gen_verdict = "failed"
            summary = f"Candidate failed all {total_trials} held-out validation scenarios."

        report = ValidationReport(
            mission_id=mission.mission_id,
            candidate_plan_id=candidate_plan.plan_id,
            held_out_scenarios=held_out_scenario_ids,
            trial_results=trial_results,
            generalization_verdict=gen_verdict,
            summary=summary
        )

        mission.validation_report = report
        mission.action_log.append(ActionLogEntry(
            step="held_out_validation",
            action="evaluate_generalization",
            status="success" if gen_verdict == "generalized" else "warning",
            details={
                "held_out_scenarios": held_out_scenario_ids,
                "passed_count": passed_count,
                "total_count": total_trials,
                "verdict": gen_verdict
            },
            message=summary
        ))
        self.memory.save_mission(mission)
        return report


# Global instance
research_validator = ResearchValidator()
