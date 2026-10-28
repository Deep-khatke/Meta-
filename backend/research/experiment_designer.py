"""
Experiment Designer for RoboForge Scientific Intelligence.
Designs controlled experiments with isolated variable modifications, sets statistical budgets,
and strictly validates parameter sanity before simulation execution.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.research.models import (
    ResearchMission, Hypothesis, ControlledExperimentPlan, ActionLogEntry
)
from backend.research.reasoning_provider import get_reasoning_provider
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory

from backend.research.candidate_ranker import candidate_ranker, CandidateRanker

logger = logging.getLogger("roboforge.research.designer")


class ExperimentDesigner:
    """Designs and validates controlled single-variable or ablative experiment plans."""

    PHYSICAL_BOUNDS = CandidateRanker.PHYSICAL_BOUNDS

    def __init__(self, tools=None, memory=None, ranker=None):
        self.tools = tools or simulation_tools
        self.memory = memory or research_memory
        self.ranker = ranker or candidate_ranker

    def design_plan(
        self,
        mission: ResearchMission,
        hypothesis: Hypothesis
    ) -> ControlledExperimentPlan:
        """
        Synthesizes a controlled experiment plan targeting the specific hypothesis,
        ranking candidates by Expected Improvement (EI), parameter uncertainty, and history.
        """
        # 1. Evaluate and rank candidate test points using the principled ranker
        ranked_candidates = self.ranker.rank_candidates_for_hypothesis(mission, hypothesis)
        top_cand = ranked_candidates[0] if ranked_candidates else None

        if top_cand and top_cand.is_valid:
            modifications = top_cand.modifications
            exp_improvement = top_cand.expected_improvement
            info_gain = top_cand.information_gain
            acq_score = top_cand.acquisition_score
            rationale = top_cand.rationale
            prior_influence = top_cand.prior_influence
        else:
            provider = get_reasoning_provider()
            base_scen = self.tools.get_base_scenario(mission.base_scenario_id)
            base_dict = base_scen.model_dump() if base_scen else {}
            spec = provider.design_controlled_experiment(
                hypothesis=hypothesis.model_dump(),
                base_scenario=base_dict
            )
            modifications = spec.get("parameter_modifications", {
                hypothesis.controlled_variable: hypothesis.proposed_value
            })
            exp_improvement = 0.5
            info_gain = 0.5
            acq_score = 0.5
            rationale = spec.get("rationale", f"Testing {hypothesis.statement}")
            prior_influence = "Default heuristic plan"

        plan = ControlledExperimentPlan(
            mission_id=mission.mission_id,
            hypothesis_id=hypothesis.hypothesis_id,
            title=f"Experiment for: {hypothesis.title}",
            rationale=rationale,
            base_scenario_id=mission.base_scenario_id,
            parameter_modifications=modifications,
            seeds=[100, 101, 102],
            expected_outcome=hypothesis.statement,
            candidate_rank=1,
            expected_improvement=exp_improvement,
            information_gain=info_gain,
            acquisition_score=acq_score,
            selection_rationale=rationale,
            prior_evidence_influence=prior_influence,
            status="ready"
        )

        # Validate proposed parameters
        val_errors = self.validate_plan(plan)
        plan.validation_messages = val_errors
        plan.is_validated = (len(val_errors) == 0)

        if not plan.is_validated:
            plan.status = "failed"
            logger.warning("Experiment plan %s failed validation: %s", plan.plan_id, val_errors)
        elif mission.mode == "supervised":
            plan.status = "pending_approval"
            plan.user_approved = False
        else:
            plan.status = "ready"
            plan.user_approved = True

        self.memory.save_plan(plan)
        mission.plans.append(plan)
        mission.active_plan_id = plan.plan_id

        mission.action_log.append(ActionLogEntry(
            step="experiment_design",
            action="design_controlled_experiment",
            status="success" if plan.is_validated else "warning",
            details={
                "plan_id": plan.plan_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "modifications": plan.parameter_modifications,
                "seeds": plan.seeds,
                "status": plan.status
            },
            message=f"Designed experiment plan '{plan.title}' with {len(plan.seeds)} trial seeds. Validation: {'PASSED' if plan.is_validated else 'FAILED'}."
        ))
        self.memory.save_mission(mission)
        return plan

    def validate_plan(self, plan: ControlledExperimentPlan) -> List[str]:
        """Strict physical boundary and sanity verification."""
        errors: List[str] = []
        if not plan.parameter_modifications:
            errors.append("Experiment plan contains no parameter modifications.")

        for var_path, val in plan.parameter_modifications.items():
            if var_path in self.PHYSICAL_BOUNDS:
                min_v, max_v = self.PHYSICAL_BOUNDS[var_path]
                if not isinstance(val, (int, float)):
                    errors.append(f"Parameter '{var_path}' value must be numeric, got {type(val)}.")
                elif val < min_v or val > max_v:
                    errors.append(f"Parameter '{var_path}'={val} violates physical safety bounds [{min_v}, {max_v}].")
            elif not (var_path.startswith("task.") or var_path.startswith("robot.") or var_path.startswith("settings.")):
                errors.append(f"Unauthorized or unrecognized parameter modification target: '{var_path}'.")

        if not plan.seeds or len(plan.seeds) < 1:
            errors.append("Experiment requires at least one deterministic seed.")

        return errors


# Global instance
experiment_designer = ExperimentDesigner()
