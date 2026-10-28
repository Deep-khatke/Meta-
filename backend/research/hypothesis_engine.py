"""
Hypothesis Engine for RoboForge Scientific Intelligence.
Formulates and maintains competing, testable physical explanations for observed simulation behavior,
attaching empirical evidence and maintaining scientific integrity.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from backend.research.models import ResearchMission, Hypothesis, ExperimentOutcome, ActionLogEntry
from backend.research.reasoning_provider import get_reasoning_provider
from backend.research.research_memory import research_memory

logger = logging.getLogger("roboforge.research.hypothesis")


class HypothesisEngine:
    """Manages competing hypotheses, causal mechanisms, and evidence updates."""

    def __init__(self, memory=None):
        self.memory = memory or research_memory

    def formulate_competing_hypotheses(
        self,
        mission: ResearchMission,
        baseline_metrics: Dict[str, Any],
        scenario_info: Optional[Dict[str, Any]] = None
    ) -> List[Hypothesis]:
        """
        Generates competing testable physical hypotheses for baseline behavior.
        """
        provider = get_reasoning_provider()
        candidates = provider.propose_hypotheses(
            goal=mission.goal_description,
            baseline_metrics=baseline_metrics,
            scenario_info=scenario_info or {}
        )

        hypotheses: List[Hypothesis] = []
        for c in candidates:
            hypo = Hypothesis(
                mission_id=mission.mission_id,
                title=c["title"],
                statement=c["statement"],
                mechanism=c["mechanism"],
                controlled_variable=c["controlled_variable"],
                baseline_value=c["baseline_value"],
                proposed_value=c["proposed_value"],
                status="untested",
                confidence=0.5
            )
            self.memory.save_hypothesis(hypo)
            hypotheses.append(hypo)

        mission.hypotheses.extend(hypotheses)
        mission.action_log.append(ActionLogEntry(
            step="hypothesis_formulation",
            action="formulate_competing_hypotheses",
            status="success",
            details={
                "hypotheses_count": len(hypotheses),
                "titles": [h.title for h in hypotheses],
                "reasoning_engine": provider.name
            },
            message=f"Formulated {len(hypotheses)} competing testable hypotheses grounded in physical dynamics."
        ))
        self.memory.save_mission(mission)
        return hypotheses

    def select_next_hypothesis(self, mission: ResearchMission) -> Optional[Hypothesis]:
        """Selects the highest-priority untested hypothesis for investigation."""
        untested = [h for h in mission.hypotheses if h.status == "untested"]
        if not untested:
            return None
        # Sort by initial confidence
        untested.sort(key=lambda h: h.confidence, reverse=True)
        return untested[0]

    def update_evidence(
        self,
        hypothesis_id: str,
        outcome: ExperimentOutcome,
        delta_metric: float,
        optimization_direction: str = "minimize"
    ) -> Hypothesis:
        """
        Updates a hypothesis with measured empirical evidence and adjusts confidence truthfully.
        """
        hypo = self.memory.get_hypothesis(hypothesis_id)
        if not hypo:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found.")

        hypo.experiment_ids.append(outcome.plan_id)
        hypo.updated_at = time.time()

        is_improvement = (delta_metric < 0.0) if optimization_direction == "minimize" else (delta_metric > 0.0)
        is_neutral = abs(delta_metric) < 1e-4

        evidence_str = (
            f"Experiment {outcome.plan_id}: Observed delta={delta_metric:+.3f}. "
            f"Verdict: {outcome.empirical_verdict} ({outcome.verdict_explanation})."
        )

        if is_improvement and outcome.empirical_verdict == "improved":
            hypo.status = "supported"
            hypo.supporting_evidence.append(evidence_str)
            hypo.confidence = min(0.95, round(hypo.confidence + 0.30, 2))
        elif not is_neutral and outcome.empirical_verdict == "degraded":
            hypo.status = "refuted"
            hypo.contradicting_evidence.append(evidence_str)
            hypo.confidence = max(0.05, round(hypo.confidence - 0.35, 2))
        else:
            hypo.status = "inconclusive"
            hypo.contradicting_evidence.append(f"{evidence_str} (Did not produce statistically distinguishable change)")
            hypo.confidence = round(hypo.confidence * 0.9, 2)

        self.memory.save_hypothesis(hypo)
        return hypo


# Global instance
hypothesis_engine = HypothesisEngine()
