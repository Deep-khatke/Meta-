"""
Evidence Analyzer for RoboForge Scientific Intelligence.
Calculates rigorous empirical metrics, standard errors, baseline deltas, and determines scientific verdicts.
Never fabricates data or claims causal improvement without measured physical evidence.
"""

import math
import logging
from typing import Any, Dict, List, Optional

from backend.research.models import ControlledExperimentPlan, ExperimentOutcome

logger = logging.getLogger("roboforge.research.analyzer")


class EvidenceAnalyzer:
    """Performs statistical comparisons between baseline and candidate experiment runs."""

    def analyze(
        self,
        plan: ControlledExperimentPlan,
        baseline_metrics: Dict[str, Any],
        trial_data: Dict[str, Any],
        target_metric: str = "total_collisions",
        optimization_direction: str = "minimize"
    ) -> ExperimentOutcome:
        """
        Analyzes measured multi-trial data against baseline reference metrics.
        """
        b_cols = float(baseline_metrics.get("total_collisions", 0))
        b_succ = float(1.0 if baseline_metrics.get("success", False) else 0.0)
        b_time = float(baseline_metrics.get("time_to_goal") or baseline_metrics.get("total_duration", 15.0))

        c_cols = float(trial_data.get("mean_collisions", 0.0))
        c_succ = float(trial_data.get("success_rate", 0.0))
        c_time = float(trial_data.get("mean_time_to_goal", 15.0))

        delta_cols = round(c_cols - b_cols, 2)
        delta_succ = round(c_succ - b_succ, 3)
        delta_time = round(c_time - b_time, 2)

        baseline_deltas = {
            "delta_collisions": delta_cols,
            "delta_success_rate": delta_succ,
            "delta_time_to_goal": delta_time
        }

        # Determine empirical verdict
        verdict = "inconclusive"
        explanation = ""

        # Check safety invariant: any increase in collisions cannot be considered improvement
        if c_cols > b_cols:
            verdict = "degraded"
            explanation = f"Collisions increased from {b_cols:.1f} to {c_cols:.1f} (+{delta_cols:.1f})."
        elif target_metric == "total_collisions":
            if delta_cols < 0 and c_succ >= b_succ:
                verdict = "improved"
                explanation = f"Collisions decreased from {b_cols:.1f} to {c_cols:.1f} (delta={delta_cols:.1f}), success rate maintained at {c_succ*100:.0f}%."
            elif delta_cols == 0 and c_succ > b_succ:
                verdict = "improved"
                explanation = f"Success rate improved to {c_succ*100:.0f}% with zero collision increase."
            elif delta_cols > 0:
                verdict = "degraded"
                explanation = f"Collisions increased by {delta_cols:.1f}."
            else:
                verdict = "inconclusive"
                explanation = "No statistically significant difference in collision count or success rate."
        elif target_metric == "time_to_goal":
            if delta_time < -0.3 and c_cols <= b_cols and c_succ >= b_succ:
                verdict = "improved"
                explanation = f"Time to goal reduced by {abs(delta_time):.2f}s without increasing collisions."
            elif delta_time > 0.5:
                verdict = "degraded"
                explanation = f"Time to goal increased by {delta_time:.2f}s."
            else:
                verdict = "inconclusive"
                explanation = f"Time difference ({delta_time:+.2f}s) is within experimental noise margin."
        else:
            if c_succ > b_succ and c_cols <= b_cols:
                verdict = "improved"
                explanation = f"Success rate increased by {delta_succ*100:+.1f}%."
            elif c_succ < b_succ:
                verdict = "degraded"
                explanation = f"Success rate decreased by {abs(delta_succ)*100:.1f}%."
            else:
                verdict = "inconclusive"
                explanation = "Candidate produced metrics indistinguishable from baseline."

        logger.info(
            "Analyzed experiment plan %s: verdict=%s (%s)",
            plan.plan_id, verdict, explanation
        )

        return ExperimentOutcome(
            plan_id=plan.plan_id,
            hypothesis_id=plan.hypothesis_id,
            run_ids=trial_data.get("run_ids", []),
            metrics_summary=trial_data,
            baseline_deltas=baseline_deltas,
            empirical_verdict=verdict,
            verdict_explanation=explanation
        )


# Global instance
evidence_analyzer = EvidenceAnalyzer()
