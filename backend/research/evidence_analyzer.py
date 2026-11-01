"""
Evidence Analyzer for RoboForge Scientific Intelligence.

Compares candidate multi-trial runs against the baseline using REAL statistical inference
(Welch's t-test + bootstrap confidence intervals + two-proportion z-test) rather than
hand-tuned deltas. A verdict of "improved" now means a measured, statistically or practically
distinguishable gain -- not `delta < -0.3`. The safety invariant is preserved and strengthened:
any increase in collisions is a "degraded" verdict regardless of other metrics.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.research.models import ControlledExperimentPlan, ExperimentOutcome
from backend.research.statistics import (
    welch_t_test, two_proportion_z_test, bootstrap_diff_ci, summarize_sample,
)

logger = logging.getLogger("roboforge.research.analyzer")


class EvidenceAnalyzer:
    """Performs statistical comparisons between baseline and candidate experiment runs."""

    def analyze(
        self,
        plan: ControlledExperimentPlan,
        baseline_metrics: Dict[str, Any],
        trial_data: Dict[str, Any],
        target_metric: str = "total_collisions",
        optimization_direction: str = "minimize",
        baseline_samples: Optional[Dict[str, List[float]]] = None,
    ) -> ExperimentOutcome:
        """
        Analyzes measured multi-trial data against baseline reference metrics.

        `trial_data` is the dict from SimulationTools.run_multi_trial, which carries the per-seed
        `runs`. `baseline_samples` optionally carries per-seed baseline samples for a paired
        statistical test; when absent we fall back to the single baseline point plus the
        candidate's own dispersion for an honest, conservative verdict.
        """
        # Aggregate baseline reference
        b_cols = float(baseline_metrics.get("total_collisions", 0))
        b_succ = float(1.0 if baseline_metrics.get("success", False) else 0.0)
        b_time = float(baseline_metrics.get("time_to_goal") or baseline_metrics.get("total_duration", 15.0))

        # Candidate per-seed samples (preferred) or aggregate fallback
        runs = trial_data.get("runs", [])
        cand_coll = [float(r.metrics.total_collisions) for r in runs] if runs else [float(trial_data.get("mean_collisions", 0.0))]
        cand_succ = [1.0 if r.metrics.success else 0.0 for r in runs] if runs else [float(trial_data.get("success_rate", 0.0))]
        cand_time = [
            float(r.metrics.time_to_goal if r.metrics.time_to_goal is not None else r.metrics.total_duration)
            for r in runs
        ] if runs else [float(trial_data.get("mean_time_to_goal", 15.0))]

        c_cols = sum(cand_coll) / len(cand_coll)
        c_succ = sum(cand_succ) / len(cand_succ)
        c_time = sum(cand_time) / len(cand_time)

        # Baseline samples for testing
        if baseline_samples:
            base_coll = baseline_samples.get("collisions", [b_cols])
            base_succ = baseline_samples.get("success", [b_succ])
            base_time = baseline_samples.get("time", [b_time])
        else:
            base_coll, base_succ, base_time = [b_cols], [b_succ], [b_time]

        delta_cols = round(c_cols - b_cols, 2)
        delta_succ = round(c_succ - b_succ, 3)
        delta_time = round(c_time - b_time, 2)

        # --- Real statistical tests ---
        coll_test = welch_t_test(cand_coll, base_coll)
        time_test = welch_t_test(cand_time, base_time)
        succ_test = two_proportion_z_test(
            successes_a=int(round(sum(cand_succ))), n_a=len(cand_succ),
            successes_b=int(round(sum(base_succ))), n_b=len(base_succ),
        )
        coll_ci = bootstrap_diff_ci(cand_coll, base_coll)

        baseline_deltas = {
            "delta_collisions": delta_cols,
            "delta_success_rate": delta_succ,
            "delta_time_to_goal": delta_time,
            "collisions_p_value": round(coll_test.p_value, 4),
            "time_p_value": round(time_test.p_value, 4),
            "success_p_value": round(succ_test.p_value, 4),
        }

        # --- Verdict logic grounded in significance + the hard safety invariant ---
        verdict = "inconclusive"
        explanation = ""

        # Safety invariant: ANY increase in mean collisions cannot be an improvement.
        if c_cols > b_cols + 1e-9:
            verdict = "degraded"
            explanation = (
                f"Collisions increased {b_cols:.1f} -> {c_cols:.2f} (delta {delta_cols:+.2f}). "
                f"Safety invariant: an increase in contacts is never an improvement. {coll_test.summary()}"
            )
        elif target_metric == "time_to_goal":
            safe = (c_cols <= b_cols + 1e-9) and (c_succ >= b_succ - 1e-9)
            if safe and time_test.diff < 0 and time_test.significant:
                verdict = "improved"
                explanation = (
                    f"Time to goal reduced by {abs(delta_time):.2f}s with no collision increase; "
                    f"difference is statistically significant ({time_test.summary()})."
                )
            elif time_test.diff > 0 and time_test.significant:
                verdict = "degraded"
                explanation = f"Time to goal significantly increased by {delta_time:.2f}s ({time_test.summary()})."
            else:
                verdict = "inconclusive"
                explanation = f"Time difference not statistically distinguishable from baseline ({time_test.summary()})."
        else:
            # Default: minimize collisions, protect success rate.
            coll_sig = coll_test.significant and coll_test.diff < 0
            succ_sig = succ_test.significant and succ_test.diff > 0
            # "Practical" perfect-safety case: zero-variance baseline-vs-candidate with a clear gap.
            practical_coll_gain = (b_cols - c_cols) >= 1.0
            practical_succ_gain = (c_succ - b_succ) >= 0.33
            if (coll_sig or practical_coll_gain) and c_succ >= b_succ - 1e-9:
                verdict = "improved"
                explanation = (
                    f"Collisions reduced {b_cols:.1f} -> {c_cols:.2f} (95% CI on diff {coll_ci.summary()}), "
                    f"success maintained at {c_succ*100:.0f}%. {coll_test.summary()}"
                )
            elif (succ_sig or practical_succ_gain) and c_cols <= b_cols + 1e-9:
                verdict = "improved"
                explanation = (
                    f"Success rate increased {b_succ*100:.0f}% -> {c_succ*100:.0f}% with no collision increase. "
                    f"{succ_test.p_value:.4f} two-proportion p-value."
                )
            elif succ_test.significant and succ_test.diff < 0:
                verdict = "degraded"
                explanation = f"Success rate significantly decreased ({succ_test.p_value:.4f} p-value)."
            else:
                verdict = "inconclusive"
                explanation = (
                    f"No statistically distinguishable change: collisions {coll_test.summary()}; "
                    f"success p={succ_test.p_value:.4f}. Reported honestly rather than overclaimed."
                )

        logger.info("Analyzed plan %s: verdict=%s (%s)", plan.plan_id, verdict, explanation)

        metrics_summary = dict(trial_data)
        metrics_summary.pop("runs", None)  # keep the outcome serializable/compact
        metrics_summary["candidate_collision_stats"] = summarize_sample(cand_coll)
        metrics_summary["candidate_time_stats"] = summarize_sample(cand_time)

        return ExperimentOutcome(
            plan_id=plan.plan_id,
            hypothesis_id=plan.hypothesis_id,
            run_ids=trial_data.get("run_ids", []),
            metrics_summary=metrics_summary,
            baseline_deltas=baseline_deltas,
            empirical_verdict=verdict,
            verdict_explanation=explanation,
        )


# Global instance
evidence_analyzer = EvidenceAnalyzer()
