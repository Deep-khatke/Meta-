"""Evidence-Based Improvement Engine: inspects navigation failures, formulates a single controlled hypothesis, executes identical reruns, and records empirical outcomes without fabrication."""

import json
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.experiment_models import ExperimentConfig, ExperimentRunResult
from backend.experiment_manager import experiment_manager
from backend.rl_models import HypothesisRefinementResult

logger = logging.getLogger("roboforge.refinement")

HYPOTHESES_DIR = Path("experiments/hypotheses")
HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)


class HypothesisRefinementEngine:
    """Manages empirical hypotheses, controlled parameter changes, and honest verification."""

    @classmethod
    def analyze_and_refine(
        cls,
        scenario_id: str,
        focus_metric: str = "collision_rate"
    ) -> HypothesisRefinementResult:
        hypothesis_id = f"hypo_{uuid.uuid4().hex[:8]}"
        scen = experiment_manager.get_scenario(scenario_id)
        if not scen:
            # Fall back to first available scenario
            scens = experiment_manager.list_scenarios()
            if scens:
                scen = scens[0]
            else:
                raise FileNotFoundError(f"Scenario '{scenario_id}' not found.")

        # 1. Run baseline trial under deterministic seed
        seed = scen.seed
        logger.info("Executing baseline trial for hypothesis %s on scenario %s (seed %s)", hypothesis_id, scen.id, seed)
        baseline_res = experiment_manager.run_experiment(scen, seed_override=seed)

        b_cols = baseline_res.metrics.total_collisions
        b_succ = baseline_res.metrics.success
        b_time = baseline_res.metrics.time_to_goal or baseline_res.metrics.total_duration

        # 2. Formulate exactly ONE controlled hypothesis based on observed metrics
        current_policy = scen.task.policy
        curr_margin = current_policy.safety_margin if current_policy else 0.70
        curr_repulsion = current_policy.repulsion_gain if current_policy else 2.0
        curr_turn_rate = current_policy.max_turn_rate if current_policy else 3.0

        if b_cols > 0:
            observed_failure = f"Baseline experienced {b_cols} collision event(s) during obstacle evasion."
            # Hypothesis 1: Increase repulsive safety clearance
            controlled_param = "task.policy.safety_margin"
            baseline_val = curr_margin
            proposed_val = round(curr_margin + 0.22, 2)
            hypothesis = (
                f"Increasing obstacle safety margin from {baseline_val:.2f}m to {proposed_val:.2f}m will "
                f"trigger earlier evasive swerving and eliminate collision contact."
            )
            # Create modified scenario with only this single change
            refined_scen = scen.model_copy(deep=True)
            if refined_scen.task.policy:
                refined_scen.task.policy.safety_margin = proposed_val
        elif not b_succ:
            observed_failure = f"Baseline timed out without reaching the goal (final distance: {baseline_res.metrics.final_distance_to_target:.2f}m)."
            # Hypothesis 2: Increase maximum turn rate
            controlled_param = "task.policy.max_turn_rate"
            baseline_val = curr_turn_rate
            proposed_val = round(curr_turn_rate + 1.2, 1)
            hypothesis = (
                f"Increasing robot maximum turn rate from {baseline_val:.1f} to {proposed_val:.1f} rad/s will "
                f"reduce cornering radius and allow timely goal convergence within timeout."
            )
            refined_scen = scen.model_copy(deep=True)
            if refined_scen.task.policy:
                refined_scen.task.policy.max_turn_rate = proposed_val
        else:
            observed_failure = f"Baseline succeeded with 0 collisions in {b_time:.2f}s."
            # Hypothesis 3: Optimize trajectory duration by increasing repulsion responsiveness
            controlled_param = "task.policy.repulsion_gain"
            baseline_val = curr_repulsion
            proposed_val = round(curr_repulsion + 0.8, 1)
            hypothesis = (
                f"Sharpening obstacle repulsion gain from {baseline_val:.1f} to {proposed_val:.1f} will "
                f"allow tighter clearance and reduce time to goal."
            )
            refined_scen = scen.model_copy(deep=True)
            if refined_scen.task.policy:
                refined_scen.task.policy.repulsion_gain = proposed_val

        # 3. Execute refinement trial under identical seed and geometry
        logger.info("Executing controlled refinement trial for hypothesis %s...", hypothesis_id)
        refinement_res = experiment_manager.run_experiment(refined_scen, seed_override=seed)

        r_cols = refinement_res.metrics.total_collisions
        r_succ = refinement_res.metrics.success
        r_time = refinement_res.metrics.time_to_goal or refinement_res.metrics.total_duration

        # 4. Rigorous empirical verdict evaluation
        if b_cols > 0 and r_cols < b_cols:
            verdict = "SUPPORTED"
            evidence = (
                f"Hypothesis validated: Collisions reduced from {b_cols} to {r_cols}. "
                f"Time to goal: {b_time:.2f}s -> {r_time:.2f}s."
            )
            # Update scenario with the empirically validated improvement
            experiment_manager.save_scenario(refined_scen)
        elif not b_succ and r_succ:
            verdict = "SUPPORTED"
            evidence = (
                f"Hypothesis validated: Goal was successfully reached in {r_time:.2f}s "
                f"(baseline timed out). Collisions remained at {r_cols}."
            )
            experiment_manager.save_scenario(refined_scen)
        elif b_cols == 0 and r_cols == 0 and r_time < (b_time - 0.2):
            verdict = "SUPPORTED"
            evidence = (
                f"Hypothesis validated: Navigation time improved from {b_time:.2f}s to {r_time:.2f}s "
                f"while maintaining 0 collisions."
            )
            experiment_manager.save_scenario(refined_scen)
        elif r_cols > b_cols:
            verdict = "REFUTED"
            evidence = (
                f"Hypothesis refuted: Collisions increased from {b_cols} to {r_cols}. "
                f"Reverting proposed parameter change."
            )
        elif r_time > (b_time + 1.0) and r_cols == b_cols:
            verdict = "REFUTED"
            evidence = (
                f"Hypothesis refuted: Navigation time increased by {r_time - b_time:.2f}s "
                f"without safety improvement."
            )
        else:
            verdict = "INCONCLUSIVE"
            evidence = (
                f"Outcome neutral: Collisions remained {r_cols} vs {b_cols}, "
                f"time changed from {b_time:.2f}s to {r_time:.2f}s."
            )

        res = HypothesisRefinementResult(
            hypothesis_id=hypothesis_id,
            target_scenario_id=scen.id,
            observed_failure=observed_failure,
            hypothesis=hypothesis,
            controlled_parameter=controlled_param,
            baseline_value=baseline_val,
            proposed_value=proposed_val,
            baseline_run_id=baseline_res.run_id,
            refinement_run_id=refinement_res.run_id,
            baseline_collisions=b_cols,
            refinement_collisions=r_cols,
            baseline_success=b_succ,
            refinement_success=r_succ,
            baseline_time=round(b_time, 2),
            refinement_time=round(r_time, 2),
            outcome_verdict=verdict,
            empirical_evidence=evidence,
        )

        # Save hypothesis record to disk
        hypo_file = HYPOTHESES_DIR / f"{hypothesis_id}.json"
        with open(hypo_file, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=2)

        return res

    @classmethod
    def list_hypotheses(cls) -> List[Dict[str, Any]]:
        records = []
        for fpath in HYPOTHESES_DIR.glob("*.json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records.append({
                        "hypothesis_id": data.get("hypothesis_id"),
                        "timestamp": data.get("timestamp"),
                        "target_scenario_id": data.get("target_scenario_id"),
                        "controlled_parameter": data.get("controlled_parameter"),
                        "outcome_verdict": data.get("outcome_verdict"),
                        "empirical_evidence": data.get("empirical_evidence"),
                    })
            except Exception:
                pass
        records.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return records
