"""
Candidate Ranker and Experiment Selection Engine for RoboForge Scientific Intelligence.
Implements principled candidate ranking based on Expected Improvement (EI), parameter uncertainty,
information gain, prior empirical outcomes, and physical boundary enforcement.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.research.models import (
    ResearchMission, Hypothesis, ControlledExperimentPlan, ExperimentOutcome,
    TaskCompatibilitySignature
)
from backend.research.research_memory import research_memory

logger = logging.getLogger("roboforge.research.ranker")


class CandidateEvaluation:
    def __init__(
        self,
        candidate_id: str,
        hypothesis_id: str,
        modifications: Dict[str, Any],
        expected_improvement: float,
        information_gain: float,
        acquisition_score: float,
        rationale: str,
        prior_influence: str = "",
        is_valid: bool = True,
        validation_error: str = ""
    ):
        self.candidate_id = candidate_id
        self.hypothesis_id = hypothesis_id
        self.modifications = modifications
        self.expected_improvement = expected_improvement
        self.information_gain = information_gain
        self.acquisition_score = acquisition_score
        self.rationale = rationale
        self.prior_influence = prior_influence
        self.is_valid = is_valid
        self.validation_error = validation_error


class CandidateRanker:
    """
    Ranks candidate experiment configurations using an acquisition function combining:
    1. Expected Improvement (EI) grounded in causal physical mechanisms.
    2. Information Gain / Parameter-Space Uncertainty.
    3. History-Aware Prior Outcome Penalization/Boosting.
    4. Task & Robot Compatibility Fingerprint Isolation.
    5. Strict Physical Parameter Bounds Enforcement.
    """

    PHYSICAL_BOUNDS = {
        "task.policy.safety_margin": (0.10, 2.50),
        "task.policy.repulsion_gain": (0.10, 15.0),
        "task.policy.tangential_gain": (0.0, 10.0),
        "task.policy.attraction_gain": (0.10, 10.0),
        "task.policy.max_turn_rate": (0.50, 8.0),
        "task.policy.slowdown_distance": (0.10, 2.0),
        "robot.max_linear_velocity": (0.10, 3.0),
        "robot.max_angular_velocity": (0.20, 8.0),
        "settings.timestep": (0.001, 0.05)
    }

    def __init__(self, memory=None):
        self.memory = memory or research_memory

    def rank_candidates_for_hypothesis(
        self,
        mission: ResearchMission,
        hypothesis: Hypothesis,
        num_candidates: int = 5
    ) -> List[CandidateEvaluation]:
        """
        Synthesizes, evaluates, and ranks candidate parameter modifications for the given hypothesis.
        """
        var_name = hypothesis.controlled_variable
        base_val = hypothesis.baseline_value
        prop_val = hypothesis.proposed_value

        # 1. Inspect compatible prior research history
        compatible_history, incompatible_count = self._get_compatible_prior_outcomes(mission)
        history_summary = (
            f"Evaluated against {len(compatible_history)} compatible prior runs "
            f"({incompatible_count} incompatible runs excluded)."
        )

        # 2. Generate candidate points along and around the hypothesis proposal
        candidate_values = self._generate_candidate_values(var_name, base_val, prop_val, num_candidates)

        evaluated_candidates: List[CandidateEvaluation] = []

        for idx, val in enumerate(candidate_values):
            cand_mods = {var_name: val}
            
            # Check physical safety bounds
            is_valid, err_msg = self._validate_bounds(cand_mods)
            if not is_valid:
                evaluated_candidates.append(CandidateEvaluation(
                    candidate_id=f"cand_{hypothesis.hypothesis_id}_{idx}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    modifications=cand_mods,
                    expected_improvement=0.0,
                    information_gain=0.0,
                    acquisition_score=-1000.0,
                    rationale=f"Rejected: {err_msg}",
                    prior_influence=history_summary,
                    is_valid=False,
                    validation_error=err_msg
                ))
                continue

            # Check for repetition of identical failed experiment in current mission or history
            is_repeat, repeat_verdict = self._check_for_failed_repeat(cand_mods, mission, compatible_history)
            if is_repeat:
                evaluated_candidates.append(CandidateEvaluation(
                    candidate_id=f"cand_{hypothesis.hypothesis_id}_{idx}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    modifications=cand_mods,
                    expected_improvement=0.0,
                    information_gain=0.0,
                    acquisition_score=-500.0,
                    rationale=f"Penalized: Repeats equivalent prior {repeat_verdict} experiment without new evidence.",
                    prior_influence=history_summary,
                    is_valid=True,
                    validation_error="Repeats failed prior parameter without justification."
                ))
                continue

            # Expected Improvement (EI)
            ei = self._calculate_expected_improvement(val, prop_val, hypothesis.confidence, mission)

            # Information Gain / Parameter Uncertainty
            info_gain = self._calculate_information_gain(var_name, val, compatible_history, mission)

            # Prior History Adjustment
            prior_adj, prior_note = self._calculate_prior_history_adjustment(var_name, val, compatible_history)

            # Combined Acquisition Score
            # Score = 0.5 * EI + 0.3 * InfoGain + 0.2 * Confidence + PriorAdj
            acquisition_score = round(
                (0.50 * ei) + (0.30 * info_gain) + (0.20 * hypothesis.confidence) + prior_adj,
                3
            )

            rationale = (
                f"Candidate {var_name}={val}: Expected Improvement={ei:.2f}, "
                f"Info Gain={info_gain:.2f}, Prior Influence={prior_note}. "
                f"Mechanism: {hypothesis.mechanism}"
            )

            evaluated_candidates.append(CandidateEvaluation(
                candidate_id=f"cand_{hypothesis.hypothesis_id}_{idx}",
                hypothesis_id=hypothesis.hypothesis_id,
                modifications=cand_mods,
                expected_improvement=round(ei, 3),
                information_gain=round(info_gain, 3),
                acquisition_score=acquisition_score,
                rationale=rationale,
                prior_influence=f"{history_summary} {prior_note}".strip(),
                is_valid=True
            ))

        # Sort descending by acquisition score
        evaluated_candidates.sort(key=lambda c: c.acquisition_score, reverse=True)
        return evaluated_candidates

    def _generate_candidate_values(
        self,
        var_name: str,
        base_val: Any,
        prop_val: Any,
        count: int
    ) -> List[float]:
        """Generates candidate test points, with primary proposed value first."""
        if not isinstance(prop_val, (int, float)):
            return [prop_val]

        b = float(base_val) if isinstance(base_val, (int, float)) else prop_val
        p = float(prop_val)
        delta = p - b

        candidates = [round(p, 3)]
        if abs(delta) > 1e-4:
            # Conservative probe (70% of delta)
            candidates.append(round(b + 0.70 * delta, 3))
            # Aggressive probe (130% of delta)
            candidates.append(round(b + 1.30 * delta, 3))
            # Midpoint (50% of delta)
            candidates.append(round(b + 0.50 * delta, 3))
            # Counter-probe
            candidates.append(round(b - 0.30 * delta, 3))

        unique_cands: List[float] = []
        for c in candidates:
            if not any(abs(c - u) < 1e-4 for u in unique_cands):
                unique_cands.append(c)

        return unique_cands[:count]

    def _validate_bounds(self, modifications: Dict[str, Any]) -> Tuple[bool, str]:
        for var_path, val in modifications.items():
            if var_path in self.PHYSICAL_BOUNDS:
                min_v, max_v = self.PHYSICAL_BOUNDS[var_path]
                if not isinstance(val, (int, float)):
                    return False, f"Parameter '{var_path}' must be numeric."
                if val < min_v or val > max_v:
                    return False, f"Parameter '{var_path}'={val} violates safety bounds [{min_v}, {max_v}]."
        return True, ""

    def _check_for_failed_repeat(
        self,
        modifications: Dict[str, Any],
        current_mission: ResearchMission,
        compatible_history: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Checks if identical parameter values previously yielded degraded outcomes."""
        # 1. Current mission: do not repeat any already executed plan (degraded or inconclusive)
        for p in current_mission.plans:
            matching_outcomes = [o for o in current_mission.outcomes if o.plan_id == p.plan_id]
            if matching_outcomes:
                verdict = matching_outcomes[0].empirical_verdict
                if verdict in ("degraded", "inconclusive"):
                    all_match = True
                    for k, v in modifications.items():
                        if k not in p.parameter_modifications or abs(float(p.parameter_modifications[k]) - float(v)) > 1e-4:
                            all_match = False
                            break
                    if all_match and len(modifications) == len(p.parameter_modifications):
                        return True, verdict

        # 2. Across compatible past missions: only penalize if it was explicitly DEGRADED (harmful/unsafe)
        for h in compatible_history:
            verdict = h.get("verdict", "")
            if verdict == "degraded":
                past_mods = h.get("parameters", {})
                all_match = True
                for k, v in modifications.items():
                    if k not in past_mods or abs(float(past_mods[k]) - float(v)) > 1e-4:
                        all_match = False
                        break
                if all_match and len(modifications) == len(past_mods):
                    return True, "degraded"

        return False, ""

    def _calculate_expected_improvement(
        self,
        val: float,
        proposed_target: float,
        hypothesis_confidence: float,
        mission: ResearchMission
    ) -> float:
        """Estimates expected reduction in objective metric or probability of success improvement."""
        # Distance to proposed theoretical optimum
        dist = abs(val - proposed_target)
        # Closer to proposed hypothesis target yields higher prior confidence
        closeness = math.exp(-0.5 * (dist / (abs(proposed_target) + 1e-5)))
        return hypothesis_confidence * closeness

    def _calculate_information_gain(
        self,
        var_name: str,
        val: float,
        compatible_history: List[Dict[str, Any]],
        mission: ResearchMission
    ) -> float:
        """Measures parameter uncertainty / dispersion from previously evaluated trial points in this mission."""
        sampled_points = []
        for p in mission.plans:
            if var_name in p.parameter_modifications:
                sampled_points.append(float(p.parameter_modifications[var_name]))

        if not sampled_points:
            return 1.0  # Max uncertainty/information gain for unexplored parameter in this mission

        min_dist = min(abs(val - sp) for sp in sampled_points)
        param_range = 1.0
        if var_name in self.PHYSICAL_BOUNDS:
            param_range = self.PHYSICAL_BOUNDS[var_name][1] - self.PHYSICAL_BOUNDS[var_name][0]
        norm_dist = min(1.0, min_dist / (param_range * 0.25 + 1e-5))
        return norm_dist

    def _calculate_prior_history_adjustment(
        self,
        var_name: str,
        val: float,
        compatible_history: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Boosts parameter directions that showed improvement in prior compatible research."""
        bonus = 0.0
        notes = []
        for h in compatible_history:
            params = h.get("parameters", {})
            verdict = h.get("verdict", "")
            if var_name in params:
                past_val = float(params[var_name])
                if verdict == "improved" and abs(val - past_val) < 0.2:
                    bonus += 0.25
                    notes.append(f"+0.25 prior improved near {past_val}")
                elif verdict == "degraded" and abs(val - past_val) < 0.1:
                    bonus -= 0.35
                    notes.append(f"-0.35 prior degraded near {past_val}")

        return bonus, "; ".join(notes) if notes else "No direct prior bias"

    def _get_compatible_prior_outcomes(
        self,
        current_mission: ResearchMission
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieves experiment outcomes from persistent research memory,
        strictly filtering out results from incompatible robots or simulator physics configurations.
        """
        all_missions = self.memory.list_missions()
        compatible_outcomes: List[Dict[str, Any]] = []
        incompatible_count = 0

        curr_sig = current_mission.task_signature

        for m in all_missions:
            if m.mission_id == current_mission.mission_id:
                continue

            # Compatibility check
            is_compat, reason = curr_sig.is_compatible_with(m.task_signature)
            if not is_compat:
                incompatible_count += 1
                logger.debug("Excluded prior mission %s from history reuse: %s", m.mission_id, reason)
                continue

            for p in m.plans:
                matching_outcomes = [o for o in m.outcomes if o.plan_id == p.plan_id]
                verdict = matching_outcomes[0].empirical_verdict if matching_outcomes else "inconclusive"
                compatible_outcomes.append({
                    "mission_id": m.mission_id,
                    "plan_id": p.plan_id,
                    "parameters": p.parameter_modifications,
                    "verdict": verdict,
                    "deltas": matching_outcomes[0].baseline_deltas if matching_outcomes else {}
                })

        return compatible_outcomes, incompatible_count


# Global singleton instance
candidate_ranker = CandidateRanker()
