"""
Research Orchestrator for RoboForge Scientific Intelligence.
Manages the end-to-end scientific research lifecycle, state transitions, budget tracking,
supervised/autonomous modes, cancellation, and reproducible report generation.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from backend.research.models import (
    ResearchMission, MissionBudget, Hypothesis, ControlledExperimentPlan,
    ExperimentOutcome, ValidationReport, ActionLogEntry
)
from backend.research.mission_compiler import mission_compiler
from backend.research.hypothesis_engine import hypothesis_engine
from backend.research.experiment_designer import experiment_designer
from backend.research.evidence_analyzer import evidence_analyzer
from backend.research.research_validator import research_validator
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory

logger = logging.getLogger("roboforge.research.orchestrator")


class ResearchOrchestrator:
    """Orchestrates autonomous and supervised scientific robotics research cycles."""

    def __init__(self, memory=None, tools=None):
        self.memory = memory or research_memory
        self.tools = tools or simulation_tools

    def start_mission(
        self,
        goal_description: str,
        mode: str = "autonomous",
        budget: Optional[MissionBudget] = None,
        base_scenario_id: Optional[str] = None
    ) -> ResearchMission:
        """Initializes a new research mission and begins execution."""
        mission = mission_compiler.compile(
            goal_description=goal_description,
            mode=mode,
            budget=budget,
            base_scenario_id=base_scenario_id
        )

        logger.info("Started research mission '%s' in %s mode", mission.mission_id, mode)

        # Advance until paused (awaiting approval in supervised mode), completed, or budget exhausted
        mission = self.run_until_pause_or_complete(mission.mission_id)
        return mission

    def step_mission(self, mission_id: str) -> ResearchMission:
        """Executes one state transition step in the scientific research loop."""
        mission = self.memory.get_mission(mission_id)
        if not mission:
            raise KeyError(f"Mission '{mission_id}' not found.")

        if mission.status in ("completed", "failed", "cancelled"):
            return mission

        # Budget & Timeout Check
        if self._check_budget_exceeded(mission):
            return self._finalize_mission(mission, status="completed", reason="Budget limit reached.")

        start_time = time.time()

        try:
            # STATE 1: CREATED -> RUN BASELINE
            if mission.status == "created":
                mission.status = "baseline_running"
                self.memory.save_mission(mission)

                logger.info("Executing authoritative baseline simulation for mission '%s'", mission.mission_id)
                base_run = self.tools.run_scenario(mission.base_scenario_id, seed=100)

                b_dict = {
                    "run_id": base_run.run_id,
                    "total_collisions": base_run.metrics.total_collisions,
                    "success": base_run.metrics.success,
                    "time_to_goal": base_run.metrics.time_to_goal,
                    "total_duration": base_run.metrics.total_duration,
                    "max_speed": base_run.metrics.max_linear_speed,
                    "distance_traveled": base_run.metrics.total_distance_traveled
                }

                mission.baseline_run_id = base_run.run_id
                mission.baseline_metrics = b_dict
                mission.budget.used_experiments += 1

                mission.action_log.append(ActionLogEntry(
                    step="baseline_collection",
                    action="run_baseline_simulation",
                    status="success",
                    details=b_dict,
                    message=f"Baseline established: {b_dict['total_collisions']} collisions, success={b_dict['success']}, duration={b_dict['total_duration']:.2f}s."
                ))

                mission.status = "hypothesizing"
                self.memory.save_mission(mission)

            # STATE 2: HYPOTHESIZING -> FORMULATE HYPOTHESES & DESIGN PLAN
            if mission.status == "hypothesizing":
                if not mission.hypotheses:
                    hypothesis_engine.formulate_competing_hypotheses(
                        mission=mission,
                        baseline_metrics=mission.baseline_metrics or {}
                    )

                # Select next untested hypothesis
                active_hypo = hypothesis_engine.select_next_hypothesis(mission)
                if not active_hypo:
                    # No more hypotheses to test -> finalize
                    logger.info("No further untested hypotheses for mission '%s'", mission.mission_id)
                    return self._finalize_mission(mission, status="completed", reason="All candidate hypotheses evaluated.")

                mission.active_hypothesis_id = active_hypo.hypothesis_id

                # Design experiment plan for this hypothesis
                plan = experiment_designer.design_plan(mission, active_hypo)
                mission.budget.used_iterations += 1

                if mission.mode == "supervised" and plan.status == "pending_approval":
                    mission.status = "awaiting_approval"
                    logger.info("Mission '%s' pausing: awaiting user approval for plan '%s'", mission.mission_id, plan.plan_id)
                    self.memory.save_mission(mission)
                    return mission
                else:
                    mission.status = "experimenting"
                    self.memory.save_mission(mission)

            # STATE 3: EXPERIMENTING -> RUN CONTROLLED SIMULATION TRIALS
            if mission.status == "experimenting":
                active_plan = self.memory.get_plan(mission.active_plan_id)
                if not active_plan:
                    raise RuntimeError("No active experiment plan found to execute.")

                logger.info("Executing controlled experiment trials for plan '%s'", active_plan.plan_id)
                trial_data = self.tools.run_multi_trial(
                    scenario_id=active_plan.base_scenario_id,
                    seeds=active_plan.seeds,
                    parameter_overrides=active_plan.parameter_modifications
                )
                mission.budget.used_experiments += len(active_plan.seeds)

                # STATE 4: ANALYZING -> EVIDENCE COMPARISON
                outcome = evidence_analyzer.analyze(
                    plan=active_plan,
                    baseline_metrics=mission.baseline_metrics or {},
                    trial_data=trial_data,
                    target_metric=mission.target_metric,
                    optimization_direction=mission.optimization_direction
                )
                mission.outcomes.append(outcome)

                # Update hypothesis confidence
                active_hypo = hypothesis_engine.update_evidence(
                    hypothesis_id=active_plan.hypothesis_id,
                    outcome=outcome,
                    delta_metric=outcome.baseline_deltas.get("delta_collisions", 0.0),
                    optimization_direction=mission.optimization_direction
                )
                for idx, h in enumerate(mission.hypotheses):
                    if h.hypothesis_id == active_hypo.hypothesis_id:
                        mission.hypotheses[idx] = active_hypo
                        break

                mission.action_log.append(ActionLogEntry(
                    step="evidence_analysis",
                    action="analyze_experiment_outcome",
                    status="success" if outcome.empirical_verdict == "improved" else "info",
                    details={
                        "plan_id": active_plan.plan_id,
                        "hypothesis_id": active_hypo.hypothesis_id,
                        "verdict": outcome.empirical_verdict,
                        "deltas": outcome.baseline_deltas,
                        "new_confidence": active_hypo.confidence
                    },
                    message=f"Outcome: {outcome.empirical_verdict.upper()} - {outcome.verdict_explanation}"
                ))

                if outcome.empirical_verdict == "improved":
                    mission.best_candidate_plan_id = active_plan.plan_id
                    mission.status = "validating"
                else:
                    # Check if more hypotheses exist and budget allows
                    next_hypo = hypothesis_engine.select_next_hypothesis(mission)
                    if next_hypo and not self._check_budget_exceeded(mission):
                        mission.status = "hypothesizing"
                    else:
                        mission.status = "validating"

                self.memory.save_mission(mission)

            # STATE 5: VALIDATING -> HELD-OUT SCENARIO GENERALIZATION CHECK
            if mission.status == "validating":
                cand_plan = None
                if mission.best_candidate_plan_id:
                    cand_plan = self.memory.get_plan(mission.best_candidate_plan_id)
                elif mission.plans:
                    cand_plan = mission.plans[0]

                if cand_plan:
                    val_report = research_validator.validate_candidate(
                        mission=mission,
                        candidate_plan=cand_plan
                    )
                    mission.validation_report = val_report

                return self._finalize_mission(mission, status="completed", reason="Research loop completed successfully.")

        except Exception as e:
            logger.exception("Error during mission step for '%s': %s", mission_id, e)
            mission.status = "failed"
            mission.error_message = str(e)
            mission.action_log.append(ActionLogEntry(
                step="mission_execution",
                action="step_error",
                status="failed",
                details={"error": str(e)},
                message=f"Execution halted due to unexpected error: {e}"
            ))
            self.memory.save_mission(mission)

        return mission

    def run_until_pause_or_complete(self, mission_id: str, max_steps: int = 15) -> ResearchMission:
        """Loops until mission completes, pauses for approval, or hits maximum steps."""
        for _ in range(max_steps):
            mission = self.memory.get_mission(mission_id)
            if not mission or mission.status in ("completed", "failed", "cancelled", "awaiting_approval"):
                break
            mission = self.step_mission(mission_id)
        return mission

    def approve_plan(self, mission_id: str) -> ResearchMission:
        """Approves the pending experiment plan in supervised mode and resumes execution."""
        mission = self.memory.get_mission(mission_id)
        if not mission:
            raise KeyError(f"Mission '{mission_id}' not found.")

        if mission.status != "awaiting_approval":
            raise RuntimeError(f"Mission is in state '{mission.status}', not awaiting approval.")

        plan = self.memory.get_plan(mission.active_plan_id)
        if plan:
            plan.user_approved = True
            plan.status = "ready"
            self.memory.save_plan(plan)

        mission.action_log.append(ActionLogEntry(
            step="supervisor_interaction",
            action="user_approval_granted",
            status="success",
            details={"plan_id": mission.active_plan_id},
            message=f"Supervisor approved experiment plan '{mission.active_plan_id}'."
        ))

        mission.status = "experimenting"
        self.memory.save_mission(mission)

        # Advance execution
        if mission.mode == "autonomous" or True:
            mission = self.step_mission(mission.mission_id)
        return mission

    def cancel_mission(self, mission_id: str, reason: str = "User cancellation request") -> ResearchMission:
        """Cancels an ongoing research mission."""
        mission = self.memory.get_mission(mission_id)
        if not mission:
            raise KeyError(f"Mission '{mission_id}' not found.")

        mission.status = "cancelled"
        mission.error_message = reason
        mission.action_log.append(ActionLogEntry(
            step="mission_control",
            action="cancel_mission",
            status="cancelled",
            details={"reason": reason},
            message=f"Mission cancelled: {reason}."
        ))
        self.memory.save_mission(mission)
        logger.info("Mission '%s' was cancelled: %s", mission_id, reason)
        return mission

    def _check_budget_exceeded(self, mission: ResearchMission) -> bool:
        """Checks iteration, experiment count, or time budget exhaustion."""
        b = mission.budget
        if b.used_iterations >= b.max_iterations:
            return True
        if b.used_experiments >= b.max_experiments:
            return True
        return False

    def _finalize_mission(self, mission: ResearchMission, status: str = "completed", reason: str = "") -> ResearchMission:
        """Generates reproducible report and finalizes mission record."""
        # Ensure held-out validation was run if any candidate plan was tested
        if not mission.validation_report and mission.plans:
            cand_plan = None
            if mission.best_candidate_plan_id:
                cand_plan = self.memory.get_plan(mission.best_candidate_plan_id)
            if not cand_plan and mission.plans:
                cand_plan = mission.plans[0]
            if cand_plan:
                try:
                    mission.validation_report = research_validator.validate_candidate(
                        mission=mission,
                        candidate_plan=cand_plan
                    )
                except Exception as e:
                    logger.warning("Could not run validation report during finalization: %s", e)

        mission.status = status
        report_md = self.generate_final_report(mission)
        mission.final_report_markdown = report_md
        mission.updated_at = time.time()

        mission.action_log.append(ActionLogEntry(
            step="mission_completion",
            action="generate_final_report",
            status="success",
            details={"reason": reason, "best_candidate": mission.best_candidate_plan_id},
            message=f"Mission finalized. Reason: {reason}."
        ))

        self.memory.save_mission(mission)
        self.memory.save_report(mission.mission_id, report_md, metadata=mission.model_dump())
        return mission

    def generate_final_report(self, mission: ResearchMission) -> str:
        """Synthesizes a reproducible, evidence-grounded scientific research report."""
        b = mission.baseline_metrics or {}
        val = mission.validation_report

        md = []
        md.append(f"# RoboForge Scientific Intelligence -- Research Report")
        md.append(f"**Mission ID:** `{mission.mission_id}`  ")
        md.append(f"**Goal:** {mission.goal_description}  ")
        md.append(f"**Status:** {mission.status.upper()}  ")
        md.append(f"**Mode:** {mission.mode.title()}  ")
        md.append(f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(mission.updated_at))}  ")
        md.append("")
        md.append("---")
        md.append("")

        # 1. Baseline Performance
        md.append("## 1. Baseline Simulation Evidence")
        md.append(f"- **Scenario ID:** `{mission.base_scenario_id}` (Seed 100)")
        md.append(f"- **Total Collisions:** `{b.get('total_collisions', 'N/A')}`")
        md.append(f"- **Task Success:** `{b.get('success', False)}`")
        md.append(f"- **Time to Goal:** `{b.get('time_to_goal', 'N/A')}s`")
        md.append(f"- **Distance Traveled:** `{b.get('distance_traveled', 'N/A')}m`")
        md.append("")

        # 2. Competing Hypotheses & Evidence
        md.append("## 2. Competing Scientific Hypotheses & Evidence")
        md.append("| Hypothesis ID | Title | Controlled Variable | Status | Confidence | Supporting Evidence |")
        md.append("|---|---|---|---|---|---|")
        for h in mission.hypotheses:
            supp = "; ".join(h.supporting_evidence) if h.supporting_evidence else "None"
            md.append(f"| `{h.hypothesis_id}` | {h.title} | `{h.controlled_variable}` | **{h.status.upper()}** | `{h.confidence*100:.0f}%` | {supp} |")
        md.append("")

        # 3. Controlled Experiment Outcomes
        md.append("## 3. Controlled Experiments & Measured Deltas")
        if mission.outcomes:
            md.append("| Plan ID | Hypothesis | Verdict | $\\Delta$ Collisions | $\\Delta$ Success | Explanation |")
            md.append("|---|---|---|---|---|---|")
            for o in mission.outcomes:
                d_c = o.baseline_deltas.get("delta_collisions", 0.0)
                d_s = o.baseline_deltas.get("delta_success_rate", 0.0)
                md.append(f"| `{o.plan_id}` | `{o.hypothesis_id}` | **{o.empirical_verdict.upper()}** | `{d_c:+.2f}` | `{d_s*100:+.0f}%` | {o.verdict_explanation} |")
        else:
            md.append("*No controlled experiments completed.*")
        md.append("")

        # 4. Held-Out Generalization Validation
        md.append("## 4. Held-Out Generalization Verification")
        if val:
            md.append(f"**Generalization Verdict:** `{val.generalization_verdict.upper()}`  ")
            md.append(f"**Summary:** {val.summary}  ")
            md.append("")
            md.append("| Scenario | Baseline Succ / Coll | Candidate Succ / Coll | Verdict |")
            md.append("|---|---|---|---|")
            for t in val.trial_results:
                md.append(f"| `{t.scenario_name}` | {t.baseline_success} / {t.baseline_collisions} | {t.candidate_success} / {t.candidate_collisions} | **{t.verdict.upper()}** |")
        else:
            md.append("*Held-out evaluation was not executed.*")
        md.append("")

        # 5. Scientific Conclusions & Recommended Next Steps
        md.append("## 5. Conclusions & Limitations")
        if mission.best_candidate_plan_id:
            md.append(f"- **Recommended Candidate:** Plan `{mission.best_candidate_plan_id}` demonstrated measurable, repeatable improvement over baseline.")
        else:
            md.append("- **Conclusion:** None of the tested hypotheses outperformed baseline under strict safety constraints.")
        md.append("- **Limitations:** Evaluated with 2D planar physics in MuJoCo; dynamic surface friction variations were held constant.")
        md.append("- **Next Step:** Deploy validated parameters to production navigation controller or expand exploration to motor gain tuning.")

        return "\n".join(md)


# Global instance
research_orchestrator = ResearchOrchestrator()
