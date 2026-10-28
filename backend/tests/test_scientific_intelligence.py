"""
Comprehensive regression test suite for RoboForge Scientific Intelligence Subsystem (Core Engine v1).
Tests end-to-end research loop, competing hypotheses, controlled experimentation, simulation evidence,
held-out generalization, supervised approval gating, cancellation, budget enforcement, and memory persistence.
"""

import time
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.research.models import (
    ResearchMission, MissionBudget, Hypothesis, ControlledExperimentPlan, ExperimentOutcome
)
from backend.research.mission_compiler import mission_compiler
from backend.research.hypothesis_engine import hypothesis_engine
from backend.research.experiment_designer import experiment_designer
from backend.research.evidence_analyzer import evidence_analyzer
from backend.research.research_validator import research_validator
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory
from backend.research.research_orchestrator import research_orchestrator
from backend.research.reasoning_provider import (
    DeterministicRuleEngine, LanguageModelReasoningProvider, get_reasoning_provider
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Mission Compilation & Goal Interpretation
# ---------------------------------------------------------------------------

def test_mission_compiler_translates_goal_to_quantitative_targets():
    """Verify mission compiler parses natural language goals into quantitative targets and constraints."""
    mission = mission_compiler.compile(
        goal_description="Eliminate collisions around obstacles and reach goal under 15 seconds",
        mode="autonomous"
    )

    assert mission.mission_id.startswith("mission_")
    assert mission.target_metric == "total_collisions"
    assert mission.target_threshold == 0.0
    assert mission.optimization_direction == "minimize"
    assert any("time_to_goal <= 15" in c for c in mission.constraints)
    assert len(mission.action_log) >= 1
    assert mission.action_log[0].step == "mission_compilation"


# ---------------------------------------------------------------------------
# 2. Competing Hypotheses & Evidence Tracking
# ---------------------------------------------------------------------------

def test_competing_hypotheses_formulation_and_mechanisms():
    """Verify hypothesis engine generates multiple distinct, competing physical hypotheses."""
    mission = mission_compiler.compile(
        goal_description="Safely navigate dense obstacle field",
        mode="autonomous"
    )
    # Simulate baseline failure with 2 collisions
    baseline_metrics = {
        "total_collisions": 2,
        "success": True,
        "time_to_goal": 14.2,
        "total_duration": 14.2
    }

    hypotheses = hypothesis_engine.formulate_competing_hypotheses(
        mission=mission,
        baseline_metrics=baseline_metrics
    )

    assert len(hypotheses) >= 2, "Must formulate at least 2 competing hypotheses"
    
    # Check that variables and mechanisms differ across hypotheses
    variables = [h.controlled_variable for h in hypotheses]
    assert len(set(variables)) == len(variables), "Competing hypotheses must test different controlled variables"
    
    for h in hypotheses:
        assert h.status == "untested"
        assert h.confidence == 0.5
        assert len(h.mechanism) > 10, "Each hypothesis must propose a causal mechanism"


def test_hypothesis_evidence_update_integrity():
    """Verify empirical evidence updates adjust confidence truthfully without fabrication."""
    mission = mission_compiler.compile(goal_description="Clear obstacle course", mode="autonomous")
    hypo = Hypothesis(
        mission_id=mission.mission_id,
        title="Safety Margin Increase",
        statement="Increasing safety margin eliminates collisions",
        mechanism="Earlier repulsion",
        controlled_variable="task.policy.safety_margin",
        baseline_value=0.70,
        proposed_value=0.95
    )
    research_memory.save_hypothesis(hypo)

    # 1. Test supported hypothesis update (improved outcome)
    outcome_improved = ExperimentOutcome(
        plan_id="plan_test_1",
        hypothesis_id=hypo.hypothesis_id,
        empirical_verdict="improved",
        baseline_deltas={"delta_collisions": -2.0},
        verdict_explanation="Collisions reduced to 0"
    )
    updated_hypo = hypothesis_engine.update_evidence(
        hypothesis_id=hypo.hypothesis_id,
        outcome=outcome_improved,
        delta_metric=-2.0,
        optimization_direction="minimize"
    )
    assert updated_hypo.status == "supported"
    assert updated_hypo.confidence > 0.5
    assert len(updated_hypo.supporting_evidence) == 1

    # 2. Test refuted hypothesis update (degraded outcome)
    outcome_degraded = ExperimentOutcome(
        plan_id="plan_test_2",
        hypothesis_id=hypo.hypothesis_id,
        empirical_verdict="degraded",
        baseline_deltas={"delta_collisions": 3.0},
        verdict_explanation="Collisions increased"
    )
    updated_hypo_2 = hypothesis_engine.update_evidence(
        hypothesis_id=hypo.hypothesis_id,
        outcome=outcome_degraded,
        delta_metric=3.0,
        optimization_direction="minimize"
    )
    assert updated_hypo_2.status == "refuted"
    assert updated_hypo_2.confidence < 0.5
    assert len(updated_hypo_2.contradicting_evidence) == 1


# ---------------------------------------------------------------------------
# 3. Controlled Experiment Designer & Physics Validation
# ---------------------------------------------------------------------------

def test_controlled_experiment_design_and_validation():
    """Verify experiment designer produces validated plans with statistical seed budgets."""
    mission = mission_compiler.compile(goal_description="Reduce collisions", mode="autonomous")
    hypo = Hypothesis(
        mission_id=mission.mission_id,
        title="Turn Rate Scaling",
        statement="Increasing turn rate improves swerving",
        mechanism="Faster angular response",
        controlled_variable="task.policy.max_turn_rate",
        baseline_value=3.0,
        proposed_value=4.5
    )

    plan = experiment_designer.design_plan(mission, hypo)
    assert plan.is_validated is True
    assert len(plan.seeds) >= 3, "Must include multi-trial seeds for statistical significance"
    assert plan.parameter_modifications["task.policy.max_turn_rate"] == 4.5
    assert plan.status == "ready"


def test_invalid_experiment_plan_rejection():
    """Verify validator rejects unphysical parameters and out-of-bounds modifications."""
    # Test extreme velocity violating physical bounds
    invalid_plan = ControlledExperimentPlan(
        mission_id="m_test",
        hypothesis_id="h_test",
        title="Unphysical Speed",
        rationale="Testing extreme speed",
        base_scenario_id="scen_mvqgu87z",
        parameter_modifications={"robot.max_linear_velocity": 99.0},  # Bound is <= 3.0
        expected_outcome="Too fast"
    )
    errors = experiment_designer.validate_plan(invalid_plan)
    assert len(errors) > 0
    assert any("violates physical safety bounds" in e for e in errors)

    # Test unauthorized target path
    unauthorized_plan = ControlledExperimentPlan(
        mission_id="m_test",
        hypothesis_id="h_test",
        title="Unauthorized Path",
        rationale="Testing injection",
        base_scenario_id="scen_mvqgu87z",
        parameter_modifications={"os.system": "malicious_code"},
        expected_outcome="Fail"
    )
    errors_sec = experiment_designer.validate_plan(unauthorized_plan)
    assert any("Unauthorized or unrecognized" in e for e in errors_sec)


# ---------------------------------------------------------------------------
# 4. Evidence Analyzer & Statistical Comparison
# ---------------------------------------------------------------------------

def test_evidence_analyzer_safety_invariants():
    """Verify evidence analyzer never claims improvement if collisions increase."""
    plan = ControlledExperimentPlan(
        mission_id="m_test",
        hypothesis_id="h_test",
        title="Test Plan",
        rationale="Test",
        base_scenario_id="scen_mvqgu87z",
        parameter_modifications={"task.policy.safety_margin": 0.5}
    )
    baseline_metrics = {"total_collisions": 1, "success": True, "time_to_goal": 10.0}
    
    # Candidate was 2 seconds faster, but had 3 collisions (+2 collisions)
    trial_data = {
        "mean_collisions": 3.0,
        "success_rate": 1.0,
        "mean_time_to_goal": 8.0,
        "run_ids": ["run_1", "run_2"]
    }

    outcome = evidence_analyzer.analyze(
        plan=plan,
        baseline_metrics=baseline_metrics,
        trial_data=trial_data,
        target_metric="time_to_goal"  # Even if user wanted faster time, safety invariant must block
    )

    assert outcome.empirical_verdict == "degraded", "Collisions increased; cannot claim improvement"
    assert outcome.baseline_deltas["delta_collisions"] == 2.0


# ---------------------------------------------------------------------------
# 5. Supervised vs. Autonomous Operating Modes
# ---------------------------------------------------------------------------

def test_supervised_mode_approval_gating():
    """Verify supervised mode halts at awaiting_approval until user approval is received."""
    mission = research_orchestrator.start_mission(
        goal_description="Supervised obstacle clearance test",
        mode="supervised",
        budget=MissionBudget(max_iterations=2, max_experiments=4)
    )

    # In supervised mode, it must pause awaiting approval
    assert mission.status == "awaiting_approval"
    assert mission.active_plan_id is not None
    plan = research_memory.get_plan(mission.active_plan_id)
    assert plan.status == "pending_approval"
    assert plan.user_approved is False

    # Grant approval
    approved_mission = research_orchestrator.approve_plan(mission.mission_id)
    assert approved_mission.status in ("experimenting", "validating", "completed", "hypothesizing", "awaiting_approval")
    approved_plan = research_memory.get_plan(mission.active_plan_id)
    assert approved_plan.user_approved is True


# ---------------------------------------------------------------------------
# 6. Mission Cancellation & Budget Enforcement
# ---------------------------------------------------------------------------

def test_mission_cancellation():
    """Verify cancellation stops research execution and persists clean cancelled state."""
    mission = mission_compiler.compile(goal_description="Test cancellation", mode="supervised")
    cancelled = research_orchestrator.cancel_mission(mission.mission_id, reason="User cancelled test")
    
    assert cancelled.status == "cancelled"
    assert "User cancelled" in (cancelled.error_message or "")
    
    # Attempting to step a cancelled mission does not alter its status
    stepped = research_orchestrator.step_mission(mission.mission_id)
    assert stepped.status == "cancelled"


def test_budget_exhaustion_stops_mission():
    """Verify mission stops when iteration or experiment budget is exhausted."""
    budget = MissionBudget(max_iterations=1, max_experiments=2)
    mission = mission_compiler.compile(goal_description="Budget limit test", mode="autonomous", budget=budget)
    mission.budget.used_iterations = 1
    research_memory.save_mission(mission)
    
    finalized = research_orchestrator.step_mission(mission.mission_id)
    assert finalized.status == "completed"
    assert "Budget limit reached" in finalized.action_log[-1].details.get("reason", "")


# ---------------------------------------------------------------------------
# 7. Persistent Memory & Rebound
# ---------------------------------------------------------------------------

def test_research_memory_persistence_and_search():
    """Verify research memory persists to disk, survives retrieval, and supports keyword search."""
    mission = ResearchMission(
        title="Searchable Acoustic SLAM Navigation Mission",
        goal_description="Investigate acoustic resonance tracking in cluttered warehouse",
        mode="autonomous"
    )
    research_memory.save_mission(mission)

    # Retrieve from disk
    loaded = research_memory.get_mission(mission.mission_id)
    assert loaded is not None
    assert loaded.title == mission.title

    # Search
    search_results = research_memory.search_missions("acoustic resonance")
    assert any(m.mission_id == mission.mission_id for m in search_results)


# ---------------------------------------------------------------------------
# 8. Model Provider Transparency & Blocked State
# ---------------------------------------------------------------------------

def test_model_provider_blocked_state_on_missing_credentials():
    """Verify LLM reasoning provider truthfully reports blocked status when no API key is set."""
    provider = LanguageModelReasoningProvider(api_key=None)
    assert provider.is_available() is False
    assert provider.provider_type == "blocked"

    # Attempting LLM reasoning without key raises clear error rather than faking
    with pytest.raises(RuntimeError) as exc_info:
        provider.interpret_goal("Test goal", {})
    assert "missing API credentials" in str(exc_info.value)


def test_deterministic_rule_engine_honesty():
    """Verify deterministic provider reports truthful non-LLM provider type."""
    engine = DeterministicRuleEngine()
    assert engine.provider_type == "deterministic_rule_engine"
    assert engine.is_available() is True
    res = engine.interpret_goal("Avoid obstacles with zero collisions", {})
    assert res["target_metric"] == "total_collisions"


# ---------------------------------------------------------------------------
# 9. Held-Out Generalization Validation
# ---------------------------------------------------------------------------

def test_held_out_validation_execution():
    """Verify research validator tests candidate on separate held-out scenarios."""
    mission = mission_compiler.compile(goal_description="Generalization test", mode="autonomous")
    plan = ControlledExperimentPlan(
        mission_id=mission.mission_id,
        hypothesis_id="h_gen",
        title="Conservative Safety Margin",
        rationale="Test generalization",
        base_scenario_id=mission.base_scenario_id,
        parameter_modifications={"task.policy.safety_margin": 0.88}
    )

    report = research_validator.validate_candidate(mission, plan)
    assert report.report_id.startswith("val_")
    assert report.generalization_verdict in ("generalized", "overfit", "failed")
    assert len(report.trial_results) >= 1
    for t in report.trial_results:
        assert t.scenario_id != ""
        assert t.verdict in ("passed", "failed")


# ---------------------------------------------------------------------------
# 10. Complete End-to-End Autonomous Research Cycle & Reproducible Report
# ---------------------------------------------------------------------------

def test_end_to_end_autonomous_navigation_research_cycle():
    """
    Demonstrable end-to-end acceptance test:
    1. Compiles navigation goal.
    2. Runs real baseline in MuJoCo.
    3. Proposes competing hypotheses.
    4. Executes controlled multi-trial experiment in real simulator.
    5. Analyzes measured evidence against baseline.
    6. Validates candidate on held-out scenarios.
    7. Produces reproducible Markdown report.
    """
    budget = MissionBudget(max_iterations=2, max_experiments=6, max_time_seconds=60.0)
    mission = research_orchestrator.start_mission(
        goal_description="Eliminate obstacle collisions and reach goal safely",
        mode="autonomous",
        budget=budget,
        base_scenario_id="scen_mvqgu87z"
    )

    assert mission.status == "completed", f"Mission ended in status {mission.status}"
    assert mission.baseline_metrics is not None, "Real baseline must be measured"
    assert len(mission.hypotheses) >= 2, "Must formulate competing hypotheses"
    assert len(mission.plans) >= 1, "Must design controlled experiment"
    assert len(mission.outcomes) >= 1, "Must measure empirical outcome"
    assert mission.validation_report is not None, "Held-out validation must be executed"
    assert mission.final_report_markdown is not None, "Reproducible Markdown report must be generated"

    # Verify report contents
    report_text = mission.final_report_markdown
    assert "## 1. Baseline Simulation Evidence" in report_text
    assert "## 2. Competing Scientific Hypotheses & Evidence" in report_text
    assert "## 3. Controlled Experiments & Measured Deltas" in report_text
    assert "## 4. Held-Out Generalization Verification" in report_text
    assert "## 5. Conclusions & Limitations" in report_text
    assert f"Mission ID:** `{mission.mission_id}`" in report_text


# ---------------------------------------------------------------------------
# 11. FastAPI REST API Integration
# ---------------------------------------------------------------------------

def test_research_api_endpoints_integration():
    """Verify FastAPI endpoints for mission creation, query, step, providers, and report export."""
    # 1. GET /api/research/providers
    res_p = client.get("/api/research/providers")
    assert res_p.status_code == 200
    data_p = res_p.json()
    assert "active_provider" in data_p
    assert data_p["allows_arbitrary_code_execution"] is False

    # 2. POST /api/research/missions (supervised)
    req_payload = {
        "goal_description": "Supervised testing via API",
        "mode": "supervised",
        "max_iterations": 2,
        "max_experiments": 4
    }
    res_m = client.post("/api/research/missions", json=req_payload)
    assert res_m.status_code == 200
    m_data = res_m.json()
    mid = m_data["mission_id"]
    assert m_data["mode"] == "supervised"
    assert m_data["status"] == "awaiting_approval"

    # 3. GET /api/research/missions/{mission_id}
    res_g = client.get(f"/api/research/missions/{mid}")
    assert res_g.status_code == 200
    assert res_g.json()["mission_id"] == mid

    # 4. POST /api/research/missions/{mission_id}/approve
    res_a = client.post(f"/api/research/missions/{mid}/approve")
    assert res_a.status_code == 200

    # 5. GET /api/research/missions/{mission_id}/report
    res_r = client.get(f"/api/research/missions/{mid}/report")
    assert res_r.status_code == 200
    assert "markdown_report" in res_r.json()

    # 6. GET /api/research/missions
    res_l = client.get("/api/research/missions?limit=10")
    assert res_l.status_code == 200
    assert len(res_l.json()) >= 1
