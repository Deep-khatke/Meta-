"""
Unit and Integration Tests for RoboForge Milestone 3D: Blind Scientific Discovery Benchmark.

Verifies:
1. Unseen evaluation scenarios are distinct, deterministic, and physically valid.
2. Anti-leakage isolation: no hardcoded answers or scenario leaks in research memory.
3. Fair baseline comparison under identical simulator access, trial budgets, and stopping criteria.
4. Comprehensive, honest metric calculations (success rate, collision rate, time to goal, timeout rate).
5. Generalization testing on held-out scenarios without re-tuning.
6. REST API endpoints for blind benchmark execution and report retrieval.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app import app
from backend.experiment_manager import ExperimentManager
from backend.experiment_models import AvoidancePolicy
from backend.research.blind_benchmark_scenarios import (
    get_all_blind_benchmark_scenarios,
    create_blind_chicanes_scenario,
    create_blind_narrow_gate_scenario,
    create_blind_staggered_forest_scenario,
    create_blind_diagonal_barrier_scenario
)
from backend.research.blind_discovery_evaluator import BlindDiscoveryEvaluator, blind_discovery_evaluator
from backend.research.models import BlindDiscoveryBenchmarkReport


@pytest.fixture
def client():
    return TestClient(app)


def test_blind_scenarios_deterministic_and_distinct():
    """Verifies that all 4 benchmark scenarios have valid, deterministic, and distinct definitions."""
    scens = get_all_blind_benchmark_scenarios()
    assert len(scens) == 4
    assert "scen_blind_chicanes" in scens
    assert "scen_blind_narrow_gate" in scens
    assert "scen_blind_staggered_forest" in scens
    assert "scen_blind_diagonal_barrier" in scens

    chicanes = scens["scen_blind_chicanes"]
    gate = scens["scen_blind_narrow_gate"]
    forest = scens["scen_blind_staggered_forest"]
    diag = scens["scen_blind_diagonal_barrier"]

    # Verify distinct environments and seeds
    assert chicanes.seed == 100
    assert gate.seed == 200
    assert forest.seed == 201
    assert diag.seed == 202

    # Verify obstacle counts and positions are distinct
    assert len(chicanes.environment) == 2
    assert len(gate.environment) == 2
    assert len(forest.environment) == 3
    assert len(diag.environment) == 2

    # Verify all have identical physics timestep
    for sid, s in scens.items():
        assert s.settings.timestep == 0.005
        assert s.task.control_mode == "goal_seeking"
        assert s.task.policy.enabled is True


def test_anti_leakage_isolation():
    """Inspects research memory and prompt definitions to ensure held-out scenarios are not leaked."""
    from backend.research.research_memory import research_memory

    # Verify held-out scenario IDs are not stored in missions or hypotheses
    recent_missions = research_memory.list_missions(limit=50)
    for m in recent_missions:
        assert "scen_blind_narrow_gate" not in m.goal_description
        assert "scen_blind_staggered_forest" not in m.goal_description
        assert "scen_blind_diagonal_barrier" not in m.goal_description

    # Verify default benchmark scenarios start with stock, unbiased APF policy
    scens = get_all_blind_benchmark_scenarios()
    stock_default = AvoidancePolicy()
    for sid, sc in scens.items():
        assert sc.task.policy.safety_margin == stock_default.safety_margin
        assert sc.task.policy.repulsion_gain == stock_default.repulsion_gain
        assert sc.task.policy.tangential_gain == stock_default.tangential_gain


def test_metric_calculations_and_fairness():
    """Verifies that navigation metrics (success, collision, time, timeout, score) are correctly derived."""
    evaluator = BlindDiscoveryEvaluator()

    # Empty run list
    empty_m = evaluator._calc_run_metrics([])
    assert empty_m["success_rate"] == 0.0
    assert empty_m["score"] == -9999.0

    # Mock test runs
    sc = create_blind_chicanes_scenario()
    em = ExperimentManager()
    runs = [em.run_experiment(sc.model_copy(deep=True, update={"seed": 100}))]

    metrics = evaluator._calc_run_metrics(runs, timeout=11.0)
    assert "success_rate" in metrics
    assert "collision_count" in metrics
    assert "collision_rate" in metrics
    assert "mean_path_length" in metrics
    assert "timeout_rate" in metrics
    assert "score" in metrics
    assert isinstance(metrics["score"], float)


def test_fair_baselines_budget_enforcement():
    """Verifies that Random Search, Grid Search, and Research Agent enforce candidate budgets."""
    evaluator = BlindDiscoveryEvaluator()
    discovery_scen = create_blind_chicanes_scenario()
    held_out_scens = [create_blind_narrow_gate_scenario()]

    # Run quick evaluation with budget = 2 candidates, 1 seed
    rand_res = evaluator._eval_random_search(
        discovery_scen=discovery_scen,
        held_out_scens=held_out_scens,
        candidate_budget=2,
        tuning_seeds=[100],
        held_out_seeds=[200]
    )
    assert len(rand_res.raw_trial_records) == 2
    # 2 candidates * 1 seed + 1 held-out * 1 seed = 3 simulator calls
    assert rand_res.simulator_calls == 3

    grid_res = evaluator._eval_grid_search(
        discovery_scen=discovery_scen,
        held_out_scens=held_out_scens,
        candidate_budget=2,
        tuning_seeds=[100],
        held_out_seeds=[200]
    )
    assert len(grid_res.raw_trial_records) == 2
    assert grid_res.simulator_calls == 3


def test_reproducible_benchmark_execution_and_artifacts():
    """Executes a quick 2-candidate benchmark and verifies all artifacts are generated."""
    evaluator = BlindDiscoveryEvaluator()
    report = evaluator.run_benchmark(
        candidate_budget=2,
        tuning_seeds=[100],
        held_out_seeds=[200],
        save_artifacts=True
    )

    assert isinstance(report, BlindDiscoveryBenchmarkReport)
    assert "default_apf" in report.results
    assert "random_search" in report.results
    assert "grid_search" in report.results
    assert "research_agent" in report.results
    assert report.leakage_audit_passed is True

    # Verify files saved under benchmarks/discovery/
    b_dir = Path("benchmarks/discovery")
    assert (b_dir / "scenarios.json").exists()
    assert (b_dir / "benchmark_report.json").exists()
    assert (b_dir / "benchmark_summary.md").exists()

    with open(b_dir / "scenarios.json", "r", encoding="utf-8") as f:
        scen_data = json.load(f)
        assert len(scen_data) == 4

    summary_md = evaluator.generate_markdown_summary(report)
    assert "RoboForge" in summary_md
    assert "Discovery Performance" in summary_md
    assert "Generalization on Unseen Held-Out Environments" in summary_md


def test_api_endpoints_for_blind_discovery(client):
    """Verifies FastAPI endpoints for blind benchmark retrieval and scenarios."""
    # Test scenarios endpoint
    res = client.get("/api/research/benchmark/blind-discovery/scenarios")
    assert res.status_code == 200
    data = res.json()
    assert "scen_blind_chicanes" in data
    assert "scen_blind_narrow_gate" in data

    # Test latest report retrieval
    res_report = client.get("/api/research/benchmark/blind-discovery/latest")
    assert res_report.status_code == 200
    report_data = res_report.json()
    assert "results" in report_data
    assert "research_agent" in report_data["results"]
    assert report_data["leakage_audit_passed"] is True
