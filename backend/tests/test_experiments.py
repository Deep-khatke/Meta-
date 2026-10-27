import pytest
import tempfile
import json
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

from backend.models import SCHEMA_VERSION, SceneObject
from backend.experiment_models import (
    ExperimentConfig, TaskObjective, ScriptedCommand,
    SimulatedMeasurement, DerivedMetrics, ExperimentRunResult, RunComparison
)
from backend.experiment_manager import ExperimentManager
from backend.app import app

client = TestClient(app)

# =====================================================================
# 1. EXPERIMENT CONFIGURATION & VERSIONING
# =====================================================================

def test_experiment_config_versioning():
    """Verify experiment configurations have versioned schemas and sensible defaults."""
    config = ExperimentConfig(
        name="Test Experiment",
        seed=123
    )
    assert config.schema_version == SCHEMA_VERSION
    assert config.seed == 123
    assert config.duration == 8.0
    assert len(config.task.target_position) == 3

def test_invalid_task_objective():
    """Verify invalid task objective boundaries raise validation errors."""
    with pytest.raises(ValueError):
        # 2 coordinates instead of 3
        TaskObjective(target_position=[1.0, 2.0])

    with pytest.raises(ValueError):
        # negative distance threshold
        TaskObjective(distance_threshold=-0.1)

# =====================================================================
# 2. SCENARIO MANAGEMENT (Save, Duplicate, Load, Reset, Delete)
# =====================================================================

def test_scenario_crud_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        em = ExperimentManager(base_dir=Path(tmpdir))
        
        # List initial default scenarios
        scenarios = em.list_scenarios()
        assert len(scenarios) >= 2
        
        # Create new scenario
        custom = ExperimentConfig(
            id="scen_custom_1",
            name="Custom Obstacle Field",
            seed=77,
            environment=[
                SceneObject(id="box_test", name="Box", type="box", position=[1.5, 0.0, 0.2], size=[0.2, 0.2, 0.2])
            ],
            task=TaskObjective(target_position=[2.0, 1.0, 0.25], timeout=6.0)
        )
        saved_path = em.save_scenario(custom)
        assert Path(saved_path).exists()
        
        # Get by ID
        fetched = em.get_scenario("scen_custom_1")
        assert fetched is not None
        assert fetched.name == "Custom Obstacle Field"
        assert fetched.seed == 77
        assert len(fetched.environment) == 1
        
        # Duplicate
        duplicated = em.duplicate_scenario("scen_custom_1", new_name="Duplicated Field")
        assert duplicated.id != "scen_custom_1"
        assert duplicated.name == "Duplicated Field"
        assert duplicated.seed == 77
        
        # Delete original
        deleted = em.delete_scenario("scen_custom_1")
        assert deleted is True
        assert em.get_scenario("scen_custom_1") is None

# =====================================================================
# 3. REPRODUCIBILITY (Seeded Execution Determinism)
# =====================================================================

def test_seeded_execution_reproducibility():
    """Verify that re-running identical configurations with the same seed produces equivalent results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        em = ExperimentManager(base_dir=Path(tmpdir))
        scen = em.list_scenarios()[0]
        
        # Run 1 with seed=42
        run1 = em.run_experiment(scen, seed_override=42, duration_override=3.0)
        # Run 2 with seed=42
        run2 = em.run_experiment(scen, seed_override=42, duration_override=3.0)
        
        # Check measurements count matches
        assert len(run1.measurements) == len(run2.measurements)
        
        # Check positions match step-for-step within strict numerical tolerance (< 1e-4)
        for m1, m2 in zip(run1.measurements, run2.measurements):
            assert pytest.approx(m1.sim_time, 1e-4) == m2.sim_time
            assert pytest.approx(m1.position[0], 1e-3) == m2.position[0]
            assert pytest.approx(m1.position[1], 1e-3) == m2.position[1]
            assert pytest.approx(m1.linear_speed, 1e-3) == m2.linear_speed
            
        # Check derived metrics match
        assert pytest.approx(run1.metrics.total_distance_traveled, 1e-3) == run2.metrics.total_distance_traveled
        assert pytest.approx(run1.metrics.final_distance_to_target, 1e-3) == run2.metrics.final_distance_to_target
        assert run1.metrics.success == run2.metrics.success

# =====================================================================
# 4. METRICS COLLECTION & DISTINCTION
# =====================================================================

def test_metrics_collection_and_distinction():
    """Verify simulated raw measurements vs derived summary metrics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        em = ExperimentManager(base_dir=Path(tmpdir))
        scen = em.list_scenarios()[0]
        
        res = em.run_experiment(scen, duration_override=2.0)
        
        # Raw simulated measurements
        assert len(res.measurements) > 5
        first_m = res.measurements[0]
        assert hasattr(first_m, "position")
        assert hasattr(first_m, "linear_velocity")
        assert hasattr(first_m, "contacts_count")
        assert first_m.sim_time >= 0.0
        
        # Derived metrics
        m = res.metrics
        assert m.total_duration > 0.0
        assert m.total_distance_traveled >= 0.0
        assert m.initial_distance_to_target > 0.0
        assert m.final_distance_to_target >= 0.0
        assert m.min_distance_to_target <= m.initial_distance_to_target
        assert m.max_linear_speed >= m.mean_linear_speed

# =====================================================================
# 5. PERSISTENCE & RESTARTS
# =====================================================================

def test_result_persistence_survives_restart():
    """Verify saving run result to disk and retrieving it with a new manager instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        em1 = ExperimentManager(base_dir=Path(tmpdir))
        scen = em1.list_scenarios()[0]
        run_res = em1.run_experiment(scen, duration_override=1.5)
        run_id = run_res.run_id
        
        # Instantiate fresh manager to simulate restart
        em2 = ExperimentManager(base_dir=Path(tmpdir))
        runs_list = em2.list_runs()
        assert any(r["run_id"] == run_id for r in runs_list)
        
        loaded_run = em2.get_run_result(run_id)
        assert loaded_run is not None
        assert loaded_run.run_id == run_id
        assert loaded_run.experiment_name == scen.name
        assert len(loaded_run.measurements) == len(run_res.measurements)
        assert loaded_run.metrics.total_distance_traveled == run_res.metrics.total_distance_traveled

# =====================================================================
# 6. RUN COMPARISON
# =====================================================================

def test_run_comparison_deltas():
    """Verify run comparison produces metric deltas and summary between two runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        em = ExperimentManager(base_dir=Path(tmpdir))
        scen = em.list_scenarios()[0]
        
        # Run A: short run (0.8s, before reaching goal)
        run_a = em.run_experiment(scen, seed_override=1, duration_override=0.8)
        # Run B: longer run (3.0s, reaching goal)
        run_b = em.run_experiment(scen, seed_override=2, duration_override=3.0)
        
        comp = em.compare_runs(run_a.run_id, run_b.run_id)
        assert comp.run_a_id == run_a.run_id
        assert comp.run_b_id == run_b.run_id
        assert "duration_delta" in comp.metric_deltas
        assert "final_distance_delta" in comp.metric_deltas
        assert comp.metric_deltas["duration_delta"] > 0.4
        assert comp.metric_deltas["final_distance_delta"] < 0  # Run B got closer to goal
        assert len(comp.summary) > 10

# =====================================================================
# 7. REST API ENDPOINTS
# =====================================================================

def test_api_scenario_endpoints():
    # List scenarios
    res = client.get("/api/experiments/scenarios")
    assert res.status_code == 200
    scenarios = res.json()["scenarios"]
    assert len(scenarios) >= 2
    scen_id = scenarios[0]["id"]
    
    # Get scenario
    get_res = client.get(f"/api/experiments/scenarios/{scen_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == scen_id
    
    # Run experiment via API
    run_res = client.post("/api/experiments/run", json={
        "scenario_id": scen_id,
        "seed": 42,
        "duration": 2.0
    })
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert "run_id" in run_data
    assert "measurements" in run_data
    assert "metrics" in run_data
    
    # List runs
    runs_res = client.get("/api/experiments/runs")
    assert runs_res.status_code == 200
    assert any(r["run_id"] == run_data["run_id"] for r in runs_res.json()["runs"])
    
    # Load scenario to scene
    load_res = client.post(f"/api/experiments/scenarios/{scen_id}/load-to-scene")
    assert load_res.status_code == 200
    assert load_res.json()["status"] == "loaded"

def test_api_invalid_experiment_requests():
    # Nonexistent scenario
    res = client.post("/api/experiments/run", json={"scenario_id": "nonexistent_scenario_123"})
    assert res.status_code == 404
    
    # Compare nonexistent runs
    res_comp = client.post("/api/experiments/compare", json={"run_a_id": "fake_a", "run_b_id": "fake_b"})
    assert res_comp.status_code == 404

def test_policy_training_optimizes_avoidance():
    with tempfile.TemporaryDirectory() as tmpdir:
        em = ExperimentManager(base_dir=Path(tmpdir))
        scens = em.list_scenarios()
        slalom = next(s for s in scens if s.id == "scen_slalom_beta")
        train_res = em.train_avoidance_policy(slalom.id)
        assert len(train_res.iterations) == 5
        assert train_res.best_policy is not None
        assert train_res.scenario_id == slalom.id
        
        # Verify best policy was saved to scenario
        updated_scen = em.get_scenario(slalom.id)
        assert updated_scen.task.policy.safety_margin == train_res.best_policy.safety_margin

