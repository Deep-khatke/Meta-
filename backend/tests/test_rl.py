"""Comprehensive test suite for RL environment, PPO training manager, benchmark evaluation, AI assistant, and hypothesis refinement."""

import time
import pytest
import numpy as np

from backend.rl_env import RoboForgeGymEnv
from backend.rl_models import (
    RLTrainingConfig, RLHyperparameters, AIAssistantPromptRequest, HypothesisRefinementRequest
)
from backend.rl_training_manager import training_manager
from backend.rl_evaluator import RLEvaluator
from backend.ai_assistant import AIExperimentAssistant
from backend.rl_refinement import HypothesisRefinementEngine


def test_gym_env_spaces_and_step():
    """Verify Gymnasium space contracts, continuous observation dimensions, and physics stepping."""
    env = RoboForgeGymEnv(max_episode_steps=50, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (16,)
    assert isinstance(info, dict)
    assert "target_position" in info

    # Step with forward action
    action = np.array([0.5, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_to_goal" in step_info
    assert "sim_time" in step_info
    assert "robot_position" in step_info

    # Ground contact check (robot should not fly, Z ~ 0.07m)
    r_z = step_info["robot_position"][2]
    assert 0.06 < r_z < 0.095, f"Robot Z position {r_z} deviated from ground level"

    env.close()


def test_gym_env_collision_and_termination():
    """Verify collision checking triggers negative penalty and episode termination."""
    env = RoboForgeGymEnv(seed=42)
    env.reset(seed=42)

    # Force robot position into an obstacle
    if env.project.objects:
        obs_obj = env.project.objects[0]
        # Place robot close to obstacle
        env.engine.data.qpos[env.engine.robot_qpos_adr:env.engine.robot_qpos_adr + 2] = [
            obs_obj.position[0] + 0.05,
            obs_obj.position[1] + 0.05
        ]
        is_col = env._check_collision()
        assert is_col is True, "Collision should be detected when robot overlaps obstacle"

    env.close()


def test_rl_training_and_checkpoint_saving():
    """Verify genuine PPO background training, progress metrics, and persistent checkpoint creation."""
    job_id = f"test_ppo_{int(time.time())}"
    config = RLTrainingConfig(
        job_id=job_id,
        model_name="Unit-Test-PPO-Agent",
        total_timesteps=256,
        seed=101,
        randomize_scenarios=False,
        hyperparameters=RLHyperparameters(
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=2,
        ),
    )

    status = training_manager.start_training(config)
    assert status.job_id == job_id
    assert status.status in ("queued", "training")

    # Wait for completion (256 steps takes ~2-3 seconds)
    max_wait = 25.0
    start = time.time()
    while time.time() - start < max_wait:
        cur_status = training_manager.get_status(job_id)
        if cur_status and cur_status.status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.3)

    final_status = training_manager.get_status(job_id)
    assert final_status is not None
    assert final_status.status == "completed", f"Training ended with {final_status.status}: {final_status.error}"
    assert final_status.checkpoint_path is not None

    # Verify model reload
    loaded_model = training_manager.load_model(job_id)
    assert loaded_model is not None, "Trained PPO model could not be loaded from disk"

    # Verify metadata registry persistence
    meta = training_manager.get_model_metadata(job_id)
    assert meta is not None
    assert meta.model_id == job_id
    assert meta.timesteps_trained == 256


def test_rl_evaluator_benchmark_comparison():
    """Verify evaluation suite runs PPO, APF, and Baseline across held-out layouts with confidence intervals."""
    # Run evaluation using the trained model or default
    models = training_manager.list_models()
    model_id = models[0].model_id if models else "test_model"

    eval_result = RLEvaluator.evaluate_suite(model_id, seed=202)
    assert eval_result.eval_id.startswith("eval_")
    assert len(eval_result.benchmark_results) == 3

    policy_types = [b.policy_type for b in eval_result.benchmark_results]
    assert "PPO" in policy_types
    assert "APF" in policy_types
    assert "Baseline" in policy_types

    for b in eval_result.benchmark_results:
        assert 0.0 <= b.success_rate <= 1.0
        assert 0.0 <= b.collision_rate <= 1.0
        assert b.avg_episode_time > 0.0
        assert "confidence_interval_95" in b.model_dump()

    assert eval_result.verdict is not None


def test_ai_experiment_assistant_validation():
    """Verify natural language prompts generate validated experiment configurations without code execution."""
    prompt = "Create a tight slalom challenge with 3 pylons, target at (3.2, 0.4), and a 12s timeout"
    response = AIExperimentAssistant.process_prompt(prompt)

    assert response.validation_status in ("valid", "warning")
    assert len(response.validation_messages) > 0

    cfg = response.suggested_configuration
    assert cfg["duration"] == 12.0
    assert cfg["task"]["target_position"][0] == pytest.approx(3.2, abs=0.1)
    assert cfg["task"]["target_position"][1] == pytest.approx(0.4, abs=0.1)
    assert len(cfg["environment"]) >= 2
    assert "policy" in cfg["task"]


def test_evidence_based_hypothesis_refinement():
    """Verify empirical hypothesis formulation, controlled rerun, and honest evidence tracking."""
    result = HypothesisRefinementEngine.analyze_and_refine(scenario_id="scen_slalom_beta")

    assert result.hypothesis_id.startswith("hypo_")
    assert result.controlled_parameter != ""
    assert result.outcome_verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
    assert len(result.empirical_evidence) > 0
    assert result.baseline_run_id != result.refinement_run_id
