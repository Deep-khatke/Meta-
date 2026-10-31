"""Unit and integration test suite for Live Visual Drone RL Training (Isaac Lab Mode)."""

import os
import time
import pytest
from backend.models import Project, SceneObject
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.rl_training_manager import training_manager, MODELS_DIR
from backend.ai_reasoning_service import AIReasoningService


def test_live_interactive_drone_curriculum_execution():
    """Verify that live visual Isaac Lab RL training executes, animates failures & convergence, and saves PPO model."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_drone_training(
        engine=engine,
        drone_identifier="drone",
        target_altitude=1.5,
        total_episodes=4
    )

    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None
    assert engine.rl_training_active is True

    # Monitor training progression through episodes
    max_wait = 18.0
    start = time.time()
    observed_episodes = set()
    saw_crash = False

    while engine.rl_training_active and (time.time() - start) < max_wait:
        st = engine.get_state()
        if st.rl_training_episode > 0:
            observed_episodes.add(st.rl_training_episode)
        if st.rl_training_status and ("CRASH" in st.rl_training_status or "TILT" in st.rl_training_status):
            saw_crash = True
        time.sleep(0.1)

    # Verify that curriculum completed
    assert not engine.rl_training_active, "Training did not complete within allotted timeout"
    final_st = engine.get_state()

    # Check observed learning behavior
    assert len(observed_episodes) >= 2, f"Expected multiple learning episodes, observed {observed_episodes}"
    assert saw_crash is True, "Expected visual crash in Episode 1"

    # Check final state and PPO model storage
    assert final_st.rl_policy_active is True
    assert final_st.active_rl_policy_name is not None
    assert os.path.exists(MODELS_DIR / f"{job_id}.zip"), f"PPO checkpoint not found at {job_id}.zip"
    assert os.path.exists(MODELS_DIR / f"{job_id}.json"), f"PPO metadata JSON not found at {job_id}.json"

    # Check that model can be loaded by training manager
    loaded_model = training_manager.load_model(job_id)
    assert loaded_model is not None


def test_ai_agent_recognizes_live_drone_training_prompt():
    """Verify AI Assistant maps user request to train_drone_live_interactive with full deep-thinking."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    svc = AIReasoningService(project_manager=pm, engine=engine)

    prompt = (
        "i want to see it happen live in the scene where a drone is learning "
        "then failing again then trying again and after at the end it finally gets it "
        "and that model gets stored in ppo like it happens in isac sim and lab"
    )

    from backend.rl_models import AIChatRequest

    resp = svc.chat(AIChatRequest(prompt=prompt))
    assert resp.reply is not None
    assert any("train_drone_live_interactive" in t for t in resp.tools_invoked)
    assert len(resp.thought_steps) >= 2
    assert "Isaac Lab" in resp.reply or "PPO" in resp.reply

