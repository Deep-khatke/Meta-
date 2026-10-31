"""Comprehensive integration test suite for Live Visual Multi-Robot RL Training across Drones, Arms, and Rovers (Isaac Lab Mode)."""

import os
import time
import pytest
from backend.models import Project, SceneObject
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.rl_training_manager import training_manager, MODELS_DIR
from backend.ai_reasoning_service import AIReasoningService
from backend.rl_models import AIChatRequest


def test_ai_agent_recognizes_all_robot_training_prompts():
    """Verify AI Assistant maps user requests across all robotics paradigms to their respective live interactive curricula."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    svc = AIReasoningService(project_manager=pm, engine=engine)

    # 1. Robotic Arm / Pick & Place
    arm_prompts = [
        "create an arm and make it pick an object",
        "an arm and make it pick a object",
        "train arm to pick",
        "make the arm pick an object"
    ]
    for prompt in arm_prompts:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert resp.reply is not None
        assert "train_arm_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"
        assert len(resp.thought_steps) >= 2
        assert "Arm" in resp.reply or "Pick" in resp.reply

    # 2. Ground Rover / Navigation & Obstacle Avoidance
    rover_prompts = [
        "create a robot and make it train",
        "train rover",
        "make the robot train",
        "train the robot"
    ]
    for prompt in rover_prompts:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert resp.reply is not None
        assert "train_rover_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"
        assert len(resp.thought_steps) >= 2
        assert "Rover" in resp.reply or "Ground" in resp.reply or "Navigation" in resp.reply

    # 3. Aerial Drone / Hover & Attitude
    drone_prompts = [
        "create a drone and make it train",
        "train drone live",
        "drone is learning then failing again then trying again and finally gets it"
    ]
    for prompt in drone_prompts:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert resp.reply is not None
        assert "train_drone_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"
        assert len(resp.thought_steps) >= 2
        assert "Drone" in resp.reply or "Quadcopter" in resp.reply or "Hover" in resp.reply


def test_live_arm_curriculum_execution():
    """Verify that live visual Isaac Lab Robotic Arm Pick & Place executes, animates joint slip & lift, and saves PPO model."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_arm_training(
        engine=engine,
        arm_identifier="arm",
        total_episodes=4
    )

    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None
    assert engine.rl_training_active is True

    # Monitor progression
    max_wait = 18.0
    start = time.time()
    observed_episodes = set()
    saw_slip_or_overshoot = False

    while engine.rl_training_active and (time.time() - start) < max_wait:
        st = engine.get_state()
        if st.rl_training_episode > 0:
            observed_episodes.add(st.rl_training_episode)
        if st.rl_training_status and any(w in st.rl_training_status for w in ["FAIL", "SLIP", "OVERSHOOT"]):
            saw_slip_or_overshoot = True
        time.sleep(0.08)

    assert not engine.rl_training_active, "Arm training did not complete within timeout"
    assert len(observed_episodes) >= 2, f"Expected multiple episodes, got {observed_episodes}"
    assert saw_slip_or_overshoot is True, "Expected visual overshoot/slip failure state"

    # Verify PPO model files
    assert os.path.exists(MODELS_DIR / f"{job_id}.zip")
    assert os.path.exists(MODELS_DIR / f"{job_id}.json")

    # Verify model is loadable
    loaded = training_manager.load_model(job_id)
    assert loaded is not None


def test_live_rover_curriculum_execution():
    """Verify that live visual Isaac Lab Ground Rover Navigation executes, animates collisions & avoidance, and saves PPO model."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_rover_training(
        engine=engine,
        robot_identifier="robot",
        total_episodes=4
    )

    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None
    assert engine.rl_training_active is True

    # Monitor progression
    max_wait = 18.0
    start = time.time()
    observed_episodes = set()
    saw_crash_or_retry = False

    while engine.rl_training_active and (time.time() - start) < max_wait:
        st = engine.get_state()
        if st.rl_training_episode > 0:
            observed_episodes.add(st.rl_training_episode)
        if st.rl_training_status and any(w in st.rl_training_status for w in ["CRASH", "RETRY", "IMPACT"]):
            saw_crash_or_retry = True
        time.sleep(0.08)

    assert not engine.rl_training_active, "Rover training did not complete within timeout"
    assert len(observed_episodes) >= 2, f"Expected multiple episodes, got {observed_episodes}"
    assert saw_crash_or_retry is True, "Expected visual obstacle collision failure state"

    # Verify PPO model files
    assert os.path.exists(MODELS_DIR / f"{job_id}.zip")
    assert os.path.exists(MODELS_DIR / f"{job_id}.json")

    # Verify model is loadable
    loaded = training_manager.load_model(job_id)
    assert loaded is not None

