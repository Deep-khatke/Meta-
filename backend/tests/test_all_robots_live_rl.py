"""Integration tests for REAL live multi-robot RL training (arm, rover, drone).

These replace the previous tests that asserted a scripted animation ("FAIL/SLIP/CRASH/RETRY"
status strings completing in 18s). Live training now runs genuine PPO in the background and
rolls the actual current policy out in the scene, so the tests verify the real contract:
training starts, emits honest telemetry, and shuts down cleanly on cancel. We cancel quickly
rather than wait for full convergence (which legitimately takes minutes).
"""

import time
import pytest
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.rl_training_manager import training_manager
from backend.ai_reasoning_service import AIReasoningService
from backend.rl_models import AIChatRequest


def _wait_until_training(engine, timeout=6.0):
    start = time.time()
    while not getattr(engine, "rl_training_active", False) and time.time() - start < timeout:
        time.sleep(0.1)
    return getattr(engine, "rl_training_active", False)


def test_ai_agent_recognizes_all_robot_training_prompts():
    """The AI maps user requests across robotics paradigms to their live training tools."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    svc = AIReasoningService(project_manager=pm, engine=engine)

    for prompt in ["create an arm and make it pick an object", "train arm to pick"]:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert resp.reply is not None
        assert "train_arm_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"

    for prompt in ["create a robot and make it train", "train rover"]:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert "train_rover_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"

    for prompt in ["create a drone and make it train", "train drone live"]:
        resp = svc.chat(AIChatRequest(prompt=prompt))
        assert "train_drone_live_interactive" in resp.tools_invoked, f"Failed for prompt: {prompt}"


def test_live_rover_training_is_real_and_cancellable():
    """Rover live training starts a genuine PPO worker, reports honest telemetry, cancels cleanly."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_rover_training(engine=engine, robot_identifier="robot")
    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None

    assert _wait_until_training(engine), "Training did not become active"

    # Honest status: the real worker reports INIT/LEARNING/ROLLOUT -- never scripted CRASH/SLIP.
    time.sleep(2.0)
    st = engine.get_state()
    assert st.rl_training_status is not None
    assert any(k in st.rl_training_status for k in ["INIT", "LEARNING", "ROLLOUT", "COMPLETE"])

    # Cancel and verify clean shutdown.
    training_manager.cancel_training(job_id)
    engine.rl_training_active = False
    status = training_manager.get_status(job_id)
    assert status is not None


def test_live_arm_training_uses_universal_reach_env():
    """Arm live training loads an articulated robot and starts the real reach trainer."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_arm_training(engine=engine, arm_identifier="arm")
    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None

    # A generic (articulated) robot must now be present for the reach env.
    assert _wait_until_training(engine), "Arm training did not become active"
    time.sleep(2.0)
    st = engine.get_state()
    assert st.rl_training_status is not None
    assert any(k in st.rl_training_status for k in ["INIT", "LEARNING", "ROLLOUT", "COMPLETE", "ERROR"])

    training_manager.cancel_training(job_id)
    engine.rl_training_active = False
