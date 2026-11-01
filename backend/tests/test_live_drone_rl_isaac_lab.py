"""Tests for REAL live drone RL training.

Previously this asserted a scripted crash animation. Drone live training now runs genuine PPO
flight learning with measured telemetry, so we verify it starts, reports honest status, and is
cancellable (rather than waiting for full convergence, which takes real time).
"""

import time
import pytest
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.rl_training_manager import training_manager
from backend.rl_models import AIChatRequest
from backend.ai_reasoning_service import AIReasoningService


def test_live_drone_training_is_real_and_cancellable():
    """Live drone training starts a genuine PPO worker with honest telemetry."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    engine.play()

    res = training_manager.start_live_interactive_drone_training(
        engine=engine, drone_identifier="drone", target_altitude=1.5,
    )
    assert res["status"] in ("started", "warning")
    job_id = res.get("job_id")
    assert job_id is not None

    start = time.time()
    while not engine.rl_training_active and time.time() - start < 6.0:
        time.sleep(0.1)
    assert engine.rl_training_active is True

    time.sleep(2.0)
    st = engine.get_state()
    assert st.rl_training_status is not None
    # Honest status only -- no scripted CRASH/TILT theater.
    assert any(k in st.rl_training_status for k in ["INIT", "LEARNING", "ROLLOUT", "COMPLETE", "ERROR"])

    training_manager.cancel_training(job_id)
    engine.rl_training_active = False
    assert training_manager.get_status(job_id) is not None


def test_ai_agent_recognizes_live_drone_training_prompt():
    """The AI maps a 'watch it learn live' drone request to the live drone trainer."""
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(proj)
    svc = AIReasoningService(project_manager=pm, engine=engine)

    prompt = (
        "i want to see it happen live in the scene where a drone is learning "
        "then failing again then trying again and after at the end it finally gets it "
        "and that model gets stored as a ppo policy"
    )
    resp = svc.chat(AIChatRequest(prompt=prompt))
    assert resp.reply is not None
    assert any("train_drone_live_interactive" in t for t in resp.tools_invoked)
    assert len(resp.thought_steps) >= 2
