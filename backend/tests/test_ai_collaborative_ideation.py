"""
Tests for Anti-Meta AI Collaborative Ideation, Natural Conversational Intelligence,
and Clean Product-Ready Behavior.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.ai_reasoning_service import get_ai_reasoning_service, AIReasoningService
from backend.rl_models import AIChatRequest


@pytest.fixture
def reasoning_service(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return get_ai_reasoning_service()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return TestClient(app)


def test_brainstorming_ideas_does_not_spawn_obstacles(reasoning_service: AIReasoningService):
    """Verify brainstorming and ideation returns articulate engineering ideas without creating fake obstacles."""
    req = AIChatRequest(
        prompt="Can you brainstorm some ideas for our autonomous robotics project?",
        mode="supervised"
    )
    res = reasoning_service.chat(req)

    assert res is not None
    assert "Collaborative Project Ideation" in res.reply or "Ideation" in res.reply
    assert "Autonomous Dynamic Warehouse Scout" in res.reply or "Hybrid Classical-RL" in res.reply
    # Ensure no fake suggested configuration or obstacles were dumped into the scene
    assert res.suggested_configuration is None
    assert len(res.actions_executed or []) == 0
    # Thought steps should be transparent and clean
    assert any("Ideation" in s.title or "Collaborative" in s.title for s in res.thought_steps)


def test_control_theory_comparison_does_not_spawn_obstacles(reasoning_service: AIReasoningService):
    """Verify comparing MPC vs APF vs RL returns comprehensive technical analysis without obstacles."""
    req = AIChatRequest(
        prompt="What are the trade-offs between Model Predictive Control and Artificial Potential Fields for obstacle avoidance?",
        mode="supervised"
    )
    res = reasoning_service.chat(req)

    assert res is not None
    assert "Artificial Potential Fields (APF)" in res.reply
    assert "Model Predictive Control (MPC)" in res.reply
    assert "Local Minima" in res.reply or "vortex" in res.reply
    assert res.suggested_configuration is None
    assert len(res.actions_executed or []) == 0


def test_question_about_differential_robot_does_not_spawn_robot(reasoning_service: AIReasoningService):
    """Verify asking an informational question about differential robots does NOT accidentally spawn a robot entity."""
    proj = reasoning_service.pm.get_current()
    initial_obj_count = len(proj.objects)

    req = AIChatRequest(
        prompt="What is a differential robot and how does it steer?",
        mode="supervised"
    )
    res = reasoning_service.chat(req)

    assert res is not None
    # No action should have been executed to add a robot
    assert not any(a.get("action") == "add_robot" for a in (res.actions_executed or []))
    assert len(proj.objects) == initial_obj_count


def test_question_about_drone_sensors_does_not_spawn_drone(reasoning_service: AIReasoningService):
    """Verify asking an informational question about drones does NOT accidentally spawn a drone entity."""
    proj = reasoning_service.pm.get_current()
    initial_obj_count = len(proj.objects)

    req = AIChatRequest(
        prompt="Can drones use 2D LiDAR for altitude stabilization?",
        mode="supervised"
    )
    res = reasoning_service.chat(req)

    assert res is not None
    assert not any(a.get("action") == "add_drone" for a in (res.actions_executed or []))
    assert len(proj.objects) == initial_obj_count


def test_explicit_layout_request_still_synthesizes_obstacles(reasoning_service: AIReasoningService):
    """Verify that when an engineer explicitly asks to create an obstacle course, it does synthesize validated layout."""
    req = AIChatRequest(
        prompt="Create an obstacle course challenge in the arena",
        mode="supervised"
    )
    res = reasoning_service.chat(req)

    assert res is not None
    assert (res.suggested_configuration is not None) or (len(res.actions_executed or []) > 0)


def test_general_thinking_together_conversation(client: TestClient):
    """Verify conversational Q&A via API endpoint returns clean, natural dialogue."""
    res = client.post("/api/ai/chat", json={
        "prompt": "How can we think together about optimizing our robot for narrow aisles?",
        "mode": "supervised"
    })
    assert res.status_code == 200
    data = res.json()

    assert "reply" in data
    assert len(data["reply"]) > 50
    assert "Thought Title" not in data["reply"]
    assert "Thought Detail" not in data["reply"]
    assert data["suggested_configuration"] is None
