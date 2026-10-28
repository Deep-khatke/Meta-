"""Tests for RoboForge Milestone 3B: Autonomous Robotics AI Chat & Provider Integration."""

import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
import httpx

from backend.app import app
from backend.ai_reasoning_service import get_ai_reasoning_service, AIReasoningService
from backend.research.reasoning_provider import LanguageModelReasoningProvider
from backend.research.research_memory import research_memory
from backend.research.models import ResearchMission, MissionBudget, ControlledExperimentPlan, ExperimentOutcome
from backend.rl_models import AIChatRequest, AIApprovalRequest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return TestClient(app)


@pytest.fixture
def reasoning_service(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return get_ai_reasoning_service()


# ==========================================
# 1. Chat Autonomous Mode
# ==========================================

def test_ai_chat_autonomous_mission_execution(client: TestClient):
    """Verify autonomous mode plans, runs bounded trials, and returns measured deltas."""
    req_payload = {
        "prompt": "Run an autonomous research mission to optimize navigation speed and minimize transit time",
        "mode": "autonomous",
        "max_iterations": 2,
        "max_experiments": 4,
    }

    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert "reply" in data
    assert data["mode"] == "autonomous"
    assert data["awaiting_approval"] is False
    assert len(data["thought_steps"]) >= 3

    # Verify thought trace steps
    step_types = [s["step"] for s in data["thought_steps"]]
    assert "inspect" in step_types
    assert "plan" in step_types
    assert "execute" in step_types

    # Verify mission object was generated and tracked
    assert data["mission"] is not None
    assert "mission_id" in data["mission"]
    assert data["mission"]["status"] in ["completed", "running", "paused"]
    assert "propose_research_mission" in data["tools_invoked"]
    assert "execute_research_mission" in data["tools_invoked"]


# ==========================================
# 2. Chat Supervised Mode & Approval Gate
# ==========================================

def test_ai_chat_supervised_mission_approval_gate(client: TestClient):
    """Verify supervised mode pauses with awaiting_approval=True and returns approval_plan."""
    req_payload = {
        "prompt": "Propose a supervised research mission to improve safety margin and avoid obstacles",
        "mode": "supervised",
        "max_iterations": 2,
        "max_experiments": 4,
    }

    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["awaiting_approval"] is True
    assert data["approval_plan"] is not None
    assert "parameter_modifications" in data["approval_plan"]
    assert "Supervisor Approval Gate" in [s["title"] for s in data["thought_steps"]]
    assert "Approve & Execute" in data["reply"] or "Approval Required" in data["reply"]


def test_ai_approve_action_positive(client: TestClient):
    """Verify approving a paused plan executes trials and returns empirical deltas."""
    # First, formulate a supervised mission
    req_payload = {
        "prompt": "Propose a supervised research mission for collision avoidance",
        "mode": "supervised",
    }
    chat_res = client.post("/api/ai/chat", json=req_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["awaiting_approval"] is True
    mission_id = chat_data["mission"]["mission_id"]

    # Now approve the action
    approval_payload = {
        "mission_id": mission_id,
        "approved": True,
        "reason": "Supervisor confirmed experiment safety parameters",
    }
    app_res = client.post("/api/ai/approve-action", json=approval_payload)
    assert app_res.status_code == 200
    app_data = app_res.json()

    assert app_data["awaiting_approval"] is False
    assert "Approved & Executed" in app_data["reply"]
    assert any(s["step"] == "approve" for s in app_data["thought_steps"])
    assert any(s["step"] == "execute" for s in app_data["thought_steps"])
    assert app_data["mission"]["budget"]["used_experiments"] > 0


def test_ai_approve_action_rejection(client: TestClient):
    """Verify rejecting a paused plan cancels execution and runs 0 trials."""
    # First, formulate a supervised mission
    req_payload = {
        "prompt": "Propose a supervised research mission to test aggressive speed",
        "mode": "supervised",
    }
    chat_res = client.post("/api/ai/chat", json=req_payload)
    mission_id = chat_res.json()["mission"]["mission_id"]

    # Reject the action
    approval_payload = {
        "mission_id": mission_id,
        "approved": False,
        "reason": "Speed is excessively risky in crowded arena",
    }
    app_res = client.post("/api/ai/approve-action", json=approval_payload)
    assert app_res.status_code == 200
    app_data = app_res.json()

    assert app_data["awaiting_approval"] is False
    assert "Plan Execution Cancelled" in app_data["reply"]
    assert app_data["mission"]["status"] == "cancelled"
    assert any(s["step"] == "reject" for s in app_data["thought_steps"])


def test_ai_cancel_mission_endpoint(client: TestClient):
    """Verify /api/ai/cancel-mission terminates an ongoing mission."""
    # Create mission
    req_payload = {
        "prompt": "Propose a supervised research mission for obstacle clearance",
        "mode": "supervised",
    }
    chat_res = client.post("/api/ai/chat", json=req_payload)
    mission_id = chat_res.json()["mission"]["mission_id"]

    cancel_res = client.post("/api/ai/cancel-mission", json={
        "mission_id": mission_id,
        "reason": "Aborted by supervisor test"
    })
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.json()
    assert cancel_data["status"] == "cancelled"
    assert cancel_data["mission_id"] == mission_id


# ==========================================
# 3. Grounded Robotics Queries
# ==========================================

def test_ai_chat_robot_capabilities_query(client: TestClient):
    """Verify capabilities query returns actual hardware specs and LiDAR/IMU details."""
    req_payload = {
        "prompt": "Inspect this robot and explain its capabilities and sensors",
    }
    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert "Robot Capabilities" in data["reply"]
    assert "ForgeBot-DD1" in data["reply"]
    assert "LiDAR" in data["reply"]
    assert "max_linear_velocity" in data["observed_facts"]
    assert data["observed_facts"]["sensors"]["lidar"]["enabled"] is True


def test_ai_chat_failure_diagnosis_query(client: TestClient):
    """Verify failure diagnosis query grounds explanation in empirical run telemetry."""
    # Seed a mission with failure telemetry
    mission = ResearchMission(
        mission_id="m_diag_test",
        title="Diagnostic Failure Test",
        goal_description="Test diagnostic telemetry grounding",
        base_scenario_id="scen_slalom_beta",
        mode="supervised",
        target_metric="collision_rate",
        target_threshold=0.0,
        budget=MissionBudget(max_iterations=1, max_experiments=1),
        baseline_metrics={
            "total_collisions": 3,
            "success": False,
            "total_duration": 12.0,
            "time_to_goal": 12.0,
            "max_speed": 1.2
        }
    )
    research_memory.save_mission(mission)

    req_payload = {
        "prompt": "Diagnose why the robot collided or failed in recent runs",
    }
    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert "Telemetry Diagnosis Report" in data["reply"]
    assert "3" in data["reply"]  # Measured collisions
    assert "explain_telemetry" in data["tools_invoked"]


def test_ai_chat_candidate_deltas_explanation(client: TestClient):
    """Verify candidate delta explanation summarizes baseline vs candidate outcomes."""
    # Seed a mission with an outcome
    mission = ResearchMission(
        mission_id="m_delta_test",
        title="Delta Comparison Test",
        goal_description="Test candidate comparison explanation",
        base_scenario_id="scen_slalom_beta",
        mode="autonomous",
        target_metric="collision_rate",
        target_threshold=0.0,
        budget=MissionBudget(max_iterations=1, max_experiments=2),
        outcomes=[
            ExperimentOutcome(
                plan_id="plan_delta_1",
                hypothesis_id="hyp_delta_1",
                empirical_verdict="improved",
                baseline_deltas={"delta_collisions": -2, "delta_duration": -1.4},
                verdict_explanation="Increasing repulsion gain by 0.6 successfully reduced collisions from 2 to 0 with faster corner clearance."
            )
        ]
    )
    research_memory.save_mission(mission)

    req_payload = {
        "prompt": "Explain why the candidate improved or degraded compared to baseline",
    }
    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert "Candidate Outcome Delta Analysis" in data["reply"]
    assert "IMPROVED" in data["reply"]
    assert "-2" in data["reply"]
    assert "repulsion gain" in data["reply"]


# ==========================================
# 4. Real Language Model Provider Mock Tests
# ==========================================

def test_gemini_provider_mock_live_inference(reasoning_service: AIReasoningService):
    """Verify Gemini REST call, JSON schema parsing, boundary clamping, and safety checks."""
    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "interpreted_goal": "Gemini: Synthesize wide obstacle corridor",
                                "assistant_reply": "I have used Gemini 2.0 Flash to design a high-speed corridor challenge with clamped boundary safety.",
                                "layout": {
                                    "target_position": [3.2, 0.0, 0.25],
                                    "timeout": 14.0,
                                    "obstacles": [
                                        {"name": "Left Pylon", "type": "cylinder", "position": [1.5, 0.8, 0.3], "size": [0.2, 0.3]},
                                        {"name": "Right Pylon", "type": "cylinder", "position": [1.5, -0.8, 0.3], "size": [0.2, 0.3]},
                                    ]
                                },
                                "suggested_actions": ["Execute in 3D Viewport", "Train PPO Policy"]
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_response
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test_mock_gemini_key", "GEMINI_MODEL": "gemini-2.0-flash"}, clear=True):
        with patch("httpx.Client.post", return_value=mock_resp):
            provider_status = reasoning_service.get_provider_status()
            assert provider_status["has_credentials"] is True
            assert provider_status["active_provider"] == "Google Gemini (gemini-2.0-flash)"

            # Call process_prompt
            res = reasoning_service.process_prompt("Design a high-speed corridor challenge")
            assert res.is_model_inference is True
            assert res.is_fallback is False
            assert res.provider == "Google Gemini (gemini-2.0-flash)"
            assert "Gemini 2.0 Flash" in res.assistant_reply
            assert len(res.suggested_configuration["environment"]) == 2
            assert res.validation_status == "valid"


def test_openai_provider_mock_live_inference(reasoning_service: AIReasoningService):
    """Verify OpenAI REST call when OPENAI_API_KEY is configured."""
    mock_openai_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "interpreted_goal": "OpenAI: Synthesize 3-cylinder slalom challenge",
                        "assistant_reply": "Synthesized a 3-cylinder slalom challenge using GPT-4o-mini with verified clearance.",
                        "layout": {
                            "target_position": [2.6, 0.5, 0.25],
                            "timeout": 12.0,
                            "obstacles": [
                                {"name": "Slalom 1", "type": "cylinder", "position": [1.2, 0.3, 0.3], "size": [0.2, 0.3]},
                                {"name": "Slalom 2", "type": "cylinder", "position": [1.8, -0.3, 0.3], "size": [0.2, 0.3]},
                            ]
                        },
                        "suggested_actions": ["Execute in 3D Viewport"]
                    })
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_openai_response
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test_mock_openai_key", "OPENAI_MODEL": "gpt-4o-mini"}, clear=True):
        with patch("httpx.Client.post", return_value=mock_resp):
            provider_status = reasoning_service.get_provider_status()
            assert provider_status["has_credentials"] is True
            assert provider_status["active_provider"] == "OpenAI (gpt-4o-mini)"

            res = reasoning_service.process_prompt("Create a slalom challenge")
            assert res.is_model_inference is True
            assert res.is_fallback is False
            assert "GPT-4o-mini" in res.assistant_reply


def test_provider_error_transparent_fallback(reasoning_service: AIReasoningService):
    """Verify that if model inference raises an HTTP error or timeout, deterministic fallback is safely invoked."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Internal Server Error", request=MagicMock(), response=mock_resp)

    with patch.dict("os.environ", {"GEMINI_API_KEY": "mock_failing_key"}):
        with patch("httpx.Client.post", side_effect=httpx.ConnectTimeout("Connection timed out")):
            # Must fall back gracefully to deterministic synthesis without crashing
            res = reasoning_service.process_prompt("Create a slalom course")
            assert res.is_model_inference is False
            assert res.is_fallback is True
            assert res.provider == "DeterministicRoboticsHeuristicEngine-v1"
            assert res.validation_status == "valid"
            assert len(res.suggested_configuration["environment"]) >= 2
            assert res.suggested_configuration["task"]["target_position"] is not None


def test_offline_mode_without_credentials_reports_honestly(client: TestClient):
    """Verify /api/ai/status honestly reports offline fallback when no keys are configured."""
    with patch.dict("os.environ", {"OLLAMA_DISABLED": "1"}, clear=True):
        res = client.get("/api/ai/status")
        assert res.status_code == 200
        data = res.json()
        assert data["has_credentials"] is False
        assert data["active_provider"] == "DeterministicRoboticsHeuristicEngine-v1"
        assert data["offline_mode"] is True


def test_ai_chat_conversational_greeting_natural(client: TestClient):
    """Verify casual greetings like 'hey' receive an articulate, natural, non-robotic response without leaked thoughts."""
    for greeting in ["hey", "hello", "hi there", "who are you"]:
        res = client.post("/api/ai/chat", json={"prompt": greeting})
        assert res.status_code == 200
        data = res.json()
        assert "RoboForge AI" in data["reply"]
        assert "Thought Title" not in data["reply"]
        assert "Thought Detail" not in data["reply"]
        assert "## Intent" not in data["reply"]
        assert len(data["thought_steps"]) == 0  # No cluttered trace on simple greeting


def test_ai_chat_cleans_leaked_thought_headers(reasoning_service: AIReasoningService):
    """Verify any leaked thought headers in model json output are stripped cleanly."""
    mock_ollama_resp = MagicMock()
    mock_ollama_resp.status_code = 200
    mock_ollama_resp.json.return_value = {
        "message": {
            "content": json.dumps({
                "reply": "Ready to simulate!\n\n## Thought Title: Internal\n## Thought Detail: Secret\n## Intent: research_mission\n## Suggested Actions:\n- Do X\n\nLet's begin.",
                "thought_title": "Internal",
                "thought_detail": "Secret",
                "intent": "general_qa",
                "suggested_actions": ["Do X"]
            })
        }
    }

    with patch("httpx.Client.post", return_value=mock_ollama_resp):
        res = reasoning_service._call_model_chat_reasoning(
            prompt="What should we do?",
            context={},
            provider_info={"provider_type": "ollama", "endpoint": "http://127.0.0.1:11434", "model": "qwen2.5:7b"}
        )
        assert res is not None
        assert "Thought Title" not in res["reply"]
        assert "Thought Detail" not in res["reply"]
        assert "## Intent" not in res["reply"]
        assert "Ready to simulate!" in res["reply"]
        assert "Let's begin." in res["reply"]


def test_ai_chat_add_differential_robot(reasoning_service: AIReasoningService):
    """Verify requesting another differential robot directly spawns ForgeBot-DD2 in the project."""
    req = AIChatRequest(
        prompt="can you add another differential robot",
        mode="autonomous"
    )
    res = reasoning_service.chat(req)
    assert res is not None
    assert any(a["action"] == "add_robot" for a in (res.actions_executed or []))
    assert "ForgeBot-DD2" in res.reply
    # Verify in active project
    proj = reasoning_service.pm.get_current()
    added_robots = [o for o in proj.objects if "forgebot" in o.name.lower() or "robot" in o.name.lower()]
    assert len(added_robots) >= 1


def test_ai_chat_contextual_followup_robot_spawning(reasoning_service: AIReasoningService):
    """Verify contextual follow-ups like 'same as the first one' or 'no add them' correctly resolve from history."""
    history = [
        {"role": "user", "content": "can you add another differential robot"},
        {"role": "assistant", "content": "Let's name it ForgeBot-DD2. What settings would you like?"}
    ]
    req = AIChatRequest(
        prompt="same as the first one",
        mode="autonomous",
        chat_history=history
    )
    res = reasoning_service.chat(req)
    assert res is not None
    assert any(a["action"] == "add_robot" for a in (res.actions_executed or []))
    assert "ForgeBot-DD2" in res.reply


def test_chat_memory_store_and_endpoints(client):
    """Verify chat memory persists messages and history endpoints operate properly."""
    import time
    from backend.chat_memory import chat_memory
    test_session = f"test_session_{int(time.time())}"
    chat_memory.clear_history(test_session)

    chat_memory.append_message(test_session, "user", "Hello robot")
    chat_memory.append_message(test_session, "assistant", "Hello human!")

    hist = chat_memory.get_history(test_session)
    assert len(hist) == 2
    assert hist[0]["content"] == "Hello robot"
    assert hist[1]["content"] == "Hello human!"

    res = client.get(f"/api/ai/chat/history?session_id={test_session}")
    assert res.status_code == 200
    assert len(res.json()["history"]) == 2

    del_res = client.delete(f"/api/ai/chat/history?session_id={test_session}")
    assert del_res.status_code == 200
    assert len(chat_memory.get_history(test_session)) == 0


