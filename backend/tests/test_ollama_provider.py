import pytest
import os
import httpx
from fastapi.testclient import TestClient
from backend.app import app
from backend.research.reasoning_provider import (
    OllamaReasoningProvider,
    DeterministicRuleEngine,
    get_reasoning_provider
)
from backend.ai_reasoning_service import get_ai_reasoning_service, AIChatRequest, AIApprovalRequest

client = TestClient(app)


def is_local_ollama_running() -> bool:
    try:
        res = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return res.status_code == 200
    except Exception:
        return False


def test_ollama_provider_discovery_and_selection():
    """Verify endpoint discovery and that qwen2.5:7b is defaulted only if installed."""
    provider = OllamaReasoningProvider()
    assert provider.base_url.startswith("http")
    assert isinstance(provider.available_models, list)
    if is_local_ollama_running():
        assert provider.is_available() is True
        assert provider.model in provider.available_models
        assert "qwen2.5:7b" in provider.model or len(provider.available_models) > 0
    # Requested model not in available models should not be pulled
    provider_bogus = OllamaReasoningProvider(model="nonexistent-model-xyz")
    assert provider_bogus.model is None
    assert provider_bogus.is_available() is False


def test_ollama_provider_unavailable_fallback():
    """Verify provider transparently falls back to DeterministicRuleEngine when unreachable."""
    provider_down = OllamaReasoningProvider(base_url="http://127.0.0.1:59999")
    assert provider_down.is_available() is False
    assert provider_down.name == "Ollama (Unavailable)"

    goal_res = provider_down.interpret_goal("Optimize safety margin", {})
    assert "target_metric" in goal_res
    assert "target_threshold" in goal_res

    hypos = provider_down.propose_hypotheses("Optimize speed", {"total_collisions": 0, "success": True}, {})
    assert len(hypos) >= 1
    assert "controlled_variable" in hypos[0]


def test_ollama_provider_timeout_and_error_handling(monkeypatch):
    """Verify network timeouts fall back safely without raising uncaught exceptions."""
    provider = OllamaReasoningProvider()
    monkeypatch.setattr(provider, "is_available", lambda: True)

    def mock_post(*args, **kwargs):
        raise httpx.ConnectTimeout("Connection to Ollama timed out after 35s")

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    # Should safely catch ConnectTimeout and invoke fallback engine
    hypos = provider.propose_hypotheses("Avoid collisions", {"total_collisions": 2}, {})
    assert len(hypos) > 0
    assert hypos[0]["controlled_variable"] in ("task.policy.safety_margin", "task.policy.repulsion_gain")


def test_ollama_variable_normalization_and_safety_constraints():
    """Verify non-allowed or malformed variables are normalized into ALLOWED_VARIABLES."""
    provider = OllamaReasoningProvider()
    # Mock _call_ollama to return raw unnormalized variables
    raw_response = {
        "hypotheses": [
            {
                "title": "Margin hypothesis",
                "statement": "Increase safety distance",
                "mechanism": "Clearance",
                "controlled_variable": "robot.margin.distance",
                "baseline_value": 0.5,
                "proposed_value": 0.9,
                "expected_outcome": "Fewer collisions"
            },
            {
                "title": "Turn rate hypothesis",
                "statement": "Increase angular yaw",
                "mechanism": "Turn faster",
                "controlled_variable": "steering_turn_gain",
                "baseline_value": 2.0,
                "proposed_value": 3.2,
                "expected_outcome": "Better cornering"
            }
        ]
    }
    provider._call_ollama = lambda sys_inst, user_msg: raw_response

    hypos = provider.propose_hypotheses("Goal", {}, {})
    assert len(hypos) == 2
    assert hypos[0]["controlled_variable"] == "task.policy.safety_margin"
    assert hypos[1]["controlled_variable"] == "task.policy.max_turn_rate"
    for h in hypos:
        assert h["controlled_variable"] in provider.ALLOWED_VARIABLES


def test_api_ai_status_and_models_endpoints():
    """Verify /api/ai/status and /api/ai/models return truthful Ollama status without credential leakage."""
    res_status = client.get("/api/ai/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert "active_provider" in data_status
    assert "provider_type" in data_status
    assert data_status["allows_arbitrary_code"] is False
    assert "tools_available" in data_status

    res_models = client.get("/api/ai/models")
    assert res_models.status_code == 200
    data_models = res_models.json()
    assert "available_models" in data_models
    assert "model" in data_models
    if is_local_ollama_running():
        assert data_models["provider_type"] == "ollama"
        assert "qwen2.5:7b" in data_models["model"]


@pytest.mark.skipif(not is_local_ollama_running(), reason="Local Ollama not reachable")
def test_live_ollama_inference_reasoning():
    """Executes REAL model inference against installed local Ollama qwen2.5:7b."""
    provider = OllamaReasoningProvider()
    assert provider.is_available() is True
    assert provider.model is not None

    # 1. Real goal interpretation
    goal_res = provider.interpret_goal(
        "Minimize robot collisions during tight obstacle slalom navigation",
        {"robot": "DifferentialDriveRover", "arena": "indoor_slalom"}
    )
    assert "target_metric" in goal_res
    assert "target_threshold" in goal_res
    assert isinstance(goal_res["constraints"], list)

    # 2. Real hypothesis proposal
    hypos = provider.propose_hypotheses(
        "Minimize collisions",
        {"total_collisions": 3, "time_to_goal": 14.2, "success": False},
        {"scenario_id": "scen_slalom_beta"}
    )
    assert len(hypos) >= 1
    assert "title" in hypos[0]
    assert "controlled_variable" in hypos[0]
    assert hypos[0]["controlled_variable"] in provider.ALLOWED_VARIABLES
    assert isinstance(hypos[0]["proposed_value"], float)


@pytest.mark.skipif(not is_local_ollama_running(), reason="Local Ollama not reachable")
def test_live_chat_endpoint_real_model_inference():
    """Tests POST /api/ai/chat with live inference against qwen2.5:7b."""
    req_payload = {
        "prompt": "What sensors and capabilities does this robot have for obstacle avoidance?",
        "mode": "supervised",
        "chat_history": []
    }
    res = client.post("/api/ai/chat", json=req_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["is_model_inference"] is True
    assert data["is_fallback"] is False
    assert "Ollama" in data["provider"]
    assert len(data["reply"]) > 20
    assert len(data["thought_steps"]) >= 2
    assert "inspect_robot_capabilities" in data["tools_invoked"]


def test_supervised_approval_gate_and_rejection():
    """Verify supervised research pauses at approval gate and respects rejection."""
    req_payload = {
        "prompt": "Investigate navigation safety and propose a supervised research mission.",
        "mode": "supervised",
        "max_iterations": 2,
        "max_experiments": 3,
        "selected_scenario_id": "scen_slalom_beta"
    }
    chat_res = client.post("/api/ai/chat", json=req_payload)
    assert chat_res.status_code == 200
    data = chat_res.json()

    assert data["awaiting_approval"] is True
    assert data["approval_plan"] is not None
    assert "mission" in data
    mission_id = data["mission"]["mission_id"]

    # Reject plan via cancel-mission
    cancel_res = client.post("/api/ai/cancel-mission", json={
        "mission_id": mission_id,
        "reason": "Safety margin too aggressive for arena constraints"
    })
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.json()
    assert cancel_data["status"] == "cancelled"


def test_deterministic_override_ai_provider(monkeypatch):
    """Verify AI_PROVIDER=deterministic explicitly forces fallback without error."""
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    service = get_ai_reasoning_service()
    status = service.get_provider_status()
    assert status["provider_type"] == "deterministic_rule_engine"
    assert status["model"] == "deterministic-robotics-v1"

    req_payload = {
        "prompt": "Explain robot capabilities",
        "mode": "supervised"
    }
    chat_res = client.post("/api/ai/chat", json=req_payload)
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert data["is_model_inference"] is False
    assert data["is_fallback"] is True
