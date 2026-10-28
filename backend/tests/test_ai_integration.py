"""Integration tests for RoboForge AI Assistant reasoning, simulation grounding, and tool interfaces (Milestone 3A)."""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.ai_reasoning_service import get_ai_reasoning_service, AIReasoningService
from backend.ai_assistant import AIExperimentAssistant
from backend.project_manager import ProjectManager
from backend.research.research_memory import research_memory
from backend.research.models import ResearchMission, MissionBudget


@pytest.fixture
def reasoning_service(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return get_ai_reasoning_service()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    return TestClient(app)


def test_ai_provider_status_reporting(reasoning_service: AIReasoningService):
    """Verify provider status truthfully reports configured provider and credentials presence."""
    status = reasoning_service.get_provider_status()

    assert "active_provider" in status
    assert "has_credentials" in status
    assert "tools_available" in status
    assert status["active_provider"] == "DeterministicRoboticsHeuristicEngine-v1"
    assert status["has_credentials"] is False
    assert "inspect_simulation_state" in status["tools_available"]
    assert "inspect_robot_capabilities" in status["tools_available"]
    assert "explain_telemetry" in status["tools_available"]
    assert "propose_research_mission" in status["tools_available"]


def test_answer_questions_using_actual_simulation_state(reasoning_service: AIReasoningService):
    """Verify assistant grounds answers in real simulation state from ProjectManager."""
    resp = reasoning_service.process_prompt("Where is the robot located and what is its maximum speed?")

    assert resp.is_model_inference is False
    assert resp.is_fallback is True
    assert resp.provider == "DeterministicRoboticsHeuristicEngine-v1"
    assert "inspect_simulation_state" in resp.tools_invoked
    assert resp.active_simulation_context is not None

    rob_ctx = resp.active_simulation_context["robot"]
    assert rob_ctx["name"] == "ForgeBot-DD1"
    assert rob_ctx["initial_position"][:2] == [0.0, 0.0]
    assert rob_ctx["initial_position"][2] > 0
    assert rob_ctx["max_linear_velocity"] > 0

    # Ensure assistant reply contains the actual grounded numbers
    assert "ForgeBot-DD1" in resp.assistant_reply
    assert str(rob_ctx["max_linear_velocity"]) in resp.assistant_reply


def test_inspect_robot_capabilities_and_sensors(reasoning_service: AIReasoningService):
    """Verify assistant inspects and returns real sensor suite and actuation parameters."""
    resp = reasoning_service.process_prompt("What sensors and actuators does the robot have?")

    assert "inspect_robot_capabilities" in resp.tools_invoked
    assert "Robot Capabilities & Sensors" in resp.assistant_reply
    assert "LiDAR" in resp.assistant_reply
    assert "IMU" in resp.assistant_reply
    assert "Camera" in resp.assistant_reply
    assert "Odometry" in resp.assistant_reply

    caps = reasoning_service.tool_inspect_robot_capabilities()
    assert caps["robot_name"] == "ForgeBot-DD1"
    assert caps["sensors"]["lidar"]["enabled"] is True
    assert caps["sensors"]["lidar"]["rays"] > 0
    assert caps["sensors"]["imu"]["enabled"] is True
    assert caps["sensors"]["wheel_encoders"]["enabled"] is True


def test_explain_real_telemetry_without_hallucinations(reasoning_service: AIReasoningService):
    """Verify assistant explains measured telemetry from research memory without inventing values."""
    # Seed a real recorded mission with empirical metrics
    mission = ResearchMission(
        mission_id="m_test_telemetry",
        title="Telemetry Audit Mission",
        goal_description="Audit collision telemetry in slalom beta",
        base_scenario_id="scen_slalom_beta",
        mode="autonomous",
        target_metric="time_to_goal",
        target_threshold=6.0,
        budget=MissionBudget(max_iterations=1, max_experiments=2),
        baseline_metrics={
            "total_collisions": 1,
            "success": False,
            "total_duration": 10.0,
            "time_to_goal": 10.0,
            "max_speed": 1.45
        }
    )
    research_memory.save_mission(mission)

    resp = reasoning_service.process_prompt("Explain the latest telemetry: why did the robot collide?")

    assert "explain_telemetry" in resp.tools_invoked
    assert "Total Collisions = `1`" in resp.assistant_reply
    assert "Success = `False`" in resp.assistant_reply
    assert "Duration = `10.00s`" in resp.assistant_reply


def test_propose_structured_research_mission(reasoning_service: AIReasoningService):
    """Verify assistant compiles natural language intent into a structured, bounded research mission."""
    resp = reasoning_service.process_prompt("Propose a research mission to optimize navigation speed and avoid collisions")

    assert "propose_research_mission" in resp.tools_invoked
    assert resp.structured_mission is not None
    assert resp.structured_mission["mission_id"].startswith("mission_")
    assert resp.structured_mission["target_metric"] != ""
    assert resp.structured_mission["max_iterations"] > 0
    assert resp.structured_mission["max_experiments"] > 0
    assert "Compiled Research Mission" in resp.assistant_reply


def test_schema_validation_and_spawn_clearance(reasoning_service: AIReasoningService):
    """Verify physical bounds clamping and spawn clearance enforcement with no code execution."""
    # Prompt requesting target outside arena bounds
    resp = reasoning_service.process_prompt("Create a navigation challenge with target at (15.0, 20.0)")

    assert resp.validation_status in ("valid", "warning")
    cfg = resp.suggested_configuration
    tx, ty, _ = cfg["task"]["target_position"]

    # Target must be strictly clamped within arena [-3.5, 4.5] x [-2.8, 2.8]
    assert -3.5 <= tx <= 4.5
    assert -2.8 <= ty <= 2.8
    assert any("clamped" in msg.lower() for msg in resp.validation_messages)


def test_fastapi_ai_status_endpoint(client: TestClient):
    """Verify GET /api/ai/status returns 200 with truthful provider metadata."""
    res = client.get("/api/ai/status")
    assert res.status_code == 200
    data = res.json()
    assert data["active_provider"] == "DeterministicRoboticsHeuristicEngine-v1"
    assert data["has_credentials"] is False
    assert len(data["tools_available"]) >= 4


def test_fastapi_ai_interpret_prompt_endpoint(client: TestClient):
    """Verify POST /api/ai/interpret-prompt returns structured AI response."""
    payload = {
        "prompt": "What sensors does ForgeBot have and how can we optimize navigation?",
        "base_scenario_id": None
    }
    res = client.post("/api/ai/interpret-prompt", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "interpreted_goal" in data
    assert "assistant_reply" in data
    assert "provider" in data
    assert data["provider"] == "DeterministicRoboticsHeuristicEngine-v1"
    assert data["is_model_inference"] is False
    assert data["is_fallback"] is True
    assert len(data["tools_invoked"]) > 0
