"""
Tests for 6-DOF dynamic drone aerodynamic flight physics,
autonomous drone policy training & deployment, and deep multi-step agent thinking.
"""

import pytest
from starlette.testclient import TestClient
from backend.app import app
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.models import SceneObject


@pytest.fixture
def client():
    return TestClient(app)


def test_drone_flight_controller_and_physics():
    """Verifies 6-DOF aerodynamic force generation and altitude control."""
    pm = ProjectManager()
    proj = pm.get_current()
    
    # Add a dynamic drone
    drone_obj = SceneObject(
        id="drone_test_01",
        name="Drone-Alpha",
        type="box",
        position=[1.5, 0.5, 0.1],
        size=[0.25, 0.25, 0.08],
        mass=1.2,
        color="#38bdf8",
        is_static=False
    )
    # Remove any existing drone_test_01
    proj.objects = [o for o in proj.objects if o.id != "drone_test_01"]
    proj.objects.append(drone_obj)
    pm.update_project(proj)

    engine = MuJoCoEngine(proj)
    engine.rebuild_model()

    ctrl = engine._find_drone_controller("drone_test_01")
    assert ctrl is not None
    assert ctrl.name == "Drone-Alpha"
    assert ctrl.mass >= 0.4

    # Issue Takeoff command
    res_takeoff = engine.command_drone_takeoff("drone_test_01", altitude=1.8)
    assert res_takeoff["status"] == "success"
    assert ctrl.flight_mode == "takeoff"
    assert ctrl.target_altitude == 1.8
    assert ctrl.rotors_active is True

    # Step simulation and check that aerodynamic thrust is generated
    for _ in range(50):
        engine.step()

    # Verify altitude PID applied upward thrust in xfrc_applied
    drone_body_id = ctrl.body_id
    if drone_body_id is not None and engine.data is not None:
        fx, fy, fz, tx, ty, tz = engine.data.xfrc_applied[drone_body_id]
        assert fz > 0.0, f"Expected positive upward aerodynamic thrust, got fz={fz}"

    # Issue Landing command
    res_land = engine.command_drone_land("drone_test_01")
    assert res_land["status"] == "success"
    assert ctrl.flight_mode == "landing"


def test_deep_agent_thinking_and_drone_flow(client: TestClient):
    """Verifies multi-step transparent thinking and end-to-end drone workflow."""
    
    # 1. User says: "i want you to add a drone to the scene"
    res1 = client.post("/api/ai/chat", json={"prompt": "i want you to add a drone to the scene"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1.get("thought_steps", [])) >= 4
    assert "thought_process" in data1 and data1["thought_process"]
    assert "### 🧠 1. Intent Decomposition" in data1["thought_process"]
    assert "### 📐 2. Scene Telemetry" in data1["thought_process"]
    assert "### 🔬 5. Physics & Dynamic Stability" in data1["thought_process"]
    assert any("drone" in a.get("action", "") for a in data1.get("actions_executed", []))

    # 2. User says: "okay now i want you to train the drone with the rl policy and train it to take of and land correctly"
    res2 = client.post("/api/ai/chat", json={
        "prompt": "okay now i want you to train the drone with the rl policy and train it to take of and land correctly"
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2.get("thought_steps", [])) >= 4
    assert any(a.get("action") == "train_drone_policy" for a in data2.get("actions_executed", []))

    # 3. User says: "is the rl trainig done ?"
    res3 = client.post("/api/ai/chat", json={"prompt": "is the rl trainig done ?"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert any(a.get("action") == "get_rl_training_status" for a in data3.get("actions_executed", []))

    # 4. User says: "load it into the scene"
    res4 = client.post("/api/ai/chat", json={"prompt": "load it into the scene"})
    assert res4.status_code == 200
    data4 = res4.json()
    assert any(a.get("action") == "deploy_drone_policy" for a in data4.get("actions_executed", []))

    # 5. User says: "I want you to start the simulation for it"
    res5 = client.post("/api/ai/chat", json={"prompt": "I want you to start the simulation for it"})
    assert res5.status_code == 200
    data5 = res5.json()
    actions5 = [a.get("action") for a in data5.get("actions_executed", [])]
    assert any(a in ("simulation_control", "simulation_play") for a in actions5)
    assert "control_drone" in actions5
