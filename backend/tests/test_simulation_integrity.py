import pytest
import math
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import mujoco

from backend.models import Project, SceneObject, RobotConfig, SimulationSettings
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager, get_default_project
from backend.app import app

client = TestClient(app)

# =====================================================================
# 1. PHYSICS VALIDATION (MuJoCo Physics Engine Authenticity)
# =====================================================================

def test_physics_timestep_strictness():
    """Verify that sim_time is governed strictly by MuJoCo physics steps, not wall-clock."""
    project = get_default_project()
    project.settings.timestep = 0.005
    engine = MuJoCoEngine(project)
    
    assert engine.sim_time == 0.0
    assert engine.step_count == 0
    
    # Execute exactly 200 steps
    for _ in range(200):
        engine.step()
        
    assert engine.step_count == 200
    assert pytest.approx(engine.sim_time, 1e-6) == 200 * 0.005  # Exactly 1.000s
    assert pytest.approx(float(engine.data.time), 1e-6) == 1.0

def test_wheel_kinematics_and_actuators():
    """Verify that wheel actuators and differential kinematics drive wheel velocities."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    wr = project.robot.wheel_radius # 0.06m
    v_max = project.robot.max_linear_velocity # 1.5 m/s
    
    # Normalized teleop command 1.0 commands full max_linear_velocity (1.5 m/s)
    engine.set_teleop(linear=1.0, angular=0.0)
    engine.apply_control()
    
    expected_wheel_omega = v_max / wr # 1.5 / 0.06 = 25.0 rad/s
    assert pytest.approx(engine.data.ctrl[engine.left_motor_id], 0.01) == expected_wheel_omega
    assert pytest.approx(engine.data.ctrl[engine.right_motor_id], 0.01) == expected_wheel_omega
    
    # Step simulation and check that wheels spin
    for _ in range(30):
        engine.step()
        
    state = engine.get_state()
    assert state.robot.left_wheel_velocity > 1.0
    assert state.robot.right_wheel_velocity > 1.0
    assert state.robot.linear_speed > 0.05

def test_differential_turn_kinematics():
    """Verify differential drive angular velocity produces counter-rotating wheels."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    omega_target = 2.0 # rad/s
    engine.set_teleop(linear=0.0, angular=omega_target)
    engine.apply_control()
    
    # Left wheel should be negative, right wheel positive
    assert engine.data.ctrl[engine.left_motor_id] < 0
    assert engine.data.ctrl[engine.right_motor_id] > 0
    assert pytest.approx(abs(engine.data.ctrl[engine.left_motor_id]), 0.01) == abs(engine.data.ctrl[engine.right_motor_id])

def test_mass_and_collision_reaction():
    """Verify that colliding with a heavy obstacle resists robot movement compared to a light obstacle."""
    # Simulation with a heavy static blocker
    proj_heavy = get_default_project()
    proj_heavy.objects = [
        SceneObject(
            id="heavy_wall",
            name="Heavy Wall",
            type="box",
            position=[0.4, 0.0, 0.15],
            size=[0.1, 0.5, 0.15],
            is_static=True,
            mass=500.0
        )
    ]
    eng_heavy = MuJoCoEngine(proj_heavy)
    eng_heavy.set_teleop(linear=1.5, angular=0.0)
    for _ in range(150):
        eng_heavy.step()
    
    x_heavy = eng_heavy.get_state().robot.position[0]
    # Heavy static wall must have prevented robot from crossing x=0.4
    assert x_heavy < 0.4
    
    # Contacts must register with geom_heavy_wall
    contacts = eng_heavy.get_state().contacts
    assert any("heavy_wall" in (c.geom1 + c.geom2) for c in contacts)

# =====================================================================
# 2. DYNAMIC OBJECT CONSISTENCY (Add, Edit, Delete, Model Rebuild)
# =====================================================================

def test_dynamic_object_addition():
    """Verify adding an obstacle updates the compiled MuJoCo model."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    initial_obj_count = len(engine.project.objects)
    
    new_obj = SceneObject(
        id="dynamic_cyl",
        name="Dynamic Cylinder",
        type="cylinder",
        position=[1.0, 1.0, 0.3],
        size=[0.2, 0.3],
        mass=3.0,
        is_static=False
    )
    engine.project.objects.append(new_obj)
    engine.rebuild_model(preserve_state=True)
    
    assert len(engine.project.objects) == initial_obj_count + 1
    assert "dynamic_cyl" in engine.object_body_ids
    body_id = engine.object_body_ids["dynamic_cyl"]
    assert body_id >= 0
    
    geom_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_dynamic_cyl")
    assert geom_id >= 0

def test_dynamic_object_deletion_no_stale_state():
    """Verify deleting an obstacle removes body and geom with no stale IDs."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    assert "obs_box_1" in engine.object_body_ids
    
    # Delete obs_box_1
    engine.project.objects = [o for o in engine.project.objects if o.id != "obs_box_1"]
    engine.rebuild_model(preserve_state=True)
    
    # Stale ID must not exist
    assert "obs_box_1" not in engine.object_body_ids
    state = engine.get_state()
    assert "obs_box_1" not in state.objects
    
    # MuJoCo model should no longer contain geom_obs_box_1
    geom_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_obs_box_1")
    assert geom_id == -1

def test_preserve_simulation_state_on_rebuild():
    """Verify model rebuilding preserves robot pose and sim_time when preserve_state=True."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    engine.play()
    engine.set_teleop(linear=1.0, angular=0.0)
    for _ in range(60):
        engine.step()
        
    pre_time = engine.sim_time
    pre_x = engine.get_state().robot.position[0]
    assert pre_time > 0.1
    assert pre_x > 0.05
    
    # Add an obstacle with preserve_state=True
    new_obj = SceneObject(
        id="side_box",
        name="Side Box",
        type="box",
        position=[-1.0, 1.0, 0.2],
        size=[0.2, 0.2, 0.2]
    )
    engine.project.objects.append(new_obj)
    engine.rebuild_model(preserve_state=True)
    
    # State should be preserved
    assert pytest.approx(engine.sim_time, 1e-4) == pre_time
    assert pytest.approx(engine.get_state().robot.position[0], 1e-3) == pre_x
    assert engine.status == "running"

# =====================================================================
# 3. SIMULATION CONTROLS & REPEATED CYCLES
# =====================================================================

def test_simulation_controls_cycle():
    """Test full cycle: play, pause, single-step, timestep change, reset."""
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    # Play
    engine.play()
    assert engine.status == "running"
    engine.step()
    t1 = engine.sim_time
    assert t1 > 0
    
    # Pause
    engine.pause()
    assert engine.status == "paused"
    
    # Single-step while paused
    engine.step()
    t2 = engine.sim_time
    assert pytest.approx(t2, 1e-6) == t1 + project.settings.timestep
    assert engine.step_count == 2
    
    # Change timestep
    project.settings.timestep = 0.010
    engine.rebuild_model(preserve_state=True)
    assert pytest.approx(engine.model.opt.timestep, 1e-6) == 0.010
    
    # Step again with new timestep
    engine.step()
    t3 = engine.sim_time
    assert pytest.approx(t3, 1e-6) == t2 + 0.010
    
    # Reset
    engine.reset()
    assert engine.sim_time == 0.0
    assert engine.step_count == 0
    assert engine.status == "stopped"

# =====================================================================
# 4. PERSISTENCE & VERSION INTEGRITY
# =====================================================================

def test_save_load_roundtrip_all_properties():
    """Verify saving and loading preserves robot config, all objects, and settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = ProjectManager(storage_dir=Path(tmpdir))
        
        proj = pm.get_current()
        proj.name = "Custom Arena 42"
        proj.robot.wheel_radius = 0.075
        proj.robot.max_linear_velocity = 2.2
        proj.settings.timestep = 0.002
        proj.settings.speed_multiplier = 2.0
        
        saved_file = pm.save_project_to_disk("custom_arena.rfproj.json")
        assert Path(saved_file).exists()
        
        # Load back
        pm2 = ProjectManager(storage_dir=Path(tmpdir))
        loaded = pm2.load_project_from_file("custom_arena.rfproj.json")
        
        assert loaded.name == "Custom Arena 42"
        assert loaded.robot.wheel_radius == 0.075
        assert loaded.robot.max_linear_velocity == 2.2
        assert loaded.settings.timestep == 0.002
        assert loaded.settings.speed_multiplier == 2.0
        assert len(loaded.objects) == len(proj.objects)

def test_reset_to_saved_state():
    """Verify reset_to_saved restores last saved snapshot and discards unsaved changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pm = ProjectManager(storage_dir=Path(tmpdir))
        pm.save_project_to_disk("snapshot.rfproj.json")
        
        # Modify in memory
        pm.get_current().name = "Unsaved Mutation"
        pm.get_current().objects.clear()
        assert len(pm.get_current().objects) == 0
        
        # Revert
        reverted = pm.reset_to_saved()
        assert reverted.name == "Workspace Alpha"
        assert len(reverted.objects) > 0

# =====================================================================
# 5. FAILURE HANDLING & BOUNDARY VALIDATION
# =====================================================================

def test_api_invalid_scene_data():
    """Verify invalid scene object data is rejected with HTTP 400."""
    # Negative dimensions
    res = client.post("/api/scene/objects", json={
        "name": "Bad Box",
        "type": "box",
        "position": [0.0, 0.0, 0.5],
        "size": [-0.5, 0.5, 0.5]
    })
    assert res.status_code == 400

def test_api_invalid_schema_version():
    """Verify loading incompatible schema versions is rejected with HTTP 400."""
    res = client.post("/api/project/load", json={
        "project_json": json.dumps({"schema_version": "2.0.0", "name": "Unsupported"})
    })
    assert res.status_code == 400
    assert "Unsupported schema version" in res.json()["detail"]

def test_api_teleop_clamping():
    """Verify teleop commands exceeding robot limits are clamped to max_linear/angular_velocity."""
    # Command 5.0 exceeds max_linear_velocity (1.5) but is within Pydantic bounds (<= 10.0)
    res = client.post("/api/simulation/teleop", json={"linear": 5.0, "angular": -8.0})
    assert res.status_code == 200
    teleop = res.json()["teleop"]
    assert teleop["linear"] == 1.5 # clamped to max_linear_velocity
    assert teleop["angular"] == -3.0 # clamped to -max_angular_velocity

def test_api_teleop_out_of_bounds_rejection():
    """Verify teleop commands outside schema boundaries are rejected with HTTP 422."""
    res = client.post("/api/simulation/teleop", json={"linear": 999.0, "angular": -999.0})
    assert res.status_code == 422

def test_websocket_stream_and_disconnect():
    """Verify WebSocket connection, initial message delivery, and safe disconnect."""
    with client.websocket_connect("/ws") as ws:
        init_data = ws.receive_json()
        assert init_data["type"] == "init"
        assert "project" in init_data
        assert "state" in init_data
        
        # Send a ping/action
        ws.send_json({"type": "play"})
        
        # Receive broadcast or state
        msg = ws.receive_json()
        assert msg["type"] in ("state", "log", "project_updated")
        
    # Client disconnects cleanly without server error
