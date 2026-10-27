import pytest
import math
from backend.models import Project, SceneObject, RobotConfig, SimulationSettings
from backend.simulation import MuJoCoEngine, quat_to_forward, euler_deg_to_quat_str
from backend.project_manager import get_default_project

def test_engine_initialization():
    project = get_default_project()
    engine = MuJoCoEngine(project)
    assert engine.model is not None
    assert engine.data is not None
    assert engine.status == "stopped"
    assert engine.sim_time == 0.0

def test_single_step():
    project = get_default_project()
    engine = MuJoCoEngine(project)
    initial_time = engine.sim_time
    engine.step()
    assert engine.sim_time > initial_time
    assert engine.step_count == 1

def test_teleoperation_movement():
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    # Send forward linear command
    engine.set_teleop(linear=1.0, angular=0.0)
    
    initial_x = engine.data.qpos[0]
    # Step simulation 100 times (approx 0.5s)
    for _ in range(100):
        engine.step()
        
    final_x = engine.data.qpos[0]
    # Robot should have moved forward in X
    assert final_x > initial_x + 0.05
    state = engine.get_state()
    assert state.robot.linear_speed > 0.01

def test_teleoperation_turn():
    project = get_default_project()
    engine = MuJoCoEngine(project)
    
    # Send pure angular rotation
    engine.set_teleop(linear=0.0, angular=1.0)
    
    for _ in range(60):
        engine.step()
        
    state = engine.get_state()
    assert abs(state.robot.angular_velocity[2]) > 0.1

def test_collision_with_obstacle():
    project = get_default_project()
    # Place a dynamic box directly in front of robot
    project.objects = [
        SceneObject(
            id="test_box",
            name="Blocker Box",
            type="box",
            position=[0.5, 0.0, 0.1],
            size=[0.1, 0.1, 0.1],
            mass=1.0,
            is_static=False
        )
    ]
    engine = MuJoCoEngine(project)
    engine.set_teleop(linear=1.5, angular=0.0)
    
    # Drive forward into obstacle
    detected_contacts = []
    for _ in range(250):
        engine.step()
        st = engine.get_state()
        if len(st.contacts) > 0:
            for c in st.contacts:
                if "geom_test_box" in (c.geom1, c.geom2):
                    detected_contacts.append(c)
            
    assert len(detected_contacts) > 0, "Contact between robot and obstacle box must be detected"

def test_reset_simulation():
    project = get_default_project()
    engine = MuJoCoEngine(project)
    engine.set_teleop(linear=1.0, angular=0.0)
    for _ in range(50):
        engine.step()
    
    assert engine.sim_time > 0
    engine.reset()
    assert engine.sim_time == 0.0
    assert engine.step_count == 0
    assert engine.status == "stopped"

def test_quat_to_forward():
    # Identity quat [1, 0, 0, 0] forward is [1, 0, 0]
    fwd = quat_to_forward([1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(fwd[0], 0.01) == 1.0
    assert pytest.approx(fwd[1], 0.01) == 0.0
