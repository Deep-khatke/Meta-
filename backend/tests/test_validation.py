import pytest
from pydantic import ValidationError
from backend.models import SceneObject, RobotConfig, SimulationSettings, Project

def test_valid_scene_object():
    obj = SceneObject(
        id="box1",
        name="Valid Box",
        type="box",
        position=[1.0, 2.0, 0.5],
        rotation=[0.0, 0.0, 45.0],
        size=[0.2, 0.2, 0.2],
        mass=2.5
    )
    assert obj.name == "Valid Box"
    assert obj.type == "box"
    assert obj.mass == 2.5

def test_invalid_position_dimension():
    with pytest.raises(ValidationError):
        # Only 2 coordinates instead of 3
        SceneObject(name="Bad Pos", type="box", position=[1.0, 2.0])

def test_invalid_negative_size():
    with pytest.raises(ValidationError):
        SceneObject(name="Bad Size", type="box", size=[-0.1, 0.2, 0.3])

def test_invalid_zero_mass():
    with pytest.raises(ValidationError):
        SceneObject(name="Zero Mass", type="box", mass=0.0)

def test_robot_velocity_limits():
    cfg = RobotConfig(max_linear_velocity=2.5, max_angular_velocity=4.0)
    assert cfg.max_linear_velocity == 2.5
    
    with pytest.raises(ValidationError):
        # Negative limit
        RobotConfig(max_linear_velocity=-1.0)

def test_simulation_settings_bounds():
    s = SimulationSettings(timestep=0.005, speed_multiplier=2.0)
    assert s.timestep == 0.005
    
    with pytest.raises(ValidationError):
        # Timestep too large
        SimulationSettings(timestep=0.2)
        
    with pytest.raises(ValidationError):
        # Speed multiplier negative
        SimulationSettings(speed_multiplier=-1.0)
