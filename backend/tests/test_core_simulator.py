import pytest
import numpy as np
import mujoco

from backend.models import (
    Project, SceneObject, RobotConfig, SimulationSettings,
    GenericRobotModel, LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial
)
from backend.simulation import MuJoCoEngine
from backend.project_manager import get_default_project, ProjectManager
from backend.urdf_importer import parse_urdf_string

SAMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.4 0.4 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1.0"/>
      </material>
    </visual>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.08"/>
    </inertial>
  </link>

  <link name="arm_link">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </visual>
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.005"/>
    </inertial>
  </link>

  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
  </joint>
</robot>
"""

SAMPLE_URDF_WITH_UNSUPPORTED = """<?xml version="1.0"?>
<robot name="mesh_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://robot/meshes/base.stl"/>
      </geometry>
    </visual>
  </link>
  <transmission name="trans_1">
    <type>transmission_interface/SimpleTransmission</type>
  </transmission>
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>
</robot>
"""

def test_urdf_importer_valid():
    res = parse_urdf_string(SAMPLE_URDF)
    assert res.status == "success"
    assert res.total_links == 2
    assert res.total_joints == 1
    assert res.robot is not None
    assert res.robot.name == "test_arm"
    assert res.robot.root_link == "base_link"
    assert len(res.robot.joints) == 1
    assert res.robot.joints[0].name == "shoulder_joint"
    assert res.robot.joints[0].type == "revolute"
    assert res.robot.joints[0].limit.lower == pytest.approx(-1.57)

def test_urdf_importer_reports_unsupported():
    res = parse_urdf_string(SAMPLE_URDF_WITH_UNSUPPORTED)
    assert res.status == "warning"
    assert len(res.warnings) >= 2
    warning_texts = " ".join(res.warnings)
    assert "mesh" in warning_texts.lower()
    assert "transmission" in warning_texts.lower()

def test_simulation_lidar_and_imu():
    proj = get_default_project()
    proj.sensors.lidar.enabled = True
    proj.sensors.lidar.num_beams = 16
    proj.sensors.lidar.max_range = 8.0
    
    # Place an obstacle directly in front of the robot at x = 1.0
    proj.objects = [
        SceneObject(
            id="front_wall",
            name="Front Wall",
            type="box",
            position=[1.0, 0.0, 0.25],
            size=[0.1, 1.0, 0.5],
            is_static=True
        )
    ]
    
    engine = MuJoCoEngine(proj)
    state = engine.get_state()
    
    assert len(state.lidar_readings) == 16
    assert len(state.lidar_angles) == 16
    # Front-facing beams should hit the wall before max_range
    min_dist = min(state.lidar_readings)
    assert min_dist < 8.0
    # Wall is at x=1.0, half-width is 0.1, so surface is at x=0.9
    # Robot is at x=0.0, lidar mount is at x=0.10, so distance ~ 0.8m
    assert 0.6 <= min_dist <= 1.0

    # Test IMU readings exist
    assert len(state.imu_linear_acceleration) == 3
    assert len(state.imu_angular_velocity) == 3

def test_world_boundaries_generation():
    proj = get_default_project()
    proj.settings.enable_boundaries = True
    proj.settings.world_boundaries = [-5.0, 5.0, -5.0, 5.0, 0.0, 3.0]
    
    engine = MuJoCoEngine(proj)
    xml = engine.generate_mjcf_xml()
    assert "boundary_walls" in xml
    assert "wall_north" in xml
    assert "wall_south" in xml
    assert "wall_east" in xml
    assert "wall_west" in xml

def test_obstacle_friction_mjcf():
    proj = get_default_project()
    proj.objects = [
        SceneObject(
            id="fric_box",
            name="Friction Box",
            type="box",
            position=[1.0, 0.0, 0.2],
            size=[0.2, 0.2, 0.2],
            friction=[0.45, 0.02, 0.003],
            is_static=False,
            mass=2.5
        )
    ]
    engine = MuJoCoEngine(proj)
    xml = engine.generate_mjcf_xml()
    assert 'friction="0.45 0.02 0.003"' in xml

def test_generic_robot_kinematics_and_joint_control():
    urdf_res = parse_urdf_string(SAMPLE_URDF)
    assert urdf_res.robot is not None
    
    proj = get_default_project()
    urdf_res.robot.base_position = [2.0, 2.0, 0.05]
    proj.generic_robot = urdf_res.robot
    
    engine = MuJoCoEngine(proj)
    xml = engine.generate_mjcf_xml()
    assert "shoulder_joint" in xml
    assert "act_shoulder_joint" in xml
    
    # Test setting joint target
    engine.set_joint_target("shoulder_joint", 0.75)
    assert engine.joint_targets["shoulder_joint"] == 0.75
    
    # Step physics
    for _ in range(10):
        engine.step()
    
    state = engine.get_state()
    assert "shoulder_joint" in state.joint_positions
    # Joint angle should move towards commanded 0.75 rad
    assert state.joint_positions["shoulder_joint"] > 0.0
