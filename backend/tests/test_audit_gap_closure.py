import pytest
import math
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.models import (
    Project, SceneObject, RobotConfig, GenericRobotModel,
    LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial
)
from backend.simulation import MuJoCoEngine
from backend.scene_validator import validate_scene, get_object_aabb, check_aabb_intersection
from backend.mjcf_importer import export_robot_mjcf, parse_mjcf_string
from backend.controllers import PIDController, TrajectoryPoint, TrajectoryTracker
from backend.motion_planning import plan_collision_free_path, PathPlanningRequest
from backend.synthetic_data import SyntheticDataGenerator, DomainRandomizationConfig
from backend.ros2_bridge import ROS2Bridge
from backend.experiment_manager import experiment_manager

client = TestClient(app)

def get_test_robot() -> GenericRobotModel:
    return GenericRobotModel(
        name="TestArm",
        root_link="base_link",
        base_position=[1.0, 1.0, 0.05],
        links=[
            LinkModel(
                name="base_link",
                visual=LinkGeom(type="cylinder", size=[0.1, 0.05], origin_pos=[0, 0, 0.05]),
                collision=LinkGeom(type="cylinder", size=[0.1, 0.05], origin_pos=[0, 0, 0.05]),
                inertial=LinkInertial(mass=2.0)
            ),
            LinkModel(
                name="arm_link",
                visual=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0]),
                collision=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0]),
                inertial=LinkInertial(mass=1.0)
            )
        ],
        joints=[
            JointModel(
                name="joint_1",
                type="revolute",
                parent_link="base_link",
                child_link="arm_link",
                origin_pos=[0.0, 0.0, 0.1],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-1.57, upper=1.57, effort=30.0, velocity=3.0)
            )
        ]
    )


# --- 1. Physics Core & Snapshots ---

def test_physics_snapshots_and_rollback():
    project = Project()
    engine = MuJoCoEngine(project)
    
    # Save initial snapshot at t=0
    snap0 = engine.save_snapshot()
    assert snap0.sim_time == 0.0
    assert len(engine.list_snapshots()) >= 1
    
    # Step forward 20 steps
    for _ in range(20):
        engine.step()
    assert engine.sim_time > 0.0
    assert engine.step_count == 20
    
    # Restore back to initial snapshot
    restored = engine.restore_snapshot(snap0.snapshot_id)
    assert restored is True
    assert engine.sim_time == 0.0
    assert engine.step_count == 0


def test_contact_forces_and_rtf():
    project = Project()
    # Add a dynamic box dropped on the floor
    project.objects.append(SceneObject(
        id="falling_box",
        name="Falling Box",
        type="box",
        position=[0.0, 1.0, 0.15],
        size=[0.1, 0.1, 0.1],
        mass=2.0,
        is_static=False
    ))
    engine = MuJoCoEngine(project)
    
    for _ in range(30):
        engine.step()
        
    state = engine.get_state()
    assert state.real_time_factor > 0.0
    assert state.odometry is not None
    # Contacts should include normal force
    assert len(state.contacts) > 0
    # At least one contact should register normal force balancing gravity
    normal_forces = [c.force_normal for c in state.contacts]
    assert any(fn > 0.0 for fn in normal_forces)


# --- 2. Scene Validation & Diagnostics ---

def test_scene_validation_diagnostics():
    project = Project()
    # 1. Object penetrating floor (z=0.0 with half-height 0.2 -> penetrates by 0.2m)
    project.objects.append(SceneObject(
        id="pen_box",
        name="Penetrating Box",
        type="box",
        position=[3.0, 3.0, 0.0],
        size=[0.2, 0.2, 0.2],
        is_static=True
    ))
    # 2. Pairwise overlapping objects
    project.objects.append(SceneObject(
        id="overlap_1",
        name="Overlap Box A",
        type="box",
        position=[2.0, 2.0, 0.3],
        size=[0.2, 0.2, 0.2],
        is_static=True
    ))
    project.objects.append(SceneObject(
        id="overlap_2",
        name="Overlap Box B",
        type="box",
        position=[2.1, 2.1, 0.3],
        size=[0.2, 0.2, 0.2],
        is_static=True
    ))
    # 3. Object overlapping robot spawn position
    rx, ry, _ = project.robot.initial_position
    project.objects.append(SceneObject(
        id="spawn_blocker",
        name="Spawn Blocker",
        type="box",
        position=[rx, ry, 0.1],
        size=[0.2, 0.2, 0.2],
        is_static=True
    ))

    report = validate_scene(project)
    assert report.is_valid is False
    assert any(e.category == "robot_clearance" for e in report.errors)
    assert any(w.category == "floor_penetration" for w in report.warnings)
    assert any(w.category == "overlap" for w in report.warnings)


# --- 3. MJCF Importer and Exporter ---

def test_mjcf_export_and_import():
    robot = get_test_robot()
    # Export GenericRobotModel to MJCF XML string
    xml_str = export_robot_mjcf(robot)
    assert "<mujoco" in xml_str
    assert 'model="TestArm"' in xml_str
    assert 'name="arm_link"' in xml_str
    assert 'name="joint_1"' in xml_str
    assert "<actuator>" in xml_str

    # Re-import from MJCF XML string
    result = parse_mjcf_string(xml_str)
    assert result.status in ("success", "warning")
    assert result.robot is not None
    assert result.robot.name == "TestArm"
    assert len(result.robot.links) == 2
    assert len(result.robot.joints) == 1
    assert result.robot.joints[0].name == "joint_1"


# --- 4. Controllers (PID & Trajectory) ---

def test_pid_controller_tracking_and_antiwindup():
    pid = PIDController(kp=2.0, ki=1.0, kd=0.1, min_output=-10.0, max_output=10.0, integral_limit=5.0)
    
    # Step response with error = 5.0
    out1 = pid.compute(target=10.0, current=5.0, dt=0.1)
    assert out1 > 0.0
    assert out1 <= 10.0

    # Test integral accumulation and anti-windup clamping
    for _ in range(50):
        out = pid.compute(target=10.0, current=0.0, dt=0.2)
        assert out <= 10.0
    assert pid._integral <= 5.0


def test_trajectory_tracker_interpolation():
    points = [
        TrajectoryPoint(time_from_start=0.0, positions={"j1": 0.0, "j2": 0.0}),
        TrajectoryPoint(time_from_start=2.0, positions={"j1": 1.0, "j2": 2.0}),
        TrajectoryPoint(time_from_start=4.0, positions={"j1": 2.0, "j2": 0.0})
    ]
    tracker = TrajectoryTracker(points)

    # At t=0
    assert tracker.sample(0.0) == {"j1": 0.0, "j2": 0.0}
    # At t=1.0 (midway between 0 and 2)
    s1 = tracker.sample(1.0)
    assert abs(s1["j1"] - 0.5) < 1e-4
    assert abs(s1["j2"] - 1.0) < 1e-4
    # At t=4.0
    assert tracker.sample(4.0) == {"j1": 2.0, "j2": 0.0}


# --- 5. Collision-Aware Motion Planning (A*) ---

def test_grid_astar_motion_planner():
    project = Project()
    # Place an obstacle directly in the direct line between (0, 0) and (2, 0)
    project.objects.append(SceneObject(
        id="wall_blocker",
        name="Wall Blocker",
        type="box",
        position=[1.0, 0.0, 0.2],
        size=[0.2, 0.6, 0.2],
        is_static=True
    ))

    req = PathPlanningRequest(
        start_pos=[0.0, 0.0],
        goal_pos=[2.0, 0.0],
        grid_resolution=0.1,
        robot_radius=0.25,
        smooth_path=True
    )
    result = plan_collision_free_path(project, req)
    assert result.success is True
    assert len(result.path) >= 2
    assert result.path_length > 2.0  # Must detour around the blocker
    
    # Verify no waypoint in path penetrates the blocker
    for wp in result.path:
        dx = abs(wp[0] - 1.0)
        dy = abs(wp[1] - 0.0)
        # Must not be inside the obstacle box
        assert not (dx < 0.15 and dy < 0.5)


# --- 6. Offscreen Camera Rendering ---

def test_offscreen_camera_rendering():
    project = Project()
    engine = MuJoCoEngine(project)
    
    frame = engine.capture_camera_frame(width=160, height=120, render_depth=True)
    assert frame["width"] == 160
    assert frame["height"] == 120
    assert frame["image_data_url"].startswith("data:image/png;base64,")
    assert frame["depth_info"] is not None
    assert frame["depth_info"]["mean_m"] > 0.0


# --- 7. Synthetic Perception Data & COCO Export ---

def test_synthetic_data_coco_export():
    project = Project()
    project.objects.append(SceneObject(
        id="perception_box",
        name="Target Cube",
        type="box",
        position=[1.5, 0.0, 0.15],
        size=[0.15, 0.15, 0.15],
        color="#e74c3c",
        is_static=True
    ))
    
    generator = SyntheticDataGenerator(project)
    coco = generator.generate_coco_dataset(num_samples=2, width=160, height=120)
    
    assert len(coco.images) == 2
    assert len(coco.categories) >= 1
    assert "images" in coco.model_dump()
    assert "annotations" in coco.model_dump()


# --- 8. ROS 2 Bridge Serialization ---

def test_ros2_bridge_serialization():
    project = Project()
    engine = MuJoCoEngine(project)
    for _ in range(5):
        engine.step()
    state = engine.get_state()

    # Odometry
    odom = ROS2Bridge.serialize_odometry(state)
    assert odom["header"]["frame_id"] == "odom"
    assert "pose" in odom and "twist" in odom

    # Scan
    scan = ROS2Bridge.serialize_laser_scan(state)
    assert scan is not None
    assert scan["header"]["frame_id"] == "lidar_link"
    assert len(scan["ranges"]) == len(state.lidar_readings)

    # Joint States
    js = ROS2Bridge.serialize_joint_states(state)
    assert js["header"]["frame_id"] == "base_link"
    assert isinstance(js["name"], list)

    # Cmd Vel parsing
    twist_payload = {"linear": {"x": 1.2, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": -0.8}}
    teleop = ROS2Bridge.parse_cmd_vel(twist_payload)
    assert abs(teleop.linear - 1.2) < 1e-4
    assert abs(teleop.angular - (-0.8)) < 1e-4


# --- 9. Multi-Trial Statistical Experiments ---

def test_multi_trial_experiment_aggregation():
    # Use baseline scenario
    res = experiment_manager.run_multi_trial_experiment("scen_nav_alpha", num_trials=3)
    assert res["scenario_id"] == "scen_nav_alpha"
    assert res["num_trials"] == 3
    assert 0.0 <= res["success_rate"] <= 1.0
    assert "mean" in res["distance_stats"]
    assert "std" in res["collision_stats"]
    assert len(res["trials"]) == 3


# --- 10. End-to-End API Integration ---

def test_api_gap_closure_endpoints():
    # Snapshots
    resp_snap = client.post("/api/simulation/snapshot")
    assert resp_snap.status_code == 200
    snap_id = resp_snap.json()["snapshot"]["snapshot_id"]

    resp_list = client.get("/api/simulation/snapshots")
    assert resp_list.status_code == 200
    assert any(s["snapshot_id"] == snap_id for s in resp_list.json()["snapshots"])

    resp_restore = client.post("/api/simulation/snapshot/restore", json={"snapshot_id": snap_id})
    assert resp_restore.status_code == 200

    # Offscreen Camera Capture
    resp_cam = client.post("/api/simulation/camera/capture", json={"width": 160, "height": 120, "render_depth": True})
    assert resp_cam.status_code == 200
    assert resp_cam.json()["image_data_url"].startswith("data:image/png;base64,")

    # Scene Validation
    resp_val = client.post("/api/scene/validate")
    assert resp_val.status_code == 200
    assert "is_valid" in resp_val.json()

    # Motion Planning
    resp_plan = client.post("/api/motion/plan-path", json={
        "start_pos": [0.0, 0.0],
        "goal_pos": [0.0, 2.0],
        "grid_resolution": 0.15,
        "robot_radius": 0.25
    })
    assert resp_plan.status_code == 200
    assert resp_plan.json()["success"] is True

    # ROS 2 Topics
    resp_topics = client.get("/api/ros2/topics")
    assert resp_topics.status_code == 200
    assert len(resp_topics.json()["topics"]) >= 5

    resp_odom = client.get("/api/ros2/messages/odom")
    assert resp_odom.status_code == 200
    assert "pose" in resp_odom.json()
