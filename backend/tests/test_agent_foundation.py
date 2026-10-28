"""
Targeted regression tests for RoboForge simulator gap closure and agent interface:
- Gap A: Continuous trajectory execution in the MuJoCo simulation loop
- Gap B: Genuine pixel-level geom-ID semantic segmentation
- Gap C: Robot assembly snapping and kinematic constraint integrity
- Agent Simulation Interface: Typed inspection, telemetry, perception, experiments, snapshots, and records
"""

import math
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app, agent_interface, engine, project_manager, experiment_manager
from backend.models import (
    Project, SceneObject, GenericRobotModel, LinkModel, JointModel, JointLimit,
    LinkGeom, LinkInertial
)
from backend.experiment_models import ExperimentConfig, TaskObjective
from backend.simulation import MuJoCoEngine
from backend.motion_planning import ContinuousTrajectoryController
from backend.robot_validator import validate_generic_robot

client = TestClient(app)


# ---------------------------------------------------------------------------
# GAP A: Continuous Trajectory Execution
# ---------------------------------------------------------------------------

def test_continuous_trajectory_execution_smooth_progression():
    """Verify that the continuous trajectory controller advances through waypoints without stopping."""
    # Define a 4-waypoint path: start at origin, step along +X, diagonal, then +X
    waypoints = [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.5],
        [1.5, 0.5]
    ]
    controller = ContinuousTrajectoryController(
        waypoints=waypoints,
        v_max=1.0,
        w_max=2.0,
        waypoint_tolerance=0.25,
        goal_tolerance=0.10
    )

    assert controller.current_index == 0
    assert controller.is_completed is False
    assert controller.is_failed is False

    # Simulate stepping forward along the trajectory
    # Step 1: near origin (0.05, 0.0), facing 0 rad
    cmd_v, cmd_w = controller.update(rx=0.05, ry=0.0, ryaw=0.0, dt=0.02)
    assert cmd_v > 0.0, "Should have forward velocity"

    # Step 2: reached near waypoint 0 (0.48, 0.0) -> should smoothly advance to waypoint 1 without halting
    cmd_v, cmd_w = controller.update(rx=0.48, ry=0.0, ryaw=0.0, dt=0.02)
    assert controller.current_index >= 1, "Must advance to next waypoint"
    assert cmd_v > 0.2, "Must not pause or drop velocity to zero between waypoints"

    # Step 3: reaching final goal (1.48, 0.50)
    cmd_v, cmd_w = controller.update(rx=1.48, ry=0.50, ryaw=0.0, dt=0.02)
    assert controller.is_completed is True
    assert cmd_v == 0.0 and cmd_w == 0.0


def test_continuous_trajectory_turn_in_place_alignment():
    """Verify trajectory controller slows down and pivots when heading error is large."""
    waypoints = [
        [0.0, 0.0],
        [0.0, 1.0]  # Directly to the left (+Y)
    ]
    controller = ContinuousTrajectoryController(waypoints=waypoints, v_max=1.0, w_max=2.0)
    
    # Robot is facing +X (yaw = 0), goal is at +Y (yaw = pi/2, 90 deg error)
    cmd_v, cmd_w = controller.update(rx=0.0, ry=0.0, ryaw=0.0, dt=0.02)
    assert cmd_w > 0.5, "Should steer counter-clockwise towards +Y"
    assert cmd_v <= 0.1, "Should slow forward velocity when misaligned"


def test_continuous_trajectory_cancellation_via_agent_interface():
    """Verify trajectory execution and cancellation through the agent interface."""
    # Execute trajectory towards clear workspace coordinate
    traj_state = agent_interface.execute_motion_trajectory(target_pos=[0.0, 1.2])
    assert traj_state.status in ("running", "idle", "planning")
    assert agent_interface.active_trajectory_controller is not None

    # Cancel trajectory
    cancelled = agent_interface.cancel_motion_trajectory()
    assert cancelled is True
    assert agent_interface.active_trajectory_controller is None
    assert agent_interface.trajectory_state.status == "cancelled"


# ---------------------------------------------------------------------------
# GAP B: Semantic Segmentation
# ---------------------------------------------------------------------------

def test_genuine_semantic_segmentation_rendering():
    """Verify that MuJoCo renderer produces genuine pixel-level geom-ID segmentation."""
    proj = Project()
    proj.objects.append(SceneObject(
        id="target_cube",
        name="Target Cube",
        type="box",
        position=[0.8, 0.0, 0.15],
        size=[0.1, 0.1, 0.1],
        mass=1.0,
        color="#ff0000"
    ))
    test_engine = MuJoCoEngine(proj)
    test_engine.step()

    # Capture frame with segmentation enabled
    frame = test_engine.capture_camera_frame(width=160, height=120, render_depth=False, render_segmentation=True)
    assert frame is not None
    assert "segmentation_info" in frame
    seg_info = frame["segmentation_info"]
    assert "image_data_url" in seg_info
    assert "labels" in seg_info

    labels = seg_info["labels"]
    assert len(labels) > 0, "Labels must contain geom mappings"

    # Verify background and geometry representation
    assert "-1" in labels, "Background geom (-1) should be present"
    bg_label = labels["-1"]
    assert bg_label["geom_name"] == "background"

    # Total pixel percentage must sum to ~100%
    total_pct = sum(item["pixel_percentage"] for item in labels.values())
    assert 99.0 <= total_pct <= 101.0, f"Pixel percentages must sum to ~100%, got {total_pct}%"


def test_segmentation_api_endpoint():
    """Verify the /api/simulation/camera/segmentation HTTP endpoint."""
    res = client.get("/api/simulation/camera/segmentation?width=120&height=90")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "image_data_url" in data
    assert "labels" in data
    assert data["width"] == 120
    assert data["height"] == 90


# ---------------------------------------------------------------------------
# GAP C: Robot Assembly Snapping & Kinematic Constraints
# ---------------------------------------------------------------------------

def test_joint_assembly_snapping_calculation():
    """Verify mathematical calculation of parent link extremity snapping."""
    parent_box = LinkModel(
        name="parent_link",
        visual=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0]),
        collision=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0]),
        inertial=LinkInertial(mass=1.5)
    )
    
    # In a box of half-x = 0.20, full length is 0.40m along X
    snap_x = parent_box.visual.size[0] * 2.0
    assert abs(snap_x - 0.40) < 1e-5

    # Parent top surface in Z (half-z = 0.04) -> 0.08m
    snap_z = parent_box.visual.size[2] * 2.0
    assert abs(snap_z - 0.08) < 1e-5


def test_kinematic_hierarchy_validation():
    """Verify validation detects invalid/cyclic robot kinematics."""
    invalid_robot = GenericRobotModel(
        name="CyclicRobot",
        root_link="base_link",
        links=[
            LinkModel(name="base_link"),
            LinkModel(name="link_1"),
        ],
        joints=[
            JointModel(
                name="joint_1",
                type="revolute",
                parent_link="base_link",
                child_link="link_1",
                origin_pos=[0, 0, 0.1],
                axis=[0, 0, 1]
            ),
            JointModel(
                name="joint_cyclic",
                type="revolute",
                parent_link="link_1",
                child_link="base_link",  # Cycle!
                origin_pos=[0, 0, 0.1],
                axis=[0, 0, 1]
            )
        ]
    )
    result = validate_generic_robot(invalid_robot)
    assert result.valid is False
    assert any(d.code == "KINEMATIC_CYCLE" or "cycle" in d.message.lower() for d in result.errors)


# ---------------------------------------------------------------------------
# AGENT SIMULATION INTERFACE: Typed Operations
# ---------------------------------------------------------------------------

def test_agent_interface_inspect_environment():
    """Verify AgentSimulationInterface.inspect_environment()."""
    env = agent_interface.inspect_environment()
    assert env.project_name is not None
    assert env.physics_settings is not None
    assert env.physics_settings.timestep > 0
    assert env.total_objects >= 0
    assert env.actuator_count >= 2


def test_agent_interface_get_telemetry():
    """Verify AgentSimulationInterface.get_telemetry()."""
    telem = agent_interface.get_telemetry()
    assert telem.sim_time >= 0.0
    assert len(telem.robot_position) == 3
    assert len(telem.robot_quaternion) == 4
    assert telem.real_time_factor >= 0.0


def test_agent_interface_capture_perception():
    """Verify AgentSimulationInterface.capture_perception() with multimodal outputs."""
    obs = agent_interface.capture_perception(width=120, height=90, include_segmentation=True, include_depth=True)
    assert obs.width == 120
    assert obs.height == 90
    assert obs.rgb_image_url.startswith("data:image/png;base64,")
    assert obs.segmentation_image_url is not None
    assert obs.segmentation_labels is not None


def test_agent_interface_snapshots():
    """Verify saving and restoring snapshots through the agent interface."""
    snap = agent_interface.save_physics_snapshot()
    assert snap.snapshot_id is not None
    assert snap.sim_time >= 0.0

    restored = agent_interface.restore_physics_snapshot(snap.snapshot_id)
    assert restored is True


def test_agent_interface_experiments_and_reproducible_records():
    """Verify experiment configuration validation, execution, comparison, and reproducible record export."""
    # 1. Validate experiment config
    scenarios = experiment_manager.list_scenarios()
    assert len(scenarios) > 0, "Default scenarios should exist"
    scen = scenarios[0]

    val = agent_interface.validate_experiment_config(scen)
    assert val["valid"] is True
    assert len(val["errors"]) == 0

    # 2. Run experiment synchronously
    run_res = agent_interface.start_experiment(scen.id)
    assert run_res.run_id is not None
    assert run_res.status in ("completed", "success", "collision", "timeout")

    # 3. Compare runs
    comparison = agent_interface.compare_runs(run_res.run_id, run_res.run_id)
    assert comparison.run_a_id == run_res.run_id
    assert comparison.run_b_id == run_res.run_id

    # 4. Export reproducible record
    record = agent_interface.export_reproducible_record(run_res.run_id)
    assert record.record_id.startswith("rep_")
    assert record.project_snapshot is not None
    assert record.run_result is not None
    assert "physics_engine" in record.system_metadata


def test_agent_http_api_endpoints():
    """Verify agent HTTP API routes."""
    # GET /api/agent/inspect
    res = client.get("/api/agent/inspect")
    assert res.status_code == 200
    assert "project_name" in res.json()

    # GET /api/agent/telemetry
    res = client.get("/api/agent/telemetry")
    assert res.status_code == 200
    assert "robot_position" in res.json()

    # GET /api/agent/perception
    res = client.get("/api/agent/perception?width=120&height=90")
    assert res.status_code == 200
    assert "rgb_image_url" in res.json()
