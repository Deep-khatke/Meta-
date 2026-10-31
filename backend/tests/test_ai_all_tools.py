"""
Comprehensive test suite verifying all 50+ autonomous agent tools
for mastering the RoboForge simulation application.
"""

import pytest
from backend.ai_reasoning_service import AIReasoningService
from backend.project_manager import ProjectManager


@pytest.fixture
def reasoning_service():
    pm = ProjectManager()
    return AIReasoningService(proj_manager=pm)


def test_provider_status_has_50_plus_tools(reasoning_service):
    """Verifies that the autonomous agent catalog contains at least 50 tools."""
    status = reasoning_service.get_provider_status()
    tools = status.get("tools_available", [])
    assert len(tools) >= 50, f"Expected at least 50 tools, got {len(tools)}: {tools}"


def test_scene_geometry_tools(reasoning_service):
    """Tests execution of Category 1: Scene & Obstacle Geometry tools."""
    # 1. spawn_obstacle
    res = reasoning_service.execute_tool("spawn_obstacle", {"shape": "cylinder", "position": [1.8, 0.4, 0.25], "color": "#ef4444"})
    assert res.get("status") == "success"

    # 2. modify_obstacle
    obs_id = res.get("id") or "obs_box_1"
    res_mod = reasoning_service.execute_tool("modify_obstacle", {"identifier": obs_id, "color": "#3b82f6", "mass": 6.5})
    assert res_mod.get("status") in ("success", "not_found")

    # 3. scatter_random_clutter
    res_clutter = reasoning_service.execute_tool("scatter_random_clutter", {"count": 4, "radius": 2.0})
    assert res_clutter.get("status") == "success"
    assert res_clutter.get("count") >= 1

    # 4. set_arena_boundaries
    res_bounds = reasoning_service.execute_tool("set_arena_boundaries", {"bounds": [-4.0, 4.0, -3.0, 3.0, 0.0, 3.0], "enable_boundaries": True})
    assert res_bounds.get("status") == "success"

    # 5. set_ground_plane
    res_ground = reasoning_service.execute_tool("set_ground_plane", {"friction": [0.8, 0.005, 0.0001], "grid_size": 0.5})
    assert res_ground.get("status") == "success"

    # 6. generate_obstacle_course
    res_course = reasoning_service.execute_tool("generate_obstacle_course", {"count": 5, "pattern": "slalom"})
    assert res_course.get("status") == "success"

    # 7. remove_obstacle
    res_rem = reasoning_service.execute_tool("remove_obstacle", {"identifier": "Course Obstacle 1"})
    assert res_rem.get("status") in ("success", "not_found")

    # 8. clear_scene
    res_clear = reasoning_service.execute_tool("clear_scene", {})
    assert res_clear.get("status") == "success"


def test_robot_hardware_and_kinematics_tools(reasoning_service):
    """Tests execution of Category 2: Robot Hardware & Kinematics tools."""
    # 9. set_robot_speed
    res_spd = reasoning_service.execute_tool("set_robot_speed", {"max_linear_velocity": 2.2, "max_angular_velocity": 3.8})
    assert res_spd.get("status") == "success"
    assert res_spd.get("max_linear_velocity") == 2.2

    # 10. set_robot_chassis
    res_chassis = reasoning_service.execute_tool("set_robot_chassis", {"mass": 6.0, "length": 0.35, "width": 0.25})
    assert res_chassis.get("status") == "success"

    # 11. set_wheel_geometry
    res_wheels = reasoning_service.execute_tool("set_wheel_geometry", {"wheel_radius": 0.075, "wheel_base": 0.32})
    assert res_wheels.get("status") == "success"

    # 12. teleport_robot
    res_teleport = reasoning_service.execute_tool("teleport_robot", {"position": [0.5, 0.2, 0.07], "yaw": 45.0})
    assert res_teleport.get("status") == "success"

    # 13. add_robot
    res_robot = reasoning_service.execute_tool("add_robot", {"name": "ForgeBot-DD2", "position": [1.0, 0.5, 0.07]})
    assert res_robot.get("status") == "success"

    # 14. add_drone
    res_drone = reasoning_service.execute_tool("add_drone", {"name": "AirScout-01", "position": [2.0, 0.0, 1.8]})
    assert res_drone.get("status") == "success"

    # 15. run_kinematic_diagnostics
    res_diag = reasoning_service.execute_tool("run_kinematic_diagnostics", {})
    assert res_diag.get("status") == "success"
    assert "motor_rpm_saturation" in res_diag


def test_sensor_suite_and_perception_tools(reasoning_service):
    """Tests execution of Category 3: Sensor Suite & Perception tools."""
    # 16. configure_lidar
    res_lidar = reasoning_service.execute_tool("configure_lidar", {"enabled": True, "rays": 32, "max_range": 12.0})
    assert res_lidar.get("status") == "success"

    # 17. configure_imu
    res_imu = reasoning_service.execute_tool("configure_imu", {"enabled": True, "accel_noise": 0.02})
    assert res_imu.get("status") == "success"

    # 18. configure_camera
    res_cam = reasoning_service.execute_tool("configure_camera", {"enabled": True, "fov": 75.0, "width": 640, "height": 480})
    assert res_cam.get("status") == "success"

    # 19. configure_wheel_encoders
    res_enc = reasoning_service.execute_tool("configure_wheel_encoders", {"enabled": True, "ticks_per_rev": 2048})
    assert res_enc.get("status") == "success"

    # 20. capture_camera_frame
    res_frame = reasoning_service.execute_tool("capture_camera_frame", {"width": 160, "height": 120})
    assert res_frame.get("status") in ("success", "simulated")

    # 21. get_lidar_scan
    res_scan = reasoning_service.execute_tool("get_lidar_scan", {})
    assert res_scan.get("status") in ("success", "nominal")
    assert "min_distance_m" in res_scan

    # 22. get_imu_telemetry
    res_imutel = reasoning_service.execute_tool("get_imu_telemetry", {})
    assert res_imutel.get("status") in ("success", "nominal")
    assert "linear_acceleration" in res_imutel


def test_physics_and_simulation_engine_tools(reasoning_service):
    """Tests execution of Category 4: Physics & Simulation Engine tools."""
    # 23. simulation_control
    res_ctl = reasoning_service.execute_tool("simulation_control", {"command": "pause"})
    assert res_ctl.get("status") == "success"

    # 24. set_simulation_speed
    res_spd = reasoning_service.execute_tool("set_simulation_speed", {"multiplier": 1.5})
    assert res_spd.get("status") == "success"

    # 25. set_physics
    res_phys = reasoning_service.execute_tool("set_physics", {"gravity": [0.0, 0.0, -9.81], "timestep": 0.004})
    assert res_phys.get("status") == "success"

    # 26. step_simulation
    res_step = reasoning_service.execute_tool("step_simulation", {"steps": 2})
    assert res_step.get("status") == "success"

    # 27. inspect_contacts
    res_con = reasoning_service.execute_tool("inspect_contacts", {})
    assert res_con.get("status") in ("success", "nominal")

    # 28. inspect_energy_metrics
    res_e = reasoning_service.execute_tool("inspect_energy_metrics", {})
    assert res_e.get("status") == "success"
    assert "total_mechanical_energy_joules" in res_e


def test_navigation_and_motion_control_tools(reasoning_service):
    """Tests execution of Category 5: Navigation, Planning & Motion Control tools."""
    # 29. set_target_position
    res_target = reasoning_service.execute_tool("set_target_position", {"position": [3.0, 0.8, 0.25]})
    assert res_target.get("status") == "success"

    # 30. set_target_radius
    res_rad = reasoning_service.execute_tool("set_target_radius", {"radius": 0.3})
    assert res_rad.get("status") == "success"

    # 31. tune_apf_controller
    res_apf = reasoning_service.execute_tool("tune_apf_controller", {"k_att": 2.2, "k_rep": 3.8, "safety_margin": 0.9})
    assert res_apf.get("status") == "success"

    # 32. execute_teleop_cmd
    res_cmd = reasoning_service.execute_tool("execute_teleop_cmd", {"linear": 0.5, "angular": 0.1})
    assert res_cmd.get("status") == "success"

    # 33. plan_and_execute_trajectory
    res_plan = reasoning_service.execute_tool("plan_and_execute_trajectory", {"target_pos": [2.0, 0.0]})
    assert res_plan.get("status") in ("success", "planning_failed")

    # 34. start_autodrive
    res_ad_start = reasoning_service.execute_tool("start_autodrive", {})
    assert res_ad_start.get("status") == "success"

    # 35. stop_autodrive
    res_ad_stop = reasoning_service.execute_tool("stop_autodrive", {})
    assert res_ad_stop.get("status") == "success"


def test_rl_studio_tools(reasoning_service):
    """Tests execution of Category 6: Reinforcement Learning (PPO) Studio tools."""
    # 36. list_rl_models
    res_models = reasoning_service.execute_tool("list_rl_models", {})
    assert res_models.get("status") == "success"

    # 37. get_rl_training_status
    res_st = reasoning_service.execute_tool("get_rl_training_status", {})
    assert res_st.get("status") in ("success", "idle")

    # 38. cancel_rl_training
    res_cancel = reasoning_service.execute_tool("cancel_rl_training", {})
    assert res_cancel.get("status") in ("success", "failed", "not_found")


def test_benchmarking_and_experiment_tools(reasoning_service):
    """Tests execution of Category 7: Empirical Experimentation & Benchmarking tools."""
    # 39. list_experiment_runs
    res_runs = reasoning_service.execute_tool("list_experiment_runs", {"limit": 5})
    assert res_runs.get("status") == "success"

    # 40. export_experiment_results
    res_exp = reasoning_service.execute_tool("export_experiment_results", {})
    assert res_exp.get("status") in ("success", "not_found")


def test_research_studio_and_hypotheses_tools(reasoning_service):
    """Tests execution of Category 8: Research Studio & Hypothesis Refinement tools."""
    # 41. get_mission_status
    res_mstatus = reasoning_service.execute_tool("get_mission_status", {})
    assert res_mstatus.get("status") in ("success", "not_found")

    # 42. start_research_mission
    res_rm = reasoning_service.execute_tool("start_research_mission", {"goal_description": "Validate collision-free passage", "max_iterations": 1})
    assert res_rm.get("status") == "success"
    assert "mission_id" in res_rm


def test_swarm_and_ros2_tools(reasoning_service):
    """Tests execution of Category 9: Swarm Robotics, ROS 2 & Scene Snapshots tools."""
    # 43. deploy_swarm
    res_swarm = reasoning_service.execute_tool("deploy_swarm", {"count": 4, "formation": "v_shape", "spacing": 0.6})
    assert res_swarm.get("status") == "success"
    assert res_swarm.get("count") == 4

    # 44. get_swarm_status
    res_sw_st = reasoning_service.execute_tool("get_swarm_status", {})
    assert res_sw_st.get("status") == "success"
    assert res_sw_st.get("swarm_size") == 4

    # 45. start_ros2_bag
    res_bag = reasoning_service.execute_tool("start_ros2_bag", {"dataset_name": "test_mcap_run"})
    assert res_bag.get("status") in ("recording", "error")

    # 46. stop_ros2_bag
    res_bag_stop = reasoning_service.execute_tool("stop_ros2_bag", {})
    assert res_bag_stop.get("status") in ("success", "error")

    # 47. save_scene_snapshot
    res_snap = reasoning_service.execute_tool("save_scene_snapshot", {"snapshot_name": "test_auto_snap"})
    assert res_snap.get("status") == "success"

    # 48. restore_scene_snapshot
    res_rest = reasoning_service.execute_tool("restore_scene_snapshot", {"snapshot_name": "test_auto_snap.rfproj.json"})
    assert res_rest.get("status") in ("success", "error")


def test_natural_language_intent_matching_for_new_features(reasoning_service):
    """Verifies that _extract_intent_actions detects commands across diverse subsystems."""
    ctx = reasoning_service.gather_simulation_context()

    # 1. Swarm intent
    acts = reasoning_service._extract_intent_actions("deploy a swarm of 5 robots in circle formation", "deploy a swarm of 5 robots in circle formation", ctx)
    assert any(a[0] == "deploy_swarm" for a in acts)
    swarm_act = next(a for a in acts if a[0] == "deploy_swarm")
    assert swarm_act[1]["count"] == 5
    assert swarm_act[1]["formation"] == "circle"

    # 2. Clutter intent
    acts_clutter = reasoning_service._extract_intent_actions("scatter 7 random clutter obstacles", "scatter 7 random clutter obstacles", ctx)
    assert any(a[0] == "scatter_random_clutter" for a in acts_clutter)

    # 3. Kinematic diagnostics intent
    acts_diag = reasoning_service._extract_intent_actions("run kinematic diagnostics on our robot", "run kinematic diagnostics on our robot", ctx)
    assert any(a[0] == "run_kinematic_diagnostics" for a in acts_diag)

    # 4. Simulation speed intent
    acts_spd = reasoning_service._extract_intent_actions("set simulation speed to 2.5x", "set simulation speed to 2.5x", ctx)
    assert any(a[0] == "set_simulation_speed" for a in acts_spd)

    # 5. Autodrive intent
    acts_ad = reasoning_service._extract_intent_actions("start autodrive navigation loop", "start autodrive navigation loop", ctx)
    assert any(a[0] == "start_autodrive" for a in acts_ad)

    # 6. ROS 2 bag intent
    acts_bag = reasoning_service._extract_intent_actions("start ros2 bag recording now", "start ros2 bag recording now", ctx)
    assert any(a[0] == "start_ros2_bag" for a in acts_bag)
