import pytest
from backend.ai_reasoning_service import AIReasoningService
from backend.rl_models import AIChatRequest
from backend.project_manager import ProjectManager

def test_synthesize_environments_maze():
    pm = ProjectManager()
    service = AIReasoningService(proj_manager=pm)
    
    # 1. Test Maze Synthesis
    res_maze = service.chat(AIChatRequest(prompt="build a maze for the robot to navigate"))
    assert res_maze is not None
    assert any(a["action"] == "synthesize_environment" for a in res_maze.actions_executed)
    act_maze = next(a for a in res_maze.actions_executed if a["action"] == "synthesize_environment")
    assert act_maze["env_type"] == "maze"
    assert act_maze["obstacle_count"] >= 5
    assert len(res_maze.thought_steps) >= 3
    assert res_maze.thought_process is not None
    assert "🧠" in res_maze.thought_process

def test_synthesize_environments_warehouse_and_drone_course():
    pm = ProjectManager()
    service = AIReasoningService(proj_manager=pm)
    
    # 2. Test Warehouse Synthesis
    res_wh = service.chat(AIChatRequest(prompt="create a warehouse environment with pallets and storage racks"))
    assert any(a["action"] == "synthesize_environment" for a in res_wh.actions_executed)
    act_wh = next(a for a in res_wh.actions_executed if a["action"] == "synthesize_environment")
    assert act_wh["env_type"] == "warehouse"
    assert act_wh["obstacle_count"] >= 6
    
    # 3. Test Drone Aerial Course Synthesis
    res_drone = service.chat(AIChatRequest(prompt="build an aerial drone course with gates"))
    assert any(a["action"] == "synthesize_environment" for a in res_drone.actions_executed)
    act_drone = next(a for a in res_drone.actions_executed if a["action"] == "synthesize_environment")
    assert act_drone["env_type"] == "drone_course"
    assert act_drone["target_position"][2] >= 1.5

def test_arbitrary_robot_and_sensor_customization():
    pm = ProjectManager()
    service = AIReasoningService(proj_manager=pm)
    
    # Test setting robot speed to 3.8 m/s
    res_spd = service.chat(AIChatRequest(prompt="set robot speed to 3.8"))
    assert any(a["action"] == "set_robot_speed" for a in res_spd.actions_executed)
    
    # Test configuring LiDAR beams
    res_lidar = service.chat(AIChatRequest(prompt="configure lidar to 64 beams"))
    assert any(a["action"] == "configure_lidar" for a in res_lidar.actions_executed)
    
    # Test custom reward script
    reward_code = "def compute_reward(obs, act):\n    return obs[0] * 2.0 - abs(act[1]) * 0.1"
    res_rew = service.tool_set_custom_reward_script(reward_code)
    assert res_rew["status"] == "success"
    assert service.tool_get_custom_reward_script()["script"] == reward_code
