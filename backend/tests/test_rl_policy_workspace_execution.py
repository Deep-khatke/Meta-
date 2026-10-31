import pytest
import numpy as np
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.ai_reasoning_service import AIReasoningService
from backend.rl_models import AIChatRequest

def test_drone_flight_policy_workspace_execution():
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(project=proj)
    engine.rebuild_model()
    
    service = AIReasoningService(proj_manager=pm, engine=engine)
    
    # 1. Deploy drone
    service.tool_add_drone(name="TestDrone-01", position=[2.0, 0.0, 0.5])
    
    # 2. Deploy drone flight policy
    deploy_res = service.tool_deploy_drone_policy(drone_name="TestDrone-01")
    assert deploy_res["status"] == "success"
    assert engine.status == "running"
    
    # 3. Step physics and verify altitude increases (lifting off the ground)
    for _ in range(50):
        engine.step()
        
    state = engine.get_state()
    drone_state = next((o for o in state.objects.values() if o.is_drone), None)
    assert drone_state is not None
    assert drone_state.rotors_active is True
    assert drone_state.flight_mode in ("takeoff", "hover", "flying")
    assert drone_state.position[2] > 0.4  # Successfully airborne in MuJoCo workspace

def test_rover_rl_policy_deployment_and_active_inference():
    pm = ProjectManager()
    proj = pm.get_current()
    engine = MuJoCoEngine(project=proj)
    engine.rebuild_model()
    
    service = AIReasoningService(proj_manager=pm, engine=engine)
    
    # Mock / verify policy deployment
    class DummyPolicy:
        def predict(self, obs, deterministic=True):
            # Return forward velocity and slight steer
            return np.array([0.8, 0.1], dtype=np.float32), None
            
    engine.active_rl_policy_id = "test_ppo_nav"
    engine.active_rl_policy_name = "PPO-Nav-Trained"
    engine.active_rl_policy_model = DummyPolicy()
    engine.play()
    
    state_before = engine.get_state()
    assert state_before.rl_policy_active is True
    assert state_before.active_rl_policy_name == "PPO-Nav-Trained"
    
    # Step simulation: policy predicts [0.8, 0.1] -> translates to wheel speeds
    for _ in range(40):
        engine.step()
        
    state_after = engine.get_state()
    assert state_after.robot.linear_speed > 0.05
    assert not np.isnan(state_after.robot.position[0])
