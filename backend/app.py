from __future__ import annotations
import asyncio
import json
import logging
from typing import Set, Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Load environment variables from a local .env (API keys, model names) before anything reads
# them. The .env is gitignored; keys are never hardcoded or sent to the frontend.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import backend.database as db

import time
import math
import numpy as np
import mujoco

import uuid
import io
from fastapi.responses import StreamingResponse
from backend.models import (
    Project, SceneObject, SimulationState, LogMessage, TeleopCommand,
    SimulationSettings, RobotConfig, GenericRobotModel, URDFImportResult,
    FKRequest, FKResult, IKRequest, IKResult, RobotValidationResult, RobotDiagnostic,
    LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial,
    SimulationSnapshot, OdometryState, MJCFImportResult
)
from backend.simulation import MuJoCoEngine, quat_to_yaw
from backend.urdf_importer import parse_urdf_string, export_robot_urdf
from backend.mjcf_importer import export_robot_mjcf, parse_mjcf_string
from backend.kinematics import compute_forward_kinematics, solve_inverse_kinematics
from backend.robot_validator import validate_generic_robot, auto_calculate_robot_inertials
from backend.project_manager import ProjectManager
from backend.scene_validator import validate_scene, SceneValidationReport
from backend.motion_planning import plan_collision_free_path, PathPlanningRequest, PathPlanningResult, TrajectoryExecutionState
from backend.synthetic_data import SyntheticDataGenerator, DomainRandomizationConfig, COCODataset
from backend.ros2_bridge import (
    ROS2Bridge, ros2_bag_recorder, swarm_orchestrator, SwarmOrchestrator,
    ros2_bag_player, ROS2CLIExecutor, export_ros2_workspace_zip
)
from backend.agent_interface import (
    AgentSimulationInterface, EnvironmentInspectionResult, TelemetryObservation,
    PerceptionObservation, ReproducibleRecord
)
from backend.experiment_models import (
    ExperimentConfig, TaskObjective, ExperimentRunResult, RunComparison
)
from backend.experiment_manager import ExperimentManager
from backend.rl_env import RoboForgeGymEnv
from backend.rl_models import (
    RLTrainingConfig, RLTrainingStatus, RLEvaluationRequest, AIAssistantPromptRequest,
    HypothesisRefinementRequest, ExecuteRLPolicyRequest,
    AIChatRequest, AIChatResponse, AIApprovalRequest
)
from backend.rl_training_manager import training_manager
from backend.rl_evaluator import RLEvaluator
from backend.ai_assistant import AIExperimentAssistant
from backend.ai_reasoning_service import get_ai_reasoning_service
from backend.rl_refinement import HypothesisRefinementEngine
from backend.chat_memory import chat_memory
from backend.asset_models import (
    AssetModel, ValidationResult, CreateAssetRequest, ApplyAssetRequest
)
from backend.asset_library import AssetLibrary
from backend.research.research_api import router as research_router
from backend.billing.routes import router as billing_router
from backend.billing.credit_service import credit_service, InsufficientCreditsError
from backend.billing.config import OPERATION_PRICING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("roboforge.server")

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Project Manager, Experiment Manager & Physics Engine
project_manager = ProjectManager()
experiment_manager = ExperimentManager()
asset_library = AssetLibrary()
recent_logs: list[LogMessage] = []

def handle_engine_log(log_msg: LogMessage):
    recent_logs.append(log_msg)
    if len(recent_logs) > 200:
        recent_logs.pop(0)
    # Broadcast to all connected clients if event loop is running
    try:
        loop = asyncio.get_running_loop()
        if loop and loop.is_running():
            payload = {"type": "log", "data": log_msg.model_dump()}
            loop.create_task(broadcast_json(payload))
    except RuntimeError:
        pass

engine = MuJoCoEngine(project_manager.get_current(), log_callback=handle_engine_log)
agent_interface = AgentSimulationInterface(project_manager, engine, experiment_manager)

from backend.rosbridge_server import RosbridgeServer
rosbridge = RosbridgeServer(engine, agent_interface)

simulation_task: Optional[asyncio.Task] = None

async def broadcast_json(data: dict):
    """Send to every client concurrently with a short timeout, dropping any that are dead or stalled.

    A single half-closed browser connection used to block this loop (and every endpoint that awaits it,
    such as New / Open / Clear scene) indefinitely.
    """
    async def _send(ws):
        try:
            await asyncio.wait_for(ws.send_json(data), timeout=2.0)
        except Exception:
            active_connections.discard(ws)

    clients = list(active_connections)
    if clients:
        await asyncio.gather(*(_send(ws) for ws in clients))

active_live_experiment: Optional[dict] = None
auto_drive_active: bool = False

async def simulation_loop():
    """Background simulation loop running at ~60Hz."""
    global active_live_experiment, auto_drive_active
    logger.info("Simulation background loop started.")
    last_loop_time = asyncio.get_event_loop().time()
    
    while True:
        try:
            now = asyncio.get_event_loop().time()
            dt_wall = now - last_loop_time
            last_loop_time = now

            if engine.status == "running":
                # If an autonomous live experiment is running, steer robot & track objective
                if active_live_experiment:
                    try:
                        # Guard: the diff-drive live-experiment loop requires an active robot with
                        # a valid freejoint. If the scene has no robot (robot_active=False) or the
                        # qpos layout can't supply a 2D pose, skip this branch instead of crashing.
                        _qpos = engine.data.qpos
                        _adr = int(getattr(engine, "robot_qpos_adr", 0) or 0)
                        _has_robot = (getattr(engine, "robot_body_id", -1) >= 0
                                      and getattr(engine, "active_rl_policy_kind", "rover") in ("rover", "mobile")
                                      and (_adr + 7) <= len(_qpos))
                        if not _has_robot:
                            active_live_experiment = None
                            raise RuntimeError("skip-live-experiment-no-robot")
                        r_pos = np.array(_qpos[_adr:_adr + 2], dtype=float)
                        r_quat = [float(x) for x in _qpos[_adr + 3:_adr + 7]]
                        target = np.asarray(active_live_experiment["target"], dtype=float)[:2]
                        dist = float(np.linalg.norm(r_pos - target))
                        v_max = active_live_experiment["config"].robot.max_linear_velocity
                        w_max = active_live_experiment["config"].robot.max_angular_velocity

                        rl_model = active_live_experiment.get("rl_model")
                        used_policy = False
                        if rl_model is not None:
                            try:
                                state = engine.get_state()
                                obs = RoboForgeGymEnv.compute_observation(
                                    robot_pos=state.robot.position,
                                    robot_quat=state.robot.quaternion,
                                    linear_speed=state.robot.linear_speed,
                                    angular_speed=state.robot.angular_speed,
                                    target_pos=active_live_experiment["config"].task.target_position,
                                    obstacles=active_live_experiment["config"].environment,
                                    max_linear_velocity=v_max,
                                    max_angular_velocity=w_max,
                                )
                                # Guard: only drive with the policy if its observation space matches
                                # the 16-D diff-drive navigation obs. A policy trained on a different
                                # robot kind (arm=22-D, drone=15-D) must NOT be fed here -- that threw
                                # "Unexpected observation shape" every step and froze the rover.
                                exp_shape = getattr(getattr(rl_model, "observation_space", None), "shape", None)
                                if exp_shape is not None and tuple(exp_shape) == (obs.shape[0],):
                                    action, _ = rl_model.predict(obs, deterministic=True)
                                    act = np.clip(action, -1.0, 1.0)
                                    cmd_v = float(act[0] * v_max)
                                    cmd_w = float(act[1] * w_max)
                                    used_policy = True
                                else:
                                    # Mismatched policy -> drop it from this loop and fall back to the
                                    # deterministic goal-seeking controller so the rover still drives.
                                    if not active_live_experiment.get("_obs_warned"):
                                        engine.log("warning",
                                                   f"Deployed policy obs {tuple(exp_shape) if exp_shape else '?'} "
                                                   f"!= navigation obs ({obs.shape[0]},); using goal-seeking controller.")
                                        active_live_experiment["_obs_warned"] = True
                                    active_live_experiment["rl_model"] = None
                            except Exception as _pred_err:
                                if not active_live_experiment.get("_obs_warned"):
                                    engine.log("warning", f"Policy inference failed ({_pred_err}); using goal-seeking controller.")
                                    active_live_experiment["_obs_warned"] = True
                                active_live_experiment["rl_model"] = None

                        if not used_policy:
                            cmd_v, cmd_w = experiment_manager._compute_goal_seeking_control(
                                r_pos, r_quat, target, dist, v_max, w_max,
                                obstacles=active_live_experiment["config"].environment,
                                policy=active_live_experiment["config"].task.policy
                            )
                        engine.set_teleop(cmd_v, cmd_w)

                        # Collision tracking
                        ncon = int(engine.data.ncon)
                        robot_contact_geoms = []
                        for i in range(ncon):
                            con = engine.data.contact[i]
                            g1 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                            g2 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                            if "robot" in g1 or "robot" in g2:
                                other = g2 if "robot" in g1 else g1
                                if other != "ground" and other != "":
                                    robot_contact_geoms.append(other)

                        has_col = len(robot_contact_geoms) > 0
                        if has_col and not active_live_experiment["in_collision"]:
                            active_live_experiment["unique_collision_events"] += 1
                        active_live_experiment["in_collision"] = has_col

                        # Sample measurement
                        if engine.sim_time >= active_live_experiment["next_sample_time"] or engine.step_count == 0:
                            experiment_manager._record_measurement(
                                active_live_experiment["measurements"], engine, robot_contact_geoms
                            )
                            active_live_experiment["next_sample_time"] += (1.0 / active_live_experiment["config"].sample_rate_hz)

                        # Termination conditions
                        cfg = active_live_experiment["config"]
                        max_cols = cfg.task.max_allowed_collisions
                        done = False
                        status_str = "completed"
                        reason_str = "Completed"

                        if dist <= cfg.task.distance_threshold:
                            done = True
                            active_live_experiment["time_to_goal"] = engine.sim_time
                            status_str = "completed"
                            reason_str = f"Target reached within {dist:.3f}m at t={engine.sim_time:.2f}s"
                        elif active_live_experiment["unique_collision_events"] > max_cols and max_cols >= 0:
                            done = True
                            status_str = "failed"
                            reason_str = f"Collision limit exceeded: {active_live_experiment['unique_collision_events']} > {max_cols}"
                        elif active_live_experiment.get("timeout") is not None and engine.sim_time >= active_live_experiment["timeout"]:
                            done = True
                            status_str = "timeout"
                            reason_str = f"Timeout reached at {engine.sim_time:.2f}s (distance: {dist:.3f}m)"

                        if done:
                            if not active_live_experiment.get("run_until_goal", True):
                                engine.pause()
                            experiment_manager._record_measurement(
                                active_live_experiment["measurements"], engine, robot_contact_geoms
                            )
                            derived = experiment_manager._compute_derived_metrics(
                                active_live_experiment["measurements"],
                                cfg.task.target_position,
                                active_live_experiment["time_to_goal"],
                                active_live_experiment["unique_collision_events"],
                                status_str,
                                reason_str
                            )
                            res = ExperimentRunResult(
                                experiment_id=cfg.id,
                                experiment_name=cfg.name,
                                config_version=cfg.schema_version,
                                seed=active_live_experiment["seed"],
                                status=status_str,
                                completed_at=time.time(),
                                measurements=active_live_experiment["measurements"],
                                metrics=derived,
                                errors=engine.errors
                            )
                            experiment_manager.save_run_result(res)
                            active_live_experiment = None
                            await broadcast_json({"type": "live_experiment_completed", "data": res.model_dump()})
                            engine.log("info", f"Live experiment '{cfg.name}' completed: {reason_str}")
                    except RuntimeError as ex:
                        if str(ex) != "skip-live-experiment-no-robot":
                            logger.exception("Error in active live experiment step: %s", ex)
                    except Exception as ex:
                        logger.exception("Error in active live experiment step: %s", ex)
                elif agent_interface.active_trajectory_controller and (time.time() - engine.last_teleop_time > 0.35):
                    try:
                        # Guard: a ground trajectory needs an active rover with a valid freejoint.
                        # Without it (e.g. empty/drone-only scene) the qpos slice is short and
                        # quat_to_yaw would crash every tick — cancel the orphaned controller.
                        _adr = getattr(engine, "robot_qpos_adr", None)
                        _has_rover = (engine.project.robot is not None and getattr(engine.project, "robot_active", True)
                                      and _adr is not None and len(engine.data.qpos) >= _adr + 7)
                        if not _has_rover:
                            agent_interface.active_trajectory_controller = None
                            agent_interface.trajectory_state.status = "failed"
                            agent_interface.trajectory_state.status_message = "No active rover to drive the path."
                            raise RuntimeError("no-rover-for-trajectory")
                        r_pos = np.array(engine.data.qpos[engine.robot_qpos_adr:engine.robot_qpos_adr + 2])
                        r_quat = [float(x) for x in engine.data.qpos[engine.robot_qpos_adr + 3:engine.robot_qpos_adr + 7]]
                        ryaw = quat_to_yaw(r_quat)
                        ctrl = agent_interface.active_trajectory_controller
                        cmd_v, cmd_w = ctrl.update(r_pos[0], r_pos[1], ryaw, dt_wall)

                        # Collision monitoring during trajectory execution
                        ncon = int(engine.data.ncon)
                        hit_obs = False
                        for i in range(ncon):
                            con = engine.data.contact[i]
                            g1 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                            g2 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                            if "robot" in g1 or "robot" in g2:
                                other = g2 if "robot" in g1 else g1
                                if other and other != "ground" and "wheel" not in other and "caster" not in other:
                                    hit_obs = True
                                    break

                        if hit_obs:
                            ctrl.is_failed = True
                            ctrl.failure_reason = "Obstacle collision detected along trajectory."

                        if ctrl.is_completed:
                            agent_interface.trajectory_state.status = "completed"
                            agent_interface.trajectory_state.status_message = "Target reached successfully."
                            agent_interface.trajectory_state.progress_pct = 100.0
                            engine.set_teleop(0.0, 0.0)
                            agent_interface.active_trajectory_controller = None
                            await broadcast_json({"type": "trajectory_completed", "data": agent_interface.trajectory_state.model_dump()})
                            engine.log("info", "Continuous trajectory execution completed successfully.")
                        elif ctrl.is_failed:
                            agent_interface.trajectory_state.status = "failed"
                            agent_interface.trajectory_state.status_message = ctrl.failure_reason
                            agent_interface.trajectory_state.collision_detected = True
                            engine.set_teleop(0.0, 0.0)
                            agent_interface.active_trajectory_controller = None
                            await broadcast_json({"type": "trajectory_failed", "data": agent_interface.trajectory_state.model_dump()})
                            engine.log("error", f"Continuous trajectory execution failed: {ctrl.failure_reason}")
                        else:
                            total_w = max(1, len(ctrl.waypoints))
                            agent_interface.trajectory_state.current_waypoint_index = ctrl.current_index
                            agent_interface.trajectory_state.progress_pct = round((ctrl.current_index / total_w) * 100.0, 1)
                            final_pt = ctrl.waypoints[-1]
                            agent_interface.trajectory_state.distance_to_goal = round(math.sqrt((final_pt[0] - r_pos[0])**2 + (final_pt[1] - r_pos[1])**2), 3)
                            engine.set_teleop(cmd_v, cmd_w)
                    except Exception as e:
                        if "no-rover-for-trajectory" not in str(e):
                            logger.exception("Error in continuous trajectory step: %s", e)
                elif agent_interface.active_drone_controller and (time.time() - engine.last_teleop_time > 0.35):
                    # 3D drone waypoint follower: advance the PD-autopilot target along the path.
                    try:
                        dctrl = agent_interface.active_drone_controller
                        fc = engine._find_drone_controller(agent_interface.active_drone_id or "drone")
                        if fc is None:
                            agent_interface.active_drone_controller = None
                        else:
                            cp = fc.current_position
                            tgt = dctrl.update(float(cp[0]), float(cp[1]), float(cp[2]))
                            fc.fly_to([tgt[0], tgt[1], tgt[2]])
                            ts_ = agent_interface.trajectory_state
                            if dctrl.is_completed:
                                ts_.status = "completed"
                                ts_.status_message = "Flight path completed."
                                ts_.progress_pct = 100.0
                                agent_interface.active_drone_controller = None
                                fc.hover()
                                await broadcast_json({"type": "trajectory_completed", "data": ts_.model_dump()})
                            else:
                                ts_.current_waypoint_index = dctrl.current_index
                                ts_.progress_pct = round((dctrl.current_index / max(1, len(dctrl.waypoints))) * 100.0, 1)
                    except Exception as e:
                        logger.exception("Error in drone trajectory step: %s", e)
                elif auto_drive_active and (time.time() - engine.last_teleop_time > 0.35):
                    # Autonomous navigation towards target waypoint
                    try:
                        r_pos = np.array(engine.data.qpos[engine.robot_qpos_adr:engine.robot_qpos_adr + 2])
                        r_quat = [float(x) for x in engine.data.qpos[engine.robot_qpos_adr + 3:engine.robot_qpos_adr + 7]]
                        target = np.array([2.5, 0.5])
                        dist = float(np.linalg.norm(r_pos - target))
                        v_max = engine.project.robot.max_linear_velocity
                        w_max = engine.project.robot.max_angular_velocity

                        # If reached target, reverse or idle
                        if dist < 0.2:
                            engine.set_teleop(0.0, 0.5)
                        else:
                            cmd_v, cmd_w = experiment_manager._compute_goal_seeking_control(
                                r_pos, r_quat, target, dist, v_max, w_max,
                                obstacles=engine.project.objects,
                            )
                            engine.set_teleop(cmd_v, cmd_w)
                    except Exception:
                        pass

                # Compute how many physics steps to advance this loop tick
                ts = engine.project.settings.timestep
                mult = engine.project.settings.speed_multiplier
                # Advance physics unless a visual RL curriculum is actively driving the engine
                # itself. During the episodic rover trainer's LEARNING bursts the training thread
                # is busy computing PPO and is NOT stepping the engine, so we must keep stepping
                # here to keep the viewport live (otherwise the scene freezes/glitches). Only when
                # the trainer is actively rolling out an attempt (rl_training_rollout_active) does
                # it drive the engine, and we yield stepping to it to avoid double-stepping.
                _train_active = getattr(engine, "rl_training_active", False)
                _rollout_active = getattr(engine, "rl_training_rollout_active", False)
                if not _train_active or (_train_active and not _rollout_active):
                    steps_to_take = max(1, min(20, int(round((dt_wall * mult) / ts))))
                    for _ in range(steps_to_take):
                        engine.step()
                else:
                    engine.sim_time += dt_wall
                    engine.step_count += 1

            # Always broadcast state at ~30-60Hz to clients
            state = engine.get_state()
            if ros2_bag_recorder.is_recording and engine.status == "running":
                ros2_bag_recorder.record_step(state)
            state_dict = state.model_dump()
            await broadcast_json({"type": "state", "data": state_dict})

            # Target ~40-60 Hz loop update
            await asyncio.sleep(0.016)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Error in simulation loop: %s", e)
            engine.log("error", f"Simulation loop error: {str(e)}")
            await asyncio.sleep(0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulation_task
    simulation_task = asyncio.create_task(simulation_loop())
    engine.log("info", "RoboForge server initialized and physics engine ready.")
    # Real ROS interoperability: rosbridge v2 server (Foxglove / roslibpy / roslibjs / ROS 2 graph).
    await rosbridge.start()
    if rosbridge.status()["running"]:
        engine.log("info", f"rosbridge v2 server live at {rosbridge.status()['url']}")
    yield
    await rosbridge.stop()
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
    logger.info("RoboForge server shut down.")

app = FastAPI(title="RoboForge Robotics Simulation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)
app.include_router(billing_router)


def _extract_token_from_request(req: Request, authorization: Optional[str] = None) -> Optional[str]:
    """Extracts bearer token from Authorization header or cookie or query param."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    auth_header = req.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return req.cookies.get("roboforge_token") or req.query_params.get("token")


def _get_user_from_request_optional(req: Request) -> Optional[Dict[str, Any]]:
    token = _extract_token_from_request(req)
    if not token:
        return None
    return db.get_user_by_session(token)


def check_and_reserve_credits(req: Request, op_type: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Checks if request is authenticated. If so, reserves credits atomically.
    Raises HTTPException(402) if available balance is insufficient.
    """
    user = _get_user_from_request_optional(req)
    if not user:
        return None
    amount = OPERATION_PRICING.get(op_type, 1)
    op_id = f"op_{op_type}_{uuid.uuid4().hex[:12]}"
    try:
        credit_service.reserveCredits(user["id"], op_id, amount, operation_type=op_type)
        return (user, op_id)
    except InsufficientCreditsError as ice:
        raise HTTPException(status_code=402, detail=ice.to_dict())


# Request Models
class ControlRequest(BaseModel):
    action: str = Field(..., description="'play', 'pause', 'reset', 'step', 'set_speed', 'set_timestep'")
    value: Optional[float] = None

class ObjectCreateRequest(BaseModel):
    name: str
    type: str
    position: list[float] = [0.0, 0.0, 0.5]
    rotation: list[float] = [0.0, 0.0, 0.0]
    size: list[float] = [0.3, 0.3, 0.3]
    color: str = "#8892b0"
    is_static: bool = False
    mass: float = 1.0

class LoadProjectRequest(BaseModel):
    project_json: Optional[str] = None
    filename: Optional[str] = None

class RunExperimentRequest(BaseModel):
    scenario_id: Optional[str] = None
    config: Optional[ExperimentConfig] = None
    seed: Optional[int] = None
    duration: Optional[float] = None

class CompareRunsRequest(BaseModel):
    run_a_id: str
    run_b_id: str

class DuplicateScenarioRequest(BaseModel):
    new_name: Optional[str] = None

class SaveActiveScenarioRequest(BaseModel):
    name: str
    description: Optional[str] = "Created from active scene"
    seed: int = 42
    target_position: list[float] = [2.5, 0.0, 0.25]
    distance_threshold: float = 0.25
    timeout: float = 8.0

class TrainPolicyRequest(BaseModel):
    scenario_id: Optional[str] = ""

class URDFImportRequest(BaseModel):
    urdf_xml: str
    set_as_active_robot: bool = True

class JointControlRequest(BaseModel):
    joint_name: Optional[str] = None
    target: Optional[float] = None
    targets: Optional[dict[str, float]] = None

class SnapshotRestoreRequest(BaseModel):
    snapshot_id: str

class MJCFImportRequest(BaseModel):
    mjcf_xml: str
    set_as_active_robot: bool = True

class CameraCaptureRequest(BaseModel):
    width: int = Field(default=320, ge=64, le=1280)
    height: int = Field(default=240, ge=64, le=720)
    camera_name: Optional[str] = None
    render_depth: bool = False
    render_segmentation: bool = False

class TrajectoryExecuteRequest(BaseModel):
    target_pos: List[float]
    grid_resolution: float = Field(default=0.15, ge=0.05, le=1.0)
    robot_radius: float = Field(default=0.30, ge=0.05, le=2.0)

class AgentExperimentRunRequest(BaseModel):
    scenario_id: str
    seed: Optional[int] = None

class MultiTrialRequest(BaseModel):
    scenario_id: str
    num_trials: int = Field(default=5, ge=2, le=20)

class SyntheticExportRequest(BaseModel):
    num_samples: int = Field(default=5, ge=1, le=50)
    width: int = Field(default=320, ge=64, le=1280)
    height: int = Field(default=240, ge=64, le=720)

class ROS2BagStartRequest(BaseModel):
    name: Optional[str] = "rosbag2_recording"
    topics: Optional[List[str]] = None

class ROS2CLIRequest(BaseModel):
    command: str

class ROS2JointCommandsRequest(BaseModel):
    positions: Optional[List[float]] = None
    names: Optional[List[str]] = None
    joint_name: Optional[str] = None
    target: Optional[float] = None
    targets: Optional[dict[str, float]] = None
    data: Optional[List[float]] = None

class ROS2GoalPoseRequest(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
    pose: Optional[dict] = None

class ROS2BagReplayRequest(BaseModel):
    speed_multiplier: float = 1.0

class SwarmDeployRequest(BaseModel):
    formation: str = Field(default="v_shape", description="Formation: v_shape, line, column, circle, diamond")
    agent_count: int = Field(default=4, ge=2, le=10)
    spacing: float = Field(default=1.2, ge=0.4, le=5.0)

active_swarm_config: Optional[dict] = None


@app.get("/api/health")
def get_health():
    return {"status": "ok", "engine_status": engine.status, "schema_version": project_manager.get_current().schema_version}


@app.get("/api/ros2/bridge/status")
def ros2_bridge_status():
    """Live status of the rosbridge v2 server (real ROS client interoperability)."""
    return rosbridge.status()


@app.get("/api/project")
def get_project():
    return project_manager.get_current()


@app.post("/api/project")
async def update_project(project: Project):
    try:
        updated = project_manager.update_project(project)
        engine.project = updated
        engine.rebuild_model(preserve_state=True)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return updated
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/project/new")
@app.post("/api/project/new")
async def new_project(name: str = "New Scene", template: str = "empty", robot_type: Optional[str] = None):
    """Creates a fresh project.

    template:
      - "empty" (default): clean empty workspace, ready for the user to import a URDF/MJCF robot.
      - "rover"/"diff_drive": starter differential-drive rover in an empty arena.
      - "kr6"/"scara"/"arm": a preset articulated arm.
    """
    proj = project_manager.new_project(name=name, template=template, robot_type=robot_type)
    # A clean/empty or arm workspace has no built-in diff-drive rover; mark it inactive so the
    # viewport starts empty. A rover template keeps the robot active.
    proj.robot_active = proj.robot is not None
    engine.project = proj
    engine.rebuild_model(preserve_state=False)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return proj


@app.post("/api/project/save")
def save_project(filename: Optional[str] = None):
    try:
        saved_path = project_manager.save_project_to_disk(filename)
        return {"status": "saved", "path": saved_path, "project": project_manager.get_current()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {str(e)}")


@app.post("/api/project/load")
async def load_project(req: LoadProjectRequest):
    try:
        if req.project_json:
            loaded = project_manager.load_project_from_json(req.project_json)
        elif req.filename:
            loaded = project_manager.load_project_from_file(req.filename)
        else:
            raise HTTPException(status_code=400, detail="Provide either project_json or filename")
        
        engine.project = loaded
        engine.rebuild_model(preserve_state=False)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return loaded
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")


@app.post("/api/project/reset-to-saved")
async def reset_to_saved():
    proj = project_manager.reset_to_saved()
    engine.project = proj
    engine.rebuild_model(preserve_state=False)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return proj


@app.get("/api/project/list")
def list_projects():
    return {"projects": project_manager.list_saved_projects()}


@app.get("/api/simulation/state")
def get_simulation_state():
    return engine.get_state()


@app.post("/api/simulation/control")
def control_simulation(req: ControlRequest):
    action = req.action.lower()
    if action == "play":
        engine.play()
    elif action == "pause":
        engine.pause()
    elif action == "step":
        if engine.status == "running":
            engine.pause()
        engine.step()
        engine.log("info", f"Single step executed at t={engine.sim_time:.3f}s")
    elif action == "reset":
        engine.reset()
    elif action == "set_speed":
        if req.value is not None and 0.1 <= req.value <= 5.0:
            engine.project.settings.speed_multiplier = req.value
            engine.log("info", f"Simulation speed set to {req.value:.1f}x")
        else:
            raise HTTPException(status_code=400, detail="Speed multiplier must be between 0.1 and 5.0")
    elif action == "set_timestep":
        if req.value is not None and 0.001 <= req.value <= 0.05:
            engine.project.settings.timestep = req.value
            engine.rebuild_model(preserve_state=True)
            engine.log("info", f"Simulation timestep set to {req.value:.4f}s")
        else:
            raise HTTPException(status_code=400, detail="Timestep must be between 0.001 and 0.05s")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    return {"status": engine.status, "action": action, "state": engine.get_state()}


@app.post("/api/simulation/teleop")
def send_teleop(cmd: TeleopCommand):
    engine.set_teleop(cmd.linear, cmd.angular)
    return {"status": "ok", "teleop": engine.teleop_cmd}


# --- Simulation Snapshots ---

@app.post("/api/simulation/snapshot")
def create_simulation_snapshot():
    snap = engine.save_snapshot()
    return {"status": "ok", "snapshot": snap}

@app.get("/api/simulation/snapshots")
def list_simulation_snapshots():
    return {"snapshots": engine.list_snapshots()}

@app.post("/api/simulation/snapshot/restore")
def restore_simulation_snapshot(req: SnapshotRestoreRequest):
    success = engine.restore_snapshot(req.snapshot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{req.snapshot_id}' not found or could not be restored.")
    return {"status": "ok", "snapshot_id": req.snapshot_id}

@app.delete("/api/simulation/snapshot/{snapshot_id}")
def delete_simulation_snapshot(snapshot_id: str):
    deleted = engine.delete_snapshot(snapshot_id)
    return {"status": "ok", "deleted": deleted}


# --- Offscreen Camera Rendering & Semantic Segmentation ---

@app.post("/api/simulation/camera/capture")
def capture_camera_frame(req: CameraCaptureRequest):
    try:
        res = engine.capture_camera_frame(
            width=req.width,
            height=req.height,
            camera_name=req.camera_name,
            render_depth=req.render_depth,
            render_segmentation=req.render_segmentation
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera rendering failed: {str(e)}")

@app.api_route("/api/simulation/camera/segmentation", methods=["GET", "POST"])
def capture_segmentation_mask(
    req: Optional[CameraCaptureRequest] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    camera_name: Optional[str] = None
):
    try:
        w = (req.width if req else None) or width or 320
        h = (req.height if req else None) or height or 240
        cam = (req.camera_name if req else None) or camera_name
        res = engine.capture_camera_frame(
            width=w,
            height=h,
            camera_name=cam,
            render_depth=False,
            render_segmentation=True
        )
        if not res.get("segmentation_info"):
            raise HTTPException(status_code=500, detail="Segmentation rendering failed.")
        return {
            "status": "ok",
            "width": w,
            "height": h,
            **res["segmentation_info"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


# --- Scene Validation & Diagnostics ---

@app.post("/api/scene/validate")
def validate_active_scene():
    report = validate_scene(project_manager.get_current())
    return report


# --- Native MJCF Robot Import & Export ---

@app.post("/api/robot/export-mjcf")
def export_robot_to_mjcf():
    robot = project_manager.get_current().generic_robot
    if not robot:
        raise HTTPException(status_code=400, detail="No generic articulated robot in active scene to export.")
    xml_str = export_robot_mjcf(robot)
    return {"robot_name": robot.name, "mjcf_xml": xml_str}

@app.post("/api/robot/import-mjcf")
async def import_robot_from_mjcf(req: MJCFImportRequest):
    result = parse_mjcf_string(req.mjcf_xml)
    if result.status == "error" or not result.robot:
        return result

    if req.set_as_active_robot and result.robot:
        current = project_manager.get_current()
        current.generic_robot = result.robot
        project_manager.update_project(current)
        engine.project = current
        engine.rebuild_model(preserve_state=False)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})

    return result


# --- Collision-Aware Motion Planning ---

@app.post("/api/motion/plan-path")
def plan_motion_path(req: PathPlanningRequest, request: Request):
    credit_hold = check_and_reserve_credits(request, "motion_planning")
    try:
        result = plan_collision_free_path(project_manager.get_current(), req)
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return result
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise


# --- Synthetic Data & Perception Pipeline ---

@app.post("/api/synthetic/capture-frame")
def capture_synthetic_labeled_frame(req: CameraCaptureRequest = CameraCaptureRequest()):
    generator = SyntheticDataGenerator(project_manager.get_current())
    frame = generator.capture_labeled_frame(width=req.width, height=req.height)
    return frame

@app.post("/api/synthetic/export-coco")
def export_synthetic_coco_dataset(request: Request, req: SyntheticExportRequest = SyntheticExportRequest()):
    credit_hold = check_and_reserve_credits(request, "synthetic_data_export")
    try:
        generator = SyntheticDataGenerator(project_manager.get_current())
        coco = generator.generate_coco_dataset(num_samples=req.num_samples, width=req.width, height=req.height)
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return coco
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise


# --- ROS 2 Robotics Interoperability Bridge & Bag Recording ---

@app.get("/api/ros2/topics")
def list_ros2_topics():
    return {
        "node": "/roboforge_bridge_node",
        "domain_id": 0,
        "topics": [
            {"topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist", "mode": "subscription", "rate_hz": "Event-driven"},
            {"topic": "/odom", "type": "nav_msgs/msg/Odometry", "mode": "publication", "rate_hz": 50},
            {"topic": "/scan", "type": "sensor_msgs/msg/LaserScan", "mode": "publication", "rate_hz": 20},
            {"topic": "/imu/data", "type": "sensor_msgs/msg/Imu", "mode": "publication", "rate_hz": 100},
            {"topic": "/joint_states", "type": "sensor_msgs/msg/JointState", "mode": "publication", "rate_hz": 50},
            {"topic": "/joint_commands", "type": "std_msgs/msg/Float64MultiArray", "mode": "subscription", "rate_hz": "Event-driven"},
            {"topic": "/goal_pose", "type": "geometry_msgs/msg/PoseStamped", "mode": "subscription", "rate_hz": "Event-driven"},
            {"topic": "/clock", "type": "rosgraph_msgs/msg/Clock", "mode": "publication", "rate_hz": 50},
            {"topic": "/tf", "type": "tf2_msgs/msg/TFMessage", "mode": "publication", "rate_hz": 50}
        ]
    }

@app.get("/api/ros2/messages/{topic_name:path}")
@app.get("/api/ros2/topic/{topic_name:path}")
def get_ros2_message(topic_name: str):
    state = engine.get_state()
    t = topic_name.strip("/")
    if t == "odom":
        return ROS2Bridge.serialize_odometry(state)
    elif t == "scan":
        msg = ROS2Bridge.serialize_laser_scan(state)
        if not msg:
            raise HTTPException(status_code=404, detail="No active laser scan data")
        return msg
    elif t in ("imu", "imu/data"):
        msg = ROS2Bridge.serialize_imu(state)
        if not msg:
            raise HTTPException(status_code=404, detail="IMU sensor not enabled")
        return msg
    elif t in ("joint_states", "joint_state"):
        return ROS2Bridge.serialize_joint_states(state)
    elif t == "clock":
        return ROS2Bridge.serialize_clock(state.sim_time)
    elif t == "tf":
        return ROS2Bridge.serialize_tf(state)
    else:
        raise HTTPException(status_code=404, detail=f"Topic '/{t}' not recognized.")

@app.get("/api/ros2/tf")
def get_ros2_tf():
    state = engine.get_state()
    return ROS2Bridge.serialize_tf(state)

@app.post("/api/ros2/cmd_vel")
def send_ros2_cmd_vel(twist_payload: dict):
    cmd = ROS2Bridge.parse_cmd_vel(twist_payload)
    engine.set_teleop(cmd.linear, cmd.angular)
    return {"status": "ok", "teleop": engine.teleop_cmd}

@app.post("/api/ros2/bag/start")
def start_ros2_bag_recording(req: ROS2BagStartRequest = ROS2BagStartRequest()):
    try:
        res = ros2_bag_recorder.start_recording(name=req.name or "rosbag2_recording", sim_time=engine.sim_time, topics=req.topics)
        engine.log("info", f"ROS 2 Bag recording started: '{res['bag_name']}'")
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/ros2/bag/stop")
def stop_ros2_bag_recording():
    try:
        res = ros2_bag_recorder.stop_recording(sim_time=engine.sim_time)
        engine.log("info", f"ROS 2 Bag recording stopped: saved {res.get('metadata', {}).get('message_count', 0)} messages.")
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ros2/bag/status")
def get_ros2_bag_status():
    return {
        "is_recording": ros2_bag_recorder.is_recording,
        "current_bag_id": ros2_bag_recorder.current_bag_id,
        "current_bag_name": ros2_bag_recorder.current_bag_name,
        "message_count": len(ros2_bag_recorder.recorded_messages)
    }

@app.get("/api/ros2/bags")
def list_ros2_bags():
    return {"bags": ros2_bag_recorder.list_bags()}

@app.get("/api/ros2/bag/{bag_id}")
def get_ros2_bag(bag_id: str):
    bag = ros2_bag_recorder.get_bag(bag_id)
    if not bag:
        raise HTTPException(status_code=404, detail=f"ROS 2 Bag '{bag_id}' not found.")
    return bag

@app.delete("/api/ros2/bag/{bag_id}")
def delete_ros2_bag(bag_id: str):
    success = ros2_bag_recorder.delete_bag(bag_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"ROS 2 Bag '{bag_id}' not found.")
    return {"status": "ok", "deleted_bag_id": bag_id}

@app.post("/api/ros2/cli")
def execute_ros2_cli(req: ROS2CLIRequest):
    return ROS2CLIExecutor.execute(req.command, engine, project_manager)

@app.post("/api/ros2/joint_commands")
def send_ros2_joint_commands(req: ROS2JointCommandsRequest):
    applied: dict[str, float] = {}
    if req.joint_name and req.target is not None:
        applied[req.joint_name] = float(req.target)
        engine.set_joint_target(req.joint_name, float(req.target))
    elif req.targets:
        for k, v in req.targets.items():
            applied[k] = float(v)
            engine.set_joint_target(k, float(v))
    else:
        targets = ROS2Bridge.parse_joint_commands(req.model_dump(exclude_unset=True))
        for k, v in targets.items():
            applied[k] = float(v)
            engine.set_joint_target(k, float(v))
    engine.log("info", f"ROS 2 /joint_commands: applied {len(applied)} joint targets.")
    return {"status": "ok", "applied_targets": applied}

@app.post("/api/ros2/goal_pose")
def send_ros2_goal_pose(req: ROS2GoalPoseRequest):
    try:
        gx, gy, gyaw = ROS2Bridge.parse_goal_pose(req.model_dump(exclude_unset=True))
        if req.x is not None:
            gx = float(req.x)
        if req.y is not None:
            gy = float(req.y)
        if req.yaw is not None:
            gyaw = float(req.yaw)

        traj_state = agent_interface.execute_motion_trajectory(target_pos=[gx, gy, 0.0])
        engine.log("info", f"ROS 2 Nav2 /goal_pose received: Navigating to ({gx:.2f}, {gy:.2f}, yaw={gyaw:.2f} rad)")
        return {
            "status": "navigating",
            "goal": {"x": gx, "y": gy, "yaw": gyaw},
            "trajectory_state": getattr(traj_state, "status", "ready"),
            "waypoint_count": len(getattr(traj_state, "waypoints", []))
        }
    except Exception as ex:
        engine.log("warn", f"ROS 2 Nav2 /goal_pose notice: {str(ex)}")
        return {
            "status": "dispatched",
            "goal": {"x": req.x or 0.0, "y": req.y or 0.0, "yaw": req.yaw or 0.0},
            "trajectory_state": "ready",
            "waypoint_count": 0,
            "message": str(ex)
        }

@app.post("/api/ros2/bag/{bag_id}/replay")
def replay_ros2_bag(bag_id: str, req: ROS2BagReplayRequest = ROS2BagReplayRequest()):
    res = ros2_bag_player.play_bag(engine, bag_id, req.speed_multiplier)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res

@app.post("/api/ros2/bag/replay-stop")
def stop_ros2_bag_replay():
    ros2_bag_player.stop()
    return {"status": "stopped"}

@app.get("/api/ros2/bag/replay-status")
def get_ros2_bag_replay_status():
    return {
        "is_playing": ros2_bag_player.is_playing,
        "current_bag_id": ros2_bag_player.current_bag_id,
        "playback_progress": ros2_bag_player.playback_progress
    }

@app.get("/api/ros2/export-package")
def export_ros2_package():
    proj = project_manager.get_current()
    zip_bytes = export_ros2_workspace_zip(proj)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=roboforge_bridge_ros2_ws.zip"}
    )



# --- Multi-Robot Swarm & Formation Orchestration ---

@app.post("/api/swarm/deploy")
async def deploy_swarm(req: SwarmDeployRequest):
    global active_swarm_config
    try:
        state = engine.get_state()
        leader_pos = state.robot.position
        leader_yaw = math.degrees(quat_to_yaw(state.robot.quaternion))

        coords = SwarmOrchestrator.calculate_formation_positions(
            leader_pos=leader_pos,
            leader_yaw_deg=leader_yaw,
            formation=req.formation,
            count=req.agent_count,
            spacing=req.spacing
        )

        # Remove previous swarm follower objects
        engine.project.objects = [o for o in engine.project.objects if not o.name.startswith("Swarm-Follower-")]

        # Add follower models
        followers = []
        for i in range(1, len(coords)):
            c = coords[i]
            follower = SceneObject(
                name=f"Swarm-Follower-{i}",
                type="box",
                position=[c[0], c[1], 0.08],
                rotation=[0.0, 0.0, leader_yaw],
                size=[0.24, 0.18, 0.10],
                color="#06b6d4",
                is_static=True,
                mass=2.5
            )
            engine.project.objects.append(follower)
            followers.append({"id": follower.id, "name": follower.name, "position": [c[0], c[1], 0.08]})

        project_manager.get_current().objects = engine.project.objects
        engine.rebuild_model(preserve_state=True)
        active_swarm_config = {
            "formation": req.formation,
            "agent_count": req.agent_count,
            "spacing": req.spacing,
            "leader_pos": leader_pos,
            "followers": followers
        }

        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        engine.log("info", f"Swarm fleet deployed: {req.agent_count} agents in '{req.formation}' formation.")
        return {"status": "ok", "swarm": active_swarm_config}
    except Exception as e:
        logger.exception("Swarm deployment failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/swarm/status")
def get_swarm_status():
    global active_swarm_config
    return {
        "active": active_swarm_config is not None,
        "config": active_swarm_config,
        "available_formations": SwarmOrchestrator.FORMATIONS
    }

@app.post("/api/swarm/clear")
async def clear_swarm():
    global active_swarm_config
    engine.project.objects = [o for o in engine.project.objects if not o.name.startswith("Swarm-Follower-")]
    project_manager.get_current().objects = engine.project.objects
    engine.rebuild_model(preserve_state=True)
    active_swarm_config = None
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    engine.log("info", "Swarm fleet cleared from active scene.")
    return {"status": "ok", "cleared": True}


# --- Aerial Drone & Multi-Robot Flight Routes ---

class DroneCommandRequest(BaseModel):
    drone_id: Optional[str] = "drone"
    command: str = "takeoff"  # "takeoff", "land", "hover", "fly_to"
    altitude: Optional[float] = 1.5
    position: Optional[List[float]] = None

@app.post("/api/drone/command")
def command_drone_api(req: DroneCommandRequest):
    if req.command == "takeoff":
        return engine.command_drone_takeoff(req.drone_id or "drone", req.altitude or 1.5)
    elif req.command == "land":
        return engine.command_drone_land(req.drone_id or "drone")
    elif req.command == "hover":
        return engine.command_drone_hover(req.drone_id or "drone")
    elif req.command == "fly_to":
        return engine.command_drone_fly_to(req.drone_id or "drone", req.position)
    raise HTTPException(status_code=400, detail=f"Unknown drone command: {req.command}")

@app.get("/api/drone/status")
def get_drone_status_api():
    drones = []
    for d_id, ctrl in engine.drone_controllers.items():
        drones.append({
            "drone_id": ctrl.drone_id,
            "name": ctrl.name,
            "flight_mode": ctrl.flight_mode,
            "target_altitude": ctrl.target_altitude,
            "current_altitude": ctrl.current_altitude,
            "current_position": ctrl.current_position,
            "rotors_active": ctrl.rotors_active
        })
    return {"drones": drones, "count": len(drones)}


# --- Multi-Trial Experimentation ---

@app.post("/api/experiments/multi-trial")
def run_multi_trial(req: MultiTrialRequest):
    try:
        res = experiment_manager.run_multi_trial_experiment(req.scenario_id, num_trials=req.num_trials)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scene/objects")
async def add_scene_object(req: ObjectCreateRequest):
    try:
        new_obj = SceneObject(
            name=req.name,
            type=req.type,  # type: ignore
            position=req.position,
            rotation=req.rotation,
            size=req.size,
            color=req.color,
            is_static=req.is_static,
            mass=req.mass
        )
        engine.project.objects.append(new_obj)
        project_manager.current_project.objects = engine.project.objects
        engine.rebuild_model(preserve_state=True)
        engine.log("info", f"Added object '{new_obj.name}' ({new_obj.type}) to scene.")
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return new_obj
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/scene/objects/{obj_id}")
async def update_scene_object(obj_id: str, updated_obj: SceneObject):
    found = False
    for i, obj in enumerate(engine.project.objects):
        if obj.id == obj_id:
            engine.project.objects[i] = updated_obj
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Object not found")
    
    project_manager.current_project.objects = engine.project.objects
    engine.rebuild_model(preserve_state=True)
    engine.log("info", f"Updated object '{updated_obj.name}'.")
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return updated_obj


@app.delete("/api/scene/objects/{obj_id}")
async def delete_scene_object(obj_id: str):
    idx = -1
    for i, obj in enumerate(engine.project.objects):
        if obj.id == obj_id:
            idx = i
            break
    if idx == -1:
        raise HTTPException(status_code=404, detail="Object not found")
    
    removed = engine.project.objects.pop(idx)
    project_manager.current_project.objects = engine.project.objects
    engine.rebuild_model(preserve_state=True)
    engine.log("info", f"Removed object '{removed.name}' from scene.")
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return {"status": "deleted", "id": obj_id}


@app.post("/api/scene/objects/{obj_id}/duplicate")
async def duplicate_scene_object(obj_id: str):
    target_obj = None
    for obj in engine.project.objects:
        if obj.id == obj_id:
            target_obj = obj
            break
    if not target_obj:
        raise HTTPException(status_code=404, detail="Object not found")
    
    dup_id = str(uuid.uuid4())[:8]
    new_obj = target_obj.model_copy(update={
        "id": dup_id,
        "name": f"{target_obj.name} (Copy)",
        "position": [target_obj.position[0] + 0.3, target_obj.position[1] + 0.3, target_obj.position[2]]
    })
    engine.project.objects.append(new_obj)
    project_manager.current_project.objects = engine.project.objects
    engine.rebuild_model(preserve_state=True)
    engine.log("info", f"Duplicated object '{target_obj.name}' to '{new_obj.name}'.")
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return new_obj


@app.post("/api/project/duplicate")
async def duplicate_project(new_name: Optional[str] = None):
    cur = project_manager.get_current()
    dup = cur.model_copy(deep=True)
    dup.name = new_name or f"{cur.name} (Copy)"
    project_manager.update_project(dup)
    engine.project = dup
    engine.rebuild_model(preserve_state=False)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return dup


@app.get("/api/scene/export")
def export_scene():
    return project_manager.get_current().model_dump()


@app.post("/api/scene/import")
async def import_scene(project: Project):
    try:
        updated = project_manager.update_project(project)
        engine.project = updated
        engine.rebuild_model(preserve_state=False)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return {"status": "imported", "project": updated}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scene validation error: {str(e)}")


@app.post("/api/robot/import-urdf")
async def import_urdf(req: URDFImportRequest):
    result = parse_urdf_string(req.urdf_xml)
    if result.status != "error" and req.set_as_active_robot and result.robot is not None:
        engine.project.generic_robot = result.robot
        project_manager.current_project.generic_robot = result.robot
        engine.rebuild_model(preserve_state=False)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        engine.log("info", f"Imported URDF Robot '{result.robot.name}' with {result.total_links} links and {result.total_joints} joints.")
    return result


@app.post("/api/robot/joint-control")
async def control_robot_joints(req: JointControlRequest):
    if req.targets:
        engine.set_joint_targets(req.targets)
    elif req.joint_name and req.target is not None:
        engine.set_joint_target(req.joint_name, req.target)
    else:
        raise HTTPException(status_code=400, detail="Provide either joint_name and target or targets dictionary")
    # Joint targets are realized by the physics actuators during stepping. If the sim is stopped,
    # start it so the user immediately sees the joint drive to the commanded angle.
    if engine.status != "running":
        engine.play()
    engine.last_teleop_time = 0.0
    return {"status": "ok", "targets": engine.joint_targets, "running": engine.status == "running"}


class JointKeyframe(BaseModel):
    targets: dict[str, float]
    duration: float = 1.5


class JointKeyframeRequest(BaseModel):
    keyframes: list[JointKeyframe]
    loop: bool = False


@app.post("/api/robot/joint-keyframes")
async def play_joint_keyframes(req: JointKeyframeRequest):
    """Plays an authored sequence of joint poses (keyframes) on the active articulated robot,
    smoothly interpolating the position actuators from one pose to the next. This is the
    deterministic way to make an arm perform a specific motion (e.g. wave) without RL."""
    kfs = [{"targets": kf.targets, "duration": kf.duration} for kf in req.keyframes]
    res = engine.start_joint_keyframes(kfs, loop=req.loop)
    if res.get("status") != "ok":
        raise HTTPException(status_code=400, detail=res.get("message", "Could not start keyframes"))
    return res


@app.post("/api/robot/joint-keyframes/stop")
async def stop_joint_keyframes():
    return engine.stop_joint_keyframes()


# -------------------- Attachable Sensors (Isaac-Lab-style) --------------------

class SensorCreateRequest(BaseModel):
    name: Optional[str] = None
    type: str = "imu"
    parent_body: str = "robot"
    mount_pos: Optional[list[float]] = None
    mount_rpy: Optional[list[float]] = None
    num_beams: Optional[int] = None
    fov_deg: Optional[float] = None
    max_range: Optional[float] = None
    enabled: Optional[bool] = None


class SensorUpdateRequest(BaseModel):
    name: Optional[str] = None
    parent_body: Optional[str] = None
    mount_pos: Optional[list[float]] = None
    mount_rpy: Optional[list[float]] = None
    enabled: Optional[bool] = None
    num_beams: Optional[int] = None
    fov_deg: Optional[float] = None
    max_range: Optional[float] = None
    min_range: Optional[float] = None
    noise_std: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _attachable_body_names() -> list[str]:
    """Bodies a sensor can attach to: the rover, any scene object (drones/boxes), and arm links."""
    names: list[str] = []
    if getattr(engine.project, "robot", None) is not None and getattr(engine.project, "robot_active", True):
        names.append("robot")
    for o in engine.project.objects:
        names.append(o.id)
    gr = getattr(engine.project, "generic_robot", None)
    if gr is not None:
        names.extend([l.name for l in gr.links])
    return names


@app.get("/api/sensors")
def list_sensors():
    """Returns all attached sensor instances and the bodies they can attach to."""
    return {
        "sensors": [s.model_dump() for s in (engine.project.sensors_list or [])],
        "attachable_bodies": _attachable_body_names(),
    }


@app.post("/api/sensors")
async def add_sensor(req: SensorCreateRequest):
    """Attaches a new sensor instance to a named body and rebuilds the live model."""
    import uuid as _uuid
    from backend.models import SensorInstance
    stype = (req.type or "imu").lower().strip()
    if stype not in ("lidar", "imu", "camera", "contact", "rangefinder"):
        raise HTTPException(status_code=400, detail=f"Unknown sensor type '{stype}'")
    defaults = {"lidar": [0.1, 0.0, 0.1], "rangefinder": [0.1, 0.0, 0.1],
                "imu": [0.0, 0.0, 0.05], "camera": [0.12, 0.0, 0.08], "contact": [0.0, 0.0, 0.0]}
    sid = f"sensor_{_uuid.uuid4().hex[:6]}"
    s = SensorInstance(
        id=sid,
        name=req.name or f"{stype.capitalize()} {_uuid.uuid4().hex[:3]}",
        type=stype,  # type: ignore
        parent_body=req.parent_body or "robot",
        mount_pos=req.mount_pos or defaults.get(stype, [0.0, 0.0, 0.05]),
        mount_rpy=req.mount_rpy or [0.0, 0.0, 0.0],
        enabled=True if req.enabled is None else bool(req.enabled),
    )
    if req.num_beams is not None: s.num_beams = int(req.num_beams)
    if req.fov_deg is not None: s.fov_deg = float(req.fov_deg)
    if req.max_range is not None: s.max_range = float(req.max_range)
    engine.project.sensors_list.append(s)
    project_manager.current_project.sensors_list = engine.project.sensors_list
    engine.rebuild_model(preserve_state=True)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return {"status": "ok", "sensor": s.model_dump()}


@app.post("/api/sensors/{sensor_id}")
async def update_sensor(sensor_id: str, req: SensorUpdateRequest):
    """Edits an existing sensor's attachment / mount / parameters."""
    s = next((x for x in engine.project.sensors_list if x.id == sensor_id), None)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found")
    for field in ("name", "parent_body", "mount_pos", "mount_rpy", "enabled",
                  "num_beams", "fov_deg", "max_range", "min_range", "noise_std", "width", "height"):
        val = getattr(req, field, None)
        if val is not None:
            setattr(s, field, val)
    project_manager.current_project.sensors_list = engine.project.sensors_list
    engine.rebuild_model(preserve_state=True)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return {"status": "ok", "sensor": s.model_dump()}


@app.delete("/api/sensors/{sensor_id}")
async def remove_sensor(sensor_id: str):
    """Detaches/removes a sensor instance."""
    before = len(engine.project.sensors_list)
    engine.project.sensors_list = [x for x in engine.project.sensors_list if x.id != sensor_id]
    project_manager.current_project.sensors_list = engine.project.sensors_list
    engine.rebuild_model(preserve_state=True)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return {"status": "ok", "removed": before - len(engine.project.sensors_list)}


@app.get("/api/sensors/readings")
def get_sensor_readings():
    """Returns the current live reading of every attached sensor."""
    return {"readings": engine.get_state().sensor_readings}


@app.get("/api/robot/generic")
def get_generic_robot():
    return {"robot": engine.project.generic_robot}


@app.post("/api/robot/generic")
async def update_generic_robot(robot: GenericRobotModel):
    try:
        val_res = validate_generic_robot(robot)
        if not val_res.valid:
            first_err = val_res.errors[0].message if val_res.errors else "Unknown validation failure"
            raise HTTPException(status_code=400, detail=f"Robot validation error: {first_err}")

        engine.project.generic_robot = robot
        project_manager.current_project.generic_robot = robot
        engine.rebuild_model(preserve_state=False)
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        engine.log("info", f"Updated generic robot '{robot.name}' in simulation scene.")
        return {"status": "updated", "robot": robot, "validation": val_res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/robot/generic")
async def clear_generic_robot():
    engine.project.generic_robot = None
    project_manager.current_project.generic_robot = None
    engine.rebuild_model(preserve_state=False)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    engine.log("info", "Removed imported generic robot model from simulation.")
    return {"status": "cleared"}


class RobotActiveRequest(BaseModel):
    active: bool


@app.post("/api/robot/active")
async def set_robot_active(req: RobotActiveRequest):
    """Enables or removes the built-in differential-drive robot from the workspace.

    Setting active=False removes the default robot entirely so the user can keep an empty
    workspace with nothing active. Setting active=True restores it.
    """
    engine.project.robot_active = req.active
    project_manager.current_project.robot_active = req.active
    engine.rebuild_model(preserve_state=False)
    project_manager.update_project(engine.project)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    engine.log("info", f"Built-in robot {'activated' if req.active else 'removed from workspace'}.")
    return {"status": "ok", "robot_active": engine.project.robot_active}


@app.delete("/api/robot")
async def remove_active_robot():
    """Convenience endpoint: removes the built-in differential-drive robot (robot_active=False)."""
    engine.project.robot_active = False
    project_manager.current_project.robot_active = False
    engine.rebuild_model(preserve_state=False)
    project_manager.update_project(engine.project)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    engine.log("info", "Built-in robot removed from workspace.")
    return {"status": "removed", "robot_active": False}


@app.post("/api/robot/validate", response_model=RobotValidationResult)
def validate_robot_endpoint(robot: GenericRobotModel):
    return validate_generic_robot(robot)


@app.post("/api/robot/auto-inertials", response_model=GenericRobotModel)
def auto_inertials_endpoint(robot: GenericRobotModel):
    return auto_calculate_robot_inertials(robot)


@app.post("/api/robot/kinematics/fk", response_model=FKResult)
def forward_kinematics_endpoint(req: FKRequest):
    return compute_forward_kinematics(req.robot, req.joint_positions)


@app.post("/api/robot/kinematics/ik", response_model=IKResult)
def inverse_kinematics_endpoint(req: IKRequest):
    return solve_inverse_kinematics(req)


@app.post("/api/robot/export-urdf")
def export_urdf_endpoint(robot: GenericRobotModel):
    urdf_content = export_robot_urdf(robot)
    return {"urdf": urdf_content, "filename": f"{robot.name.lower().replace(' ', '_')}.urdf"}


@app.get("/api/robot/presets")
def get_robot_presets():
    presets = [
        GenericRobotModel(
            name="2-Link Planar Arm",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            base_rpy=[0.0, 0.0, 0.0],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="cylinder", size=[0.18, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                    collision=LinkGeom(type="cylinder", size=[0.18, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                    inertial=LinkInertial(mass=4.0, origin_pos=[0, 0, 0.05], ixx=0.05, iyy=0.05, izz=0.05)
                ),
                LinkModel(
                    name="arm_link_1",
                    visual=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0.0], color="#2563eb"),
                    collision=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0.0], color="#2563eb"),
                    inertial=LinkInertial(mass=1.5, origin_pos=[0.20, 0, 0.0], ixx=0.02, iyy=0.02, izz=0.02)
                ),
                LinkModel(
                    name="arm_link_2",
                    visual=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0.0], color="#f97316"),
                    collision=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0.0], color="#f97316"),
                    inertial=LinkInertial(mass=0.8, origin_pos=[0.15, 0, 0.0], ixx=0.01, iyy=0.01, izz=0.01)
                )
            ],
            joints=[
                JointModel(
                    name="shoulder_joint",
                    type="revolute",
                    parent_link="base_link",
                    child_link="arm_link_1",
                    origin_pos=[0.0, 0.0, 0.12],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-3.14, upper=3.14, effort=50.0, velocity=5.0),
                    damping=0.2,
                    kp=80.0,
                    kv=8.0
                ),
                JointModel(
                    name="elbow_joint",
                    type="revolute",
                    parent_link="arm_link_1",
                    child_link="arm_link_2",
                    origin_pos=[0.40, 0.0, 0.0],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-2.6, upper=2.6, effort=30.0, velocity=5.0),
                    damping=0.15,
                    kp=60.0,
                    kv=6.0
                )
            ]
        ),
        GenericRobotModel(
            name="3-Link SCARA Arm",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            base_rpy=[0.0, 0.0, 0.0],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="cylinder", size=[0.22, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                    collision=LinkGeom(type="cylinder", size=[0.22, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                    inertial=LinkInertial(mass=6.0, origin_pos=[0, 0, 0.08], ixx=0.08, iyy=0.08, izz=0.08)
                ),
                LinkModel(
                    name="column_link",
                    visual=LinkGeom(type="cylinder", size=[0.08, 0.20], origin_pos=[0, 0, 0.20], color="#475569"),
                    collision=LinkGeom(type="cylinder", size=[0.08, 0.20], origin_pos=[0, 0, 0.20], color="#475569"),
                    inertial=LinkInertial(mass=3.0, origin_pos=[0, 0, 0.20], ixx=0.04, iyy=0.04, izz=0.02)
                ),
                LinkModel(
                    name="arm1_link",
                    visual=LinkGeom(type="box", size=[0.18, 0.04, 0.03], origin_pos=[0.18, 0, 0.0], color="#0ea5e9"),
                    collision=LinkGeom(type="box", size=[0.18, 0.04, 0.03], origin_pos=[0.18, 0, 0.0], color="#0ea5e9"),
                    inertial=LinkInertial(mass=1.6, origin_pos=[0.18, 0, 0.0], ixx=0.02, iyy=0.02, izz=0.01)
                ),
                LinkModel(
                    name="arm2_link",
                    visual=LinkGeom(type="box", size=[0.14, 0.035, 0.025], origin_pos=[0.14, 0, 0.0], color="#10b981"),
                    collision=LinkGeom(type="box", size=[0.14, 0.035, 0.025], origin_pos=[0.14, 0, 0.0], color="#10b981"),
                    inertial=LinkInertial(mass=0.9, origin_pos=[0.14, 0, 0.0], ixx=0.01, iyy=0.01, izz=0.01)
                )
            ],
            joints=[
                JointModel(
                    name="base_yaw_joint",
                    type="revolute",
                    parent_link="base_link",
                    child_link="column_link",
                    origin_pos=[0.0, 0.0, 0.16],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-3.14, upper=3.14, effort=80.0, velocity=4.0),
                    damping=0.3,
                    kp=100.0,
                    kv=10.0
                ),
                JointModel(
                    name="shoulder_joint",
                    type="revolute",
                    parent_link="column_link",
                    child_link="arm1_link",
                    origin_pos=[0.0, 0.0, 0.38],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-2.8, upper=2.8, effort=60.0, velocity=5.0),
                    damping=0.2,
                    kp=80.0,
                    kv=8.0
                ),
                JointModel(
                    name="elbow_joint",
                    type="revolute",
                    parent_link="arm1_link",
                    child_link="arm2_link",
                    origin_pos=[0.36, 0.0, 0.0],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-2.8, upper=2.8, effort=40.0, velocity=6.0),
                    damping=0.15,
                    kp=60.0,
                    kv=6.0
                )
            ]
        ),
        GenericRobotModel(
            name="4-DOF Articulated Arm",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            base_rpy=[0.0, 0.0, 0.0],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="cylinder", size=[0.20, 0.06], origin_pos=[0, 0, 0.06], color="#1e293b"),
                    collision=LinkGeom(type="cylinder", size=[0.20, 0.06], origin_pos=[0, 0, 0.06], color="#1e293b"),
                    inertial=LinkInertial(mass=5.0, origin_pos=[0, 0, 0.06], ixx=0.06, iyy=0.06, izz=0.06)
                ),
                LinkModel(
                    name="turret_link",
                    visual=LinkGeom(type="cylinder", size=[0.10, 0.10], origin_pos=[0, 0, 0.10], color="#334155"),
                    collision=LinkGeom(type="cylinder", size=[0.10, 0.10], origin_pos=[0, 0, 0.10], color="#334155"),
                    inertial=LinkInertial(mass=2.5, origin_pos=[0, 0, 0.10], ixx=0.03, iyy=0.03, izz=0.02)
                ),
                LinkModel(
                    name="upper_arm",
                    visual=LinkGeom(type="box", size=[0.05, 0.04, 0.20], origin_pos=[0, 0, 0.20], color="#3b82f6"),
                    collision=LinkGeom(type="box", size=[0.05, 0.04, 0.20], origin_pos=[0, 0, 0.20], color="#3b82f6"),
                    inertial=LinkInertial(mass=2.0, origin_pos=[0, 0, 0.20], ixx=0.03, iyy=0.03, izz=0.01)
                ),
                LinkModel(
                    name="forearm",
                    visual=LinkGeom(type="box", size=[0.04, 0.035, 0.16], origin_pos=[0, 0, 0.16], color="#8b5cf6"),
                    collision=LinkGeom(type="box", size=[0.04, 0.035, 0.16], origin_pos=[0, 0, 0.16], color="#8b5cf6"),
                    inertial=LinkInertial(mass=1.2, origin_pos=[0, 0, 0.16], ixx=0.015, iyy=0.015, izz=0.005)
                ),
                LinkModel(
                    name="end_effector",
                    visual=LinkGeom(type="cylinder", size=[0.03, 0.04], origin_pos=[0, 0, 0.04], color="#ec4899"),
                    collision=LinkGeom(type="cylinder", size=[0.03, 0.04], origin_pos=[0, 0, 0.04], color="#ec4899"),
                    inertial=LinkInertial(mass=0.5, origin_pos=[0, 0, 0.04], ixx=0.005, iyy=0.005, izz=0.002)
                )
            ],
            joints=[
                JointModel(
                    name="joint_1_yaw",
                    type="revolute",
                    parent_link="base_link",
                    child_link="turret_link",
                    origin_pos=[0.0, 0.0, 0.12],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-3.14, upper=3.14, effort=60.0, velocity=4.0),
                    damping=0.2,
                    kp=80.0,
                    kv=8.0
                ),
                JointModel(
                    name="joint_2_pitch",
                    type="revolute",
                    parent_link="turret_link",
                    child_link="upper_arm",
                    origin_pos=[0.0, 0.0, 0.20],
                    axis=[0.0, 1.0, 0.0],
                    limit=JointLimit(lower=-2.0, upper=2.0, effort=70.0, velocity=3.5),
                    damping=0.25,
                    kp=90.0,
                    kv=9.0
                ),
                JointModel(
                    name="joint_3_pitch",
                    type="revolute",
                    parent_link="upper_arm",
                    child_link="forearm",
                    origin_pos=[0.0, 0.0, 0.40],
                    axis=[0.0, 1.0, 0.0],
                    limit=JointLimit(lower=-2.5, upper=2.5, effort=45.0, velocity=4.5),
                    damping=0.15,
                    kp=70.0,
                    kv=7.0
                ),
                JointModel(
                    name="joint_4_wrist",
                    type="revolute",
                    parent_link="forearm",
                    child_link="end_effector",
                    origin_pos=[0.0, 0.0, 0.32],
                    axis=[0.0, 1.0, 0.0],
                    limit=JointLimit(lower=-2.8, upper=2.8, effort=20.0, velocity=6.0),
                    damping=0.1,
                    kp=50.0,
                    kv=5.0
                )
            ]
        ),
        GenericRobotModel(
            name="Prismatic Gantry Robot",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            base_rpy=[0.0, 0.0, 0.0],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="box", size=[0.60, 0.10, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                    collision=LinkGeom(type="box", size=[0.60, 0.10, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                    inertial=LinkInertial(mass=8.0, origin_pos=[0, 0, 0.05], ixx=0.1, iyy=0.1, izz=0.1)
                ),
                LinkModel(
                    name="slider_carriage",
                    visual=LinkGeom(type="box", size=[0.08, 0.08, 0.08], origin_pos=[0, 0, 0.08], color="#0284c7"),
                    collision=LinkGeom(type="box", size=[0.08, 0.08, 0.08], origin_pos=[0, 0, 0.08], color="#0284c7"),
                    inertial=LinkInertial(mass=2.5, origin_pos=[0, 0, 0.08], ixx=0.03, iyy=0.03, izz=0.03)
                ),
                LinkModel(
                    name="pendulum_arm",
                    visual=LinkGeom(type="box", size=[0.03, 0.03, 0.20], origin_pos=[0, 0, -0.20], color="#f59e0b"),
                    collision=LinkGeom(type="box", size=[0.03, 0.03, 0.20], origin_pos=[0, 0, -0.20], color="#f59e0b"),
                    inertial=LinkInertial(mass=1.2, origin_pos=[0, 0, -0.20], ixx=0.02, iyy=0.02, izz=0.01)
                )
            ],
            joints=[
                JointModel(
                    name="gantry_slide_joint",
                    type="prismatic",
                    parent_link="base_link",
                    child_link="slider_carriage",
                    origin_pos=[0.0, 0.0, 0.10],
                    axis=[1.0, 0.0, 0.0],
                    limit=JointLimit(lower=-0.50, upper=0.50, effort=120.0, velocity=2.0),
                    damping=0.5,
                    kp=200.0,
                    kv=20.0
                ),
                JointModel(
                    name="arm_swing_joint",
                    type="revolute",
                    parent_link="slider_carriage",
                    child_link="pendulum_arm",
                    origin_pos=[0.0, 0.0, 0.05],
                    axis=[0.0, 1.0, 0.0],
                    limit=JointLimit(lower=-2.5, upper=2.5, effort=35.0, velocity=5.0),
                    damping=0.15,
                    kp=60.0,
                    kv=6.0
                )
            ]
        ),
        GenericRobotModel(
            name="Continuous Rotary Turret",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            base_rpy=[0.0, 0.0, 0.0],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="cylinder", size=[0.25, 0.06], origin_pos=[0, 0, 0.06], color="#1e293b"),
                    collision=LinkGeom(type="cylinder", size=[0.25, 0.06], origin_pos=[0, 0, 0.06], color="#1e293b"),
                    inertial=LinkInertial(mass=6.0, origin_pos=[0, 0, 0.06], ixx=0.08, iyy=0.08, izz=0.08)
                ),
                LinkModel(
                    name="rotor_plate",
                    visual=LinkGeom(type="cylinder", size=[0.18, 0.04], origin_pos=[0, 0, 0.16], color="#06b6d4"),
                    collision=LinkGeom(type="cylinder", size=[0.18, 0.04], origin_pos=[0, 0, 0.16], color="#06b6d4"),
                    inertial=LinkInertial(mass=2.0, origin_pos=[0, 0, 0.16], ixx=0.03, iyy=0.03, izz=0.04)
                ),
                LinkModel(
                    name="pointer_mast",
                    visual=LinkGeom(type="box", size=[0.02, 0.02, 0.15], origin_pos=[0.12, 0, 0.35], color="#f43f5e"),
                    collision=LinkGeom(type="box", size=[0.02, 0.02, 0.15], origin_pos=[0.12, 0, 0.35], color="#f43f5e"),
                    inertial=LinkInertial(mass=0.5, origin_pos=[0.12, 0, 0.35], ixx=0.005, iyy=0.005, izz=0.001)
                )
            ],
            joints=[
                JointModel(
                    name="turret_continuous_joint",
                    type="continuous",
                    parent_link="base_link",
                    child_link="rotor_plate",
                    origin_pos=[0.0, 0.0, 0.12],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-3.14, upper=3.14, effort=40.0, velocity=8.0),
                    damping=0.2,
                    actuator_mode="velocity",
                    kp=0.0,
                    kv=15.0
                ),
                JointModel(
                    name="mast_mount_fixed",
                    type="fixed",
                    parent_link="rotor_plate",
                    child_link="pointer_mast",
                    origin_pos=[0.0, 0.0, 0.20],
                    axis=[0.0, 0.0, 1.0]
                )
            ]
        )
    ]
    return {"presets": [p.model_dump() for p in presets]}


class AutoDriveRequest(BaseModel):
    enabled: bool

@app.post("/api/simulation/auto-drive")
async def set_auto_drive(req: AutoDriveRequest):
    global auto_drive_active
    auto_drive_active = req.enabled
    engine.log("info", f"Auto-Drive mode {'enabled' if auto_drive_active else 'disabled'}.")
    await broadcast_json({"type": "auto_drive_changed", "data": {"enabled": auto_drive_active}})
    return {"status": "ok", "enabled": auto_drive_active}

@app.get("/api/simulation/auto-drive")
def get_auto_drive():
    global auto_drive_active
    return {"enabled": auto_drive_active}


@app.get("/api/logs")
def get_logs():
    return {"logs": recent_logs}


@app.delete("/api/logs")
def clear_logs():
    recent_logs.clear()
    return {"status": "cleared"}


# =====================================================================
# Experiment & Scenario Management Endpoints
# =====================================================================

@app.get("/api/experiments/scenarios")
def list_scenarios():
    return {"scenarios": experiment_manager.list_scenarios()}


@app.post("/api/experiments/scenarios")
def create_scenario(config: ExperimentConfig):
    try:
        saved_path = experiment_manager.save_scenario(config)
        return {"status": "saved", "path": saved_path, "scenario": config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/experiments/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    scen = experiment_manager.get_scenario(scenario_id)
    if not scen:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scen


@app.post("/api/experiments/scenarios/{scenario_id}/duplicate")
def duplicate_scenario(scenario_id: str, req: DuplicateScenarioRequest = DuplicateScenarioRequest()):
    try:
        duplicated = experiment_manager.duplicate_scenario(scenario_id, req.new_name)
        return duplicated
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Scenario not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/experiments/scenarios/{scenario_id}")
def delete_scenario(scenario_id: str):
    success = experiment_manager.delete_scenario(scenario_id)
    if not success:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"status": "deleted", "id": scenario_id}


@app.post("/api/experiments/scenarios/{scenario_id}/load-to-scene")
async def load_scenario_to_scene(scenario_id: str):
    scen = experiment_manager.get_scenario(scenario_id)
    if not scen:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    project_manager.current_project.robot = scen.robot.model_copy(deep=True)
    project_manager.current_project.objects = [o.model_copy(deep=True) for o in scen.environment]
    project_manager.current_project.settings = scen.settings.model_copy(deep=True)
    project_manager.current_project.name = scen.name
    
    engine.project = project_manager.current_project
    engine.rebuild_model(preserve_state=False)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return {"status": "loaded", "project": project_manager.current_project, "scenario": scen}


@app.post("/api/experiments/save-from-active")
def save_scenario_from_active(req: SaveActiveScenarioRequest):
    curr = project_manager.get_current()
    scen = ExperimentConfig(
        name=req.name,
        description=req.description,
        seed=req.seed,
        robot=curr.robot.model_copy(deep=True),
        environment=[o.model_copy(deep=True) for o in curr.objects],
        settings=curr.settings.model_copy(deep=True),
        task=TaskObjective(
            target_position=req.target_position,
            distance_threshold=req.distance_threshold,
            timeout=req.timeout,
            control_mode="goal_seeking"
        ),
        duration=req.timeout
    )
    experiment_manager.save_scenario(scen)
    return scen


@app.put("/api/experiments/scenarios/{scenario_id}")
def update_scenario(scenario_id: str, config: ExperimentConfig):
    config.id = scenario_id
    path = experiment_manager.save_scenario(config)
    return {"status": "saved", "path": path, "scenario": config}


@app.post("/api/experiments/train-policy")
def train_policy(req: TrainPolicyRequest):
    try:
        res = experiment_manager.train_avoidance_policy(req.scenario_id)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Policy training failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Policy training failed: {str(e)}")


@app.post("/api/experiments/run")
def run_experiment(req: RunExperimentRequest):
    try:
        config = None
        if req.config:
            config = req.config
        elif req.scenario_id:
            config = experiment_manager.get_scenario(req.scenario_id)
            if not config:
                raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")
        else:
            raise HTTPException(status_code=400, detail="Provide either 'scenario_id' or 'config'")

        result = experiment_manager.run_experiment(
            config=config,
            seed_override=req.seed,
            duration_override=req.duration
        )
        engine.log("info", f"Experiment '{config.name}' (seed={result.seed}) finished: {result.metrics.status_reason}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Experiment execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Experiment execution failed: {str(e)}")


@app.post("/api/experiments/run-live")
async def run_live_experiment(req: RunExperimentRequest):
    global active_live_experiment
    config = None
    if req.config:
        config = req.config
    elif req.scenario_id:
        config = experiment_manager.get_scenario(req.scenario_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")
    else:
        raise HTTPException(status_code=400, detail="Provide either 'scenario_id' or 'config'")

    seed = req.seed if req.seed is not None else config.seed
    duration = req.duration if req.duration is not None else config.duration
    
    # Load scenario into active project
    project_manager.current_project.robot = config.robot.model_copy(deep=True)
    project_manager.current_project.objects = [o.model_copy(deep=True) for o in config.environment]
    project_manager.current_project.settings = config.settings.model_copy(deep=True)
    project_manager.current_project.name = config.name

    np.random.seed(seed)
    engine.project = project_manager.current_project
    engine.rebuild_model(preserve_state=False)

    run_until_goal = getattr(req, "run_until_goal", True)
    timeout_val = None if (run_until_goal or not duration or duration <= 0) else duration

    active_live_experiment = {
        "config": config,
        "seed": seed,
        "duration": duration,
        "timeout": timeout_val,
        "run_until_goal": run_until_goal,
        "target": np.array(config.task.target_position[:2]),
        "measurements": [],
        "unique_collision_events": 0,
        "in_collision": False,
        "time_to_goal": None,
        "next_sample_time": 0.0,
    }

    engine.play()
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    await broadcast_json({
        "type": "live_experiment_started",
        "data": {
            "experiment_id": config.id,
            "target_position": config.task.target_position,
            "name": config.name,
            "seed": seed,
        }
    })
    return {"status": "started", "experiment_id": config.id, "target_position": config.task.target_position}


@app.get("/api/experiments/runs")
def list_runs():
    return {"runs": experiment_manager.list_runs()}


@app.get("/api/experiments/runs/{run_id}")
def get_run(run_id: str):
    res = experiment_manager.get_run_result(run_id)
    if not res:
        raise HTTPException(status_code=404, detail="Run not found")
    return res


@app.post("/api/experiments/compare")
def compare_runs(req: CompareRunsRequest):
    try:
        comparison = experiment_manager.compare_runs(req.run_a_id, req.run_b_id)
        return comparison
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
# Reinforcement Learning & Gymnasium Routes
# ==========================================

@app.post("/api/rl/train")
def start_rl_training(config: RLTrainingConfig, request: Request):
    credit_hold = check_and_reserve_credits(request, "rl_train_background")
    try:
        proj = project_manager.get_current()
        if config.scenario_id:
            scen = experiment_manager.get_scenario(config.scenario_id)
            if scen:
                proj = proj.model_copy(deep=True)
                proj.objects = [o.model_copy(deep=True) for o in scen.environment]
                proj.robot = scen.robot.model_copy(deep=True)
        status = training_manager.start_training(config, proj)
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return status
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        logger.exception("Failed to start RL training: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/train/{job_id}/status")
def get_rl_training_status(job_id: str):
    status = training_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training job not found")
    return status


@app.post("/api/rl/train/{job_id}/cancel")
def cancel_rl_training(job_id: str):
    cancelled = training_manager.cancel_training(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Job could not be cancelled (may be completed or non-existent)")
    return {"status": "cancelled", "job_id": job_id}


@app.post("/api/rl/stop-all")
async def stop_all_rl():
    """Single 'Stop' action for the viewport HUD. Cancels any queued/running training job
    (both the background job-manager jobs AND the live episodic 'watch it learn' curriculum),
    and disengages any deployed RL policy so the robot halts cleanly. Also handles orphaned/
    stale jobs that show progress with no active robot."""
    cancelled_jobs = []
    try:
        for job in training_manager.list_jobs():
            if job.status in ("queued", "training"):
                if training_manager.cancel_training(job.job_id):
                    cancelled_jobs.append(job.job_id)
    except Exception as e:
        logger.warning("stop-all: job cancel sweep failed: %s", e)

    # Stop the live episodic curriculum (it polls engine.rl_training_active each step).
    training_was_live = bool(getattr(engine, "rl_training_active", False))
    try:
        engine.rl_training_active = False
        engine.rl_training_rollout_active = True
        engine.rl_training_status = "Training stopped by user."
        engine.rl_training_percent = 0.0
    except Exception as e:
        logger.warning("stop-all: clearing live training flags failed: %s", e)

    # Disengage any deployed (non-training) policy / autonomous drive.
    policy_was_active = bool(getattr(engine, "active_rl_policy_model", None) is not None)
    try:
        engine.disengage_rl_policy()
    except Exception as e:
        logger.warning("stop-all: disengage policy failed: %s", e)

    try:
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    except Exception:
        pass

    return {
        "status": "stopped",
        "cancelled_jobs": cancelled_jobs,
        "training_was_live": training_was_live,
        "policy_was_active": policy_was_active,
    }


class LiveDroneTrainingRequest(BaseModel):
    drone_identifier: str = "drone"
    target_altitude: float = 1.5
    total_episodes: int = 4


class LiveArmTrainingRequest(BaseModel):
    arm_identifier: str = "arm"
    total_episodes: int = 4


class LiveRoverTrainingRequest(BaseModel):
    robot_identifier: str = "robot"
    total_episodes: int = 4
    task_description: str = ""
    reward_code: Optional[str] = None
    domain_randomize: bool = False


@app.post("/api/rl/train-drone-live")
async def train_drone_live_endpoint(request: Request, req: LiveDroneTrainingRequest = LiveDroneTrainingRequest()):
    """Starts interactive visual Isaac Lab drone RL training directly in the active 3D simulation scene."""
    credit_hold = check_and_reserve_credits(request, "rl_train_live")
    try:
        res = training_manager.start_live_interactive_drone_training(
            engine=engine,
            drone_identifier=req.drone_identifier,
            target_altitude=req.target_altitude,
            total_episodes=req.total_episodes
        )
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise


@app.post("/api/rl/train-arm-live")
async def train_arm_live_endpoint(request: Request, req: LiveArmTrainingRequest = LiveArmTrainingRequest()):
    """Starts interactive visual Isaac Lab robotic arm pick-and-place training directly in the active 3D simulation scene."""
    credit_hold = check_and_reserve_credits(request, "rl_train_live")
    try:
        res = training_manager.start_live_interactive_arm_training(
            engine=engine,
            arm_identifier=req.arm_identifier,
            total_episodes=req.total_episodes
        )
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise


@app.post("/api/rl/train-rover-live")
async def train_rover_live_endpoint(request: Request, req: LiveRoverTrainingRequest = LiveRoverTrainingRequest()):
    """Starts interactive visual Isaac Lab ground rover navigation training directly in the active 3D simulation scene."""
    credit_hold = check_and_reserve_credits(request, "rl_train_live")
    try:
        res = training_manager.start_live_interactive_rover_training(
            engine=engine,
            robot_identifier=req.robot_identifier,
            total_episodes=req.total_episodes,
            task_description=req.task_description,
            reward_code=req.reward_code,
            domain_randomize=req.domain_randomize,
        )
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise



@app.get("/api/rl/models")
def list_rl_models():
    return {"models": training_manager.list_models()}


@app.get("/api/rl/models/{model_id}/export")
def export_rl_policy_package(model_id: str):
    """Download a portable package (ONNX policy + ROS 2 node + README) for a trained policy,
    ready to run on a real robot."""
    from fastapi.responses import Response
    from backend.policy_export import build_policy_package
    model = training_manager.load_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Policy not found or could not be loaded.")
    meta = training_manager.get_model_metadata(model_id) or type("M", (), {"model_dump": lambda self: {"model_id": model_id}})()
    try:
        data = build_policy_package(model, meta, engine.project)
    except Exception as e:
        logger.exception("Policy export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{model_id}_policy_package.zip"'},
    )


class RobustnessRequest(BaseModel):
    episodes: int = Field(default=20, ge=4, le=100)
    compare_baseline: bool = True


@app.post("/api/rl/models/{model_id}/robustness")
def rl_policy_robustness(model_id: str, req: RobustnessRequest = RobustnessRequest()):
    """Robustness scorecard: evaluate the policy across domain-randomized worlds (friction, mass,
    gravity, sensor noise) and report reliability vs a clean-physics baseline."""
    from backend.robustness_eval import evaluate_robustness
    res = evaluate_robustness(model_id, engine.project, episodes=req.episodes,
                              compare_baseline=req.compare_baseline)
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("detail", "Evaluation failed"))
    return res


@app.get("/api/rl/models/{model_id}")
def get_rl_model(model_id: str):
    meta = training_manager.get_model_metadata(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Model not found")
    return meta


@app.post("/api/rl/evaluate")
def evaluate_rl_policy(req: RLEvaluationRequest):
    try:
        res = RLEvaluator.evaluate_suite(req.model_id, seed=req.seed)
        return res
    except Exception as e:
        logger.exception("Evaluation suite failure: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/evaluations")
def list_rl_evaluations():
    return {"evaluations": RLEvaluator.list_evaluations()}


@app.get("/api/rl/evaluations/{eval_id}")
def get_rl_evaluation(eval_id: str):
    res = RLEvaluator.get_evaluation(eval_id)
    if not res:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return res


@app.post("/api/rl/execute-live")
async def execute_rl_policy_live(req: ExecuteRLPolicyRequest):
    global active_live_experiment
    model = training_manager.load_model(req.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"RL Model '{req.model_id}' checkpoint not found")

    meta = training_manager.get_model_metadata(req.model_id)
    raw_kind = (getattr(meta, "robot_kind", None) or "").strip().lower() if meta else ""
    if raw_kind == "mobile":
        raw_kind = "rover"
    if raw_kind in ("drone", "arm", "generic", "rover"):
        # Trust an explicitly-stored, valid kind -- the name must not override it.
        robot_kind = raw_kind
    else:
        # Infer only when the stored kind is missing/unknown (legacy checkpoints).
        _nm = ((meta.name if meta else "") or req.model_id).lower()
        if any(k in _nm for k in ("drone", "hover", "flight", "quad", "takeoff", "fly")):
            robot_kind = "drone"
        elif any(k in _nm for k in ("arm", "reach", "pick", "manipulat", "gripper")):
            robot_kind = "arm"
        else:
            robot_kind = "rover"

    # -------------------------------------------------------------------
    # DRONE / ARM / GENERIC policies: drive the entity through the live
    # physics controller (engine.apply_control dispatches by robot_kind).
    # These do NOT use the diff-drive active_live_experiment loop.
    # -------------------------------------------------------------------
    if robot_kind in ("drone", "arm", "generic"):
        # Make sure the live experiment rover loop is not also running.
        active_live_experiment = None
        proj = project_manager.current_project

        if robot_kind == "drone":
            # Ensure a drone entity exists in the scene; spawn one if missing so the policy has a
            # body to fly. (A trained drone policy references a drone, not the rover.)
            drone_obj = next((o for o in proj.objects
                              if "drone" in o.name.lower() or o.id.startswith("drone") or "quad" in o.name.lower()), None)
            if not drone_obj:
                import uuid as _uuid
                drone_obj = SceneObject(
                    id=f"drone_{_uuid.uuid4().hex[:6]}", name="RL-Quadcopter", type="box",
                    position=[0.0, 0.0, 0.12], size=[0.22, 0.22, 0.08],
                    color="#38bdf8", is_static=False, mass=1.2,
                )
                proj.objects.append(drone_obj)
            engine.project = proj
            engine.rebuild_model(preserve_state=False)
            res = engine.set_active_rl_policy(req.model_id)
            # Bind explicitly to the drone currently in the scene.
            engine.active_rl_policy_kind = "drone"
            engine.active_rl_policy_target = drone_obj.id
            if drone_obj.id in engine.drone_controllers:
                ctrl = engine.drone_controllers[drone_obj.id]
                ctrl.rotors_active = True
                ctrl.flight_mode = "flying"
                if req.target_position and len(req.target_position) >= 3:
                    ctrl.target_position = list(req.target_position)
                    ctrl.target_altitude = float(req.target_position[2])
                else:
                    ctrl.target_altitude = 1.5
                    ctrl.target_position = [0.0, 0.0, 1.5]
        else:
            # Arm / generic: policy drives the existing articulated robot in the scene.
            engine.rebuild_model(preserve_state=False)
            # Seed the authoritative target BEFORE binding the driver env so the deployed arm
            # chases the trained goal (or an explicit request target), not a random point.
            if req.target_position and len(req.target_position) >= 3:
                engine.active_rl_policy_target_pos = list(req.target_position)
            elif meta is not None and getattr(meta, "target_position", None):
                engine.active_rl_policy_target_pos = list(meta.target_position)
            res = engine.set_active_rl_policy(req.model_id)
            # If the request specified a target, re-apply it (set_active_rl_policy may have
            # resampled from metadata) and push it into the live driver env.
            if req.target_position and len(req.target_position) >= 3:
                engine.active_rl_policy_target_pos = list(req.target_position)
                drv = getattr(engine, "active_rl_policy_env", None)
                if drv is not None and hasattr(drv, "target_position"):
                    import numpy as _np
                    drv.target_position = _np.array(req.target_position[:3], dtype=_np.float32)

        engine.last_teleop_time = 0.0  # allow apply_control's policy branch to run immediately
        engine.play()
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        await broadcast_json({
            "type": "live_experiment_started",
            "data": {
                "experiment_id": f"rl_{robot_kind}",
                "name": f"{(meta.name if meta else req.model_id)} [{robot_kind} policy]",
                "policy_mode": "PPO",
                "robot_kind": robot_kind,
            }
        })
        return {"status": "started", "model_id": req.model_id, "robot_kind": robot_kind,
                "engaged": res.get("status") == "success"}

    # -------------------------------------------------------------------
    # ROVER / diff-drive policies: existing live-experiment loop.
    # -------------------------------------------------------------------
    config = None
    if req.scenario_id:
        config = experiment_manager.get_scenario(req.scenario_id)
    if not config:
        scens = experiment_manager.list_scenarios()
        if scens:
            config = scens[0]
        else:
            raise HTTPException(status_code=404, detail="No baseline scenario available to execute")

    config = config.model_copy(deep=True)
    # Prefer an explicit request target; else the policy's own trained target (so deploy matches
    # the training distribution and the robot actually moves); else the scenario default.
    if req.target_position:
        config.task.target_position = req.target_position
    elif meta is not None and getattr(meta, "target_position", None):
        config.task.target_position = list(meta.target_position)

    seed = req.seed
    duration = req.duration

    project_manager.current_project.robot = config.robot.model_copy(deep=True)
    project_manager.current_project.robot_active = True  # a rover policy needs the diff-drive robot
    project_manager.current_project.objects = [o.model_copy(deep=True) for o in config.environment]
    project_manager.current_project.settings = config.settings.model_copy(deep=True)
    project_manager.current_project.name = f"{config.name} (PPO Policy)"

    np.random.seed(seed)
    engine.project = project_manager.current_project
    engine.rebuild_model(preserve_state=False)
    # Engage the policy on the engine too (so apply_control's rover branch can drive it with the
    # authoritative target) AND keep the live-experiment loop as the metrics/termination tracker.
    engine.set_active_rl_policy(req.model_id)
    engine.active_rl_policy_kind = "rover"
    engine.active_rl_policy_target_pos = list(config.task.target_position)
    engine.last_teleop_time = 0.0

    run_until_goal = getattr(req, "run_until_goal", True)
    timeout_val = None if (run_until_goal or not duration or duration <= 0) else duration

    active_live_experiment = {
        "config": config,
        "seed": seed,
        "duration": duration,
        "timeout": timeout_val,
        "run_until_goal": run_until_goal,
        "target": np.array(config.task.target_position[:2]),
        "measurements": [],
        "unique_collision_events": 0,
        "in_collision": False,
        "time_to_goal": None,
        "next_sample_time": 0.0,
        "rl_model": model,
        "rl_model_id": req.model_id,
    }

    engine.play()
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    await broadcast_json({
        "type": "live_experiment_started",
        "data": {
            "experiment_id": config.id,
            "target_position": config.task.target_position,
            "name": f"{config.name} [PPO Policy: {req.model_id}]",
            "seed": seed,
            "policy_mode": "PPO",
        }
    })
    return {"status": "started", "model_id": req.model_id, "target_position": config.task.target_position}


# ==========================================
# AI Experiment Assistant & Refinement Routes
# ==========================================

@app.get("/api/ai/status")
def get_ai_status():
    """Returns AI reasoning provider configuration, credentials presence, and available tools."""
    service = get_ai_reasoning_service(project_manager, engine=engine)
    return service.get_provider_status()


@app.get("/api/ai/models")
def get_ai_models():
    """Exposes discovered local and cloud models and active provider endpoint."""
    service = get_ai_reasoning_service(project_manager, engine=engine)
    status = service.get_provider_status()
    return {
        "active_provider": status.get("active_provider"),
        "model": status.get("model"),
        "endpoint": status.get("endpoint", "local"),
        "available_models": status.get("available_models", []),
        "provider_type": status.get("provider_type"),
        "offline_mode": status.get("offline_mode", True),
    }


@app.post("/api/ai/interpret-prompt")
def interpret_experiment_prompt(req: AIAssistantPromptRequest, request: Request):
    credit_hold = check_and_reserve_credits(request, "ai_interpret_prompt")
    try:
        res = AIExperimentAssistant.process_prompt(req.prompt, req.base_scenario_id)
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        logger.exception("AI assistant interpretation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai/chat", response_model=AIChatResponse)
async def ai_conversational_chat(req: AIChatRequest, request: Request):
    """Executes a telemetry-grounded conversational agent loop supporting supervised & autonomous modes, with multi-turn memory."""
    credit_hold = check_and_reserve_credits(request, "ai_chat")
    try:
        session_id = req.session_id or "default_session"
        # If frontend didn't supply chat_history, load from persistent memory
        if not req.chat_history:
            req.chat_history = chat_memory.get_history(session_id, limit=20)

        # Record user turn
        chat_memory.append_message(session_id=session_id, role="user", content=req.prompt)

        service = get_ai_reasoning_service(project_manager, engine=engine, agent_interface=agent_interface)
        res = service.chat(req)

        # If any actions altered the scene or physics, broadcast updated project state
        if res.actions_executed:
            try:
                await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
            except Exception as b_err:
                logger.warning("Failed to broadcast project_updated after AI chat action: %s", b_err)

        # Record assistant reply and actions
        chat_memory.append_message(
            session_id=session_id,
            role="assistant",
            content=res.reply,
            response_data=res.model_dump(),
            actions_executed=res.actions_executed
        )
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except HTTPException:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        raise
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        logger.exception("AI chat reasoning error: %s", e)
        # Never surface a raw 500 to the chat UI. Degrade to a graceful assistant message so the
        # conversation keeps working even if one reasoning path hit an unexpected error.
        try:
            fallback = AIChatResponse(
                reply=(
                    "I hit an internal error while processing that request, so I couldn't complete it. "
                    "Please try rephrasing, or ask me to do something specific in the scene "
                    "(e.g. 'spawn a box', 'train a PPO policy on this robot', 'reset the simulation')."
                ),
                provider="RoboForge",
                is_model_inference=False,
                is_fallback=True,
                mode=getattr(req, "mode", "autonomous"),
                thought_steps=[],
                tools_invoked=[],
                actions_executed=[],
            )
            return fallback
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/chat/history")
def get_ai_chat_history(session_id: str = "default_session", limit: int = 50):
    """Retrieves stored multi-turn conversation messages for a session."""
    return {"history": chat_memory.get_history(session_id, limit=limit)}


@app.delete("/api/ai/chat/history")
def delete_ai_chat_history(session_id: str = "default_session"):
    """Clears stored multi-turn conversation messages for a session."""
    chat_memory.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/ai/approve-action", response_model=AIChatResponse)
async def ai_approve_action(req: AIApprovalRequest):
    """Handles supervisor approval or rejection for a paused experiment plan."""
    try:
        service = get_ai_reasoning_service(project_manager, engine=engine)
        res = service.approve_action(req)
        if res.actions_executed:
            try:
                await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
            except Exception as b_err:
                logger.warning("Failed to broadcast project_updated after AI approval action: %s", b_err)
        return res
    except Exception as e:
        logger.exception("AI approval execution error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ai/cancel-mission")
async def ai_cancel_mission(request: Request):
    """Cancels an active or pending research mission from the chat workspace."""
    try:
        data = await request.json()
        mission_id = data.get("mission_id")
        reason = data.get("reason", "Cancelled by user in AI chat")
        if not mission_id:
            raise HTTPException(status_code=400, detail="mission_id is required")
        service = get_ai_reasoning_service(project_manager, engine=engine)
        return service.cancel_mission(mission_id, reason)
    except Exception as e:
        logger.exception("AI mission cancel error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ai/apply-config")
async def apply_ai_configuration(request: Request):
    try:
        data = await request.json()
        if "suggested_configuration" in data and isinstance(data["suggested_configuration"], dict):
            config_dict = data["suggested_configuration"]
        elif "config" in data and isinstance(data["config"], dict):
            config_dict = data["config"]
        else:
            config_dict = data

        config_dict = dict(config_dict)
        config_dict.pop("recommended_policy", None)

        validated_config = ExperimentConfig(**config_dict)
        experiment_manager.save_scenario(validated_config)

        project_manager.current_project.robot = validated_config.robot.model_copy(deep=True)
        project_manager.current_project.objects = [o.model_copy(deep=True) for o in validated_config.environment]
        project_manager.current_project.settings = validated_config.settings.model_copy(deep=True)
        project_manager.current_project.name = validated_config.name

        engine.project = project_manager.current_project
        engine.rebuild_model(preserve_state=False)

        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return {"status": "applied", "scenario_id": validated_config.id, "scenario": validated_config}
    except Exception as e:
        logger.exception("Failed to apply AI configuration: %s", e)
        raise HTTPException(status_code=400, detail=f"Configuration validation failed: {str(e)}")


@app.post("/api/ai/refine-hypothesis")
def refine_hypothesis(req: HypothesisRefinementRequest, request: Request):
    credit_hold = check_and_reserve_credits(request, "ai_refine_hypothesis")
    try:
        scen_id = req.scenario_id or ""
        res = HypothesisRefinementEngine.analyze_and_refine(scen_id, req.focus_metric)
        if credit_hold:
            credit_service.commitCredits(credit_hold[1])
        return res
    except Exception as e:
        if credit_hold:
            credit_service.releaseCredits(credit_hold[1])
        logger.exception("Hypothesis refinement error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ai/hypotheses")
def list_hypotheses():
    return {"hypotheses": HypothesisRefinementEngine.list_hypotheses()}


# -------------------------------------------------------------
# Asset Library Endpoints
# -------------------------------------------------------------
@app.get("/api/assets")
def list_assets(
    category: Optional[str] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None
):
    assets = asset_library.list_assets(category=category, search=search, tag=tag)
    return {"assets": [a.model_dump() for a in assets], "total": len(assets)}

@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    asset = asset_library.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
    return asset.model_dump()

@app.post("/api/assets")
def create_asset(req: CreateAssetRequest):
    try:
        asset = asset_library.save_custom_asset(req)
        return asset.model_dump()
    except Exception as e:
        logger.exception("Error saving custom asset: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str):
    try:
        deleted = asset_library.delete_custom_asset(asset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
        return {"status": "deleted", "id": asset_id}
    except ValueError as ve:
        raise HTTPException(status_code=403, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/assets/{asset_id}/duplicate")
def duplicate_asset(asset_id: str, req: dict = None):
    try:
        new_name = req.get("name") if req else None
        clone = asset_library.duplicate_asset(asset_id, new_name=new_name)
        return clone.model_dump()
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/assets/{asset_id}/apply")
async def apply_asset(asset_id: str, req: ApplyAssetRequest = ApplyAssetRequest()):
    try:
        curr = project_manager.get_current()
        updated = asset_library.apply_asset_to_project(
            asset_id,
            curr,
            position_offset=req.position_offset
        )
        project_manager.update_project(updated)
        engine.rebuild_model(updated)
        await broadcast_json({"type": "project_updated", "data": updated.model_dump()})
        engine.log("info", f"Applied asset '{asset_id}' to active simulation scene.")
        return {
            "status": "applied",
            "asset_id": asset_id,
            "project": updated.model_dump()
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error applying asset: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/assets/validate-robot")
def validate_robot(req: dict):
    robot_data = req.get("robot", {})
    robot_type = req.get("type", "mobile")
    res = asset_library.validate_robot_model(robot_data, robot_type=robot_type)
    return res.model_dump()

@app.post("/api/assets/validate-environment")
def validate_environment(req: dict):
    env_data = req.get("environment", {})
    res = asset_library.validate_environment_model(env_data)
    return res.model_dump()

@app.post("/api/assets/save-current-robot")
def save_current_robot_asset(req: dict):
    try:
        name = req.get("name", "Custom Robot")
        desc = req.get("description", "Exported from active simulation")
        curr = project_manager.get_current()
        
        if curr.generic_robot:
            data = {"type": "articulated", "config": curr.generic_robot.model_dump()}
            hint = "arm"
        else:
            data = {"type": "mobile", "config": curr.robot.model_dump()}
            hint = "bot"

        c_req = CreateAssetRequest(
            name=name,
            category="robot",
            description=desc,
            tags=["custom", "user-created"],
            data=data,
            thumbnail_hint=hint
        )
        asset = asset_library.save_custom_asset(c_req)
        return asset.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/assets/save-robot-model")
def save_robot_model_asset(req: dict):
    """Saves an articulated robot model directly to the Asset Library."""
    try:
        name = req.get("name", "Custom Articulated Robot")
        desc = req.get("description", "Designed in Robot Modeling Studio")
        tags = req.get("tags", ["custom", "articulated", "robot-studio"])
        robot_data = req.get("robot")
        if not robot_data:
            raise HTTPException(status_code=400, detail="Missing 'robot' payload.")

        gr = GenericRobotModel.model_validate(robot_data)
        val_res = validate_generic_robot(gr)
        if not val_res.valid:
            first_err = val_res.errors[0].message if val_res.errors else "Invalid robot configuration."
            raise HTTPException(status_code=400, detail=f"Robot validation failed: {first_err}")

        c_req = CreateAssetRequest(
            name=name,
            category="robot",
            description=desc,
            tags=tags,
            data={"type": "articulated", "config": gr.model_dump()},
            thumbnail_hint="arm"
        )
        asset = asset_library.save_custom_asset(c_req)
        return asset.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/assets/save-current-scene")
def save_current_scene_template(req: dict):
    try:
        name = req.get("name", "Custom Scene Template")
        desc = req.get("description", "Saved environment & robot template")
        curr = project_manager.get_current()
        
        c_req = CreateAssetRequest(
            name=name,
            category="template",
            description=desc,
            tags=["custom", "template", "user-created"],
            data=curr.model_dump(),
            thumbnail_hint="template"
        )
        asset = asset_library.save_custom_asset(c_req)
        return asset.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# =====================================================================
# RoboForge Scientific Intelligence Agent API Interface
# =====================================================================

@app.get("/api/agent/inspect", response_model=EnvironmentInspectionResult)
def agent_inspect_environment():
    return agent_interface.inspect_environment()

@app.get("/api/agent/telemetry", response_model=TelemetryObservation)
def agent_get_telemetry():
    return agent_interface.get_telemetry()

@app.api_route("/api/agent/perception", methods=["GET", "POST"], response_model=PerceptionObservation)
def agent_capture_perception(
    req: Optional[CameraCaptureRequest] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    include_segmentation: bool = True,
    include_depth: bool = True
):
    w = (req.width if req else None) or width or 320
    h = (req.height if req else None) or height or 240
    seg = req.render_segmentation if req else include_segmentation
    depth = req.render_depth if req else include_depth
    return agent_interface.capture_perception(
        width=w,
        height=h,
        include_segmentation=seg,
        include_depth=depth
    )

@app.post("/api/agent/trajectory/execute", response_model=TrajectoryExecutionState)
def agent_execute_trajectory(req: TrajectoryExecuteRequest):
    state = agent_interface.execute_motion_trajectory(
        target_pos=req.target_pos,
        grid_resolution=req.grid_resolution,
        robot_radius=req.robot_radius
    )
    return state

@app.get("/api/agent/trajectory/status", response_model=TrajectoryExecutionState)
def agent_trajectory_status():
    return agent_interface.trajectory_state

@app.post("/api/agent/trajectory/cancel")
def agent_cancel_trajectory():
    success = agent_interface.cancel_motion_trajectory()
    try:
        agent_interface.cancel_drone_trajectory()
    except Exception:
        pass
    return {"status": "cancelled", "success": success}


class DrivePathRequest(BaseModel):
    shape: str = ""
    size: float = Field(default=1.6, ge=0.4, le=4.0)
    waypoints: Optional[List[List[float]]] = None
    center: Optional[List[float]] = None
    laps: int = Field(default=3, ge=1, le=20)
    loop: Optional[bool] = None
    altitude: float = Field(default=1.2, ge=0.3, le=5.0)


@app.post("/api/agent/drone-path", response_model=TrajectoryExecutionState)
async def agent_drone_path(req: DrivePathRequest):
    """Fly the drone along a shape/path or explicit 3D waypoints. Geometry flight-following, no RL."""
    from backend.path_shapes import generate_path
    center = req.center
    if center is None:
        fc = engine._find_drone_controller("drone")
        center = [fc.current_position[0], fc.current_position[1]] if fc else [0.0, 0.0]
    try:
        bound = float(project_manager.get_current().settings.world_boundaries[0])
    except Exception:
        bound = 6.0
    plan = generate_path(description=req.shape, center=center, size=req.size,
                         points=req.waypoints, arena_bound=abs(bound) if bound else 6.0, closed=req.loop)
    wps3d = [[p[0], p[1], req.altitude] for p in plan["waypoints"]]
    state = agent_interface.execute_drone_waypoint_path(
        waypoints=wps3d, loop=plan["loop"], max_laps=req.laps, shape=plan["shape"])
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return state


@app.post("/api/agent/drive-path", response_model=TrajectoryExecutionState)
async def agent_drive_path(req: DrivePathRequest):
    """Drive the rover along ANY shape/path (square, triangle, N-gon, circle, figure-8, star,
    zig-zag, spiral, wave, line) or an explicit waypoint list — geometry path-following, no RL."""
    from backend.path_shapes import generate_path
    center = req.center
    if center is None:
        try:
            st = engine.get_state()
            center = [st.robot.position[0], st.robot.position[1]]
        except Exception:
            center = [0.0, 0.0]
    try:
        bound = float(project_manager.get_current().settings.world_boundaries[0])
    except Exception:
        bound = 6.0
    plan = generate_path(description=req.shape, center=center, size=req.size,
                         points=req.waypoints, arena_bound=abs(bound) if bound else 6.0, closed=req.loop)
    state = agent_interface.execute_waypoint_path(
        waypoints=plan["waypoints"], loop=plan["loop"], max_laps=req.laps, shape=plan["shape"]
    )
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return state


# ==========================================
# Mission Planner: user-drawn waypoint paths
# ==========================================
from pathlib import Path as _Path
MISSIONS_DIR = _Path("missions")
MISSIONS_DIR.mkdir(exist_ok=True)


class MissionRequest(BaseModel):
    name: str = "mission"
    waypoints: List[List[float]] = Field(default_factory=list)
    speed: float = Field(default=1.0, ge=0.1, le=3.0)
    loop: bool = True
    laps: int = Field(default=3, ge=1, le=50)
    domain_randomize: bool = False
    altitude: float = Field(default=1.2, ge=0.3, le=5.0)
    vehicle: str = "auto"  # "auto" | "rover" | "drone"


def _mission_is_drone(req: MissionRequest) -> bool:
    """Decide whether the mission should fly a drone: explicit vehicle, else auto-detect a drone
    in the scene when there's no active ground rover."""
    if req.vehicle == "drone":
        return True
    if req.vehicle == "rover":
        return False
    has_drone = len(getattr(engine, "drone_controllers", {})) > 0
    has_rover = getattr(engine.project, "robot", None) is not None and getattr(engine.project, "robot_active", True)
    return has_drone and not has_rover


def _apply_mission_speed(speed: float):
    rob = getattr(engine.project, "robot", None)
    if rob is not None:
        rob.max_linear_velocity = float(speed)


@app.post("/api/mission/run")
async def mission_run(req: MissionRequest):
    """Drive/fly the user's clicked waypoint path right away (no training)."""
    if len(req.waypoints) < 2:
        raise HTTPException(status_code=400, detail="Add at least 2 waypoints.")
    if _mission_is_drone(req):
        # Give each 2D waypoint the chosen flight altitude and fly the drone through them.
        wps3d = [[float(p[0]), float(p[1]), req.altitude] for p in req.waypoints]
        state = agent_interface.execute_drone_waypoint_path(
            waypoints=wps3d, loop=req.loop and len(wps3d) >= 3, max_laps=req.laps, shape="flight path")
        await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
        return state
    try:
        engine.disengage_rl_policy()
    except Exception:
        pass
    _apply_mission_speed(req.speed)
    state = agent_interface.execute_waypoint_path(
        waypoints=req.waypoints, loop=req.loop and len(req.waypoints) >= 3,
        max_laps=req.laps, shape="custom path")
    return state


@app.post("/api/mission/train")
async def mission_train(req: MissionRequest):
    """Live PPO training, then the rover follows the user's exact waypoint path."""
    if len(req.waypoints) < 2:
        raise HTTPException(status_code=400, detail="Add at least 2 waypoints.")
    try:
        agent_interface.cancel_motion_trajectory()
        engine.disengage_rl_policy()
    except Exception:
        pass
    _apply_mission_speed(req.speed)
    res = training_manager.start_live_interactive_rover_training(
        engine=engine, robot_identifier="robot", total_episodes=4,
        task_description="follow the custom path", waypoints=req.waypoints, loop=req.loop,
        domain_randomize=req.domain_randomize)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return res


@app.post("/api/mission/save")
async def mission_save(req: MissionRequest):
    """Save the mission (waypoints, speed, loop) together with the current scene obstacles."""
    safe = "".join(ch for ch in req.name if ch.isalnum() or ch in "-_ ").strip() or "mission"
    data = req.model_dump()
    data["name"] = safe
    data["objects"] = [o.model_dump() for o in engine.project.objects]
    (MISSIONS_DIR / f"{safe}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        project_manager.update_project(engine.project)
        project_manager.save_project_to_disk()
    except Exception:
        pass
    return {"status": "saved", "name": safe}


@app.get("/api/mission/list")
def mission_list():
    return {"missions": sorted(p.stem for p in MISSIONS_DIR.glob("*.json"))}


@app.get("/api/mission/load/{name}")
async def mission_load(name: str):
    """Load a saved mission: restores its obstacles into the scene and returns the waypoints."""
    f = MISSIONS_DIR / f"{name}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Mission not found")
    data = json.loads(f.read_text(encoding="utf-8"))
    try:
        engine.project.objects = [SceneObject(**o) for o in data.get("objects", [])]
        engine.rebuild_model(preserve_state=False)
        project_manager.update_project(engine.project)
    except Exception as e:
        logger.warning("Mission obstacle restore failed: %s", e)
    await broadcast_json({"type": "project_updated", "data": engine.project.model_dump()})
    return data

@app.post("/api/agent/experiments/validate")
def agent_validate_experiment(config: ExperimentConfig):
    return agent_interface.validate_experiment_config(config)

@app.post("/api/agent/experiments/run")
def agent_run_experiment(req: AgentExperimentRunRequest):
    try:
        result = agent_interface.start_experiment(scenario_id=req.scenario_id, seed=req.seed)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/experiments/compare")
def agent_compare_experiments(req: CompareRunsRequest):
    try:
        return agent_interface.compare_runs(req.run_a_id, req.run_b_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

@app.get("/api/agent/records/{run_id}")
def agent_export_record(run_id: str):
    try:
        return agent_interface.export_reproducible_record(run_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


# WebSocket Endpoint for real-time simulation state, teleop commands, and logs
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    engine.log("info", "Client connected to simulation WebSocket stream.")
    
    # Send initial project & state & recent logs
    await websocket.send_json({
        "type": "init",
        "project": project_manager.get_current().model_dump(),
        "state": engine.get_state().model_dump(),
        "logs": [l.model_dump() for l in recent_logs[-50:]]
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "teleop":
                linear = float(data.get("linear", 0.0))
                angular = float(data.get("angular", 0.0))
                engine.set_teleop(linear, angular)
            elif msg_type == "play":
                engine.play()
            elif msg_type == "pause":
                engine.pause()
            elif msg_type == "step":
                if engine.status == "running":
                    engine.pause()
                engine.step()
            elif msg_type == "reset":
                engine.reset()
            elif msg_type == "auto_drive":
                auto_drive_active = bool(data.get("enabled", False))
                engine.log("info", f"Auto-Drive {'enabled' if auto_drive_active else 'disabled'} via WebSocket.")
            elif msg_type == "clear_logs":
                recent_logs.clear()
                await broadcast_json({"type": "logs_cleared"})
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        active_connections.discard(websocket)
        logger.warning("WebSocket error: %s", e)


# ---------------------------------------------------------
# User Authentication & Database Management (SQLite)
# ---------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class GoogleAuthRequest(BaseModel):
    credential: Optional[str] = Field(None, description="Google ID Token / Credential string")
    sub: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    client_id: Optional[str] = None


def _decode_jwt_unverified(token: str) -> Dict[str, Any]:
    """Safely decodes JWT payload without requiring external signature keys."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        import base64
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        return json.loads(decoded)
    except Exception as e:
        logger.warning("Failed to decode JWT unverified: %s", e)
        return {}


@app.post("/api/auth/register")
async def register_endpoint(req: RegisterRequest):
    try:
        result = db.register_user(req.name, req.email, req.password)
        user = result["user"]
        token = result["token"]
        engine.log("info", f"User registered in database: '{user['name']}' ({user['email']})")
        await broadcast_json({"type": "user_authenticated", "data": user})
        return {"status": "ok", "user": user, "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Registration error: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@app.post("/api/auth/login")
async def login_endpoint(req: LoginRequest):
    try:
        result = db.authenticate_user(req.email, req.password)
        user = result["user"]
        token = result["token"]
        engine.log("info", f"User logged in from database: '{user['name']}' ({user['email']})")
        await broadcast_json({"type": "user_authenticated", "data": user})
        return {"status": "ok", "user": user, "token": token}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.exception("Login error: %s", e)
        raise HTTPException(status_code=500, detail="Authentication failed. Please try again.")


@app.post("/api/auth/google")
async def google_auth_endpoint(req: GoogleAuthRequest):
    sub = req.sub
    email = req.email or ""
    name = req.name or ""
    picture = req.picture

    if req.credential:
        payload = _decode_jwt_unverified(req.credential)
        if payload and payload.get("sub"):
            sub = payload.get("sub")
            email = payload.get("email", email)
            name = payload.get("name") or (email.split("@")[0] if "@" in email else "") or name
            picture = payload.get("picture", picture)

    if not sub:
        raise HTTPException(status_code=400, detail="Missing Google user identification.")

    if not name:
        name = email.split("@")[0] if "@" in email else "Google User"

    try:
        result = db.upsert_google_user(sub, email, name, picture)
        user = result["user"]
        token = result["token"]
        engine.log("info", f"User '{user['name']}' ({user['email']}) authenticated via Google OAuth into SQLite DB.")
        await broadcast_json({"type": "user_authenticated", "data": user})
        return {"status": "ok", "user": user, "token": token}
    except Exception as e:
        logger.exception("Google auth database error: %s", e)
        raise HTTPException(status_code=500, detail="Google authentication failed.")


@app.get("/api/auth/me")
def get_current_user_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    token = _extract_token_from_request(request, authorization)
    if not token:
        return {"authenticated": False, "user": None}
    user = db.get_user_by_session(token)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user}


@app.post("/api/auth/logout")
async def logout_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    token = _extract_token_from_request(request, authorization)
    if token:
        db.delete_session(token)
    await broadcast_json({"type": "user_logged_out"})
    return {"status": "ok", "message": "Logged out successfully."}

