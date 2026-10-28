from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Any, Literal
import numpy as np
from pydantic import BaseModel, Field

from backend.models import (
    Project, SceneObject, SimulationSettings, SimulationSnapshot,
    SimulationState, TeleopCommand
)
from backend.simulation import MuJoCoEngine
from backend.project_manager import ProjectManager
from backend.scene_validator import validate_scene, SceneValidationReport
from backend.motion_planning import (
    plan_collision_free_path, PathPlanningRequest,
    ContinuousTrajectoryController, TrajectoryExecutionState
)
from backend.synthetic_data import SyntheticDataGenerator, BoundingBox2D
from backend.experiment_models import (
    ExperimentConfig, TaskObjective, ExperimentRunResult, RunComparison
)
from backend.experiment_manager import ExperimentManager


class EnvironmentInspectionResult(BaseModel):
    project_name: str
    schema_version: str
    physics_settings: SimulationSettings
    robot_type: str
    robot_config: Dict[str, Any]
    sensor_suite: Dict[str, Any]
    scene_objects: List[SceneObject]
    total_objects: int
    actuator_count: int
    boundary_limits: List[float]
    is_scene_valid: bool
    validation_issues: List[str] = Field(default_factory=list)


class TelemetryObservation(BaseModel):
    sim_time: float
    status: str
    real_time_factor: float
    robot_position: List[float]
    robot_quaternion: List[float]
    linear_speed: float
    angular_speed: float
    joint_states: Dict[str, Dict[str, float]] # joint_name -> {position, velocity, torque}
    active_contacts: List[Dict[str, Any]]
    nearest_obstacle_distance: float
    imu_linear_acceleration: Optional[List[float]] = None
    imu_angular_velocity: Optional[List[float]] = None
    odometry_distance_traveled: float = 0.0


class PerceptionObservation(BaseModel):
    width: int
    height: int
    rgb_image_url: str
    depth_summary: Optional[Dict[str, float]] = None
    segmentation_image_url: Optional[str] = None
    segmentation_labels: Optional[Dict[str, Any]] = None
    bounding_boxes: List[BoundingBox2D] = Field(default_factory=list)


class ReproducibleRecord(BaseModel):
    record_id: str
    timestamp: float
    project_snapshot: Project
    experiment_config: Optional[ExperimentConfig] = None
    run_result: Optional[ExperimentRunResult] = None
    system_metadata: Dict[str, Any]


class AgentSimulationInterface:
    """Safe, typed, high-level interface designed for autonomous scientific simulation agents."""
    def __init__(self, project_manager: ProjectManager, engine: MuJoCoEngine, experiment_manager: ExperimentManager):
        self.pm = project_manager
        self.engine = engine
        self.em = experiment_manager
        self.active_trajectory_controller: Optional[ContinuousTrajectoryController] = None
        self.trajectory_state = TrajectoryExecutionState(trajectory_id="idle")

    def inspect_environment(self) -> EnvironmentInspectionResult:
        """Inspects authoritative scene, robot, sensors, actuators, and physics configuration."""
        proj = self.pm.get_current()
        val_report = validate_scene(proj)

        r_type = "generic_articulated" if proj.generic_robot else "differential_drive"
        r_conf = proj.generic_robot.model_dump() if proj.generic_robot else proj.robot.model_dump()

        issues = [f"[{e.severity.upper()}] {e.message}" for e in val_report.errors + val_report.warnings]

        act_count = len(self.engine.joint_actuator_ids) if proj.generic_robot else 2

        return EnvironmentInspectionResult(
            project_name=proj.name,
            schema_version=proj.schema_version,
            physics_settings=proj.settings,
            robot_type=r_type,
            robot_config=r_conf,
            sensor_suite=proj.sensors.model_dump(),
            scene_objects=proj.objects,
            total_objects=len(proj.objects),
            actuator_count=act_count,
            boundary_limits=proj.settings.world_boundaries,
            is_scene_valid=val_report.is_valid,
            validation_issues=issues
        )

    def get_telemetry(self) -> TelemetryObservation:
        """Reads ground-truth state, joint states, contacts, sensor observations, and odometry."""
        st = self.engine.get_state()

        joint_dict = {}
        for jname in st.joint_positions:
            joint_dict[jname] = {
                "position": st.joint_positions.get(jname, 0.0),
                "velocity": st.joint_velocities.get(jname, 0.0),
                "torque": st.joint_torques.get(jname, 0.0)
            }

        contact_list = []
        for c in st.contacts:
            contact_list.append({
                "geom1": c.geom1,
                "geom2": c.geom2,
                "distance": c.distance,
                "force_normal": c.force_normal,
                "force_tangent": c.force_tangent,
                "penetration": c.penetration
            })

        min_lidar = min(st.lidar_readings) if st.lidar_readings else 50.0

        return TelemetryObservation(
            sim_time=round(st.sim_time, 4),
            status=st.status,
            real_time_factor=round(st.real_time_factor, 2),
            robot_position=st.robot.position,
            robot_quaternion=st.robot.quaternion,
            linear_speed=round(st.robot.linear_speed, 3),
            angular_speed=round(st.robot.angular_speed, 3),
            joint_states=joint_dict,
            active_contacts=contact_list,
            nearest_obstacle_distance=round(min_lidar, 3),
            imu_linear_acceleration=st.imu_linear_acceleration,
            imu_angular_velocity=st.imu_angular_velocity,
            odometry_distance_traveled=round(st.odometry.distance_traveled, 3) if st.odometry else 0.0
        )

    def capture_perception(
        self,
        width: int = 320,
        height: int = 240,
        include_segmentation: bool = True,
        include_depth: bool = True
    ) -> PerceptionObservation:
        """Captures multi-modal camera perception including genuine pixel geom-ID segmentation."""
        frame = self.engine.capture_camera_frame(
            width=width,
            height=height,
            render_depth=include_depth,
            render_segmentation=include_segmentation
        )

        # Also project 2D bounding boxes for detected objects
        generator = SyntheticDataGenerator(self.pm.get_current())
        labeled_frame = generator.capture_labeled_frame(width=width, height=height)

        seg_url = None
        seg_labels = None
        if frame.get("segmentation_info"):
            seg_url = frame["segmentation_info"]["image_data_url"]
            seg_labels = frame["segmentation_info"]["labels"]

        return PerceptionObservation(
            width=width,
            height=height,
            rgb_image_url=frame["image_data_url"],
            depth_summary=frame.get("depth_info"),
            segmentation_image_url=seg_url,
            segmentation_labels=seg_labels,
            bounding_boxes=labeled_frame.annotations
        )

    def execute_motion_trajectory(
        self,
        target_pos: List[float],
        grid_resolution: float = 0.15,
        robot_radius: float = 0.30
    ) -> TrajectoryExecutionState:
        """Plans a collision-free path with A* and executes it continuously in the MuJoCo engine."""
        st = self.engine.get_state()
        start_2d = [st.robot.position[0], st.robot.position[1]]
        goal_2d = [target_pos[0], target_pos[1]]

        req = PathPlanningRequest(
            start_pos=start_2d,
            goal_pos=goal_2d,
            grid_resolution=grid_resolution,
            robot_radius=robot_radius,
            smooth_path=True
        )
        plan_res = plan_collision_free_path(self.pm.get_current(), req)

        if not plan_res.success:
            self.trajectory_state = TrajectoryExecutionState(
                trajectory_id=f"traj_{int(time.time())}",
                status="failed",
                start_pos=start_2d,
                goal_pos=goal_2d,
                status_message=plan_res.message
            )
            return self.trajectory_state

        v_max = self.pm.get_current().robot.max_linear_velocity
        w_max = self.pm.get_current().robot.max_angular_velocity

        self.active_trajectory_controller = ContinuousTrajectoryController(
            waypoints=plan_res.path,
            v_max=v_max,
            w_max=w_max
        )

        traj_id = f"traj_{int(time.time())}"
        self.trajectory_state = TrajectoryExecutionState(
            trajectory_id=traj_id,
            status="running",
            start_pos=start_2d,
            goal_pos=goal_2d,
            waypoints=plan_res.path,
            current_waypoint_index=0,
            total_waypoints=len(plan_res.path),
            distance_to_goal=plan_res.path_length,
            progress_pct=0.0,
            status_message="Tracking active waypoints continuously"
        )

        if self.engine.status != "running":
            self.engine.play()

        return self.trajectory_state

    def cancel_motion_trajectory(self) -> bool:
        """Halts the active continuous trajectory controller and stops robot motors."""
        self.active_trajectory_controller = None
        self.trajectory_state.status = "cancelled"
        self.trajectory_state.status_message = "Cancelled by agent request"
        self.engine.set_teleop(0.0, 0.0)
        return True

    def save_physics_snapshot(self, name: Optional[str] = None) -> SimulationSnapshot:
        """Captures a reproducible physics snapshot for experiment branching."""
        return self.engine.save_snapshot(snapshot_id=name)

    def restore_physics_snapshot(self, snapshot_id: str) -> bool:
        """Restores exact physics state from a previously saved snapshot."""
        return self.engine.restore_snapshot(snapshot_id)

    def validate_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validates an experiment configuration before execution."""
        errors = []
        warnings = []
        if config.duration <= 0.0 or config.duration > 300.0:
            errors.append("Duration must be between 0.1 and 300.0 seconds")
        if not config.task.target_position:
            errors.append("TaskObjective requires a non-empty target_position")

        # Validate obstacle layout
        temp_project = Project(
            name=config.name,
            robot=config.robot,
            objects=config.environment,
            settings=config.settings
        )
        scene_rep = validate_scene(temp_project)
        for e in scene_rep.errors:
            errors.append(f"Obstacle error: {e.message}")
        for w in scene_rep.warnings:
            warnings.append(f"Obstacle warning: {w.message}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config_id": config.id
        }

    def start_experiment(self, scenario_id: str, seed: Optional[int] = None) -> ExperimentRunResult:
        """Executes an authoritative experiment run synchronously with guaranteed completion."""
        scen = self.em.get_scenario(scenario_id)
        if not scen:
            raise ValueError(f"Scenario '{scenario_id}' not found.")
        
        run_scen = scen.model_copy(deep=True)
        if seed is not None:
            run_scen.seed = seed

        return self.em.run_experiment(run_scen)

    def compare_runs(self, run_a_id: str, run_b_id: str) -> RunComparison:
        """Compares two completed experiment runs and produces metric deltas."""
        return self.em.compare_runs(run_a_id, run_b_id)

    def export_reproducible_record(self, run_id: str) -> ReproducibleRecord:
        """Exports a self-contained, reproducible record with project snapshot and run results."""
        run_res = self.em.get_run_result(run_id)
        if not run_res:
            raise ValueError(f"Run result '{run_id}' not found.")

        scen = self.em.get_scenario(run_res.experiment_id)
        proj = self.pm.get_current()

        return ReproducibleRecord(
            record_id=f"rep_{run_id}",
            timestamp=time.time(),
            project_snapshot=proj,
            experiment_config=scen,
            run_result=run_res,
            system_metadata={
                "physics_engine": "MuJoCo 3.x",
                "timestep": proj.settings.timestep,
                "gravity": proj.settings.gravity
            }
        )
