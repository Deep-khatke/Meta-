from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from backend.models import (
    Project, SceneObject, RobotConfig, SimulationSettings, SCHEMA_VERSION,
    GenericRobotModel, LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial
)

PROJECTS_DIR = Path("projects")

def get_clean_empty_project(name: str = "Clean Workspace") -> Project:
    """Returns a completely clean, empty project with no robot and no obstacles."""
    return Project(
        schema_version=SCHEMA_VERSION,
        name=name,
        description="Fresh empty 3D robotics simulation workspace",
        robot=None,
        generic_robot=None,
        objects=[],
        settings=SimulationSettings(
            timestep=0.005,
            speed_multiplier=1.0,
            gravity=[0.0, 0.0, -9.81],
            enable_boundaries=True
        )
    )

def get_kr6_arm_robot() -> GenericRobotModel:
    """Returns high-precision KUKA KR-6 Pro Articulated 6-Axis Industrial Manipulator."""
    return GenericRobotModel(
        name="KR-6 Pro Articulated",
        root_link="base_link",
        base_position=[0.0, 0.0, 0.0],
        base_rpy=[0.0, 0.0, 0.0],
        is_fixed_base=True,
        links=[
            LinkModel(
                name="base_link",
                visual=LinkGeom(type="cylinder", size=[0.20, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                collision=LinkGeom(type="cylinder", size=[0.20, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                inertial=LinkInertial(mass=8.0, origin_pos=[0, 0, 0.08], ixx=0.08, iyy=0.08, izz=0.08)
            ),
            LinkModel(
                name="link_1_turret",
                visual=LinkGeom(type="cylinder", size=[0.13, 0.12], origin_pos=[0, 0, 0.12], color="#ea580c"),
                collision=LinkGeom(type="cylinder", size=[0.13, 0.12], origin_pos=[0, 0, 0.12], color="#ea580c"),
                inertial=LinkInertial(mass=4.5, origin_pos=[0, 0, 0.12], ixx=0.05, iyy=0.05, izz=0.03)
            ),
            LinkModel(
                name="link_2_shoulder",
                visual=LinkGeom(type="box", size=[0.09, 0.08, 0.28], origin_pos=[0, 0, 0.28], color="#e2e8f0"),
                collision=LinkGeom(type="box", size=[0.09, 0.08, 0.28], origin_pos=[0, 0, 0.28], color="#e2e8f0"),
                inertial=LinkInertial(mass=3.8, origin_pos=[0, 0, 0.28], ixx=0.04, iyy=0.04, izz=0.02)
            ),
            LinkModel(
                name="link_3_elbow",
                visual=LinkGeom(type="box", size=[0.07, 0.07, 0.24], origin_pos=[0, 0, 0.24], color="#ea580c"),
                collision=LinkGeom(type="box", size=[0.07, 0.07, 0.24], origin_pos=[0, 0, 0.24], color="#ea580c"),
                inertial=LinkInertial(mass=2.5, origin_pos=[0, 0, 0.24], ixx=0.03, iyy=0.03, izz=0.015)
            ),
            LinkModel(
                name="link_4_forearm",
                visual=LinkGeom(type="cylinder", size=[0.06, 0.14], origin_pos=[0, 0, 0.14], color="#334155"),
                collision=LinkGeom(type="cylinder", size=[0.06, 0.14], origin_pos=[0, 0, 0.14], color="#334155"),
                inertial=LinkInertial(mass=1.6, origin_pos=[0, 0, 0.14], ixx=0.02, iyy=0.02, izz=0.01)
            ),
            LinkModel(
                name="link_5_wrist",
                visual=LinkGeom(type="cylinder", size=[0.05, 0.06], origin_pos=[0, 0, 0.06], color="#94a3b8"),
                collision=LinkGeom(type="cylinder", size=[0.05, 0.06], origin_pos=[0, 0, 0.06], color="#94a3b8"),
                inertial=LinkInertial(mass=0.9, origin_pos=[0, 0, 0.06], ixx=0.01, iyy=0.01, izz=0.005)
            ),
            LinkModel(
                name="tool_flange",
                visual=LinkGeom(type="box", size=[0.05, 0.04, 0.04], origin_pos=[0, 0, 0.04], color="#06b6d4"),
                collision=LinkGeom(type="box", size=[0.05, 0.04, 0.04], origin_pos=[0, 0, 0.04], color="#06b6d4"),
                inertial=LinkInertial(mass=0.5, origin_pos=[0, 0, 0.04], ixx=0.005, iyy=0.005, izz=0.005)
            )
        ],
        joints=[
            JointModel(
                name="joint_1_yaw",
                type="revolute",
                parent_link="base_link",
                child_link="link_1_turret",
                origin_pos=[0.0, 0.0, 0.16],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-3.14159, upper=3.14159, effort=120.0, velocity=5.0),
                damping=0.3,
                kp=120.0,
                kv=12.0
            ),
            JointModel(
                name="joint_2_shoulder",
                type="revolute",
                parent_link="link_1_turret",
                child_link="link_2_shoulder",
                origin_pos=[0.0, 0.0, 0.24],
                axis=[0.0, 1.0, 0.0],
                limit=JointLimit(lower=-1.9, upper=1.5, effort=140.0, velocity=4.5),
                damping=0.3,
                kp=140.0,
                kv=14.0
            ),
            JointModel(
                name="joint_3_elbow",
                type="revolute",
                parent_link="link_2_shoulder",
                child_link="link_3_elbow",
                origin_pos=[0.0, 0.0, 0.56],
                axis=[0.0, 1.0, 0.0],
                limit=JointLimit(lower=-2.6, upper=2.6, effort=90.0, velocity=5.5),
                damping=0.25,
                kp=100.0,
                kv=10.0
            ),
            JointModel(
                name="joint_4_wrist_roll",
                type="revolute",
                parent_link="link_3_elbow",
                child_link="link_4_forearm",
                origin_pos=[0.0, 0.0, 0.48],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-3.14159, upper=3.14159, effort=50.0, velocity=6.0),
                damping=0.2,
                kp=70.0,
                kv=7.0
            ),
            JointModel(
                name="joint_5_wrist_pitch",
                type="revolute",
                parent_link="link_4_forearm",
                child_link="link_5_wrist",
                origin_pos=[0.0, 0.0, 0.28],
                axis=[0.0, 1.0, 0.0],
                limit=JointLimit(lower=-2.1, upper=2.1, effort=35.0, velocity=6.5),
                damping=0.15,
                kp=50.0,
                kv=5.0
            ),
            JointModel(
                name="joint_6_wrist_yaw",
                type="revolute",
                parent_link="link_5_wrist",
                child_link="tool_flange",
                origin_pos=[0.0, 0.0, 0.12],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-6.28, upper=6.28, effort=20.0, velocity=8.0),
                damping=0.1,
                kp=40.0,
                kv=4.0
            )
        ]
    )

def get_scara_arm_robot() -> GenericRobotModel:
    """Returns 3-Axis SCARA Manipulator for high-speed pick and place."""
    return GenericRobotModel(
        name="3-Axis SCARA Manipulator",
        root_link="base_link",
        base_position=[0.0, 0.0, 0.0],
        base_rpy=[0.0, 0.0, 0.0],
        is_fixed_base=True,
        links=[
            LinkModel(name="base_link", visual=LinkGeom(type="cylinder", size=[0.22, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b")),
            LinkModel(name="column_link", visual=LinkGeom(type="cylinder", size=[0.08, 0.20], origin_pos=[0, 0, 0.20], color="#475569")),
            LinkModel(name="arm1_link", visual=LinkGeom(type="box", size=[0.18, 0.04, 0.03], origin_pos=[0.18, 0, 0.0], color="#0ea5e9")),
            LinkModel(name="arm2_link", visual=LinkGeom(type="box", size=[0.14, 0.035, 0.025], origin_pos=[0.14, 0, 0.0], color="#10b981"))
        ],
        joints=[
            JointModel(name="base_yaw_joint", type="revolute", parent_link="base_link", child_link="column_link", origin_pos=[0.0, 0.0, 0.16], axis=[0.0, 0.0, 1.0], limit=JointLimit(lower=-3.14, upper=3.14, effort=80.0, velocity=4.0)),
            JointModel(name="shoulder_joint", type="revolute", parent_link="column_link", child_link="arm1_link", origin_pos=[0.0, 0.0, 0.38], axis=[0.0, 0.0, 1.0], limit=JointLimit(lower=-2.8, upper=2.8, effort=60.0, velocity=5.0)),
            JointModel(name="elbow_joint", type="revolute", parent_link="arm1_link", child_link="arm2_link", origin_pos=[0.36, 0.0, 0.0], axis=[0.0, 0.0, 1.0], limit=JointLimit(lower=-2.8, upper=2.8, effort=40.0, velocity=6.0))
        ]
    )

def get_default_project() -> Project:
    """Returns a clean default robotics simulation project for a new user with no clutter."""
    return Project(
        schema_version=SCHEMA_VERSION,
        name="Workspace Alpha",
        description="Clean RoboForge differential-drive arena",
        robot=RobotConfig(
            id="robot_main",
            name="ForgeBot-DD1",
            initial_position=[0.0, 0.0, 0.07],
            initial_yaw=0.0,
            wheel_radius=0.06,
            wheel_base=0.28,
            max_linear_velocity=1.5,
            max_angular_velocity=3.0,
        ),
        objects=[],
        settings=SimulationSettings(
            timestep=0.005,
            speed_multiplier=1.0,
            gravity=[0.0, 0.0, -9.81]
        )
    )

def get_default_project_demo() -> Project:
    """Archived demo project with primitive obstacles preserved for reference."""
    return Project(
        schema_version=SCHEMA_VERSION,
        name="Workspace Alpha Demo",
        description="Archived RoboForge differential-drive arena with demo obstacles",
        robot=RobotConfig(
            id="robot_main",
            name="ForgeBot-DD1",
            initial_position=[0.0, 0.0, 0.07],
            initial_yaw=0.0,
            wheel_radius=0.06,
            wheel_base=0.28,
            max_linear_velocity=1.5,
            max_angular_velocity=3.0,
        ),
        objects=[
            SceneObject(
                id="obs_box_1",
                name="Cargo Box A",
                type="box",
                position=[1.5, 0.0, 0.20],
                rotation=[0.0, 0.0, 15.0],
                size=[0.25, 0.25, 0.20],
                color="#e06c75",
                is_static=False,
                mass=2.0
            ),
            SceneObject(
                id="obs_cyl_1",
                name="Pylon Cylinder",
                type="cylinder",
                position=[2.0, 1.2, 0.35],
                rotation=[0.0, 0.0, 0.0],
                size=[0.18, 0.35],
                color="#98c379",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="obs_sph_1",
                name="Target Sphere",
                type="sphere",
                position=[1.2, -1.2, 0.25],
                rotation=[0.0, 0.0, 0.0],
                size=[0.25],
                color="#d19a66",
                is_static=False,
                mass=1.2
            ),
        ],
        settings=SimulationSettings(
            timestep=0.005,
            speed_multiplier=1.0,
            gravity=[0.0, 0.0, -9.81]
        )
    )

class ProjectManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or PROJECTS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_project = get_default_project()
        
        # Recover last session if available
        last_session_path = self.storage_dir / "last_session.json"
        if last_session_path.exists():
            try:
                with open(last_session_path, "r", encoding="utf-8") as f:
                    content = f.read()
                raw_dict = json.loads(content)
                self.current_project = Project.model_validate(raw_dict)
            except Exception:
                pass

        self.saved_project_snapshot = self.current_project.model_copy(deep=True)

    def new_project(self, name: str = "New Scene", template: str = "empty", robot_type: Optional[str] = None) -> Project:
        t = (template or "empty").lower().strip()
        if t in ("empty", "clean", "blank"):
            proj = get_clean_empty_project(name=name)
        elif t in ("kr6", "kr6_arm", "arm", "industrial_arm"):
            proj = get_clean_empty_project(name=name)
            proj.generic_robot = get_kr6_arm_robot()
        elif t in ("scara", "scara_arm"):
            proj = get_clean_empty_project(name=name)
            proj.generic_robot = get_scara_arm_robot()
        elif t in ("rover", "diff_drive", "ground"):
            proj = get_default_project()
            proj.name = name
            proj.objects = []
        else:
            proj = get_clean_empty_project(name=name)

        self.current_project = proj
        self.saved_project_snapshot = proj.model_copy(deep=True)
        self._auto_save_session()
        return self.current_project

    def get_current(self) -> Project:
        return self.current_project

    def update_project(self, project: Project) -> Project:
        self.current_project = project
        self._auto_save_session()
        return self.current_project

    def _auto_save_session(self):
        try:
            last_session_path = self.storage_dir / "last_session.json"
            with open(last_session_path, "w", encoding="utf-8") as f:
                json.dump(self.current_project.model_dump(), f, indent=2)
        except Exception:
            pass

    def save_project_to_disk(self, filename: Optional[str] = None) -> str:
        """Saves current project to disk as versioned JSON."""
        clean_name = "".join(c for c in self.current_project.name if c.isalnum() or c in ("-", "_")).rstrip()
        if not clean_name:
            clean_name = "project"
        if not filename:
            filename = f"{clean_name}.rfproj.json"
        elif not filename.endswith(".rfproj.json") and not filename.endswith(".json"):
            filename = f"{filename}.rfproj.json"

        file_path = self.storage_dir / filename
        data = self.current_project.model_dump()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.saved_project_snapshot = self.current_project.model_copy(deep=True)
        return str(file_path)

    def load_project_from_json(self, json_content: str) -> Project:
        """Parses and validates project JSON."""
        try:
            raw_dict = json.loads(json_content)
        except Exception as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")

        # Check schema version
        version = raw_dict.get("schema_version")
        if not version:
            raise ValueError("Missing 'schema_version' in project file.")
        
        # In this first version, we support 1.0.0
        if not version.startswith("1."):
            raise ValueError(f"Unsupported schema version: {version}. Expected 1.x.x")

        try:
            project = Project.model_validate(raw_dict)
            self.current_project = project
            self.saved_project_snapshot = project.model_copy(deep=True)
            return project
        except ValidationError as e:
            raise ValueError(f"Project validation failed: {str(e)}")

    def load_project_from_file(self, filename: str) -> Project:
        file_path = self.storage_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Project file {filename} not found.")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.load_project_from_json(content)

    def reset_to_saved(self) -> Project:
        """Restores project configuration to the last saved snapshot."""
        self.current_project = self.saved_project_snapshot.model_copy(deep=True)
        return self.current_project

    def list_saved_projects(self) -> list[str]:
        return [f.name for f in self.storage_dir.glob("*.json")]
