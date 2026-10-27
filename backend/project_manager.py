from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from backend.models import Project, SceneObject, RobotConfig, SimulationSettings, SCHEMA_VERSION

PROJECTS_DIR = Path("projects")

def get_default_project() -> Project:
    """Returns a sensible default robotics simulation project with primitive obstacles."""
    return Project(
        schema_version=SCHEMA_VERSION,
        name="Workspace Alpha",
        description="Default RoboForge differential-drive arena",
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
                size=[0.25, 0.25, 0.20], # half-sizes
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
                size=[0.18, 0.35], # radius, half_height
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
                size=[0.25], # radius
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

    def new_project(self, name: str = "New Scene") -> Project:
        proj = get_default_project()
        proj.name = name
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
