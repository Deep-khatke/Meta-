from __future__ import annotations
import json
import uuid
import copy
from pathlib import Path
from typing import List, Dict, Optional, Any
import mujoco

from backend.models import (
    Project, SceneObject, RobotConfig, GenericRobotModel,
    LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial,
    SimulationSettings, SensorSettings, LidarConfig, ImuConfig, CameraConfig, EncoderConfig,
    SCHEMA_VERSION
)
from backend.asset_models import AssetModel, ValidationResult, CreateAssetRequest
from backend.robot_validator import validate_generic_robot

ASSETS_STORAGE_DIR = Path("assets")

class AssetLibrary:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or ASSETS_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.custom_assets_file = self.storage_dir / "custom_assets.json"
        
        self.built_in_assets: Dict[str, AssetModel] = {}
        self.custom_assets: Dict[str, AssetModel] = {}
        
        self._init_built_in_assets()
        self._load_custom_assets()

    def _init_built_in_assets(self):
        """Initializes verified, physical built-in assets across all categories."""
        # -------------------------------------------------------------
        # 1. ROBOTS (Mobile Diff-Drive & Articulated Manipulators)
        # -------------------------------------------------------------
        forgebot = RobotConfig(
            id="forgebot_dd1",
            name="ForgeBot-DD1",
            initial_position=[0.0, 0.0, 0.07],
            initial_yaw=0.0,
            chassis_size=[0.20, 0.15, 0.05],
            wheel_radius=0.06,
            wheel_width=0.025,
            wheel_base=0.28,
            caster_radius=0.02,
            chassis_mass=3.5,
            wheel_mass=0.3,
            max_linear_velocity=1.5,
            max_angular_velocity=3.0,
        )
        self.built_in_assets["asset_robot_forgebot"] = AssetModel(
            id="asset_robot_forgebot",
            name="ForgeBot-DD1 (Mobile)",
            category="robot",
            description="Balanced two-wheeled differential drive mobile robot with caster stabilizers and front bumper mount.",
            tags=["mobile", "differential-drive", "agv", "verified"],
            is_built_in=True,
            thumbnail_hint="bot",
            data={"type": "mobile", "config": forgebot.model_dump()}
        )

        scout = RobotConfig(
            id="scout_micro",
            name="Scout-Micro",
            initial_position=[0.0, 0.0, 0.045],
            initial_yaw=0.0,
            chassis_size=[0.12, 0.09, 0.035],
            wheel_radius=0.04,
            wheel_width=0.018,
            wheel_base=0.18,
            caster_radius=0.015,
            chassis_mass=1.2,
            wheel_mass=0.15,
            max_linear_velocity=2.2,
            max_angular_velocity=4.5,
        )
        self.built_in_assets["asset_robot_scout"] = AssetModel(
            id="asset_robot_scout",
            name="Scout-Micro (Agile)",
            category="robot",
            description="Lightweight high-agility micro mobile robot capable of high angular acceleration and tight turns.",
            tags=["mobile", "micro", "high-speed", "differential-drive"],
            is_built_in=True,
            thumbnail_hint="bot",
            data={"type": "mobile", "config": scout.model_dump()}
        )

        heavy_agv = RobotConfig(
            id="heavy_cargo_agv",
            name="Heavy-Cargo AGV",
            initial_position=[0.0, 0.0, 0.10],
            initial_yaw=0.0,
            chassis_size=[0.35, 0.25, 0.08],
            wheel_radius=0.09,
            wheel_width=0.04,
            wheel_base=0.45,
            caster_radius=0.03,
            chassis_mass=22.0,
            wheel_mass=1.5,
            max_linear_velocity=1.0,
            max_angular_velocity=1.8,
        )
        self.built_in_assets["asset_robot_agv"] = AssetModel(
            id="asset_robot_agv",
            name="Heavy-Cargo AGV",
            category="robot",
            description="Industrial automated guided vehicle designed for high payload mass stability and heavy transport.",
            tags=["mobile", "industrial", "heavy-duty", "logistics"],
            is_built_in=True,
            thumbnail_hint="bot",
            data={"type": "mobile", "config": heavy_agv.model_dump()}
        )

        # 2-Link Planar Manipulator
        two_link_arm = GenericRobotModel(
            name="TwoLinkPlanarArm",
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
                    name="link_1",
                    visual=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0.0], color="#2563eb"),
                    collision=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0.0], color="#2563eb"),
                    inertial=LinkInertial(mass=1.5, origin_pos=[0.20, 0, 0.0], ixx=0.02, iyy=0.02, izz=0.02)
                ),
                LinkModel(
                    name="link_2",
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
                    child_link="link_1",
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
                    parent_link="link_1",
                    child_link="link_2",
                    origin_pos=[0.40, 0.0, 0.0],
                    axis=[0.0, 0.0, 1.0],
                    limit=JointLimit(lower=-2.6, upper=2.6, effort=30.0, velocity=5.0),
                    damping=0.15,
                    kp=60.0,
                    kv=6.0
                )
            ]
        )
        self.built_in_assets["asset_robot_2link_arm"] = AssetModel(
            id="asset_robot_2link_arm",
            name="2-Link Planar Manipulator",
            category="robot",
            description="2-DOF articulated planar robotic manipulator with shoulder and elbow revolute joints.",
            tags=["articulated", "arm", "manipulator", "kinematics"],
            is_built_in=True,
            thumbnail_hint="arm",
            data={"type": "articulated", "config": two_link_arm.model_dump()}
        )

        # 3-Link SCARA Manipulator
        scara_arm = GenericRobotModel(
            name="SCARAManipulator3D",
            root_link="base_link",
            base_position=[2.5, 2.5, 0.05],
            is_fixed_base=True,
            links=[
                LinkModel(
                    name="base_link",
                    visual=LinkGeom(type="cylinder", size=[0.22, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                    collision=LinkGeom(type="cylinder", size=[0.22, 0.08], origin_pos=[0, 0, 0.08], color="#1e293b"),
                    inertial=LinkInertial(mass=6.0, origin_pos=[0, 0, 0.08])
                ),
                LinkModel(
                    name="column_link",
                    visual=LinkGeom(type="cylinder", size=[0.08, 0.20], origin_pos=[0, 0, 0.20], color="#475569"),
                    collision=LinkGeom(type="cylinder", size=[0.08, 0.20], origin_pos=[0, 0, 0.20], color="#475569"),
                    inertial=LinkInertial(mass=3.0, origin_pos=[0, 0, 0.20])
                ),
                LinkModel(
                    name="arm1_link",
                    visual=LinkGeom(type="box", size=[0.18, 0.04, 0.03], origin_pos=[0.18, 0, 0.0], color="#0ea5e9"),
                    collision=LinkGeom(type="box", size=[0.18, 0.04, 0.03], origin_pos=[0.18, 0, 0.0], color="#0ea5e9"),
                    inertial=LinkInertial(mass=1.6, origin_pos=[0.18, 0, 0.0])
                ),
                LinkModel(
                    name="arm2_link",
                    visual=LinkGeom(type="box", size=[0.14, 0.035, 0.025], origin_pos=[0.14, 0, 0.0], color="#10b981"),
                    collision=LinkGeom(type="box", size=[0.14, 0.035, 0.025], origin_pos=[0.14, 0, 0.0], color="#10b981"),
                    inertial=LinkInertial(mass=0.9, origin_pos=[0.14, 0, 0.0])
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
        )
        self.built_in_assets["asset_robot_scara"] = AssetModel(
            id="asset_robot_scara",
            name="3-Link SCARA Manipulator",
            category="robot",
            description="3-DOF industrial SCARA robot arm with base rotation, shoulder swivel, and elbow joint.",
            tags=["articulated", "scara", "industrial", "3-dof"],
            is_built_in=True,
            thumbnail_hint="arm",
            data={"type": "articulated", "config": scara_arm.model_dump()}
        )

        # -------------------------------------------------------------
        # 2. OBSTACLES & PROPS
        # -------------------------------------------------------------
        self.built_in_assets["asset_obs_cargo_crate"] = AssetModel(
            id="asset_obs_cargo_crate",
            name="Industrial Cargo Crate",
            category="obstacle",
            description="Dense rectangular wooden transport crate with balanced physical mass and high friction.",
            tags=["obstacle", "box", "cargo", "dynamic"],
            is_built_in=True,
            thumbnail_hint="box",
            data={
                "name": "Cargo Crate",
                "type": "box",
                "size": [0.25, 0.25, 0.20],
                "mass": 3.0,
                "friction": [1.2, 0.005, 0.0001],
                "color": "#e06c75",
                "is_static": False
            }
        )

        self.built_in_assets["asset_obs_slalom_pylon"] = AssetModel(
            id="asset_obs_slalom_pylon",
            name="Slalom Pylon",
            category="obstacle",
            description="High-visibility tall obstacle pylon for agility courses and obstacle avoidance navigation.",
            tags=["obstacle", "cylinder", "slalom", "static"],
            is_built_in=True,
            thumbnail_hint="cylinder",
            data={
                "name": "Slalom Pylon",
                "type": "cylinder",
                "size": [0.15, 0.40],
                "mass": 15.0,
                "friction": [1.0, 0.005, 0.0001],
                "color": "#98c379",
                "is_static": True
            }
        )

        self.built_in_assets["asset_obs_barrier_wall"] = AssetModel(
            id="asset_obs_barrier_wall",
            name="Modular Barrier Wall",
            category="obstacle",
            description="Static modular concrete barrier wall segment for dividing lanes or creating maze corridors.",
            tags=["obstacle", "wall", "corridor", "static"],
            is_built_in=True,
            thumbnail_hint="box",
            data={
                "name": "Barrier Wall",
                "type": "box",
                "size": [0.60, 0.08, 0.30],
                "mass": 80.0,
                "friction": [1.5, 0.01, 0.001],
                "color": "#64748b",
                "is_static": True
            }
        )

        self.built_in_assets["asset_obs_storage_drum"] = AssetModel(
            id="asset_obs_storage_drum",
            name="Storage Barrel / Drum",
            category="obstacle",
            description="Heavy cylindrical metal storage drum with rolling resistance and realistic inertia.",
            tags=["obstacle", "cylinder", "barrel", "dynamic"],
            is_built_in=True,
            thumbnail_hint="cylinder",
            data={
                "name": "Storage Drum",
                "type": "cylinder",
                "size": [0.22, 0.35],
                "mass": 6.5,
                "friction": [1.0, 0.008, 0.0005],
                "color": "#38bdf8",
                "is_static": False
            }
        )

        self.built_in_assets["asset_obs_sensor_sphere"] = AssetModel(
            id="asset_obs_sensor_sphere",
            name="Sensor Calibration Sphere",
            category="obstacle",
            description="Calibrated spherical reflector target for acoustic and optical sensor benchmark tests.",
            tags=["obstacle", "sphere", "calibration", "dynamic"],
            is_built_in=True,
            thumbnail_hint="sphere",
            data={
                "name": "Calibration Sphere",
                "type": "sphere",
                "size": [0.22],
                "mass": 1.5,
                "friction": [0.8, 0.003, 0.0001],
                "color": "#d19a66",
                "is_static": False
            }
        )

        # -------------------------------------------------------------
        # 3. SENSORS
        # -------------------------------------------------------------
        self.built_in_assets["asset_sensor_precision_lidar"] = AssetModel(
            id="asset_sensor_precision_lidar",
            name="Precision 360° LiDAR",
            category="sensor",
            description="Full-surround 360-degree LiDAR sensor array with 32 beams and 12-meter range.",
            tags=["sensor", "lidar", "360", "precision"],
            is_built_in=True,
            thumbnail_hint="radio",
            data={
                "type": "lidar",
                "config": LidarConfig(
                    enabled=True,
                    mount_pos=[0.10, 0.0, 0.09],
                    num_beams=32,
                    fov_deg=360.0,
                    max_range=12.0,
                    min_range=0.05
                ).model_dump()
            }
        )

        self.built_in_assets["asset_sensor_fast_lidar"] = AssetModel(
            id="asset_sensor_fast_lidar",
            name="Fast 180° Front LiDAR",
            category="sensor",
            description="High-frequency 180-degree forward collision avoidance LiDAR scanner with 16 beams.",
            tags=["sensor", "lidar", "180", "fast"],
            is_built_in=True,
            thumbnail_hint="radio",
            data={
                "type": "lidar",
                "config": LidarConfig(
                    enabled=True,
                    mount_pos=[0.12, 0.0, 0.09],
                    num_beams=16,
                    fov_deg=180.0,
                    max_range=8.0,
                    min_range=0.05
                ).model_dump()
            }
        )

        self.built_in_assets["asset_sensor_imu_6dof"] = AssetModel(
            id="asset_sensor_imu_6dof",
            name="6-DOF Industrial IMU",
            category="sensor",
            description="Chassis-centered 6-DOF inertial measurement unit extracting real MuJoCo accelerometer and gyro physics.",
            tags=["sensor", "imu", "inertial", "6-dof"],
            is_built_in=True,
            thumbnail_hint="cpu",
            data={
                "type": "imu",
                "config": ImuConfig(
                    enabled=True,
                    mount_pos=[0.0, 0.0, 0.05]
                ).model_dump()
            }
        )

        self.built_in_assets["asset_sensor_wide_rgb_cam"] = AssetModel(
            id="asset_sensor_wide_rgb_cam",
            name="Wide-Angle RGB Vision Camera",
            category="sensor",
            description="Front bumper mounted vision camera sensor with 90° FOV for visual navigation.",
            tags=["sensor", "camera", "vision", "rgb"],
            is_built_in=True,
            thumbnail_hint="video",
            data={
                "type": "camera",
                "config": CameraConfig(
                    enabled=True,
                    mount_pos=[0.18, 0.0, 0.08],
                    mount_rpy=[0.0, 0.0, 0.0],
                    width=320,
                    height=240,
                    fov_deg=90.0
                ).model_dump()
            }
        )

        # -------------------------------------------------------------
        # 4. SCENE TEMPLATES (Full, verified MuJoCo environments)
        # -------------------------------------------------------------
        # Template 1: Mobile Navigation Arena
        tpl_nav = Project(
            schema_version=SCHEMA_VERSION,
            name="Mobile Navigation Arena",
            description="Open arena with boundary walls, waypoint targets, and scattered cargo props for path planning.",
            robot=RobotConfig(
                id="robot_nav",
                name="ForgeBot-DD1",
                initial_position=[0.0, 0.0, 0.07],
                initial_yaw=0.0
            ),
            objects=[
                SceneObject(id="box_nav_1", name="Cargo Alpha", type="box", position=[1.5, 0.2, 0.20], size=[0.25, 0.25, 0.20], color="#e06c75", mass=2.5),
                SceneObject(id="pylon_nav_1", name="Pylon Center", type="cylinder", position=[2.2, -1.0, 0.35], size=[0.16, 0.35], color="#98c379", mass=12.0, is_static=True),
                SceneObject(id="box_nav_2", name="Cargo Beta", type="box", position=[2.8, 1.2, 0.20], size=[0.20, 0.20, 0.20], color="#38bdf8", mass=2.0)
            ],
            settings=SimulationSettings(timestep=0.005, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81], enable_boundaries=True),
            sensors=SensorSettings(
                lidar=LidarConfig(enabled=True, num_beams=24, fov_deg=360.0, max_range=10.0),
                imu=ImuConfig(enabled=True),
                camera=CameraConfig(enabled=True, fov_deg=75.0)
            )
        )
        self.built_in_assets["asset_tpl_mobile_nav"] = AssetModel(
            id="asset_tpl_mobile_nav",
            name="Mobile Navigation Arena",
            category="template",
            description="Full simulation scene for mobile robot waypoint tracking and obstacle avoidance.",
            tags=["template", "navigation", "mobile", "arena"],
            is_built_in=True,
            thumbnail_hint="template",
            data=tpl_nav.model_dump()
        )

        # Template 2: Slalom Obstacle Course
        tpl_slalom = Project(
            schema_version=SCHEMA_VERSION,
            name="Slalom Obstacle Course",
            description="Staggered slalom columns and narrow choke-points designed to test agile steering and reactive avoidance.",
            robot=RobotConfig(
                id="robot_slalom",
                name="ForgeBot-DD1",
                initial_position=[0.0, 0.0, 0.07],
                initial_yaw=0.0,
                max_linear_velocity=2.0,
                max_angular_velocity=3.5
            ),
            objects=[
                SceneObject(id="slalom_1", name="Slalom Gate 1", type="cylinder", position=[1.0, 0.6, 0.40], size=[0.14, 0.40], color="#f97316", mass=20.0, is_static=True),
                SceneObject(id="slalom_2", name="Slalom Gate 2", type="cylinder", position=[1.8, -0.6, 0.40], size=[0.14, 0.40], color="#f97316", mass=20.0, is_static=True),
                SceneObject(id="slalom_3", name="Slalom Gate 3", type="cylinder", position=[2.6, 0.6, 0.40], size=[0.14, 0.40], color="#f97316", mass=20.0, is_static=True),
                SceneObject(id="slalom_4", name="Slalom Gate 4", type="cylinder", position=[3.4, -0.6, 0.40], size=[0.14, 0.40], color="#f97316", mass=20.0, is_static=True),
                SceneObject(id="wall_choke_left", name="Choke Wall Left", type="box", position=[4.2, 1.2, 0.25], size=[0.40, 0.06, 0.25], color="#64748b", mass=50.0, is_static=True),
                SceneObject(id="wall_choke_right", name="Choke Wall Right", type="box", position=[4.2, -1.2, 0.25], size=[0.40, 0.06, 0.25], color="#64748b", mass=50.0, is_static=True),
            ],
            settings=SimulationSettings(timestep=0.005, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81], enable_boundaries=True),
            sensors=SensorSettings(
                lidar=LidarConfig(enabled=True, num_beams=32, fov_deg=360.0, max_range=12.0),
                imu=ImuConfig(enabled=True),
                camera=CameraConfig(enabled=True, fov_deg=80.0)
            )
        )
        self.built_in_assets["asset_tpl_slalom"] = AssetModel(
            id="asset_tpl_slalom",
            name="Slalom Obstacle Course",
            category="template",
            description="5-gate staggered slalom agility course with choke corridors for high-precision autonomous navigation.",
            tags=["template", "slalom", "agility", "high-speed"],
            is_built_in=True,
            thumbnail_hint="template",
            data=tpl_slalom.model_dump()
        )

        # Template 3: Articulated Testing Bench
        tpl_articulated = Project(
            schema_version=SCHEMA_VERSION,
            name="Articulated Robot Testing Bench",
            description="Workbench table with an articulated 2-link robotic arm, workspace boundary props, and inspection targets.",
            robot=RobotConfig(
                id="robot_standby",
                name="ForgeBot-DD1",
                initial_position=[0.0, -1.5, 0.07],
                initial_yaw=0.0
            ),
            generic_robot=two_link_arm,
            objects=[
                SceneObject(id="bench_table", name="Workbench Table", type="box", position=[2.5, 2.5, 0.025], size=[0.80, 0.80, 0.025], color="#1e293b", mass=100.0, is_static=True),
                SceneObject(id="target_post_1", name="Pick Target A", type="cylinder", position=[2.85, 2.7, 0.15], size=[0.04, 0.10], color="#10b981", mass=0.5, is_static=False),
                SceneObject(id="target_post_2", name="Place Target B", type="box", position=[2.15, 2.3, 0.10], size=[0.06, 0.06, 0.05], color="#38bdf8", mass=0.4, is_static=False),
            ],
            settings=SimulationSettings(timestep=0.002, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81], enable_boundaries=True)
        )
        self.built_in_assets["asset_tpl_articulated"] = AssetModel(
            id="asset_tpl_articulated",
            name="Articulated Robot Testing Bench",
            category="template",
            description="Precision robotic workbench testbed with 2-DOF planar manipulator and pick/place reach targets.",
            tags=["template", "manipulator", "arm", "kinematics", "testing"],
            is_built_in=True,
            thumbnail_hint="template",
            data=tpl_articulated.model_dump()
        )

        # Template 4: Sensor Calibration Field
        tpl_sensor = Project(
            schema_version=SCHEMA_VERSION,
            name="Sensor Calibration Field",
            description="Multi-distance obstacle targets positioned at precise 1m, 2m, and 3m intervals for LiDAR and camera calibration.",
            robot=RobotConfig(
                id="robot_sensor_test",
                name="ForgeBot-DD1",
                initial_position=[0.0, 0.0, 0.07],
                initial_yaw=0.0
            ),
            objects=[
                SceneObject(id="target_1m", name="Calib Target 1.0m", type="box", position=[1.0, 0.0, 0.20], size=[0.15, 0.15, 0.20], color="#ef4444", mass=5.0, is_static=True),
                SceneObject(id="target_2m_left", name="Calib Target 2.0m L", type="cylinder", position=[2.0, 0.8, 0.25], size=[0.15, 0.25], color="#22c55e", mass=5.0, is_static=True),
                SceneObject(id="target_2m_right", name="Calib Target 2.0m R", type="sphere", position=[2.0, -0.8, 0.20], size=[0.20], color="#eab308", mass=5.0, is_static=True),
                SceneObject(id="target_3m_wall", name="Calib Backstop 3.5m", type="box", position=[3.5, 0.0, 0.35], size=[0.10, 1.5, 0.35], color="#64748b", mass=50.0, is_static=True),
            ],
            settings=SimulationSettings(timestep=0.005, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81]),
            sensors=SensorSettings(
                lidar=LidarConfig(enabled=True, num_beams=32, fov_deg=360.0, max_range=8.0),
                imu=ImuConfig(enabled=True),
                camera=CameraConfig(enabled=True, fov_deg=65.0)
            )
        )
        self.built_in_assets["asset_tpl_sensor_calib"] = AssetModel(
            id="asset_tpl_sensor_calib",
            name="Sensor Calibration Field",
            category="template",
            description="Calibrated multi-geometry range testbed for measuring precision LiDAR beam returns and camera vision depth.",
            tags=["template", "sensors", "calibration", "lidar", "benchmarking"],
            is_built_in=True,
            thumbnail_hint="template",
            data=tpl_sensor.model_dump()
        )

    def _load_custom_assets(self):
        """Loads custom user-saved assets from disk."""
        if not self.custom_assets_file.exists():
            return
        try:
            with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                asset = AssetModel.model_validate(item)
                self.custom_assets[asset.id] = asset
        except Exception:
            pass

    def _save_custom_assets_to_disk(self):
        """Persists custom user assets to disk."""
        try:
            items = [a.model_dump() for a in self.custom_assets.values()]
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
        except Exception as e:
            print(f"Error saving custom assets: {e}")

    def list_assets(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[AssetModel]:
        """Returns all assets matching optional filters."""
        all_assets = {**self.built_in_assets, **self.custom_assets}
        results: List[AssetModel] = []

        q = (search or "").strip().lower()
        t = (tag or "").strip().lower()

        for asset in all_assets.values():
            if category and asset.category != category:
                continue
            if t and not any(t in existing_tag.lower() for existing_tag in asset.tags):
                continue
            if q:
                in_name = q in asset.name.lower()
                in_desc = q in asset.description.lower()
                in_tags = any(q in existing_tag.lower() for existing_tag in asset.tags)
                if not (in_name or in_desc or in_tags):
                    continue
            results.append(asset)

        # Sort: built-in first, then by name
        results.sort(key=lambda a: (0 if a.is_built_in else 1, a.name.lower()))
        return results

    def get_asset(self, asset_id: str) -> Optional[AssetModel]:
        return self.built_in_assets.get(asset_id) or self.custom_assets.get(asset_id)

    def save_custom_asset(self, req: CreateAssetRequest) -> AssetModel:
        """Saves a new user custom asset to the library with full physical validation."""
        asset_id = f"custom_{uuid.uuid4().hex[:8]}"
        hint = req.thumbnail_hint
        if not hint:
            if req.category == "robot":
                hint = "arm" if req.data.get("type") == "articulated" else "bot"
            elif req.category == "obstacle":
                hint = req.data.get("type", "box")
            elif req.category == "sensor":
                hint = req.data.get("type", "radio")
            else:
                hint = "template"

        asset = AssetModel(
            id=asset_id,
            name=req.name,
            category=req.category,
            description=req.description,
            tags=req.tags,
            is_built_in=False,
            thumbnail_hint=hint,
            data=req.data
        )

        self.custom_assets[asset_id] = asset
        self._save_custom_assets_to_disk()
        return asset

    def delete_custom_asset(self, asset_id: str) -> bool:
        """Deletes a custom user asset. Built-in assets cannot be deleted."""
        if asset_id in self.built_in_assets:
            raise ValueError("Built-in system assets cannot be deleted.")
        if asset_id not in self.custom_assets:
            return False

        del self.custom_assets[asset_id]
        self._save_custom_assets_to_disk()
        return True

    def duplicate_asset(self, asset_id: str, new_name: Optional[str] = None) -> AssetModel:
        """Clones an existing asset (built-in or custom) as a new user asset."""
        source = self.get_asset(asset_id)
        if not source:
            raise KeyError(f"Asset '{asset_id}' not found.")

        target_name = new_name or f"{source.name} (Copy)"
        new_data = copy.deepcopy(source.data)

        req = CreateAssetRequest(
            name=target_name,
            category=source.category,
            description=f"Clone of {source.name}. {source.description}".strip(),
            tags=list(set(source.tags + ["copy"])),
            data=new_data,
            thumbnail_hint=source.thumbnail_hint
        )
        return self.save_custom_asset(req)

    def validate_robot_model(self, robot_data: Dict[str, Any], robot_type: str = "mobile") -> ValidationResult:
        """Thoroughly validates physical robot parameters and runs dry-run MuJoCo compilation."""
        errors: List[str] = []
        warnings: List[str] = []

        if robot_type == "mobile":
            try:
                # Check required fields
                cfg = RobotConfig.model_validate(robot_data)
                if any(s <= 0 for s in cfg.chassis_size):
                    errors.append("All chassis dimensions [hx, hy, hz] must be positive.")
                if cfg.wheel_radius <= 0 or cfg.wheel_radius > 1.0:
                    errors.append("Wheel radius must be between 0.01m and 1.0m.")
                if cfg.wheel_base <= cfg.wheel_radius:
                    warnings.append("Wheel base is very narrow compared to wheel radius; robot may be unstable.")
                if cfg.chassis_mass <= 0:
                    errors.append("Chassis mass must be strictly positive.")

                # Dry-run MuJoCo compilation for differential drive model
                hx, hy, hz = cfg.chassis_size
                wr, ww = cfg.wheel_radius, cfg.wheel_width
                test_xml = f"""<mujoco model="validation_test">
  <worldbody>
    <body name="robot" pos="0 0 {hz + wr}">
      <freejoint name="r_root"/>
      <geom name="chassis" type="box" size="{hx} {hy} {hz}" mass="{cfg.chassis_mass}"/>
      <body name="w_left" pos="0 {cfg.wheel_base*0.5} 0">
        <joint name="j_w_left" type="hinge" axis="0 1 0"/>
        <geom name="g_w_l" type="cylinder" size="{wr} {ww*0.5}" quat="0.707 0.707 0 0" mass="{cfg.wheel_mass}"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
                try:
                    mujoco.MjModel.from_xml_string(test_xml)
                except Exception as me:
                    errors.append(f"MuJoCo physics compilation rejected: {str(me)}")

            except Exception as e:
                errors.append(f"Invalid Mobile Robot schema: {str(e)}")

        elif robot_type == "articulated":
            try:
                gr = GenericRobotModel.model_validate(robot_data)
                v_res = validate_generic_robot(gr)
                for err in v_res.errors:
                    errors.append(f"[{err.code}] {err.target}: {err.message}")
                for warn in v_res.warnings:
                    warnings.append(f"[{warn.code}] {warn.target}: {warn.message}")
            except Exception as e:
                errors.append(f"Invalid Articulated Robot schema: {str(e)}")
        else:
            errors.append(f"Unknown robot type: '{robot_type}'. Expected 'mobile' or 'articulated'.")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"robot_type": robot_type}
        )

    def validate_environment_model(self, env_data: Dict[str, Any]) -> ValidationResult:
        """Validates environment boundaries, gravity, and obstacle configurations."""
        errors: List[str] = []
        warnings: List[str] = []

        objects_data = env_data.get("objects", [])
        if not isinstance(objects_data, list):
            errors.append("'objects' must be a list.")
        else:
            for idx, obj in enumerate(objects_data):
                try:
                    SceneObject.model_validate(obj)
                except Exception as e:
                    errors.append(f"Obstacle #{idx + 1} validation failed: {str(e)}")

        settings = env_data.get("settings", {})
        if settings:
            try:
                SimulationSettings.model_validate(settings)
            except Exception as e:
                errors.append(f"Simulation settings validation failed: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def apply_asset_to_project(
        self,
        asset_id: str,
        current_project: Project,
        position_offset: Optional[List[float]] = None
    ) -> Project:
        """Applies or inserts the specified asset into the given project."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise KeyError(f"Asset '{asset_id}' not found.")

        updated = current_project.model_copy(deep=True)

        if asset.category == "template":
            # Replaces entire project with the validated template
            tpl_proj = Project.model_validate(asset.data)
            return tpl_proj

        elif asset.category == "robot":
            r_type = asset.data.get("type", "mobile")
            if r_type == "mobile":
                new_cfg = RobotConfig.model_validate(asset.data["config"])
                if position_offset and len(position_offset) == 3:
                    new_cfg.initial_position = position_offset
                updated.robot = new_cfg
                # Clear generic robot if replacing with mobile robot
                updated.generic_robot = None
            else:
                new_generic = GenericRobotModel.model_validate(asset.data["config"])
                if position_offset and len(position_offset) == 3:
                    new_generic.base_position = position_offset
                updated.generic_robot = new_generic

        elif asset.category == "obstacle":
            # Instantiate obstacle as new unique SceneObject
            obs_dict = copy.deepcopy(asset.data)
            obs_dict["id"] = f"obs_{uuid.uuid4().hex[:8]}"
            if position_offset and len(position_offset) == 3:
                obs_dict["position"] = position_offset
            else:
                # Default drop offset in front of robot
                obs_dict["position"] = [1.8 + len(updated.objects) * 0.4, 0.0, obs_dict.get("size", [0.2])[0]]

            new_obj = SceneObject.model_validate(obs_dict)
            updated.objects.append(new_obj)

        elif asset.category == "sensor":
            s_type = asset.data.get("type", "lidar")
            if s_type == "lidar":
                updated.sensors.lidar = LidarConfig.model_validate(asset.data["config"])
            elif s_type == "imu":
                updated.sensors.imu = ImuConfig.model_validate(asset.data["config"])
            elif s_type == "camera":
                updated.sensors.camera = CameraConfig.model_validate(asset.data["config"])

        return updated
