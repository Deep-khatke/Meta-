from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import uuid

SCHEMA_VERSION = "1.0.0"

class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class SceneObject(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., min_length=1, max_length=50)
    type: Literal["box", "sphere", "cylinder"]
    position: List[float] = Field(default=[0.0, 0.0, 0.5])
    # Euler rotation in degrees [roll, pitch, yaw]
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0])
    # Size dimensions:
    # box: [half_x, half_y, half_z] in meters
    # sphere: [radius] in meters
    # cylinder: [radius, half_height] in meters
    size: List[float] = Field(default=[0.3, 0.3, 0.3])
    color: str = Field(default="#8892b0")
    is_static: bool = Field(default=False, description="If True, object is fixed in world. If False, dynamic physics body.")
    mass: float = Field(default=1.0, gt=0.0, le=1000.0)
    # [sliding, torsional, rolling] friction
    friction: List[float] = Field(default=[1.0, 0.005, 0.0001], description="MuJoCo geom friction [sliding, torsional, rolling]")
    visible: bool = Field(default=True)
    is_locked: bool = Field(default=False, description="If True, object is locked and cannot be transformed via gizmo.")

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("Position must have exactly 3 coordinates [x, y, z]")
        for val in v:
            if not (-100.0 <= val <= 100.0):
                raise ValueError("Position coordinates must be between -100 and 100 meters")
        return v

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("Rotation must have exactly 3 angles [roll, pitch, yaw]")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: List[float], info) -> List[float]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("All dimensions must be positive numbers")
        return v

    @field_validator("friction")
    @classmethod
    def validate_friction(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("friction must have exactly 3 values: [sliding, torsional, rolling]")
        for f in v:
            if f < 0:
                raise ValueError("friction values must be non-negative")
        return v


class RobotConfig(BaseModel):
    id: str = "diff_drive_robot"
    name: str = "ForgeBot-DD1"
    initial_position: List[float] = Field(default=[0.0, 0.0, 0.07])
    initial_yaw: float = Field(default=0.0) # in degrees
    chassis_size: List[float] = Field(default=[0.20, 0.14, 0.05]) # [half_x, half_y, half_z]
    wheel_radius: float = Field(default=0.06, gt=0.01, le=0.5)
    wheel_width: float = Field(default=0.03, gt=0.005, le=0.2)
    wheel_base: float = Field(default=0.28, gt=0.05, le=1.5) # distance between left and right wheels
    caster_radius: float = Field(default=0.025, gt=0.005, le=0.1)
    chassis_mass: float = Field(default=3.5, gt=0.1, le=100.0)
    wheel_mass: float = Field(default=0.5, gt=0.05, le=10.0)
    max_linear_velocity: float = Field(default=1.5, gt=0.05, le=10.0) # m/s
    max_angular_velocity: float = Field(default=3.0, gt=0.1, le=15.0) # rad/s

    @field_validator("initial_position")
    @classmethod
    def validate_init_pos(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("initial_position must be [x, y, z]")
        return v


# --- Generic Robot & Joint Models ---
JointType = Literal["revolute", "continuous", "prismatic", "fixed"]

class JointLimit(BaseModel):
    lower: float = -3.14159
    upper: float = 3.14159
    effort: float = 100.0
    velocity: float = 10.0

class JointModel(BaseModel):
    name: str
    type: JointType = "revolute"
    parent_link: str
    child_link: str
    origin_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    origin_rpy: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    limit: Optional[JointLimit] = None
    damping: float = 0.1
    armature: float = 0.01
    actuator_mode: Optional[Literal["position", "velocity", "torque"]] = "position"
    kp: float = 100.0
    kv: float = 10.0

class LinkInertial(BaseModel):
    mass: float = 1.0
    origin_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    ixx: float = 0.01
    iyy: float = 0.01
    izz: float = 0.01

class LinkGeom(BaseModel):
    type: Literal["box", "sphere", "cylinder"]
    size: List[float] # box: [hx, hy, hz], sphere: [r], cylinder: [r, hh]
    origin_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    origin_rpy: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    color: str = "#4a90e2"

class LinkModel(BaseModel):
    name: str
    visual: Optional[LinkGeom] = None
    collision: Optional[LinkGeom] = None
    inertial: Optional[LinkInertial] = None

class GenericRobotModel(BaseModel):
    name: str = "GenericRobot"
    root_link: str = "base_link"
    links: List[LinkModel] = Field(default_factory=list)
    joints: List[JointModel] = Field(default_factory=list)
    base_position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_rpy: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_fixed_base: bool = True


# --- Sensor Configs ---
class LidarConfig(BaseModel):
    enabled: bool = True
    mount_pos: List[float] = Field(default_factory=lambda: [0.10, 0.0, 0.09])
    num_beams: int = Field(default=16, ge=4, le=128)
    fov_deg: float = Field(default=360.0, ge=10.0, le=360.0)
    max_range: float = Field(default=8.0, ge=0.5, le=50.0)
    min_range: float = Field(default=0.05, ge=0.01, le=1.0)
    noise_std: float = Field(default=0.0, ge=0.0, le=0.5, description="Gaussian noise standard deviation in meters")
    dropout_prob: float = Field(default=0.0, ge=0.0, le=0.5, description="Probability of dropped beam / missed return")
    channels: int = Field(default=1, ge=1, le=64, description="Vertical channel count (1 for 2D LiDAR, >1 for 3D LiDAR)")

class ImuConfig(BaseModel):
    enabled: bool = True
    mount_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.05])
    accel_noise_std: float = Field(default=0.0, ge=0.0, le=0.5, description="Accelerometer Gaussian noise std (m/s^2)")
    gyro_noise_std: float = Field(default=0.0, ge=0.0, le=0.5, description="Gyroscope Gaussian noise std (rad/s)")
    accel_bias: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Accelerometer constant drift bias")
    gyro_bias: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Gyroscope constant drift bias")

class CameraConfig(BaseModel):
    enabled: bool = False
    mount_pos: List[float] = Field(default_factory=lambda: [0.18, 0.0, 0.08])
    mount_rpy: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: int = Field(default=320, ge=64, le=1920)
    height: int = Field(default=240, ge=64, le=1080)
    fov_deg: float = Field(default=60.0, ge=10.0, le=140.0)

class EncoderConfig(BaseModel):
    enabled: bool = True
    ticks_per_rev: int = Field(default=1024, ge=16, le=16384)
    noise_std: float = Field(default=0.0, ge=0.0, le=0.1)

class SensorSettings(BaseModel):
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    imu: ImuConfig = Field(default_factory=ImuConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    encoders: EncoderConfig = Field(default_factory=EncoderConfig)

class URDFImportResult(BaseModel):
    status: Literal["success", "warning", "error"]
    robot: Optional[GenericRobotModel] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    total_links: int = 0
    total_joints: int = 0

class MJCFImportResult(BaseModel):
    status: Literal["success", "warning", "error"]
    robot: Optional[GenericRobotModel] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    total_links: int = 0
    total_joints: int = 0


class SimulationSettings(BaseModel):
    timestep: float = Field(default=0.01, ge=0.001, le=0.05, description="Physics timestep in seconds")
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=5.0, description="Realtime playback speed factor")
    gravity: List[float] = Field(default=[0.0, 0.0, -9.81])
    world_boundaries: List[float] = Field(default=[-15.0, 15.0, -15.0, 15.0, 0.0, 8.0], description="[xmin, xmax, ymin, ymax, zmin, zmax]")
    enable_boundaries: bool = Field(default=True, description="Containment boundary walls")

    @field_validator("gravity")
    @classmethod
    def validate_gravity(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("gravity must have 3 components [gx, gy, gz]")
        return v

    @field_validator("world_boundaries")
    @classmethod
    def validate_boundaries(cls, v: List[float]) -> List[float]:
        if len(v) != 6:
            raise ValueError("world_boundaries must have 6 values [xmin, xmax, ymin, ymax, zmin, zmax]")
        if v[0] >= v[1] or v[2] >= v[3] or v[4] >= v[5]:
            raise ValueError("Min boundaries must be strictly less than max boundaries")
        return v


class Project(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    name: str = Field(default="Default Scene", min_length=1, max_length=100)
    description: Optional[str] = "RoboForge Simulation Project"
    robot: RobotConfig = Field(default_factory=RobotConfig)
    generic_robot: Optional[GenericRobotModel] = None
    sensors: SensorSettings = Field(default_factory=SensorSettings)
    objects: List[SceneObject] = Field(default_factory=list)
    settings: SimulationSettings = Field(default_factory=SimulationSettings)


class TeleopCommand(BaseModel):
    # Normalized command [-1.0 to 1.0] or direct velocity [m/s, rad/s]
    linear: float = Field(default=0.0, ge=-10.0, le=10.0)
    angular: float = Field(default=0.0, ge=-15.0, le=15.0)


class RobotState(BaseModel):
    position: List[float] = [0.0, 0.0, 0.0]
    quaternion: List[float] = [1.0, 0.0, 0.0, 0.0] # [w, x, y, z] Three.js compatible
    linear_velocity: List[float] = [0.0, 0.0, 0.0]
    angular_velocity: List[float] = [0.0, 0.0, 0.0]
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    left_wheel_velocity: float = 0.0
    right_wheel_velocity: float = 0.0
    forward_vector: List[float] = [1.0, 0.0, 0.0]


class ObjectPhysicsState(BaseModel):
    id: str
    position: List[float]
    quaternion: List[float] # [w, x, y, z]
    linear_velocity: List[float] = [0.0, 0.0, 0.0]


class ContactInfo(BaseModel):
    geom1: str
    geom2: str
    position: List[float]
    distance: float
    force_normal: float = 0.0
    force_tangent: float = 0.0
    penetration: float = 0.0


class OdometryState(BaseModel):
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw: float = 0.0
    linear_velocity: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    distance_traveled: float = 0.0


class SimulationSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = 0.0
    sim_time: float = 0.0
    step_count: int = 0
    qpos: List[float] = Field(default_factory=list)
    qvel: List[float] = Field(default_factory=list)
    act: List[float] = Field(default_factory=list)
    joint_targets: Dict[str, float] = Field(default_factory=dict)
    teleop_linear: float = 0.0
    teleop_angular: float = 0.0


class SimulationState(BaseModel):
    status: Literal["running", "paused", "stopped", "error"] = "stopped"
    sim_time: float = 0.0
    wall_time: float = 0.0
    step_count: int = 0
    real_time_factor: float = 1.0
    robot: RobotState = Field(default_factory=RobotState)
    odometry: Optional[OdometryState] = None
    objects: Dict[str, ObjectPhysicsState] = Field(default_factory=dict)
    contacts: List[ContactInfo] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # Ground-truth sensors extracted directly from MuJoCo simulation
    lidar_readings: List[float] = Field(default_factory=list)
    lidar_angles: List[float] = Field(default_factory=list)
    imu_linear_acceleration: Optional[List[float]] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    imu_angular_velocity: Optional[List[float]] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Generic Joint Telemetry (Positions, Velocities, Torques)
    joint_positions: Dict[str, float] = Field(default_factory=dict)
    joint_velocities: Dict[str, float] = Field(default_factory=dict)
    joint_torques: Dict[str, float] = Field(default_factory=dict)
    encoder_ticks: Dict[str, int] = Field(default_factory=dict)


class LogMessage(BaseModel):
    timestamp: float
    level: Literal["info", "warning", "error", "debug"]
    message: str
    source: str = "simulation"


# --- Robot Studio Validation & Kinematics Models ---

class RobotDiagnostic(BaseModel):
    code: str
    severity: Literal["error", "warning", "info"]
    target: str # e.g. "link:base_link" or "joint:shoulder" or "tree"
    message: str
    suggestion: Optional[str] = None

class RobotValidationResult(BaseModel):
    valid: bool
    errors: List[RobotDiagnostic] = Field(default_factory=list)
    warnings: List[RobotDiagnostic] = Field(default_factory=list)
    summary: str = ""

class LinkPose(BaseModel):
    link_name: str
    position: List[float] # [x, y, z] in world/base coordinates
    quaternion: List[float] # [w, x, y, z]
    rpy_deg: List[float] # [roll, pitch, yaw] in degrees
    parent_link: Optional[str] = None

class JointFrame(BaseModel):
    joint_name: str
    origin_position: List[float]
    axis_vector: List[float]
    joint_type: str
    current_position: float
    limits: Optional[JointLimit] = None

class FKResult(BaseModel):
    success: bool
    end_effector_link: str
    end_effector_position: List[float]
    end_effector_quaternion: List[float]
    end_effector_rpy_deg: List[float]
    link_poses: Dict[str, LinkPose] = Field(default_factory=dict)
    joint_frames: Dict[str, JointFrame] = Field(default_factory=dict)
    total_reach: float = 0.0

class FKRequest(BaseModel):
    robot: GenericRobotModel
    joint_positions: Dict[str, float] = Field(default_factory=dict)

class IKRequest(BaseModel):
    robot: GenericRobotModel
    target_position: List[float]
    target_rpy_deg: Optional[List[float]] = None
    initial_joint_positions: Optional[Dict[str, float]] = None
    tolerance: float = 0.005 # 5mm
    max_iterations: int = 150
    end_effector_link: Optional[str] = None

class IKResult(BaseModel):
    success: bool
    status: Literal["SUCCESS", "TARGET_OUT_OF_REACH", "JOINT_LIMITS_BLOCKED", "CONVERGENCE_FAILED", "ERROR"]
    iterations: int
    error_distance: float # in meters
    joint_solution: Dict[str, float]
    achieved_position: List[float]
    achieved_rpy_deg: List[float]
    message: str


