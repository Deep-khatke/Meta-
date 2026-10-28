from __future__ import annotations
import time
import math
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from backend.models import SimulationState, TeleopCommand

class ROS2Header(BaseModel):
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str

class ROS2Vector3(BaseModel):
    x: float
    y: float
    z: float

class ROS2Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float

class ROS2Twist(BaseModel):
    linear: ROS2Vector3
    angular: ROS2Vector3

class ROS2LaserScan(BaseModel):
    header: ROS2Header
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float = 0.0
    scan_time: float = 0.05
    range_min: float
    range_max: float
    ranges: List[float]

class ROS2Odometry(BaseModel):
    header: ROS2Header
    child_frame_id: str = "base_link"
    pose: Dict[str, Any]
    twist: Dict[str, Any]

class ROS2Imu(BaseModel):
    header: ROS2Header
    orientation: ROS2Quaternion
    orientation_covariance: List[float] = Field(default_factory=lambda: [0.0] * 9)
    angular_velocity: ROS2Vector3
    angular_velocity_covariance: List[float] = Field(default_factory=lambda: [0.0] * 9)
    linear_acceleration: ROS2Vector3
    linear_acceleration_covariance: List[float] = Field(default_factory=lambda: [0.0] * 9)

class ROS2JointState(BaseModel):
    header: ROS2Header
    name: List[str]
    position: List[float]
    velocity: List[float]
    effort: List[float]

class ROS2Clock(BaseModel):
    clock: Dict[str, int]


class ROS2Bridge:
    """Standard ROS 2 message serialization bridge mapping RoboForge simulation telemetry to ROS 2 topics."""

    @staticmethod
    def sim_time_to_stamp(sim_time: float) -> Tuple[int, int]:
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        return sec, nanosec

    @staticmethod
    def build_header(sim_time: float, frame_id: str) -> ROS2Header:
        sec, nanosec = ROS2Bridge.sim_time_to_stamp(sim_time)
        return ROS2Header(stamp_sec=sec, stamp_nanosec=nanosec, frame_id=frame_id)

    @classmethod
    def serialize_clock(cls, sim_time: float) -> Dict[str, Any]:
        sec, nanosec = cls.sim_time_to_stamp(sim_time)
        return {"clock": {"sec": sec, "nanosec": nanosec}}

    @classmethod
    def serialize_laser_scan(cls, sim_state: SimulationState, frame_id: str = "lidar_link") -> Optional[Dict[str, Any]]:
        if not sim_state.lidar_readings:
            return None
        
        n = len(sim_state.lidar_readings)
        header = cls.build_header(sim_state.sim_time, frame_id)
        
        # Recover angle range
        if sim_state.lidar_angles:
            min_deg = min(sim_state.lidar_angles)
            max_deg = max(sim_state.lidar_angles)
            ang_min = math.radians(min_deg)
            ang_max = math.radians(max_deg)
            inc = (ang_max - ang_min) / max(1, n - 1)
        else:
            ang_min = -math.pi
            ang_max = math.pi
            inc = (2.0 * math.pi) / n

        msg = ROS2LaserScan(
            header=header,
            angle_min=round(ang_min, 4),
            angle_max=round(ang_max, 4),
            angle_increment=round(inc, 4),
            range_min=0.05,
            range_max=8.0,
            ranges=sim_state.lidar_readings
        )
        return msg.model_dump()

    @classmethod
    def serialize_odometry(cls, sim_state: SimulationState, frame_id: str = "odom") -> Dict[str, Any]:
        header = cls.build_header(sim_state.sim_time, frame_id)
        rx, ry, rz = sim_state.robot.position
        qw, qx, qy, qz = sim_state.robot.quaternion
        vx, vy, vz = sim_state.robot.linear_velocity
        wx, wy, wz = sim_state.robot.angular_velocity

        msg = {
            "header": header.model_dump(),
            "child_frame_id": "base_footprint",
            "pose": {
                "pose": {
                    "position": {"x": round(rx, 4), "y": round(ry, 4), "z": round(rz, 4)},
                    "orientation": {"x": round(qx, 4), "y": round(qy, 4), "z": round(qz, 4), "w": round(qw, 4)}
                },
                "covariance": [0.0] * 36
            },
            "twist": {
                "twist": {
                    "linear": {"x": round(vx, 4), "y": round(vy, 4), "z": round(vz, 4)},
                    "angular": {"x": round(wx, 4), "y": round(wy, 4), "z": round(wz, 4)}
                },
                "covariance": [0.0] * 36
            }
        }
        return msg

    @classmethod
    def serialize_imu(cls, sim_state: SimulationState, frame_id: str = "imu_link") -> Optional[Dict[str, Any]]:
        if not sim_state.imu_linear_acceleration or not sim_state.imu_angular_velocity:
            return None
        
        header = cls.build_header(sim_state.sim_time, frame_id)
        qw, qx, qy, qz = sim_state.robot.quaternion
        ax, ay, az = sim_state.imu_linear_acceleration
        wx, wy, wz = sim_state.imu_angular_velocity

        msg = ROS2Imu(
            header=header,
            orientation=ROS2Quaternion(x=qx, y=qy, z=qz, w=qw),
            angular_velocity=ROS2Vector3(x=wx, y=wy, z=wz),
            linear_acceleration=ROS2Vector3(x=ax, y=ay, z=az)
        )
        return msg.model_dump()

    @classmethod
    def serialize_joint_states(cls, sim_state: SimulationState, frame_id: str = "base_link") -> Dict[str, Any]:
        header = cls.build_header(sim_state.sim_time, frame_id)
        names = list(sim_state.joint_positions.keys())
        positions = [sim_state.joint_positions[n] for n in names]
        velocities = [sim_state.joint_velocities.get(n, 0.0) for n in names]
        efforts = [sim_state.joint_torques.get(n, 0.0) for n in names]

        msg = ROS2JointState(
            header=header,
            name=names,
            position=positions,
            velocity=velocities,
            effort=efforts
        )
        return msg.model_dump()

    @classmethod
    def parse_cmd_vel(cls, twist_payload: Dict[str, Any]) -> TeleopCommand:
        linear_val = 0.0
        angular_val = 0.0
        if "linear" in twist_payload and isinstance(twist_payload["linear"], dict):
            linear_val = float(twist_payload["linear"].get("x", 0.0))
        if "angular" in twist_payload and isinstance(twist_payload["angular"], dict):
            angular_val = float(twist_payload["angular"].get("z", 0.0))
        return TeleopCommand(linear=linear_val, angular=angular_val)
