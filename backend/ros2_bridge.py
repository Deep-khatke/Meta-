from __future__ import annotations
import time
import math
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
    def serialize_tf(cls, sim_state: SimulationState) -> Dict[str, Any]:
        """Serializes full TF2 coordinate transformation tree matching ROS 2 standard conventions."""
        header_map = cls.build_header(sim_state.sim_time, "map")
        header_odom = cls.build_header(sim_state.sim_time, "odom")
        header_base = cls.build_header(sim_state.sim_time, "base_link")
        
        rx, ry, rz = sim_state.robot.position
        qw, qx, qy, qz = sim_state.robot.quaternion

        transforms = [
            # map -> odom (identity in local frame)
            {
                "header": header_map.model_dump(),
                "child_frame_id": "odom",
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            },
            # odom -> base_footprint (robot planar position)
            {
                "header": header_odom.model_dump(),
                "child_frame_id": "base_footprint",
                "transform": {
                    "translation": {"x": round(rx, 4), "y": round(ry, 4), "z": 0.0},
                    "rotation": {"x": round(qx, 4), "y": round(qy, 4), "z": round(qz, 4), "w": round(qw, 4)}
                }
            },
            # base_footprint -> base_link (chassis elevation)
            {
                "header": cls.build_header(sim_state.sim_time, "base_footprint").model_dump(),
                "child_frame_id": "base_link",
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": round(rz, 4)},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            },
            # base_link -> lidar_link (mounted sensor turret)
            {
                "header": header_base.model_dump(),
                "child_frame_id": "lidar_link",
                "transform": {
                    "translation": {"x": 0.10, "y": 0.0, "z": 0.09},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            },
            # base_link -> imu_link (geometric center)
            {
                "header": header_base.model_dump(),
                "child_frame_id": "imu_link",
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.05},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            },
            # base_link -> camera_link (forward optical payload)
            {
                "header": header_base.model_dump(),
                "child_frame_id": "camera_link",
                "transform": {
                    "translation": {"x": 0.18, "y": 0.0, "z": 0.08},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                }
            }
        ]
        return {"transforms": transforms}

    @classmethod
    def parse_cmd_vel(cls, twist_payload: Dict[str, Any]) -> TeleopCommand:
        linear_val = 0.0
        angular_val = 0.0
        if "linear" in twist_payload and isinstance(twist_payload["linear"], dict):
            linear_val = float(twist_payload["linear"].get("x", 0.0))
        elif "linear" in twist_payload and isinstance(twist_payload["linear"], (int, float)):
            linear_val = float(twist_payload["linear"])
        if "angular" in twist_payload and isinstance(twist_payload["angular"], dict):
            angular_val = float(twist_payload["angular"].get("z", 0.0))
        elif "angular" in twist_payload and isinstance(twist_payload["angular"], (int, float)):
            angular_val = float(twist_payload["angular"])
        return TeleopCommand(linear=linear_val, angular=angular_val)


class ROS2BagRecorder:
    """Manages recording of live simulation telemetry into structured ROS 2 bag datasets."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("experiments/rosbags")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.current_bag_id: Optional[str] = None
        self.current_bag_name: Optional[str] = None
        self.start_wall_time: float = 0.0
        self.start_sim_time: float = 0.0
        self.recorded_messages: List[Dict[str, Any]] = []
        self.topic_counts: Dict[str, int] = {}

    def start_recording(self, name: Optional[str] = None, sim_time: float = 0.0, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        bag_id = f"rosbag_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        bag_name = name or f"simulation_run_{int(time.time())}"
        self.is_recording = True
        self.current_bag_id = bag_id
        self.current_bag_name = bag_name
        self.start_wall_time = time.time()
        self.start_sim_time = sim_time
        self.recorded_messages = []
        self.topic_counts = {
            "/odom": 0,
            "/scan": 0,
            "/imu": 0,
            "/joint_states": 0,
            "/tf": 0,
            "/clock": 0
        }
        return {
            "status": "recording_started",
            "bag_id": bag_id,
            "bag_name": bag_name,
            "timestamp": self.start_wall_time
        }

    def record_step(self, sim_state: SimulationState):
        if not self.is_recording or not self.current_bag_id:
            return

        t = sim_state.sim_time

        # 1. /clock
        clock_msg = ROS2Bridge.serialize_clock(t)
        self.recorded_messages.append({
            "topic": "/clock",
            "type": "rosgraph_msgs/Clock",
            "timestamp": t,
            "data": clock_msg
        })
        self.topic_counts["/clock"] = self.topic_counts.get("/clock", 0) + 1

        # 2. /odom
        odom_msg = ROS2Bridge.serialize_odometry(sim_state)
        self.recorded_messages.append({
            "topic": "/odom",
            "type": "nav_msgs/Odometry",
            "timestamp": t,
            "data": odom_msg
        })
        self.topic_counts["/odom"] = self.topic_counts.get("/odom", 0) + 1

        # 3. /scan (if lidar present)
        scan_msg = ROS2Bridge.serialize_laser_scan(sim_state)
        if scan_msg:
            self.recorded_messages.append({
                "topic": "/scan",
                "type": "sensor_msgs/LaserScan",
                "timestamp": t,
                "data": scan_msg
            })
            self.topic_counts["/scan"] = self.topic_counts.get("/scan", 0) + 1

        # 4. /imu
        imu_msg = ROS2Bridge.serialize_imu(sim_state)
        if imu_msg:
            self.recorded_messages.append({
                "topic": "/imu",
                "type": "sensor_msgs/Imu",
                "timestamp": t,
                "data": imu_msg
            })
            self.topic_counts["/imu"] = self.topic_counts.get("/imu", 0) + 1

        # 5. /tf
        tf_msg = ROS2Bridge.serialize_tf(sim_state)
        self.recorded_messages.append({
            "topic": "/tf",
            "type": "tf2_msgs/TFMessage",
            "timestamp": t,
            "data": tf_msg
        })
        self.topic_counts["/tf"] = self.topic_counts.get("/tf", 0) + 1

    def stop_recording(self, sim_time: float = 0.0) -> Dict[str, Any]:
        if not self.is_recording or not self.current_bag_id:
            return {"status": "not_recording"}

        duration = round(time.time() - self.start_wall_time, 2)
        sim_duration = round(sim_time - self.start_sim_time, 2)
        total_messages = len(self.recorded_messages)

        bag_metadata = {
            "bag_id": self.current_bag_id,
            "bag_name": self.current_bag_name,
            "created_at": self.start_wall_time,
            "wall_duration_sec": duration,
            "sim_duration_sec": sim_duration,
            "total_messages": total_messages,
            "topic_counts": self.topic_counts,
            "storage_format": "mcap_json",
            "topics": [
                {"name": "/clock", "type": "rosgraph_msgs/Clock"},
                {"name": "/odom", "type": "nav_msgs/Odometry"},
                {"name": "/scan", "type": "sensor_msgs/LaserScan"},
                {"name": "/imu", "type": "sensor_msgs/Imu"},
                {"name": "/tf", "type": "tf2_msgs/TFMessage"}
            ]
        }

        bag_data = {
            "metadata": bag_metadata,
            "messages": self.recorded_messages
        }

        file_path = self.storage_dir / f"{self.current_bag_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(bag_data, f, indent=2)
        except Exception as e:
            return {"status": "error", "error": str(e)}

        self.is_recording = False
        res = {
            "status": "recording_saved",
            "bag_id": self.current_bag_id,
            "bag_name": self.current_bag_name,
            "file_path": str(file_path),
            "metadata": bag_metadata
        }
        self.current_bag_id = None
        self.current_bag_name = None
        return res

    def list_bags(self) -> List[Dict[str, Any]]:
        bags = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                meta = content.get("metadata", {})
                meta["file_size_bytes"] = file.stat().st_size
                bags.append(meta)
            except Exception:
                pass
        bags.sort(key=lambda b: b.get("created_at", 0), reverse=True)
        return bags

    def get_bag(self, bag_id: str) -> Optional[Dict[str, Any]]:
        file_path = self.storage_dir / f"{bag_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def delete_bag(self, bag_id: str) -> bool:
        file_path = self.storage_dir / f"{bag_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class SwarmOrchestrator:
    """Manages multi-robot swarm deployment, formation geometry, and cooperative control."""

    FORMATIONS = ["line", "column", "v_shape", "circle", "diamond"]

    @classmethod
    def calculate_formation_positions(
        cls,
        leader_pos: List[float],
        leader_yaw_deg: float,
        formation: str = "v_shape",
        count: int = 3,
        spacing: float = 1.2
    ) -> List[List[float]]:
        """Calculates collision-free follower coordinates relative to leader pose."""
        lx, ly = leader_pos[0], leader_pos[1]
        yaw_rad = math.radians(leader_yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        positions = [[round(lx, 3), round(ly, 3), 0.07]]  # Leader is index 0

        for i in range(1, count):
            dx_rel, dy_rel = 0.0, 0.0
            idx = i  # 1-indexed follower
            
            if formation == "v_shape":
                # Alternating wings of the V trailing behind leader
                side = 1.0 if (idx % 2 == 1) else -1.0
                rank = (idx + 1) // 2
                dx_rel = -rank * spacing * 0.8
                dy_rel = side * rank * spacing
            elif formation == "line":
                # Lateral abreast line
                side = 1.0 if (idx % 2 == 1) else -1.0
                rank = (idx + 1) // 2
                dx_rel = 0.0
                dy_rel = side * rank * spacing
            elif formation == "column":
                # In-trail convoy behind leader
                dx_rel = -idx * spacing
                dy_rel = 0.0
            elif formation == "circle":
                # Radial ring around leader
                angle = (2.0 * math.pi * idx) / max(1, count - 1)
                dx_rel = math.cos(angle) * spacing
                dy_rel = math.sin(angle) * spacing
            elif formation == "diamond":
                # Tactical diamond
                if idx == 1:
                    dx_rel, dy_rel = -spacing * 0.7, spacing * 0.7
                elif idx == 2:
                    dx_rel, dy_rel = -spacing * 0.7, -spacing * 0.7
                elif idx == 3:
                    dx_rel, dy_rel = -spacing * 1.4, 0.0
                else:
                    dx_rel, dy_rel = -spacing * (idx * 0.5), (1 if idx % 2 else -1) * spacing

            # Rotate relative offsets into world frame
            world_x = lx + (dx_rel * cos_y - dy_rel * sin_y)
            world_y = ly + (dx_rel * sin_y + dy_rel * cos_y)
            positions.append([round(world_x, 3), round(world_y, 3), 0.07])

        return positions


# Global Singletons
ros2_bag_recorder = ROS2BagRecorder()
swarm_orchestrator = SwarmOrchestrator()

