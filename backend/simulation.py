from __future__ import annotations
import asyncio
import time
import math
import logging
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import mujoco

import zlib
import struct
import base64
import uuid

from backend.models import (
    Project, SceneObject, RobotConfig, GenericRobotModel, LinkModel, JointModel,
    SimulationSettings, SimulationState, RobotState, ObjectPhysicsState,
    ContactInfo, LogMessage, TeleopCommand, SimulationSnapshot, OdometryState
)

logger = logging.getLogger("roboforge.simulation")

def hex_to_rgba_str(hex_code: str, alpha: float = 1.0) -> str:
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        r = int(hex_code[0:2], 16) / 255.0
        g = int(hex_code[2:4], 16) / 255.0
        b = int(hex_code[4:6], 16) / 255.0
        return f"{r:.3f} {g:.3f} {b:.3f} {alpha:.3f}"
    return f"0.5 0.5 0.5 {alpha:.3f}"

def quat_to_forward(q: List[float]) -> List[float]:
    """Convert quaternion [w, x, y, z] to forward direction vector in XY plane."""
    w, x, y, z = q
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm > 1e-6:
        return [fx / norm, fy / norm, fz / norm]
    return [1.0, 0.0, 0.0]

def quat_to_yaw(q: List[float]) -> float:
    """Extract yaw angle (radians) from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def euler_deg_to_quat_str(euler_deg: List[float]) -> str:
    """Euler angles [roll, pitch, yaw] in degrees to MuJoCo quat 'w x y z'."""
    r = math.radians(euler_deg[0])
    p = math.radians(euler_deg[1])
    y = math.radians(euler_deg[2])
    
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_q = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:.6f} {x:.6f} {y_q:.6f} {z:.6f}"


def rgb_to_png(rgb_array: np.ndarray) -> bytes:
    """Encode an RGB uint8 numpy array to a valid PNG byte stream using standard library."""
    h, w, _ = rgb_array.shape
    raw_data = bytearray()
    for row in rgb_array:
        raw_data.append(0)  # filter type 0 (None)
        raw_data.extend(row.tobytes())
    compressed = zlib.compress(bytes(raw_data))

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")
    return header + ihdr + idat + iend


class MuJoCoEngine:
    def __init__(self, project: Project, log_callback: Optional[Callable[[LogMessage], Any]] = None):
        self.project = project
        self.log_callback = log_callback
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        
        self.status: str = "stopped"  # "running" | "paused" | "stopped" | "error"
        self.sim_time: float = 0.0
        self.wall_time_start: float = time.time()
        self.step_count: int = 0
        self.errors: List[str] = []
        
        self.teleop_cmd = TeleopCommand(linear=0.0, angular=0.0)
        self.last_teleop_time: float = 0.0
        
        # Generic joint control targets: joint_name -> commanded position/velocity/torque
        self.joint_targets: Dict[str, float] = {}
        self.joint_actuator_ids: Dict[str, int] = {}
        
        # Actuator IDs and joint addresses for diff drive robot
        self.left_motor_id: int = -1
        self.right_motor_id: int = -1
        self.robot_body_id: int = -1
        self.left_wheel_body_id: int = -1
        self.right_wheel_body_id: int = -1
        self.left_wheel_joint_qposadr: int = -1
        self.right_wheel_joint_qposadr: int = -1
        self.left_wheel_dofadr: int = -1
        self.right_wheel_dofadr: int = -1
        
        # Sensor addresses
        self.imu_accel_adr: int = -1
        self.imu_gyro_adr: int = -1
        
        # Object mapping: scene_object_id -> body_name & geom_name
        self.object_body_ids: Dict[str, int] = {}

        # Snapshots
        self.snapshots: Dict[str, SimulationSnapshot] = {}

        # Real-time factor metrics
        self._last_step_wall_time: float = time.perf_counter()
        self.real_time_factor: float = 1.0

        # Odometry tracking (dead-reckoning)
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_yaw: float = 0.0
        self._last_lw_pos: Optional[float] = None
        self._last_rw_pos: Optional[float] = None
        self._distance_traveled: float = 0.0

        # Offscreen camera renderer
        self._renderer: Optional[mujoco.Renderer] = None

        # Build initial MuJoCo model
        self.rebuild_model()

    def log(self, level: str, message: str):
        msg = LogMessage(
            timestamp=time.time(),
            level=level,  # type: ignore
            message=message,
            source="MuJoCoEngine"
        )
        if level == "error":
            self.errors.append(message)
            if len(self.errors) > 20:
                self.errors.pop(0)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def _generate_generic_robot_mjcf(self, robot_model: GenericRobotModel) -> Tuple[str, str]:
        """Constructs MJCF XML snippet for a generic kinematic tree robot."""
        links_by_name = {link.name: link for link in robot_model.links}
        joints_by_parent: Dict[str, List[JointModel]] = {}
        for j in robot_model.joints:
            joints_by_parent.setdefault(j.parent_link, []).append(j)

        actuators_xml: List[str] = []

        def build_link_body(link_name: str, joint: Optional[JointModel] = None) -> str:
            link = links_by_name.get(link_name)
            if not link:
                return ""

            if joint is not None:
                pos_str = f"{joint.origin_pos[0]} {joint.origin_pos[1]} {joint.origin_pos[2]}"
                quat_str = euler_deg_to_quat_str(joint.origin_rpy)
            else:
                pos_str = f"{robot_model.base_position[0]} {robot_model.base_position[1]} {robot_model.base_position[2]}"
                quat_str = euler_deg_to_quat_str(robot_model.base_rpy)

            body_lines = [f'<body name="{link.name}" pos="{pos_str}" quat="{quat_str}">']

            # If this is the root link and not fixed base, add freejoint
            if joint is None and not robot_model.is_fixed_base:
                body_lines.append(f'<freejoint name="{link.name}_freejoint"/>')

            # If a joint connects to this link
            if joint is not None and joint.type != "fixed":
                mj_jtype = "hinge" if joint.type in ("revolute", "continuous") else "slide"
                axis_str = f"{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"
                jnt_attrs = [
                    f'name="{joint.name}"',
                    f'type="{mj_jtype}"',
                    f'axis="{axis_str}"',
                    f'damping="{joint.damping}"',
                    f'armature="{joint.armature}"'
                ]
                if joint.type in ("revolute", "prismatic") and joint.limit:
                    jnt_attrs.append(f'range="{joint.limit.lower} {joint.limit.upper}"')
                    jnt_attrs.append('limited="true"')

                body_lines.append(f'<joint {" ".join(jnt_attrs)}/>')

                # Actuator
                act_name = f"act_{joint.name}"
                mode = joint.actuator_mode or "position"
                ctrl_str = ""
                force_str = ""
                if joint.limit:
                    if mode == "position" and joint.type in ("revolute", "prismatic"):
                        ctrl_str = f' ctrlrange="{joint.limit.lower} {joint.limit.upper}"'
                    elif mode == "velocity" and joint.limit.velocity > 0:
                        ctrl_str = f' ctrlrange="{-joint.limit.velocity} {joint.limit.velocity}"'
                    elif mode == "torque" and joint.limit.effort > 0:
                        ctrl_str = f' ctrlrange="{-joint.limit.effort} {joint.limit.effort}"'
                    if joint.limit.effort > 0 and mode != "torque":
                        force_str = f' forcerange="{-joint.limit.effort} {joint.limit.effort}"'

                if mode == "position":
                    actuators_xml.append(f'<position name="{act_name}" joint="{joint.name}" kp="{joint.kp}" kv="{joint.kv}"{ctrl_str}{force_str}/>')
                elif mode == "velocity":
                    actuators_xml.append(f'<velocity name="{act_name}" joint="{joint.name}" kv="{joint.kv}"{ctrl_str}{force_str}/>')
                elif mode == "torque":
                    actuators_xml.append(f'<motor name="{act_name}" joint="{joint.name}"{ctrl_str}/>')

            # Inertial
            if link.inertial:
                ipos = f"{link.inertial.origin_pos[0]} {link.inertial.origin_pos[1]} {link.inertial.origin_pos[2]}"
                body_lines.append(f'<inertial pos="{ipos}" mass="{link.inertial.mass}" diaginertia="{link.inertial.ixx} {link.inertial.iyy} {link.inertial.izz}"/>')

            # Visual / Collision Geoms
            if link.collision and link.visual and (link.collision.type != link.visual.type or link.collision.size != link.visual.size or link.collision.origin_pos != link.visual.origin_pos):
                # Distinct collision and visual geometries
                c_pos = f"{link.collision.origin_pos[0]} {link.collision.origin_pos[1]} {link.collision.origin_pos[2]}"
                c_quat = euler_deg_to_quat_str(link.collision.origin_rpy)
                if link.collision.type == "box":
                    c_size = f"{link.collision.size[0]} {link.collision.size[1]} {link.collision.size[2]}"
                elif link.collision.type == "sphere":
                    c_size = f"{link.collision.size[0]}"
                else:
                    c_size = f"{link.collision.size[0]} {link.collision.size[1]}"
                body_lines.append(f'<geom name="col_{link.name}" type="{link.collision.type}" size="{c_size}" pos="{c_pos}" quat="{c_quat}"/>')

                v_pos = f"{link.visual.origin_pos[0]} {link.visual.origin_pos[1]} {link.visual.origin_pos[2]}"
                v_quat = euler_deg_to_quat_str(link.visual.origin_rpy)
                rgba_str = hex_to_rgba_str(link.visual.color)
                if link.visual.type == "box":
                    v_size = f"{link.visual.size[0]} {link.visual.size[1]} {link.visual.size[2]}"
                elif link.visual.type == "sphere":
                    v_size = f"{link.visual.size[0]}"
                else:
                    v_size = f"{link.visual.size[0]} {link.visual.size[1]}"
                body_lines.append(f'<geom name="vis_{link.name}" type="{link.visual.type}" size="{v_size}" pos="{v_pos}" quat="{v_quat}" rgba="{rgba_str}" contype="0" conaffinity="0"/>')
            else:
                geom_to_render = link.visual or link.collision
                if geom_to_render:
                    g_pos = f"{geom_to_render.origin_pos[0]} {geom_to_render.origin_pos[1]} {geom_to_render.origin_pos[2]}"
                    g_quat = euler_deg_to_quat_str(geom_to_render.origin_rpy)
                    rgba_str = hex_to_rgba_str(geom_to_render.color)
                    if geom_to_render.type == "box":
                        g_size = f"{geom_to_render.size[0]} {geom_to_render.size[1]} {geom_to_render.size[2]}"
                    elif geom_to_render.type == "sphere":
                        g_size = f"{geom_to_render.size[0]}"
                    elif geom_to_render.type == "cylinder":
                        g_size = f"{geom_to_render.size[0]} {geom_to_render.size[1]}"
                    else:
                        g_size = "0.1 0.1 0.1"

                    body_lines.append(f'<geom name="geom_{link.name}" type="{geom_to_render.type}" size="{g_size}" pos="{g_pos}" quat="{g_quat}" rgba="{rgba_str}"/>')

            # Recursively append child links
            for child_jnt in joints_by_parent.get(link_name, []):
                child_xml = build_link_body(child_jnt.child_link, child_jnt)
                if child_xml:
                    body_lines.append(child_xml)

            body_lines.append('</body>')
            return "\n".join(body_lines)

        root_xml = build_link_body(robot_model.root_link, None)
        return root_xml, "\n".join(actuators_xml)

    def generate_mjcf_xml(self) -> str:
        """Construct a validated MJCF XML string from the current Project definition."""
        robot = self.project.robot
        settings = self.project.settings
        sensors = self.project.sensors
        
        r_init_x, r_init_y, r_init_z = robot.initial_position
        init_quat_str = euler_deg_to_quat_str([0.0, 0.0, robot.initial_yaw])
        
        hx, hy, hz = robot.chassis_size
        wr = robot.wheel_radius
        ww = robot.wheel_width
        wb = robot.wheel_base
        cr = robot.caster_radius
        
        wheel_z_offset = -0.01
        wheel_y_offset = wb * 0.5
        
        caster_x = hx * 0.75
        caster_y = wb * 0.35
        caster_z_offset = wheel_z_offset - wr + cr
        
        grav_x, grav_y, grav_z = settings.gravity
        
        lidar_mount = sensors.lidar.mount_pos
        imu_mount = sensors.imu.mount_pos
        cam_mount = sensors.camera.mount_pos if (hasattr(sensors, 'camera') and sensors.camera) else [0.18, 0.0, 0.08]
        cam_fov = sensors.camera.fov_deg if (hasattr(sensors, 'camera') and sensors.camera) else 60.0

        xml_parts = [
            f"""<mujoco model="{self.project.name}">
  <option timestep="{settings.timestep}" integrator="implicitfast" gravity="{grav_x} {grav_y} {grav_z}"/>
  <default>
    <geom friction="1.0 0.005 0.0001" condim="3" solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>
  </default>
  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.20 0.23" rgb2="0.14 0.16 0.19"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="20 20" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light directional="true" pos="5 -5 8" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="ground" type="plane" size="50 50 0.1" material="grid_mat"/>
"""
        ]

        # Configurable World Boundaries
        if settings.enable_boundaries:
            b = settings.world_boundaries
            xmin, xmax, ymin, ymax, zmin, zmax = b
            b_hx = (xmax - xmin) * 0.5
            b_hy = (ymax - ymin) * 0.5
            b_hz = (zmax - zmin) * 0.5
            mid_x = (xmin + xmax) * 0.5
            mid_y = (ymin + ymax) * 0.5
            mid_z = (zmin + zmax) * 0.5
            wall_thick = 0.05

            xml_parts.append(f"""
    <!-- World Containment Boundaries -->
    <body name="boundary_walls" pos="0 0 0">
      <geom name="wall_north" type="box" size="{b_hx} {wall_thick} {b_hz}" pos="{mid_x} {ymax} {mid_z}" rgba="0.3 0.3 0.4 0.15" friction="1.0 0.005 0.0001"/>
      <geom name="wall_south" type="box" size="{b_hx} {wall_thick} {b_hz}" pos="{mid_x} {ymin} {mid_z}" rgba="0.3 0.3 0.4 0.15" friction="1.0 0.005 0.0001"/>
      <geom name="wall_east" type="box" size="{wall_thick} {b_hy} {b_hz}" pos="{xmax} {mid_y} {mid_z}" rgba="0.3 0.3 0.4 0.15" friction="1.0 0.005 0.0001"/>
      <geom name="wall_west" type="box" size="{wall_thick} {b_hy} {b_hz}" pos="{xmin} {mid_y} {mid_z}" rgba="0.3 0.3 0.4 0.15" friction="1.0 0.005 0.0001"/>
    </body>""")

        # Robot Definition (Differential-Drive)
        xml_parts.append(f"""
    <!-- Differential-Drive Robot -->
    <body name="robot" pos="{r_init_x} {r_init_y} {r_init_z}" quat="{init_quat_str}">
      <freejoint name="robot_freejoint"/>
      <!-- Main Chassis -->
      <geom name="robot_chassis" type="box" size="{hx} {hy} {hz}" mass="{robot.chassis_mass}" rgba="0.22 0.55 0.88 1.0"/>
      <!-- Direction Indicator Nose -->
      <geom name="robot_nose" type="box" size="{hx * 0.25} {hy * 0.3} {hz * 0.4}" pos="{hx * 0.85} 0 {hz * 0.4}" mass="0.05" rgba="1.0 0.4 0.1 1.0"/>
      <!-- Sensor Mounting Sites -->
      <site name="robot_lidar_site" pos="{lidar_mount[0]} {lidar_mount[1]} {lidar_mount[2]}"/>
      <site name="robot_imu_site" pos="{imu_mount[0]} {imu_mount[1]} {imu_mount[2]}"/>
      <site name="robot_camera_site" pos="{cam_mount[0]} {cam_mount[1]} {cam_mount[2]}"/>
      <camera name="robot_camera" pos="{cam_mount[0]} {cam_mount[1]} {cam_mount[2]}" fovy="{cam_fov}" xyaxes="0 -1 0 0 0 1"/>
      
      <!-- 4 Corner Stabilizer Casters -->
      <geom name="robot_caster_fl" type="sphere" size="{cr}" pos="{caster_x} {caster_y} {caster_z_offset}" friction="0.001 0.001 0.001" mass="0.05" rgba="0.2 0.2 0.2 1.0"/>
      <geom name="robot_caster_fr" type="sphere" size="{cr}" pos="{caster_x} {-caster_y} {caster_z_offset}" friction="0.001 0.001 0.001" mass="0.05" rgba="0.2 0.2 0.2 1.0"/>
      <geom name="robot_caster_rl" type="sphere" size="{cr}" pos="{-caster_x} {caster_y} {caster_z_offset}" friction="0.001 0.001 0.001" mass="0.05" rgba="0.2 0.2 0.2 1.0"/>
      <geom name="robot_caster_rr" type="sphere" size="{cr}" pos="{-caster_x} {-caster_y} {caster_z_offset}" friction="0.001 0.001 0.001" mass="0.05" rgba="0.2 0.2 0.2 1.0"/>

      <!-- Left Wheel -->
      <body name="left_wheel" pos="0 {wheel_y_offset} {wheel_z_offset}">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom name="left_wheel_geom" type="cylinder" size="{wr} {ww * 0.5}" quat="0.7071068 0.7071068 0 0" mass="{robot.wheel_mass}" friction="1.5 0.01 0.001" rgba="0.15 0.15 0.18 1.0"/>
      </body>

      <!-- Right Wheel -->
      <body name="right_wheel" pos="0 {-wheel_y_offset} {wheel_z_offset}">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom name="right_wheel_geom" type="cylinder" size="{wr} {ww * 0.5}" quat="0.7071068 0.7071068 0 0" mass="{robot.wheel_mass}" friction="1.5 0.01 0.001" rgba="0.15 0.15 0.18 1.0"/>
      </body>
    </body>
""")

        # Generic Kinematic Robot Model (if present)
        generic_actuators = ""
        if self.project.generic_robot is not None and len(self.project.generic_robot.links) > 0:
            gen_body_xml, gen_acts_xml = self._generate_generic_robot_mjcf(self.project.generic_robot)
            xml_parts.append(gen_body_xml)
            generic_actuators = gen_acts_xml

        # Add Scene Objects (Obstacles) with physical friction and body mode
        for obj in self.project.objects:
            pos_str = f"{obj.position[0]} {obj.position[1]} {obj.position[2]}"
            rot_quat_str = euler_deg_to_quat_str(obj.rotation)
            rgba_str = hex_to_rgba_str(obj.color)
            fric_str = f"{obj.friction[0]} {obj.friction[1]} {obj.friction[2]}"
            
            if obj.type == "box":
                size_str = f"{obj.size[0]} {obj.size[1]} {obj.size[2]}"
            elif obj.type == "sphere":
                size_str = f"{obj.size[0]}"
            elif obj.type == "cylinder":
                size_str = f"{obj.size[0]} {obj.size[1]}"
            else:
                size_str = "0.3 0.3 0.3"

            body_name = f"obj_{obj.id}"
            geom_name = f"geom_{obj.id}"
            
            if obj.is_static:
                # Static body (fixed in space)
                xml_parts.append(f"""
    <body name="{body_name}" pos="{pos_str}" quat="{rot_quat_str}">
      <geom name="{geom_name}" type="{obj.type}" size="{size_str}" mass="{obj.mass}" friction="{fric_str}" rgba="{rgba_str}"/>
    </body>""")
            else:
                # Dynamic body with freejoint
                xml_parts.append(f"""
    <body name="{body_name}" pos="{pos_str}" quat="{rot_quat_str}">
      <freejoint name="joint_{obj.id}"/>
      <geom name="{geom_name}" type="{obj.type}" size="{size_str}" mass="{obj.mass}" friction="{fric_str}" rgba="{rgba_str}"/>
    </body>""")

        xml_parts.append("""
  </worldbody>
  <sensor>
    <accelerometer name="imu_accel" site="robot_imu_site"/>
    <gyro name="imu_gyro" site="robot_imu_site"/>
  </sensor>
  <actuator>
    <velocity name="left_motor" joint="left_wheel_joint" kv="40" ctrlrange="-50 50" forcerange="-15 15"/>
    <velocity name="right_motor" joint="right_wheel_joint" kv="40" ctrlrange="-50 50" forcerange="-15 15"/>
""")
        if generic_actuators:
            xml_parts.append(generic_actuators)

        xml_parts.append("  </actuator>\n</mujoco>\n")
        return "".join(xml_parts)

    def rebuild_model(self, preserve_state: bool = False):
        """Compiles the MuJoCo model and initializes MjData."""
        prev_data = None
        prev_model = None
        prev_status = self.status
        prev_sim_time = self.sim_time
        prev_step_count = self.step_count
        prev_robot_qpos = None
        prev_robot_qvel = None
        prev_obj_states = {}

        if preserve_state and self.model is not None and self.data is not None:
            try:
                prev_data = self.data
                prev_model = self.model
                r_qpos_adr = getattr(self, "robot_qpos_adr", 0)
                r_qvel_adr = getattr(self, "robot_qvel_adr", 0)
                prev_robot_qpos = np.array(prev_data.qpos[r_qpos_adr:r_qpos_adr + 7], copy=True)
                prev_robot_qvel = np.array(prev_data.qvel[r_qvel_adr:r_qvel_adr + 6], copy=True)

                for obj in self.project.objects:
                    if not obj.is_static:
                        jnt = mujoco.mj_name2id(prev_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{obj.id}")
                        if jnt >= 0:
                            j_qpos = prev_model.jnt_qposadr[jnt]
                            j_dof = prev_model.jnt_dofadr[jnt]
                            prev_obj_states[obj.id] = (
                                np.array(prev_data.qpos[j_qpos:j_qpos + 7], copy=True),
                                np.array(prev_data.qvel[j_dof:j_dof + 6], copy=True)
                            )
            except Exception as e:
                logger.warning("Could not snapshot previous state for model rebuild: %s", e)

        try:
            # Ensure generic robot is never placed right on top of mobile robot chassis
            if self.project.generic_robot and hasattr(self.project.generic_robot, "base_position"):
                bx, by, bz = self.project.generic_robot.base_position
                if abs(bx) < 0.4 and abs(by) < 0.4:
                    self.project.generic_robot.base_position = [2.5, 2.5, 0.05]

            xml_str = self.generate_mjcf_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)

            # Resolve Actuators
            self.left_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
            self.right_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
            
            # Resolve generic joint actuators
            self.joint_actuator_ids = {}
            if self.project.generic_robot is not None:
                for jnt in self.project.generic_robot.joints:
                    act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jnt.name}")
                    if act_id >= 0:
                        self.joint_actuator_ids[jnt.name] = act_id

            # Resolve Bodies
            self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            self.left_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
            self.right_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
            
            # Resolve Robot Joint qpos/qvel addresses
            robot_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_freejoint")
            self.robot_qpos_adr = self.model.jnt_qposadr[robot_jnt] if robot_jnt >= 0 else 0
            self.robot_qvel_adr = self.model.jnt_dofadr[robot_jnt] if robot_jnt >= 0 else 0

            # Resolve wheel joint DOF addresses and qpos addresses
            left_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
            right_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
            self.left_wheel_dofadr = self.model.jnt_dofadr[left_jnt] if left_jnt >= 0 else -1
            self.right_wheel_dofadr = self.model.jnt_dofadr[right_jnt] if right_jnt >= 0 else -1
            self.left_wheel_joint_qposadr = self.model.jnt_qposadr[left_jnt] if left_jnt >= 0 else -1
            self.right_wheel_joint_qposadr = self.model.jnt_qposadr[right_jnt] if right_jnt >= 0 else -1
            
            # Resolve IMU sensors
            accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
            gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
            self.imu_accel_adr = self.model.sensor_adr[accel_id] if accel_id >= 0 else -1
            self.imu_gyro_adr = self.model.sensor_adr[gyro_id] if gyro_id >= 0 else -1

            # Map object bodies
            self.object_body_ids = {}
            for obj in self.project.objects:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj.id}")
                if bid >= 0:
                    self.object_body_ids[obj.id] = bid

            if preserve_state and prev_robot_qpos is not None:
                self.data.qpos[self.robot_qpos_adr:self.robot_qpos_adr + 7] = prev_robot_qpos
                self.data.qvel[self.robot_qvel_adr:self.robot_qvel_adr + 6] = prev_robot_qvel

                for obj_id, (qpos, qvel) in prev_obj_states.items():
                    jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{obj_id}")
                    if jnt >= 0:
                        j_qpos = self.model.jnt_qposadr[jnt]
                        j_dof = self.model.jnt_dofadr[jnt]
                        self.data.qpos[j_qpos:j_qpos + 7] = qpos
                        self.data.qvel[j_dof:j_dof + 6] = qvel

                self.data.time = prev_sim_time
                self.sim_time = prev_sim_time
                self.step_count = prev_step_count
                self.status = prev_status
            else:
                self.sim_time = 0.0
                self.step_count = 0
                self.status = "stopped"

            mujoco.mj_forward(self.model, self.data)
            self.log("info", f"MuJoCo model successfully built with {len(self.project.objects)} scene objects.")
        except Exception as e:
            err_msg = f"Failed to build MuJoCo model: {str(e)}"
            self.status = "error"
            self.log("error", err_msg)
            logger.exception(err_msg)

    def set_teleop(self, linear: float, angular: float):
        """Sets the teleoperation command with validation against robot velocity limits."""
        v_max = self.project.robot.max_linear_velocity
        w_max = self.project.robot.max_angular_velocity
        
        cmd_linear = max(-v_max, min(v_max, linear * v_max if abs(linear) <= 1.0 else linear))
        cmd_angular = max(-w_max, min(w_max, angular * w_max if abs(angular) <= 1.0 else angular))
        
        self.teleop_cmd = TeleopCommand(linear=cmd_linear, angular=cmd_angular)
        self.last_teleop_time = time.time()

        if (abs(cmd_linear) > 0.01 or abs(cmd_angular) > 0.01) and self.status != "running":
            self.play()

    def set_joint_target(self, joint_name: str, target: float):
        """Sets target actuation value (position/velocity/torque) for a generic robot joint."""
        self.joint_targets[joint_name] = target

    def set_joint_targets(self, targets: Dict[str, float]):
        self.joint_targets.update(targets)

    def apply_control(self):
        """Applies differential drive and generic joint actuator controls to MuJoCo."""
        if not self.model or not self.data:
            return
            
        # Teleop decay: if no key input in last 0.35s, stop motors
        if time.time() - self.last_teleop_time > 0.35:
            self.teleop_cmd.linear = 0.0
            self.teleop_cmd.angular = 0.0

        v = self.teleop_cmd.linear
        omega = self.teleop_cmd.angular
        
        wb = self.project.robot.wheel_base
        wr = self.project.robot.wheel_radius
        
        v_left = v - omega * (wb * 0.5)
        v_right = v + omega * (wb * 0.5)
        
        omega_left = v_left / wr
        omega_right = v_right / wr
        
        if self.left_motor_id >= 0:
            self.data.ctrl[self.left_motor_id] = omega_left
        if self.right_motor_id >= 0:
            self.data.ctrl[self.right_motor_id] = omega_right

        # Apply generic joint actuator controls with joint limit enforcement
        joint_map = {}
        if self.project.generic_robot and self.project.generic_robot.joints:
            joint_map = {j.name: j for j in self.project.generic_robot.joints}

        for jname, target in self.joint_targets.items():
            act_id = self.joint_actuator_ids.get(jname, -1)
            if act_id >= 0 and act_id < len(self.data.ctrl):
                clamped_target = float(target)
                j_obj = joint_map.get(jname)
                if j_obj:
                    mode = j_obj.actuator_mode or "position"
                    if mode == "position" and j_obj.limit and j_obj.type in ("revolute", "prismatic"):
                        clamped_target = max(j_obj.limit.lower, min(j_obj.limit.upper, clamped_target))
                    elif mode == "velocity" and j_obj.limit and j_obj.limit.velocity > 0:
                        v_max = j_obj.limit.velocity
                        clamped_target = max(-v_max, min(v_max, clamped_target))
                    elif mode == "torque" and j_obj.limit and j_obj.limit.effort > 0:
                        eff_max = j_obj.limit.effort
                        clamped_target = max(-eff_max, min(eff_max, clamped_target))
                self.data.ctrl[act_id] = clamped_target

    def step(self):
        """Advances physics by one simulation timestep and computes rolling real-time factor."""
        if not self.model or not self.data:
            return
        
        try:
            self.apply_control()
            mujoco.mj_step(self.model, self.data)
            
            # Compute rolling real-time factor
            now_wall = time.perf_counter()
            dt_wall = now_wall - self._last_step_wall_time
            self._last_step_wall_time = now_wall
            if dt_wall > 1e-6:
                instant_rtf = (self.project.settings.timestep * self.project.settings.speed_multiplier) / dt_wall
                self.real_time_factor = float(0.95 * self.real_time_factor + 0.05 * instant_rtf)

            if np.isnan(self.data.qpos[0]) or np.isnan(self.data.qvel[0]):
                self.reset()
                self.log("warning", "Physics numerical divergence detected; state restored to initial conditions.")
            else:
                self.sim_time = float(self.data.time)
                self.step_count += 1
        except Exception as e:
            self.status = "error"
            self.log("error", f"Simulation step failure: {str(e)}")

    def play(self):
        self.status = "running"
        self._last_step_wall_time = time.perf_counter()
        self.log("info", "Simulation started.")

    def pause(self):
        self.status = "paused"
        self.log("info", "Simulation paused.")

    def reset(self):
        """Resets physics state and robot to initial conditions."""
        self.teleop_cmd.linear = 0.0
        self.teleop_cmd.angular = 0.0
        self.joint_targets.clear()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._last_lw_pos = None
        self._last_rw_pos = None
        self._distance_traveled = 0.0
        self.real_time_factor = 1.0
        self.rebuild_model()
        self.status = "stopped"
        self.log("info", "Simulation reset to initial conditions.")

    def save_snapshot(self, snapshot_id: Optional[str] = None) -> SimulationSnapshot:
        """Captures an authoritative simulation state snapshot for reproducible rollbacks."""
        if not self.model or not self.data:
            raise RuntimeError("Simulation model not initialized")
        
        sid = snapshot_id or str(uuid.uuid4())[:8]
        act_vals = [float(x) for x in self.data.act] if self.model.na > 0 else []
        snapshot = SimulationSnapshot(
            snapshot_id=sid,
            timestamp=time.time(),
            sim_time=self.sim_time,
            step_count=self.step_count,
            qpos=[float(x) for x in self.data.qpos],
            qvel=[float(x) for x in self.data.qvel],
            act=act_vals,
            joint_targets=dict(self.joint_targets),
            teleop_linear=self.teleop_cmd.linear,
            teleop_angular=self.teleop_cmd.angular
        )
        self.snapshots[sid] = snapshot
        self.log("info", f"Saved simulation snapshot '{sid}' at t={self.sim_time:.3f}s")
        return snapshot

    def restore_snapshot(self, snapshot: Union[str, SimulationSnapshot]) -> bool:
        """Restores authoritative physics state from a previously saved snapshot."""
        if not self.model or not self.data:
            return False

        if isinstance(snapshot, str):
            if snapshot not in self.snapshots:
                self.log("error", f"Snapshot '{snapshot}' not found.")
                return False
            snap_obj = self.snapshots[snapshot]
        else:
            snap_obj = snapshot

        try:
            if len(snap_obj.qpos) == len(self.data.qpos):
                self.data.qpos[:] = snap_obj.qpos
            if len(snap_obj.qvel) == len(self.data.qvel):
                self.data.qvel[:] = snap_obj.qvel
            if self.model.na > 0 and len(snap_obj.act) == len(self.data.act):
                self.data.act[:] = snap_obj.act
            
            self.data.time = snap_obj.sim_time
            self.sim_time = snap_obj.sim_time
            self.step_count = snap_obj.step_count
            self.joint_targets = dict(snap_obj.joint_targets)
            self.teleop_cmd.linear = snap_obj.teleop_linear
            self.teleop_cmd.angular = snap_obj.teleop_angular
            
            mujoco.mj_forward(self.model, self.data)
            self.log("info", f"Restored simulation snapshot '{snap_obj.snapshot_id}' to t={self.sim_time:.3f}s")
            return True
        except Exception as e:
            self.log("error", f"Failed to restore snapshot: {e}")
            return False

    def list_snapshots(self) -> List[SimulationSnapshot]:
        return list(self.snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> bool:
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            return True
        return False

    def capture_camera_frame(
        self,
        width: int = 320,
        height: int = 240,
        camera_name: Optional[str] = None,
        render_depth: bool = False,
        render_segmentation: bool = False
    ) -> Dict[str, Any]:
        """Renders an offscreen camera frame using MuJoCo Renderer with optional depth and true geom-ID semantic segmentation."""
        if not self.model or not self.data:
            raise RuntimeError("Simulation model not initialized")

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            cam_id = -1
            if camera_name:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            
            # Ensure kinematics are fully evaluated prior to rendering
            mujoco.mj_forward(self.model, self.data)

            # 1. RGB Image Render
            if cam_id >= 0:
                renderer.update_scene(self.data, camera=cam_id)
            else:
                renderer.update_scene(self.data)

            rgb_img = renderer.render()
            png_bytes = rgb_to_png(rgb_img)
            b64_rgb = base64.b64encode(png_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64_rgb}"

            # 2. Depth Buffer Render
            depth_data = None
            if render_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(self.data, camera=cam_id if cam_id >= 0 else -1)
                depth_img = renderer.render()
                depth_data = {
                    "min_m": float(np.min(depth_img)),
                    "max_m": float(np.max(depth_img)),
                    "mean_m": float(np.mean(depth_img)),
                    "center_distance_m": float(depth_img[height // 2, width // 2])
                }

            # 3. Genuine MuJoCo Pixel-Level Geom-ID Semantic Segmentation
            seg_data = None
            if render_segmentation:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(self.data, camera=cam_id if cam_id >= 0 else -1)
                seg_raw = renderer.render()  # Shape: (height, width, 2), dtype int32
                geom_ids = seg_raw[:, :, 0]  # Channel 0 contains the exact geom ID (-1 for background)

                palette = [
                    (0, 0, 0),         # -1 Background (Black)
                    (230, 76, 60),     # Red
                    (52, 152, 219),    # Blue
                    (46, 204, 113),    # Green
                    (241, 196, 15),    # Yellow
                    (155, 89, 182),    # Purple
                    (26, 188, 156),    # Teal
                    (230, 126, 34),    # Orange
                    (52, 73, 94),      # Dark Slate
                    (236, 240, 241),   # Off-White
                    (149, 165, 166),   # Gray
                    (255, 107, 129),   # Coral
                    (112, 161, 255),   # Sky
                    (123, 237, 159),   # Mint
                    (255, 165, 2),     # Amber
                    (164, 176, 190)    # Slate
                ]

                seg_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                label_mapping: Dict[str, Any] = {}
                total_pixels = height * width

                unique_gids = np.unique(geom_ids)
                for gid in unique_gids:
                    gid_int = int(gid)
                    pixel_mask = (geom_ids == gid)
                    count = int(np.sum(pixel_mask))
                    pct = round((count / total_pixels) * 100.0, 2)

                    if gid_int == -1:
                        color = palette[0]
                        label_name = "background"
                        body_name = "none"
                        obj_id = None
                    else:
                        color = palette[(gid_int % (len(palette) - 1)) + 1]
                        label_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid_int) or f"geom_{gid_int}"
                        body_id = int(self.model.geom_bodyid[gid_int])
                        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
                        
                        # Match to project SceneObject if applicable
                        obj_id = None
                        for so in self.project.objects:
                            if so.id in label_name or so.name == label_name or f"obj_{so.id}" == body_name:
                                obj_id = so.id
                                break

                    seg_rgb[pixel_mask] = color
                    label_mapping[str(gid_int)] = {
                        "geom_id": gid_int,
                        "geom_name": label_name,
                        "body_name": body_name,
                        "scene_object_id": obj_id,
                        "pixel_count": count,
                        "pixel_percentage": pct,
                        "color_rgb": list(color)
                    }

                seg_png = rgb_to_png(seg_rgb)
                b64_seg = base64.b64encode(seg_png).decode("ascii")
                seg_data = {
                    "image_data_url": f"data:image/png;base64,{b64_seg}",
                    "labels": label_mapping,
                    "total_pixels": total_pixels,
                    "unique_geometries": len(label_mapping)
                }

            return {
                "width": width,
                "height": height,
                "camera_name": camera_name or "free_camera",
                "sim_time": self.sim_time,
                "image_data_url": data_url,
                "depth_info": depth_data,
                "segmentation_info": seg_data
            }
        finally:
            renderer.close()

    def get_state(self) -> SimulationState:
        """Extracts current physics state, robot pose, velocities, contacts, LiDAR, and IMU data."""
        if not self.model or not self.data:
            return SimulationState(status=self.status, errors=self.errors)

        qadr = getattr(self, "robot_qpos_adr", 0)
        vadr = getattr(self, "robot_qvel_adr", 0)
        r_pos = [float(x) for x in self.data.qpos[qadr:qadr + 3]]
        r_quat = [float(x) for x in self.data.qpos[qadr + 3:qadr + 7]]
        r_lvel = [float(x) for x in self.data.qvel[vadr:vadr + 3]]
        r_avel = [float(x) for x in self.data.qvel[vadr + 3:vadr + 6]]
        
        l_speed = float(math.sqrt(r_lvel[0]**2 + r_lvel[1]**2 + r_lvel[2]**2))
        a_speed = float(math.sqrt(r_avel[0]**2 + r_avel[1]**2 + r_avel[2]**2))
        
        fwd_vec = quat_to_forward(r_quat)
        
        left_w_vel = float(self.data.qvel[self.left_wheel_dofadr]) if getattr(self, "left_wheel_dofadr", -1) >= 0 else 0.0
        right_w_vel = float(self.data.qvel[self.right_wheel_dofadr]) if getattr(self, "right_wheel_dofadr", -1) >= 0 else 0.0

        robot_state = RobotState(
            position=r_pos,
            quaternion=r_quat,
            linear_velocity=r_lvel,
            angular_velocity=r_avel,
            linear_speed=l_speed,
            angular_speed=a_speed,
            left_wheel_velocity=left_w_vel,
            right_wheel_velocity=right_w_vel,
            forward_vector=fwd_vec
        )

        # Scene object states
        objects_state: Dict[str, ObjectPhysicsState] = {}
        for obj in self.project.objects:
            bid = self.object_body_ids.get(obj.id, -1)
            if bid >= 0:
                pos = [float(x) for x in self.data.xpos[bid]]
                quat = [float(x) for x in self.data.xquat[bid]]
                objects_state[obj.id] = ObjectPhysicsState(
                    id=obj.id,
                    position=pos,
                    quaternion=quat
                )

        # Contacts / Collisions with MuJoCo contact normal & tangent forces and penetration
        contacts: List[ContactInfo] = []
        ncon = min(int(self.data.ncon), 50)
        c_force_buf = np.zeros(6, dtype=np.float64)
        for i in range(ncon):
            con = self.data.contact[i]
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
            try:
                mujoco.mj_contactForce(self.model, self.data, i, c_force_buf)
                fn = float(abs(c_force_buf[0]))
                ft = float(math.sqrt(c_force_buf[1]**2 + c_force_buf[2]**2))
            except Exception:
                fn = 0.0
                ft = 0.0
            penetration = float(max(0.0, -float(con.dist)))
            contacts.append(ContactInfo(
                geom1=g1_name,
                geom2=g2_name,
                position=[float(x) for x in con.pos],
                distance=float(con.dist),
                force_normal=fn,
                force_tangent=ft,
                penetration=penetration
            ))

        # Ground-Truth LiDAR Simulation using real MuJoCo mj_ray with noise & dropout
        lidar_readings: List[float] = []
        lidar_angles: List[float] = []
        l_cfg = self.project.sensors.lidar
        if l_cfg.enabled and self.robot_body_id >= 0:
            num_beams = l_cfg.num_beams
            fov_rad = math.radians(l_cfg.fov_deg)
            half_fov = fov_rad * 0.5
            angles = np.linspace(-half_fov, half_fov, num_beams)
            
            yaw = quat_to_yaw(r_quat)
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            m_pos = l_cfg.mount_pos
            ray_origin = np.array([
                r_pos[0] + m_pos[0] * cos_y - m_pos[1] * sin_y,
                r_pos[1] + m_pos[0] * sin_y + m_pos[1] * cos_y,
                r_pos[2] + m_pos[2]
            ], dtype=np.float64)

            geomid_buf = np.zeros(1, dtype=np.int32)
            for a in angles:
                beam_yaw = yaw + a
                vec = np.array([math.cos(beam_yaw), math.sin(beam_yaw), 0.0], dtype=np.float64)
                dist = mujoco.mj_ray(self.model, self.data, ray_origin, vec, None, 1, self.robot_body_id, geomid_buf)
                if dist >= l_cfg.min_range and dist <= l_cfg.max_range:
                    measured = float(dist)
                elif dist > l_cfg.max_range or dist < 0:
                    measured = float(l_cfg.max_range)
                else:
                    measured = float(l_cfg.min_range)

                # Apply sensor noise if configured
                if l_cfg.noise_std > 0.0:
                    measured = float(np.clip(measured + np.random.normal(0.0, l_cfg.noise_std), l_cfg.min_range, l_cfg.max_range))

                # Apply sensor dropout if configured
                if l_cfg.dropout_prob > 0.0 and np.random.random() < l_cfg.dropout_prob:
                    measured = float(l_cfg.max_range)

                lidar_readings.append(measured)
                lidar_angles.append(float(math.degrees(a)))

        # Ground-Truth IMU Data with bias and noise modeling
        imu_acc = None
        imu_gyr = None
        imu_cfg = getattr(self.project.sensors, "imu", None)
        if imu_cfg and imu_cfg.enabled:
            if self.imu_accel_adr >= 0:
                raw_acc = [float(x) for x in self.data.sensordata[self.imu_accel_adr:self.imu_accel_adr + 3]]
                imu_acc = [
                    raw_acc[i] + (imu_cfg.accel_bias[i] if len(imu_cfg.accel_bias) > i else 0.0) +
                    (float(np.random.normal(0.0, imu_cfg.accel_noise_std)) if imu_cfg.accel_noise_std > 0.0 else 0.0)
                    for i in range(3)
                ]
            if self.imu_gyro_adr >= 0:
                raw_gyr = [float(x) for x in self.data.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3]]
                imu_gyr = [
                    raw_gyr[i] + (imu_cfg.gyro_bias[i] if len(imu_cfg.gyro_bias) > i else 0.0) +
                    (float(np.random.normal(0.0, imu_cfg.gyro_noise_std)) if imu_cfg.gyro_noise_std > 0.0 else 0.0)
                    for i in range(3)
                ]

        # Dead-reckoning odometry computation from wheel encoder rotations
        odometry_state = None
        if self.left_wheel_joint_qposadr >= 0 and self.right_wheel_joint_qposadr >= 0:
            lw_cur = float(self.data.qpos[self.left_wheel_joint_qposadr])
            rw_cur = float(self.data.qpos[self.right_wheel_joint_qposadr])
            if self._last_lw_pos is not None and self._last_rw_pos is not None:
                dl = (lw_cur - self._last_lw_pos) * self.project.robot.wheel_radius
                dr = (rw_cur - self._last_rw_pos) * self.project.robot.wheel_radius
                dc = (dl + dr) * 0.5
                dtheta = (dr - dl) / max(0.01, self.project.robot.wheel_base)
                self._odom_yaw += dtheta
                self._odom_x += dc * math.cos(self._odom_yaw)
                self._odom_y += dc * math.sin(self._odom_yaw)
                self._distance_traveled += abs(dc)
            self._last_lw_pos = lw_cur
            self._last_rw_pos = rw_cur
            odometry_state = OdometryState(
                position=[self._odom_x, self._odom_y, self.project.robot.wheel_radius],
                yaw=self._odom_yaw,
                linear_velocity=r_lvel,
                angular_velocity=r_avel,
                distance_traveled=self._distance_traveled
            )

        # Joint telemetry for all model joints
        joint_positions: Dict[str, float] = {}
        joint_velocities: Dict[str, float] = {}
        joint_torques: Dict[str, float] = {}
        for j_id in range(self.model.njnt):
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if not j_name or "freejoint" in j_name:
                continue
            q_adr = self.model.jnt_qposadr[j_id]
            d_adr = self.model.jnt_dofadr[j_id]
            joint_positions[j_name] = float(self.data.qpos[q_adr])
            joint_velocities[j_name] = float(self.data.qvel[d_adr])
            joint_torques[j_name] = float(self.data.qfrc_actuator[d_adr])

        # Encoders
        encoder_ticks: Dict[str, int] = {}
        enc_cfg = self.project.sensors.encoders
        if enc_cfg.enabled:
            if self.left_wheel_joint_qposadr >= 0:
                lw_pos = float(self.data.qpos[self.left_wheel_joint_qposadr])
                encoder_ticks["left_wheel"] = int((lw_pos / (2.0 * math.pi)) * enc_cfg.ticks_per_rev)
            if self.right_wheel_joint_qposadr >= 0:
                rw_pos = float(self.data.qpos[self.right_wheel_joint_qposadr])
                encoder_ticks["right_wheel"] = int((rw_pos / (2.0 * math.pi)) * enc_cfg.ticks_per_rev)

        wall_time = time.time() - self.wall_time_start

        return SimulationState(
            status=self.status,  # type: ignore
            sim_time=self.sim_time,
            wall_time=wall_time,
            step_count=self.step_count,
            real_time_factor=self.real_time_factor,
            robot=robot_state,
            odometry=odometry_state,
            objects=objects_state,
            contacts=contacts,
            errors=self.errors,
            lidar_readings=lidar_readings,
            lidar_angles=lidar_angles,
            imu_linear_acceleration=imu_acc,
            imu_angular_velocity=imu_gyr,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            encoder_ticks=encoder_ticks
        )

# Alias for backwards and external compatibility
SimulationEngine = MuJoCoEngine
