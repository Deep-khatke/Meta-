"""Gymnasium-compliant reinforcement learning environment for RoboForge differential-drive MuJoCo robot."""

import math
from typing import Any, Dict, List, Optional, Tuple, Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from backend.models import Project, RobotConfig, SceneObject, SimulationSettings
from backend.simulation import SimulationEngine, quat_to_yaw


class RoboForgeGymEnv(gym.Env):
    """Reinforcement learning environment wrapping real MuJoCo differential-drive physics."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        project: Optional[Project] = None,
        target_position: Tuple[float, float, float] = (2.5, 0.5, 0.25),
        control_frequency_hz: float = 20.0,
        max_episode_steps: int = 300,
        distance_threshold: float = 0.25,
        randomize_training: bool = False,
        collision_penalty: float = 50.0,
        goal_reward: float = 100.0,
        seed: Optional[int] = None,
        custom_reward_code: Optional[str] = None,
    ):
        super().__init__()

        # Copy or build default project
        if project is not None:
            self.project = project.model_copy(deep=True)
        else:
            from backend.project_manager import get_default_project
            self.project = get_default_project()

        self.target_position = np.array(target_position, dtype=np.float32)
        self.default_target = np.array(target_position, dtype=np.float32)
        self.control_frequency_hz = control_frequency_hz
        self.max_episode_steps = max_episode_steps
        self.distance_threshold = distance_threshold
        self.randomize_training = randomize_training
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.custom_reward_code = custom_reward_code
        self._custom_reward_fn = None
        if self.custom_reward_code and self.custom_reward_code.strip():
            self._compile_custom_reward(self.custom_reward_code)

        # Compute substeps per RL action
        sim_dt = self.project.settings.timestep
        self.substeps = max(1, int(round((1.0 / control_frequency_hz) / sim_dt)))

        # Create MuJoCo simulation engine
        self.engine = SimulationEngine(self.project)
        self.robot_cfg: RobotConfig = self.project.robot

        # Action space: normalized [linear_vel, angular_vel] in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space (16 dimensions):
        # [0, 1]: Target vector in robot local frame (x_rel, y_rel) in meters [-10, 10]
        # [2]: Euclidean distance to target (m) [0, 15]
        # [3]: Heading error angle relative to robot forward [-pi, pi] / pi -> [-1, 1]
        # [4]: Current robot linear speed / max_linear_vel [-1, 1]
        # [5]: Current robot angular speed / max_angular_vel [-1, 1]
        # [6..13]: 8 rangefinder / synthetic LiDAR beams at [-180, -135, -90, -45, 0, 45, 90, 135] deg [0, 1]
        # [14, 15]: Global robot coordinates x / 5.0, y / 5.0 [-2, 2]
        obs_low = np.full(16, -10.0, dtype=np.float32)
        obs_high = np.full(16, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(16,), dtype=np.float32)

        self.current_step = 0
        self.prev_distance = 0.0
        self.trajectory: List[Tuple[float, float, float]] = []

        if seed is not None:
            self.reset(seed=seed)

    def _compile_custom_reward(self, code_str: str):
        try:
            scope: Dict[str, Any] = {"math": math, "np": np}
            exec(code_str, scope)
            if "compute_reward" in scope and callable(scope["compute_reward"]):
                self._custom_reward_fn = scope["compute_reward"]
            elif "calculate_reward" in scope and callable(scope["calculate_reward"]):
                self._custom_reward_fn = scope["calculate_reward"]
            else:
                logger.warning("Custom reward code provided but does not define 'compute_reward(state, action, info)'.")
        except Exception as e:
            logger.error("Failed to compile custom reward script: %s", e)

    @staticmethod
    def compute_lidar_distances(r_pos: Sequence[float], r_yaw: float, obstacles: Sequence[SceneObject]) -> np.ndarray:
        """Simulates 8 planar rangefinder beams checking distance to all active obstacles."""
        angles = np.array([
            -np.pi,
            -3.0 * np.pi / 4.0,
            -np.pi / 2.0,
            -np.pi / 4.0,
            0.0,
            np.pi / 4.0,
            np.pi / 2.0,
            3.0 * np.pi / 4.0,
        ], dtype=np.float32)

        max_range = 3.0
        distances = np.full(8, max_range, dtype=np.float32)
        rx, ry = float(r_pos[0]), float(r_pos[1])

        # Arena bounds perimeter check (X in [-4, 5], Y in [-3, 3])
        for i, rel_angle in enumerate(angles):
            beam_yaw = r_yaw + rel_angle
            dx = math.cos(beam_yaw)
            dy = math.sin(beam_yaw)

            dist_wall = max_range
            if dx > 1e-4:
                dist_wall = min(dist_wall, (4.5 - rx) / dx)
            elif dx < -1e-4:
                dist_wall = min(dist_wall, (-3.5 - rx) / dx)
            if dy > 1e-4:
                dist_wall = min(dist_wall, (3.0 - ry) / dy)
            elif dy < -1e-4:
                dist_wall = min(dist_wall, (-3.0 - ry) / dy)

            if dist_wall < distances[i]:
                distances[i] = dist_wall

            for obj in obstacles:
                ox, oy = float(obj.position[0]), float(obj.position[1])
                if obj.type == "box":
                    obs_radius = math.hypot(obj.size[0], obj.size[1])
                elif obj.type in ("sphere", "cylinder"):
                    obs_radius = float(obj.size[0])
                else:
                    obs_radius = 0.35

                vx = ox - rx
                vy = oy - ry
                proj = vx * dx + vy * dy
                if proj > 0:
                    perp_sq = (vx**2 + vy**2) - (proj**2)
                    if perp_sq >= 0 and perp_sq < (obs_radius**2):
                        intersect_dist = proj - math.sqrt(max(0.0, obs_radius**2 - perp_sq))
                        if 0 <= intersect_dist < distances[i]:
                            distances[i] = intersect_dist

        return np.clip(distances / max_range, 0.0, 1.0)

    @classmethod
    def compute_observation(
        cls,
        robot_pos: Sequence[float],
        robot_quat: Sequence[float],
        linear_speed: float,
        angular_speed: float,
        target_pos: Sequence[float],
        obstacles: Sequence[SceneObject],
        max_linear_velocity: float,
        max_angular_velocity: float,
    ) -> np.ndarray:
        r_yaw = quat_to_yaw(list(robot_quat))
        tx, ty = float(target_pos[0]), float(target_pos[1])
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        dx = tx - rx
        dy = ty - ry

        cos_yaw = math.cos(r_yaw)
        sin_yaw = math.sin(r_yaw)
        x_rel = dx * cos_yaw + dy * sin_yaw
        y_rel = -dx * sin_yaw + dy * cos_yaw

        dist = math.hypot(dx, dy)
        heading_error = math.atan2(y_rel, x_rel)

        max_lin = max(0.1, max_linear_velocity)
        max_ang = max(0.1, max_angular_velocity)
        lin_speed_norm = linear_speed / max_lin
        ang_speed_norm = angular_speed / max_ang

        lidar_readings = cls.compute_lidar_distances(robot_pos, r_yaw, obstacles)

        obs = np.zeros(16, dtype=np.float32)
        obs[0] = np.clip(x_rel, -10.0, 10.0)
        obs[1] = np.clip(y_rel, -10.0, 10.0)
        obs[2] = np.clip(dist, 0.0, 15.0)
        obs[3] = heading_error / math.pi
        obs[4] = np.clip(lin_speed_norm, -2.0, 2.0)
        obs[5] = np.clip(ang_speed_norm, -2.0, 2.0)
        obs[6:14] = lidar_readings
        obs[14] = rx / 5.0
        obs[15] = ry / 5.0
        return obs

    def _get_obs(self) -> np.ndarray:
        state = self.engine.get_state()
        return self.compute_observation(
            robot_pos=state.robot.position,
            robot_quat=state.robot.quaternion,
            linear_speed=state.robot.linear_speed,
            angular_speed=state.robot.angular_speed,
            target_pos=self.target_position,
            obstacles=self.project.objects,
            max_linear_velocity=self.robot_cfg.max_linear_velocity,
            max_angular_velocity=self.robot_cfg.max_angular_velocity,
        )

    def _check_collision(self) -> bool:
        """Inspects MuJoCo contacts for robot-obstacle and robot-ground penetration."""
        if not self.engine.model or not self.engine.data:
            return False

        # 1. Direct MuJoCo contact geoms
        num_contacts = self.engine.data.ncon
        for i in range(num_contacts):
            con = self.engine.data.contact[i]
            g1 = con.geom1
            g2 = con.geom2
            g1_name = self.engine.model.geom(g1).name
            g2_name = self.engine.model.geom(g2).name

            is_robot1 = "robot" in g1_name
            is_robot2 = "robot" in g2_name
            if (is_robot1 or is_robot2) and not (is_robot1 and is_robot2):
                other_name = g2_name if is_robot1 else g1_name
                # Ground plane is normal support, other objects or arena geoms are collisions
                if other_name != "ground":
                    return True

        # 2. Geometric clearance check against obstacles
        state = self.engine.get_state()
        rx, ry = state.robot.position[0], state.robot.position[1]
        robot_footprint = 0.22

        for obj in self.project.objects:
            ox, oy = obj.position[0], obj.position[1]
            if obj.type == "box":
                obs_radius = math.hypot(obj.size[0], obj.size[1])
            elif obj.type in ("sphere", "cylinder"):
                obs_radius = obj.size[0]
            else:
                obs_radius = 0.25

            clearance = math.hypot(rx - ox, ry - oy)
            if clearance < (robot_footprint + obs_radius * 0.85):
                return True

        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Denormalize action to linear and angular velocity
        act = np.clip(action, -1.0, 1.0)
        target_lin_vel = float(act[0] * self.robot_cfg.max_linear_velocity)
        target_ang_vel = float(act[1] * self.robot_cfg.max_angular_velocity)

        # Apply to MuJoCo simulation and step physics
        self.engine.set_teleop(target_lin_vel, target_ang_vel)
        for _ in range(self.substeps):
            self.engine.step()

        # Gather state
        state = self.engine.get_state()
        r_pos = state.robot.position
        self.trajectory.append((r_pos[0], r_pos[1], r_pos[2]))

        dist_to_goal = math.hypot(self.target_position[0] - r_pos[0], self.target_position[1] - r_pos[1])

        # Reward components
        # 1. Progress reward: delta distance towards target
        reward_progress = 6.0 * (self.prev_distance - dist_to_goal)

        # 2. Heading alignment reward
        r_yaw = quat_to_yaw(state.robot.quaternion)
        dx = self.target_position[0] - r_pos[0]
        dy = self.target_position[1] - r_pos[1]
        target_bearing = math.atan2(dy, dx)
        angle_diff = (target_bearing - r_yaw + math.pi) % (2.0 * math.pi) - math.pi
        reward_heading = 0.3 * math.cos(angle_diff)

        # 3. Effort and time step penalty
        reward_step = -0.02
        reward_effort = -0.01 * float(act[0]**2 + act[1]**2)

        # Check termination
        terminated = False
        truncated = False
        is_collision = self._check_collision()
        is_success = dist_to_goal <= self.distance_threshold

        info: Dict[str, Any] = {
            "distance_to_goal": dist_to_goal,
            "prev_distance": self.prev_distance,
            "angle_diff": angle_diff,
            "is_success": is_success,
            "is_collision": is_collision,
            "sim_time": state.sim_time,
            "step": self.current_step,
            "robot_position": list(r_pos),
        }

        # Check if custom reward function is supplied
        if self._custom_reward_fn is not None:
            try:
                res = self._custom_reward_fn(state, act, info)
                if isinstance(res, (tuple, list)):
                    reward = float(res[0])
                    if len(res) > 1 and bool(res[1]):
                        terminated = True
                else:
                    reward = float(res)
            except Exception as cr_err:
                logger.warning("Custom reward execution error (%s). Falling back to baseline.", cr_err)
                reward = reward_progress + reward_heading + reward_step + reward_effort
                if is_collision:
                    reward -= self.collision_penalty
                    terminated = True
                elif is_success:
                    reward += self.goal_reward
                    terminated = True
        else:
            reward = reward_progress + reward_heading + reward_step + reward_effort
            if is_collision:
                reward -= self.collision_penalty
                terminated = True
            elif is_success:
                reward += self.goal_reward
                terminated = True

        self.prev_distance = dist_to_goal

        if is_collision and not info.get("termination_reason"):
            info["termination_reason"] = "collision"
        elif is_success and not info.get("termination_reason"):
            info["termination_reason"] = "goal_reached"
        elif self.current_step >= self.max_episode_steps and not terminated:
            truncated = True
            info["termination_reason"] = "timeout"

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.trajectory.clear()

        # Randomize or reset target position
        if self.randomize_training:
            # Random goal in arena forward workspace
            rand_x = float(np.random.uniform(1.2, 3.6))
            rand_y = float(np.random.uniform(-1.5, 1.5))
            self.target_position = np.array([rand_x, rand_y, 0.25], dtype=np.float32)
        else:
            self.target_position = np.array(self.default_target, dtype=np.float32)

        # Reset simulation engine to initial state
        self.engine.reset()

        state = self.engine.get_state()
        r_pos = state.robot.position
        self.prev_distance = math.hypot(self.target_position[0] - r_pos[0], self.target_position[1] - r_pos[1])
        self.trajectory.append((r_pos[0], r_pos[1], r_pos[2]))

        obs = self._get_obs()
        info = {
            "target_position": list(self.target_position),
            "initial_distance": self.prev_distance,
        }
        return obs, info


class RoboForgeDroneGymEnv(gym.Env):
    """Reinforcement learning environment for 6-DOF aerial drone takeoff, hovering, and landing."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str = "takeoff_and_land",
        target_altitude: float = 1.5,
        max_episode_steps: int = 300,
        custom_reward_code: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.task = task
        self.target_altitude = target_altitude
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.custom_reward_code = custom_reward_code

        # 4 continuous flight actions: [vertical_thrust, roll_cmd, pitch_cmd, yaw_cmd]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # 14 observation dimensions:
        # [0..2]: [x, y, z]
        # [3..5]: [vx, vy, vz]
        # [6..8]: [roll, pitch, yaw]
        # [9..11]: [target_x, target_y, target_z]
        # [12]: altitude_error
        # [13]: planar_distance_to_target
        self.observation_space = spaces.Box(
            low=-25.0, high=25.0, shape=(14,), dtype=np.float32
        )

        self.pos = np.array([2.0, 0.0, 0.05], dtype=np.float32)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([2.0, 0.0, target_altitude], dtype=np.float32)
        self.phase = "takeoff"  # "takeoff", "hover", "landing", "grounded"

        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.pos = np.array([2.0, 0.0, 0.05], dtype=np.float32)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([2.0, 0.0, self.target_altitude], dtype=np.float32)
        self.phase = "takeoff"

        obs = self._get_obs()
        return obs, {"phase": self.phase, "altitude": float(self.pos[2])}

    def _get_obs(self) -> np.ndarray:
        alt_err = float(self.target_pos[2] - self.pos[2])
        p_dist = float(math.hypot(self.target_pos[0] - self.pos[0], self.target_pos[1] - self.pos[1]))
        return np.array([
            self.pos[0], self.pos[1], self.pos[2],
            self.vel[0], self.vel[1], self.vel[2],
            self.rot[0], self.rot[1], self.rot[2],
            self.target_pos[0], self.target_pos[1], self.target_pos[2],
            alt_err, p_dist
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        self.current_step += 1
        dt = 0.02
        act = np.clip(action, -1.0, 1.0)

        # Simplified 6-DOF quadcopter kinematic update for RL training
        thrust_z = (act[0] + 1.0) * 0.5 * 19.62  # [0, 2G]
        gravity = -9.81
        acc_z = (thrust_z / 1.2) + gravity

        # Update velocities and positions
        self.vel[2] = float(np.clip(self.vel[2] + acc_z * dt, -4.0, 4.0))
        self.vel[0] = float(np.clip(self.vel[0] + act[1] * 3.0 * dt, -3.0, 3.0))
        self.vel[1] = float(np.clip(self.vel[1] + act[2] * 3.0 * dt, -3.0, 3.0))

        self.pos[2] = float(max(0.04, self.pos[2] + self.vel[2] * dt))
        self.pos[0] = float(self.pos[0] + self.vel[0] * dt)
        self.pos[1] = float(self.pos[1] + self.vel[1] * dt)

        # Multi-stage task progression (Takeoff -> Hover -> Land)
        terminated = False
        reward = 0.0

        alt_err = abs(self.target_pos[2] - self.pos[2])
        reward -= 2.0 * alt_err
        reward -= 0.5 * abs(self.vel[2])
        reward -= 0.2 * (abs(self.vel[0]) + abs(self.vel[1]))

        if self.phase == "takeoff":
            if alt_err < 0.15:
                reward += 20.0
                if self.task == "takeoff_and_land" and self.current_step > 50:
                    self.phase = "landing"
                    self.target_pos[2] = 0.05
                else:
                    self.phase = "hover"
        elif self.phase == "hover":
            if alt_err < 0.1:
                reward += 5.0
            if self.task == "takeoff_and_land" and self.current_step > 120:
                self.phase = "landing"
                self.target_pos[2] = 0.05
        elif self.phase == "landing":
            if self.pos[2] <= 0.08:
                if abs(self.vel[2]) < 0.5:
                    reward += 100.0  # Safe touchdown bonus
                    terminated = True
                else:
                    reward -= 30.0  # Hard touchdown penalty
                    terminated = True

        truncated = self.current_step >= self.max_episode_steps
        obs = self._get_obs()
        info = {
            "phase": self.phase,
            "altitude": float(self.pos[2]),
            "vertical_speed": float(self.vel[2]),
            "success": terminated and reward > 50.0
        }
        return obs, float(reward), terminated, truncated, info
