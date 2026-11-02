"""Gymnasium-compliant reinforcement learning environment for RoboForge differential-drive MuJoCo robot."""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from backend.models import Project, RobotConfig, SceneObject, SimulationSettings
from backend.simulation import SimulationEngine, quat_to_yaw

logger = logging.getLogger("roboforge.rl_env")


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
    """Real MuJoCo-physics RL environment for a quadcopter. The 4 actions are the 4 rotor thrust
    commands applied to the genuine rotor sites of the real drone body; observations are read from
    the real MuJoCo state; reward rewards reaching/holding a target position with stability. No
    scripted phases or kinematic shortcuts -- the agent must learn to fly real differential-thrust
    dynamics."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str = "hover",
        target_altitude: float = 1.5,
        max_episode_steps: int = 300,
        custom_reward_code: Optional[str] = None,
        substeps: int = 5,
        randomize_target: bool = True,
        project: Optional[Project] = None,
        drone_id: str = "drone_rl",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.task = task
        self.target_altitude = float(target_altitude)
        self.max_episode_steps = max_episode_steps
        self.substeps = max(1, int(substeps))
        self.randomize_target = randomize_target
        self.current_step = 0
        self.custom_reward_code = custom_reward_code
        self.drone_id = drone_id

        # 4 continuous actions = normalized thrust command per rotor [FL, FR, RL, RR].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 15 observations: pos(3), lin vel(3), roll/pitch/yaw(3), ang vel(3), target(3).
        self.observation_space = spaces.Box(low=-30.0, high=30.0, shape=(15,), dtype=np.float32)

        # Build a dedicated real MuJoCo scene containing just this drone.
        from backend.simulation import MuJoCoEngine
        if project is None:
            project = Project(
                name="DroneRL",
                robot_active=False,
                objects=[SceneObject(
                    id=drone_id, name="RL-Quad", type="box",
                    position=[0.0, 0.0, 0.08], size=[0.22, 0.22, 0.08],
                    color="#38bdf8", is_static=False, mass=1.2,
                )],
            )
        else:
            # Ensure a drone entity with our id exists in the provided project.
            if not any(o.id == drone_id for o in project.objects):
                project.objects.append(SceneObject(
                    id=drone_id, name="RL-Quad", type="box",
                    position=[0.0, 0.0, 0.08], size=[0.22, 0.22, 0.08],
                    color="#38bdf8", is_static=False, mass=1.2,
                ))
        self.engine = MuJoCoEngine(project)
        self.engine.rebuild_model(preserve_state=False)
        # Arm the drone's rotors and disable the built-in PID so only the policy drives thrust.
        ctrl = self.engine.drone_controllers.get(drone_id)
        if ctrl is not None:
            ctrl.rotors_active = True
            ctrl.flight_mode = "flying"
        self.target_pos = np.array([0.0, 0.0, self.target_altitude], dtype=np.float32)

        self._rng = np.random.RandomState(seed)
        self._compiled_reward = None
        if custom_reward_code:
            self._compiled_reward = self._compile_custom_reward(custom_reward_code)

    def _drone_state(self):
        """Reads real pose/velocity of the drone body from MuJoCo."""
        import mujoco
        e = self.engine
        bid = e.object_body_ids.get(self.drone_id, -1)
        if bid < 0:
            return (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        pos = np.array(e.data.xpos[bid], dtype=np.float32)
        bm = np.array(e.data.xmat[bid], dtype=np.float64).reshape(3, 3)
        roll = math.atan2(bm[2, 1], bm[2, 2])
        pitch = math.atan2(-bm[2, 0], math.sqrt(bm[2, 1] ** 2 + bm[2, 2] ** 2))
        yaw = math.atan2(bm[1, 0], bm[0, 0])
        rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        jnt = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{self.drone_id}")
        if jnt >= 0:
            dof = e.model.jnt_dofadr[jnt]
            lvel = np.array(e.data.qvel[dof:dof + 3], dtype=np.float32)
            avel = np.array(e.data.qvel[dof + 3:dof + 6], dtype=np.float32)
        else:
            lvel = np.zeros(3, dtype=np.float32); avel = np.zeros(3, dtype=np.float32)
        return pos, lvel, rpy, avel

    def _get_obs(self) -> np.ndarray:
        pos, lvel, rpy, avel = self._drone_state()
        obs = np.concatenate([pos, lvel, rpy, avel, self.target_pos]).astype(np.float32)
        return np.clip(obs, -30.0, 30.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.current_step = 0
        self.engine.reset()
        ctrl = self.engine.drone_controllers.get(self.drone_id)
        if ctrl is not None:
            ctrl.rotors_active = True
            ctrl.flight_mode = "flying"
        # Randomize the target hover point within a sensible envelope.
        if self.randomize_target:
            tx = float(self._rng.uniform(-1.5, 1.5))
            ty = float(self._rng.uniform(-1.5, 1.5))
            tz = float(self._rng.uniform(0.8, 2.2))
            self.target_pos = np.array([tx, ty, tz], dtype=np.float32)
        else:
            self.target_pos = np.array([0.0, 0.0, self.target_altitude], dtype=np.float32)
        if ctrl is not None:
            ctrl.target_position = [float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2])]
            ctrl.target_altitude = float(self.target_pos[2])
        pos, lvel, rpy, avel = self._drone_state()
        self.prev_dist = float(np.linalg.norm(self.target_pos - pos))
        return self._get_obs(), {"altitude": float(pos[2])}

    def step(self, action: np.ndarray):
        import mujoco
        self.current_step += 1
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Map 4 normalized actions to per-rotor thrust around hover: [0, 2*hover] each.
        hover = self.engine.rotor_hover_thrust(self.drone_id)
        thrusts = [float((a + 1.0) * hover) for a in act[:4]]

        # Apply REAL per-rotor thrust and step REAL MuJoCo physics.
        for _ in range(self.substeps):
            if self.engine.data is not None and hasattr(self.engine.data, "qfrc_applied"):
                self.engine.data.qfrc_applied[:] = 0.0
            self.engine.apply_rotor_thrusts(self.drone_id, thrusts)
            mujoco.mj_step(self.engine.model, self.engine.data)

        pos, lvel, rpy, avel = self._drone_state()
        dist = float(np.linalg.norm(self.target_pos - pos))
        speed = float(np.linalg.norm(lvel))
        tilt = float(abs(rpy[0]) + abs(rpy[1]))
        is_crash = (tilt > 1.4 or pos[2] < 0.03 or abs(pos[0]) > 8.0 or abs(pos[1]) > 8.0)
        is_success = (dist < 0.25 and speed < 0.4 and tilt < 0.25)

        terminated = False
        truncated = False
        # Rich, cross-compatible info dict. Drone-native keys AND ground-robot aliases are both
        # provided so a reward function written for EITHER robot type works unchanged.
        info = {
            # Drone-native
            "altitude": float(pos[2]),
            "distance_to_target": dist,
            "tilt": tilt,
            "vertical_speed": float(lvel[2]),
            # Universal aliases (match the ground/arm envs so pasted rewards "just work")
            "distance_to_goal": dist,
            "prev_distance": float(self.prev_dist),
            "speed": speed,
            "velocity": [float(lvel[0]), float(lvel[1]), float(lvel[2])],
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "target_position": [float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2])],
            "roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2]),
            "angle_diff": 0.0,                 # not meaningful for a drone, provided for compatibility
            "is_collision": bool(is_crash),
            "is_success": bool(is_success),
            "step": int(self.current_step),
        }

        if self._compiled_reward is not None:
            try:
                res = self._compiled_reward(self.engine.get_state(), act, info)
                # Support both `return reward` and `return (reward, done)` forms.
                if isinstance(res, (tuple, list)):
                    reward = float(res[0])
                    if len(res) > 1 and bool(res[1]):
                        terminated = True
                else:
                    reward = float(res)
            except Exception as e:
                logger.warning("Custom drone reward error (%s); using default.", e)
                reward = self._default_reward(pos, lvel, rpy, avel, dist)
        else:
            reward = self._default_reward(pos, lvel, rpy, avel, dist)

        self.prev_dist = dist

        # Termination: crashed/flipped (excessive tilt) or left the arena.
        if is_crash:
            reward -= 20.0
            terminated = True
            info["termination_reason"] = "crash" if (pos[2] < 0.03 or tilt > 1.4) else "out_of_bounds"
        elif is_success:
            reward += 5.0
            info["success"] = True

        if self.current_step >= self.max_episode_steps and not terminated:
            truncated = True
            info.setdefault("termination_reason", "timeout")

        return self._get_obs(), float(reward), terminated, truncated, info

    def _default_reward(self, pos, lvel, rpy, avel, dist) -> float:
        """Dense reward: progress to target, proximity, and stability (upright, low spin)."""
        r = 0.0
        r += 4.0 * (self.prev_dist - dist)          # progress toward target
        r -= 0.6 * dist                              # proximity shaping
        r -= 0.08 * float(np.linalg.norm(lvel))      # discourage high speed
        r -= 0.25 * (abs(float(rpy[0])) + abs(float(rpy[1])))  # stay upright
        r -= 0.05 * float(np.linalg.norm(avel))      # discourage spinning
        r -= 0.01                                    # mild time penalty
        return r

    def _compile_custom_reward(self, code: str):
        try:
            scope: Dict[str, Any] = {}
            exec(code, {"np": np, "math": math}, scope)
            fn = scope.get("compute_reward")
            return fn if callable(fn) else None
        except Exception as e:
            logger.warning("Failed to compile custom drone reward: %s", e)
            return None

    def close(self):
        try:
            if self.engine is not None:
                self.engine.data = None
                self.engine.model = None
        except Exception:
            pass


class RoboForgeArmGymEnv(gym.Env):
    """Reinforcement learning environment for articulated robotic arms (reach, target tracking, pick-and-place)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        project: Optional[Project] = None,
        target_position: Tuple[float, float, float] = (0.35, 0.20, 0.25),
        max_episode_steps: int = 200,
        custom_reward_code: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if project is not None:
            self.project = project.model_copy(deep=True)
        else:
            from backend.project_manager import get_default_project
            self.project = get_default_project()

        # If project has no generic_robot, construct default 2-link/3-link arm for environment
        if self.project.generic_robot is None or not self.project.generic_robot.joints:
            from backend.models import GenericRobotModel, LinkModel, JointModel, JointLimit, LinkGeom, LinkInertial
            self.project.generic_robot = GenericRobotModel(
                name="2-Link Planar Arm",
                root_link="base_link",
                base_position=[0.0, 0.0, 0.0],
                links=[
                    LinkModel(name="base_link", visual=LinkGeom(type="cylinder", size=[0.18, 0.05], origin_pos=[0, 0, 0.05], color="#334155")),
                    LinkModel(name="arm_link_1", visual=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0.0], color="#2563eb")),
                    LinkModel(name="arm_link_2", visual=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0.0], color="#f97316"))
                ],
                joints=[
                    JointModel(name="shoulder_joint", type="revolute", parent_link="base_link", child_link="arm_link_1", origin_pos=[0, 0, 0.1], axis=[0, 0, 1], limit=JointLimit(lower=-3.14, upper=3.14)),
                    JointModel(name="elbow_joint", type="revolute", parent_link="arm_link_1", child_link="arm_link_2", origin_pos=[0.4, 0, 0], axis=[0, 0, 1], limit=JointLimit(lower=-2.6, upper=2.6))
                ]
            )

        self.target_position = np.array(target_position, dtype=np.float32)
        self.default_target = np.array(target_position, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.custom_reward_code = custom_reward_code
        self._custom_reward_fn = None
        if self.custom_reward_code and self.custom_reward_code.strip():
            self._compile_custom_reward(self.custom_reward_code)

        # Controllable joints
        self.controllable_joints = [
            j for j in self.project.generic_robot.joints if j.type in ("revolute", "continuous", "prismatic")
        ]
        num_actuators = max(1, len(self.controllable_joints))

        # Joint angle bounds
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for j in self.controllable_joints:
            lower = j.limit.lower if j.limit else -3.14159
            upper = j.limit.upper if j.limit else 3.14159
            self.joint_limits_lower.append(lower)
            self.joint_limits_upper.append(upper)

        self.joint_limits_lower = np.array(self.joint_limits_lower, dtype=np.float32)
        self.joint_limits_upper = np.array(self.joint_limits_upper, dtype=np.float32)

        # Action: normalized delta joint command [-1.0, 1.0] for each joint
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float32)

        # Obs: [joint_angles_norm (N), joint_vels (N), target_pos (3), ee_pos (3), delta_to_target (3), distance (1)]
        obs_dim = 2 * num_actuators + 10
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(obs_dim,), dtype=np.float32)

        self.current_qpos = np.zeros(num_actuators, dtype=np.float32)
        self.current_qvel = np.zeros(num_actuators, dtype=np.float32)
        self.prev_distance = 1.0

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
        except Exception as e:
            pass

    def _compute_ee_position(self) -> np.ndarray:
        try:
            from backend.kinematics import compute_forward_kinematics
            joint_dict = {
                j.name: float(self.current_qpos[i]) for i, j in enumerate(self.controllable_joints)
            }
            fk_res = compute_forward_kinematics(self.project.generic_robot, joint_dict)
            return np.array(fk_res.end_effector_position, dtype=np.float32)
        except Exception:
            return np.array([0.35, 0.20, 0.25], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._compute_ee_position()
        delta = self.target_position - ee_pos
        dist = float(np.linalg.norm(delta))

        # Normalized joint angles in [-1, 1]
        span = np.maximum(self.joint_limits_upper - self.joint_limits_lower, 1e-4)
        mid = (self.joint_limits_upper + self.joint_limits_lower) * 0.5
        norm_qpos = (self.current_qpos - mid) / (span * 0.5)

        obs_parts = [
            norm_qpos,
            self.current_qvel,
            self.target_position,
            ee_pos,
            delta,
            np.array([dist], dtype=np.float32)
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.current_qpos = np.zeros(len(self.controllable_joints), dtype=np.float32)
        self.current_qvel = np.zeros(len(self.controllable_joints), dtype=np.float32)

        # Slightly randomize initial joint positions
        for i, j in enumerate(self.controllable_joints):
            low = self.joint_limits_lower[i] * 0.2
            high = self.joint_limits_upper[i] * 0.2
            self.current_qpos[i] = float(np.random.uniform(low, high))

        ee_pos = self._compute_ee_position()
        self.prev_distance = float(np.linalg.norm(self.target_position - ee_pos))

        obs = self._get_obs()
        return obs, {"ee_pos": list(ee_pos), "initial_distance": self.prev_distance}

    def step(self, action: np.ndarray):
        self.current_step += 1
        dt = 0.05
        act = np.clip(action, -1.0, 1.0)

        # Update velocities and positions
        delta_q = act * dt * 1.5
        self.current_qvel = delta_q / dt
        self.current_qpos = np.clip(
            self.current_qpos + delta_q,
            self.joint_limits_lower,
            self.joint_limits_upper
        )

        ee_pos = self._compute_ee_position()
        delta = self.target_position - ee_pos
        dist = float(np.linalg.norm(delta))

        # Dense shaping reward
        reward_dist = -dist * 2.0
        reward_progress = (self.prev_distance - dist) * 10.0
        reward_effort = -0.01 * float(np.sum(act ** 2))
        reward = reward_dist + reward_progress + reward_effort

        is_success = dist < 0.06
        if is_success:
            reward += 100.0

        terminated = is_success
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "distance_to_goal": dist,
            "prev_distance": self.prev_distance,
            "is_success": is_success,
            "ee_pos": list(ee_pos),
            "step": self.current_step
        }

        if self._custom_reward_fn is not None:
            try:
                res = self._custom_reward_fn(None, act, info)
                if isinstance(res, (tuple, list)):
                    reward = float(res[0])
                    if len(res) > 1 and bool(res[1]):
                        terminated = True
                else:
                    reward = float(res)
            except Exception:
                pass

        self.prev_distance = dist
        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info
