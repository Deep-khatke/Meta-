"""
Universal Gymnasium RL environment for ANY imported articulated robot in RoboForge.

Unlike RoboForgeGymEnv (which is hardwired to the differential-drive chassis and a 2-D
linear/angular action), this environment introspects whatever generic robot the user imported
(URDF/MJCF -> project.generic_robot) and builds its action/observation spaces from the robot's
actual actuated joints. It therefore works for an arm, a gantry, a leg, or any custom kinematic
tree -- not just a rover.

Task: drive the end-effector link to a Cartesian target pose ("reach"). This is the canonical
manipulation task Isaac Lab ships as its first tutorial, implemented here on real MuJoCo physics.

The action is a per-joint normalized command in [-1, 1] mapped to each joint's limit (position
mode) or to a torque range (torque mode). The reward is dense shaping (distance progress +
posture/effort penalties) with a sparse success bonus when the end-effector reaches the goal,
plus a self-collision penalty. Everything is measured from genuine MuJoCo state -- no scripting.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from backend.models import Project, GenericRobotModel, JointModel
from backend.simulation import SimulationEngine

logger = logging.getLogger("roboforge.rl_generic_env")


# Joints that can actually be actuated / learned on.
ACTUATED_JOINT_TYPES = ("revolute", "continuous", "prismatic")


def _actuated_joints(robot: GenericRobotModel) -> List[JointModel]:
    return [j for j in robot.joints if j.type in ACTUATED_JOINT_TYPES]


def _joint_range(joint: JointModel) -> Tuple[float, float]:
    """Returns a sensible (low, high) command range for a joint."""
    if joint.type == "continuous" or joint.limit is None:
        return (-math.pi, math.pi)
    return (float(joint.limit.lower), float(joint.limit.upper))


class RoboForgeReachEnv(gym.Env):
    """RL env that trains an arbitrary articulated robot to reach a Cartesian target."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        project: Project,
        target_position: Optional[Tuple[float, float, float]] = None,
        end_effector_link: Optional[str] = None,
        control_frequency_hz: float = 20.0,
        max_episode_steps: int = 250,
        distance_threshold: float = 0.08,
        randomize_target: bool = True,
        seed: Optional[int] = None,
        custom_reward_code: Optional[str] = None,
    ):
        super().__init__()
        if project.generic_robot is None or not _actuated_joints(project.generic_robot):
            raise ValueError(
                "RoboForgeReachEnv requires project.generic_robot with at least one actuated joint. "
                "Import a URDF/MJCF arm or articulated robot first."
            )

        self.project = project.model_copy(deep=True)
        self.robot: GenericRobotModel = self.project.generic_robot  # type: ignore
        self.joints = _actuated_joints(self.robot)
        self.joint_names = [j.name for j in self.joints]
        self.joint_ranges = [_joint_range(j) for j in self.joints]
        self.n_joints = len(self.joints)

        self.end_effector_link = end_effector_link or self._infer_end_effector()
        self.control_frequency_hz = control_frequency_hz
        self.max_episode_steps = max_episode_steps
        self.distance_threshold = distance_threshold
        self.randomize_target = randomize_target
        self.custom_reward_code = custom_reward_code
        self._custom_reward_fn = None
        if custom_reward_code and custom_reward_code.strip():
            self._compile_custom_reward(custom_reward_code)

        sim_dt = self.project.settings.timestep
        self.substeps = max(1, int(round((1.0 / control_frequency_hz) / sim_dt)))

        # Engine
        self.engine = SimulationEngine(self.project)
        self._ee_body_id = self._resolve_ee_body_id()
        self._base_pos = np.array(self.robot.base_position, dtype=np.float32)
        self._reach = self._estimate_reach()

        # Default/target position
        if target_position is not None:
            self.default_target = np.array(target_position, dtype=np.float32)
        else:
            self.default_target = self._sample_target(np.random.RandomState(0))
        self.target_position = self.default_target.copy()

        # Action: one normalized command per actuated joint.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        # Observation: joint pos (n) + joint vel (n) + ee_pos(3) + target(3) + ee->target vec(3) + dist(1)
        obs_dim = self.n_joints * 2 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(low=-30.0, high=30.0, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.prev_distance = 0.0

        if seed is not None:
            self.reset(seed=seed)

    # --- introspection helpers -------------------------------------------------

    def _infer_end_effector(self) -> str:
        """The end-effector is the child link of the last actuated joint in the chain."""
        if self.joints:
            return self.joints[-1].child_link
        return self.robot.links[-1].name if self.robot.links else self.robot.root_link

    def _resolve_ee_body_id(self) -> int:
        import mujoco
        if not self.engine.model:
            return -1
        for candidate in (self.end_effector_link, f"link_{self.end_effector_link}"):
            bid = mujoco.mj_name2id(self.engine.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid >= 0:
                return bid
        # Fallback: last body in the model.
        return self.engine.model.nbody - 1

    def _ee_world_pos(self) -> np.ndarray:
        if not self.engine.data or self._ee_body_id < 0:
            return self._base_pos.copy()
        return np.array(self.engine.data.xpos[self._ee_body_id], dtype=np.float32)

    def _estimate_reach(self) -> float:
        """Rough maximum reach = sum of joint origin offsets + link extents."""
        total = 0.0
        for j in self.robot.joints:
            total += float(np.linalg.norm(j.origin_pos))
        # add a link half-extent cushion
        total += 0.3
        return max(0.3, total)

    def _sample_target(self, rng: np.random.RandomState) -> np.ndarray:
        """Sample a reachable Cartesian target inside the robot's workspace sphere."""
        r = self._reach * rng.uniform(0.35, 0.85)
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(math.pi * 0.15, math.pi * 0.6)  # bias above the base
        x = self._base_pos[0] + r * math.sin(phi) * math.cos(theta)
        y = self._base_pos[1] + r * math.sin(phi) * math.sin(theta)
        z = max(0.05, self._base_pos[2] + r * math.cos(phi))
        return np.array([x, y, z], dtype=np.float32)

    def _compile_custom_reward(self, code_str: str) -> None:
        try:
            scope: Dict[str, Any] = {"math": math, "np": np}
            exec(code_str, scope)  # noqa: S102 -- sandboxed scope, user-authored reward only
            fn = scope.get("compute_reward") or scope.get("calculate_reward")
            if callable(fn):
                self._custom_reward_fn = fn
            else:
                logger.warning("Custom reward code did not define compute_reward(state, action, info).")
        except Exception as e:
            logger.error("Failed to compile custom reward: %s", e)

    # --- observation -----------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        state = self.engine.get_state()
        jp = state.joint_positions
        jv = state.joint_velocities
        q = np.array([jp.get(n, 0.0) for n in self.joint_names], dtype=np.float32)
        qd = np.array([jv.get(n, 0.0) for n in self.joint_names], dtype=np.float32)
        ee = self._ee_world_pos()
        tgt = self.target_position
        vec = tgt - ee
        dist = float(np.linalg.norm(vec))
        obs = np.concatenate([q, qd, ee, tgt, vec, np.array([dist], dtype=np.float32)]).astype(np.float32)
        return np.clip(obs, -30.0, 30.0)

    # --- gym API ---------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        act = np.clip(action, -1.0, 1.0)

        # Map normalized action to per-joint targets.
        targets: Dict[str, float] = {}
        for i, j in enumerate(self.joints):
            mode = j.actuator_mode or "position"
            lo, hi = self.joint_ranges[i]
            if mode == "torque":
                eff = (j.limit.effort if j.limit else 50.0) or 50.0
                targets[j.name] = float(act[i] * eff)
            elif mode == "velocity":
                vmax = (j.limit.velocity if j.limit else 5.0) or 5.0
                targets[j.name] = float(act[i] * vmax)
            else:  # position: map [-1,1] -> [lo, hi]
                mid = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                targets[j.name] = float(mid + act[i] * half)

        self.engine.set_joint_targets(targets)
        for _ in range(self.substeps):
            self.engine.step()

        ee = self._ee_world_pos()
        dist = float(np.linalg.norm(self.target_position - ee))

        # Reward shaping
        reward_progress = 8.0 * (self.prev_distance - dist)
        reward_dist = -0.4 * dist
        reward_effort = -0.005 * float(np.sum(act ** 2))
        reward_step = -0.01

        is_success = dist <= self.distance_threshold
        is_collision = self._check_self_collision()

        info: Dict[str, Any] = {
            "distance_to_goal": dist,
            "prev_distance": self.prev_distance,
            "is_success": is_success,
            "is_collision": is_collision,
            "ee_position": ee.tolist(),
            "step": self.current_step,
        }

        terminated = False
        truncated = False

        if self._custom_reward_fn is not None:
            try:
                res = self._custom_reward_fn(self.engine.get_state(), act, info)
                if isinstance(res, (tuple, list)):
                    reward = float(res[0])
                    terminated = bool(res[1]) if len(res) > 1 else False
                else:
                    reward = float(res)
            except Exception as e:
                logger.warning("Custom reward error (%s); using default.", e)
                reward = reward_progress + reward_dist + reward_effort + reward_step
        else:
            reward = reward_progress + reward_dist + reward_effort + reward_step

        if is_collision:
            reward -= 5.0
        if is_success:
            reward += 100.0
            terminated = True

        self.prev_distance = dist
        if self.current_step >= self.max_episode_steps and not terminated:
            truncated = True
            info["termination_reason"] = "timeout"
        elif is_success:
            info["termination_reason"] = "goal_reached"

        return self._get_obs(), float(reward), terminated, truncated, info

    def _check_self_collision(self) -> bool:
        """Counts contacts between two robot links (excludes ground/base support)."""
        if not self.engine.model or not self.engine.data:
            return False
        import mujoco
        for i in range(int(self.engine.data.ncon)):
            con = self.engine.data.contact[i]
            n1 = mujoco.mj_id2name(self.engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            n2 = mujoco.mj_id2name(self.engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            if "link_" in n1 and "link_" in n2 and n1 != n2:
                return True
        return False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2**31 - 1))
        self.current_step = 0

        if self.randomize_target:
            self.target_position = self._sample_target(rng)
        else:
            self.target_position = self.default_target.copy()

        self.engine.reset()
        ee = self._ee_world_pos()
        self.prev_distance = float(np.linalg.norm(self.target_position - ee))

        info = {"target_position": self.target_position.tolist(), "initial_distance": self.prev_distance}
        return self._get_obs(), info

    def close(self):
        pass


def detect_robot_kind(project: Project) -> str:
    """Classifies the active robot so the trainer picks the right environment.

    Returns one of: "arm" (fixed-base articulated), "mobile" (diff-drive chassis),
    "generic" (free-base articulated), or "drone".
    """
    gr = project.generic_robot
    if gr is not None and _actuated_joints(gr):
        return "arm" if gr.is_fixed_base else "generic"
    return "mobile"
