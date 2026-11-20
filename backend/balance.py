"""Push-recovery balance training for the free-standing humanoid and robot dog.

Everything here is real physics and real reinforcement learning:

  * `BalanceEnv` is a Gymnasium environment around the SAME MuJoCo model the studio simulates. The robot starts
    standing, is shoved by random horizontal pushes, and must stay upright.
  * `BalanceTrainer` runs PPO (Stable-Baselines3) in a background thread with a push-strength curriculum.
  * `BalanceController` deploys the learned policy on the live simulation by setting joint targets at 50 Hz.
  * A freshly spawned robot has NO controller: it only holds its default pose, so a firm push topples it.
    Training attaches the learned reflex, and `evaluate` measures survival before and after.
"""
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    class _SpacesMock:
        Box = object
        Discrete = object
    spaces = _SpacesMock()
import mujoco
import numpy as np

from backend.robot_presets import build_preset

logger = logging.getLogger("roboforge.balance")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "balance")

# fall_height: base height (m) below which the robot counts as fallen. min_up: minimum uprightness (cosine of
# the tilt angle). push_max: strongest training push (N, applied for 0.1 s). demo_push: the push used by the
# demo button and by the survival evaluation. scale: how far (rad) the policy may move a joint from standing.
CFG: Dict[str, Dict[str, float]] = {
    "humanoid": dict(fall_height=0.42, min_up=0.6, push_max=330.0, demo_push=220.0, scale=0.5),
    "dog": dict(fall_height=0.26, min_up=0.6, push_max=330.0, demo_push=220.0, scale=0.5),
}
PUSH_SECONDS = 0.1
CTRL_DT = 0.02  # 50 Hz policy


def make_project(kind: str):
    """A clean project containing only a free-standing preset robot (used by training environments)."""
    from backend.project_manager import get_default_project
    proj = get_default_project().model_copy(deep=True)
    proj.robot = None
    proj.robot_active = False
    proj.objects = []
    proj.generic_robot = build_preset(kind, free=True)
    return proj


def compute_obs(model, data, root: int, qadr, dadr, last_a: np.ndarray) -> np.ndarray:
    """Observation shared by training and the live controller (so a policy transfers exactly)."""
    R = data.xmat[root].reshape(3, 3)
    g_b = R.T @ np.array([0.0, 0.0, -1.0])           # gravity direction in the body frame
    v_b = R.T @ data.qvel[0:3]                        # base linear velocity in the body frame
    w_b = data.qvel[3:6]                              # base angular velocity (body frame)
    qj = data.qpos[qadr]
    qd = data.qvel[dadr]
    return np.concatenate([g_b, w_b * 0.25, v_b, qj, qd * 0.05, last_a]).astype(np.float32)


class BalanceEnv(gym.Env):
    metadata: Dict[str, Any] = {}

    def __init__(self, kind: str = "dog", seed: int = 0, episode_seconds: float = 6.0):
        super().__init__()
        from backend.simulation import MuJoCoEngine
        self.kind = kind
        self.cfg = CFG[kind]
        self.engine = MuJoCoEngine(make_project(kind))
        self.model, self.data = self.engine.model, self.engine.data
        gr = self.engine.project.generic_robot
        joints = [j for j in gr.joints if j.type != "fixed"]
        self.names = [j.name for j in joints]
        self.n = len(joints)
        self.act = np.array([self.engine.joint_actuator_ids[nm] for nm in self.names])
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in self.names]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in jid])
        self.dadr = np.array([self.model.jnt_dofadr[i] for i in jid])
        self.lo = np.array([j.limit.lower for j in joints])
        self.hi = np.array([j.limit.upper for j in joints])
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gr.root_link)
        self.dt = float(self.model.opt.timestep)
        self.sub = max(1, int(round(CTRL_DT / self.dt)))
        self.max_steps = int(episode_seconds / (self.sub * self.dt))
        self.observation_space = spaces.Box(-np.inf, np.inf, (9 + 3 * self.n,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.push_scale = 1.0
        self.eval_push: Optional[tuple] = None   # (angle_rad, newtons): one fixed push, used by `evaluate`
        self.t = 0
        self.last_a = np.zeros(self.n)
        self.push_left = 0
        self.push_vec = np.zeros(3)
        self.next_push = 30
        self.xy0 = np.zeros(2)

    # --- training-time helpers ---
    def set_push_scale(self, s: float) -> None:
        self.push_scale = float(s)

    def _obs(self) -> np.ndarray:
        return compute_obs(self.model, self.data, self.root, self.qadr, self.dadr, self.last_a)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.sub * 2):                     # let the contacts settle
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        self.last_a = np.zeros(self.n)
        self.push_left = 0
        self.next_push = 25 if self.eval_push else int(self.rng.integers(20, 45))
        self.xy0 = np.array(self.data.xpos[self.root][:2])
        return self._obs(), {}

    def _begin_push(self) -> None:
        if self.eval_push is not None:
            ang, mag = self.eval_push
        else:
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            mag = float(self.rng.uniform(0.3, 1.0)) * self.cfg["push_max"] * self.push_scale
            self.next_push = self.t + int(self.rng.integers(40, 75))
        self.push_vec = np.array([math.cos(ang) * mag, math.sin(ang) * mag, 0.0])
        self.push_left = int(round(PUSH_SECONDS / CTRL_DT))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[self.act] = np.clip(a * self.cfg["scale"], self.lo, self.hi)
        if self.t == self.next_push:
            self._begin_push()
        if self.push_left > 0:
            self.data.xfrc_applied[self.root, 0:3] = self.push_vec
            self.push_left -= 1
        else:
            self.data.xfrc_applied[self.root, :] = 0.0
        for _ in range(self.sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1

        obs_g = compute_obs(self.model, self.data, self.root, self.qadr, self.dadr, a)
        up = float(-obs_g[2])
        h = float(self.data.xpos[self.root][2])
        v_xy = obs_g[6:8]
        w = obs_g[3:6] * 4.0
        qd = self.data.qvel[self.dadr]
        drift = float(np.linalg.norm(self.data.xpos[self.root][:2] - self.xy0))
        fell = (h < self.cfg["fall_height"]) or (up < self.cfg["min_up"]) or bool(np.isnan(h))
        r = (1.0 + 1.0 * up - 0.4 * float(v_xy @ v_xy) - 0.02 * float(w @ w)
             - 0.03 * float(np.mean((a - self.last_a) ** 2)) - 0.0002 * float(qd @ qd) - 0.3 * drift)
        if fell:
            r -= 10.0
        self.last_a = a
        return self._obs(), float(r), fell, (self.t >= self.max_steps), {"fell": fell}


def evaluate(kind: str, policy, scale: float = 1.0, trials: int = 24, seed: int = 1234) -> float:
    """Fraction of trials in which the robot is still standing after one fixed push (eight-ish directions)."""
    env = BalanceEnv(kind, seed=seed, episode_seconds=3.5)
    survived = 0
    for i in range(trials):
        env.eval_push = (2.0 * math.pi * i / trials, CFG[kind]["demo_push"] * scale)
        obs, _ = env.reset()
        done, fell = False, False
        while not done:
            act = np.zeros(env.n) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(act)
            fell, done = info["fell"], (term or trunc)
        survived += 0 if fell else 1
    return survived / trials


class BalanceController:
    """Runs a trained policy on the LIVE engine: every 50 Hz tick it sets the robot's joint targets."""

    def __init__(self, ppo, kind: str):
        self.ppo, self.kind, self.cfg = ppo, kind, CFG[kind]
        self._bound: Optional[tuple] = None
        self.last_a: Optional[np.ndarray] = None

    @staticmethod
    def substeps_for(engine) -> int:
        return max(1, int(round(CTRL_DT / engine.project.settings.timestep)))

    def _bind(self, engine):
        key = id(engine.model)
        if self._bound is None or self._bound[0] != key:
            gr = engine.project.generic_robot
            joints = [j for j in gr.joints if j.type != "fixed"]
            names = [j.name for j in joints]
            jid = [mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in names]
            self._bound = (
                key, names,
                np.array([engine.model.jnt_qposadr[i] for i in jid]), np.array([engine.model.jnt_dofadr[i] for i in jid]),
                np.array([j.limit.lower for j in joints]), np.array([j.limit.upper for j in joints]),
                mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, gr.root_link),
            )
            self.last_a = np.zeros(len(names))
        return self._bound

    def tick(self, engine) -> None:
        _, names, qadr, dadr, lo, hi, root = self._bind(engine)
        obs = compute_obs(engine.model, engine.data, root, qadr, dadr, self.last_a)
        act = np.clip(self.ppo.predict(obs, deterministic=True)[0], -1.0, 1.0)
        self.last_a = act
        tgt = np.clip(act * self.cfg["scale"], lo, hi)
        for nm, v in zip(names, tgt):
            engine.joint_targets[nm] = float(v)


class BalanceTrainer:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = False
        self.status: Dict[str, Any] = self._blank()

    @staticmethod
    def _blank() -> Dict[str, Any]:
        return dict(phase="idle", kind=None, progress=0.0, timesteps=0, total=0, mean_reward=None, mean_length=None,
                    history=[], eval=None, message="", model_path=None, started_at=None, finished_at=None)

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, kind: str, timesteps: int, engine) -> Dict[str, Any]:
        if self.running():
            return {"status": "busy", "details": "A balance training run is already in progress."}
        if kind not in CFG:
            return {"status": "error", "details": f"Unknown robot '{kind}'."}
        self.stop_flag = False
        self.status = self._blank()
        self.status.update(phase="training", kind=kind, total=int(timesteps), started_at=time.time(),
                           message="Building environments...")
        self.thread = threading.Thread(target=self._run, args=(kind, int(timesteps), engine), daemon=True)
        self.thread.start()
        return {"status": "started", "kind": kind, "timesteps": int(timesteps)}

    def stop(self) -> None:
        self.stop_flag = True

    def _run(self, kind: str, total: int, engine) -> None:
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv

            torch.set_num_threads(2)
            n_envs = 6
            venv = DummyVecEnv([(lambda s=i: Monitor(BalanceEnv(kind, seed=100 + s))) for i in range(n_envs)])
            model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, learning_rate=3e-4, gamma=0.98,
                        gae_lambda=0.95, ent_coef=0.0, clip_range=0.2, device="cpu", verbose=0,
                        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.2))
            trainer = self
            engine.balance_ghost_joints = list(venv.envs[0].unwrapped.names)
            last_pub = [0.0]

            class Progress(BaseCallback):
                def _on_step(self) -> bool:
                    if trainer.stop_flag:
                        return False
                    now = time.time()
                    if now - last_pub[0] > 0.066:        # ~15 Hz: live poses of the real PPO training envs
                        last_pub[0] = now
                        ghosts = []
                        for e in venv.envs:
                            b = e.unwrapped
                            ghosts.append([float(v) for v in b.data.qpos[:7]] + [1.0 if b.push_left > 0 else 0.0]
                                          + [float(b.data.qpos[a]) for a in b.qadr])
                        engine.balance_ghosts = ghosts
                    t = self.num_timesteps
                    if t % (256 * n_envs) < n_envs:
                        s = 0.35 + 0.65 * min(1.0, t / (0.6 * total))     # push-strength curriculum
                        self.training_env.env_method("set_push_scale", s)
                        buf = self.model.ep_info_buffer
                        if len(buf) > 0:
                            mr = float(np.mean([e["r"] for e in buf]))
                            ml = float(np.mean([e["l"] for e in buf]))
                            st = trainer.status
                            st.update(mean_reward=round(mr, 2), mean_length=round(ml, 1))
                            st["history"].append([int(t), round(mr, 2)])
                            st["history"] = st["history"][-80:]
                        trainer.status.update(timesteps=int(t), progress=round(min(1.0, t / total), 4),
                                              message=f"Training PPO ({int(t):,} / {total:,} steps)")
                    return True

            model.learn(total_timesteps=total, callback=Progress())
            engine.balance_ghosts = []
            venv.close()
            stopped = self.stop_flag

            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"{kind}_balance.zip")
            model.save(path)
            self.status.update(phase="evaluating", message="Measuring survival under a firm push...", model_path=path)
            before = evaluate(kind, None)
            after = evaluate(kind, model)
            self.status["eval"] = {"untrained": before, "trained": after, "push_newtons": CFG[kind]["demo_push"],
                                   "trials": 24}
            # Deploy the learned reflex onto the live robot (only if it is still the same kind of robot).
            gr = engine.project.generic_robot
            live_kind = "humanoid" if gr and any(l.name == "pelvis" for l in gr.links) else (
                "dog" if gr and any(l.name == "fl_thigh" for l in gr.links) else None)
            if live_kind == kind:
                engine.balance_controller = BalanceController(model, kind)
            self.status.update(phase="done", progress=1.0, finished_at=time.time(),
                               message=("Stopped early. " if stopped else "") + "Policy deployed to the live robot.")
        except Exception as e:  # noqa: BLE001
            engine.balance_ghosts = []
            logger.exception("Balance training failed")
            self.status.update(phase="error", message=f"Training failed: {e}")


trainer = BalanceTrainer()


# ---------------------------------------------------------------- service functions (REST + assistant tools)
def spawn(kind: str, pm, engine, clear: bool = True) -> Dict[str, Any]:
    robot = build_preset(kind, free=True)
    if robot is None:
        return {"status": "error", "details": f"Unknown robot '{kind}'."}
    proj = pm.get_current()
    if clear:                                   # the balance lab is a clean stage: no rover, no props
        proj.robot, proj.robot_active, proj.objects = None, False, []
    proj.generic_robot = robot
    pm.update_project(proj)
    engine.project = proj
    engine.balance_controller = None            # a fresh robot has no balance reflex
    engine._push_left = 0
    engine.rebuild_model(preserve_state=False)
    engine.play()
    return {"status": "success", "action": "spawn_balance_robot", "kind": kind, "name": robot.name, "trained": False,
            "details": f"Spawned an untrained, free-standing {robot.name}. Push it, then train it to stay upright."}


def push(engine, angle_deg: Optional[float] = None, strength: Optional[float] = None) -> Dict[str, Any]:
    gr = engine.project.generic_robot
    if gr is None or gr.is_fixed_base:
        return {"status": "error", "details": "Spawn a free-standing robot first."}
    kind = "humanoid" if any(l.name == "pelvis" for l in gr.links) else "dog"
    mag = float(strength) if strength else CFG[kind]["demo_push"]
    ang = math.radians(float(angle_deg)) if angle_deg is not None else float(np.random.uniform(0, 2 * math.pi))
    engine.push_root([math.cos(ang) * mag, math.sin(ang) * mag, 0.0], PUSH_SECONDS)
    return {"status": "success", "action": "push_robot", "newtons": mag, "angle_deg": round(math.degrees(ang), 1),
            "controller": engine.balance_controller is not None,
            "details": f"Pushed with {mag:.0f} N for {PUSH_SECONDS:.1f} s ({'trained policy active' if engine.balance_controller else 'untrained'})."}


def untrain(engine) -> Dict[str, Any]:
    engine.balance_controller = None
    engine.joint_targets.clear()
    return {"status": "success", "details": "Balance policy removed; the robot is untrained again."}


def get_status(engine) -> Dict[str, Any]:
    s = dict(trainer.status)
    s["controller_active"] = getattr(engine, "balance_controller", None) is not None
    gr = engine.project.generic_robot
    s["robot"] = gr.name if gr is not None else None
    s["free_standing"] = bool(gr is not None and not gr.is_fixed_base)
    return s
