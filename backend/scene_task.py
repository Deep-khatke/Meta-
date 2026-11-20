"""Train the live robot on a task in the scene built from a photo (real PPO on the same MuJoCo model).

Tasks
  reach            fixed-base arm: move the arm's end (last joint frame) to the target object. Success < 6 cm.
  go_to / climb    free-standing legged robot: walk to a target (for stairs, the top landing) and stay upright.

The training environments are copies of the live project (same robot, same stairs/objects), PPO runs in a
background thread, the live poses of the parallel environments are streamed as ghosts, and the learned policy is
deployed on the live robot (joint targets at 50 Hz). Nothing is scripted: if the policy does not learn the task, the
measured success rate says so.
"""
import logging
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

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
from backend import scene_vision

logger = logging.getLogger("roboforge.scene_task")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scene_tasks")
CTRL_DT = 0.02
LEG_SCALE = 0.5


def make_ctx(model, gr, goal, task, engine=None) -> SimpleNamespace:
    joints = [j for j in gr.joints if j.type != "fixed"]
    names = [j.name for j in joints]
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    parents = {j.parent_link for j in gr.joints}
    leaves = [l.name for l in gr.links if l.name not in parents] or [gr.links[-1].name]
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    return SimpleNamespace(
        free=not gr.is_fixed_base, task=task, names=names, n=len(names), goal=np.asarray(goal, dtype=float),
        qadr=np.array([model.jnt_qposadr[i] for i in jid]), dadr=np.array([model.jnt_dofadr[i] for i in jid]),
        lo=np.array([j.limit.lower for j in joints]), hi=np.array([j.limit.upper for j in joints]),
        root=bid(gr.root_link), ee=bid(leaves[-1]), delta=(task == "imitation"),
        act=np.array([engine.joint_actuator_ids[n] for n in names]) if engine is not None else None)


DELTA = 0.25  # rad per 50 Hz tick for relative (imitation) control


def to_targets(c, a: np.ndarray, q=None) -> np.ndarray:
    a = np.clip(a, -1.0, 1.0)
    if c.delta and q is not None:
        return np.clip(q + a * DELTA, c.lo, c.hi)
    if c.free:
        return np.clip(a * LEG_SCALE, c.lo, c.hi)
    return c.lo + (a + 1.0) * 0.5 * (c.hi - c.lo)


def task_obs(model, data, c, last_a) -> np.ndarray:
    qj, qd = data.qpos[c.qadr], data.qvel[c.dadr] * 0.05
    if c.free:
        R = data.xmat[c.root].reshape(3, 3)
        rel = np.array([c.goal[0], 0.0, 0.0]) if c.task == "walk" else np.clip(R.T @ (c.goal - data.xpos[c.root]), -4.0, 4.0)
        return np.concatenate([R.T @ [0, 0, -1.0], data.qvel[3:6] * 0.25, R.T @ data.qvel[0:3], rel, qj, qd, last_a]).astype(np.float32)
    ee = data.xpos[c.ee]
    err = (c.goal - ee) * (10.0 if c.delta else 1.0)      # imitation policy: amplify the tip error for precision
    return np.concatenate([err, ee, qj, qd, last_a]).astype(np.float32)


class SceneTaskEnv(gym.Env):
    metadata: Dict[str, Any] = {}

    def __init__(self, project, task: str, goal, seed: int = 0, episode_seconds: Optional[float] = None):
        super().__init__()
        from backend.simulation import MuJoCoEngine
        p = project.model_copy(deep=True)
        p.robot, p.robot_active = None, False
        for o in p.objects:
            o.is_static = True
        self.engine = MuJoCoEngine(p)
        self.model, self.data = self.engine.model, self.engine.data
        self.c = make_ctx(self.model, self.engine.project.generic_robot, goal, task, self.engine)
        self.goal0 = np.asarray(goal, dtype=float)
        self.n = self.c.n
        self.sub = max(1, int(round(CTRL_DT / float(self.model.opt.timestep))))
        eps = episode_seconds or (4.0 if not self.c.free else (10.0 if task == "walk" else 8.0))
        self.max_steps = int(eps / (self.sub * float(self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self._obs_dummy()),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self.last_a = np.zeros(self.n)
        self.prev = 0.0
        self.t = 0
        self.z0 = 0.4
        self.x0 = 0.0
        self.fall_h = 0.2

    def _obs_dummy(self):
        mujoco.mj_forward(self.model, self.data)
        return task_obs(self.model, self.data, self.c, np.zeros(self.n))

    def _dist(self) -> float:
        d = self.data
        if self.c.free:
            return float(np.linalg.norm(self.c.goal - d.xpos[self.c.root]))
        return float(np.linalg.norm(self.c.goal - d.xpos[self.c.ee]))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d, c = self.model, self.data, self.c
        mujoco.mj_resetData(m, d)
        if not c.free:
            d.qpos[c.qadr] = c.lo + self.rng.uniform(0.2, 0.8, c.n) * (c.hi - c.lo)
            self.c.goal = self.goal0 + self.rng.uniform(-0.02, 0.02, 3)
        mujoco.mj_forward(m, d)
        for _ in range(self.sub * 2):
            d.ctrl[c.act] = to_targets(c, np.zeros(c.n)) if c.free else d.qpos[c.qadr]
            mujoco.mj_step(m, d)
        if c.free:
            self.z0 = float(d.xpos[c.root][2])
            self.fall_h = 0.6 * self.z0
        self.last_a = np.zeros(c.n)
        self.t = 0
        self.prev = self._dist()
        self.x0 = float(d.xpos[c.root][0])
        return task_obs(m, d, c, self.last_a), {}

    def step(self, action):
        c, m, d = self.c, self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        d.ctrl[c.act] = to_targets(c, a, d.qpos[c.qadr])
        for _ in range(self.sub):
            mujoco.mj_step(m, d)
        self.t += 1
        dist = self._dist()
        if c.task == "walk":
            dist = self.prev
        r = 6.0 * (self.prev - dist) - 0.01 * float(np.mean((a - self.last_a) ** 2))
        self.prev, self.last_a = dist, a
        term, success = False, False
        if c.free:
            R = d.xmat[c.root].reshape(3, 3)
            up = float((R.T @ [0, 0, -1.0])[2] * -1)
            h = float(d.xpos[c.root][2])
            r += 0.05 + 0.1 * up - 0.002 * float(d.qvel[c.dadr] @ d.qvel[c.dadr]) * 0.01
            if c.task == "walk":
                vb = R.T @ d.qvel[0:3]
                cmd = c.goal[0]
                r += (0.5 + 1.5 * math.exp(-((vb[0] - cmd) / 0.15) ** 2) + 0.3 * up - 0.5 * vb[1] ** 2
                      - 0.1 * float(d.qvel[5]) ** 2 - 0.3 * abs(h - self.z0))
            if h < self.fall_h or up < 0.4 or np.isnan(h):
                r -= 30.0 if c.task == "walk" else 5.0
                term = True
            elif c.task == "walk":
                pass
            # at the goal: horizontal distance small and body is at (or above) the goal height
            elif c.task != "walk" and np.linalg.norm(c.goal[:2] - d.xpos[c.root][:2]) < 0.35 and h > c.goal[2] - 0.08:
                r += 20.0
                term = success = True
        elif c.delta:                       # imitation: no early stop; must reach AND hold, judged at 3 cm
            r -= 0.02 * dist
            success = dist < 0.03
        else:
            r -= 0.02 * dist
            if dist < 0.06:
                r += 10.0
                term = success = True
        trunc = self.t >= self.max_steps
        if c.task == "walk" and (trunc or term):
            self.speed = (float(d.xpos[c.root][0]) - self.x0) / (self.t * self.sub * float(m.opt.timestep))
            success = (not term) and self.speed >= 0.5 * c.goal[0]
        return task_obs(m, d, c, a), float(r), term, trunc, {"success": success}


def evaluate(env: SceneTaskEnv, policy, trials: int = 20) -> Dict[str, float]:
    ok, best, spd = 0, [], []
    for i in range(trials):
        obs, _ = env.reset(seed=1000 + i)
        done, success = False, False
        while not done:
            act = np.zeros(env.n) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, _, te, tr, info = env.step(act)
            success, done = info["success"], te or tr
        ok += int(success)
        best.append(env._dist())
        spd.append(getattr(env, "speed", 0.0))
    out = {"success_rate": ok / trials, "final_distance_m": float(np.mean(best))}
    if env.c.task == "walk":
        out["mean_speed_mps"] = float(np.mean(spd))
    return out


class TaskController:
    """Runs the trained policy on the live engine (works for fixed-base arms and free-standing robots)."""

    def __init__(self, ppo, goal, task):
        self.ppo, self.goal, self.task = ppo, goal, task
        self._k = None
        self.last_a = None

    @staticmethod
    def substeps_for(engine) -> int:
        return max(1, int(round(CTRL_DT / engine.project.settings.timestep)))

    def tick(self, engine) -> None:
        if self._k != id(engine.model):
            self.c = make_ctx(engine.model, engine.project.generic_robot, self.goal, self.task)
            self.last_a, self._k = np.zeros(self.c.n), id(engine.model)
        obs = task_obs(engine.model, engine.data, self.c, self.last_a)
        a = np.clip(self.ppo.predict(obs, deterministic=True)[0], -1, 1)
        self.last_a = a
        for nm, v in zip(self.c.names, to_targets(self.c, a, engine.data.qpos[self.c.qadr])):
            engine.joint_targets[nm] = float(v)


# ------------------------------------------------------------------ target + robot resolution
def resolve_target(text: Optional[str]) -> tuple:
    """Find a scene target by label (fuzzy). Returns (label, [x, y, z]) or raises ValueError."""
    t = scene_vision.CTX["targets"]
    if not scene_vision.CTX["active"] or not t:
        raise ValueError("No scene yet. Upload a photo of the environment first.")
    q = (text or "").lower().strip()
    if not q:
        st = scene_vision.CTX.get("stairs")
        k = st["label"].lower() if st else next(iter(t))
        return k, t[k]
    if q in t:
        return q, t[q]
    words = set(q.replace("the ", "").split())
    best = max(t, key=lambda k: len(words & set(k.split())) + (2 if (k in q or q in k) else 0))
    if len(words & set(best.split())) == 0 and best not in q and q not in best:
        if "stair" in q or "step" in q:
            st = scene_vision.CTX.get("stairs")
            if st:
                return st["label"].lower(), t[st["label"].lower()]
        raise ValueError(f"I can't find '{text}' in the scene. Known objects: {', '.join(t)}.")
    return best, t[best]


def ensure_robot(task: str, pm, engine) -> str:
    """Make sure a suitable robot is in the scene (without touching the photo's objects)."""
    proj = pm.get_current()
    gr = proj.generic_robot
    if task == "reach":
        if gr is None or not gr.is_fixed_base or any(l.name in ("pelvis", "fl_thigh") for l in gr.links):
            from backend.ai_reasoning_service import get_ai_reasoning_service
            get_ai_reasoning_service(pm, engine=engine).tool_spawn_arm(joints=4)
            proj = pm.get_current()
    else:
        if gr is None or gr.is_fixed_base:
            proj.generic_robot = build_preset("dog", free=True)
    proj.robot, proj.robot_active = None, False
    pm.update_project(proj)
    engine.project = proj
    engine.balance_controller = None
    engine.rebuild_model(preserve_state=False)
    engine.play()
    return proj.generic_robot.name


# ------------------------------------------------------------------ trainer
class SceneTrainer:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = False
        self.status: Dict[str, Any] = {"phase": "idle"}

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self, task: str, target: Optional[str], timesteps: int, pm, engine) -> Dict[str, Any]:
        if self.running():
            return {"status": "busy", "details": "A scene training run is already in progress."}
        tl = task.lower()
        if any(k in tl for k in ("reach", "pick", "grab", "grasp", "touch", "lift", "arm")):
            task = "reach"
        elif any(k in tl for k in ("climb", "stair", "navigate", "step")) or (target and any(k in tl for k in ("go", "walk", "run"))):
            task = "go_to"
        elif any(k in tl for k in ("walk", "run", "trot", "gait", "go", "forward")):
            task = "walk"
        if task == "go_to" and not target and not scene_vision.CTX["active"]:
            task = "walk"
        if task not in ("reach", "go_to", "walk"):
            return {"status": "error", "details": "task must be 'walk', 'climb', 'go_to' or 'reach'."}
        if task == "walk":
            label, pos = "forward", [0.0, 0.0, 0.0]
        else:
            try:
                label, pos = resolve_target(target)
            except ValueError as e:
                return {"status": "error", "details": str(e)}
        robot = ensure_robot(task, pm, engine)
        proj = engine.project
        goal = list(pos)
        if task == "walk":
            humanoid = any(l.name == "pelvis" for l in proj.generic_robot.links)
            goal = [0.2 if humanoid else 0.3, 0.0, 0.0]   # commanded forward speed, m/s
            probe = None
        elif task == "go_to":                                   # stand ON the target: add the robot's standing height
            probe = SceneTaskEnv(proj, task, goal)
            probe.reset()
            goal[2] = float(pos[2]) + probe.z0
            probe.goal0 = probe.c.goal = np.array(goal)
        else:
            probe = SceneTaskEnv(proj, task, goal)
            probe.reset()
            rng, near = np.random.default_rng(0), 9.9
            for _ in range(20000):
                probe.data.qpos[probe.c.qadr] = probe.c.lo + rng.uniform(0, 1, probe.c.n) * (probe.c.hi - probe.c.lo)
                mujoco.mj_forward(probe.model, probe.data)
                near = min(near, float(np.linalg.norm(probe.data.xpos[probe.c.ee] - np.asarray(goal))))
            if near > 0.10:
                return {"status": "error", "details": f"'{label}' is out of the {robot}'s reach (closest the arm can get is "
                                                        f"{near:.2f} m). Move the object closer or use a longer arm."}
        self.stop_flag = False
        self.status = {"phase": "training", "task": task, "target": label, "goal": [round(g, 3) for g in goal], "robot": robot,
                       "progress": 0.0, "timesteps": 0, "total": int(timesteps), "mean_reward": None, "history": [],
                       "eval": None, "message": "Building environments...", "started_at": time.time()}
        self.thread = threading.Thread(target=self._run, args=(task, goal, int(timesteps), proj, engine), daemon=True)
        self.thread.start()
        return {"status": "started", "task": task, "target": label, "robot": robot, "timesteps": int(timesteps),
                "details": f"Training {robot} to {'reach' if task == 'reach' else ('walk forward at ' + str(goal[0]) + ' m/s' if task == 'walk' else 'walk to ' + chr(39) + label + chr(39))} with real PPO. "
                           "Watch the ghosts: they are the parallel training environments."}

    def stop(self) -> None:
        self.stop_flag = True

    def _run(self, task, goal, total, proj, engine) -> None:
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
            torch.set_num_threads(2)
            venv = DummyVecEnv([(lambda s=i: Monitor(SceneTaskEnv(proj, task, goal, seed=50 + s))) for i in range(6)])
            model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, learning_rate=3e-4, gamma=0.98, gae_lambda=0.95,
                        device="cpu", verbose=0, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0))
            engine.balance_ghost_joints = list(venv.envs[0].unwrapped.c.names)
            me, last = self, [0.0]

            class CB(BaseCallback):
                def _on_step(self) -> bool:
                    if me.stop_flag:
                        return False
                    now = time.time()
                    if now - last[0] > 0.066:
                        last[0] = now
                        gh = []
                        for e in venv.envs:
                            b = e.unwrapped
                            pose = [float(v) for v in b.data.qpos[:7]] if b.c.free else [0, 0, 0, 1, 0, 0, 0]
                            gh.append(pose + [0.0] + [float(b.data.qpos[x]) for x in b.c.qadr])
                        engine.balance_ghosts = gh
                    t = self.num_timesteps
                    if t % 1536 < 6:
                        buf = self.model.ep_info_buffer
                        if len(buf):
                            mr = round(float(np.mean([x["r"] for x in buf])), 2)
                            me.status["mean_reward"] = mr
                            me.status["history"] = (me.status["history"] + [[int(t), mr]])[-80:]
                        me.status.update(timesteps=int(t), progress=round(min(1.0, t / total), 4),
                                         message=f"Training PPO ({int(t):,} / {total:,} steps)")
                    return True

            model.learn(total_timesteps=total, callback=CB())
            engine.balance_ghosts = []
            venv.close()
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.status.update(phase="evaluating", message="Measuring success rate...")
            env = SceneTaskEnv(proj, task, goal, seed=7)
            res = {"untrained": evaluate(env, None), "trained": evaluate(env, model)}
            model.save(os.path.join(MODEL_DIR, f"{task}_{self.status['target'].replace(' ', '_')}.zip"))
            engine.balance_controller = TaskController(model, np.array(goal), task)
            self.status.update(phase="done", progress=1.0, eval=res, finished_at=time.time(),
                               message=("Stopped early. " if self.stop_flag else "") + "Policy deployed to the live robot.")
        except Exception as e:  # noqa: BLE001
            engine.balance_ghosts = []
            logger.exception("Scene training failed")
            self.status.update(phase="error", message=f"Training failed: {e}")


trainer = SceneTrainer()
