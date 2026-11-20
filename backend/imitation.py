"""Imitation learning for the robot arm: demonstrations -> behaviour cloning -> DAgger -> optional PPO fine-tune.

Demonstrations come from two real sources, stored in one format (observation, action) at 50 Hz:
  * the scripted expert: damped-least-squares inverse kinematics on the same MuJoCo model, which
    computes the joint targets that put the arm's tip on a goal point;
  * live recording: the user moves the arm in the studio (joint sliders, gizmo, motions) and every
    50 Hz tick of the real simulation is recorded; the goal is where the tip ended up.

Training clones the demos into an SB3 PPO actor (so PPO can fine-tune the exact same network), optionally
runs DAgger (the policy drives, the expert labels what it should have done, the data is aggregated and the
policy retrained), then evaluates the success rate (tip within 6 cm of a new random goal) for every stage.
"""
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from backend.scene_task import DELTA, SceneTaskEnv, TaskController, evaluate, make_ctx, task_obs

logger = logging.getLogger("roboforge.imitation")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "imitation", "arm_reach_bc.zip")


# ------------------------------------------------------------------ environment with random reachable goals
class ReachEnv(SceneTaskEnv):
    def __init__(self, project, seed: int = 0):
        super().__init__(project, "imitation", [0.3, 0.0, 0.3], seed=seed, episode_seconds=3.0)
        self.scratch = mujoco.MjData(self.model)

    def sample_goal(self) -> np.ndarray:
        c = self.c
        for _ in range(200):
            self.scratch.qpos[c.qadr] = c.lo + self.rng.uniform(0, 1, c.n) * (c.hi - c.lo)
            mujoco.mj_kinematics(self.model, self.scratch)
            p = self.scratch.xpos[c.ee].copy()
            if p[2] > 0.08 and np.linalg.norm(p[:2]) > 0.12:
                return p
        return p

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self.c.goal = self.sample_goal()
        self.prev = self._dist()
        return task_obs(self.model, self.data, self.c, self.last_a), info


def ik(model, scratch, c, q_now: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Damped-least-squares IK from the current joint angles; returns the joint angles that reach the goal."""
    q = np.array(q_now, dtype=float)
    jacp = np.zeros((3, model.nv))
    for _ in range(120):
        scratch.qpos[c.qadr] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        err = goal - scratch.xpos[c.ee]
        if np.linalg.norm(err) < 0.004:
            break
        mujoco.mj_jacBody(model, scratch, jacp, None, c.ee)
        J = jacp[:, c.dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        q = np.clip(q + np.clip(dq, -0.25, 0.25), c.lo, c.hi)
    return q


EXPERT_GAIN = 3.0  # over-commands the joint error so the position servos beat gravity sag (measured: 27% -> 97.5% holds)


def delta_action(q_star: np.ndarray, q_now: np.ndarray, gain: float = EXPERT_GAIN) -> np.ndarray:
    return np.clip(gain * (q_star - q_now) / DELTA, -1.0, 1.0)


def expert_action(model, scratch, c, q_now, goal) -> np.ndarray:
    return delta_action(ik(model, scratch, c, q_now, goal), np.asarray(q_now))


def expert_demos(env: ReachEnv, n: int) -> Dict[str, Any]:
    O, A, ok = [], [], 0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            a = expert_action(env.model, env.scratch, env.c, env.data.qpos[env.c.qadr], env.c.goal)
            O.append(obs)
            A.append(a)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        ok += int(info["success"])
    return {"obs": O, "act": A, "success": ok}


# ------------------------------------------------------------------ live recorder (user demonstrates in the studio)
class LiveRecorder:
    def __init__(self, engine):
        gr = engine.project.generic_robot
        if gr is None or not gr.is_fixed_base:
            raise ValueError("Spawn the robot arm first.")
        self.c = make_ctx(engine.model, gr, [0, 0, 0], "imitation")
        self.rows: List[tuple] = []
        self.model_id = id(engine.model)

    @staticmethod
    def substeps_for(engine) -> int:
        return max(1, int(round(0.02 / engine.project.settings.timestep)))

    def tick(self, engine) -> None:
        if id(engine.model) != self.model_id:
            return
        d, c = engine.data, self.c
        tgt = np.array([engine.joint_targets.get(nm, d.qpos[a]) for nm, a in zip(c.names, c.qadr)])
        self.rows.append((d.qpos[c.qadr].copy(), d.qvel[c.dadr].copy(), d.xpos[c.ee].copy(), tgt))

    def to_demo(self) -> Dict[str, Any]:
        c = self.c
        if len(self.rows) < 10:
            return {"obs": [], "act": []}
        goal = self.rows[-1][2]
        moved = np.linalg.norm(self.rows[0][2] - goal)
        if moved < 0.03:
            return {"obs": [], "act": [], "moved": float(moved)}
        O, A, last = [], [], np.zeros(c.n)
        for q, qd, ee, tgt in self.rows:
            a = delta_action(tgt, q, gain=1.0)
            O.append(np.concatenate([(goal - ee) * 10.0, ee, q, qd * 0.05, last]).astype(np.float32))
            A.append(a)
            last = a
        return {"obs": O, "act": A, "moved": float(moved)}


# ------------------------------------------------------------------ the lab
def _arm_project(engine):
    p = engine.project.model_copy(deep=True)
    p.objects, p.robot, p.robot_active = [], None, False
    return p


class ImitationLab:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.act: List[np.ndarray] = []
        self.sources = {"expert": 0, "user": 0}
        self.expert_success = None
        self.recorder: Optional[LiveRecorder] = None
        self.thread: Optional[threading.Thread] = None
        self.status: Dict[str, Any] = {"phase": "idle"}
        self.model = None
        self.goal: Optional[List[float]] = None

    def summary(self, engine=None) -> Dict[str, Any]:
        return {"samples": len(self.obs), "demos": dict(self.sources), "recording": self.recorder is not None,
                "expert_success": self.expert_success, "training": dict(self.status),
                "deployed": bool(engine is not None and isinstance(getattr(engine, "balance_controller", None), TaskController)
                                 and engine.balance_controller.task == "imitation"),
                "goal": self.goal, "tip_distance_m": self._tip(engine)}

    def _tip(self, engine):
        ctrl = getattr(engine, "balance_controller", None) if engine is not None else None
        if not isinstance(ctrl, TaskController) or ctrl.task != "imitation" or getattr(ctrl, "c", None) is None or self.goal is None:
            return None
        return round(float(np.linalg.norm(engine.data.xpos[ctrl.c.ee] - np.asarray(self.goal))), 4)

    def ensure_arm(self, pm, engine) -> None:
        gr = engine.project.generic_robot
        if gr is None or not gr.is_fixed_base or any(l.name in ("pelvis", "fl_thigh") for l in gr.links):
            from backend.scene_task import ensure_robot
            ensure_robot("reach", pm, engine)

    def add_expert(self, n: int, pm, engine) -> Dict[str, Any]:
        self.ensure_arm(pm, engine)
        env = ReachEnv(_arm_project(engine), seed=int(time.time()) % 10000)
        d = expert_demos(env, max(1, min(500, int(n))))
        self.obs += d["obs"]
        self.act += d["act"]
        self.sources["expert"] += int(n)
        self.expert_success = round(d["success"] / n, 3)
        return {"status": "success", "added_demos": int(n), "samples": len(self.obs), "expert_success_rate": self.expert_success,
                "details": f"Recorded {n} scripted-expert demonstrations ({len(d['obs'])} samples); the expert reached the goal in {d['success']}/{n}."}

    def start_recording(self, pm, engine) -> Dict[str, Any]:
        self.ensure_arm(pm, engine)
        engine.balance_controller = None
        self.recorder = LiveRecorder(engine)
        engine.demo_recorder = self.recorder
        engine.play()
        return {"status": "success", "details": "Recording. Move the arm's tip to a point (joint sliders or gizmo), then stop."}

    def stop_recording(self, engine) -> Dict[str, Any]:
        rec, self.recorder = self.recorder, None
        engine.demo_recorder = None
        if rec is None:
            return {"status": "error", "details": "Not recording."}
        d = rec.to_demo()
        if not d["obs"]:
            return {"status": "error", "details": "The arm's tip barely moved, so the demo was discarded. Move it at least 3 cm."}
        self.obs += d["obs"]
        self.act += d["act"]
        self.sources["user"] += 1
        return {"status": "success", "samples_added": len(d["obs"]), "samples": len(self.obs),
                "details": f"Saved your demonstration ({len(d['obs'])} samples, tip moved {d['moved'] * 100:.0f} cm)."}

    def clear(self) -> Dict[str, Any]:
        self.obs, self.act, self.sources, self.expert_success = [], [], {"expert": 0, "user": 0}, None
        return {"status": "success", "details": "All demonstrations deleted."}

    # --- training
    def train(self, pm, engine, dagger_rounds: int = 3, finetune_steps: int = 0) -> Dict[str, Any]:
        if self.thread and self.thread.is_alive():
            return {"status": "busy", "details": "Imitation training is already running."}
        if len(self.obs) < 200:
            return {"status": "error", "details": "Not enough demonstrations. Record some first (e.g. 40 expert demos)."}
        self.ensure_arm(pm, engine)
        self.status = {"phase": "training", "stage": "behaviour cloning", "progress": 0.0, "results": {}, "message": "Starting...",
                       "dagger_rounds": int(dagger_rounds), "finetune_steps": int(finetune_steps)}
        self.thread = threading.Thread(target=self._run, args=(engine, int(dagger_rounds), int(finetune_steps)), daemon=True)
        self.thread.start()
        return {"status": "started", "details": f"Training from {len(self.obs)} demonstration samples "
                                                f"(behaviour cloning + {dagger_rounds} DAgger rounds"
                                                + (f" + {finetune_steps:,} PPO fine-tune steps" if finetune_steps else "") + ")."}

    def _bc(self, model, O, A, epochs: int, lr: float = 1e-3):
        import torch
        pol = model.policy
        o = torch.as_tensor(np.asarray(O), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(A), dtype=torch.float32)
        opt = torch.optim.Adam(pol.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        n = len(o)
        for ep in range(epochs):
            sched.step() if ep else None
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                mean = pol.get_distribution(o[idx]).distribution.mean
                loss = ((mean - a[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return float(loss.item())

    def _run(self, engine, rounds: int, ft: int) -> None:
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
            torch.set_num_threads(2)
            proj = _arm_project(engine)
            venv = DummyVecEnv([(lambda s=i: Monitor(ReachEnv(proj, seed=300 + s))) for i in range(4)])
            model = PPO("MlpPolicy", venv, n_steps=256, batch_size=256, learning_rate=1e-4, gamma=0.98, device="cpu",
                        verbose=0, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-2.5))
            ev = ReachEnv(proj, seed=777)
            st, res = self.status, self.status["results"]
            res["untrained"] = evaluate(ev, None, trials=40)
            O, A = list(self.obs), list(self.act)
            st["message"] = f"Behaviour cloning on {len(O)} samples..."
            loss = self._bc(model, O, A, epochs=60)
            res["behaviour_cloning"] = dict(evaluate(ev, model, trials=40), loss=round(loss, 5))
            st["progress"] = 0.3
            # DAgger: the policy drives; the IK expert labels every visited state; aggregate and retrain.
            dg = ReachEnv(proj, seed=900)
            for r in range(rounds):
                st.update(stage="DAgger", message=f"DAgger round {r + 1}/{rounds}: policy drives, expert labels...")
                for _ in range(40):
                    obs, _ = dg.reset()
                    done = False
                    while not done:
                        O.append(obs)
                        A.append(expert_action(dg.model, dg.scratch, dg.c, dg.data.qpos[dg.c.qadr], dg.c.goal))
                        obs, _, te, tr, _ = dg.step(model.predict(obs, deterministic=True)[0])
                        done = te or tr
                loss = self._bc(model, O, A, epochs=30)
                st["progress"] = 0.3 + 0.5 * (r + 1) / max(1, rounds)
            if rounds:
                st["message"] = f"Consolidating on {len(O)} aggregated samples..."
                loss = self._bc(model, O, A, epochs=80, lr=1e-3)
                res["dagger"] = dict(evaluate(ev, model, trials=40), loss=round(loss, 5), samples=len(O))
            if ft > 0:
                st.update(stage="PPO fine-tune", message=f"PPO fine-tuning for {ft:,} steps...")
                before = res.get("dagger", res["behaviour_cloning"])["success_rate"]
                snapshot = {k: v.clone() for k, v in model.policy.state_dict().items()}
                model.learn(total_timesteps=ft)
                res["ppo_finetune"] = evaluate(ev, model, trials=40)
                if res["ppo_finetune"]["success_rate"] < before:      # keep whichever is better, and say so
                    model.policy.load_state_dict(snapshot)
                    res["ppo_finetune"]["kept"] = False
                else:
                    res["ppo_finetune"]["kept"] = True
            venv.close()
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)
            self.model = model
            self.deploy(engine)
            st.update(phase="done", progress=1.0, stage="done", message="Imitation policy deployed to the live arm.")
        except Exception as e:  # noqa: BLE001
            logger.exception("Imitation training failed")
            self.status.update(phase="error", message=f"Training failed: {e}")

    # --- deployment
    def deploy(self, engine, goal: Optional[List[float]] = None) -> Dict[str, Any]:
        if self.model is None:
            if os.path.exists(MODEL_PATH):
                from stable_baselines3 import PPO
                self.model = PPO.load(MODEL_PATH, device="cpu")
            else:
                return {"status": "error", "details": "Train from demonstrations first."}
        if goal is None:
            ev = ReachEnv(_arm_project(engine), seed=int(time.time()) % 10000)
            goal = ev.sample_goal()
        self.goal = [float(v) for v in goal]
        ctrl = getattr(engine, "balance_controller", None)
        if isinstance(ctrl, TaskController) and ctrl.task == "imitation":
            ctrl.goal = np.array(self.goal)
            if getattr(ctrl, "c", None) is not None:
                ctrl.c.goal = np.array(self.goal)
        else:
            ctrl = TaskController(self.model, np.array(self.goal), "imitation")
            engine.balance_controller = ctrl
        engine.task_goal = self.goal
        engine.play()
        return {"status": "success", "goal": [round(v, 3) for v in self.goal],
                "details": f"Imitation policy is driving the arm to ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})."}


for _name, _act in (("add_expert", "record_demos"), ("start_recording", "record_demos"), ("stop_recording", "record_demos"),
                     ("clear", "record_demos"), ("train", "train_from_demos"), ("deploy", "imitation_goal")):
    def _wrap(f, a=_act):
        def g(*args, **kw):
            r = f(*args, **kw)
            if isinstance(r, dict):
                r.setdefault("action", a)
            return r
        return g
    setattr(ImitationLab, _name, _wrap(getattr(ImitationLab, _name)))

lab = ImitationLab()
