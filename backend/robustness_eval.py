"""Robustness scorecard: evaluate a trained policy across many DOMAIN-RANDOMIZED worlds.

Runs the policy over N episodes with randomized friction, mass, gravity and sensor noise, and
reports research-grade reliability metrics (success rate, collisions, time-to-goal, path error).
Optionally compares a baseline (clean-physics) run so you can show the robustness gain.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("roboforge.robustness")


def _run_episodes(model, project, target, n: int, randomize: bool, max_steps: int, seed0: int) -> Dict[str, Any]:
    from backend.rl_env import RoboForgeGymEnv
    env = RoboForgeGymEnv(
        project=project, target_position=tuple(target), randomize_training=False,
        distance_threshold=0.9, max_episode_steps=max_steps, domain_randomize=randomize,
    )
    successes = 0
    collisions = 0
    times: List[float] = []
    final_dists: List[float] = []
    rewards: List[float] = []
    try:
        for i in range(n):
            obs, _ = env.reset(seed=seed0 + i)
            ep_r = 0.0
            done = False
            steps = 0
            crashed = False
            reached = False
            while not done and steps < max_steps:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                ep_r += float(r)
                steps += 1
                if info.get("is_collision"):
                    crashed = True
                if info.get("is_success"):
                    reached = True
                done = term or trunc
            times.append(steps * (1.0 / env.control_frequency_hz))
            final_dists.append(float(info.get("distance_to_goal", 9.9)))
            rewards.append(ep_r)
            if reached:
                successes += 1
            if crashed:
                collisions += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    succ_times = [t for t, d in zip(times, final_dists) if d <= 0.9]
    return {
        "episodes": n,
        "success_rate": round(successes / max(1, n), 3),
        "collision_rate": round(collisions / max(1, n), 3),
        "mean_time_to_goal": round(float(np.mean(succ_times)), 2) if succ_times else None,
        "mean_final_distance": round(float(np.mean(final_dists)), 3),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "reward_std": round(float(np.std(rewards)), 2),
    }


def evaluate_robustness(model_id: str, project, episodes: int = 20, compare_baseline: bool = True,
                        max_steps: int = 300, seed: int = 500) -> Dict[str, Any]:
    from backend.rl_training_manager import training_manager
    model = training_manager.load_model(model_id)
    if model is None:
        return {"status": "error", "detail": f"Policy '{model_id}' not found."}
    meta = training_manager.get_model_metadata(model_id)
    target = (meta.target_position if meta and getattr(meta, "target_position", None) else [2.5, 0.5, 0.25])
    if len(target) < 3:
        target = [target[0], target[1], 0.25]

    t0 = time.time()
    randomized = _run_episodes(model, project, target, episodes, True, max_steps, seed)
    baseline = _run_episodes(model, project, target, episodes, False, max_steps, seed + 9000) if compare_baseline else None

    # A simple robustness score: how well it holds up under randomization (0-100).
    score = round(100.0 * randomized["success_rate"] * (1.0 - randomized["collision_rate"]), 1)
    verdict = ("Robust — holds up well under randomized physics & noise." if score >= 70
               else "Moderately robust — degrades under randomization." if score >= 40
               else "Fragile — overfit to clean simulation; needs domain-randomized training.")
    retention = None
    if baseline and baseline["success_rate"] > 0:
        retention = round(100.0 * randomized["success_rate"] / baseline["success_rate"], 0)

    return {
        "status": "success",
        "model_id": model_id,
        "model_name": getattr(meta, "name", model_id) if meta else model_id,
        "episodes": episodes,
        "target_position": target,
        "randomized": randomized,
        "baseline_clean": baseline,
        "robustness_score": score,
        "success_retention_pct": retention,  # randomized success as % of clean success
        "verdict": verdict,
        "randomization": {
            "friction": "60–160% of nominal",
            "mass": "80–130% per body",
            "gravity": "±3%",
            "sensor_noise_std": "0.01–0.06 (per episode)",
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
