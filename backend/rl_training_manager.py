"""Background training manager for Reinforcement Learning (PPO) using Stable-Baselines3."""

import os
import json
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from backend.models import Project
from backend.rl_env import RoboForgeGymEnv
from backend.rl_models import (
    RLTrainingConfig, RLTrainingStatus, RLTrainingMetrics, RLModelMetadata
)
from backend.project_manager import get_default_project

logger = logging.getLogger("roboforge.rl")

MODELS_DIR = Path("models/rl")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = MODELS_DIR / "registry.json"


class RLProgressCallback(BaseCallback):
    """Callback to capture episode statistics and check for cancellation during training."""

    def __init__(self, manager: "RLTrainingManager", job_id: str, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.manager = manager
        self.job_id = job_id
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[bool] = []
        self.collisions: List[bool] = []

    def _on_step(self) -> bool:
        # Check cancellation
        if self.manager.is_cancelled(self.job_id):
            logger.info("Cancellation requested for job %s. Aborting PPO training loop.", self.job_id)
            return False

        # Check for episode termination info in env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info and ("termination_reason" in info or "is_collision" in info):
                if info.get("termination_reason") in ("goal_reached", "collision", "timeout"):
                    self.successes.append(bool(info.get("is_success", False)))
                    self.collisions.append(bool(info.get("is_collision", False)))

        # Update metrics every 128 steps
        if self.num_timesteps % 128 == 0 or self.num_timesteps >= self.total_timesteps:
            elapsed = max(0.1, time.time() - self.start_time)
            fps = float(self.num_timesteps / elapsed)

            recent_succ = self.successes[-20:] if self.successes else []
            recent_cols = self.collisions[-20:] if self.collisions else []

            succ_rate = float(sum(recent_succ) / len(recent_succ)) if recent_succ else 0.0
            col_rate = float(sum(recent_cols) / len(recent_cols)) if recent_cols else 0.0

            # SB3 ep_info_buffer
            ep_info_buf = self.model.ep_info_buffer
            mean_rew = float(np.mean([ep["r"] for ep in ep_info_buf])) if ep_info_buf else 0.0

            self.manager.update_metrics(
                job_id=self.job_id,
                timestep=self.num_timesteps,
                total_timesteps=self.total_timesteps,
                episodes=len(self.successes),
                mean_reward=mean_rew,
                success_rate=succ_rate,
                collision_rate=col_rate,
                elapsed_seconds=elapsed,
                fps=fps,
            )

        return True


class RLTrainingManager:
    """Manages background asynchronous PPO training jobs, checkpoints, and model registry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, RLTrainingStatus] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._loaded_models: Dict[str, PPO] = {}
        self._load_registry()

    def _load_registry(self):
        """Loads existing model metadata from disk so trained policies persist across restarts."""
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        meta = RLModelMetadata(**item)
                        # Create completed status entry
                        status = RLTrainingStatus(
                            job_id=meta.model_id,
                            model_name=meta.name,
                            status="completed",
                            checkpoint_path=meta.checkpoint_path,
                            created_at=meta.created_at,
                            updated_at=meta.created_at,
                            metrics=RLTrainingMetrics(
                                timestep=meta.timesteps_trained,
                                total_timesteps=meta.timesteps_trained,
                                percent_complete=100.0,
                                mean_reward=meta.mean_reward,
                                success_rate=meta.success_rate,
                                collision_rate=meta.collision_rate,
                                elapsed_seconds=meta.training_duration_seconds,
                            ),
                        )
                        self._jobs[meta.model_id] = status
            except Exception as e:
                logger.warning("Failed to load RL model registry: %s", e)

    def _save_registry(self):
        """Persists model metadata to disk."""
        models = self.list_models()
        try:
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in models], f, indent=2)
        except Exception as e:
            logger.error("Failed to write RL model registry: %s", e)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return self._cancel_flags.get(job_id, False)

    def update_metrics(
        self,
        job_id: str,
        timestep: int,
        total_timesteps: int,
        episodes: int,
        mean_reward: float,
        success_rate: float,
        collision_rate: float,
        elapsed_seconds: float,
        fps: float,
    ):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                pct = min(100.0, round((timestep / max(1, total_timesteps)) * 100.0, 1))
                job.metrics.timestep = timestep
                job.metrics.total_timesteps = total_timesteps
                job.metrics.percent_complete = pct
                job.metrics.episodes = episodes
                job.metrics.mean_reward = round(mean_reward, 2)
                job.metrics.success_rate = round(success_rate, 3)
                job.metrics.collision_rate = round(collision_rate, 3)
                job.metrics.elapsed_seconds = round(elapsed_seconds, 1)
                job.metrics.fps = round(fps, 1)
                job.updated_at = time.time()

    def start_training(self, config: RLTrainingConfig, project: Optional[Project] = None) -> RLTrainingStatus:
        job_id = config.job_id or f"rl_{uuid.uuid4().hex[:8]}"
        config.job_id = job_id

        status = RLTrainingStatus(
            job_id=job_id,
            model_name=config.model_name,
            status="queued",
            metrics=RLTrainingMetrics(total_timesteps=config.total_timesteps),
        )

        with self._lock:
            self._jobs[job_id] = status
            self._cancel_flags[job_id] = False

        thread = threading.Thread(
            target=self._run_training_worker,
            args=(config, project or get_default_project()),
            daemon=True,
            name=f"RLWorker-{job_id}",
        )
        self._threads[job_id] = thread
        thread.start()
        return status

    def _run_training_worker(self, config: RLTrainingConfig, project: Project):
        job_id = config.job_id or "unknown"
        logger.info("Starting RL training worker for job %s (%s steps)", job_id, config.total_timesteps)

        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "training"

        env = None
        try:
            # Create training environment
            env = RoboForgeGymEnv(
                project=project,
                target_position=tuple(config.target_position),  # type: ignore
                randomize_training=config.randomize_scenarios,
                seed=config.seed,
            )

            hp = config.hyperparameters
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=hp.learning_rate,
                n_steps=hp.n_steps,
                batch_size=hp.batch_size,
                n_epochs=hp.n_epochs,
                gamma=hp.gamma,
                gae_lambda=hp.gae_lambda,
                clip_range=hp.clip_range,
                ent_coef=hp.ent_coef,
                seed=config.seed,
                verbose=0,
            )

            callback = RLProgressCallback(
                manager=self,
                job_id=job_id,
                total_timesteps=config.total_timesteps,
            )

            model.learn(total_timesteps=config.total_timesteps, callback=callback)

            if self.is_cancelled(job_id):
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id].status = "cancelled"
                logger.info("Training job %s marked as cancelled.", job_id)
                return

            # Save model checkpoint
            checkpoint_path = str(MODELS_DIR / f"{job_id}.zip")
            model.save(checkpoint_path)
            self._loaded_models[job_id] = model

            # Build metadata
            elapsed = time.time() - callback.start_time
            recent_succ = callback.successes[-30:] if callback.successes else []
            recent_cols = callback.collisions[-30:] if callback.collisions else []
            final_succ_rate = float(sum(recent_succ) / len(recent_succ)) if recent_succ else 0.0
            final_col_rate = float(sum(recent_cols) / len(recent_cols)) if recent_cols else 0.0

            metadata = RLModelMetadata(
                model_id=job_id,
                name=config.model_name,
                checkpoint_path=checkpoint_path,
                timesteps_trained=config.total_timesteps,
                training_duration_seconds=round(elapsed, 2),
                success_rate=round(final_succ_rate, 3),
                collision_rate=round(final_col_rate, 3),
                mean_reward=self._jobs[job_id].metrics.mean_reward,
                hyperparameters=config.hyperparameters.model_dump(),
                scenario_id=config.scenario_id,
            )

            meta_file = MODELS_DIR / f"{job_id}.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2)

            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].status = "completed"
                    self._jobs[job_id].checkpoint_path = checkpoint_path
                    self._jobs[job_id].metrics.percent_complete = 100.0

            self._save_registry()
            logger.info("RL training for %s completed successfully! Saved to %s", job_id, checkpoint_path)

        except Exception as e:
            logger.exception("Error during RL training job %s: %s", job_id, e)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].status = "failed"
                    self._jobs[job_id].error = str(e)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def cancel_training(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs and self._jobs[job_id].status in ("queued", "training"):
                self._cancel_flags[job_id] = True
                self._jobs[job_id].status = "cancelled"
                return True
            return False

    def get_status(self, job_id: str) -> Optional[RLTrainingStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[RLTrainingStatus]:
        with self._lock:
            return list(self._jobs.values())

    def list_models(self) -> List[RLModelMetadata]:
        models = []
        for meta_file in MODELS_DIR.glob("*.json"):
            if meta_file.name == "registry.json":
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    models.append(RLModelMetadata(**json.load(f)))
            except Exception:
                pass
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def get_model_metadata(self, model_id: str) -> Optional[RLModelMetadata]:
        meta_file = MODELS_DIR / f"{model_id}.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return RLModelMetadata(**json.load(f))
            except Exception:
                pass
        return None

    def load_model(self, model_id: str) -> Optional[PPO]:
        if not model_id or model_id == "latest":
            models = self.list_models()
            if models:
                return self.load_model(models[0].model_id)
            return None

        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        meta = self.get_model_metadata(model_id)
        if not meta:
            # Try finding by name in saved models
            for m in self.list_models():
                if m.name.lower() == model_id.lower() or m.model_id == model_id:
                    meta = m
                    break

        if not meta or not os.path.exists(meta.checkpoint_path):
            # Check direct zip file
            direct_path = MODELS_DIR / f"{model_id}.zip"
            if direct_path.exists():
                model = PPO.load(str(direct_path))
                self._loaded_models[model_id] = model
                return model
            
            # Fallback to most recent model if available
            models = self.list_models()
            if models and os.path.exists(models[0].checkpoint_path):
                model = PPO.load(models[0].checkpoint_path)
                self._loaded_models[model_id] = model
                return model
            return None

        model = PPO.load(meta.checkpoint_path)
        self._loaded_models[model_id] = model
        return model


# Global singleton instance
training_manager = RLTrainingManager()
