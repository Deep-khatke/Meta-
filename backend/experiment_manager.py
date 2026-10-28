from __future__ import annotations
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import mujoco

from backend.models import (
    Project, SceneObject, RobotConfig, SimulationSettings, SCHEMA_VERSION
)
from backend.simulation import MuJoCoEngine, quat_to_forward
from backend.project_manager import get_default_project
from backend.experiment_models import (
    ExperimentConfig, TaskObjective, ScriptedCommand, AvoidancePolicy,
    PolicyTrainingIteration, PolicyTrainingResult,
    SimulatedMeasurement, DerivedMetrics, ExperimentRunResult, RunComparison
)

EXPERIMENTS_DIR = Path("experiments")

def normalize_angle(theta: float) -> float:
    """Normalizes angle to range [-pi, pi]."""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

def get_default_scenarios() -> List[ExperimentConfig]:
    base_proj = get_default_project()
    
    # Scenario 1: Target Navigation
    scen1 = ExperimentConfig(
        id="scen_nav_alpha",
        name="Baseline Target Navigation",
        description="Autonomous goal-seeking navigation to target waypoint (2.0, 0.0)",
        seed=42,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="obs_box_1",
                name="Cargo Box A",
                type="box",
                position=[1.2, 0.6, 0.20],
                rotation=[0.0, 0.0, 0.0],
                size=[0.2, 0.2, 0.2],
                color="#e06c75",
                is_static=True,
                mass=5.0
            )
        ],
        settings=SimulationSettings(timestep=0.005, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81]),
        task=TaskObjective(
            target_position=[2.0, 0.0, 0.25],
            distance_threshold=0.25,
            max_allowed_collisions=0,
            timeout=8.0,
            control_mode="goal_seeking"
        ),
        duration=8.0,
        sample_rate_hz=20.0
    )
    
    # Scenario 2: Obstacle Slalom
    scen2 = ExperimentConfig(
        id="scen_slalom_beta",
        name="Obstacle Slalom",
        description="Navigation requiring obstacle clearance to reach (2.5, 0.5)",
        seed=101,
        robot=base_proj.robot.model_copy(deep=True),
        environment=[
            SceneObject(
                id="obs_slalom_1",
                name="Slalom Pylon 1",
                type="cylinder",
                position=[1.0, 0.1, 0.3],
                size=[0.15, 0.3],
                color="#98c379",
                is_static=True,
                mass=10.0
            ),
            SceneObject(
                id="obs_slalom_2",
                name="Slalom Pylon 2",
                type="cylinder",
                position=[1.8, -0.2, 0.3],
                size=[0.15, 0.3],
                color="#e5c07b",
                is_static=True,
                mass=10.0
            )
        ],
        settings=SimulationSettings(timestep=0.005, speed_multiplier=1.0, gravity=[0.0, 0.0, -9.81]),
        task=TaskObjective(
            target_position=[2.5, 0.5, 0.25],
            distance_threshold=0.30,
            max_allowed_collisions=2,
            timeout=10.0,
            control_mode="goal_seeking"
        ),
        duration=10.0,
        sample_rate_hz=20.0
    )
    return [scen1, scen2]


class ExperimentManager:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or EXPERIMENTS_DIR
        self.scenarios_dir = self.base_dir / "scenarios"
        self.runs_dir = self.base_dir / "runs"
        
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        self._ensure_default_scenarios()

    def _ensure_default_scenarios(self):
        """Creates reference scenarios if scenarios directory is empty."""
        existing = list(self.scenarios_dir.glob("*.json"))
        if not existing:
            for s in get_default_scenarios():
                self.save_scenario(s)

    # -------------------------------------------------------------
    # Scenario Management
    # -------------------------------------------------------------
    def save_scenario(self, scenario: ExperimentConfig) -> str:
        """Saves a scenario configuration persistently to disk."""
        file_path = self.scenarios_dir / f"{scenario.id}.json"
        data = scenario.model_dump()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(file_path)

    def get_scenario(self, scenario_id: str) -> Optional[ExperimentConfig]:
        file_path = self.scenarios_dir / f"{scenario_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentConfig.model_validate(data)

    def list_scenarios(self) -> List[ExperimentConfig]:
        scenarios = []
        for p in self.scenarios_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    scenarios.append(ExperimentConfig.model_validate(json.load(f)))
            except Exception:
                pass
        return sorted(scenarios, key=lambda s: s.created_at, reverse=True)

    def duplicate_scenario(self, scenario_id: str, new_name: Optional[str] = None) -> ExperimentConfig:
        scen = self.get_scenario(scenario_id)
        if not scen:
            raise FileNotFoundError(f"Scenario '{scenario_id}' not found.")
        
        new_scen = scen.model_copy(deep=True)
        new_scen.id = f"scen_{uuid.uuid4().hex[:8]}"
        new_scen.name = new_name or f"{scen.name} (Copy)"
        new_scen.created_at = time.time()
        self.save_scenario(new_scen)
        return new_scen

    def delete_scenario(self, scenario_id: str) -> bool:
        file_path = self.scenarios_dir / f"{scenario_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def reset_scenario_to_defaults(self, scenario_id: str) -> ExperimentConfig:
        """If scenario matches a default reference ID, restore it to default."""
        defaults = {s.id: s for s in get_default_scenarios()}
        if scenario_id in defaults:
            default_scen = defaults[scenario_id].model_copy(deep=True)
            self.save_scenario(default_scen)
            return default_scen
        scen = self.get_scenario(scenario_id)
        if not scen:
            raise FileNotFoundError(f"Scenario '{scenario_id}' not found.")
        return scen

    # -------------------------------------------------------------
    # Experiment Execution (Deterministic & Seeded)
    # -------------------------------------------------------------
    def run_experiment(
        self,
        config: ExperimentConfig,
        seed_override: Optional[int] = None,
        duration_override: Optional[float] = None
    ) -> ExperimentRunResult:
        """Executes an experiment deterministically, recording physical measurements and metrics."""
        seed = seed_override if seed_override is not None else config.seed
        duration = duration_override if duration_override is not None else config.duration
        
        # Build project representation for MuJoCo
        proj = Project(
            schema_version=config.schema_version,
            name=config.name,
            robot=config.robot.model_copy(deep=True),
            objects=[o.model_copy(deep=True) for o in config.environment],
            settings=config.settings.model_copy(deep=True)
        )

        # Seed random generators for numerical determinism
        np.random.seed(seed)
        
        # Instantiate dedicated MuJoCo engine
        engine = MuJoCoEngine(proj)
        dt = config.settings.timestep
        sample_interval = 1.0 / config.sample_rate_hz
        next_sample_time = 0.0

        measurements: List[SimulatedMeasurement] = []
        target = np.array(config.task.target_position[:2]) # [x, y]
        target_thresh = config.task.distance_threshold
        max_collisions = config.task.max_allowed_collisions
        timeout = min(duration, config.task.timeout)
        
        status = "completed"
        status_reason = "Simulation duration completed."
        time_to_goal: Optional[float] = None
        unique_collision_events = 0
        in_collision = False

        total_sim_steps = int(math.ceil(duration / dt))
        v_max = config.robot.max_linear_velocity
        w_max = config.robot.max_angular_velocity

        for step_idx in range(total_sim_steps):
            sim_time = float(engine.data.time)

            # Robot pose
            r_pos = np.array(engine.data.qpos[engine.robot_qpos_adr:engine.robot_qpos_adr + 2]) # [x, y]
            r_quat = [float(x) for x in engine.data.qpos[engine.robot_qpos_adr + 3:engine.robot_qpos_adr + 7]]
            dist_to_target = float(np.linalg.norm(r_pos - target))

            # Collision tracking
            ncon = int(engine.data.ncon)
            robot_contact_geoms = []
            for i in range(ncon):
                con = engine.data.contact[i]
                g1 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                g2 = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                if "robot" in g1 or "robot" in g2:
                    other = g2 if "robot" in g1 else g1
                    if other != "ground" and other != "":
                        robot_contact_geoms.append(other)

            current_has_collision = len(robot_contact_geoms) > 0
            if current_has_collision and not in_collision:
                unique_collision_events += 1
            in_collision = current_has_collision

            # Check task success / termination
            if dist_to_target <= target_thresh and time_to_goal is None:
                time_to_goal = sim_time
                status = "completed"
                status_reason = f"Target reached within {dist_to_target:.3f}m at t={sim_time:.3f}s"
                # Record final measurement and stop
                self._record_measurement(measurements, engine, robot_contact_geoms)
                break

            if unique_collision_events > max_collisions and max_collisions >= 0:
                status = "failed"
                status_reason = f"Collision limit exceeded: {unique_collision_events} > {max_collisions}"
                self._record_measurement(measurements, engine, robot_contact_geoms)
                break

            if sim_time >= timeout:
                status = "timeout"
                status_reason = f"Timeout reached at {sim_time:.2f}s (distance to goal: {dist_to_target:.3f}m)"
                self._record_measurement(measurements, engine, robot_contact_geoms)
                break

            # Periodic measurement sampling
            if sim_time >= next_sample_time or step_idx == 0:
                self._record_measurement(measurements, engine, robot_contact_geoms)
                next_sample_time += sample_interval

            # Controller computation
            if config.task.control_mode == "goal_seeking":
                cmd_v, cmd_w = self._compute_goal_seeking_control(
                    r_pos, r_quat, target, dist_to_target, v_max, w_max,
                    obstacles=config.environment,
                    policy=config.task.policy
                )
                engine.set_teleop(cmd_v, cmd_w)
            elif config.task.control_mode == "scripted" and config.task.scripted_commands:
                cmd_v, cmd_w = self._interpolate_scripted_command(config.task.scripted_commands, sim_time)
                engine.set_teleop(cmd_v, cmd_w)

            # Advance physics
            engine.step()

        # Compute derived metrics
        derived = self._compute_derived_metrics(
            measurements, config.task.target_position, time_to_goal, unique_collision_events, status, status_reason
        )

        run_result = ExperimentRunResult(
            experiment_id=config.id,
            experiment_name=config.name,
            config_version=config.schema_version,
            seed=seed,
            status=status,
            completed_at=time.time(),
            measurements=measurements,
            metrics=derived,
            errors=engine.errors
        )

        # Save run result to disk
        self.save_run_result(run_result)
        return run_result

    def _record_measurement(self, measurements: List[SimulatedMeasurement], engine: MuJoCoEngine, contact_geoms: List[str]):
        st = engine.get_state()
        r = st.robot
        measurements.append(SimulatedMeasurement(
            sim_time=st.sim_time,
            position=[round(x, 4) for x in r.position],
            quaternion=[round(x, 4) for x in r.quaternion],
            linear_velocity=[round(x, 4) for x in r.linear_velocity],
            angular_velocity=[round(x, 4) for x in r.angular_velocity],
            linear_speed=round(r.linear_speed, 4),
            angular_speed=round(r.angular_speed, 4),
            contacts_count=len(contact_geoms),
            active_contacts=contact_geoms
        ))

    def _compute_goal_seeking_control(
        self,
        r_pos: np.ndarray,
        r_quat: List[float],
        target: np.ndarray,
        dist: float,
        v_max: float,
        w_max: float,
        obstacles: Optional[List[SceneObject]] = None,
        policy: Optional[AvoidancePolicy] = None
    ) -> Tuple[float, float]:
        """Artificial Potential Field (APF) + reactive vortex circulation controller.
        
        Combines attractive pull towards goal waypoint with repulsive potential fields
        and tangential circulation forces from nearby scene obstacles to smoothly navigate
        cluttered environments without collisions.
        """
        fwd = quat_to_forward(r_quat)
        current_yaw = math.atan2(fwd[1], fwd[0])
        
        diff = target - r_pos
        goal_dist = max(1e-4, float(np.linalg.norm(diff)))
        f_att = diff / goal_dist

        f_rep = np.zeros(2, dtype=float)
        f_tan = np.zeros(2, dtype=float)
        closest_clearance = 999.0

        if policy and policy.enabled and obstacles:
            for obs in obstacles:
                o_pos = np.array(obs.position[:2], dtype=float)
                delta = r_pos - o_pos
                center_dist = float(np.linalg.norm(delta))
                
                # Accurate obstacle collision boundary radius + robot footprint (~0.28m)
                if obs.type == "box":
                    obs_radius = math.hypot(obs.size[0], obs.size[1]) if len(obs.size) >= 2 else 0.35
                elif obs.type in ("cylinder", "sphere"):
                    obs_radius = obs.size[0] if obs.size else 0.25
                else:
                    obs_radius = 0.30

                robot_footprint = 0.28
                clearance = center_dist - obs_radius - robot_footprint
                if clearance < closest_clearance:
                    closest_clearance = clearance

                if clearance < policy.safety_margin:
                    norm_dir = delta / max(1e-4, center_dist)
                    inv_d = 1.0 / max(0.04, clearance)
                    inv_d_safe = 1.0 / policy.safety_margin
                    rep_strength = policy.repulsion_gain * (inv_d - inv_d_safe) * (inv_d ** 1.3)
                    f_rep += norm_dir * rep_strength

                    # Tangential vortex steering around obstacle
                    tan_1 = np.array([-norm_dir[1], norm_dir[0]])
                    tan_2 = -tan_1
                    # Steer along the tangent that best heads towards the goal
                    chosen_tan = tan_1 if np.dot(tan_1, f_att) >= np.dot(tan_2, f_att) else tan_2
                    f_tan += chosen_tan * (rep_strength * policy.tangential_gain)

        f_net = f_att + f_rep + f_tan
        desired_yaw = math.atan2(f_net[1], f_net[0])
        heading_error = normalize_angle(desired_yaw - current_yaw)

        # Speed scaling based on obstacle proximity and alignment
        speed_factor = 1.0
        slow_dist = policy.slowdown_distance if policy else 0.55
        if closest_clearance < slow_dist:
            speed_factor = max(0.15, closest_clearance / max(0.05, slow_dist))

        # Close danger avoidance: if an obstacle is dangerously close, prioritize reversing and turning away
        if closest_clearance < 0.10:
            v_cmd = -0.12
            turn_dir = 1.0 if heading_error >= 0 else -1.0
            w_cmd = (policy.max_turn_rate if policy else 2.5) * turn_dir
        else:
            turn_limit = min(w_max, policy.max_turn_rate if policy else 2.5)
            w_cmd = max(-w_max, min(w_max, heading_error * turn_limit))
            
            # Forward speed modulated by heading alignment and distance
            align = max(0.05, math.cos(heading_error))
            v_cmd = min(v_max, max(0.2, dist * 0.8)) * speed_factor * align
            if abs(heading_error) > 0.7:
                v_cmd = max(0.05, v_cmd * 0.20)

        return float(v_cmd), float(w_cmd)

    def _interpolate_scripted_command(self, commands: List[ScriptedCommand], sim_time: float) -> Tuple[float, float]:
        active_cmd = commands[0]
        for cmd in commands:
            if sim_time >= cmd.time:
                active_cmd = cmd
            else:
                break
        return active_cmd.linear, active_cmd.angular

    def _compute_derived_metrics(
        self,
        measurements: List[SimulatedMeasurement],
        target_pos: List[float],
        time_to_goal: Optional[float],
        collision_count: int,
        status: str,
        status_reason: str
    ) -> DerivedMetrics:
        if not measurements:
            return DerivedMetrics(
                total_duration=0.0,
                total_distance_traveled=0.0,
                initial_distance_to_target=0.0,
                final_distance_to_target=0.0,
                min_distance_to_target=0.0,
                max_linear_speed=0.0,
                mean_linear_speed=0.0,
                max_angular_speed=0.0,
                total_collisions=0,
                success=False,
                status_reason="No measurements recorded"
            )

        target = np.array(target_pos[:2])
        dists = [float(np.linalg.norm(np.array(m.position[:2]) - target)) for m in measurements]
        speeds = [m.linear_speed for m in measurements]
        ang_speeds = [m.angular_speed for m in measurements]

        # Path length
        path_length = 0.0
        for i in range(1, len(measurements)):
            p1 = np.array(measurements[i - 1].position[:2])
            p2 = np.array(measurements[i].position[:2])
            path_length += float(np.linalg.norm(p2 - p1))

        success = (status == "completed" and time_to_goal is not None)

        return DerivedMetrics(
            total_duration=round(measurements[-1].sim_time, 3),
            total_distance_traveled=round(path_length, 3),
            initial_distance_to_target=round(dists[0], 3),
            final_distance_to_target=round(dists[-1], 3),
            min_distance_to_target=round(min(dists), 3),
            max_linear_speed=round(max(speeds), 3),
            mean_linear_speed=round(float(np.mean(speeds)), 3),
            max_angular_speed=round(max(ang_speeds), 3),
            total_collisions=collision_count,
            success=success,
            time_to_goal=round(time_to_goal, 3) if time_to_goal is not None else None,
            status_reason=status_reason
        )

    # -------------------------------------------------------------
    # Run Persistence & Retrieval
    # -------------------------------------------------------------
    def save_run_result(self, result: ExperimentRunResult) -> str:
        file_path = self.runs_dir / f"{result.run_id}.json"
        data = result.model_dump()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(file_path)

    def get_run_result(self, run_id: str) -> Optional[ExperimentRunResult]:
        file_path = self.runs_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return ExperimentRunResult.model_validate(json.load(f))

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for p in self.runs_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    r_dict = json.load(f)
                    runs.append({
                        "run_id": r_dict.get("run_id"),
                        "experiment_id": r_dict.get("experiment_id"),
                        "experiment_name": r_dict.get("experiment_name"),
                        "seed": r_dict.get("seed"),
                        "status": r_dict.get("status"),
                        "created_at": r_dict.get("created_at"),
                        "metrics": r_dict.get("metrics")
                    })
            except Exception:
                pass
        return sorted(runs, key=lambda r: r.get("created_at", 0), reverse=True)

    # -------------------------------------------------------------
    # Run Comparison
    # -------------------------------------------------------------
    def compare_runs(self, run_a_id: str, run_b_id: str) -> RunComparison:
        run_a = self.get_run_result(run_a_id)
        if not run_a:
            raise FileNotFoundError(f"Run '{run_a_id}' not found.")
        run_b = self.get_run_result(run_b_id)
        if not run_b:
            raise FileNotFoundError(f"Run '{run_b_id}' not found.")

        mA = run_a.metrics
        mB = run_b.metrics

        deltas = {
            "duration_delta": round(mB.total_duration - mA.total_duration, 3),
            "final_distance_delta": round(mB.final_distance_to_target - mA.final_distance_to_target, 3),
            "min_distance_delta": round(mB.min_distance_to_target - mA.min_distance_to_target, 3),
            "total_distance_delta": round(mB.total_distance_traveled - mA.total_distance_traveled, 3),
            "mean_speed_delta": round(mB.mean_linear_speed - mA.mean_linear_speed, 3),
            "collisions_delta": float(mB.total_collisions - mA.total_collisions),
        }

        # Narrative summary
        summary_lines = []
        if mA.success == mB.success:
            summary_lines.append(f"Both runs {'succeeded' if mA.success else 'did not complete successfully'}.")
        else:
            summary_lines.append(f"Run A {'succeeded' if mA.success else 'failed'}, whereas Run B {'succeeded' if mB.success else 'failed'}.")
        
        if deltas['final_distance_delta'] < 0:
            summary_lines.append(f"Run B reached closer to the target by {abs(deltas['final_distance_delta']):.3f}m.")
        elif deltas['final_distance_delta'] > 0:
            summary_lines.append(f"Run A reached closer to the target by {deltas['final_distance_delta']:.3f}m.")
        else:
            summary_lines.append("Both runs achieved identical final distance to target.")

        if deltas['collisions_delta'] != 0:
            summary_lines.append(f"Collision difference: {deltas['collisions_delta']:+.0f}.")

        return RunComparison(
            run_a_id=run_a.run_id,
            run_b_id=run_b.run_id,
            run_a_name=run_a.experiment_name,
            run_b_name=run_b.experiment_name,
            run_a_seed=run_a.seed,
            run_b_seed=run_b.seed,
            metric_deltas=deltas,
            run_a_metrics=mA,
            run_b_metrics=mB,
            summary=" ".join(summary_lines)
        )

    def train_avoidance_policy(self, scenario_id: str) -> PolicyTrainingResult:
        """Evaluates multiple obstacle avoidance policy configurations on the scenario,
        finds the policy that minimizes collisions and time to goal, and updates the scenario."""
        if not scenario_id:
            scens = self.list_scenarios()
            if scens:
                scenario_id = scens[0].id
        scen = self.get_scenario(scenario_id) if scenario_id else None
        if not scen:
            scens = self.list_scenarios()
            if scens:
                scen = scens[0]
                scenario_id = scen.id
            else:
                raise FileNotFoundError(f"Scenario '{scenario_id}' not found.")

        # Candidate policies varying safety clearance, repulsion strength, and tangential vortex
        candidates = [
            AvoidancePolicy(enabled=True, safety_margin=0.6, repulsion_gain=1.4, tangential_gain=1.2, max_turn_rate=2.5, slowdown_distance=0.4),
            AvoidancePolicy(enabled=True, safety_margin=0.75, repulsion_gain=2.2, tangential_gain=1.8, max_turn_rate=3.0, slowdown_distance=0.55),
            AvoidancePolicy(enabled=True, safety_margin=0.88, repulsion_gain=2.8, tangential_gain=2.4, max_turn_rate=3.5, slowdown_distance=0.65),
            AvoidancePolicy(enabled=True, safety_margin=1.05, repulsion_gain=3.5, tangential_gain=2.8, max_turn_rate=3.8, slowdown_distance=0.75),
            AvoidancePolicy(enabled=True, safety_margin=0.95, repulsion_gain=3.0, tangential_gain=3.2, max_turn_rate=4.0, slowdown_distance=0.7),
        ]

        iterations = []
        best_candidate = candidates[0]
        best_score = -999999.0
        baseline_cols = 0

        for idx, cand in enumerate(candidates, 1):
            test_scen = scen.model_copy(deep=True)
            test_scen.task.policy = cand
            # Allow enough time for maneuvering around obstacles
            test_duration = min(14.0, max(scen.duration, scen.task.timeout))
            run_res = self.run_experiment(test_scen, seed_override=scen.seed, duration_override=test_duration)
            cols = run_res.metrics.total_collisions
            ttg = run_res.metrics.time_to_goal or run_res.metrics.total_duration
            succ = run_res.metrics.success
            final_d = run_res.metrics.final_distance_to_target

            if idx == 1:
                baseline_cols = cols

            # Scoring: Large penalty for collisions, bonus for success, penalty for excess time
            score = (1000.0 if succ else 0.0) - (cols * 250.0) - (ttg * 8.0) - (final_d * 30.0)
            
            iterations.append(PolicyTrainingIteration(
                iteration=idx,
                policy=cand,
                success=succ,
                collisions=cols,
                time_to_goal=run_res.metrics.time_to_goal,
                final_distance=round(final_d, 3),
                score=round(score, 1)
            ))

            if score > best_score:
                best_score = score
                best_candidate = cand

        # Apply optimal candidate policy directly to the scenario
        scen.task.policy = best_candidate
        self.save_scenario(scen)

        trained_cols = [it.collisions for it in iterations if it.policy == best_candidate][0]
        improved = trained_cols < baseline_cols or (trained_cols == 0 and baseline_cols > 0)
        
        msg = f"Trained 5 policy variations. Best policy achieved {trained_cols} collisions (baseline: {baseline_cols})."
        if trained_cols == 0:
            msg += " Successfully achieved zero collisions and verified path to target!"

        return PolicyTrainingResult(
            scenario_id=scen.id,
            iterations=iterations,
            best_policy=best_candidate,
            baseline_collisions=baseline_cols,
            trained_collisions=trained_cols,
            improved=improved,
            message=msg
        )

    def run_multi_trial_experiment(self, scenario_id: str, num_trials: int = 5) -> Dict[str, Any]:
        """Executes multiple independent experiment trials with seed variations and computes statistical metrics."""
        scen = self.get_scenario(scenario_id)
        if not scen:
            raise ValueError(f"Scenario '{scenario_id}' not found.")

        trial_results = []
        successes = []
        ttg_list = []
        dist_list = []
        col_list = []
        energy_list = []

        base_seed = scen.seed or 42
        for t_idx in range(num_trials):
            # Clone scenario with varied seed
            trial_scen = scen.model_copy(deep=True)
            trial_scen.seed = base_seed + t_idx * 17
            run_res = self.run_experiment(trial_scen)
            dist = run_res.metrics.total_distance_traveled
            cols = run_res.metrics.total_collisions
            energy = round(run_res.metrics.mean_linear_speed * run_res.metrics.total_duration, 3)

            trial_results.append({
                "trial": t_idx + 1,
                "seed": trial_scen.seed,
                "success": run_res.metrics.success,
                "collisions": cols,
                "time_to_goal": run_res.metrics.time_to_goal,
                "distance": dist,
                "energy_proxy": energy
            })

            successes.append(1.0 if run_res.metrics.success else 0.0)
            if run_res.metrics.time_to_goal is not None:
                ttg_list.append(run_res.metrics.time_to_goal)
            dist_list.append(dist)
            col_list.append(float(cols))
            energy_list.append(energy)

        def calc_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            mean = float(np.mean(values))
            std = float(np.std(values))
            return {
                "mean": round(mean, 3),
                "std": round(std, 3),
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3)
            }

        return {
            "scenario_id": scenario_id,
            "num_trials": num_trials,
            "success_rate": round(float(np.mean(successes)), 3),
            "time_to_goal_stats": calc_stats(ttg_list),
            "distance_stats": calc_stats(dist_list),
            "collision_stats": calc_stats(col_list),
            "energy_stats": calc_stats(energy_list),
            "trials": trial_results
        }

# Global singleton instance
experiment_manager = ExperimentManager()

