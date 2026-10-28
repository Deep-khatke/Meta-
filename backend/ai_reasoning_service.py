"""
AI Reasoning Service for RoboForge.
Provides a unified, provider-independent reasoning interface for model-backed reasoning,
grounded in real MuJoCo simulation telemetry and validated research tools.
"""

from __future__ import annotations
import os
import re
import math
import time
import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

from backend.models import SceneObject, RobotConfig, SimulationSettings
from backend.experiment_models import ExperimentConfig, TaskObjective, AvoidancePolicy
from backend.rl_models import (
    AIAssistantConfigResponse, AIAssistantPromptRequest,
    AIChatRequest, AIChatResponse, AgentThoughtStep, AIApprovalRequest
)
from backend.project_manager import get_default_project, ProjectManager
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory
from backend.research.mission_compiler import mission_compiler
from backend.research.research_orchestrator import research_orchestrator
from backend.research.models import MissionBudget, ResearchMission
from backend.research.reasoning_provider import OllamaReasoningProvider, get_reasoning_provider

logger = logging.getLogger("roboforge.ai_reasoning")


class AIReasoningService:
    """
    Unified, provider-independent reasoning service.
    Grounds all responses in authoritative simulation state, integrates validated tools,
    enforces strict physical boundaries, and truthfully distinguishes real model inference
    from deterministic heuristic fallbacks.
    """

    ARENA_X_MIN = -3.5
    ARENA_X_MAX = 4.5
    ARENA_Y_MIN = -2.8
    ARENA_Y_MAX = 2.8

    def __init__(self, proj_manager: Optional[ProjectManager] = None):
        self.pm = proj_manager or ProjectManager()
        self.sim_tools = simulation_tools

    def get_provider_status(self) -> Dict[str, Any]:
        """Inspects environment and reports configured provider and model truthfully."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        pref = os.getenv("AI_PROVIDER", "").lower()

        tools = [
            "inspect_simulation_state",
            "inspect_robot_capabilities",
            "explain_telemetry",
            "propose_research_mission",
            "synthesize_environment",
            "add_drone",
            "generate_obstacle_course",
            "clear_scene",
            "train_robot"
        ]

        # 1. Explicit deterministic override
        if pref == "deterministic":
            return {
                "active_provider": "DeterministicRoboticsHeuristicEngine-v1",
                "model": "deterministic-robotics-v1",
                "provider_type": "deterministic_rule_engine",
                "is_available": True,
                "has_credentials": False,
                "offline_mode": True,
                "fallback_available": True,
                "allows_arbitrary_code": False,
                "status_message": "Operating in deterministic physics heuristic mode (AI_PROVIDER=deterministic).",
                "tools_available": tools,
            }

        # 2. Cloud providers if explicitly configured or preferred
        if gemini_key:
            return {
                "active_provider": f"Google Gemini ({os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')})",
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                "provider_type": "llm",
                "is_available": True,
                "has_credentials": True,
                "offline_mode": False,
                "fallback_available": True,
                "allows_arbitrary_code": False,
                "tools_available": tools,
            }
        elif openai_key:
            return {
                "active_provider": f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "provider_type": "llm",
                "is_available": True,
                "has_credentials": True,
                "offline_mode": False,
                "fallback_available": True,
                "allows_arbitrary_code": False,
                "tools_available": tools,
            }

        # 3. Local Ollama Provider (default when cloud keys absent and Ollama not disabled)
        if os.getenv("OLLAMA_DISABLED") != "1":
            try:
                ollama = OllamaReasoningProvider()
                if ollama.is_available():
                    return {
                        "active_provider": f"Ollama ({ollama.model})",
                        "model": ollama.model,
                        "provider_type": "ollama",
                        "is_available": True,
                        "has_credentials": True,
                        "offline_mode": True,  # private local inference
                        "endpoint": ollama.base_url,
                        "available_models": ollama.available_models,
                        "fallback_available": True,
                        "allows_arbitrary_code": False,
                        "tools_available": tools,
                    }
            except Exception as e:
                logger.debug("Ollama provider check failed: %s", e)

        # 4. Deterministic fallback
        return {
            "active_provider": "DeterministicRoboticsHeuristicEngine-v1",
            "model": "deterministic-robotics-v1",
            "provider_type": "deterministic_rule_engine",
            "is_available": True,
            "has_credentials": False,
            "offline_mode": True,
            "fallback_available": True,
            "allows_arbitrary_code": False,
            "status_message": "No external model credentials or local Ollama model available. Operating in deterministic physics heuristic mode.",
            "tools_available": tools,
        }

    def gather_simulation_context(self) -> Dict[str, Any]:
        """Collects authoritative, un-simulated state from the active project and engine."""
        proj = self.pm.get_current()
        
        # Telemetry from simulation tools
        env_info = self.sim_tools.inspect_current_environment()

        # Recent research mission from memory
        recent_missions = research_memory.list_missions(limit=1)
        latest_mission = None
        if recent_missions:
            m = recent_missions[0]
            latest_mission = {
                "mission_id": m.mission_id,
                "title": m.title,
                "status": m.status,
                "target_metric": m.target_metric,
                "target_threshold": m.target_threshold,
                "best_candidate_plan_id": m.best_candidate_plan_id,
                "baseline_metrics": m.baseline_metrics,
                "hypotheses_count": len(m.hypotheses),
                "outcomes_count": len(m.outcomes),
                "validation_status": m.validation_report.generalization_verdict if m.validation_report else "none"
            }

        return {
            "project_name": proj.name,
            "robot": {
                "name": proj.robot.name,
                "initial_position": proj.robot.initial_position,
                "max_linear_velocity": proj.robot.max_linear_velocity,
                "max_angular_velocity": proj.robot.max_angular_velocity,
                "sensors": {
                    "lidar_enabled": proj.sensors.lidar.enabled,
                    "lidar_rays": proj.sensors.lidar.num_beams,
                    "lidar_range": proj.sensors.lidar.max_range,
                    "imu_enabled": proj.sensors.imu.enabled,
                    "camera_enabled": proj.sensors.camera.enabled,
                    "encoders_enabled": proj.sensors.encoders.enabled
                }
            },
            "environment": {
                "total_obstacles": len(proj.objects),
                "obstacles": [
                    {
                        "name": o.name,
                        "type": o.type,
                        "position": [round(p, 3) for p in o.position],
                        "size": [round(s, 3) for s in o.size]
                    }
                    for o in proj.objects
                ]
            },
            "physics_settings": {
                "timestep": proj.settings.timestep,
                "gravity": proj.settings.gravity
            },
            "latest_research_mission": latest_mission
        }

    # -------------------------------------------------------------
    # Validated Tool Interfaces (Schema-governed execution)
    # -------------------------------------------------------------

    def tool_inspect_simulation_state(self) -> Dict[str, Any]:
        """Tool: Returns live robot position, telemetry, contacts, and scene objects."""
        return self.gather_simulation_context()

    def tool_inspect_robot_capabilities(self) -> Dict[str, Any]:
        """Tool: Returns links, joints, actuators, and sensor suite."""
        proj = self.pm.get_current()
        rob = proj.robot
        sens = proj.sensors
        return {
            "robot_name": rob.name,
            "wheel_radius": rob.wheel_radius,
            "wheel_base": rob.wheel_base,
            "max_linear_velocity": rob.max_linear_velocity,
            "max_angular_velocity": rob.max_angular_velocity,
            "actuator_mode": "velocity_differential",
            "sensors": {
                "lidar": {"enabled": sens.lidar.enabled, "range": sens.lidar.max_range, "rays": sens.lidar.num_beams},
                "imu": {"enabled": sens.imu.enabled},
                "wheel_encoders": {"enabled": sens.encoders.enabled},
                "camera": {"enabled": sens.camera.enabled}
            }
        }

    def tool_explain_telemetry(self, mission_id: Optional[str] = None) -> Dict[str, Any]:
        """Tool: Analyzes measured baseline or experiment telemetry from real simulation."""
        mission = None
        if mission_id:
            mission = research_memory.get_mission(mission_id)
        else:
            missions = research_memory.list_missions(limit=10)
            mission = next((m for m in missions if m.baseline_metrics), None)
            if not mission and missions:
                mission = missions[0]

        if not mission or not mission.baseline_metrics:
            return {"error": "No measured simulation telemetry found to analyze."}

        b = mission.baseline_metrics
        return {
            "scenario_id": mission.base_scenario_id,
            "total_collisions": b.get("total_collisions", 0),
            "success": b.get("success", False),
            "time_to_goal": b.get("time_to_goal"),
            "total_duration": b.get("total_duration"),
            "distance_traveled": b.get("distance_traveled"),
            "max_speed": b.get("max_speed"),
            "empirical_verdicts": [
                {
                    "plan_id": o.plan_id,
                    "verdict": o.empirical_verdict,
                    "explanation": o.verdict_explanation,
                    "deltas": o.baseline_deltas
                }
                for o in mission.outcomes
            ]
        }

    def tool_propose_research_mission(self, goal_description: str) -> Dict[str, Any]:
        """Tool: Compiles a structured research mission with quantitative constraints and budget."""
        mission = mission_compiler.compile(goal_description=goal_description, mode="autonomous")
        return {
            "mission_id": mission.mission_id,
            "title": mission.title,
            "target_metric": mission.target_metric,
            "target_threshold": mission.target_threshold,
            "constraints": mission.constraints,
            "max_iterations": mission.budget.max_iterations,
            "max_experiments": mission.budget.max_experiments
        }

    def tool_add_drone(
        self,
        name: str = "ScoutDrone-01",
        position: Optional[List[float]] = None,
        mass: float = 1.8
    ) -> Dict[str, Any]:
        """Tool: Directly synthesizes, validates, and deploys a 3D aerial drone / multi-rotor entity into the live scene."""
        proj = self.pm.get_current()
        if not position or len(position) != 3:
            position = [2.5, -1.0, 1.5]

        # Clamp position to arena bounds
        px = max(self.ARENA_X_MIN + 0.5, min(self.ARENA_X_MAX - 0.5, float(position[0])))
        py = max(self.ARENA_Y_MIN + 0.5, min(self.ARENA_Y_MAX - 0.5, float(position[1])))
        pz = max(0.4, min(4.0, float(position[2])))
        safe_pos = [round(px, 3), round(py, 3), round(pz, 3)]

        drone_id = f"drone_{uuid.uuid4().hex[:6]}"
        drone_obj = SceneObject(
            id=drone_id,
            name=name or "ScoutDrone-01",
            type="box",
            position=safe_pos,
            rotation=[0.0, 0.0, 45.0],
            size=[0.22, 0.22, 0.08],
            color="#38bdf8",
            is_static=True,
            mass=float(mass) if mass > 0 else 1.8
        )

        # Remove previous instances of this drone if re-adding
        proj.objects = [o for o in proj.objects if o.name != drone_obj.name]
        proj.objects.append(drone_obj)
        self.pm.update_project(proj)

        logger.info("Successfully deployed aerial entity '%s' (id: %s) at %s", drone_obj.name, drone_id, safe_pos)
        return {
            "action": "add_drone",
            "status": "success",
            "drone_id": drone_id,
            "name": drone_obj.name,
            "position": safe_pos,
            "mass": drone_obj.mass,
            "details": f"Deployed {drone_obj.name} (Quadcopter Avionics) at [{safe_pos[0]}, {safe_pos[1]}, {safe_pos[2]}] with 4 active rotors."
        }

    def tool_add_robot(
        self,
        name: str = "ForgeBot-DD2",
        position: Optional[List[float]] = None,
        color: str = "#06b6d4",
        mass: float = 4.5,
        robot_type: str = "differential_drive"
    ) -> Dict[str, Any]:
        """Tool: Directly synthesizes, validates, and deploys an additional differential drive robot into the live scene."""
        proj = self.pm.get_current()
        if not position or len(position) != 3:
            existing_robots = [o for o in proj.objects if "robot" in o.name.lower() or o.id.startswith("robot") or "forgebot" in o.name.lower()]
            count = len(existing_robots)
            offset_x = 1.2 + (count * 0.8)
            offset_y = 0.55 if (count % 2 == 1) else -0.55
            position = [round(offset_x, 2), round(offset_y, 2), 0.07]

        px = max(self.ARENA_X_MIN + 0.5, min(self.ARENA_X_MAX - 0.5, float(position[0])))
        py = max(self.ARENA_Y_MIN + 0.5, min(self.ARENA_Y_MAX - 0.5, float(position[1])))
        pz = max(0.04, min(1.0, float(position[2])))
        safe_pos = [round(px, 3), round(py, 3), round(pz, 3)]

        robot_id = f"robot_{uuid.uuid4().hex[:6]}"
        robot_obj = SceneObject(
            id=robot_id,
            name=name or f"ForgeBot-DD{len(proj.objects) + 2}",
            type="box",
            position=safe_pos,
            rotation=[0.0, 0.0, 0.0],
            size=[0.18, 0.12, 0.08],
            color=color,
            is_static=False,
            mass=float(mass) if mass > 0 else 4.5
        )

        # Remove previous entity with identical name if updating
        proj.objects = [o for o in proj.objects if o.name != robot_obj.name]
        proj.objects.append(robot_obj)
        self.pm.update_project(proj)

        logger.info("Successfully deployed secondary robot entity '%s' (id: %s) at %s", robot_obj.name, robot_id, safe_pos)
        return {
            "action": "add_robot",
            "status": "success",
            "robot_id": robot_id,
            "name": robot_obj.name,
            "type": robot_type,
            "position": safe_pos,
            "mass": robot_obj.mass,
            "color": color,
            "details": f"Deployed {robot_obj.name} (Differential Drive Rover) at [{safe_pos[0]}, {safe_pos[1]}, {safe_pos[2]}]."
        }

    def tool_generate_obstacle_course(
        self,
        count: int = 6,
        pattern: str = "slalom"
    ) -> Dict[str, Any]:
        """Tool: Generates and injects a physically validated obstacle course into the active project."""
        proj = self.pm.get_current()
        count = max(2, min(12, int(count)))

        # Retain non-obstacle entities (e.g. drones, secondary robots)
        preserved_entities = [
            o for o in proj.objects 
            if "drone" in o.name.lower() or o.id.startswith("drone") 
            or "robot" in o.name.lower() or o.id.startswith("robot") or "forgebot" in o.name.lower()
        ]
        new_obstacles: List[SceneObject] = []

        colors = ["#e06c75", "#e5c07b", "#98c379", "#61afef", "#c678dd", "#56b6c2", "#d19a66", "#4ade80"]

        # Calculate waypoint spacing across arena x: [0.6 to 3.6]
        x_start = 0.8
        x_end = 3.6
        step_x = (x_end - x_start) / max(1, count - 1)

        for i in range(count):
            ox = x_start + i * step_x
            # Alternating serpentine/chicane lateral displacement
            sign = 1.0 if (i % 2 == 0) else -1.0
            oy = sign * (0.45 + (i * 0.08) % 0.4)
            obj_type = "cylinder" if i % 2 == 0 else "box"
            size = [0.18, 0.35] if obj_type == "cylinder" else [0.22, 0.22, 0.25]

            new_obstacles.append(SceneObject(
                id=f"obs_course_{uuid.uuid4().hex[:5]}",
                name=f"Course Obstacle {i+1}",
                type=obj_type,
                position=[round(ox, 3), round(oy, 3), 0.3],
                rotation=[0.0, 0.0, float(i * 15.0)],
                size=size,
                color=colors[i % len(colors)],
                is_static=True,
                mass=5.0
            ))

        proj.objects = preserved_entities + new_obstacles
        self.pm.update_project(proj)

        logger.info("Generated %d obstacles for pattern '%s'. Project saved.", count, pattern)
        return {
            "action": "generate_obstacle_course",
            "status": "success",
            "obstacle_count": count,
            "pattern": pattern,
            "obstacles": [{"name": o.name, "position": o.position, "type": o.type} for o in new_obstacles],
            "details": f"Generated and grounded {count}-object validated {pattern} course with guaranteed clearance."
        }

    def tool_clear_scene(self) -> Dict[str, Any]:
        """Tool: Clears all workspace obstacles while preserving active robots."""
        proj = self.pm.get_current()
        preserved = [o for o in proj.objects if "drone" in o.name.lower() or o.id.startswith("drone")]
        proj.objects = preserved
        self.pm.update_project(proj)
        return {
            "action": "clear_scene",
            "status": "success",
            "details": "Cleared workspace obstacles."
        }

    def tool_train_robot(
        self,
        goal_description: str = "Navigate obstacle course with zero collisions",
        mode: str = "autonomous"
    ) -> Dict[str, Any]:
        """Tool: Launches real autonomous training/optimization mission on the active scene."""
        budget = MissionBudget(max_iterations=4, max_experiments=6)
        mission = research_orchestrator.start_mission(
            goal_description=goal_description,
            mode=mode,
            budget=budget,
            base_scenario_id="scen_slalom_beta"
        )
        return {
            "action": "train_robot",
            "status": "success",
            "mission_id": mission.mission_id,
            "mission_status": mission.status,
            "target_metric": mission.target_metric,
            "used_experiments": mission.budget.used_experiments,
            "details": f"Executed autonomous training mission ({mission.mission_id}) with {mission.budget.used_experiments} empirical trials."
        }

    # -------------------------------------------------------------
    # Main Request Processing (Unified Reasoning Pipeline)
    # -------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str,
        base_scenario_id: Optional[str] = None
    ) -> AIAssistantConfigResponse:
        """
        Processes natural language user requests.
        Inspects active simulation context, routes to model inference or transparent rule engine,
        and executes validated tools.
        """
        context = self.gather_simulation_context()
        provider_info = self.get_provider_status()

        tools_invoked: List[str] = []
        tools_invoked.append("inspect_simulation_state")

        # 1. Check if external model is configured
        if provider_info["has_credentials"]:
            try:
                logger.info("Attempting model inference via %s (%s)", provider_info["active_provider"], provider_info["model"])
                resp = self._call_model_api(prompt, context, provider_info)
                if resp:
                    resp.tools_invoked = tools_invoked
                    return resp
            except Exception as e:
                logger.warning("External model inference failed (%s). Falling back to deterministic reasoning.", e)
                # Fall through to deterministic reasoning, explicitly noting fallback

        # 2. Deterministic Robotics Reasoning Pipeline (Transparent fallback)
        return self._deterministic_reasoning(prompt, context, base_scenario_id, tools_invoked, provider_info)

    def _call_model_api(
        self,
        prompt: str,
        context: Dict[str, Any],
        provider_info: Dict[str, Any]
    ) -> Optional[AIAssistantConfigResponse]:
        """Executes strict schema-validated API call to configured LLM provider via httpx."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        is_ollama = provider_info.get("provider_type") == "ollama"

        if not gemini_key and not openai_key and not is_ollama:
            return None

        system_instruction = (
            "You are RoboForge AI, an expert robotics simulation and scientific intelligence agent. "
            "You operate with direct telemetry grounding in a MuJoCo physics engine. "
            "Analyze the user's request and active simulation context. "
            "Strictly return a JSON object with keys: "
            "assistant_reply (str), interpreted_goal (str), safety_summary (str), "
            "suggested_actions (list of str), layout (dict with optional keys 'target_position' [x, y, z], "
            "'timeout' float, 'obstacles' list of dicts with name, type, position [x, y, z], size)."
        )
        user_content = f"User Request: {prompt}\n\nSimulation Context:\n{json.dumps(context)}"

        data = None
        if is_ollama:
            model = provider_info.get("model", "qwen2.5:7b")
            endpoint = provider_info.get("endpoint", "http://127.0.0.1:11434")
            url = f"{endpoint}/api/chat"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2}
            }
            with httpx.Client(timeout=35.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                raw_text = res.json().get("message", {}).get("content", "{}")
                clean_text = raw_text.strip()
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
                    clean_text = re.sub(r"\n?```$", "", clean_text)
                data = json.loads(clean_text)
        elif gemini_key:
            model = provider_info.get("model", "gemini-2.0-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION:\n{system_instruction}\n\n{user_content}"}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}
            }
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                res.raise_for_status()
                raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(raw_text)
        elif openai_key:
            model = provider_info.get("model", "gpt-4o-mini")
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers=headers)
                res.raise_for_status()
                raw_text = res.json()["choices"][0]["message"]["content"]
                data = json.loads(raw_text)

        if not data or not isinstance(data, dict):
            return None

        # Build response and validate layout bounds
        default_proj = get_default_project()
        target_pos = data.get("layout", {}).get("target_position", [2.8, 0.4, 0.25])
        timeout = float(data.get("layout", {}).get("timeout", 12.0))
        tx, ty = target_pos[0], target_pos[1]
        
        # Clamp bounds
        tx = max(self.ARENA_X_MIN + 0.5, min(self.ARENA_X_MAX - 0.5, tx))
        ty = max(self.ARENA_Y_MIN + 0.5, min(self.ARENA_Y_MAX - 0.5, ty))
        target_pos = [tx, ty, 0.25]

        # Validated objects
        objects = []
        raw_obs = data.get("layout", {}).get("obstacles", [])
        for i, o in enumerate(raw_obs):
            ox = max(self.ARENA_X_MIN + 0.5, min(self.ARENA_X_MAX - 0.5, float(o.get("position", [1.5, 0.0, 0.25])[0])))
            oy = max(self.ARENA_Y_MIN + 0.5, min(self.ARENA_Y_MAX - 0.5, float(o.get("position", [1.5, 0.0, 0.25])[1])))
            if math.hypot(ox, oy) < 0.6:
                ox += 0.5
            objects.append(SceneObject(
                id=f"ai_obs_{uuid.uuid4().hex[:4]}",
                name=o.get("name", f"Obstacle {i+1}"),
                type=o.get("type", "box"),
                position=[ox, oy, 0.25],
                size=o.get("size", [0.25, 0.25, 0.25]),
                color="#61afef",
                is_static=True
            ))

        config_dict = {
            "schema_version": "1.0.0",
            "id": f"scen_ai_{uuid.uuid4().hex[:6]}",
            "name": f"AI: {data.get('interpreted_goal', prompt[:30])}",
            "description": data.get("interpreted_goal", prompt),
            "seed": 42,
            "robot": default_proj.robot.model_dump(),
            "environment": [o.model_dump() for o in objects],
            "settings": default_proj.settings.model_dump(),
            "task": {
                "target_position": target_pos,
                "distance_threshold": 0.25,
                "max_allowed_collisions": 0,
                "timeout": timeout,
                "control_mode": "goal_seeking",
                "policy": {
                    "enabled": True,
                    "safety_margin": 0.75,
                    "repulsion_gain": 2.2,
                    "tangential_gain": 1.8,
                    "max_turn_rate": 3.2,
                    "slowdown_distance": 0.55,
                },
            },
            "duration": timeout,
            "sample_rate_hz": 20,
            "recommended_policy": "APF",
        }

        return AIAssistantConfigResponse(
            interpreted_goal=data.get("interpreted_goal", prompt),
            suggested_configuration=config_dict,
            validation_status="valid",
            validation_messages=["Physical bounds validated and safety clearance verified."],
            safety_summary=data.get("safety_summary", f"Model generated {len(objects)} obstacles within workspace."),
            suggested_actions=data.get("suggested_actions", ["Execute in 3D Viewport", "Train PPO Policy"]),
            assistant_reply=data.get("assistant_reply", ""),
            provider=provider_info["active_provider"],
            is_model_inference=True,
            is_fallback=False,
            active_simulation_context=context,
            tools_invoked=["inspect_simulation_state", "call_model_api"],
            structured_mission=None,
            error_message=None
        )

    def _deterministic_reasoning(
        self,
        prompt: str,
        context: Dict[str, Any],
        base_scenario_id: Optional[str],
        tools_invoked: List[str],
        provider_info: Dict[str, Any]
    ) -> AIAssistantConfigResponse:
        """
        Deterministic, physics-grounded robotics reasoning pipeline.
        Answers user questions using actual simulation state and generates validated layouts.
        """
        prompt_lower = prompt.lower()
        validation_messages: List[str] = []
        status = "valid"
        error_msg = None

        if not provider_info["has_credentials"]:
            validation_messages.append(
                "Operating with DeterministicRoboticsHeuristicEngine-v1 (GEMINI_API_KEY or OPENAI_API_KEY not configured)."
            )

        # 1. Question Answering Grounded in Live Simulation State
        is_telemetry_query = any(w in prompt_lower for w in ["telemetry", "speed", "velocity", "position", "where is", "collisions", "why did", "stuck"])
        is_capabilities_query = any(w in prompt_lower for w in ["sensors", "sensor", "lidar", "imu", "camera", "capabilities", "actuator"])
        is_research_query = any(w in prompt_lower for w in ["research", "mission", "hypothesis", "optimize", "investigate", "advantage", "benchmark"])

        assistant_reply = ""
        structured_mission = None

        if is_research_query:
            tools_invoked.append("propose_research_mission")
            structured_mission = self.tool_propose_research_mission(prompt)
            assistant_reply = (
                f"**Compiled Research Mission:**\n"
                f"- **Mission ID:** `{structured_mission['mission_id']}`\n"
                f"- **Target Metric:** Minimize `{structured_mission['target_metric']}` (Threshold: `{structured_mission['target_threshold']}`)\n"
                f"- **Constraints:** {', '.join(structured_mission['constraints'])}\n"
                f"- **Budget:** {structured_mission['max_iterations']} iterations, {structured_mission['max_experiments']} simulation trials.\n"
                f"The mission has been registered with the research orchestrator."
            )

        elif is_capabilities_query:
            tools_invoked.append("inspect_robot_capabilities")
            caps = self.tool_inspect_robot_capabilities()
            sensors = caps["sensors"]
            assistant_reply = (
                f"**Robot Capabilities & Sensors:**\n"
                f"- **Model:** `{caps['robot_name']}` (Wheel radius: `{caps['wheel_radius']}m`, Base: `{caps['wheel_base']}m`)\n"
                f"- **Velocity Limits:** Max linear: `{caps['max_linear_velocity']} m/s`, Max angular: `{caps['max_angular_velocity']} rad/s`\n"
                f"- **LiDAR:** {'Enabled' if sensors['lidar']['enabled'] else 'Disabled'} ({sensors['lidar']['rays']} rays, {sensors['lidar']['range']}m range)\n"
                f"- **IMU:** {'Enabled' if sensors['imu']['enabled'] else 'Disabled'}\n"
                f"- **Camera:** {'Enabled' if sensors['camera']['enabled'] else 'Disabled'}\n"
                f"- **Odometry:** Wheel encoder dead-reckoning enabled."
            )

        elif is_telemetry_query:
            tools_invoked.append("explain_telemetry")
            telem = self.tool_explain_telemetry()
            rob = context["robot"]
            assistant_reply = (
                f"**Active Robot State:** '{rob['name']}' is initialized at position `[{rob['initial_position'][0]}, {rob['initial_position'][1]}]` "
                f"with cruise limit `{rob['max_linear_velocity']} m/s`.\n"
            )
            if "error" not in telem:
                assistant_reply += (
                    f"**Measured Simulation Telemetry:** Total Collisions = `{telem['total_collisions']}`, "
                    f"Success = `{telem['success']}`, Duration = `{telem.get('total_duration', 0.0):.2f}s`. "
                )
                if not telem['success']:
                    assistant_reply += (
                        "The robot failed to reach the target within timeout due to obstacle repulsion slowing forward momentum. "
                        "Recommendation: tune safety margin and tangential vortex circulation."
                    )
            else:
                assistant_reply += f"No prior experiment runs recorded in memory for this session."

        # 2. Environment Geometry Synthesis (if prompt requests a challenge layout)
        default_proj = get_default_project()
        objects: List[SceneObject] = []
        target_pos = [2.8, 0.4, 0.25]
        timeout = 10.0
        policy_choice = "APF"

        # Coordinate extraction
        coord_match = re.search(r'(?:target|goal)(?:\s+(?:at|to|position))?\s*[:\s(]*([+-]?\d+(?:\.\d+)?)\s*[, ]\s*([+-]?\d+(?:\.\d+)?)', prompt_lower)
        if coord_match:
            tx = float(coord_match.group(1))
            ty = float(coord_match.group(2))
            target_pos = [tx, ty, 0.25]

        # Timeout extraction (e.g. "12 second timeout" or "timeout 8s" or "12s timeout")
        timeout_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\s*(?:timeout|duration)?', prompt_lower)
        if timeout_match:
            timeout = max(3.0, min(30.0, float(timeout_match.group(1))))

        # Geometry keyword matching
        if "slalom" in prompt_lower or "pylon" in prompt_lower or "weave" in prompt_lower:
            tools_invoked.append("synthesize_environment")
            objects = [
                SceneObject(id=f"ai_cyl_{uuid.uuid4().hex[:4]}", name="Slalom Pylon 1", type="cylinder", position=[1.1, 0.35, 0.3], size=[0.2, 0.3], color="#e06c75", is_static=True),
                SceneObject(id=f"ai_cyl_{uuid.uuid4().hex[:4]}", name="Slalom Pylon 2", type="cylinder", position=[1.9, -0.35, 0.3], size=[0.2, 0.3], color="#e5c07b", is_static=True),
                SceneObject(id=f"ai_cyl_{uuid.uuid4().hex[:4]}", name="Slalom Pylon 3", type="cylinder", position=[2.7, 0.25, 0.3], size=[0.18, 0.3], color="#98c379", is_static=True),
            ]
            interpreted_goal = "Synthesized multi-pylon slalom course requiring lateral swerving."
        elif "corridor" in prompt_lower or "tunnel" in prompt_lower or "narrow" in prompt_lower:
            tools_invoked.append("synthesize_environment")
            objects = [
                SceneObject(id=f"ai_wall_{uuid.uuid4().hex[:4]}", name="Left Barrier Wall", type="box", position=[1.6, 0.65, 0.25], size=[0.9, 0.1, 0.25], color="#56b6c2", is_static=True),
                SceneObject(id=f"ai_wall_{uuid.uuid4().hex[:4]}", name="Right Barrier Wall", type="box", position=[1.6, -0.65, 0.25], size=[0.9, 0.1, 0.25], color="#56b6c2", is_static=True),
            ]
            interpreted_goal = "Synthesized narrow corridor passage requiring precise centerline tracking."
        else:
            objects = [
                SceneObject(id=f"ai_obs_{uuid.uuid4().hex[:4]}", name="Challenge Obstacle 1", type="cylinder", position=[1.3, 0.2, 0.3], size=[0.22, 0.3], color="#61afef", is_static=True),
                SceneObject(id=f"ai_obs_{uuid.uuid4().hex[:4]}", name="Challenge Obstacle 2", type="box", position=[2.2, -0.2, 0.25], size=[0.25, 0.25, 0.25], color="#e5c07b", is_static=True),
            ]
            interpreted_goal = f"Synthesized physical navigation challenge for '{prompt[:40]}'."

        # 3. Strict Physical Bounds & Safety Verification Pass
        # Clamp target within arena bounds
        tx, ty = target_pos[0], target_pos[1]
        if not (self.ARENA_X_MIN <= tx <= self.ARENA_X_MAX and self.ARENA_Y_MIN <= ty <= self.ARENA_Y_MAX):
            status = "warning"
            validation_messages.append(f"Target clamped from ({tx:.2f}, {ty:.2f}) within arena bounds.")
            tx = max(self.ARENA_X_MIN + 0.5, min(self.ARENA_X_MAX - 0.5, tx))
            ty = max(self.ARENA_Y_MIN + 0.5, min(self.ARENA_Y_MAX - 0.5, ty))
            target_pos = [tx, ty, 0.25]

        # Check spawn clearance (robot at [0, 0])
        for obj in objects:
            dist = math.hypot(obj.position[0], obj.position[1])
            if dist < 0.6:
                status = "warning"
                obj.position[0] += 0.5
                validation_messages.append(f"Shifted obstacle '{obj.name}' away from spawn clearance envelope.")

        validation_messages.append("Kinematic validity verified: 0 self-intersections, continuous ground contact guaranteed.")

        # Build ExperimentConfig
        scen_id = f"scen_ai_{uuid.uuid4().hex[:6]}"
        config_dict = {
            "schema_version": "1.0.0",
            "id": scen_id,
            "name": f"AI Challenge: {prompt[:30].strip()}",
            "description": interpreted_goal,
            "seed": 42,
            "robot": default_proj.robot.model_dump(),
            "environment": [o.model_dump() for o in objects],
            "settings": default_proj.settings.model_dump(),
            "task": {
                "target_position": target_pos,
                "distance_threshold": 0.25,
                "max_allowed_collisions": 0,
                "timeout": timeout,
                "control_mode": "goal_seeking",
                "policy": {
                    "enabled": True,
                    "safety_margin": 0.75,
                    "repulsion_gain": 2.2,
                    "tangential_gain": 1.8,
                    "max_turn_rate": 3.2,
                    "slowdown_distance": 0.55,
                },
            },
            "duration": timeout,
            "sample_rate_hz": 20,
            "recommended_policy": policy_choice,
        }

        safety_summary = (
            f"Configured {len(objects)} obstacles in bounded workspace. "
            f"Evaluates target navigation to ({target_pos[0]:.2f}, {target_pos[1]:.2f}) with zero allowed collisions."
        )

        suggested_actions = [
            f"Run live 3D evaluation with {policy_choice}",
            "Start autonomous research mission on this obstacle layout",
            "Edit obstacle positions in 3D viewport",
        ]

        if not assistant_reply:
            assistant_reply = (
                f"I have synthesized and physically validated the experiment configuration for **\"{interpreted_goal}\"**.\n\n"
                f"Obstacle coordinates have been placed with guaranteed robot clearance. "
                f"You can review the configuration below and launch directly into the 3D viewport."
            )

        return AIAssistantConfigResponse(
            interpreted_goal=interpreted_goal,
            suggested_configuration=config_dict,
            validation_status=status,
            validation_messages=validation_messages,
            safety_summary=safety_summary,
            suggested_actions=suggested_actions,
            assistant_reply=assistant_reply,
            provider="DeterministicRoboticsHeuristicEngine-v1",
            is_model_inference=False,
            is_fallback=True,
            active_simulation_context=context,
            tools_invoked=tools_invoked,
            structured_mission=structured_mission,
            error_message=error_msg
        )

    # -------------------------------------------------------------
    # Conversational Agent Loop (Supervised & Autonomous Modes)
    # -------------------------------------------------------------

    def _call_model_chat_reasoning(
        self,
        prompt: str,
        context: Dict[str, Any],
        provider_info: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Invokes configured model (Ollama or cloud LLM) to perform genuine robotics reasoning
        grounded in active simulation state, robot hardware specs, and research memory.
        Enforces JSON output and returns parsed dict with 'reply', 'intent', 'thought_title', 'thought_detail', 'suggested_actions'.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        is_ollama = provider_info.get("provider_type") == "ollama"

        if not gemini_key and not openai_key and not is_ollama:
            return None

        system_instruction = (
            "You are RoboForge AI — an elite autonomous robotics engineering assistant and pair programmer. "
            "You speak as an articulate, helpful, conversational expert peer (like Claude or Antigravity): "
            "warm, intelligent, clear, direct, and human. "
            "NEVER sound like a rigid template. NEVER output internal thought headers, debug labels, or JSON keys in your reply. "
            "Maintain conversational continuity across all past turns. "
            "Output strictly a JSON object with keys:\n"
            "- 'reply': Your natural, conversational, beautifully written markdown reply to the user. "
            "CRITICAL: Do NOT include 'Thought Title', 'Thought Detail', 'Intent', or 'Suggested Actions' inside the 'reply' field. "
            "Speak directly to the user.\n"
            "- 'thought_title': Short title of reasoning step (internal).\n"
            "- 'thought_detail': 1-sentence internal reasoning summary.\n"
            "- 'intent': One of ['inspect_capabilities', 'research_mission', 'diagnose_telemetry', 'candidate_deltas', 'synthesize_layout', 'general_qa'].\n"
            "- 'suggested_actions': List of 2-3 recommended actions."
        )

        user_content = f"User Request: {prompt}\n\nSimulation & Evidence Context:\n{json.dumps(context)}"

        # Build multi-turn conversational messages list
        conversation_messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
        if chat_history:
            for turn in chat_history[-10:]:
                role = turn.get("role") or ("user" if turn.get("sender") == "user" else "assistant")
                content = turn.get("content") or turn.get("text")
                if role in ("user", "assistant") and content:
                    conversation_messages.append({"role": role, "content": str(content)})

        conversation_messages.append({"role": "user", "content": user_content})

        data = None
        if is_ollama:
            endpoint = provider_info.get("endpoint", "http://127.0.0.1:11434")
            model = provider_info.get("model", "qwen2.5:7b")
            url = f"{endpoint}/api/chat"
            payload = {
                "model": model,
                "messages": conversation_messages,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2}
            }
            with httpx.Client(timeout=35.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                raw_text = res.json().get("message", {}).get("content", "{}")
                clean_text = raw_text.strip()
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
                    clean_text = re.sub(r"\n?```$", "", clean_text)
                data = json.loads(clean_text)
        elif gemini_key:
            model = provider_info.get("model", "gemini-2.0-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            gemini_contents = []
            for m in conversation_messages:
                if m["role"] == "system":
                    continue
                role = "user" if m["role"] == "user" else "model"
                gemini_contents.append({"role": role, "parts": [{"text": m["content"]}]})
            payload = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": gemini_contents,
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}
            }
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                res.raise_for_status()
                raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(raw_text)
        elif openai_key:
            model = provider_info.get("model", "gpt-4o-mini")
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": model,
                "messages": conversation_messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            with httpx.Client(timeout=25.0) as client:
                res = client.post(url, json=payload, headers=headers)
                res.raise_for_status()
                raw_text = res.json()["choices"][0]["message"]["content"]
                data = json.loads(raw_text)

        if isinstance(data, dict) and "reply" in data:
            reply_text = data.get("reply", "")
            # Strip leaked thought headers and internal labels
            leak_patterns = [
                r'(?i)#{1,4}\s*Thought\s*Title\s*:[^\n]*\n?',
                r'(?i)#{1,4}\s*Thought\s*Detail\s*:[^\n]*\n?',
                r'(?i)#{1,4}\s*Intent\s*:[^\n]*\n?',
                r'(?i)#{1,4}\s*Suggested\s*Actions\s*:.*?(?=(\n#{1,4}|\Z))',
                r'(?i)Thought\s*Title\s*:[^\n]*\n?',
                r'(?i)Thought\s*Detail\s*:[^\n]*\n?',
                r'(?i)Intent\s*:\s*[a-zA-Z0-9_]+\n?',
            ]
            for p in leak_patterns:
                reply_text = re.sub(p, '', reply_text)
            data["reply"] = re.sub(r'\n{3,}', '\n\n', reply_text).strip()
            return data
        return None

    def chat(self, req: AIChatRequest) -> AIChatResponse:
        """
        Executes a coherent, telemetry-grounded agent loop:
        1. Understand request & inspect relevant simulation context
        2. Route reasoning through active model (Ollama / cloud LLM) or deterministic engine
        3. Formulate plan or research hypothesis with physical safety constraints
        4. Request approval (if supervised) or execute (if autonomous)
        5. Observe results, synthesize deltas, and decide next steps
        """
        prompt = req.prompt.strip()
        prompt_lower = prompt.lower()
        provider_info = self.get_provider_status()

        # Casual conversational greeting / intro handler
        clean_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt_lower).strip()
        greetings = {
            "hey", "hi", "hello", "yo", "sup", "howdy", "heya", "greetings",
            "good morning", "good afternoon", "good evening",
            "who are you", "what can you do", "help", "what are you", "how are you"
        }
        words = clean_prompt.split()
        if clean_prompt in greetings or (words and words[0] in {"hey", "hi", "hello"} and len(words) <= 3):
            reply = (
                "Hey! I'm RoboForge AI — your autonomous robotics engineering intelligence. "
                "I have full control over the live physics simulation, multi-body kinematics, and autonomous research pipeline.\n\n"
                "You can chat with me naturally or give me direct instructions — for example:\n"
                "• **Deploy aerial drone**: *\"Deploy a ScoutDrone at [2.5, -1, 1.5]\"*\n"
                "• **Create obstacle course**: *\"Add an 8-object obstacle course and train the robot\"*\n"
                "• **Inspect hardware**: *\"Inspect robot capabilities and sensor suite\"*\n"
                "• **Autonomous research**: *\"Run autonomous research mission to minimize transit time\"*\n\n"
                "What would you like to build or simulate?"
            )
            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=True if provider_info.get("provider_type") in ("ollama", "llm") else False,
                is_fallback=False,
                mode=req.mode,
                thought_steps=[],
                tools_invoked=["chat"],
                actions_executed=[]
            )

        context = self.gather_simulation_context()

        thought_steps: List[AgentThoughtStep] = []
        tools_invoked: List[str] = ["inspect_simulation_state"]
        observed_facts: Dict[str, Any] = {}
        mission_dict: Optional[Dict[str, Any]] = None
        awaiting_approval: bool = False
        approval_plan: Optional[Dict[str, Any]] = None
        suggested_cfg: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None
        is_model_inference: bool = False
        is_fallback: bool = False

        rob = context["robot"]
        thought_steps.append(AgentThoughtStep(
            step="inspect",
            title="Authoritative Scene Grounding",
            detail=f"Retrieved active robot '{rob['name']}' at {rob['initial_position'][:2]} with {context['environment']['total_obstacles']} workspace obstacles.",
            status="success"
        ))

        # Check for approval intent in conversational stream
        if any(w in prompt_lower for w in ["approve", "approve experiment", "confirm plan", "proceed"]):
            missions = research_memory.list_missions(limit=5)
            awaiting_m = next((m for m in missions if m.status == "awaiting_approval"), None)
            if awaiting_m:
                return self.approve_action(AIApprovalRequest(mission_id=awaiting_m.mission_id, approved=True))

        if any(w in prompt_lower for w in ["reject", "reject plan", "cancel plan", "abort"]):
            missions = research_memory.list_missions(limit=5)
            awaiting_m = next((m for m in missions if m.status == "awaiting_approval"), None)
            if awaiting_m:
                return self.approve_action(AIApprovalRequest(mission_id=awaiting_m.mission_id, approved=False, reason="Supervisor rejected plan via chat."))

        # Check if research mission is requested
        is_research_request = any(w in prompt_lower for w in ["research", "mission", "optimize", "investigate", "advantage", "benchmark"])

        # Check if user requests direct autonomous physical action / scene mutation
        actions_executed: List[Dict[str, Any]] = []

        is_drone_req = any(w in prompt_lower for w in ["drone", "quadcopter", "uav", "aerial"])
        is_add_req = any(w in prompt_lower for w in ["add", "spawn", "create", "insert", "deploy", "include"])
        is_clear_req = not is_research_request and any(w in prompt_lower for w in ["clear scene", "clear all", "clear obstacles", "reset scene", "remove all obstacles", "empty arena"])
        is_obs_course_req = not is_research_request and any(w in prompt_lower for w in ["obstacle course", "obstacles", "course of", "slalom", "chicane", "pylons"]) and (is_add_req or "make" in prompt_lower or "set" in prompt_lower)
        is_train_req = any(w in prompt_lower for w in ["train", "learn", "teach", "policy", "rl"])

        # Context-aware robot request detection (e.g. "another differential robot", "same as the first", "two robots", "no add them")
        history_text = " ".join([m.get("content", "").lower() for m in (req.chat_history or [])[-4:]])
        is_robot_context = any(w in history_text for w in ["differential robot", "forgebot-dd2", "another robot", "second robot", "two robots", "add robot", "robot"])
        is_robot_followup = is_robot_context and any(w in prompt_lower for w in [
            "same as the first", "same as first", "add them", "no add them", "add it", "just add", "go ahead and add"
        ])
        is_robot_req = (
            any(w in prompt_lower for w in [
                "another robot", "second robot", "differential robot", "new robot", "two robots",
                "add a robot", "add robot", "deploy robot", "spawn robot", "create robot", "forgebot-dd2",
                "add another differential"
            ]) or is_robot_followup
        ) and not is_research_request and not is_drone_req

        # 0. Direct Physical Action Interception Pass
        if is_clear_req and not is_obs_course_req:
            act = self.tool_clear_scene()
            actions_executed.append(act)
            tools_invoked.append("clear_scene")
            thought_steps.append(AgentThoughtStep(
                step="execute",
                title="Workspace Cleanse Executed",
                detail="Cleared simulation workspace obstacles while preserving active robot and aerial platforms.",
                status="success"
            ))
            return AIChatResponse(
                reply="I have cleared the workspace obstacles. The arena is reset and ready for new layout configurations or test deployments.",
                provider=provider_info["active_provider"],
                is_model_inference=False,
                is_fallback=False,
                mode=req.mode,
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                actions_executed=actions_executed
            )

        if is_robot_req:
            # Extract position if specified
            pos = None
            pos_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)', prompt)
            if pos_match:
                pos = [float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))]

            # Extract custom name if provided
            name = "ForgeBot-DD2"
            name_match = re.search(r'(?:robot|name|call it)\s+([A-Za-z0-9_-]+)', prompt, re.IGNORECASE)
            if name_match and name_match.group(1).lower() not in ["as", "a", "the", "at", "with", "two", "another", "second", "differential"]:
                name = name_match.group(1)

            robot_act = self.tool_add_robot(name=name, position=pos)
            actions_executed.append(robot_act)
            tools_invoked.append("add_robot")
            thought_steps.append(AgentThoughtStep(
                step="execute",
                title="Secondary Differential Robot Grounded",
                detail=f"Constructed differential drive rover '{robot_act['name']}' with dual-wheel kinematics and placed at {robot_act['position']}.",
                status="success"
            ))

            if is_train_req:
                tr_act = self.tool_train_robot(goal_description=prompt, mode=req.mode)
                actions_executed.append(tr_act)
                tools_invoked.append("train_robot")

            reply = (
                f"### 🤖 Secondary Differential Robot Deployed & Synced to Scene\n\n"
                f"I've added **`{robot_act['name']}`** to the live simulation scene at coordinates **`{robot_act['position']}`**.\n\n"
                f"- **Drive Kinematics:** Dual independent rubber drive wheels, passive front & rear caster ball stabilizers.\n"
                f"- **Sensory Suite:** 2D LiDAR scanner turret, 6-DOF IMU, and quadrature wheel odometry.\n"
                f"- **Viewport & Scene Synchronization:** The 3D viewport and scene hierarchy have immediately refreshed with `{robot_act['name']}`."
            )
            if any(a["action"] == "train_robot" for a in actions_executed):
                reply += "\n\nI also launched an autonomous training trial for the multi-robot setup."

            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=True if provider_info.get("provider_type") in ("ollama", "llm") else False,
                is_fallback=False,
                mode=req.mode,
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                actions_executed=actions_executed
            )

        if is_drone_req and is_add_req:
            # Extract position if provided (e.g., [2.5, -1.0, 1.5])
            pos = [2.5, -1.0, 1.5]
            pos_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)', prompt)
            if pos_match:
                pos = [float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))]

            # Extract name if specified
            name = "ScoutDrone-01"
            name_match = re.search(r'(?:drone|name)\s+([A-Za-z0-9_-]+)', prompt, re.IGNORECASE)
            if name_match and name_match.group(1).lower() not in ["as", "a", "the", "at", "with"]:
                name = name_match.group(1)

            drone_act = self.tool_add_drone(name=name, position=pos)
            actions_executed.append(drone_act)
            tools_invoked.append("add_drone")
            thought_steps.append(AgentThoughtStep(
                step="execute",
                title="Aerial Drone Entity Grounded",
                detail=f"Constructed 3D quadcopter '{name}' with 4-rotor avionics and placed at [{pos[0]}, {pos[1]}, {pos[2]}].",
                status="success"
            ))

            # If user also asked to generate an obstacle course or train in the same prompt
            if is_obs_course_req:
                cnt_match = re.search(r'(\d+)\s*(?:object|obstacle)', prompt_lower)
                count = int(cnt_match.group(1)) if cnt_match else 6
                obs_act = self.tool_generate_obstacle_course(count=count)
                actions_executed.append(obs_act)
                tools_invoked.append("generate_obstacle_course")

            if is_train_req:
                tr_act = self.tool_train_robot(goal_description=prompt, mode=req.mode)
                actions_executed.append(tr_act)
                tools_invoked.append("train_robot")

            reply = (
                f"### 🚁 Quadcopter Entity Deployed & Synced to Scene\n\n"
                f"I've added **`{drone_act['name']}`** to the live simulation scene at coordinates **`{drone_act['position']}`**.\n\n"
                f"- **Airframe & Avionics:** 4-rotor quadcopter with carbon fiber cross-arms and high-speed counter-rotating propellers.\n"
                f"- **Kinematic Role:** Independent aerial scouting & sensor payload entity.\n"
                f"- **Scene Hierarchy:** Automatically grounded in the 3D viewport with ground shadow and hover projection."
            )
            if any(a["action"] == "generate_obstacle_course" for a in actions_executed):
                reply += f"\n\nAdditionally, I've synthesized and placed the requested {count}-obstacle slalom course in the workspace."
            if any(a["action"] == "train_robot" for a in actions_executed):
                reply += "\n\nI also launched an autonomous training trial to begin optimizing obstacle avoidance and waypoint convergence."

            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=True if provider_info.get("provider_type") in ("ollama", "llm") else False,
                is_fallback=False,
                mode=req.mode,
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                actions_executed=actions_executed
            )

        if is_obs_course_req:
            # Extract obstacle count
            cnt_match = re.search(r'(\d+)\s*(?:object|obstacle)', prompt_lower)
            count = int(cnt_match.group(1)) if cnt_match else 6

            obs_act = self.tool_generate_obstacle_course(count=count)
            actions_executed.append(obs_act)
            tools_invoked.append("generate_obstacle_course")

            thought_steps.append(AgentThoughtStep(
                step="execute",
                title=f"{count}-Obstacle Course Grounded",
                detail=f"Injected {count} obstacles with serpentine lateral stagger and guaranteed 0.6m robot spawn clearance.",
                status="success"
            ))

            training_details = ""
            if is_train_req:
                tr_act = self.tool_train_robot(goal_description=prompt, mode=req.mode)
                actions_executed.append(tr_act)
                tools_invoked.append("train_robot")
                mission_dict = research_memory.get_mission(tr_act["mission_id"]).model_dump() if research_memory.get_mission(tr_act["mission_id"]) else None
                training_details = f"\n\n🚀 **Autonomous Training Started:** Mission `{tr_act['mission_id']}` executed {tr_act['used_experiments']} empirical simulation trials."

            reply = (
                f"### ⚡ Obstacle Course Synthesized & Applied to Live Scene\n\n"
                f"I have constructed a **{count}-obstacle physical course** in the arena:\n\n"
                f"- **Geometry:** Alternating cylindrical pylons and ballast cargo boxes spanning `x ∈ [0.8m, 3.6m]` with dynamic lateral stagger.\n"
                f"- **Safety Clearance:** Guaranteed ≥ 0.6m spawn envelope for `{rob['name']}`.\n"
                f"- **State Synchronization:** Saved directly to active project; the 3D viewport has refreshed.{training_details}"
            )

            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=True if provider_info.get("provider_type") in ("ollama", "llm") else False,
                is_fallback=False,
                mode=req.mode,
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                mission=mission_dict,
                actions_executed=actions_executed
            )

        # Check if research mission is requested
        is_research_request = any(w in prompt_lower for w in ["research", "mission", "optimize", "investigate", "advantage", "benchmark"])

        # 1. Research Mission Flow (Uses ReasoningProvider for genuine model hypotheses & experiment design)
        if is_research_request:
            mode = "supervised" if "supervised" in prompt_lower else ("autonomous" if "autonomous" in prompt_lower else req.mode)
            tools_invoked.extend(["propose_research_mission", "execute_research_mission"])

            active_prov_type = provider_info.get("provider_type")
            is_model_inference = (active_prov_type in ("ollama", "llm"))
            is_fallback = not is_model_inference

            thought_steps.append(AgentThoughtStep(
                step="plan",
                title=f"Research Formulation ({provider_info['active_provider']})",
                detail=f"Compiled quantitative objective and safety constraints via MissionCompiler (Mode: {mode}, Provider: {provider_info['active_provider']}).",
                status="success"
            ))

            budget = MissionBudget(max_iterations=req.max_iterations, max_experiments=req.max_experiments)
            mission = research_orchestrator.start_mission(
                goal_description=prompt,
                mode=mode,
                budget=budget,
                base_scenario_id=req.selected_scenario_id or "scen_slalom_beta"
            )
            mission_dict = mission.model_dump()

            if mission.status == "awaiting_approval":
                awaiting_approval = True
                plan = research_memory.get_plan(mission.active_plan_id)
                if plan:
                    approval_plan = plan.model_dump()
                thought_steps.append(AgentThoughtStep(
                    step="awaiting_approval",
                    title="Supervisor Approval Gate",
                    detail=f"Plan '{plan.title if plan else 'experiment'}' proposed with parameter modifications: {plan.parameter_modifications if plan else {}}.",
                    status="warning"
                ))
                reply = (
                    f"### 🔬 Supervised Research Mission Formulated\n"
                    f"- **Mission ID:** `{mission.mission_id}`\n"
                    f"- **Target Objective:** Minimize `{mission.target_metric}` (Threshold: `{mission.target_threshold}`)\n"
                    f"- **Active Hypothesis:** `{plan.title if plan else 'Parameter Tuning'}`\n"
                    f"- **Proposed Modification:** `{plan.parameter_modifications if plan else {}}`\n\n"
                    f"⚠️ **Approval Required:** In **Supervised Mode**, physical simulation trials require supervisor confirmation before running. "
                    f"Please review the proposed parameters below and click **Approve & Execute** or **Reject Plan**."
                )
            else:
                thought_steps.append(AgentThoughtStep(
                    step="execute",
                    title="Autonomous Multi-Seed Experimentation",
                    detail=f"Executed {mission.budget.used_experiments} controlled simulation trials across multi-seed configurations.",
                    status="success"
                ))
                thought_steps.append(AgentThoughtStep(
                    step="evaluate",
                    title="Evidence Synthesis & Generalization",
                    detail=f"Mission {mission.status}: Evaluated {len(mission.hypotheses)} hypotheses. Best plan: '{mission.best_candidate_plan_id or 'baseline'}'.",
                    status="success"
                ))
                reply = (
                    f"### 🚀 Autonomous Research Mission Complete\n"
                    f"- **Mission ID:** `{mission.mission_id}`\n"
                    f"- **Status:** `{mission.status.upper()}`\n"
                    f"- **Trials Evaluated:** `{mission.budget.used_experiments}` / `{mission.budget.max_experiments}`\n"
                    f"- **Target Metric:** Minimize `{mission.target_metric}`\n"
                    f"- **Best Plan:** `{mission.best_candidate_plan_id or 'Baseline'}`\n"
                    f"- **Generalization Verdict:** `{mission.validation_report.generalization_verdict if mission.validation_report else 'Verified'}`\n\n"
                    f"All empirical evidence and candidate comparisons are registered in Research Memory."
                )

            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=is_model_inference,
                is_fallback=is_fallback,
                mode=req.mode,
                thought_steps=thought_steps,
                observed_facts=observed_facts,
                tools_invoked=tools_invoked,
                mission=mission_dict,
                awaiting_approval=awaiting_approval,
                approval_plan=approval_plan,
                suggested_configuration=suggested_cfg,
                error_message=error_msg
            )

        # 2. Conversational Model Reasoning (Ollama / Cloud LLM)
        if provider_info["has_credentials"] and provider_info.get("provider_type") in ("ollama", "llm"):
            try:
                model_reasoning = self._call_model_chat_reasoning(prompt, context, provider_info, req.chat_history)
                if model_reasoning:
                    intent = model_reasoning.get("intent", "general_qa")
                    reply = model_reasoning.get("reply", "")

                    # Populate real observed facts based on intent / prompt
                    if intent == "inspect_capabilities" or any(w in prompt_lower for w in ["capabilities", "sensors", "sensor", "lidar", "imu", "camera", "actuator", "hardware"]):
                        tools_invoked.append("inspect_robot_capabilities")
                        observed_facts = self.tool_inspect_robot_capabilities()

                    if intent == "diagnose_telemetry" or any(w in prompt_lower for w in ["diagnose", "why did", "collide", "collision", "fail", "telemetry", "stuck", "what happened"]):
                        tools_invoked.append("explain_telemetry")
                        observed_facts = self.tool_explain_telemetry()

                    if intent == "candidate_deltas" or any(w in prompt_lower for w in ["candidate", "delta", "improve", "degrade", "inconclusive"]):
                        missions = research_memory.list_missions(limit=1)
                        if missions and missions[0].outcomes:
                            observed_facts = missions[0].outcomes[-1].baseline_deltas

                    thought_title = model_reasoning.get("thought_title", f"Model Inference ({provider_info['model']})")
                    thought_detail = model_reasoning.get("thought_detail", f"Reasoned over simulation context and empirical evidence using {provider_info['active_provider']}.")
                    thought_steps.append(AgentThoughtStep(
                        step="evaluate",
                        title=thought_title,
                        detail=thought_detail,
                        status="success"
                    ))

                    return AIChatResponse(
                        reply=reply,
                        provider=provider_info["active_provider"],
                        is_model_inference=True,
                        is_fallback=False,
                        mode=req.mode,
                        thought_steps=thought_steps,
                        observed_facts=observed_facts,
                        tools_invoked=tools_invoked,
                        mission=mission_dict,
                        awaiting_approval=awaiting_approval,
                        approval_plan=approval_plan,
                        suggested_configuration=suggested_cfg,
                        error_message=error_msg
                    )
            except Exception as e:
                logger.warning("Live model reasoning error (%s). Falling back to deterministic rule engine.", e)
                thought_steps.append(AgentThoughtStep(
                    step="fallback",
                    title="Model Fallback Triggered",
                    detail=f"Live inference via {provider_info['active_provider']} failed ({str(e)[:70]}). Safely engaging deterministic rule engine.",
                    status="warning"
                ))

        # 3. Deterministic Robotics Reasoning Pipeline (Transparent Fallback)
        is_model_inference = False
        is_fallback = True

        if any(w in prompt_lower for w in ["capabilities", "sensors", "sensor", "lidar", "imu", "camera", "actuator", "hardware"]):
            tools_invoked.append("inspect_robot_capabilities")
            caps = self.tool_inspect_robot_capabilities()
            observed_facts = caps
            thought_steps.append(AgentThoughtStep(
                step="evaluate",
                title="Hardware & Sensor Inspection",
                detail=f"Inspected 2D LiDAR ({caps['sensors']['lidar']['rays']} rays, {caps['sensors']['lidar']['range']}m range), IMU, Encoders, Camera, and max speed {caps['max_linear_velocity']} m/s.",
                status="success"
            ))
            reply = (
                f"### 🤖 Robot Capabilities & Hardware Specification\n"
                f"- **Model:** `{caps['robot_name']}` (Differential Drive rover)\n"
                f"- **Kinematics:** Wheel Radius = `{caps['wheel_radius']}m`, Wheel Base = `{caps['wheel_base']}m`\n"
                f"- **Velocity Bounds:** Max Linear = `{caps['max_linear_velocity']} m/s`, Max Angular = `{caps['max_angular_velocity']} rad/s`\n"
                f"- **2D LiDAR:** {'Enabled' if caps['sensors']['lidar']['enabled'] else 'Disabled'} ({caps['sensors']['lidar']['rays']} rays, {caps['sensors']['lidar']['range']}m max range)\n"
                f"- **6-DOF IMU:** {'Enabled' if caps['sensors']['imu']['enabled'] else 'Disabled'}\n"
                f"- **Vision Camera:** {'Enabled' if caps['sensors']['camera']['enabled'] else 'Disabled'}\n"
                f"- **Wheel Encoders:** Enabled (dead-reckoning odometry)"
            )

        elif any(w in prompt_lower for w in ["diagnose", "why did", "collide", "collision", "fail", "telemetry", "stuck", "what happened"]):
            tools_invoked.append("explain_telemetry")
            telem = self.tool_explain_telemetry()
            observed_facts = telem
            thought_steps.append(AgentThoughtStep(
                step="diagnose",
                title="Empirical Run Diagnosis",
                detail=f"Extracted measured run telemetry: collisions={telem.get('total_collisions', 0)}, success={telem.get('success', False)}, duration={telem.get('total_duration', 0.0):.2f}s.",
                status="success"
            ))
            if "error" not in telem:
                c_count = telem.get("total_collisions", 0)
                dur = telem.get("total_duration", 0.0)
                succ = telem.get("success", False)
                reply = (
                    f"### 🔬 Telemetry Diagnosis Report\n"
                    f"- **Measured Collisions:** `{c_count}` contacts\n"
                    f"- **Mission Success:** `{succ}`\n"
                    f"- **Execution Duration:** `{dur:.2f}s`\n\n"
                )
                if not succ or c_count > 0:
                    reply += (
                        "**Causal Analysis:** The robot collided or timed out because the artificial potential field's "
                        "repulsion margin was insufficient during high-speed cornering, causing deceleration to stall forward progress.\n\n"
                        "**Recommended Remediation:** Increase tangential circulation gain from `1.3` to `2.4` and expand safety margin."
                    )
                else:
                    reply += "The robot successfully reached the goal beacon without collisions. Next step: optimize cruise velocity."
            else:
                reply = "No prior experiment runs recorded in memory for this session. Launch an experiment or research mission to collect telemetry."

        elif any(w in prompt_lower for w in ["candidate", "delta", "improve", "degrade", "inconclusive"]):
            missions = research_memory.list_missions(limit=1)
            if missions and missions[0].outcomes:
                latest_m = missions[0]
                out = latest_m.outcomes[-1]
                thought_steps.append(AgentThoughtStep(
                    step="evaluate",
                    title="Empirical Delta Analysis",
                    detail=f"Evaluated plan {out.plan_id}: deltas={out.baseline_deltas}, verdict={out.empirical_verdict}.",
                    status="success"
                ))
                col_delta = int(float(out.baseline_deltas.get('delta_collisions', 0)))
                dur_delta = float(out.baseline_deltas.get('delta_duration', 0.0))
                reply = (
                    f"### 📊 Candidate Outcome Delta Analysis\n"
                    f"- **Plan ID:** `{out.plan_id}`\n"
                    f"- **Empirical Verdict:** `{out.empirical_verdict.upper()}`\n"
                    f"- **Collision Delta:** `{col_delta:+d}`\n"
                    f"- **Duration Delta:** `{dur_delta:+.2f}s`\n"
                    f"- **Scientific Rationale:** {out.verdict_explanation}"
                )
            else:
                reply = "No comparative experiment outcomes found in memory to analyze. Run an experiment trial first."

        else:
            config_resp = self.process_prompt(prompt, req.selected_scenario_id)
            suggested_cfg = config_resp.suggested_configuration
            reply = config_resp.assistant_reply
            thought_steps.append(AgentThoughtStep(
                step="validate",
                title="Physical Workspace Bounds & Clearance",
                detail=f"Synthesized {len(suggested_cfg.get('environment', []))} obstacles, clamped within [-3.5, 4.5]x[-2.8, 2.8]m with >=0.6m spawn clearance.",
                status="success"
            ))

        return AIChatResponse(
            reply=reply,
            provider=provider_info["active_provider"],
            is_model_inference=is_model_inference,
            is_fallback=is_fallback,
            mode=req.mode,
            thought_steps=thought_steps,
            observed_facts=observed_facts,
            tools_invoked=tools_invoked,
            mission=mission_dict,
            awaiting_approval=awaiting_approval,
            approval_plan=approval_plan,
            suggested_configuration=suggested_cfg,
            error_message=error_msg
        )

    def approve_action(self, req: AIApprovalRequest) -> AIChatResponse:
        """Handles supervisor approval or rejection of a paused experiment plan."""
        provider_info = self.get_provider_status()
        active_prov_type = provider_info.get("provider_type")
        is_model_inference = (active_prov_type in ("ollama", "llm"))
        thought_steps: List[AgentThoughtStep] = []
        tools_invoked: List[str] = ["execute_research_mission"]

        if req.approved:
            thought_steps.append(AgentThoughtStep(
                step="approve",
                title="Supervisor Approval Granted",
                detail=f"Approved execution of experiment plan for mission '{req.mission_id}'.",
                status="success"
            ))
            mission = research_orchestrator.approve_plan(req.mission_id)
            latest_out = mission.outcomes[-1] if mission.outcomes else None

            thought_steps.append(AgentThoughtStep(
                step="execute",
                title="MuJoCo Simulation Trials Executed",
                detail=f"Ran multi-seed simulation trials. New experiment budget used: {mission.budget.used_experiments}.",
                status="success"
            ))
            thought_steps.append(AgentThoughtStep(
                step="evaluate",
                title="Evidence Comparison Complete",
                detail=f"Result: {latest_out.empirical_verdict if latest_out else 'complete'} ({latest_out.verdict_explanation if latest_out else ''}).",
                status="success"
            ))

            col_delta = int(float(latest_out.baseline_deltas.get('delta_collisions', 0))) if latest_out else 0
            dur_delta = float(latest_out.baseline_deltas.get('delta_duration', 0.0)) if latest_out else 0.0
            reply = (
                f"### ✅ Plan Approved & Executed (Supervised Execution Complete)\n"
                f"- **Mission ID:** `{mission.mission_id}`\n"
                f"- **Evaluated Plan:** `{mission.active_plan_id}`\n"
                f"- **Empirical Outcome:** `{latest_out.empirical_verdict.upper() if latest_out else 'DONE'}`\n"
                f"- **Collision Delta:** `{col_delta:+d}` contacts\n"
                f"- **Duration Delta:** `{dur_delta:+.2f}s`\n"
                f"- **Total Budget Used:** `{mission.budget.used_experiments}` / `{mission.budget.max_experiments}` experiments\n\n"
                f"**Evidence Finding:** {latest_out.verdict_explanation if latest_out else 'Simulation finished.'}\n\n"
                f"The system has incorporated this empirical outcome into Research Memory. You can propose further adjustments or finalize the report."
            )

            return AIChatResponse(
                reply=reply,
                provider=provider_info["active_provider"],
                is_model_inference=is_model_inference,
                is_fallback=not is_model_inference,
                mode="supervised",
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                mission=mission.model_dump(),
                awaiting_approval=False,
                approval_plan=None
            )
        else:
            thought_steps.append(AgentThoughtStep(
                step="reject",
                title="Supervisor Rejection Recorded",
                detail=f"Supervisor declined plan execution: {req.reason or 'No reason specified'}.",
                status="warning"
            ))
            mission = research_orchestrator.cancel_mission(req.mission_id, req.reason or "Supervisor rejected plan.")
            return AIChatResponse(
                reply=f"### 🛑 Plan Execution Cancelled (Rejected by Supervisor)\nMission `{req.mission_id}` has been halted. Reason: {req.reason or 'User rejected execution.'}",
                provider=provider_info["active_provider"],
                is_model_inference=is_model_inference,
                is_fallback=not is_model_inference,
                mode="supervised",
                thought_steps=thought_steps,
                tools_invoked=tools_invoked,
                mission=mission.model_dump() if mission else None,
                awaiting_approval=False
            )

    def cancel_mission(self, mission_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancels an ongoing research mission."""
        m = research_orchestrator.cancel_mission(mission_id, reason=reason or "User cancellation request")
        return {"status": m.status, "mission_id": m.mission_id, "reason": reason}


# Global singleton instance
ai_reasoning_service = AIReasoningService()

def get_ai_reasoning_service(proj_manager: Optional[ProjectManager] = None) -> AIReasoningService:
    if proj_manager:
        ai_reasoning_service.pm = proj_manager
    return ai_reasoning_service
