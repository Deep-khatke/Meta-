"""
Mission Compiler for RoboForge Scientific Intelligence.
Inspects authoritative robot and scene capabilities, translates natural-language objectives into
measurable success criteria, boundary constraints, and stopping conditions.
"""

import time
import logging
from typing import Any, Dict, Optional

from backend.research.models import ResearchMission, MissionBudget, ActionLogEntry
from backend.research.reasoning_provider import get_reasoning_provider
from backend.research.simulation_tools import simulation_tools
from backend.research.research_memory import research_memory

logger = logging.getLogger("roboforge.research.compiler")


class MissionCompiler:
    """Translates user goals into structured, bounded scientific research missions."""

    def __init__(self, tools=None, memory=None):
        self.tools = tools or simulation_tools
        self.memory = memory or research_memory

    def compile(
        self,
        goal_description: str,
        mode: str = "autonomous",
        budget: Optional[MissionBudget] = None,
        base_scenario_id: Optional[str] = None
    ) -> ResearchMission:
        """
        Compiles a natural language goal into a validated, quantified ResearchMission.
        """
        logger.info("Compiling research mission for goal: '%s' (mode=%s)", goal_description, mode)

        # 1. Inspect authoritative robot and scene environment
        env_info = self.tools.inspect_current_environment()

        # 2. Derive quantitative targets and constraints via reasoning provider
        provider = get_reasoning_provider()
        interpreted = provider.interpret_goal(goal_description, env_info)

        # 3. Determine base scenario for testing
        scen_id = base_scenario_id or "scen_mvqgu87z"
        base_scen = self.tools.get_base_scenario(scen_id)
        if not base_scen:
            # Pick first available or fallback
            scen_id = "scen_mvqgu87z"

        mission = ResearchMission(
            title=interpreted.get("title", f"Investigation: {goal_description[:40]}"),
            goal_description=goal_description,
            mode=mode,
            status="compiling",
            target_metric=interpreted.get("target_metric", "total_collisions"),
            target_threshold=interpreted.get("target_threshold", 0.0),
            optimization_direction=interpreted.get("optimization_direction", "minimize"),
            constraints=interpreted.get("constraints", []),
            budget=budget or MissionBudget(),
            base_scenario_id=scen_id
        )

        # Add compiler action log
        mission.action_log.append(ActionLogEntry(
            step="mission_compilation",
            action="compile_objectives",
            status="success",
            details={
                "target_metric": mission.target_metric,
                "target_threshold": mission.target_threshold,
                "constraints": mission.constraints,
                "base_scenario_id": mission.base_scenario_id,
                "reasoning_engine": provider.name
            },
            message=f"Mission successfully compiled with target: {mission.target_metric} ({mission.optimization_direction} towards {mission.target_threshold})."
        ))

        mission.status = "created"
        self.memory.save_mission(mission)
        return mission


# Global instance
mission_compiler = MissionCompiler()
