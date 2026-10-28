"""AI Experiment Assistant: Converts natural language goals into validated experiment configurations without arbitrary code execution."""

from typing import Optional
from backend.rl_models import AIAssistantConfigResponse
from backend.ai_reasoning_service import get_ai_reasoning_service


class AIExperimentAssistant:
    """
    Unified entry point for AI experiment reasoning.
    Delegates to AIReasoningService for live simulation telemetry context,
    tool inspection, and schema-validated experiment synthesis.
    """

    ARENA_X_MIN = -3.5
    ARENA_X_MAX = 4.5
    ARENA_Y_MIN = -2.8
    ARENA_Y_MAX = 2.8

    @classmethod
    def process_prompt(cls, prompt: str, base_scenario_id: Optional[str] = None) -> AIAssistantConfigResponse:
        return get_ai_reasoning_service().process_prompt(prompt, base_scenario_id)
