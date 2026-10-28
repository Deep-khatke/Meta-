"""
RoboForge Scientific Intelligence Subsystem.
Automates robotics research: goal compilation, baseline establishment, competing hypotheses,
controlled experimentation, empirical evidence analysis, and held-out generalization validation.
"""

from backend.research.models import (
    ResearchMission, MissionBudget, Hypothesis, ControlledExperimentPlan,
    ExperimentOutcome, ValidationReport, HeldOutValidationResult, ActionLogEntry,
    TaskCompatibilitySignature, ComparativeBenchmarkResult, ComparativeBenchmarkReport
)
from backend.research.candidate_ranker import CandidateRanker, candidate_ranker
from backend.research.comparative_evaluator import ComparativeEvaluator, comparative_evaluator
from backend.research.reasoning_provider import (
    ReasoningProvider, DeterministicRuleEngine, LanguageModelReasoningProvider, get_reasoning_provider
)
from backend.research.simulation_tools import SimulationTools, simulation_tools
from backend.research.research_memory import ResearchMemory, research_memory
from backend.research.mission_compiler import MissionCompiler, mission_compiler
from backend.research.hypothesis_engine import HypothesisEngine, hypothesis_engine
from backend.research.experiment_designer import ExperimentDesigner, experiment_designer
from backend.research.evidence_analyzer import EvidenceAnalyzer, evidence_analyzer
from backend.research.research_validator import ResearchValidator, research_validator
from backend.research.research_orchestrator import ResearchOrchestrator, research_orchestrator
from backend.research.research_api import router as research_router

__all__ = [
    "ResearchMission",
    "MissionBudget",
    "Hypothesis",
    "ControlledExperimentPlan",
    "ExperimentOutcome",
    "ValidationReport",
    "HeldOutValidationResult",
    "ActionLogEntry",
    "ReasoningProvider",
    "DeterministicRuleEngine",
    "LanguageModelReasoningProvider",
    "get_reasoning_provider",
    "SimulationTools",
    "simulation_tools",
    "ResearchMemory",
    "research_memory",
    "MissionCompiler",
    "mission_compiler",
    "HypothesisEngine",
    "hypothesis_engine",
    "ExperimentDesigner",
    "experiment_designer",
    "EvidenceAnalyzer",
    "evidence_analyzer",
    "ResearchValidator",
    "research_validator",
    "ResearchOrchestrator",
    "research_orchestrator",
    "research_router",
]
