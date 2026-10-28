"""
FastAPI router for RoboForge Scientific Intelligence Subsystem.
Exposes mission lifecycle, supervised approvals, evidence queries, and reproducible report exports.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.research.models import (
    ResearchMission, MissionBudget, Hypothesis, ControlledExperimentPlan,
    ComparativeBenchmarkReport
)
from backend.research.research_orchestrator import research_orchestrator
from backend.research.research_memory import research_memory
from backend.research.reasoning_provider import get_reasoning_provider

router = APIRouter(prefix="/api/research", tags=["Scientific Intelligence"])


class CreateMissionRequest(BaseModel):
    goal_description: str = Field(..., description="Natural language robotics research goal")
    mode: str = Field(default="autonomous", description="'supervised' or 'autonomous'")
    base_scenario_id: Optional[str] = Field(default=None, description="Starting benchmark scenario ID")
    max_iterations: int = Field(default=5, ge=1, le=20)
    max_experiments: int = Field(default=8, ge=1, le=30)
    max_time_seconds: float = Field(default=120.0, ge=10.0, le=600.0)


@router.post("/missions", response_model=ResearchMission)
def create_and_start_mission(req: CreateMissionRequest):
    """Initializes and begins execution of a robotics scientific research mission."""
    try:
        budget = MissionBudget(
            max_iterations=req.max_iterations,
            max_experiments=req.max_experiments,
            max_time_seconds=req.max_time_seconds
        )
        mission = research_orchestrator.start_mission(
            goal_description=req.goal_description,
            mode=req.mode,
            budget=budget,
            base_scenario_id=req.base_scenario_id
        )
        return mission
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start mission: {str(e)}")


@router.get("/missions", response_model=List[ResearchMission])
def list_research_missions(limit: int = Query(default=30, ge=1, le=100)):
    """Lists recent research missions sorted by creation time descending."""
    return research_memory.list_missions(limit=limit)


@router.get("/missions/{mission_id}", response_model=ResearchMission)
def get_mission_status(mission_id: str):
    """Retrieves full details, hypotheses, and action log for a research mission."""
    mission = research_memory.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    return mission


@router.post("/missions/{mission_id}/approve", response_model=ResearchMission)
def approve_experiment_plan(mission_id: str):
    """Grants supervisor approval for a pending experiment plan in supervised mode."""
    try:
        return research_orchestrator.approve_plan(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/missions/{mission_id}/step", response_model=ResearchMission)
def step_research_mission(mission_id: str):
    """Manually triggers one state transition step in the scientific research loop."""
    try:
        return research_orchestrator.step_mission(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/missions/{mission_id}/cancel", response_model=ResearchMission)
def cancel_research_mission(mission_id: str, reason: str = Query(default="User cancellation request")):
    """Cancels an active or pending research mission."""
    try:
        return research_orchestrator.cancel_mission(mission_id, reason=reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/missions/{mission_id}/hypotheses", response_model=List[Hypothesis])
def get_mission_hypotheses(mission_id: str):
    """Retrieves all competing hypotheses and accumulated evidence for a mission."""
    return research_memory.list_hypotheses_for_mission(mission_id)


@router.get("/missions/{mission_id}/report")
def get_mission_report(mission_id: str):
    """Retrieves the reproducible Markdown research report for a completed mission."""
    report_md = research_memory.get_report(mission_id)
    if not report_md:
        mission = research_memory.get_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")
        report_md = research_orchestrator.generate_final_report(mission)
    return {
        "mission_id": mission_id,
        "markdown_report": report_md
    }


@router.get("/providers")
def get_reasoning_provider_status():
    """Reports status and active capabilities of the reasoning provider."""
    provider = get_reasoning_provider()
    return {
        "active_provider": provider.name,
        "provider_type": provider.provider_type,
        "is_available": provider.is_available(),
        "allows_arbitrary_code_execution": False,
        "enforces_schema_validation": True
    }


class RunBenchmarkRequest(BaseModel):
    scenario_id: str = Field(default="scen_slalom_beta", description="Scenario ID to benchmark")
    evaluation_budget: int = Field(default=9, ge=3, le=30, description="Total simulation evaluations allowed per strategy")
    seeds: Optional[List[int]] = Field(default=None)


@router.post("/benchmark/compare", response_model=ComparativeBenchmarkReport)
def run_comparative_benchmark_endpoint(req: RunBenchmarkRequest):
    """
    Executes a controlled comparative benchmark comparing:
    1. Original Baseline
    2. Random Parameter Search
    3. Simple 1D Optimization
    4. RoboForge Scientific Research Agent
    Evaluates on identical simulation budgets and held-out test scenarios.
    """
    from backend.research.comparative_evaluator import comparative_evaluator
    try:
        report = comparative_evaluator.run_comparative_benchmark(
            scenario_id=req.scenario_id,
            evaluation_budget=req.evaluation_budget,
            seeds=req.seeds
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparative benchmark failed: {str(e)}")


class RunBlindDiscoveryBenchmarkRequest(BaseModel):
    candidate_budget: int = Field(default=4, ge=1, le=10, description="Candidate budget per strategy")
    tuning_seeds: Optional[List[int]] = Field(default=None)
    held_out_seeds: Optional[List[int]] = Field(default=None)


@router.post("/benchmark/blind-discovery")
def run_blind_discovery_benchmark_endpoint(req: RunBlindDiscoveryBenchmarkRequest):
    """
    Executes Milestone 3D Blind Scientific Discovery Benchmark.
    Compares Default APF, Random Search, Grid Search, and RoboForge Autonomous Research Agent.
    Strictly keeps held-out scenarios out of research memory and prompt context.
    """
    from backend.research.blind_discovery_evaluator import blind_discovery_evaluator
    try:
        report = blind_discovery_evaluator.run_benchmark(
            candidate_budget=req.candidate_budget,
            tuning_seeds=req.tuning_seeds,
            held_out_seeds=req.held_out_seeds,
            save_artifacts=True
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blind discovery benchmark failed: {str(e)}")


@router.get("/benchmark/blind-discovery/latest")
def get_latest_blind_discovery_report():
    """Retrieves the latest cached Blind Scientific Discovery benchmark report."""
    from pathlib import Path
    report_file = Path("benchmarks/discovery/benchmark_report.json")
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="No blind discovery benchmark report found. Run benchmark first.")
    import json
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/benchmark/blind-discovery/scenarios")
def get_blind_discovery_scenarios_endpoint():
    """Retrieves all defined benchmark scenarios for blind discovery evaluation."""
    from backend.research.blind_benchmark_scenarios import get_all_blind_benchmark_scenarios
    scens = get_all_blind_benchmark_scenarios()
    return {sid: sc.model_dump() for sid, sc in scens.items()}

