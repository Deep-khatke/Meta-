"""
Persistent research memory for RoboForge Scientific Intelligence.
Stores structured records for missions, hypotheses, plans, outcomes, and reports in filesystem JSON format.
Survives backend restarts and supports indexed search.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.research.models import (
    ResearchMission, Hypothesis, ControlledExperimentPlan,
    ExperimentOutcome, ValidationReport
)

logger = logging.getLogger("roboforge.research.memory")

RESEARCH_DIR = Path("experiments/research")
MISSIONS_DIR = RESEARCH_DIR / "missions"
HYPOTHESES_DIR = RESEARCH_DIR / "hypotheses"
PLANS_DIR = RESEARCH_DIR / "plans"
REPORTS_DIR = RESEARCH_DIR / "reports"


class ResearchMemory:
    """Manages filesystem-backed persistent memory of scientific research activities."""

    def __init__(self, base_dir: Path = RESEARCH_DIR):
        self.base_dir = base_dir
        self.missions_dir = base_dir / "missions"
        self.hypotheses_dir = base_dir / "hypotheses"
        self.plans_dir = base_dir / "plans"
        self.reports_dir = base_dir / "reports"

        for d in [self.missions_dir, self.hypotheses_dir, self.plans_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # --- Missions ---

    def save_mission(self, mission: ResearchMission) -> None:
        """Persists mission state to disk."""
        path = self.missions_dir / f"{mission.mission_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(mission.model_dump_json(indent=2))
        logger.debug("Saved research mission '%s' to %s", mission.mission_id, path)

    def get_mission(self, mission_id: str) -> Optional[ResearchMission]:
        """Retrieves mission from disk."""
        path = self.missions_dir / f"{mission_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ResearchMission.model_validate(data)
        except Exception as e:
            logger.error("Failed to load mission '%s': %s", mission_id, e)
            return None

    def list_missions(self, limit: int = 50) -> List[ResearchMission]:
        """Lists all stored missions, sorted by creation time descending."""
        missions: List[ResearchMission] = []
        for path in self.missions_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                missions.append(ResearchMission.model_validate(data))
            except Exception as e:
                logger.warning("Skipping unreadable mission file '%s': %s", path, e)

        missions.sort(key=lambda m: m.created_at, reverse=True)
        return missions[:limit]

    def search_missions(self, query: str) -> List[ResearchMission]:
        """Performs keyword search across mission goals and titles."""
        q = query.lower()
        results = []
        for m in self.list_missions(limit=100):
            if q in m.title.lower() or q in m.goal_description.lower() or q in m.mission_id.lower():
                results.append(m)
        return results

    # --- Hypotheses ---

    def save_hypothesis(self, hypothesis: Hypothesis) -> None:
        path = self.hypotheses_dir / f"{hypothesis.hypothesis_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(hypothesis.model_dump_json(indent=2))

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        path = self.hypotheses_dir / f"{hypothesis_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Hypothesis.model_validate(json.load(f))
        except Exception as e:
            logger.error("Failed to load hypothesis '%s': %s", hypothesis_id, e)
            return None

    def list_hypotheses_for_mission(self, mission_id: str) -> List[Hypothesis]:
        results = []
        for path in self.hypotheses_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    hypo = Hypothesis.model_validate(json.load(f))
                if hypo.mission_id == mission_id:
                    results.append(hypo)
            except Exception:
                pass
        results.sort(key=lambda h: h.confidence, reverse=True)
        return results

    # --- Controlled Experiment Plans ---

    def save_plan(self, plan: ControlledExperimentPlan) -> None:
        path = self.plans_dir / f"{plan.plan_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

    def get_plan(self, plan_id: str) -> Optional[ControlledExperimentPlan]:
        path = self.plans_dir / f"{plan_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return ControlledExperimentPlan.model_validate(json.load(f))
        except Exception:
            return None

    # --- Reports ---

    def save_report(self, mission_id: str, markdown: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        md_path = self.reports_dir / f"{mission_id}_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        if metadata:
            json_path = self.reports_dir / f"{mission_id}_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    def get_report(self, mission_id: str) -> Optional[str]:
        path = self.reports_dir / f"{mission_id}_report.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None


# Global instance
research_memory = ResearchMemory()
