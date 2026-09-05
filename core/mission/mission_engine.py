"""core/mission/mission_engine.py — Strategic Campaign Mission Engine.

Manages long-horizon campaigns with objectives, subgoals, constraints,
deadlines, resources, blockers, active workers, evidence, progress,
abort conditions, and final review.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

from core.mission.objective_graph import ObjectiveGraph, Milestone
from core.mission.campaign_planner import CampaignPlanner
from core.mission.progress_monitor import MissionProgressMonitor
from core.runtime.action_executor import ActionExecutor

logger = logging.getLogger("Aura.MissionEngine")


class MissionStatus(StrEnum):
    PLANNING = "planning"
    ACTIVE = "active"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class MissionConstraints:
    """Operational constraints for a mission."""
    max_duration_seconds: float = 3600.0
    max_tool_calls: int = 100
    max_cost_dollars: float = 0.0
    abort_on_first_failure: bool = False
    required_confidence: float = 0.6
    allowed_action_classes: List[str] = field(default_factory=lambda: ["tool_execution", "file_write", "shell_command"])
    forbidden_action_classes: List[str] = field(default_factory=list)


@dataclass
class Campaign:
    """A durable multi-step campaign with tracking."""
    campaign_id: str
    objective: str
    status: MissionStatus = MissionStatus.PLANNING
    constraints: MissionConstraints = field(default_factory=MissionConstraints)
    started_at: float = 0.0
    completed_at: float = 0.0
    milestones_completed: int = 0
    milestones_total: int = 0
    tool_calls_used: int = 0
    blockers: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    progress_log: List[Dict[str, Any]] = field(default_factory=list)
    final_review: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MissionEngine:
    """Orchestrates long-horizon campaigns, tracking subtasks, blockers,
    resources, deadlines, and abort conditions."""

    def __init__(self) -> None:
        self.graph = ObjectiveGraph()
        self.monitor = MissionProgressMonitor()
        self.campaigns: Dict[str, Campaign] = {}
        self._campaign_counter = 0

    async def create_campaign(
        self,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Campaign:
        """Create and register a new campaign."""
        self._campaign_counter += 1
        cid = f"campaign_{self._campaign_counter}_{int(time.time())}"
        mc = MissionConstraints()
        if constraints:
            for k, v in constraints.items():
                if hasattr(mc, k):
                    setattr(mc, k, v)
        campaign = Campaign(campaign_id=cid, objective=objective, constraints=mc)
        self.campaigns[cid] = campaign
        logger.info("🎯 Created campaign '%s': %s", cid, objective[:60])
        return campaign

    async def run_mission(
        self,
        plan_steps: List[str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a mission through the campaign pipeline."""
        campaign = await self.create_campaign(" ".join(plan_steps), constraints)
        campaign.status = MissionStatus.ACTIVE
        campaign.started_at = time.time()

        # Build milestone graph
        milestones = CampaignPlanner.plan_campaign(campaign.objective, self.graph, plan_steps)
        campaign.milestones_total = len(milestones)

        for step_id in milestones:
            # Check deadline
            elapsed = time.time() - campaign.started_at
            if elapsed > campaign.constraints.max_duration_seconds:
                campaign.status = MissionStatus.ABORTED
                campaign.error = f"deadline_exceeded:{elapsed:.0f}s"
                logger.warning("⏰ Campaign aborted: deadline exceeded (%.0fs)", elapsed)
                break

            # Check tool call budget
            if campaign.tool_calls_used >= campaign.constraints.max_tool_calls:
                campaign.status = MissionStatus.ABORTED
                campaign.error = "tool_call_budget_exhausted"
                logger.warning("💰 Campaign aborted: tool call budget exhausted")
                break

            # Check blockers
            if self.graph.is_blocked(step_id):
                campaign.blockers.append(step_id)
                campaign.status = MissionStatus.BLOCKED
                logger.warning("🎯 Step %s is BLOCKED. Halting campaign.", step_id)
                break

            self.graph.set_status(step_id, "in_progress")
            self.monitor.record_progress(campaign.campaign_id, step_id, "executing")
            campaign.progress_log.append({
                "step": step_id,
                "event": "started",
                "time": time.time(),
                "elapsed_s": time.time() - campaign.started_at,
            })

            # Execute through ActionExecutor
            result = await ActionExecutor.execute(
                domain="tool_execution",
                action_name=f"mission.{step_id}",
                params={"step_id": step_id, "objective": campaign.objective},
                source="mission_engine",
            )
            campaign.tool_calls_used += 1

            if result.get("ok"):
                self.graph.set_status(step_id, "completed")
                campaign.milestones_completed += 1
                campaign.evidence.append({
                    "step": step_id,
                    "result": "success",
                    "time": time.time(),
                })
                campaign.progress_log.append({
                    "step": step_id,
                    "event": "completed",
                    "time": time.time(),
                })
            else:
                self.graph.set_status(step_id, "failed")
                campaign.evidence.append({
                    "step": step_id,
                    "result": "failure",
                    "error": result.get("error", "unknown"),
                    "time": time.time(),
                })
                campaign.progress_log.append({
                    "step": step_id,
                    "event": "failed",
                    "error": result.get("error"),
                    "time": time.time(),
                })

                if campaign.constraints.abort_on_first_failure:
                    campaign.status = MissionStatus.FAILED
                    campaign.error = f"step_failed:{step_id}"
                    break

        # Final review
        if campaign.status == MissionStatus.ACTIVE:
            campaign.status = MissionStatus.COMPLETED

        campaign.completed_at = time.time()
        campaign.final_review = {
            "objective": campaign.objective,
            "status": str(campaign.status),
            "milestones_completed": campaign.milestones_completed,
            "milestones_total": campaign.milestones_total,
            "tool_calls": campaign.tool_calls_used,
            "duration_s": campaign.completed_at - campaign.started_at,
            "blockers": campaign.blockers,
            "error": campaign.error,
        }

        logger.info("📊 Campaign %s finished: %s (%d/%d milestones, %.1fs)",
                     campaign.campaign_id, campaign.status,
                     campaign.milestones_completed, campaign.milestones_total,
                     campaign.completed_at - campaign.started_at)

        return {
            "ok": campaign.status == MissionStatus.COMPLETED,
            "status": str(campaign.status),
            "campaign_id": campaign.campaign_id,
            "review": campaign.final_review,
        }

    def get_active_campaigns(self) -> List[Campaign]:
        return [c for c in self.campaigns.values() if c.status in (MissionStatus.ACTIVE, MissionStatus.PLANNING)]

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        return self.campaigns.get(campaign_id)

    def abort_campaign(self, campaign_id: str, reason: str = "manual") -> bool:
        campaign = self.campaigns.get(campaign_id)
        if campaign and campaign.status == MissionStatus.ACTIVE:
            campaign.status = MissionStatus.ABORTED
            campaign.error = f"aborted:{reason}"
            campaign.completed_at = time.time()
            return True
        return False


# ── Singleton ───────────────────────────────────────────────────────────
_mission_engine_instance: MissionEngine | None = None


def get_mission_engine() -> MissionEngine:
    global _mission_engine_instance
    if _mission_engine_instance is None:
        _mission_engine_instance = MissionEngine()
    return _mission_engine_instance
