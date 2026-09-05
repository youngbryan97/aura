"""core/mission/campaign_planner.py — Long-Horizon Campaign Planner.

Splits a multi-day objective into concrete milestones with dependencies,
resource estimates, and evidence requirements.
"""
from __future__ import annotations

import logging

from core.mission.objective_graph import Milestone, ObjectiveGraph

logger = logging.getLogger("Aura.CampaignPlanner")


class CampaignPlanner:
    """Splits objectives into structured milestone sequences with dependency wiring."""

    @staticmethod
    def plan_campaign(
        objective: str,
        graph: ObjectiveGraph,
        plan_steps: list[str] | None = None,
    ) -> list[str]:
        """Generate milestones from plan steps or decompose the objective automatically."""
        logger.info("🎯 Planning campaign for: '%s'", objective[:60])

        if plan_steps and len(plan_steps) > 1:
            # Build milestones from explicit plan steps
            milestone_ids = []
            prev_id: str | None = None
            for i, step in enumerate(plan_steps):
                ms_id = f"ms_{i}_{step[:20].replace(' ', '_').lower()}"
                deps = [prev_id] if prev_id else []
                ms = Milestone(
                    milestone_id=ms_id,
                    description=step,
                    dependencies=deps,
                    priority=1.0 - (i * 0.01),
                    estimated_duration_s=60.0 + i * 10,
                )
                graph.add_milestone(ms)
                milestone_ids.append(ms_id)
                prev_id = ms_id
            return milestone_ids

        # Default decomposition: analyze → plan → execute → verify → report
        phases = [
            ("ms_analyze", "Analyze target and gather context", []),
            ("ms_plan", "Create detailed execution plan", ["ms_analyze"]),
            ("ms_execute", f"Execute core steps for: {objective[:40]}", ["ms_plan"]),
            ("ms_verify", "Run verification and tests", ["ms_execute"]),
            ("ms_report", "Compile results and update memory", ["ms_verify"]),
        ]

        milestone_ids = []
        for ms_id, desc, deps in phases:
            ms = Milestone(
                milestone_id=ms_id,
                description=desc,
                dependencies=deps,
                estimated_duration_s=120.0,
            )
            graph.add_milestone(ms)
            milestone_ids.append(ms_id)

        return milestone_ids
