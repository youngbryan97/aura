"""core/council/god_council.py — Model Parliament Orchestrator.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.council.debate import ParliamentDebate
from core.council.minority_report import MinorityDisagreement, get_minority_report_store

logger = logging.getLogger("Aura.GodCouncil")


class GodCouncil:
    """Coordinates the specialized roles, debates, and voting consensus."""

    def __init__(self) -> None:
        self.report_store = get_minority_report_store()

    async def run_debate(
        self,
        objective: str,
        simulation_data: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs the parliament debate loop, resolves consensus, and logs dissents."""
        logger.info(" God Council convened for objective: '%s'", objective)
        
        debate = ParliamentDebate(objective)
        result = await debate.conduct(
            simulation_data=simulation_data,
            memory_context=memory_context,
        )

        # If approved, but there are dissenters, record them in the minority report store
        if result.get("approved"):
            dissenters = result.get("dissenters", [])
            for diss in dissenters:
                disagreement = MinorityDisagreement(
                    timestamp=time.time(),
                    mission_id=objective[:80],
                    dissenting_role=diss,
                    dissent_content=f"Dissenting vote recorded during debate for {objective[:40]}",
                    risk_level="medium" if diss != "safety_judge" else "critical",
                    consensus_decision="approved_by_majority",
                )
                self.report_store.record_dissent(disagreement)
        elif result.get("status") == "vetoed":
            # Record veto in reports
            dissenters = result.get("dissenters", [])
            for diss in dissenters:
                disagreement = MinorityDisagreement(
                    timestamp=time.time(),
                    mission_id=objective[:80],
                    dissenting_role=diss,
                    dissent_content=f"Safety Veto triggered: {result.get('reason')}",
                    risk_level="critical",
                    consensus_decision="vetoed_and_denied",
                )
                self.report_store.record_dissent(disagreement)

        return result


# Singleton
_council_instance: GodCouncil | None = None


def get_god_council() -> GodCouncil:
    global _council_instance
    if _council_instance is None:
        _council_instance = GodCouncil()
    return _council_instance
LC = get_god_council() # convenience ref
