"""core/lab/research_memory.py — Research Memory Store.

Persists findings and outcomes of scientific research cycles.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.ResearchMemory")


class ResearchMemory:
    """Stores validated conclusions and logs of research cycles."""

    def __init__(self) -> None:
        self.findings: dict[str, dict[str, Any]] = {}

    def save_research_outcome(self, cycle_id: str, conclusion: dict[str, Any]) -> None:
        self.findings[cycle_id] = {
            "conclusion": conclusion,
            "timestamp": time.time(),
        }
        logger.info("💾 Saved research finding for cycle '%s'", cycle_id)

    def list_findings(self) -> list[dict[str, Any]]:
        return [{"cycle_id": k, **v} for k, v in self.findings.items()]
