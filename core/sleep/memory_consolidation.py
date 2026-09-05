"""core/sleep/memory_consolidation.py
Offline memory consolidator compressing raw logs into narrative summaries.
"""
import logging
from typing import Any

from core.memory.day_summary import DaySummaryManager

logger = logging.getLogger("Sleep.MemoryConsolidation")


class MemoryConsolidator:
    """Prunes fine-grained transient events and generates high-level summaries."""

    def __init__(self):
        self.day_manager = DaySummaryManager()

    async def consolidate_logs(self, state: Any) -> None:
        logger.info("MemoryConsolidator compressing autobiographical logs...")
        recent_events = state.autobiographical_memory
        
        # Compile summary
        summary = self.day_manager.generate_day_summary(recent_events)
        
        # Prune raw memory logs, keeping only consolidated summary in working state
        state.autobiographical_memory = [summary]
        logger.info("Consolidation complete. Pruned %d raw episode traces.", len(recent_events))
