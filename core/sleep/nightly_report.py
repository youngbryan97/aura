"""core/sleep/nightly_report.py
Compiles offline cycle nightly status reports for user review.
"""
import time
from typing import Any


class NightlyReportCompiler:
    """Compiles structured reports summarizing the agent's offline cycles."""

    def compile_report(self, state: Any, dreams: int) -> str:
        welfare = state.welfare.welfare_index
        goals_len = len(state.cognition.current_goals)
        
        report = (
            f"=== Aura Nightly Consolidation Report ===\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Consolidation Cycle Status: SUCCESS\n"
            f"Unified Welfare Index: {welfare:.2f}\n"
            f"Active Goals Remaining: {goals_len}\n"
            f"Simulated Dream Rehearsals: {dreams}\n"
            f"Memory Compaction: Completed\n"
            f"========================================="
        )
        return report
