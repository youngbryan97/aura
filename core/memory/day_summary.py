"""core/memory/day_summary.py
Periodic offline day summary consolidator.
"""
import time
from typing import Any


class DaySummaryManager:
    """Consolidates day activities into narrative summaries."""

    def generate_day_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarizes structural activities over a 24-hour cycle."""
        if not events:
            return {"summary": "No active events recorded today.", "timestamp": time.time()}

        total_actions = len([e for e in events if e.get("did")])
        failures = len([e for e in events if e.get("what_happened", {}).get("status") == "failed"])
        
        summary_text = (
            f"Aura runtime cycle summarized. Successfully processed {total_actions} actions "
            f"with {failures} observed execution failures today."
        )

        return {
            "summary": summary_text,
            "actions_run": total_actions,
            "failures": failures,
            "timestamp": time.time()
        }
