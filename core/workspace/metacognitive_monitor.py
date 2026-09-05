"""core/workspace/metacognitive_monitor.py
Metacognitive monitor identifying loop stalls and cognitive traps.
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Workspace.MetacognitiveMonitor")


class MetacognitiveMonitor:
    """Verifies that active goal ticks do not get stuck in infinite execution loops."""

    def audit_thought_traces(self, monologue_history: List[str]) -> bool:
        """Audits recent monologue logs. Returns True if anomalies are detected."""
        if len(monologue_history) < 5:
            return False
            
        # If the last 5 thoughts are identical, we have a stall
        recent = monologue_history[-5:]
        if len(set(recent)) == 1:
            logger.warning("Metacognitive warning: identical thought loop stall detected.")
            return True
            
        return False
