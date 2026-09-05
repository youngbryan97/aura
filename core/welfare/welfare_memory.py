"""core/welfare/welfare_memory.py
Stores historical logs of welfare state variables.
"""
from typing import List, Dict, Any
import time


class WelfareMemoryManager:
    """Maintains timeline of interoceptive variables to identify persistent distress."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record_snapshot(self, variables: Dict[str, float]) -> None:
        self._history.append({
            "timestamp": time.time(),
            "variables": variables
        })

    def get_recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._history[-limit:]
