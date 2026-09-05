"""core/welfare/welfare_memory.py
Stores historical logs of welfare state variables.
"""
import time
from typing import Any


class WelfareMemoryManager:
    """Maintains timeline of interoceptive variables to identify persistent distress."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []

    def record_snapshot(self, variables: dict[str, float]) -> None:
        self._history.append({
            "timestamp": time.time(),
            "variables": variables
        })

    def get_recent_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._history[-limit:]
