"""core/identity/identity_history.py
Tracks historical modifications and logs for self-identity parameters.
"""
from typing import List, Dict, Any
import time


class IdentityHistoryTracker:
    """Audit log compiler for tracking modifications of Aura's identity."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record_revision(self, parameter: str, old_value: Any, new_value: Any, reason: str) -> None:
        self._history.append({
            "timestamp": time.time(),
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history
