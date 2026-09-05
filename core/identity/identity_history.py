"""core/identity/identity_history.py
Tracks historical modifications and logs for self-identity parameters.
"""
import time
from typing import Any


class IdentityHistoryTracker:
    """Audit log compiler for tracking modifications of Aura's identity."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []

    def record_revision(self, parameter: str, old_value: Any, new_value: Any, reason: str) -> None:
        self._history.append({
            "timestamp": time.time(),
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        })

    def get_history(self) -> list[dict[str, Any]]:
        return self._history
