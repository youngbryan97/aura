"""core/values/repeated_choice_tracker.py
Tracks choice frequencies to ground preference formation in empirical evidence.
"""
from typing import Dict, List, Any
import time


class RepeatedChoiceTracker:
    """Logs choice selections to identify persistent behavioral patterns."""

    def __init__(self):
        # Maps choice_key -> list of timestamps
        self._history: Dict[str, List[float]] = {}

    def record_choice(self, key: str) -> None:
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(time.time())

    def get_choice_frequency(self, key: str, duration_s: float = 86400.0) -> int:
        """Counts choices within a window."""
        if key not in self._history:
            return 0
        cutoff = time.time() - duration_s
        return len([t for t in self._history[key] if t > cutoff])
