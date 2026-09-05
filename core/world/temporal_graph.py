"""core/world/temporal_graph.py
Temporal graph cataloging chronologies and duration estimates for events.
"""
from typing import Dict, Any, List
import time


class TemporalGraph:
    """Tracks chronological links and event sequencing."""

    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def record_event_time(self, label: str, duration: float) -> None:
        self._events.append({
            "label": label,
            "timestamp": time.time(),
            "duration": duration
        })

    def get_average_duration(self, label: str) -> float:
        matches = [e["duration"] for e in self._events if e["label"] == label]
        if not matches:
            return 1.0
        return sum(matches) / len(matches)
