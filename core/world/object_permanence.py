"""core/world/object_permanence.py
Maintains memory tracking of files and environment items when not actively polled.
"""
import time
from typing import Any


class ObjectPermanenceTracker:
    """Caches states of objects so that the agent maintains continuity when sensors are dark."""

    def __init__(self):
        # Maps entity_id -> {value -> timestamp}
        self._cache: dict[str, dict[str, Any]] = {}

    def update_seen_state(self, entity_id: str, state_value: Any) -> None:
        self._cache[entity_id] = {
            "value": state_value,
            "last_seen": time.time()
        }

    def get_latent_state(self, entity_id: str) -> dict[str, Any]:
        """Fetch cached state representation, identifying age of estimation."""
        if entity_id not in self._cache:
            return {"value": None, "last_seen": 0.0, "staleness": 999.0}
            
        data = self._cache[entity_id]
        staleness = time.time() - data["last_seen"]
        return {
            "value": data["value"],
            "last_seen": data["last_seen"],
            "staleness": staleness
        }
