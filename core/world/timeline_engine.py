"""core/world/timeline_engine.py — Timeline and Event Sequencing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.TimelineEngine")


@dataclass
class WorldEvent:
    event_id: str
    title: str
    description: str
    timestamp: float = field(default_factory=time.time)
    related_entities: List[str] = field(default_factory=list)


class TimelineEngine:
    """Tracks historical sequences of external events and plans."""

    def __init__(self) -> None:
        self.events: Dict[str, WorldEvent] = {}

    def record_event(self, event: WorldEvent) -> None:
        self.events[event.event_id] = event
        logger.info("📅 Event recorded: %s (Time: %.1f)", event.title, event.timestamp)

    def get_timeline(self, entity_id: Optional[str] = None) -> List[WorldEvent]:
        """Returns sorted chronologically list of events, optionally filtered by entity."""
        all_events = list(self.events.values())
        if entity_id:
            all_events = [e for e in all_events if entity_id in e.related_entities]
        return sorted(all_events, key=lambda e: e.timestamp)


# Singleton
_timeline_engine_instance: TimelineEngine | None = None


def get_timeline_engine() -> TimelineEngine:
    global _timeline_engine_instance
    if _timeline_engine_instance is None:
        _timeline_engine_instance = TimelineEngine()
    return _timeline_engine_instance
