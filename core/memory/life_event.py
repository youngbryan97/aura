"""core/memory/life_event.py
Structured Schema for Autobiographical Life Events.
Translates raw transactions into narrative autobiographical traces.
"""
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LifeEvent:
    """Structured autobiographical memory representation for Aura."""
    event_id: str
    timestamp: float = field(default_factory=time.time)
    
    # 7-factor narrative attributes
    perceived: dict[str, Any] = field(default_factory=dict)
    believed: dict[str, Any] = field(default_factory=dict)
    wanted: dict[str, Any] = field(default_factory=dict)
    chose: dict[str, Any] = field(default_factory=dict)
    did: dict[str, Any] = field(default_factory=dict)
    what_happened: dict[str, Any] = field(default_factory=dict)
    what_changed: dict[str, Any] = field(default_factory=dict)
    
    # Introspective lessons
    what_she_learned: str = ""
    what_she_should_avoid: str = ""
    what_she_should_remember: str = ""
    what_remains_unresolved: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "perceived": self.perceived,
            "believed": self.believed,
            "wanted": self.wanted,
            "chose": self.chose,
            "did": self.did,
            "what_happened": self.what_happened,
            "what_changed": self.what_changed,
            "lessons": {
                "learned": self.what_she_learned,
                "avoid": self.what_she_should_avoid,
                "remember": self.what_she_should_remember,
                "unresolved": self.what_remains_unresolved
            }
        }
