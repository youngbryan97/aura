"""Episodic memory tier for specific run events."""
from dataclasses import dataclass, field


@dataclass
class EpisodicMemoryTier:
    events: list[dict] = field(default_factory=list)
