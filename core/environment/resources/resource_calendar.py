"""Temporal resource management and trend tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.environment.homeostasis import Resource


@dataclass
class TimeSeriesPoint:
    """A single snapshot of a resource value at a given sequence/time."""
    sequence_id: int
    value: float
    timestamp: float


class ResourceCalendar:
    """Tracks resource trends over time and manages temporal constraints (cooldowns)."""

    def __init__(self):
        # Maps resource_id to a list of historical points
        self.history: dict[str, list[TimeSeriesPoint]] = {}
        # Maps action_name or event_name to the sequence_id it was last performed
        self.last_performed: dict[str, int] = {}
        # Known cooldown lengths (e.g. rate limit resets in turns/sequences)
        self.cooldowns: dict[str, int] = {}
        self.current_sequence: int = 0

    def update(self, resources: dict[str, Resource], sequence_id: int, timestamp: float) -> None:
        """Record the current resource values."""
        self.current_sequence = sequence_id
        for name, state in resources.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(TimeSeriesPoint(sequence_id, state.value, timestamp))
            
            # Keep history bounded
            if len(self.history[name]) > 1000:
                self.history[name] = self.history[name][-1000:]

    def record_action(self, action_name: str, cooldown_length: int = 0) -> None:
        """Record that an action was taken, and set an optional cooldown."""
        self.last_performed[action_name] = self.current_sequence
        if cooldown_length > 0:
            self.cooldowns[action_name] = cooldown_length

    def is_on_cooldown(self, action_name: str) -> bool:
        """Check if an action is currently restricted by a temporal cooldown."""
        last_time = self.last_performed.get(action_name)
        if last_time is None:
            return False
            
        cooldown = self.cooldowns.get(action_name, 0)
        return (self.current_sequence - last_time) < cooldown

    def predict_turns_until_critical(self, resource_name: str, critical_threshold: float) -> int | None:
        """Predicts when a resource will hit a critical threshold based on recent trends."""
        history = self.history.get(resource_name)
        if not history or len(history) < 2:
            return None
            
        # Simple linear projection based on the last 10 points
        recent = history[-10:]
        if len(recent) < 2:
            return None
            
        start_val = recent[0].value
        end_val = recent[-1].value
        d_seq = recent[-1].sequence_id - recent[0].sequence_id
        
        if d_seq == 0:
            return None
            
        rate = (end_val - start_val) / d_seq
        
        # If rate is positive and threshold is below, we will never hit it
        # If rate is negative and threshold is above, we will never hit it
        if rate == 0:
            return None
            
        if rate > 0 and end_val >= critical_threshold:
            return None
        if rate < 0 and end_val <= critical_threshold:
            return None
            
        turns = (critical_threshold - end_val) / rate
        if turns < 0:
            return None
            
        return int(turns)

__all__ = ["TimeSeriesPoint", "ResourceCalendar"]
