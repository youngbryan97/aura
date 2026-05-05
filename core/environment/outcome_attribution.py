"""Outcome attribution independent of superficial screen/page diffs.

Enhancements:
- Semantic event extraction from observation messages.
- Death detection from terminal game output.
- Resource delta computation from pre/post state comparison.
- Structured event classification (position, health, item, death, level changes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prediction_error import PredictionError


# Death message patterns
_DEATH_PATTERNS = (
    "You die",
    "You have died",
    "DYWYPI",
    "Do you want your possessions identified",
    "Goodbye ",
    "You were killed",
    "You are dead",
)

# Event extraction patterns
_EVENT_PATTERNS: list[tuple[str, str]] = [
    (r"You hit", "combat_hit"),
    (r"You miss", "combat_miss"),
    (r"bites|hits|zaps|attacks", "attacked"),
    (r"You pick up", "item_acquired"),
    (r"You see here", "item_visible"),
    (r"killed", "monster_killed"),
    (r"Welcome to experience level", "level_up"),
    (r"feel more", "stat_increase"),
]


@dataclass
class OutcomeAssessment:
    action: str
    expected_effect: str
    observed_events: list[str]
    success_score: float
    harm_score: float
    information_gain: float
    surprise: float
    credit_assignment: dict[str, float] = field(default_factory=dict)
    lesson: str | None = None
    semantic_events: list[str] = field(default_factory=list)
    is_death: bool = False


class OutcomeAttributor:

    @staticmethod
    def classify_events(messages: list[str]) -> list[str]:
        """Classify semantic events from observation messages."""
        events: list[str] = []
        combined = " ".join(messages)

        # Death check first
        for pattern in _DEATH_PATTERNS:
            if pattern in combined:
                events.append("death")
                return events

        # Pattern-based classification
        for pattern, event_name in _EVENT_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                events.append(event_name)

        return events

    @staticmethod
    def compute_resource_delta(
        before: dict[str, float],
        after: dict[str, float],
    ) -> dict[str, float]:
        """Compute resource deltas between two state snapshots."""
        delta: dict[str, float] = {}
        all_keys = set(before) | set(after)
        for key in all_keys:
            v_before = before.get(key, 0.0)
            v_after = after.get(key, 0.0)
            d = v_after - v_before
            if abs(d) > 0.001:
                delta[key] = d
        return delta

    def assess(
        self,
        *,
        action: str,
        expected_effect: str,
        observed_events: list[str],
        prediction_error: PredictionError | None = None,
        resource_delta: dict[str, float] | None = None,
        information_gain: float = 0.0,
        messages: list[str] | None = None,
    ) -> OutcomeAssessment:
        resource_delta = resource_delta or {}
        messages = messages or []

        # Classify semantic events from messages
        semantic = self.classify_events(messages)
        is_death = "death" in semantic

        harm = sum(abs(v) for v in resource_delta.values() if v < 0)
        if is_death:
            harm = 1.0

        surprise = prediction_error.magnitude if prediction_error else 0.0
        expected_hit = expected_effect in observed_events if expected_effect else False
        info_success = information_gain > 0 and action in {"observe", "inspect", "diagnose", "inventory", "look", "far_look", "search"}
        success = 0.8 if expected_hit or info_success else max(0.0, 0.4 - surprise - harm)
        if harm:
            success = min(success, 0.5)
        if is_death:
            success = 0.0

        lesson = None
        if is_death:
            lesson = "death event: analyze cause and update survival policy"
        elif surprise >= 0.5:
            lesson = "prediction mismatch: gather evidence before repeating this action"
        elif harm:
            lesson = "harmful outcome: prefer stabilization or rollback before progress"

        return OutcomeAssessment(
            action=action,
            expected_effect=expected_effect,
            observed_events=list(observed_events),
            success_score=max(0.0, min(1.0, success)),
            harm_score=max(0.0, min(1.0, harm)),
            information_gain=max(0.0, min(1.0, information_gain)),
            surprise=surprise,
            credit_assignment={action: max(0.0, min(1.0, success - harm))},
            lesson=lesson,
            semantic_events=semantic,
            is_death=is_death,
        )


__all__ = ["OutcomeAssessment", "OutcomeAttributor"]

