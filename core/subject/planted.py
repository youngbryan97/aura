"""Units planted outside K* with known answers, for validating the interventional closure test (ISC-v7).

Registered as services in a validation run only, never in hers. Each keeps its
own state, so a swap from a donor carries over and is not rewritten next turn:

    planted_cause    a level that drifts on its own clock; every turn it moves
                     her curiosity by a little. A cause: the test must find it.
    planted_gate_a,  two flags that flip every thirteen turns of their own
    planted_gate_b   counts; only when both are on does her curiosity move. An
                     interaction: swapping one may show nothing, swapping both
                     must.
    planted_record   a copy of her curiosity, written every turn and read by
                     nothing. A record: no swap of it may move her.

`install(kinds)` registers the chosen units and wraps the affect phase's
closing step so they act once a turn.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["KINDS", "install"]

KINDS: tuple[str, ...] = ("cause", "gate", "record")

#: How far each planted cause moves her curiosity in a turn.
CAUSE_GAIN: float = 0.08
GATE_GAIN: float = 0.06
FLIP_EVERY: int = 13


class PlantedCause:
    def __init__(self) -> None:
        self.count = 0
        self.level = 0.0

    def step(self) -> None:
        self.count += 1
        self.level = 0.8 * self.level + 0.2 * math.sin(0.37 * self.count)


class PlantedGate:
    def __init__(self) -> None:
        self.count = 0
        self.on = 1

    def step(self) -> None:
        self.count += 1
        if self.count % FLIP_EVERY == 0:
            self.on = 1 - self.on


class PlantedRecord:
    def __init__(self) -> None:
        self.value = 0.0


def _get(name: str) -> Any:
    from core.container import ServiceContainer

    return ServiceContainer.get(name, default=None)


def act(state: Any) -> None:
    """One turn of the planted units: step their own clocks, act on her, and record her."""
    affect = getattr(state, "affect", None)
    emotions = getattr(affect, "emotions", None)
    if not isinstance(emotions, dict):
        return
    cause = _get("planted_cause")
    if cause is not None:
        cause.step()
        emotions["curiosity"] = min(1.0, max(0.0, float(emotions.get("curiosity", 0.0) or 0.0) + CAUSE_GAIN * cause.level))
    a, b = _get("planted_gate_a"), _get("planted_gate_b")
    if a is not None and b is not None:
        a.step()
        b.step()
        if a.on and b.on:
            emotions["curiosity"] = min(1.0, float(emotions.get("curiosity", 0.0) or 0.0) + GATE_GAIN)
    record = _get("planted_record")
    if record is not None:
        record.value = float(emotions.get("curiosity", 0.0) or 0.0)


def install(kinds: tuple[str, ...]) -> None:
    """Register the chosen planted units and have them act at the close of each affect phase."""
    from core.container import ServiceContainer
    import core.phases.memory_retrieval as recall

    if "cause" in kinds:
        ServiceContainer.register_instance("planted_cause", PlantedCause(), required=False)
    if "gate" in kinds:
        ServiceContainer.register_instance("planted_gate_a", PlantedGate(), required=False)
        ServiceContainer.register_instance("planted_gate_b", PlantedGate(), required=False)
    if "record" in kinds:
        ServiceContainer.register_instance("planted_record", PlantedRecord(), required=False)
    settle = recall.settle_on_what_she_feels
    if getattr(settle, "_planted", False):
        return

    def settle_then_act(state: Any) -> bool:
        changed = settle(state)
        act(state)
        return changed

    settle_then_act._planted = True  # type: ignore[attr-defined]
    recall.settle_on_what_she_feels = settle_then_act
