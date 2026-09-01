"""core/cognition/situation.py — one picture of now, and one broadcast about it.

Two things every organ needs and each currently builds for itself.

**The situation.** ``AuraNow`` holds the self-field. ``world_state`` holds the
world. The global workspace holds candidates. Memory holds what was retrieved.
Each decision organ assembles its own view from some subset of these, at its
own moment, and two organs deciding "the same" thing routinely decided it
about different worlds. A :class:`SituationSnapshot` is immutable and content-
hashed, so two organs that consumed the same situation can prove they did, and
two that disagreed can be shown to have been looking at different frames.

**The broadcast.** When something wins the workspace, several learners should
update — episodic, procedural, perceptual, attentional. In Aura they update
from their own separate triggers, so an important event teaches whichever
learner happened to be listening. A :class:`LearningBroadcast` carries one
event to all applicable learners under one evidence id, which is what makes
"this experience taught four systems" checkable: the same id appears in four
records, or it does not.

The coordinator refuses two things. A learner that has not declared what it
learns from cannot subscribe, because a subscriber matching everything is a
subscriber nobody can reason about. And a broadcast with no prediction error
and no valence is delivered but flagged — an event with nothing surprising and
nothing at stake is not obviously worth four learners' compute, and the
counter is what will eventually say whether broadcast-triggered learning earns
its cost.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.cognitive_event import current_cycle
from core.evidence.packet import EvidencePacket

__all__ = [
    "SituationSnapshot",
    "LearningBroadcast",
    "ConsolidationCoordinator",
    "get_coordinator",
    "reset_coordinator_for_test",
    "snapshot",
]


def _digest(payload: Any) -> str:
    try:
        material = json.dumps(payload, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        material = repr(payload)
    return hashlib.blake2s(material.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True, slots=True)
class SituationSnapshot:
    """What was true, all at once, when a decision was made."""

    cycle_id: int
    at: float = field(default_factory=time.time)
    percepts: Mapping[str, Any] = field(default_factory=dict)
    entities: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    memories: tuple[str, ...] = ()
    self_state: Mapping[str, Any] = field(default_factory=dict)
    affordances: tuple[str, ...] = ()
    #: Named uncertainties, so a consumer can see what the snapshot does NOT
    #: settle rather than inferring certainty from silence.
    uncertainty: Mapping[str, float] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return _digest(
            {
                "percepts": dict(self.percepts),
                "entities": list(self.entities),
                "goals": list(self.goals),
                "memories": list(self.memories),
                "self_state": dict(self.self_state),
                "affordances": list(self.affordances),
                "uncertainty": dict(self.uncertainty),
            }
        )

    def agrees_with(self, other: "SituationSnapshot") -> bool:
        """Whether two organs were looking at the same world."""
        return self.content_hash == other.content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "at": self.at,
            "content_hash": self.content_hash,
            "percepts": dict(self.percepts),
            "entities": list(self.entities),
            "goals": list(self.goals),
            "memories": list(self.memories),
            "self_state": dict(self.self_state),
            "affordances": list(self.affordances),
            "uncertainty": dict(self.uncertainty),
        }


@dataclass(frozen=True, slots=True)
class LearningBroadcast:
    """One event, offered to every learner it applies to, under one id."""

    evidence_id: str
    content: Any
    kind: str
    cycle_id: int = 0
    situation: SituationSnapshot | None = None
    valence: float = 0.0
    prediction_error: float = 0.0
    #: Who this happened to. "self" for Aura's own action, otherwise the agent.
    ownership: str = "self"
    evidence: EvidencePacket | None = None
    at: float = field(default_factory=time.time)

    @property
    def carries_a_signal(self) -> bool:
        """Whether anything about this event was surprising or mattered."""
        return abs(self.prediction_error) > 1e-9 or abs(self.valence) > 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "cycle_id": self.cycle_id,
            "valence": self.valence,
            "prediction_error": self.prediction_error,
            "ownership": self.ownership,
            "carries_a_signal": self.carries_a_signal,
            "situation": self.situation.content_hash if self.situation else "",
            "at": self.at,
        }


@dataclass
class _Subscriber:
    name: str
    kinds: frozenset[str]
    handler: Callable[[LearningBroadcast], Any]
    delivered: int = 0
    failed: int = 0


class ConsolidationCoordinator:
    """Which learners hear about an event, and the proof that they did."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[str, _Subscriber] = {}
        self._deliveries: dict[str, set[str]] = {}
        self._broadcasts = 0
        self._signalless = 0
        self._max_tracked = 4096

    def subscribe(
        self, name: str, kinds: Sequence[str], handler: Callable[[LearningBroadcast], Any]
    ) -> None:
        """Register a learner and what it learns from.

        ``kinds`` may not be empty. A learner subscribing to everything cannot
        be lesioned meaningfully and cannot be reasoned about when an event
        teaches the wrong thing.
        """
        if not kinds:
            raise ValueError(
                f"{name!r} subscribed to nothing in particular; declare the kinds it "
                "learns from, or it cannot be ablated or explained"
            )
        with self._lock:
            self._subscribers[name] = _Subscriber(name, frozenset(kinds), handler)

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subscribers.pop(name, None)

    def broadcast(self, broadcast: LearningBroadcast) -> dict[str, Any]:
        """Deliver to every applicable learner. Returns who got it.

        A handler that raises does not stop the others: one learner failing on
        an event must not silently cost the other three their update.
        """
        with self._lock:
            targets = [s for s in self._subscribers.values() if broadcast.kind in s.kinds]
            self._broadcasts += 1
            if not broadcast.carries_a_signal:
                self._signalless += 1
        reached: list[str] = []
        errors: dict[str, str] = {}
        for subscriber in targets:
            try:
                subscriber.handler(broadcast)
            except Exception as exc:  # noqa: BLE001 - one learner must not cost the rest
                subscriber.failed += 1
                errors[subscriber.name] = f"{type(exc).__name__}: {exc}"
                continue
            subscriber.delivered += 1
            reached.append(subscriber.name)
        with self._lock:
            if len(self._deliveries) >= self._max_tracked:
                self._deliveries.pop(next(iter(self._deliveries)))
            self._deliveries[broadcast.evidence_id] = set(reached)
        return {
            "evidence_id": broadcast.evidence_id,
            "reached": sorted(reached),
            "errors": errors,
            "carried_a_signal": broadcast.carries_a_signal,
        }

    def learners_reached(self, evidence_id: str) -> frozenset[str]:
        """Which learners this experience actually updated."""
        with self._lock:
            return frozenset(self._deliveries.get(evidence_id, ()))

    def report(self) -> dict[str, Any]:
        with self._lock:
            multi = sum(1 for names in self._deliveries.values() if len(names) >= 2)
            return {
                "subscribers": {
                    s.name: {"kinds": sorted(s.kinds), "delivered": s.delivered, "failed": s.failed}
                    for s in self._subscribers.values()
                },
                "broadcasts": self._broadcasts,
                "signalless_broadcasts": self._signalless,
                "tracked_events": len(self._deliveries),
                "events_that_taught_two_or_more": multi,
            }


_coordinator_lock = threading.Lock()
_coordinator: ConsolidationCoordinator | None = None


def get_coordinator() -> ConsolidationCoordinator:
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = ConsolidationCoordinator()
        return _coordinator


def reset_coordinator_for_test() -> ConsolidationCoordinator:
    global _coordinator
    with _coordinator_lock:
        _coordinator = ConsolidationCoordinator()
        return _coordinator


def snapshot(**parts: Any) -> SituationSnapshot:
    """Build a snapshot for the current cycle."""
    return SituationSnapshot(cycle_id=current_cycle(), **parts)
