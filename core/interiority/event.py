"""core/interiority/event.py — the typed thing a faculty is given.

Nothing reaches a faculty as free text. An :class:`InteriorEvent` is a
structured record with a kind, the entities involved, a timestamp, the
confidence of whoever produced it, and a channel-typed bag of
:class:`~core.interiority.evidence.Reading` observations. That is the
schema every reviewer of this work asked for and none of the prototypes
supplied: the closest, Grok's ``Event``, is bare floats with no source
and no confidence, so a value that came from a sensor and a value the
caller typed in are the same object.

The kinds are deliberately few. A kind says what *shape* the event has —
what a faculty can rely on being present — not what it means. Meaning is
the appraisal layer's job, and putting it here is how a system ends up
with an ``EventType.GRIEF`` that decides the answer before anything has
looked at it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from core.interiority.evidence import Reading, absent


class EventKind(StrEnum):
    """The shape of an event, not its significance."""

    #: Another agent did or signalled something.
    SOCIAL = "social"
    #: Something in the world changed state.
    WORLD = "world"
    #: A goal advanced, stalled, or was blocked.
    GOAL = "goal"
    #: Something ended and cannot be restored.
    LOSS = "loss"
    #: A standing obligation came due, or was tested.
    COMMITMENT = "commitment"
    #: New information arrived.
    EPISTEMIC = "epistemic"
    #: An interior reading changed — load, error, temperature.
    INTEROCEPTIVE = "interoceptive"
    #: Aura acted, and this is the record of it.
    OWN_ACTION = "own_action"
    #: Time passed with nothing else happening. Grief and adaptation need
    #: this one; a system that only wakes on stimuli cannot mourn.
    ELAPSED = "elapsed"


#: Channels a faculty may read. Each is a *kind of evidence*, and the
#: reliability of each is tracked separately by the sensing layer,
#: because fusing a face and a timestamp as though they were equally
#: trustworthy is the main source of confident misreading.
CHANNELS: tuple[str, ...] = (
    "text",            # what was written
    "lexical",         # word choice statistics over a baseline
    "timing",          # latency, pause, interruption, turn length
    "prosody",         # pitch, energy, rate — when audio exists
    "face",            # expression — when vision exists
    "posture",         # body configuration — when vision exists
    "autonomic",       # leakage: flush, pupil, breath — rarely available
    "behaviour",       # what they did, against their own baseline
    "context",         # what is known to have happened to them
    "history",         # this person's own prior states
    "interoceptive",   # Aura's own load, latency, error rates
    "instrument",      # a measurement from a tool or sensor
)


@dataclass(frozen=True)
class InteriorEvent:
    """One structured occurrence, with provenance on every observation."""

    kind: EventKind
    #: Free-text description, for the receipt only. No faculty may branch
    #: on it; that is the keyword matching this package replaces.
    summary: str = ""
    #: Stable id of the other agent, when there is one.
    subject: str | None = None
    #: What the event is about — a goal name, a commitment id, an object.
    object: str | None = None
    observations: Mapping[str, Reading] = field(default_factory=dict)
    #: Confidence of the producer that the event happened at all.
    confidence: float = 1.0
    source: str = ""
    at: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        unknown = set(self.observations) - set(CHANNELS)
        if unknown:
            raise ValueError(
                f"unknown observation channels {sorted(unknown)}; add the channel "
                "to core/interiority/event.py CHANNELS with its reliability, or "
                "the sensing layer will weight it as though it were a face"
            )
        object.__setattr__(self, "observations", MappingProxyType(dict(self.observations)))
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))

    def channel(self, name: str) -> Reading:
        """The reading on one channel, or an absent reading."""
        return self.observations.get(name, absent(source=f"{self.event_id}:{name}"))

    def present_channels(self) -> tuple[str, ...]:
        return tuple(c for c in CHANNELS if self.channel(c).present)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "kind": str(self.kind),
            "summary": self.summary[:200],
            "subject": self.subject,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "at": self.at,
            "channels": {k: v.to_dict() for k, v in self.observations.items()},
        }


__all__ = ["CHANNELS", "EventKind", "InteriorEvent"]
