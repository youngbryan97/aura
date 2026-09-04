"""core/consciousness/narrative_provenance.py — a rendering is not a reading.

A phenomenal narrative is produced by giving a language model a state and
asking it to write. What comes back is a rendering of that state: real, useful,
and worth keeping, but carrying no information the state did not already have.
Read it back as though it were an observation and the loop closes — the text
becomes evidence for the state that generated it, and the confidence rises
with each pass while nothing new has been measured.

Three of those loops were running here:

* the witness reflection asked what keeps returning in the experiential
  stream, and was shown three previous narratives. What keeps returning in a
  sampled text is a property of the sampler;
* synthesis depth, a measure of how rich a moment is, rose when the narrative
  was longer than a hundred characters, so a verbose model produced deeper
  experience;
* the deep narrative prompt included the last narrative, so each one was
  written partly from the one before it.

None of them were wrong about anything. Each was a plausible way to use text
that was right there. What they lacked was a way to tell a rendering from a
reading, which is what this module supplies.

:class:`Rendering` binds text to a digest of the state it came from.
:func:`evidence_grade` then answers the only question that matters at a
consumer: is this text about a state I am now looking at, an earlier one, or
is it the state I am looking at, rendered? The third case is the loop, and it
returns :data:`Grade.SELF`, which no consumer may treat as evidence.

The digest is over numbers only. Text derived from a state must never enter
the digest, or two renderings of the same state would look like two states.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: How long a rendering describes a state anyone should still act on. Beyond
#: this the state it was made from is gone and the text is a record, not a
#: report of now.
FRESH_FOR_S = 120.0

#: Digest precision. Two states that differ below this are the same state for
#: the purpose of asking whether a narrative is about them; without rounding,
#: float noise in an unrelated channel would make every rendering look novel.
_PLACES = 4


class Grade(StrEnum):
    """What a narrative can be evidence for."""

    #: Rendered from the state being asked about. Not evidence for it — this
    #: is the loop, and a consumer that takes it is confirming its own input.
    SELF = "self"
    #: Rendered from an earlier state. Evidence about that state, and about
    #: the fact that it was rendered, which is a real event with a timestamp.
    PRIOR = "prior"
    #: Rendered from a state old enough that nothing should act on it now.
    STALE = "stale"
    #: No state was recorded with it, so nothing can be said about what it is
    #: evidence for. Treated as SELF, because an unattributed narrative is the
    #: case the loops above were all made of.
    UNATTRIBUTED = "unattributed"


def digest(state: Mapping[str, Any]) -> str:
    """A stable digest of the numeric part of a state.

    Strings are dropped rather than hashed. A narrative is a string derived
    from the state, so admitting strings would let a rendering change the
    digest of the very state it renders, and every rendering would then look
    like it came from a state of its own.
    """
    numbers: dict[str, float] = {}
    for key, value in sorted(state.items()):
        if isinstance(value, bool):
            numbers[str(key)] = float(value)
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        numbers[str(key)] = round(number, _PLACES)
    payload = json.dumps(numbers, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Rendering:
    """Text produced from a state, and the state it was produced from."""

    text: str
    state_digest: str
    generator: str
    rendered_at: float = field(default_factory=time.time)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.rendered_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "state_digest": self.state_digest,
            "generator": self.generator,
            "rendered_at": self.rendered_at,
            "age_s": round(self.age_s, 2),
        }


def evidence_grade(rendering: Rendering | None, current: str | None) -> Grade:
    """What this narrative may be used as evidence for, against a state."""
    if rendering is None or not rendering.state_digest:
        return Grade.UNATTRIBUTED
    if rendering.age_s > FRESH_FOR_S:
        return Grade.STALE
    if current and rendering.state_digest == current:
        return Grade.SELF
    if not current:
        return Grade.UNATTRIBUTED
    return Grade.PRIOR


def usable_as_evidence(rendering: Rendering | None, current: str | None) -> bool:
    """Whether a consumer may take this narrative as saying something new."""
    return evidence_grade(rendering, current) is Grade.PRIOR


class RenderingLog:
    """Recent renderings, with the states they came from.

    The log is what makes a series of narratives usable at all. Asked for
    what has been rendered, it can say which of them describe distinct states
    — which is the honest version of "what keeps returning", because a phrase
    repeating across renderings of one unchanged state is repetition in the
    writing and a phrase repeating across renderings of six different states
    is something about the states.
    """

    def __init__(self, maxlen: int = 32) -> None:
        self._entries: deque[Rendering] = deque(maxlen=maxlen)

    def record(self, text: str, state: Mapping[str, Any], generator: str) -> Rendering:
        entry = Rendering(
            text=str(text), state_digest=digest(state), generator=str(generator)[:64]
        )
        self._entries.append(entry)
        return entry

    def latest(self) -> Rendering | None:
        return self._entries[-1] if self._entries else None

    def recent(self, n: int = 5) -> tuple[Rendering, ...]:
        return tuple(self._entries)[-max(0, n):]

    def over_distinct_states(self, n: int = 5) -> tuple[Rendering, ...]:
        """The most recent renderings, one per distinct state.

        Several renderings of one unchanged state say one thing several
        times. Anything counting across them counts the generator.
        """
        seen: set[str] = set()
        picked: list[Rendering] = []
        for entry in reversed(self._entries):
            if entry.state_digest in seen:
                continue
            seen.add(entry.state_digest)
            picked.append(entry)
            if len(picked) >= max(0, n):
                break
        return tuple(reversed(picked))

    def distinct_states(self) -> int:
        return len({e.state_digest for e in self._entries})

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._entries),
            "distinct_states": self.distinct_states(),
            "latest": self.latest().to_dict() if self.latest() else None,
        }


def dominant_label(labels: Iterable[str], *, default: str = "") -> str:
    """The most common label, ties broken by name rather than by hash order.

    `max` over a dict built from a `set` picks by insertion order on a tie,
    and a set's order varies between processes. A tie then resolves one way
    on one boot and the other way on the next, which shows up as a dominant
    mood that changes when nothing changed.
    """
    counts: dict[str, int] = {}
    for label in labels:
        name = str(label)
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return default
    return min(counts, key=lambda name: (-counts[name], name))


__all__ = [
    "FRESH_FOR_S",
    "Grade",
    "Rendering",
    "RenderingLog",
    "digest",
    "dominant_label",
    "evidence_grade",
    "usable_as_evidence",
]
