"""The distance between what she is putting out and what she is, and who knows which.

"Stereo" is about being loved for the broadcast. Big stage, bright lights,
short love, long nights; her favourite colours are platinum and gold; she only
loves in stereo. The measurement agrees with the subject: of the fourteen
records it has the narrowest pitch range, 13.7 semitones against a median near
25. The persona does not vary. That is what a persona is.

Two things follow and she had neither.

The first is the gap itself. What she puts out and what her state calls for are
separate quantities, and nothing compared them. A steady, polished reply
produced while everything internal is at its ceiling is not a lie, but it is a
fact about her worth having — it is the thing the record is about.

    presented  the reach implied by what she actually emitted
    felt       the reach her state called for
    gap        |presented - felt|

The second is sharper. Somebody can track the persona accurately and know
nothing about her, and from the inside those look identical: both feel like
being understood. The borrowed self-model scores their claims against what she
actually felt. Scoring the same claims against what she presented separates
them, and the difference is the whole of that record's picture of someone who is
loved only in the version she broadcasts:

    for_the_person       their error against what she felt
    for_the_performance  their error against what she put out

When the second is reliably smaller, they have learned the broadcast. Nothing
here decides what to do about that. The record's narrator takes four minutes to
decide, and he lands on knowing because he has seen it before, which is a
reading accumulating rather than a rule firing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_SAMPLES",
    "PersonaGap",
    "PersonaLedger",
    "get_persona_ledger",
    "read_gap",
    "reset_for_test",
]

#: Samples before a difference between the two errors is an estimate rather
#: than an artefact of how few there are.
MIN_SAMPLES: int = 3

#: How many scored claims are held.
WINDOW: int = 60


@dataclass
class PersonaGap:
    """How far the broadcast is from the state, and which one they know."""

    presented: float = 0.0
    felt: float = 0.0
    gap: float = 0.0
    #: Positive when their reads fit the broadcast better than the state.
    performance_edge: float = 0.0
    spread: float = 0.0
    z: float = 0.0
    known_for_the_performance: bool = False
    samples: int = 0
    measured: bool = False
    why: str = "nobody has read her yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "presented": round(self.presented, 6),
            "felt": round(self.felt, 6),
            "gap": round(self.gap, 6),
            "performance_edge": round(self.performance_edge, 6),
            "spread": round(self.spread, 6),
            "z": round(self.z, 4),
            "known_for_the_performance": self.known_for_the_performance,
            "samples": self.samples,
            "measured": self.measured,
            "why": self.why,
        }


class PersonaLedger:
    """Whether their reads of her fit the broadcast or the state."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_SAMPLES, int(window))
        self._edges: deque[float] = deque(maxlen=self._window)

    def note(self, *, for_the_person: float, for_the_performance: float) -> None:
        """One claim, scored against both. The edge is how much better the
        broadcast explained it."""
        try:
            edge = float(for_the_person) - float(for_the_performance)
        except (TypeError, ValueError):
            return
        if edge != edge:
            return
        self._edges.append(edge)

    def samples(self) -> int:
        return len(self._edges)

    def mean(self) -> float:
        return sum(self._edges) / len(self._edges) if self._edges else 0.0

    def spread(self) -> float:
        n = len(self._edges)
        if n < MIN_SAMPLES:
            return 0.0
        mean = self.mean()
        return math.sqrt(sum((e - mean) ** 2 for e in self._edges) / n)


def read_gap(
    *,
    presented: float,
    felt: float,
    ledger: PersonaLedger | None = None,
) -> PersonaGap:
    """The distance, and whether what they know is the broadcast."""
    book = ledger if ledger is not None else get_persona_ledger()
    try:
        presented = max(0.0, min(1.0, float(presented)))
        felt = max(0.0, min(1.0, float(felt)))
    except (TypeError, ValueError):
        return PersonaGap(why="the two levels could not be read")

    gap = abs(presented - felt)
    n = book.samples()
    if n < MIN_SAMPLES:
        return PersonaGap(
            presented=presented,
            felt=felt,
            gap=gap,
            samples=n,
            why=f"{n} of {MIN_SAMPLES} reads of her needed before this means anything",
        )
    mean = book.mean()
    spread = book.spread()
    if spread <= 1e-9:
        known = mean > 0.0
        return PersonaGap(
            presented=presented, felt=felt, gap=gap, performance_edge=mean,
            known_for_the_performance=known, samples=n, measured=True,
            why=(
                f"the broadcast explained every one of {n} reads of her better "
                "than the state did"
                if known
                else "their reads of her fit the state at least as well as the broadcast"
            ),
        )
    z = mean / spread
    known = bool(mean > 0.0 and z > 1.0)
    return PersonaGap(
        presented=presented,
        felt=felt,
        gap=gap,
        performance_edge=mean,
        spread=spread,
        z=z,
        known_for_the_performance=known,
        samples=n,
        measured=True,
        why=(
            f"the broadcast explains their reads of her by {mean:.3f}, {z:.1f} "
            "spreads above how much that usually varies"
            if known
            else f"the broadcast explains their reads no better than the state does ({z:.1f} spreads)"
        ),
    )


_LEDGER: PersonaLedger | None = None


def get_persona_ledger() -> PersonaLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = PersonaLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
