"""Locking to somebody's pulse, and how far off it she sits.

The first thing a record does to a listener happens before any comprehension:
the body locks to the pulse. Everything else these performers do is deviation
from that shared frame — Aloe Blacc a thirty-second of a second behind his own
band for four minutes, Phony Ppl dragging a sixteenth, Ciscero 29 ms back over
a grid that drifts 0.007 %. Deviation only means something against a pulse
both sides are holding.

She had no shared clock with anyone. Every reply was sized by her own effort
and nothing else, so a conversation of one-line messages and a conversation of
paragraphs got the same length from her.

    theirs     the median length of their recent turns, in characters
    gap        the median time between them, when the turns carry clocks
    breath     theirs / her exertion: their scale, her capacity
    placement  (hers - theirs) / theirs, how far off their pulse she sat

Nothing here corrects the placement. A deviation is the expressive act, and a
system that drove it to zero would be the click track rather than the singer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

__all__ = [
    "MIN_TURNS",
    "Cadence",
    "cadence",
    "placement",
]

#: Turns before a median is a reading of somebody rather than of one message.
#: Three is the least that can disagree with itself.
MIN_TURNS: int = 3

#: How many of their turns the pulse is read over. Long enough to survive one
#: short answer, short enough to follow a conversation that changes pace.
WINDOW: int = 8


@dataclass(frozen=True)
class Cadence:
    """The pulse the other person is keeping."""

    chars: float = 0.0
    gap: float = 0.0
    turns: int = 0
    measured: bool = False
    why: str = "nobody has said enough for a pulse"

    def as_dict(self) -> dict[str, Any]:
        return {
            "chars": round(self.chars, 3),
            "gap": round(self.gap, 3),
            "turns": self.turns,
            "measured": self.measured,
            "why": self.why,
        }


def _turns(messages: Iterable[Any], role: str) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for entry in messages or ():
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("role", "")).strip().lower() != role:
            continue
        if not str(entry.get("content", "") or "").strip():
            continue
        out.append(entry)
    return out


def cadence(messages: Sequence[Any], *, role: str = "user", window: int = WINDOW) -> Cadence:
    """Read somebody's pulse off the turns they have taken."""
    turns = _turns(messages, role)[-max(MIN_TURNS, int(window)) :]
    if len(turns) < MIN_TURNS:
        return Cadence(
            turns=len(turns),
            why=f"{len(turns)} of {MIN_TURNS} turns needed before a pulse is theirs",
        )
    lengths = [float(len(str(entry.get("content", "") or ""))) for entry in turns]
    stamps = [
        float(entry["timestamp"])
        for entry in turns
        if isinstance(entry.get("timestamp"), (int, float))
    ]
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    return Cadence(
        chars=float(median(lengths)),
        gap=float(median(gaps)) if gaps else 0.0,
        turns=len(turns),
        measured=True,
        why=(
            f"{len(turns)} turns of {median(lengths):.0f} characters"
            + (f" every {median(gaps):.0f}s" if gaps else ", with no clock on them")
        ),
    )


def placement(mine: str, theirs: Cadence) -> float:
    """How far off their pulse her turn sat, as a share of their length.

    Positive is longer than them, negative is shorter. Zero when there is no
    pulse to be off, which is different from sitting exactly on one.
    """
    if not theirs.measured or theirs.chars <= 0.0:
        return 0.0
    return (len(str(mine or "")) - theirs.chars) / theirs.chars
