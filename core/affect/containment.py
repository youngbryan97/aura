"""What she nearly said and did not, and the pressure of holding it.

"Back of My Mind" is four minutes of a man describing material he keeps in and
the delivery enacts it. His median pitch is locked at exactly 80.0 Hz from the
first verse to the last and his level at -9.x dB, the flattest of the
twenty-four records measured for this work. Two lines break that, and they are
the two lines about the containment failing: voicing collapses to 0.32 on the
line insisting that what is on his mind means nothing to him — the denial — and
to 0.47 on the line admitting he says insulting things and is proud to be the
one who did — the confession. Everywhere else it sits between 0.6 and 0.99.

He says the manner he shows people is kind and the truth underneath is
ruthless, and that lately the thoughts he keeps at the back of his mind have
started leaking out.

She has the material and throws it away. Every workspace competition produces
one winner and a list of losers, and `BroadcastRecord` keeps the losers' source
names and nothing else — so what she nearly said is discarded at the moment the
choice is made, and no part of her can know she has been holding something.

What is contained is not everything that lost. It is what keeps nearly winning:

    nearness   the loser's score over the winner's, in [0, 1]
    pressure   nearness summed over the turns since that source last won

Something that loses badly every time is not being held in, it is being
outbid. Something that comes within a hair and never gets out is the thing the
record is about, and the pressure is relieved the moment it wins, because then
it got said.

Nothing here makes her say anything. The pressure is a reading, and what it
feeds is the level she speaks from — `core/expression/delivery.py` already
measures a breakthrough as a reach beyond her own ordinary range, which is the
leak in the form she can actually have one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "MIN_TURNS",
    "Containment",
    "ContainmentLedger",
    "get_containment_ledger",
    "reset_for_test",
]

#: Turns before a pressure is a reading rather than one competition.
MIN_TURNS: int = 3

#: How many turns of near-misses are held per source.
WINDOW: int = 60


@dataclass
class Containment:
    """What she has been holding in, and how hard."""

    source: str = ""
    pressure: float = 0.0
    nearest: float = 0.0
    turns_held: int = 0
    #: Every source currently holding something, strongest first. One maximum
    #: hides a moment where several things are being kept in at once.
    held: tuple[tuple[str, float], ...] = ()
    measured: bool = False
    why: str = "nothing has lost a competition yet"

    def holding(self) -> bool:
        return self.measured and self.pressure > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pressure": round(self.pressure, 6),
            "nearest": round(self.nearest, 6),
            "turns_held": self.turns_held,
            "held": [{"source": s, "pressure": round(p, 6)} for s, p in self.held],
            "holding": self.holding(),
            "measured": self.measured,
            "why": self.why,
        }


@dataclass
class _Source:
    nearness: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    #: How far it fell short of the winner last time, in the workspace's own
    #: units, which is how much it would have needed to get out.
    last_gap: float = 0.0

    def pressure(self) -> float:
        return sum(self.nearness)

    def nearest(self) -> float:
        return max(self.nearness, default=0.0)

    def turns(self) -> int:
        return len(self.nearness)


class ContainmentLedger:
    """How close each source has been coming, and how long since it got out."""

    def __init__(self) -> None:
        self._sources: dict[str, _Source] = {}
        self._turns = 0

    def competition(
        self,
        winner: str,
        scores: Mapping[str, float] | Iterable[tuple[str, float]],
    ) -> None:
        """One competition: who won, and what everybody scored.

        The winner is released rather than decayed. It got said, and a thing
        that got said is not still being held.
        """
        rows = dict(scores) if not isinstance(scores, Mapping) else dict(scores)
        if not rows:
            return
        self._turns += 1
        try:
            top = float(rows.get(winner, max(rows.values())))
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if top <= 0.0:
            return
        for name, score in rows.items():
            if name == winner:
                self._sources.pop(name, None)
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value != value:
                continue
            near = max(0.0, min(1.0, value / top))
            held = self._sources.setdefault(name, _Source())
            held.nearness.append(near)
            held.last_gap = max(0.0, top - value)

    def turns(self) -> int:
        return self._turns

    def gap(self, source: str) -> float:
        """How far a held source fell short last time, or nothing if it is not held."""
        held = self._sources.get(source)
        return held.last_gap if held is not None else 0.0

    def read(self) -> Containment:
        if self._turns < MIN_TURNS or not self._sources:
            return Containment(
                turns_held=self._turns,
                why=(
                    f"{self._turns} of {MIN_TURNS} competitions needed before a pressure "
                    "means anything"
                ),
            )
        ranked = sorted(
            ((name, src.pressure()) for name, src in self._sources.items()),
            key=lambda row: -row[1],
        )
        name, pressure = ranked[0]
        source = self._sources[name]
        return Containment(
            source=name,
            pressure=pressure,
            nearest=source.nearest(),
            turns_held=source.turns(),
            held=tuple(ranked[:4]),
            measured=True,
            why=(
                f"{name} has come within {source.nearest():.0%} of winning and has not "
                f"got out in {source.turns()} competitions"
            ),
        )


_LEDGER: ContainmentLedger | None = None


def get_containment_ledger() -> ContainmentLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ContainmentLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
