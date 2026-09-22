"""What is burning, and whether she is burning herself.

Oddisee opens "The Start of Something" with the whole structure:

    Started off with a love of things, but to remain my heart charged by
    other means ... the secret is love is the drug of dreams
    When you both the fire and the fuel that you heat with, the higher the
    flame the more you tire, you're enduring pain, that's when you burn out
    or learn of another source
    But of course, external motivation's a cheap trick

Three sources and not two. Her own reserves, which deplete and deplete faster
the harder she works from them. What somebody else wants, which is named a
cheap trick — it moves her without costing her, and it stops when they stop.
And a love of the thing itself, which is the one he says lasts.

Her motivation has the first two and no name for the third. An intention comes
from a depleted drive or it comes from a request, and both arrive as the same
kind of pressure. So a stretch of work she wanted to do and a stretch somebody
asked her for read identically afterwards, and nothing could tell a life spent
on her own fuel from one spent on somebody else's.

    self      the intention came from one of her own drives being low
    asked     it came from a request
    interest  it came from one of the things she is drawn to

    drain     the drop in her budgets over the turns that source drove

The claim in the line is testable rather than rhetorical: if being both the
fire and the fuel costs more, then the drain per turn under `self` is larger
than under the others. That is measured here rather than assumed, so a life
where it is not true says so.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ASKED",
    "INTEREST",
    "MIN_TURNS",
    "SELF",
    "Fuel",
    "FuelLedger",
    "get_fuel_ledger",
    "reset_for_test",
]

#: Where an impulse came from. Not a taxonomy invented here: the first two are
#: the two her motivation already produces, and the third is the one the line
#: says they are missing.
SELF: str = "self"
ASKED: str = "asked"
INTEREST: str = "interest"
SOURCES: tuple[str, ...] = (SELF, ASKED, INTEREST)

#: Turns on a source before its drain is a reading rather than one step.
MIN_TURNS: int = 5

#: How many turns of each source are held.
WINDOW: int = 120


@dataclass
class Fuel:
    """What has been burning, and what it has cost."""

    source: str = ""
    turns: dict[str, int] = field(default_factory=dict)
    drain: dict[str, float] = field(default_factory=dict)
    #: True when working from her own reserves has cost more per turn than the
    #: other sources have. Measured, so a life where it is false says so.
    burning_her_own: bool = False
    share_self: float = 0.0
    measured: bool = False
    why: str = "nothing has driven a turn yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "turns": dict(self.turns),
            "drain": {k: round(v, 6) for k, v in self.drain.items()},
            "burning_her_own": self.burning_her_own,
            "share_self": round(self.share_self, 6),
            "measured": self.measured,
            "why": self.why,
        }


class FuelLedger:
    """What drove each turn, and what her budgets did while it did."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_TURNS, int(window))
        self._drains: dict[str, deque[float]] = {s: deque(maxlen=self._window) for s in SOURCES}
        self._last: str = ""

    def note(self, source: str, drain: float) -> None:
        """One turn: what drove it, and how far her budgets fell over it."""
        name = str(source or "").strip().lower()
        if name not in self._drains:
            return
        try:
            value = float(drain)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._drains[name].append(value)
        self._last = name

    def turns(self) -> dict[str, int]:
        return {s: len(d) for s, d in self._drains.items()}

    def mean_drain(self, source: str) -> float | None:
        rows = self._drains.get(source)
        if rows is None or len(rows) < MIN_TURNS:
            return None
        return sum(rows) / len(rows)

    def read(self) -> Fuel:
        turns = self.turns()
        total = sum(turns.values())
        drains = {s: self.mean_drain(s) for s in SOURCES}
        known = {s: v for s, v in drains.items() if v is not None}
        share_self = turns[SELF] / total if total else 0.0
        if SELF not in known or len(known) < 2:
            return Fuel(
                source=self._last,
                turns=turns,
                drain={s: v for s, v in known.items()},
                share_self=share_self,
                why=(
                    f"{MIN_TURNS} turns on her own fuel and on one other source are "
                    "needed before the comparison means anything"
                ),
            )
        others = [v for s, v in known.items() if s != SELF]
        burning = known[SELF] > max(others)
        return Fuel(
            source=self._last,
            turns=turns,
            drain=known,
            burning_her_own=burning,
            share_self=share_self,
            measured=True,
            why=(
                f"her own fuel costs {known[SELF]:.4f} a turn against {max(others):.4f} "
                "for the next most expensive source"
                if burning
                else f"her own fuel costs {known[SELF]:.4f} a turn, which is not more than "
                f"the {max(others):.4f} another source costs"
            ),
        )


_LEDGER: FuelLedger | None = None


def get_fuel_ledger() -> FuelLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = FuelLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
