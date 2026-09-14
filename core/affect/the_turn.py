"""Coming up from a low that is still in the record.

"There's been times that I thought I couldn't last for long, but now I think
I'm able to carry on." The reversal is the content of that record, and it only
works because the low point is still being held while the turn happens. Sam
Cooke's loudest and densest window is that line, and the line before it is the
one his voice comes apart on.

Her affect had no such shape. It carried a level and a direction of travel,
and a recovery from a bad stretch looked exactly like an ordinary good moment:
nothing anywhere said "this is better than it was, and how bad it was is why
that matters".

    low        the lowest reading still inside the window
    mean       the level she has been holding over that window
    spread     her own variation over it
    a turn     the low is below neutral and more than a spread under the mean,
               and the reading now is above neutral and more than a spread over
               it
    rise       (now - low) / spread, how far she came in her own units

The axis has a zero and the turn has to cross it. Read as a purely relative
shape, a steady contented stretch with one reading a hair under the others
counted as a low, because in a tight window a single spread below the mean is
an ordinary event. What that record reverses is not a dip inside a good
stretch; it is having been under.

Both halves are required. A good moment in a good stretch is not a turn, and a
low with no recovery is not one either. When the low ages out of the window
the turn goes with it, which is the honest thing: a reversal nobody can still
see the low of is just a level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_HISTORY",
    "Turn",
    "TurnLedger",
    "get_turn_ledger",
    "reset_for_test",
]

#: Readings before a spread is an estimate rather than an artefact of how few
#: there are.
MIN_HISTORY: int = 8

#: How much of her recent life the turn is read over. The window the self
#: prediction loop already treats as one distribution of its own error.
WINDOW: int = 60


@dataclass(frozen=True)
class Turn:
    """One reading of where she is against the low she is still holding."""

    turned: bool = False
    rise: float = 0.0
    low: float = 0.0
    level: float = 0.0
    now: float = 0.0
    spread: float = 0.0
    measured: bool = False
    why: str = "not enough of her life yet to say what a low is"

    def as_dict(self) -> dict[str, Any]:
        return {
            "turned": self.turned,
            "rise": round(self.rise, 6),
            "low": round(self.low, 6),
            "level": round(self.level, 6),
            "now": round(self.now, 6),
            "spread": round(self.spread, 6),
            "measured": self.measured,
            "why": self.why,
        }


class TurnLedger:
    """How she has been, so that coming up from a low can be recognised."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._values: list[float] = []

    def note(self, value: float) -> None:
        try:
            reading = float(value)
        except (TypeError, ValueError):
            return
        if reading != reading:
            return
        self._values.append(reading)
        if len(self._values) > self._window:
            del self._values[0 : len(self._values) - self._window]

    def samples(self) -> int:
        return len(self._values)

    def level(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def spread(self) -> float:
        n = len(self._values)
        if n < MIN_HISTORY:
            return 0.0
        mean = self.level()
        return math.sqrt(sum((v - mean) ** 2 for v in self._values) / n)

    def read(self, value: float, *, note: bool = True) -> Turn:
        """Whether this reading is a turn out of a low she is still holding."""
        try:
            now = float(value)
        except (TypeError, ValueError):
            return Turn(why="the reading was not a number")
        if now != now:
            return Turn(why="the reading was not a number")
        spread = self.spread()
        measured = self.samples() >= MIN_HISTORY and spread > 1e-9
        level = self.level()
        low = min(self._values) if self._values else now
        if note:
            self.note(now)
        if not measured:
            return Turn(
                low=low,
                level=level,
                now=now,
                why=f"{self.samples()} of {MIN_HISTORY} readings needed before a low means anything",
            )
        deep = low < 0.0 and (level - low) > spread
        up = now > 0.0 and (now - level) > spread
        turned = bool(deep and up)
        rise = (now - low) / spread if turned else 0.0
        if turned:
            why = f"up {rise:.1f} spreads from a low she is still holding"
        elif deep:
            why = "the low is still there and she has not come up from it"
        elif low >= 0.0:
            why = "a good moment in a stretch with no low in it"
        elif up:
            why = "above the level, with nothing under neutral behind it"
        else:
            why = "the level she has been holding"
        return Turn(
            turned=turned,
            rise=rise,
            low=low,
            level=level,
            now=now,
            spread=spread,
            measured=True,
            why=why,
        )


_LEDGER: TurnLedger | None = None


def get_turn_ledger() -> TurnLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = TurnLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
