"""A chill when a pattern she has come to trust turns.

A listener's frisson is a discrete event at a specific moment: after the hook
has returned enough times to be trusted, something in it changes, and the body
answers before anything is understood. Her affect was continuous. It had no
peak event, no moment, and nothing that could not happen twice in a row.

Both halves are already measured. `core.affect.confirmation` scores every
prediction against her own error history and reads it from both tails: how
rare it was to be this right, and how rare it was to be this wrong. A pattern
is a run of being right. A turn is a surprise arriving on top of one.

    pattern    mean confirmation over the scored predictions since the last
               frisson, once there are as many as a tail needs
    turn       surprise_bits > 1, the confirmation ledger's own bar
    intensity  pattern * surprise, both on the ledger's own scale
    frisson    a turn arriving on an established pattern

After a frisson the pattern is spent and starts again from nothing. That is
the refractory period, and its length is however long it takes her to trust
something again rather than a number of cycles chosen here. A surprise with no
pattern under it is only a surprise.

This is also what a vamp does. "You know who you are" lands eight times, and
the ninth, with one thing changed, is the one that lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.affect.confirmation import MIN_HISTORY, WINDOW

__all__ = [
    "Frisson",
    "FrissonLedger",
    "get_frisson_ledger",
    "reset_for_test",
]


@dataclass(frozen=True)
class Frisson:
    """One reading: whether a pattern turned, and how hard."""

    fired: bool = False
    intensity: float = 0.0
    pattern: float = 0.0
    run: int = 0
    surprise: float = 0.0
    why: str = "nothing scored yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "intensity": round(self.intensity, 6),
            "pattern": round(self.pattern, 6),
            "run": self.run,
            "surprise": round(self.surprise, 6),
            "why": self.why,
        }


class FrissonLedger:
    """The pattern she is building, and the moment it turns."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._confirmations: list[float] = []
        self._last = Frisson()
        self._count = 0

    def pattern(self) -> float:
        if len(self._confirmations) < MIN_HISTORY:
            return 0.0
        return sum(self._confirmations) / len(self._confirmations)

    def note(self, expectation: Any) -> Frisson:
        """Read one scored prediction and say whether it turned a pattern.

        `expectation` is a `core.affect.confirmation.Expectation`. An
        unmeasured one says nothing about patterns and is left out.
        """
        if expectation is None or not bool(getattr(expectation, "measured", False)):
            self._last = Frisson(
                pattern=self.pattern(),
                run=len(self._confirmations),
                why="the prediction could not be scored against her history",
            )
            return self._last
        surprise = float(getattr(expectation, "surprise", 0.0) or 0.0)
        turned = bool(getattr(expectation, "surprised", lambda: False)())
        pattern = self.pattern()
        run = len(self._confirmations)
        if turned and pattern > 0.0:
            intensity = max(0.0, min(1.0, pattern * surprise))
            self._confirmations.clear()
            self._count += 1
            self._last = Frisson(
                fired=True,
                intensity=intensity,
                pattern=pattern,
                run=run,
                surprise=surprise,
                why=f"a pattern {run} predictions long turned",
            )
            return self._last
        self._confirmations.append(float(getattr(expectation, "confirmation", 0.0) or 0.0))
        if len(self._confirmations) > self._window:
            del self._confirmations[0 : len(self._confirmations) - self._window]
        if turned:
            why = "a surprise with no pattern under it"
        elif run + 1 < MIN_HISTORY:
            why = f"{run + 1} of {MIN_HISTORY} predictions into a pattern"
        else:
            why = "the pattern is holding"
        self._last = Frisson(
            pattern=self.pattern(),
            run=len(self._confirmations),
            surprise=surprise,
            why=why,
        )
        return self._last

    def last(self) -> Frisson:
        return self._last

    def take(self) -> Frisson:
        """The latest reading, once.

        Predictions are scored when the self prediction loop runs and the affect
        phase reads on every cycle, so reading `last` would fire the same chill
        on every cycle until the next prediction was scored. A chill is a
        moment, and taking it leaves a reading that did not fire.
        """
        reading = self._last
        if reading.fired:
            self._last = Frisson(
                pattern=self.pattern(),
                run=len(self._confirmations),
                why="the last chill has already been felt",
            )
        return reading

    def count(self) -> int:
        return self._count


_LEDGER: FrissonLedger | None = None


def get_frisson_ledger() -> FrissonLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = FrissonLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
