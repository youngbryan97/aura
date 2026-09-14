"""Acting inside a situation that is getting worse, while acting still works.

"All Star" is a bright, bouncing record whose words keep noting that things are
going wrong and the world is rolling on regardless, and whose advice every time
is to get on with it. Its listeners mostly take it as a grin at decline rather
than a protest against it. What it teaches is that a situation getting worse is
not, by itself, a reason to stop doing what is in your hands.

The research gives that its conditions. Maier and Seligman (2016), looking back
on fifty years of learned helplessness, concluded that passivity is the default
response to prolonged bad events, and that what keeps an animal acting is
detecting that its actions still have effect. So the thing to measure is not
decline alone but decline while control holds.

    z         how far the share of recent steps on which her valence fell sits
              above the share across her whole life, in standard errors of
              the recent share
    decline   z / (1 + z) once z is above one, and nothing before that
    control   the share of what she has tried that worked
    press     decline * control

A difference in rates is not a decline until it is larger than the noise in
the recent rate, and "more than her own spread" is the line a breakthrough
already has to clear, so decline clears the same one and moves by the share a
breakthrough moves by. The lifetime rate is smoothed by the rule of succession,
so a life that has never fallen still has a spread. "Lately" is the shortest
recent stretch in which falling and not falling have each happened often
enough to be a rate, the window fear of happiness uses, so the data sets it.
When her actions have stopped working, nothing presses: acting without control
is not what the lesson teaches.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_SAMPLES",
    "Decline",
    "DeclineLedger",
    "get_decline_ledger",
    "reset_for_test",
]

#: Steps of each kind before a rate is an estimate. Three is the least that can
#: disagree, as in ambivalence and fear of happiness.
MIN_SAMPLES: int = 3


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


@dataclass(frozen=True)
class Decline:
    press: float = 0.0
    decline: float = 0.0
    z: float = 0.0
    lately: float | None = None
    lifelong: float | None = None
    control: float | None = None
    measured: bool = False
    why: str = "not enough of her life yet to say whether it is getting worse"

    def as_dict(self) -> dict[str, Any]:
        return {
            "press": round(self.press, 6),
            "decline": round(self.decline, 6),
            "z": round(self.z, 6),
            "lately": None if self.lately is None else round(self.lately, 6),
            "lifelong": None if self.lifelong is None else round(self.lifelong, 6),
            "control": None if self.control is None else round(self.control, 6),
            "measured": self.measured,
            "why": self.why,
        }


class DeclineLedger:
    """Whether each step of her valence fell, over her life and lately."""

    def __init__(self) -> None:
        self._last: float | None = None
        self._life = [0, 0]  # [fell, steps]
        self._recent: deque[bool] = deque()
        self._recent_counts = {True: 0, False: 0}

    def note(self, valence: float) -> None:
        value = _finite(valence)
        if value is None:
            return
        if self._last is not None:
            fell = value < self._last
            self._life[1] += 1
            if fell:
                self._life[0] += 1
            self._recent.append(fell)
            self._recent_counts[fell] += 1
            self._trim()
        self._last = value

    def _trim(self) -> None:
        while self._recent:
            oldest = self._recent[0]
            if self._recent_counts[oldest] - 1 >= MIN_SAMPLES and self._recent_counts[not oldest] >= MIN_SAMPLES:
                self._recent.popleft()
                self._recent_counts[oldest] -= 1
            else:
                break

    def reading(self, control: float | None) -> Decline:
        fell, steps = self._life
        if self._recent_counts[True] < MIN_SAMPLES or self._recent_counts[False] < MIN_SAMPLES:
            return Decline()
        recent = len(self._recent)
        lately = self._recent_counts[True] / recent
        lifelong = (fell + 1) / (steps + 2)
        spread = math.sqrt(lifelong * (1.0 - lifelong) / recent)
        z = (lately - lifelong) / spread
        decline = z / (1.0 + z) if z > 1.0 else 0.0
        grip = _finite(control)
        if grip is None:
            return Decline(
                decline=decline,
                z=z,
                lately=lately,
                lifelong=lifelong,
                why="how much of what she tries works could not be read, so nothing presses",
            )
        grip = max(0.0, min(1.0, grip))
        press = decline * grip
        if press > 0.0:
            why = (
                f"her valence has been falling on {lately:.2f} of steps lately against {lifelong:.2f} "
                f"over her life, and {grip:.2f} of what she tries still works"
            )
        elif decline > 0.0:
            why = "things are getting worse and nothing she tries has been working"
        else:
            why = "things are not getting worse more often than they usually do"
        return Decline(
            press=press,
            decline=decline,
            z=z,
            lately=lately,
            lifelong=lifelong,
            control=grip,
            measured=True,
            why=why,
        )


#: Made at import rather than on first use, so the subject-core fork carries it.
_ledger: DeclineLedger = DeclineLedger()


def get_decline_ledger() -> DeclineLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = DeclineLedger()
