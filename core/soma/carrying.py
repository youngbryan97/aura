"""What she is carrying, in the body rather than in the plan.

run_031 and the probe after it both read interoception with exactly one cause.
Perturb affect and the body moves; perturb recurrent cognition, deliberation,
the workspace, memory, the world model, her self-state or her developmental
state and it reads 0.0000 with a p of exactly 1.000 — bit-identical arms, seven
times over. `robust_recurrence_kappa` asks that no single domain's removal
disconnects the rest, and removing affect leaves nothing at all able to reach
the body.

The reason is in `core/phases/proprioceptive_loop.py`. Her pulse is a function
of affect and of nothing else: five branches on valence and arousal, each
setting a chosen number. So a mind with an urgent unfinished thing and a mind
with none have the same body, as long as they feel the same about it, and
nothing she is carrying is felt anywhere.

That is the wrong way round. An intention that presses and has not been
discharged is load, and load is felt — the standing cost of holding something
open is the oldest thing a body does about a plan.

    pressure    how hard the most pressing open intention is asking
    middle      how hard her open intentions usually ask
    shift       this moment against that middle, which averages to nothing

The shift multiplies the pulse her mood set rather than replacing it, so the
expression she wears is still hers and the rate under it answers to what she
is holding. Centred on her own middle, so a life of steady pressure reads as
no pressure at all and only a change is felt, which is what a baseline is for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Carrying",
    "CarryingLedger",
    "get_carrying_ledger",
    "pressure_of",
    "reset_for_test",
]

#: How much of her own history the middle is taken over.
_WINDOW = 256

#: Readings before a middle is a middle. Below three there is no point above
#: and below it, so the shift would be about the last reading.
_ENOUGH = 3


def pressure_of(*collections: Any) -> float:
    """How much is open and how hard, over goals and intentions together.

    This was the maximum, and a maximum cannot say that she is carrying five
    urgent things rather than one. Worse, it cannot move: one standing
    intention near the top pins it, and a probe that adds a sixth open thing
    at half urgency changes nothing at all. Displacing deliberation then read
    0.0000 into the body on every trial, bit-identical arms, so the path this
    module exists to open stayed shut.

    Carrying five is more than carrying one, so it is the sum. Unbounded on
    purpose: `CarryingLedger` centres it on the middle of her own, which is
    what makes a load a load, and a bound here would put the ceiling back.

    `decided_urgency` before `urgency`, for the reason `core/subject/state.py`
    prefers it: what a proposer declares is a constant chosen at its branch,
    and what the arbiter decides is that proposal shifted by how hard the state
    is pressing. The declared value is a proposal; the decided one is the load.
    """
    carried = 0.0
    for items in collections:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            stated = item.get("decided_urgency")
            if stated is None:
                stated = item.get("urgency")
            if stated is None:
                continue
            try:
                value = float(stated)
            except (TypeError, ValueError):
                continue
            if value == value:
                carried += max(0.0, value)
    return carried


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Carrying:
    """What she is holding open, against what she usually holds open."""

    #: How much is open and how hard, summed. Unbounded: what makes it a
    #: reading is the middle it is against.
    pressure: float = 0.0
    middle: float = 0.0
    #: This moment against that middle. Negative when she is carrying less.
    shift: float = 0.0
    seen: int = 0
    measured: bool = False
    why: str = "nothing open has been weighed yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pressure": round(self.pressure, 6),
            "middle": round(self.middle, 6),
            "shift": round(self.shift, 6),
            "seen": self.seen,
            "measured": self.measured,
            "why": self.why,
        }


class CarryingLedger:
    """How hard her open intentions have been pressing, over her own history."""

    def __init__(self) -> None:
        self._seen: deque[float] = deque(maxlen=_WINDOW)
        self._last: float = 0.0

    def note(self, pressure: float) -> None:
        """One moment's load: how hard the most pressing open thing is asking."""
        try:
            value = float(pressure)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        # Not bounded into [0, 1]. The load is a sum over what is open, and
        # capping it would restore the ceiling this ledger exists to remove.
        # What makes it a reading is the middle below, not a bound here.
        self._last = max(0.0, value)
        self._seen.append(self._last)

    def read(self) -> Carrying:
        seen = len(self._seen)
        if seen < _ENOUGH:
            return Carrying(
                pressure=self._last,
                seen=seen,
                why=(
                    f"{seen} of the {_ENOUGH} readings a middle needs a point "
                    "either side of"
                ),
            )
        middle = _median(list(self._seen))
        shift = self._last - middle
        return Carrying(
            pressure=self._last,
            middle=middle,
            shift=shift,
            seen=seen,
            measured=True,
            why=(
                f"she is holding {self._last:.2f} against the {middle:.2f} she "
                f"usually holds, so the body reads {shift:+.2f}"
            ),
        )

    def shift(self) -> float:
        """What the pulse her mood set should be multiplied past -- the closed
        side, and zero until there is a middle to be above or below."""
        reading = self.read()
        return reading.shift if reading.measured else 0.0


_LEDGER: CarryingLedger | None = None


def get_carrying_ledger() -> CarryingLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = CarryingLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
