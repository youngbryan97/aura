"""Whether she gives less exactly when there is more to carry.

"How to Love" is the widest of the thirty-five recordings measured so far --
78.5 dB between its quietest and its loudest -- because the clinic scene is
left in at speaking volume before the first bar. The song does not raise its
voice when the material gets heavy. It keeps the same 152 BPM and the same
hushed delivery through a life told at its worst, and the constancy is the
point: the weight changes and the contact does not.

A reply system has the opposite reflex built into it by every gate that
shortens an answer. Heavy turns arrive with the person already strained, the
budget shrinks under caution, and the turns that needed her most get the least
of her. Nothing in the system reads that pattern, because each gate is looking
at one turn.

This reads it across turns. Each turn contributes the weight the other person
was under and the contact she actually gave back. Her own light turns are the
baseline -- not a target length, not a chosen floor -- and the question is
whether her heavy turns fall below them by more than she varies anyway.

    shortfall   how far her heavy turns fall below her light ones
    spread      how much her contact varies regardless of weight
    withdraws   shortfall > spread

`lift` is the closed side. When she withdraws, the ratio that restores her
heavy turns to her light ones multiplies the answer budget for the next heavy
turn, so the reading changes the cap, the cap changes the contact, and the
contact is what this measures next time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "PresenceUnderWeight",
    "WeightLedger",
    "get_weight_ledger",
    "reset_for_test",
]

#: A median needs a point either side of it to mean anything, and a spread
#: around that median needs one more. Below three on a side there is no
#: baseline to fall short of, only two numbers.
_ENOUGH_A_SIDE = 3


@dataclass
class PresenceUnderWeight:
    """How her contact moves against what the other person is carrying."""

    #: How far her heavy turns fall below her light ones, as a share of light.
    shortfall: float = 0.0
    #: How much her contact varies within a group, which is the variation
    #: weight does not explain, on the same scale.
    spread: float = 0.0
    #: True when the shortfall is larger than she varies anyway.
    withdraws: bool = False
    #: What her heavy turns would have to be multiplied by to sit level.
    lift: float = 1.0
    heavy: int = 0
    light: int = 0
    measured: bool = False
    why: str = "no turn has been weighed yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "shortfall": round(self.shortfall, 6),
            "spread": round(self.spread, 6),
            "withdraws": self.withdraws,
            "lift": round(self.lift, 6),
            "heavy": self.heavy,
            "light": self.light,
            "measured": self.measured,
            "why": self.why,
        }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


class WeightLedger:
    """Every turn as a pair: what they were carrying, what she gave back."""

    def __init__(self) -> None:
        self._turns: list[tuple[float, float]] = []

    def note(self, weight: float, contact: float) -> None:
        """One turn. `weight` is what the other person was under, `contact` is
        how much of her came back -- both on whatever scale the caller uses,
        because only the ratio between her own turns is read."""
        try:
            w = float(weight)
            c = float(contact)
        except (TypeError, ValueError):
            return
        if c < 0.0 or w != w or c != c:
            return
        self._turns.append((w, c))

    def read(self) -> PresenceUnderWeight:
        if not self._turns:
            return PresenceUnderWeight()
        weights = [w for w, _ in self._turns]
        middle = _median(weights)
        heavy = [c for w, c in self._turns if w > middle]
        light = [c for w, c in self._turns if w <= middle]
        if len(heavy) < _ENOUGH_A_SIDE or len(light) < _ENOUGH_A_SIDE:
            return PresenceUnderWeight(
                heavy=len(heavy),
                light=len(light),
                why=(
                    f"{len(heavy)} heavy turns and {len(light)} light ones, "
                    f"fewer than the {_ENOUGH_A_SIDE} a side that give a baseline"
                ),
            )
        base = _median(light)
        if base <= 0.0:
            return PresenceUnderWeight(
                heavy=len(heavy),
                light=len(light),
                why="her light turns carry no contact to measure against",
            )
        under = _median(heavy)
        shortfall = (base - under) / base
        # Within each group, so the spread measures what weight does not
        # explain. Taken across all turns it would include the very gap being
        # tested, and a large withdrawal would excuse itself.
        deviations = [abs(c - under) for c in heavy] + [abs(c - base) for c in light]
        spread = _median(deviations) / base
        withdraws = shortfall > spread
        lift = (base / under) if (withdraws and under > 0.0) else 1.0
        return PresenceUnderWeight(
            shortfall=shortfall,
            spread=spread,
            withdraws=withdraws,
            lift=lift,
            heavy=len(heavy),
            light=len(light),
            measured=True,
            why=(
                f"her heavy turns run {shortfall:+.0%} against her light ones "
                f"while turns of the same weight vary {spread:.0%}, so she "
                + ("gives less when there is more to carry" if withdraws
                   else "holds the same contact under weight")
            ),
        )

    def lift_for(self, weight: float) -> float:
        """The multiplier a turn this heavy has earned, and 1.0 for the rest.

        The closed side: this multiplies the answer budget, the budget sets the
        contact, and the contact is the next sample.
        """
        reading = self.read()
        if not reading.measured or not reading.withdraws:
            return 1.0
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return 1.0
        if w <= _median([x for x, _ in self._turns]):
            return 1.0
        return reading.lift


_LEDGER: WeightLedger | None = None


def get_weight_ledger() -> WeightLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = WeightLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
