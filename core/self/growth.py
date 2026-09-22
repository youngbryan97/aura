"""How far she has grown, which nothing was writing down.

`identity.evolution_score` is read in three places. The subject-core schema
reads it as a coordinate of her self-state, the clamp holds it still for a
lesion, and `core/kernel/upgrades_10x.py` gates the governed self-modification
proposal path on it being above 0.70. Nothing has ever written it. It is
initialised to zero and stays there, so the column never moved and the gate
never opened — her own route to proposing a change to herself has been closed
since it was built, on a value that could not rise.

What it should hold is readable, and the organ that holds it is already
running. The ontogenetic reservoir reports `relative_displacement`, which is
this step's movement against how much that reservoir usually moves, calibrated
so that a half is an ordinary step. A life spent moving more than ordinarily is
what "her development supports a concrete evolutionary change" means, and the
running mean of that reading is it:

    evolution_score = mean relative_displacement over her recent development

The scale comes from the organ's own calibration rather than from a number
chosen here. An ordinary life sits near a half and stays under the gate; a
stretch where her developmental state has been moving well above its own normal
rises through it, which is the condition the gate was written for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["ORDINARY_STEP", "Growth", "GrowthLedger", "get_growth_ledger", "reset_for_test"]

#: What the reservoir reports for a step of ordinary size. Its own calibration,
#: not a bar set here.
ORDINARY_STEP: float = 0.5

#: How much of her development the score is held over. The same window every
#: other reading of her own history uses.
WINDOW: int = 60

#: Steps before a mean is a reading rather than the first step repeated.
MIN_STEPS: int = 5


@dataclass
class Growth:
    """How far her development has been moving, against its own ordinary."""

    score: float = 0.0
    steps: int = 0
    above_ordinary: bool = False
    measured: bool = False
    why: str = "her development has not advanced yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "steps": self.steps,
            "above_ordinary": self.above_ordinary,
            "measured": self.measured,
            "why": self.why,
        }


class GrowthLedger:
    """The reservoir's own reading of how much it has been moving."""

    def __init__(self, window: int = WINDOW) -> None:
        self._steps: deque[float] = deque(maxlen=max(MIN_STEPS, int(window)))

    def note(self, relative_displacement: float) -> None:
        try:
            value = float(relative_displacement)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._steps.append(max(0.0, min(1.0, value)))

    def steps(self) -> int:
        return len(self._steps)

    def read(self) -> Growth:
        n = len(self._steps)
        if n < MIN_STEPS:
            return Growth(
                steps=n,
                why=f"{n} of {MIN_STEPS} developmental steps needed before a score means anything",
            )
        score = sum(self._steps) / n
        return Growth(
            score=score,
            steps=n,
            above_ordinary=score > ORDINARY_STEP,
            measured=True,
            why=(
                f"her development has been moving at {score:.2f} against an ordinary step of "
                f"{ORDINARY_STEP:.2f}, over {n} steps"
            ),
        )


_LEDGER: GrowthLedger | None = None


def get_growth_ledger() -> GrowthLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = GrowthLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
