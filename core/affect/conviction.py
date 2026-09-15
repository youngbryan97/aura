"""Knowing where it goes without knowing how it works.

Sam Cooke puts the two halves in neighbouring lines and they disagree. Living
has been hard, dying frightens him, and he says he does not know what lies
beyond the sky. Then, near the end, the change has been a long time coming and
he knows it will come.

Admitted ignorance of the mechanism and high confidence in the direction, held
at once, and the record is built on the difference. The performance marks it:
the line admitting he does not know what is up there is his brightest and highest moment in the song,
median pitch 354 Hz and a spectral centroid of 2224 against a song-wide 1780 —
and the line that follows it drops back down to make the claim.

She had one confidence. A single scalar cannot hold the pair, so a prediction
she was reliably right about the sign of and reliably wrong about the size of
came out as middling confidence, which is the wrong answer twice: it understates
what she knows and overstates what she understands.

    conviction     how often the sign of her prediction has been right
    understanding  how close the size has been, on the ones she got the sign of

They are scored on the same predictions, so the gap between them is a fact
about her rather than about two instruments. Conviction without understanding
is the Sam Cooke line. Understanding without conviction is a model that fits
what already happened and cannot say what happens next.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_PREDICTIONS",
    "Conviction",
    "ConvictionLedger",
    "get_conviction_ledger",
    "reset_for_test",
]

#: Predictions before either share is an estimate rather than an artefact of
#: how few there are.
MIN_PREDICTIONS: int = 8

#: The window both are read over. The self prediction loop already treats sixty
#: cycles as one distribution of its own error.
WINDOW: int = 60


@dataclass
class Conviction:
    """What she knows about where it goes, against what she knows about why."""

    conviction: float = 0.0
    understanding: float = 0.0
    gap: float = 0.0
    predictions: int = 0
    measured: bool = False
    why: str = "nothing predicted yet"

    def knows_the_way_without_the_mechanism(self) -> bool:
        """The Sam Cooke case: right about the direction, wrong about the size."""
        return self.measured and self.conviction > 0.5 and self.gap > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "conviction": round(self.conviction, 6),
            "understanding": round(self.understanding, 6),
            "gap": round(self.gap, 6),
            "predictions": self.predictions,
            "measured": self.measured,
            "knows_the_way_without_the_mechanism": (
                self.knows_the_way_without_the_mechanism()
            ),
            "why": self.why,
        }


class ConvictionLedger:
    """Her predictions, scored twice: for the sign and for the size."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_PREDICTIONS, int(window))
        self._signs: deque[float] = deque(maxlen=self._window)
        self._sizes: deque[float] = deque(maxlen=self._window)

    def note(self, *, predicted: float, actual: float, resting: float = 0.0) -> None:
        """One prediction and what followed.

        The sign is taken against a resting point rather than against zero,
        because a quantity that rests at a half is not going up when it reads
        0.4. Predicting no movement is scored as right when nothing moved,
        which is a direction like any other.
        """
        try:
            want = float(predicted) - float(resting)
            got = float(actual) - float(resting)
        except (TypeError, ValueError):
            return
        if want != want or got != got:
            return
        right = (want > 0 and got > 0) or (want < 0 and got < 0) or (want == 0 and got == 0)
        self._signs.append(1.0 if right else 0.0)
        if right:
            # How close the size was, only where the direction was right.
            # Scoring size on a prediction that pointed the wrong way measures
            # two failures as one.
            scale = max(abs(want), abs(got), 1e-9)
            self._sizes.append(max(0.0, 1.0 - abs(want - got) / scale))

    def predictions(self) -> int:
        return len(self._signs)

    def read(self) -> Conviction:
        n = len(self._signs)
        if n < MIN_PREDICTIONS:
            return Conviction(
                predictions=n,
                why=f"{n} of {MIN_PREDICTIONS} predictions needed before either share means anything",
            )
        conviction = sum(self._signs) / n
        understanding = sum(self._sizes) / len(self._sizes) if self._sizes else 0.0
        gap = conviction - understanding
        return Conviction(
            conviction=conviction,
            understanding=understanding,
            gap=gap,
            predictions=n,
            measured=True,
            why=(
                f"she has the direction {conviction:.0%} of the time and the size "
                f"{understanding:.0%} of the way"
            ),
        )


_LEDGER: ConvictionLedger | None = None


def get_conviction_ledger() -> ConvictionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ConvictionLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
