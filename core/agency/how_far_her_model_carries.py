"""How far ahead her model of a world is still worth using, measured while she uses it.

She searches as deep as the clock allows. The clock is the wrong bound. A
search three levels into a model that is only right about the next move is
three levels of fiction, arrived at expensively and acted on with confidence,
and the deeper it goes the surer it looks.

The right bound is a measurement, and it is one she is already producing: every
move she makes carries a prediction of what the world will look like after it,
and every move of a plan she commits to carries a prediction at its own
distance. Graded against what happened, those say how far her model carries
here — one move, three, ten — and that is where the search should stop.

Two questions, both asked of her own record and answered per world:

    is she calibrated   when her rule says it is 90% sure, is it right nine
                        times in ten? The gap between what she claims and what
                        happens is what makes a confidence worth using
    how far does it     at each distance, is her prediction closer than
    carry               predicting no change at all? Where it stops beating
                        that, a level of search stops paying for itself

Nothing here knows what a world is. A prediction is a description, an outcome
is the description that turned up, and the distance is how many acts apart
they were.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.world_model.prediction_quality import (
    CalibrationCurve,
    MultiTimescalePrediction,
    PredictionOutcome,
)

logger = logging.getLogger("Aura.HowFarHerModelCarries")

__all__ = ["HowFarHerModelCarries"]

#: How many graded predictions at a distance before that distance can say
#: anything. Below this, one lucky move decides how deep she thinks.
ENOUGH_AT_A_DISTANCE = 8

#: The distances she asks about. One act ahead is what every move gives her;
#: the rest come from plans she commits to, which is what carrying a plan is
#: for. Beyond the deepest she can measure, she does not claim to know.
HOW_FAR_TO_ASK = (1, 2, 3, 4, 6, 8)


@dataclass
class HowFarHerModelCarries:
    """What her predictions here have been worth, by distance."""

    calibration: CalibrationCurve = field(default_factory=CalibrationCurve)
    #: Errors by distance, against what predicting no change would have cost.
    carrying: MultiTimescalePrediction = field(
        default_factory=lambda: MultiTimescalePrediction(horizons=HOW_FAR_TO_ASK)
    )
    #: How many graded predictions at each distance, so a distance with too
    #: few is not mistaken for one that failed.
    graded: dict[int, int] = field(default_factory=dict)

    def it_predicted(
        self,
        *,
        distance: int,
        confidence: float,
        was_right: bool,
        would_no_change_have_been_right: bool = False,
    ) -> None:
        """One prediction at a distance, and whether it turned out.

        ``would_no_change_have_been_right`` is the baseline this has to beat:
        a world where nothing she does changes anything is predicted perfectly
        by saying nothing changes, and a model that only matches that is not
        carrying her anywhere.
        """
        if type(distance) is not int or distance < 1:
            raise ValueError("distance must count delivered acts")
        outcome = PredictionOutcome(confidence, 0.0 if was_right else 1.0, tolerance=0.5)
        far = distance
        self.graded[far] = self.graded.get(far, 0) + 1
        if far == 1:
            self.calibration.observe(outcome)
        self.carrying.observe(
            far,
            error=0.0 if was_right else 1.0,
            baseline_error=0.0 if would_no_change_have_been_right else 1.0,
        )

    def carries_to(self) -> int:
        """How deep a search in her model is worth going, by what she measured.

        The last distance her model beats predicting no change, before the
        first distance measured NOT to. Zero, which bounds nothing, where no
        measured distance says it stops.

        It was the furthest distance measured to carry, and that read a
        distance nobody measured as one where the model fails. Distances past
        one are graded only when she commits to several moves at once, which
        on a fast board she rarely does, so they never gathered enough to
        count and every search was held to one move. LIVE 2026-09-23 on 2048:
        "she thought for 0.00s" on every move, and a game that fills the board
        at 128, where the same search given its clock reaches 1024 and 2048.
        """
        found = self.carrying.useful_horizon()
        useful = 0
        rows = sorted(found.get("by_horizon") or (), key=lambda row: int(row.get("horizon") or 0))
        for row in rows:
            if int(row.get("n") or 0) < ENOUGH_AT_A_DISTANCE:
                continue
            if not row.get("beats_baseline"):
                return useful
            useful = int(row.get("horizon") or 0)
        return 0

    def measured_to(self) -> int:
        """The furthest distance graded enough, and found to carry."""
        found = self.carrying.useful_horizon()
        return max(
            (
                int(row.get("horizon") or 0)
                for row in found.get("by_horizon") or ()
                if int(row.get("n") or 0) >= ENOUGH_AT_A_DISTANCE and row.get("beats_baseline")
            ),
            default=0,
        )

    def how_sure_she_should_be(self, claimed: float) -> float:
        """A confidence calibrated against predictions in the same bin."""
        return self.calibration.calibrated_probability(claimed, minimum_count=ENOUGH_AT_A_DISTANCE)

    def says(self) -> str:
        """What her model is worth here, for a log line and for a receipt."""
        carries = self.carries_to()
        over = self.calibration.overconfidence
        parts = []
        if carries:
            parts.append(f"her model carries {carries} act(s) here")
        elif self.measured_to():
            parts.append(
                f"her model carries at least {self.measured_to()} act(s) here, "
                "as far as she has measured"
            )
        elif self.graded:
            parts.append("how far her model carries is not settled yet")
        if over is not None and self.calibration.n >= ENOUGH_AT_A_DISTANCE:
            how = "over" if over > 0 else "under"
            parts.append(f"she is {how}confident by {abs(over):.2f} across {self.calibration.n}")
        return "; ".join(parts)

    def as_memory(self) -> dict[str, Any]:
        return {
            "measurement_version": 2,
            "graded": {str(far): count for far, count in self.graded.items()},
            "calibration": self.calibration.as_memory(),
            "carrying": {
                str(far): [
                    list(self.carrying._errors.get(far) or ())[-64:],  # noqa: SLF001 - her own record
                    list(self.carrying._baseline.get(far) or ())[-64:],  # noqa: SLF001
                ]
                for far in HOW_FAR_TO_ASK
            },
        }

    @classmethod
    def from_memory(cls, held: Any) -> "HowFarHerModelCarries":
        """What she measured here before, discounted by nothing: it is her own record."""
        carried = cls()
        if not isinstance(held, dict):
            return carried
        if held.get("measurement_version") != 2:
            logger.info("legacy planning measurements lack delivered-action provenance; starting a new measurement")
            return carried
        try:
            carried.calibration = CalibrationCurve.from_memory(held.get("calibration") or {})
            for far, rows in (held.get("carrying") or {}).items():
                errors, baseline = (list(rows[0]), list(rows[1])) if len(rows) == 2 else ([], [])
                for error, base in zip(errors, baseline, strict=False):
                    carried.carrying.observe(int(far), error=float(error), baseline_error=float(base))
            carried.graded = {
                int(far): int(count) for far, count in (held.get("graded") or {}).items()
            }
        except (AttributeError, IndexError, TypeError, ValueError) as why:
            logger.info("what her model was worth here would not read back: %s", why)
            return cls()
        return carried
