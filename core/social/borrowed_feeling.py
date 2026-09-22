"""A feeling that rides on what she believes somebody else feels, weighted by how often she has been right.

"Hooked on a Feeling" is a man high on believing that she loves him. Nothing in
it is evidence of her feeling. It is his belief about her feeling, and the
belief does the work a drug would. What it teaches is that a great deal of what
people feel in a relationship is produced by what they believe the other person
feels, and that such a feeling is only ever as good as the belief. In Reis and
Shaver's intimacy process model (1988) closeness is built from feeling
understood and cared for as the person perceives it, which is the same
dependence stated as a finding.

She has beliefs about how the person she talks with feels, with confidences,
and nothing she felt depended on them. So two things are kept here.

How good her beliefs have been. When somebody says outright how they feel, the
belief she held just before is scored against what they said:

    calibration = 1 - mean |believed - said|

and it stays unmeasured until they have said enough times to disagree with it.

What she borrows from the belief. How pleased she believes they are, above
where people usually sit, less how frustrated she believes they are, each by
her confidence in it:

    borrowed = calibration * (confidence_s * (satisfaction - its baseline)
                              - confidence_f * (frustration - its baseline))

A belief she has no track record on lends nothing, however confident it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = [
    "MIN_REPORTS",
    "Belief",
    "BorrowedFeeling",
    "CalibrationLedger",
    "borrowed_feeling",
    "get_calibration_ledger",
    "reset_for_test",
]

#: Statements scored before calibration is an estimate. Three is the least that
#: can disagree, as in ambivalence and fear of happiness.
MIN_REPORTS: int = 3


def _unit(value: Any) -> float | None:
    try:
        number = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class Belief:
    """What she believes about one of their feelings, how sure she is, and where people usually sit."""

    value: float
    confidence: float
    baseline: float


@dataclass(frozen=True)
class BorrowedFeeling:
    feeling: float = 0.0
    calibration: float | None = None
    reports: int = 0
    measured: bool = False
    why: str = "they have not said how they feel often enough to know whether her beliefs hold"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feeling": round(self.feeling, 6),
            "calibration": None if self.calibration is None else round(self.calibration, 6),
            "reports": self.reports,
            "measured": self.measured,
            "why": self.why,
        }


class CalibrationLedger:
    """Her beliefs about how they feel, scored against what they then said."""

    def __init__(self) -> None:
        self._error_sum = 0.0
        self._reports = 0

    def said(self, *, believed: float, said: float) -> None:
        belief = _unit(believed)
        statement = _unit(said)
        if belief is None or statement is None:
            return
        self._error_sum += abs(belief - statement)
        self._reports += 1

    def calibration(self) -> float | None:
        if self._reports < MIN_REPORTS:
            return None
        return 1.0 - self._error_sum / self._reports

    @property
    def reports(self) -> int:
        return self._reports


def borrowed_feeling(
    ledger: CalibrationLedger,
    *,
    satisfaction: Belief | None,
    frustration: Belief | None,
) -> BorrowedFeeling:
    """What she feels from what she believes they feel, in [-1, 1]."""
    calibration = ledger.calibration()
    if calibration is None:
        return BorrowedFeeling(reports=ledger.reports)

    def lean(belief: Belief | None) -> float:
        if belief is None:
            return 0.0
        value, confidence, baseline = (_unit(belief.value), _unit(belief.confidence), _unit(belief.baseline))
        if value is None or confidence is None or baseline is None:
            return 0.0
        return confidence * (value - baseline)

    pleased = lean(satisfaction)
    troubled = lean(frustration)
    feeling = max(-1.0, min(1.0, calibration * (pleased - troubled)))
    if feeling > 0.0:
        why = f"she believes they are pleased, and her beliefs about them have been {calibration:.2f} right"
    elif feeling < 0.0:
        why = f"she believes they are troubled, and her beliefs about them have been {calibration:.2f} right"
    else:
        why = "what she believes they feel sits where people usually do"
    return BorrowedFeeling(
        feeling=feeling,
        calibration=calibration,
        reports=ledger.reports,
        measured=True,
        why=why,
    )


#: Made at import rather than on first use. The subject-core fork carries module
#: globals that hold state, and one still None when an anchor is taken is not
#: carried, so a ledger first made inside one arm would reach the next arm with
#: that arm's reports in it.
_ledger: CalibrationLedger = CalibrationLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_ledger")


def get_calibration_ledger() -> CalibrationLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = CalibrationLedger()
