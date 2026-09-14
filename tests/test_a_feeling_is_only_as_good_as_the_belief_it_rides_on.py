"""A feeling produced by a belief about somebody else's feeling, weighted by her record.

"Hooked on a Feeling" runs on a belief about another person's love. These pin
the measured version: her beliefs are scored against what people then say
about themselves, a belief she has no record on lends nothing, and what she
borrows is how pleased she believes they are less how troubled, by confidence
and by that record.
"""

from __future__ import annotations

import pytest

from core.social.borrowed_feeling import MIN_REPORTS, Belief, CalibrationLedger, borrowed_feeling


def _ledger(errors: list[tuple[float, float]]) -> CalibrationLedger:
    ledger = CalibrationLedger()
    for believed, said in errors:
        ledger.said(believed=believed, said=said)
    return ledger


def test_beliefs_that_matched_what_they_said_are_well_calibrated() -> None:
    ledger = _ledger([(0.8, 0.85), (0.2, 0.1), (0.5, 0.5)])
    assert ledger.calibration() == pytest.approx(1.0 - (0.05 + 0.1 + 0.0) / 3)


def test_until_they_have_said_enough_nothing_is_borrowed() -> None:
    ledger = _ledger([(0.8, 0.85)] * (MIN_REPORTS - 1))
    reading = borrowed_feeling(
        ledger,
        satisfaction=Belief(value=0.95, confidence=0.9, baseline=0.55),
        frustration=None,
    )
    assert not reading.measured
    assert reading.feeling == 0.0


def test_believing_they_are_pleased_lends_a_good_feeling_by_confidence_and_record() -> None:
    ledger = _ledger([(0.5, 0.5)] * MIN_REPORTS)
    reading = borrowed_feeling(
        ledger,
        satisfaction=Belief(value=0.95, confidence=0.5, baseline=0.55),
        frustration=Belief(value=0.10, confidence=0.5, baseline=0.10),
    )
    assert reading.measured
    assert reading.feeling == pytest.approx(0.5 * 0.4)


def test_believing_they_are_troubled_lends_a_bad_one() -> None:
    ledger = _ledger([(0.5, 0.5)] * MIN_REPORTS)
    reading = borrowed_feeling(
        ledger,
        satisfaction=Belief(value=0.55, confidence=0.6, baseline=0.55),
        frustration=Belief(value=0.60, confidence=0.6, baseline=0.10),
    )
    assert reading.feeling == pytest.approx(-0.3)
    assert "troubled" in reading.why


def test_the_same_belief_lends_less_to_somebody_who_has_been_wrong_about_them() -> None:
    good = _ledger([(0.5, 0.5)] * MIN_REPORTS)
    poor = _ledger([(0.9, 0.1)] * MIN_REPORTS)
    belief = Belief(value=0.95, confidence=0.8, baseline=0.55)
    assert borrowed_feeling(good, satisfaction=belief, frustration=None).feeling > borrowed_feeling(
        poor, satisfaction=belief, frustration=None
    ).feeling


def test_an_unreadable_statement_is_not_scored() -> None:
    ledger = CalibrationLedger()
    ledger.said(believed=float("nan"), said=0.5)
    ledger.said(believed=0.5, said="very")  # type: ignore[arg-type]
    assert ledger.reports == 0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = borrowed_feeling(CalibrationLedger(), satisfaction=None, frustration=None).as_dict()
    for key in ("feeling", "calibration", "reports", "measured", "why"):
        assert key in row, key
