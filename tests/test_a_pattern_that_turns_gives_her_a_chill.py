"""Frisson: a discrete event when a trusted pattern turns, and not otherwise.

Built on the confirmation ledger's two tails, so every bar here is that
ledger's own: a pattern needs as many scored predictions as a tail does, and a
turn is a surprise of more than one bit.
"""

from __future__ import annotations

import pytest

from core.affect.confirmation import MIN_HISTORY, ExpectationLedger
from core.affect.frisson import Frisson, FrissonLedger


def _feed(expectations: ExpectationLedger, chills: FrissonLedger, errors) -> list[Frisson]:
    return [chills.note(expectations.score_and_note(e)) for e in errors]


def _history(expectations: ExpectationLedger) -> None:
    """Enough ordinary errors that a tail can be read."""
    expectations.load([0.30, 0.35, 0.28, 0.40, 0.33, 0.31, 0.36, 0.29, 0.34, 0.32])


def test_a_run_of_being_right_then_a_surprise_fires_once() -> None:
    expectations, chills = ExpectationLedger(), FrissonLedger()
    _history(expectations)
    readings = _feed(expectations, chills, [0.05] * MIN_HISTORY + [0.95])
    assert not any(r.fired for r in readings[:-1])
    last = readings[-1]
    assert last.fired
    assert 0.0 < last.intensity <= 1.0
    assert last.run == MIN_HISTORY
    assert chills.count() == 1


def test_a_surprise_with_no_pattern_under_it_is_only_a_surprise() -> None:
    expectations, chills = ExpectationLedger(), FrissonLedger()
    _history(expectations)
    reading = _feed(expectations, chills, [0.95])[-1]
    assert not reading.fired
    assert "no pattern" in reading.why


def test_after_a_chill_the_pattern_has_to_be_built_again() -> None:
    """The refractory period is the time it takes to trust something again."""
    expectations, chills = ExpectationLedger(), FrissonLedger()
    _history(expectations)
    _feed(expectations, chills, [0.05] * MIN_HISTORY + [0.95])
    again = _feed(expectations, chills, [0.99])[-1]
    assert not again.fired
    assert chills.count() == 1


def test_a_stronger_pattern_turns_harder() -> None:
    weak_e, weak = ExpectationLedger(), FrissonLedger()
    strong_e, strong = ExpectationLedger(), FrissonLedger()
    _history(weak_e)
    _history(strong_e)
    weak_reading = _feed(weak_e, weak, [0.29] * MIN_HISTORY + [0.95])[-1]
    strong_reading = _feed(strong_e, strong, [0.01] * MIN_HISTORY + [0.95])[-1]
    assert strong_reading.fired
    assert strong_reading.pattern > weak_reading.pattern
    assert strong_reading.intensity > weak_reading.intensity


def test_a_chill_is_felt_once_however_often_the_phase_looks() -> None:
    expectations, chills = ExpectationLedger(), FrissonLedger()
    _history(expectations)
    _feed(expectations, chills, [0.05] * MIN_HISTORY + [0.95])
    assert chills.take().fired
    assert not chills.take().fired
    assert "already been felt" in chills.take().why


def test_an_unscored_prediction_says_nothing_about_patterns() -> None:
    chills = FrissonLedger()
    reading = chills.note(ExpectationLedger().score(0.5))
    assert not reading.fired
    assert reading.run == 0
    assert chills.note(None).why.startswith("the prediction could not")


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = FrissonLedger().note(None).as_dict()
    for key in ("fired", "intensity", "pattern", "run", "surprise", "why"):
        assert key in row, key
    assert Frisson().intensity == pytest.approx(0.0)
