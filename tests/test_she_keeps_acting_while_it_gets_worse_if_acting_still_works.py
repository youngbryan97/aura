"""Acting inside decline, measured.

"All Star" says to get on with it while things go wrong. These pin the
measured version: decline is her valence falling more often lately than over
her life, it presses only as far as her actions still work, and a life that is
not getting worse, or one where nothing works, presses nothing.
"""

from __future__ import annotations

import pytest

from core.affect.acting_in_decline import MIN_SAMPLES, DeclineLedger


def _live(ledger: DeclineLedger, steps: list[float]) -> None:
    for value in steps:
        ledger.note(value)


def _ordinary_then_worse() -> DeclineLedger:
    ledger = DeclineLedger()
    # Up and down evenly for a long while, then falling most steps.
    _live(ledger, [0.0, 0.1] * 20)
    _live(ledger, [0.1, 0.0, -0.1, -0.2, -0.15, -0.3, -0.4, -0.35, -0.5, -0.6])
    return ledger


def test_a_life_getting_worse_while_her_actions_work_presses() -> None:
    reading = _ordinary_then_worse().reading(control=0.8)
    assert reading.measured
    assert reading.lately > reading.lifelong
    assert reading.press == pytest.approx(reading.decline * 0.8)
    assert reading.press > 0.0


def test_when_nothing_she_tries_works_nothing_presses() -> None:
    reading = _ordinary_then_worse().reading(control=0.0)
    assert reading.decline > 0.0
    assert reading.press == 0.0
    assert "nothing she tries" in reading.why


def test_an_ordinary_life_is_not_decline() -> None:
    ledger = DeclineLedger()
    _live(ledger, [0.0, 0.1] * 30)
    assert ledger.reading(control=0.9).press == 0.0


def test_unreadable_control_presses_nothing_and_says_why() -> None:
    reading = _ordinary_then_worse().reading(control=None)
    assert reading.press == 0.0
    assert not reading.measured


def test_before_both_kinds_of_step_have_happened_enough_nothing_is_claimed() -> None:
    ledger = DeclineLedger()
    _live(ledger, [0.5 - 0.1 * i for i in range(MIN_SAMPLES + 3)])
    assert not ledger.reading(control=1.0).measured


def test_a_recovery_wears_the_decline_away() -> None:
    ledger = _ordinary_then_worse()
    worse = ledger.reading(control=1.0).decline
    _live(ledger, [-0.5, -0.4, -0.45, -0.3, -0.2, -0.25, -0.1, 0.0, 0.05, 0.1] * 3)
    assert ledger.reading(control=1.0).decline < worse


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = DeclineLedger().reading(control=0.5).as_dict()
    for key in ("press", "decline", "z", "lately", "lifelong", "control", "measured", "why"):
        assert key in row, key
