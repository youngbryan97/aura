"""Longing for elsewhere as a way of bearing the present, measured.

"California Dreamin'" is warm in the mind and cold in the body. These pin the
measured version: a low is her own spread below her own mean, whether a warmer
recollection helped is how she recovered with one against without, and only
then does a warmer recollection floor how she feels.

Her ordinary life here swings between 0.0 and 0.6, so her low line sits near
zero and every recovery lands inside her ordinary range. A recovery that was
itself a low would open a second low of its own, which the ledger rightly
counts.
"""

from __future__ import annotations

import pytest

from core.affect.elsewhere import MIN_SAMPLES, ElsewhereLedger

LOW = -0.4


def _settle(ledger: ElsewhereLedger) -> None:
    for value in [0.0, 0.6] * 12:
        ledger.turn(value, None)


def _lows(ledger: ElsewhereLedger, *, warm_to: float, cold_to: float, times: int = MIN_SAMPLES) -> None:
    for _ in range(times):
        ledger.turn(LOW, 0.6)
        ledger.turn(warm_to, None)
        ledger.turn(0.6, None)
        ledger.turn(LOW, None)
        ledger.turn(cold_to, None)
        ledger.turn(0.6, None)


def test_when_warmer_recollections_have_helped_a_low_is_floored_toward_them() -> None:
    ledger = ElsewhereLedger()
    _settle(ledger)
    _lows(ledger, warm_to=0.3, cold_to=0.0)
    reading = ledger.turn(LOW, 0.6)
    assert reading.measured and reading.low and reading.warmer
    assert reading.with_recollection == MIN_SAMPLES and reading.without_recollection == MIN_SAMPLES
    assert reading.helped == pytest.approx(0.7 - 0.4)
    assert reading.floor == pytest.approx(LOW + 0.3 * (0.6 - LOW))


def test_when_they_have_not_helped_nothing_is_lent() -> None:
    ledger = ElsewhereLedger()
    _settle(ledger)
    _lows(ledger, warm_to=0.0, cold_to=0.3)
    reading = ledger.turn(LOW, 0.6)
    assert reading.measured
    assert reading.helped == 0.0
    assert reading.floor is None


def test_an_ordinary_turn_is_not_a_low_whatever_comes_back() -> None:
    ledger = ElsewhereLedger()
    _settle(ledger)
    _lows(ledger, warm_to=0.3, cold_to=0.0)
    reading = ledger.turn(0.3, 0.9)
    assert not reading.low
    assert reading.floor is None


def test_a_colder_recollection_does_not_lift_a_low() -> None:
    ledger = ElsewhereLedger()
    _settle(ledger)
    _lows(ledger, warm_to=0.3, cold_to=0.0)
    reading = ledger.turn(LOW, -0.8)
    assert reading.low and not reading.warmer
    assert reading.floor is None


def test_before_both_kinds_of_low_have_happened_enough_nothing_is_claimed() -> None:
    ledger = ElsewhereLedger()
    _settle(ledger)
    _lows(ledger, warm_to=0.3, cold_to=0.0, times=MIN_SAMPLES - 1)
    reading = ledger.turn(LOW, 0.6)
    assert not reading.measured
    assert reading.floor is None


def test_an_unreadable_valence_is_not_a_turn() -> None:
    ledger = ElsewhereLedger()
    assert "could not be read" in ledger.turn(float("nan"), 0.5).why


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = ElsewhereLedger().turn(0.1, None).as_dict()
    for key in ("floor", "helped", "low", "warmer", "with_recollection", "without_recollection", "measured", "why"):
        assert key in row, key
