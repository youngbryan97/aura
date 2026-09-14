"""A recall that stands out from her ordinary recalls, and a question asked again.

"Remember the Time" works on the listener's memory rather than the singer's.
What returns is the feeling. These pin the two readings that make that
possible: which recalls stand out from the ones she usually gets, and how much
further a question goes each time it is asked again.
"""

from __future__ import annotations

import pytest

from core.memory.reliving import (
    MIN_HISTORY,
    MatchLedger,
    Reliving,
    ReturnLedger,
    deeper,
)


def _ordinary(led: MatchLedger, matches=(0.30, 0.34, 0.28, 0.32, 0.31, 0.29)) -> None:
    for value in matches:
        led.note(value)


def test_a_match_far_above_her_ordinary_recalls_is_relived() -> None:
    led = MatchLedger()
    _ordinary(led)
    reading = led.reading(0.95, valence=-0.7, note=False)
    assert reading.relived
    assert reading.z > 1.0
    assert reading.intensity == pytest.approx(0.7)


def test_an_ordinary_recall_brings_nothing_back() -> None:
    led = MatchLedger()
    _ordinary(led)
    reading = led.reading(0.31, valence=-0.7, note=False)
    assert not reading.relived
    assert reading.intensity == 0.0
    assert "ordinary recall" in reading.why


def test_the_bar_is_her_own_spread_of_matches() -> None:
    steady, varied = MatchLedger(), MatchLedger()
    _ordinary(steady, (0.50, 0.51, 0.49, 0.50, 0.51, 0.49))
    _ordinary(varied, (0.1, 0.9, 0.2, 0.8, 0.3, 0.7))
    assert steady.reading(0.62, 0.5, note=False).relived
    assert not varied.reading(0.62, 0.5, note=False).relived


def test_with_too_few_recalls_it_says_so_rather_than_claiming_one_stands_out() -> None:
    led = MatchLedger()
    led.note(0.3)
    reading = led.reading(0.99, 0.9, note=False)
    assert not reading.measured
    assert not reading.relived
    assert f"of {MIN_HISTORY} recalls" in reading.why


def test_what_comes_back_is_the_feeling_that_was_stored() -> None:
    led = MatchLedger()
    _ordinary(led)
    warm = led.reading(0.95, valence=0.4, note=False)
    heavy = led.reading(0.95, valence=-0.9, note=False)
    assert heavy.intensity > warm.intensity


def test_asking_again_looks_further_on_a_doubling_ladder() -> None:
    assert deeper(0) == 0
    assert deeper(1) == 1
    assert deeper(3) == 2
    assert deeper(7) == 3
    assert deeper(50) == 5
    assert deeper("not a count") == 0


def test_the_return_ledger_counts_each_question_separately() -> None:
    led = ReturnLedger()
    assert led.returns("what did we decide") == 0
    led.note("what did we decide")
    assert led.returns("what did we decide") == 1
    assert led.returns("something else") == 0
    led.note("")
    assert led.returns("") == 0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Reliving().as_dict()
    for key in ("relived", "intensity", "match", "z", "returns", "measured", "why"):
        assert key in row, key
