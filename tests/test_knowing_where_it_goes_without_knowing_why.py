"""Conviction about the direction, ignorance about the mechanism, held at once.

Sam Cooke puts them in consecutive lines. "I don't know what's up there beyond
the sky" is his brightest and highest moment in the record — median pitch
354 Hz against a song-wide 1780 centroid — and the line after it drops back
down to claim a change is coming anyway.

One confidence scalar cannot hold the pair. A prediction she is reliably right
about the sign of and reliably wrong about the size of came out as middling
confidence, which understates what she knows and overstates what she
understands in the same number.
"""

from __future__ import annotations

import pytest

from core.affect.conviction import MIN_PREDICTIONS, Conviction, ConvictionLedger


def test_right_about_the_direction_and_wrong_about_the_size() -> None:
    led = ConvictionLedger()
    for _ in range(12):
        led.note(predicted=0.9, actual=0.15)
    reading = led.read()
    assert reading.measured
    assert reading.conviction == pytest.approx(1.0)
    assert reading.understanding < 0.3
    assert reading.knows_the_way_without_the_mechanism()


def test_a_coin_flip_on_direction_is_not_conviction() -> None:
    led = ConvictionLedger()
    for i in range(12):
        led.note(predicted=0.5 if i % 2 else -0.5, actual=0.5)
    reading = led.read()
    assert reading.conviction == pytest.approx(0.5)
    assert not reading.knows_the_way_without_the_mechanism()


def test_right_about_both_is_neither_case() -> None:
    led = ConvictionLedger()
    for _ in range(12):
        led.note(predicted=0.6, actual=0.6)
    reading = led.read()
    assert reading.conviction == pytest.approx(1.0)
    assert reading.understanding == pytest.approx(1.0)
    assert reading.gap == pytest.approx(0.0)
    assert not reading.knows_the_way_without_the_mechanism()


def test_the_size_is_only_scored_where_the_direction_was_right() -> None:
    """Scoring size on a prediction that pointed the wrong way counts two failures as one."""
    led = ConvictionLedger()
    for i in range(12):
        if i % 2:
            led.note(predicted=0.5, actual=0.5)      # right both ways
        else:
            led.note(predicted=0.5, actual=-0.5)     # wrong direction
    reading = led.read()
    assert reading.conviction == pytest.approx(0.5)
    assert reading.understanding == pytest.approx(1.0)


def test_the_sign_is_taken_against_a_resting_point_not_against_zero() -> None:
    """A quantity that rests at a half is not going up when it reads 0.4."""
    led = ConvictionLedger()
    for _ in range(10):
        led.note(predicted=0.4, actual=0.45, resting=0.5)
    assert led.read().conviction == pytest.approx(1.0)


def test_predicting_nothing_moves_is_a_direction() -> None:
    led = ConvictionLedger()
    for _ in range(10):
        led.note(predicted=0.5, actual=0.5, resting=0.5)
    assert led.read().conviction == pytest.approx(1.0)


def test_with_too_few_predictions_it_says_so() -> None:
    led = ConvictionLedger()
    for _ in range(MIN_PREDICTIONS - 1):
        led.note(predicted=0.9, actual=0.1)
    reading = led.read()
    assert not reading.measured
    assert not reading.knows_the_way_without_the_mechanism()
    assert "predictions needed" in reading.why


def test_unreadable_values_are_not_scored() -> None:
    led = ConvictionLedger()
    before = led.predictions()
    led.note(predicted="up", actual=0.5)  # type: ignore[arg-type]
    led.note(predicted=float("nan"), actual=0.5)
    assert led.predictions() == before


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = ConvictionLedger()
    for _ in range(10):
        led.note(predicted=0.7, actual=0.2)
    row = led.read().as_dict()
    for key in ("conviction", "understanding", "gap", "predictions", "measured",
                "knows_the_way_without_the_mechanism", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Conviction().measured
    assert not Conviction().knows_the_way_without_the_mechanism()
