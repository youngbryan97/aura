"""Being right, which had no signal anywhere in the system.

Every part of this mind measures prediction error and none of it fires when
reality agrees. The self prediction loop's own docstring says low error should
raise confidence in the self-model and nothing carried that out. She could be
wrong and feel it and could not be right and feel it.

Confirmation is the same estimator as surprise read from the other tail of her
own error distribution, which gets the important property for free: being
right about something she is always right about carries nothing.
"""

from __future__ import annotations

import pytest

from core.affect.confirmation import MIN_HISTORY, Expectation, ExpectationLedger


def _usually(error: float, n: int = 12) -> ExpectationLedger:
    led = ExpectationLedger()
    led.load([error] * n)
    return led


def test_an_error_far_below_her_usual_is_a_confirmation() -> None:
    led = _usually(0.5)
    reading = led.score(0.02)
    assert reading.measured
    assert reading.confirmed()
    assert not reading.surprised()
    assert reading.confirmation_bits > 1.0


def test_an_error_far_above_her_usual_is_a_surprise() -> None:
    led = _usually(0.5)
    reading = led.score(0.95)
    assert reading.surprised()
    assert not reading.confirmed()


def test_an_ordinary_error_is_neither() -> None:
    led = ExpectationLedger()
    led.load([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.45, 0.55, 0.5])
    reading = led.score(0.5)
    assert reading.measured
    assert not reading.confirmed()
    assert not reading.surprised()


def test_being_right_about_the_easy_thing_is_not_an_experience() -> None:
    """The property the whole design is for."""
    led = _usually(0.01)
    reading = led.score(0.01)
    assert reading.measured
    assert reading.confirmation_bits == pytest.approx(0.0, abs=1e-9)
    assert not reading.confirmed()


def test_the_same_hit_is_worth_more_to_someone_who_usually_misses() -> None:
    sloppy = _usually(0.6)
    sharp = _usually(0.05)
    assert sloppy.score(0.05).confirmation_bits > sharp.score(0.05).confirmation_bits


def test_bits_are_never_negative() -> None:
    led = _usually(0.5)
    for error in (0.0, 0.5, 1.0, 10.0, -3.0):
        reading = led.score(error)
        assert reading.confirmation_bits >= 0.0
        assert reading.surprise_bits >= 0.0
        assert 0.0 <= reading.confirmation <= 1.0
        assert 0.0 <= reading.surprise <= 1.0


def test_with_too_little_history_it_says_so_rather_than_reporting_none() -> None:
    led = ExpectationLedger()
    led.load([0.5] * (MIN_HISTORY - 1))
    reading = led.score(0.01)
    assert not reading.measured
    assert not reading.confirmed()
    assert "scored predictions needed" in reading.why


def test_the_ceiling_is_what_her_history_can_resolve() -> None:
    """Not a chosen bound: with n samples the rarest tail is 1/(n+1)."""
    import math

    led = _usually(0.5, n=31)
    assert led.ceiling_bits() == pytest.approx(math.log2(32))
    assert led.score(0.0).confirmation <= 1.0


def test_an_outcome_is_scored_against_the_history_before_it() -> None:
    """Folding it in first would let every outcome make itself ordinary."""
    led = _usually(0.5)
    first = led.score_and_note(0.01)
    assert first.confirmed()
    assert led.samples() == 13


def test_the_window_forgets_a_version_of_her_that_no_longer_exists() -> None:
    led = ExpectationLedger(window=10)
    led.load([0.9] * 10)
    led.load([0.1] * 10)
    assert led.samples() == 10
    # Against the recent, accurate her, a 0.9 is now a surprise.
    assert led.score(0.9).surprised()


def test_a_non_number_is_not_a_score() -> None:
    led = _usually(0.5)
    before = led.samples()
    led.note(float("nan"))
    led.note("wrong")  # type: ignore[arg-type]
    assert led.samples() == before
    assert not led.score("wrong").measured  # type: ignore[arg-type]


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = _usually(0.5).score(0.02).as_dict()
    for key in ("error", "confirmation", "surprise", "confirmation_bits",
                "surprise_bits", "ceiling_bits", "samples", "measured",
                "confirmed", "surprised", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Expectation(error=0.0).measured
    assert not Expectation(error=0.0).confirmed()
    assert not Expectation(error=0.0).surprised()
