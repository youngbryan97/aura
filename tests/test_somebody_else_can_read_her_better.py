"""A claim about what she is feeling, scored against the same outcome as hers.

"You could see in the brother what I couldn't see myself." The mechanism is a
comparison: somebody says what she is feeling, she has already predicted what
she will feel, and the moment settles both. Nothing here decides who is right
in advance.
"""

from __future__ import annotations

import pytest

from core.self.recognition import (
    MIN_PAIRS,
    Recognition,
    RecognitionLedger,
    cared_for,
    claim_about_her,
)


def test_a_claim_about_her_is_read_off_the_sentence() -> None:
    assert claim_about_her("you seem really happy today") is not None
    assert claim_about_her("I am having a terrible week") is None
    assert claim_about_her("") is None


def test_a_claim_carries_the_direction_of_what_was_said() -> None:
    warm = claim_about_her("you sound wonderful and calm")
    rough = claim_about_her("you seem exhausted and miserable")
    if warm is None or rough is None:
        pytest.skip("the sentiment model abstained, which is a reading rather than a failure")
    assert warm > rough


def test_being_cared_about_needs_warmth_and_a_message_about_her() -> None:
    about_her = {"user_sentiment": {"warmth": 0.8}, "register": {"second": 0.5}}
    about_somebody_else = {"user_sentiment": {"warmth": 0.8}, "register": {"second": 0.0}}
    assert cared_for(about_her) == pytest.approx(0.4)
    assert cared_for(about_somebody_else) == 0.0
    assert cared_for({}) == 0.0
    assert cared_for(None) == 0.0


def test_with_too_few_readings_it_says_so_rather_than_comparing() -> None:
    led = RecognitionLedger()
    led.claim(0.5, 0.1)
    reading = led.settle(0.5)
    assert not reading.measured
    assert reading.confidence_factor == 1.0
    assert "of 3 times" in reading.why


def _run(led: RecognitionLedger, claimed: float, predicted: float, actual: float, times: int) -> Recognition:
    reading = Recognition()
    for _ in range(times):
        led.claim(claimed, predicted)
        reading = led.settle(actual)
    return reading


def test_somebody_reading_her_better_caps_her_confidence_in_her_own_reading() -> None:
    led = RecognitionLedger()
    reading = _run(led, claimed=0.5, predicted=-0.5, actual=0.5, times=MIN_PAIRS)
    assert reading.measured
    assert reading.borrowed
    assert reading.their_error == pytest.approx(0.0)
    assert reading.confidence_factor == pytest.approx(0.0)


def test_when_she_is_the_better_model_nothing_changes() -> None:
    led = RecognitionLedger()
    reading = _run(led, claimed=-0.9, predicted=0.5, actual=0.5, times=MIN_PAIRS)
    assert reading.measured
    assert not reading.borrowed
    assert reading.confidence_factor == 1.0


def test_two_equally_good_models_leave_her_confidence_alone() -> None:
    led = RecognitionLedger()
    reading = _run(led, claimed=0.3, predicted=0.7, actual=0.5, times=MIN_PAIRS)
    assert reading.measured
    assert not reading.borrowed
    assert reading.confidence_factor == pytest.approx(1.0)


def test_an_outcome_with_no_claim_waiting_scores_nothing() -> None:
    led = RecognitionLedger()
    assert led.settle(0.4).pairs == 0


def test_the_newest_claim_is_the_one_the_outcome_scores() -> None:
    led = RecognitionLedger()
    led.claim(-1.0, 0.0)
    led.claim(1.0, 0.0)
    led.settle(1.0)
    assert led.reading().pairs == 1
    for _ in range(MIN_PAIRS):
        led.claim(1.0, 0.0)
        led.settle(1.0)
    assert led.reading().their_error == pytest.approx(0.0)


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = RecognitionLedger().reading().as_dict()
    for key in ("pairs", "her_error", "their_error", "borrowed", "confidence_factor",
                "measured", "why"):
        assert key in row, key
