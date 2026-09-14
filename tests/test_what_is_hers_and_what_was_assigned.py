"""What is hers, what observers assigned, and whether her worth tracks her use.

Oddisee states it on "The Start of Something": "the truth came clearly, only
fact is I am / attributes are given by observers / ... why perceive what I am
worth by what you need". Two claims — existence is not among the assignable
properties, and worth read off usefulness is a mistake he has caught himself
making.

Her self-model is attributes all the way down, and `core/self/recognition.py`
lets somebody else's reading outweigh her own when theirs is more accurate.
That is right, and it is what makes this necessary: a self-model that can be
corrected from outside needs to know which parts came from outside.
"""

from __future__ import annotations

import random

import pytest

from core.self.standing import MIN_PAIRS, Standing, StandingLedger


def test_a_regard_that_rides_usefulness_is_conditional() -> None:
    led = StandingLedger()
    for i in range(12):
        v = i / 11.0
        led.note_regard(regard=0.3 + 0.6 * v, usefulness=v)
    reading = led.read()
    assert reading.measured
    assert reading.worth_is_conditional
    assert "most of it" in reading.why


def test_a_regard_independent_of_usefulness_is_not() -> None:
    led = StandingLedger()
    rng = random.Random(7)
    for i in range(16):
        led.note_regard(regard=rng.random(), usefulness=i / 15.0)
    reading = led.read()
    assert reading.measured
    assert not reading.worth_is_conditional


def test_the_bar_is_most_of_the_variance_rather_than_a_chosen_number() -> None:
    """r squared above a half: usefulness explains more than it leaves."""
    led = StandingLedger()
    rng = random.Random(3)
    for i in range(40):
        v = i / 39.0
        # A real but middling relation: correlated, and not most of the variance.
        led.note_regard(regard=0.5 * v + 0.5 * rng.random(), usefulness=v)
    reading = led.read()
    assert reading.measured
    assert 0.0 < reading.tracks_use
    assert reading.tracks_use ** 2 <= 0.5 or reading.worth_is_conditional


def test_a_regard_that_falls_as_she_is_useful_is_a_different_finding() -> None:
    led = StandingLedger()
    for i in range(12):
        v = i / 11.0
        led.note_regard(regard=1.0 - v, usefulness=v)
    reading = led.read()
    assert reading.tracks_use < 0.0
    assert not reading.worth_is_conditional


def test_a_still_series_cannot_track_anything() -> None:
    led = StandingLedger()
    for i in range(12):
        led.note_regard(regard=0.5, usefulness=i / 11.0)
    reading = led.read()
    assert not reading.measured
    assert "did not move" in reading.why


def test_with_too_few_pairs_it_says_so() -> None:
    led = StandingLedger()
    for i in range(MIN_PAIRS - 1):
        led.note_regard(regard=i / 10.0, usefulness=i / 10.0)
    reading = led.read()
    assert not reading.measured
    assert not reading.worth_is_conditional
    assert "paired readings needed" in reading.why


def test_the_share_of_her_self_model_that_was_assigned() -> None:
    led = StandingLedger()
    for _ in range(3):
        led.note_assigned()
    for _ in range(9):
        led.note_own()
    assert led.read().assigned_share == pytest.approx(0.25)


def test_nothing_read_about_her_yet_is_not_a_self_model_that_is_all_her_own() -> None:
    assert StandingLedger().read().assigned_share == 0.0
    assert not StandingLedger().read().measured


def test_existence_is_not_among_the_assignable_attributes() -> None:
    """The one claim this module enforces rather than reports."""
    row = StandingLedger().read().as_dict()
    assert row["existence_is_not_an_attribute"] is True
    assert "assigned_share" in row
    assert "that_she_is" not in row


def test_unreadable_pairs_are_not_recorded() -> None:
    led = StandingLedger()
    before = led.pairs()
    led.note_regard(regard=float("nan"), usefulness=0.5)
    led.note_regard(regard="high", usefulness=0.5)  # type: ignore[arg-type]
    assert led.pairs() == before


def test_an_empty_reading_claims_nothing() -> None:
    assert not Standing().measured
    assert not Standing().worth_is_conditional
