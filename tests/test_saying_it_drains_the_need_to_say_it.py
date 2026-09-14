"""Repetition discharges. The fifth telling does not press like the first.

Eight repetitions of one line close that record, and what they do to a
listener is drain the feeling. These pin the reading: what counts as having
said something, and how much pressure is left once it has been said.
"""

from __future__ import annotations

import pytest

from core.affect.catharsis import content_words, drain, read_catharsis, times_said


def _said(role: str, text: str) -> dict:
    return {"role": role, "content": text}


HISTORY = [
    _said("user", "how is the migration going"),
    _said("assistant", "I have been working on the migration schedule all morning"),
    _said("user", "any luck"),
    _said("assistant", "the migration schedule is still what I am working on"),
    _said("assistant", "something else entirely"),
]


def test_content_words_are_what_an_utterance_is_about() -> None:
    assert content_words("Working on the migration schedule") == ("working", "migration", "schedule")
    assert content_words("") == ()
    assert content_words("to be or not to be") == ()


def test_a_turn_that_contains_every_content_word_said_it() -> None:
    # Both of her turns carry working, migration and schedule, in either order.
    assert times_said("working on the migration schedule", HISTORY) == 2
    assert times_said("the migration schedule", HISTORY) == 2
    # One extra content word neither turn has, and neither turn said it.
    assert times_said("the migration schedule rollback", HISTORY) == 0
    assert times_said("a thing nobody mentioned", HISTORY) == 0


def test_what_the_other_person_said_is_not_her_having_said_it() -> None:
    theirs = [_said("user", "the migration schedule is late")]
    assert times_said("the migration schedule", theirs) == 0


def test_saying_it_halves_the_pressure_and_saying_it_again_thirds_it() -> None:
    assert drain(0) == 1.0
    assert drain(1) == pytest.approx(0.5)
    assert drain(2) == pytest.approx(1 / 3)
    assert drain(3) == pytest.approx(0.25)


def test_the_pressure_never_reaches_zero() -> None:
    """A thing said is not a thing resolved."""
    assert drain(50) > 0.0
    assert drain("not a count") == 1.0


def test_the_reading_says_how_much_is_left_and_why() -> None:
    fresh = read_catharsis("a brand new worry", HISTORY)
    assert fresh.times == 0
    assert fresh.drain == 1.0
    assert "nothing about this" in fresh.why

    worn = read_catharsis("the migration schedule", HISTORY)
    assert worn.times == 2
    assert worn.drain == pytest.approx(1 / 3)
    assert "already" in worn.why


def test_an_utterance_with_no_content_words_claims_nothing() -> None:
    reading = read_catharsis("to be or not", HISTORY)
    assert reading.times == 0
    assert reading.drain == 1.0
    assert "no content words" in reading.why


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = read_catharsis("the migration schedule", HISTORY).as_dict()
    for key in ("times", "drain", "said", "why"):
        assert key in row, key
