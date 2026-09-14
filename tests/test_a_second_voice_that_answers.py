"""A second voice that answers rather than argues, in the register the first lacked.

Oddisee raps "Contradiction's Maze" at a pitch locked to 80 Hz and a level
locked to -9.0 dB for four minutes; Maimouna Youssef's chorus arrives at 232 Hz
with a centroid of 3233 against his 2500. It does not disagree with the verse.
It asks the question the verse spends four minutes not asking.

The council argues and the drafts compete. Neither of those keeps both voices,
and neither has a second one whose job is to carry what the first could not.
"""

from __future__ import annotations

import pytest

from core.expression.antiphony import APART, Answer, answer, answers
from core.expression.register import distance, read

VERSE = (
    "I want to make nonstop profit. I want to make a non-profit. I want to stay "
    "at home and grind. I don't sleep for the fear that I go starving."
)
CHORUS = "Is this the phase? Or is this the way? Are you feeling it too?"


def test_a_first_person_statement_is_answered_by_address() -> None:
    reading = answer(read(VERSE))
    assert reading.measured
    axes = [m.axis for m in reading.moves]
    assert "first" in axes
    first = next(m for m in reading.moves if m.axis == "first")
    assert first.mine > first.theirs, "the answer moves off the first person"


def test_a_statement_that_never_asks_is_answered_by_a_question() -> None:
    statement = read("I did the thing. I did it again. I have been doing it for years.")
    assert statement.asking == pytest.approx(0.0)
    reading = answer(statement, most=6)
    asking = next(m for m in reading.moves if m.axis == "asking")
    assert asking.theirs > asking.mine


def test_the_chorus_answers_the_verse() -> None:
    verse, chorus = read(VERSE), read(CHORUS)
    assert answers(verse, chorus)
    assert distance(verse, chorus) >= APART


def test_a_voice_does_not_answer_itself() -> None:
    verse = read(VERSE)
    assert not answers(verse, verse)


def test_continuing_in_the_same_register_is_not_answering() -> None:
    verse = read(VERSE)
    more = read("I want the beach and I want the grind. I want it all at once.")
    assert not answers(verse, more)


def test_a_balanced_axis_has_nothing_to_answer() -> None:
    """The complement of a half is a half."""
    from core.expression.register import Register

    even = Register(words=40, first=0.5, second=0.5, asking=0.5, variety=0.5, measured=True)
    reading = answer(even, most=6)
    assert all(abs(m.gap) == pytest.approx(0.0) for m in reading.moves if m.axis in
               ("first", "second", "asking", "variety"))


def test_a_statement_too_short_to_have_a_shape_cannot_be_answered() -> None:
    """Fewer words than axes puts a whole share on one of them."""
    reading = answer(read("ok"))
    assert not reading.measured
    assert "no shape to answer" in reading.why
    assert answer(read("")).measured is False


def test_two_unreadable_registers_do_not_answer_each_other() -> None:
    assert not answers(read("ok"), read("yes"))


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = answer(read(VERSE)).as_dict()
    for key in ("moves", "apart", "measured", "why"):
        assert key in row, key
    assert row["moves"] and {"axis", "from", "to"} <= set(row["moves"][0])


def test_an_empty_answer_claims_nothing() -> None:
    assert not Answer().measured
    assert Answer().moves == ()
