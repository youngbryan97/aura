"""The commonest computable question in existence was not computable.

LIVE, 2026-08-19. ``computable_answer("what is 17 * 4839")`` returned None.
Eight exotic forms were covered — primality, factorials, Fibonacci, gcd, digit
counts — and plain arithmetic was not, so an ordinary sum depended entirely on
the model getting it right. When it did not, ``arithmetic_answer_missing``
destroyed every draft and the turn ended in a canned apology.

The evaluation goes through ``ast``, never ``eval``: only literals and the
arithmetic operators exist in that grammar, so there is no name to resolve and
no call to make.

The counter-tests matter as much as the tests. Numbers with an operator
between them are everywhere in ordinary prose — "the 2015 - 2020 period" is
-5, "call me at 555-1234" is -679 — and a form that answers those is worse
than one that answers nothing.
"""

from __future__ import annotations

import pytest

from core.conversation.computable_math import computable_answer, form_failures


@pytest.mark.parametrize(
    "question,expected",
    [
        ("what is 17 * 4839", 82_263),
        ("what is 2 to the power of 40", 1_099_511_627_776),
        ("what's 1024 / 8?", 128),
        ("how much is 12,500 + 3,750", 16_250),
        ("what is 7 times 6", 42),
        ("what is 100 - 58", 42),
        ("calculate 99 * 99", 9_801),
        ("work out 250 / 4", 62.5),
        ("what is 17 % 5", 2),
        # The expression alone is the request.
        ("17 * 4839", 82_263),
    ],
)
def test_ordinary_arithmetic_is_answered_exactly(question, expected):
    assert computable_answer(question) == expected


@pytest.mark.parametrize(
    "text",
    [
        "the 2015 - 2020 period was rough",
        "I need a 5-10 minute break",
        "call me at 555-1234",
        "she scored 9/10 on the test",
        "my rent went from 1200 to 1450",
        "chapter 4 section 2",
        "we shipped 3 + 2 features last week",
        "version 3.11.2 broke it",
        "a 3x5 index card",
    ],
)
def test_prose_that_merely_contains_numbers_is_not_a_sum(text):
    assert computable_answer(text) is None


def test_a_division_that_is_not_whole_keeps_its_fraction():
    value = computable_answer("what is 22 / 7")
    assert isinstance(value, float)
    assert abs(value - 3.142857142857) < 1e-9


def test_a_whole_division_reads_as_a_whole_number():
    """6/3 is 2, not 2.0 — the answer a person would write."""
    assert computable_answer("what is 6 / 3") == 2
    assert isinstance(computable_answer("what is 6 / 3"), int)


def test_an_exponent_cannot_be_used_to_hang_the_turn():
    """9**9**9 is a denial of service, not a sum."""
    assert computable_answer("what is 9 ** 9 ** 9") is None
    assert computable_answer("what is 2 to the power of 999999") is None


def test_division_by_zero_answers_nothing_rather_than_raising():
    assert computable_answer("what is 5 / 0") is None


def test_a_narrower_named_form_still_wins():
    """The general form is asked last, so "5 factorial" is not "5"."""
    assert computable_answer("what is 5 factorial?") == 120


def test_every_declared_example_across_all_forms_holds():
    """A form that cannot answer its own examples is a claim with no test."""
    assert form_failures() == []


def test_a_caret_is_exponentiation():
    """Everywhere except Python.

    Without this, "what is 2^31 - 1?" matched the FRAGMENT "31 - 1" and
    answered 30 — a regression this form introduced against a parser that had
    handled it correctly.
    """
    assert computable_answer("what is 2^31 - 1?") == 2**31 - 1
    assert computable_answer("what is 3^4") == 81


def test_a_fragment_of_a_longer_expression_is_never_answered():
    """Its value is not the answer to anything that was asked."""
    from core.conversation.computable_math import _ArithmeticPattern

    assert _ArithmeticPattern().search("what is 2 ** 31 - 1").group(0).strip() == "2 ** 31 - 1"
