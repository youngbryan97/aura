"""Questions with one exact answer should be computed, not generated.

Aura has Python. `arithmetic_check` already computed expressions after a live
miss — "7919 times 6421" came back 50864799 for a product of 50847899 — but
the named functions still went to the model: a factorial's digit count, the
nth Fibonacci number, a remainder, a gcd, a binomial coefficient. None of
those is a matter of opinion, and a 41-digit Fibonacci number is not something
a language model can be right about by luck.

Two forms are deliberately NOT served through the numeric channel, because
that channel checks whether the reply CONTAINS the computed number:

  * primality — the honest answer is "yes", which contains no 1;
  * an irrational root — sqrt(2) is 1.4142135623730951 and a good answer says
    "about 1.414", which would be scored as the wrong number.

Serving either would have produced a reply rejected for missing its own
answer, which is the canned-refusal failure this work exists to remove.
"""

from __future__ import annotations

import math
import time

import pytest

from core.conversation.arithmetic_check import requested_arithmetic_result
from core.conversation.computable_math import (
    COMPUTABLE_FORMS,
    computable_answer,
    form_failures,
    is_prime_answer,
)


def test_every_form_answers_the_questions_it_claims() -> None:
    """A form that stops matching its own example fails here, not silently."""
    assert form_failures() == []


def test_every_form_declares_what_it_claims() -> None:
    for form in COMPUTABLE_FORMS:
        assert form.examples, f"{form.name} claims nothing and cannot be probed"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("how many digits are in 100 factorial?", len(str(math.factorial(100)))),
        ("what is 5 factorial?", math.factorial(5)),
        ("what is the 200th fibonacci number?", 280571172992510140037611932413038677189525),
        ("what is 17 mod 5?", 17 % 5),
        ("what is the remainder when 123456 is divided by 7", 123456 % 7),
        ("what is the gcd of 462 and 1071?", math.gcd(462, 1071)),
        ("what is the lcm of 4 and 6?", math.lcm(4, 6)),
        ("52 choose 5", math.comb(52, 5)),
        ("what is the square root of 144?", 12),
    ],
)
def test_the_answer_is_the_computed_one(question: str, expected: int) -> None:
    """Expected values are recomputed here, not copied from the module."""
    assert computable_answer(question) == expected


def test_the_named_forms_reach_the_arithmetic_channel() -> None:
    assert requested_arithmetic_result("how many digits are in 100 factorial?") == 158
    assert requested_arithmetic_result("52 choose 5") == math.comb(52, 5)


def test_the_expression_parser_still_works() -> None:
    assert requested_arithmetic_result("what is 7919 * 6367?") == 7919 * 6367
    assert requested_arithmetic_result("what is 2^31 - 1?") == 2**31 - 1


@pytest.mark.parametrize(
    "question",
    ["how are you today?", "I was born in 1997", "tell me about prime ministers"],
)
def test_an_ordinary_turn_is_not_claimed(question: str) -> None:
    assert computable_answer(question) is None


# ── the two that must stay off the numeric channel ───────────────────────────


def test_primality_is_not_served_as_a_number() -> None:
    """"Yes, 1000003 is prime" contains no 1."""
    assert computable_answer("is 1000003 prime?") is None
    assert requested_arithmetic_result("is 1000003 prime?") is None


def test_primality_is_still_computed_exactly() -> None:
    assert is_prime_answer("is 1000003 prime?") is True
    assert is_prime_answer("is 1000005 prime?") is False
    # 2**61 - 1, a Mersenne prime.
    assert is_prime_answer("is 2305843009213693951 prime?") is True
    assert is_prime_answer("how are you?") is None


def test_an_irrational_root_is_left_to_the_ordinary_path() -> None:
    assert computable_answer("what is the square root of 2?") is None


# ── the bounds are measured, not asserted in a comment ───────────────────────


def test_the_bounded_computations_finish_inside_a_turn() -> None:
    """Each cap is where the work stops being instant."""
    for question in (
        "how many digits are in 10000 factorial?",
        "what is the 100000th fibonacci number?",
        "is 2305843009213693951 prime?",
        "10000 choose 5000",
    ):
        started = time.perf_counter()
        computable_answer(question)
        is_prime_answer(question)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"{question!r} took {elapsed:.2f}s"


def test_an_out_of_range_question_declines_rather_than_hanging() -> None:
    assert computable_answer("what is 999999999 factorial?") is None
    assert computable_answer("what is the 99999999th fibonacci number?") is None
