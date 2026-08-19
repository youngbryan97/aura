"""A fragment of an expression is not an answer to it — and ^ now computes.

LIVE 2026-08-17: "compute 2^31 - 1 and tell me if it's prime" was answered
"30". The extractor matched "31 - 1" and never looked at the "2^" sitting
immediately behind it.

Refusing the parse was the safe fix and not the right one. The correct value is
2147483647, which is exactly the kind of number a person asks a machine for
rather than working out by hand, so exponentiation is supported and the
fragment guard now only refuses operators that genuinely are not implemented.

This matters more than it used to: the turn SERVES a computed arithmetic value
directly instead of offering it to the model, so a partial parse is not a
misleading hint — it is the entire reply, with the runtime's authority behind
it.
"""

from __future__ import annotations

import pytest

from core.conversation.arithmetic_check import requested_arithmetic_result


def test_the_live_case_now_computes_correctly() -> None:
    assert requested_arithmetic_result(
        "compute 2^31 - 1 and tell me if it's prime"
    ) == 2147483647


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is 2^10", 1024),
        ("what is 2 + 2 ^ 3", 10),          # precedence, not left-to-right
        ("compute 3^3 - 2", 25),
        ("what is 7919 times 6421?", 50847899),
        ("what is 2 + 2", 4),
        ("what is 100 / 4", 25.0),
        ("what is 1,000 * 2", 2000),
    ],
)
def test_expressions_it_can_evaluate_are_evaluated(question: str, expected) -> None:
    assert requested_arithmetic_result(question) == expected


@pytest.mark.parametrize("question", ["what is 5! + 2"])
def test_operators_that_are_genuinely_unimplemented_are_refused(question: str) -> None:
    """Not implemented means not answered, never answered from the part it knows."""
    assert requested_arithmetic_result(question) is None


def test_the_remainder_operator_is_implemented_now() -> None:
    """It was on the unimplemented list until 2026-08-19.

    Refusing a question Python answers in one operator was the reason to build
    it rather than a property to preserve, so this asserts the answer instead —
    and it is recomputed here rather than copied.
    """
    assert requested_arithmetic_result("what is 10 % 3") == 10 % 3
    assert requested_arithmetic_result("what is 17 mod 5") == 17 % 5


def test_a_runaway_exponent_is_bounded_not_attempted() -> None:
    """9**9**9 would hang the turn it is supposed to answer."""
    assert requested_arithmetic_result("what is 9^9^9") is None
