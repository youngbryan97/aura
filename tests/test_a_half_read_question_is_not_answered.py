"""A question with a step this runtime cannot compute gets no computed answer.

LIVE DEFECT, 2026-09-20. "17 x 23 then subtract the letters in the word
probability" came back "391.". The brainstem had already generated the whole
correct answer — 391, eleven letters, 380, 95 characters of it in the log —
and the deterministic arithmetic check replaced it with the product alone,
because the first step parses and the second does not.

The check is served with the authority of "run as Python, not generated", so
a half-read question is not a weaker answer than the model's, it is a wrong
one that outranks a right one. Declining costs a fallback.

The same half-reading answered "what is the square root of 16 plus 9" with 4,
"what is 5 factorial minus 20" with 120, and "the gcd of 12 and 18 times 3"
with 54.
"""

from __future__ import annotations

import pytest

from core.conversation.arithmetic_check import (
    requested_arithmetic_provenance,
    requested_arithmetic_result,
)
from core.conversation.computable_math import computable_answer, operation_outside

#: Two steps, one of them beyond this runtime's forms. No computed answer.
HALF_READ = (
    "What is 17 times 23, then subtract the number of letters in the word "
    "probability? Give me the final number.",
    "17 x 23 then subtract the letters in probability",
    "what is the square root of 16 plus 9",
    "what is 5 factorial minus 20",
    "the gcd of 12 and 18 times 3",
    "multiply 47 by 89 and add 12",
    "what is 144 / 6 + seven?",
)

#: The null. Each of these is one step, and each must still be computed —
#: a guard that refuses everything would pass the test above and destroy the
#: capability it protects.
STILL_COMPUTED: tuple[tuple[str, int | float], ...] = (
    ("what is 7919 times 6421? just the number.", 50_847_899),
    ("17 times 23", 391),
    ("2+2", 4),
    ("what is 2+2", 4),
    ("compute 2^31 - 1", 2_147_483_647),
    ("What's 144 / 6 + 7? Just the number.", 31),
    ("multiply 7919 by 6421", 50_847_899),
    ("subtract 5 from 20", 15),
    ("what is 100 minus 45 minus 5", 50),
    ("what is 20% of 50", 10.0),
    ("what is 2 to the 3rd power", 8),
    ("what is the remainder when 17 is divided by 5", 2),
    ("what is 5 factorial", 120),
    ("what is the square root of 16", 4),
    ("what is the gcd of 12 and 18", 6),
    ("what is 17 * 4839", 82_263),
    ("what is 2 to the power of 40", 1_099_511_627_776),
    ("how much is 12,500 + 3,750", 16_250),
    ("what is 7 times 6", 42),
    ("what is the area of a 3 by 4 rectangle", 12),
    ("how many digits are in 100 factorial", 158),
    ("what is 1,000 * 2", 2_000),
)


@pytest.mark.parametrize("question", HALF_READ)
def test_a_step_it_cannot_compute_means_no_computed_answer(question: str) -> None:
    assert requested_arithmetic_result(question) is None
    assert requested_arithmetic_provenance(question) is None


@pytest.mark.parametrize(("question", "expected"), STILL_COMPUTED)
def test_one_step_questions_are_still_computed(
    question: str, expected: int | float
) -> None:
    assert requested_arithmetic_result(question) == expected
    assert requested_arithmetic_provenance(question)


def test_the_live_question_keeps_its_product_out_of_the_answer() -> None:
    """The product is 391 and the answer is 380, so 391 must not be served."""
    asked = "17 x 23 then subtract the letters in probability"
    assert 17 * 23 == 391
    assert requested_arithmetic_result(asked) != 391


def test_a_named_form_does_not_hand_a_composite_to_the_next_form() -> None:
    """The gcd form declining must not let the expression form take the tail."""
    assert computable_answer("the gcd of 12 and 18 times 3") is None
    assert computable_answer("what is the gcd of 12 and 18") == 6


def test_the_guard_names_the_step_it_saw() -> None:
    text = "17*23 then subtract the letters in probability"
    assert operation_outside(text, 0, 5) == "subtract"
    assert operation_outside("what is 17*23", 8, 13) is None
