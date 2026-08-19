"""An exact answer carries what produced it, into the next turn.

LIVE 2026-08-19: "Can you reverse a string for me? If so, reverse 'stressed'
and tell me exactly how you did it — model or code." The answer was right and
the account of it was invented — "I have a model capability for string
manipulation... I requested the reverse operation" — for a regex match and a
Python slice.

The answer travelled and its provenance did not. Nothing was wrong with the
computation, so no test about computation could have caught it.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.arithmetic_check import (
    requested_arithmetic_provenance,
    requested_arithmetic_result,
)
from core.conversation.computable_math import computable_result
from core.conversation.computable_text import computed_text_result
from core.conversation.computation_receipts import (
    asks_how_it_was_computed,
    clear_computation_receipts,
    how_it_was_computed_block,
    last_computation,
    record_computation,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_computation_receipts()
    yield
    clear_computation_receipts()


def test_the_provenance_names_the_code_object_that_ran():
    result = computed_text_result("reverse the word stressed")

    assert result is not None
    assert result.value == "desserts"
    assert result.module == "core.conversation.computable_text"
    assert result.function == "_reversed"
    assert "not generated" in result.provenance()


def test_the_provenance_is_read_off_the_function_not_written_down():
    """A sentence about a mechanism can drift from the mechanism."""
    import core.conversation.computable_text as module

    result = computed_text_result("reverse the word stressed")

    assert result is not None
    assert getattr(module, result.function, None) is not None


@pytest.mark.parametrize(
    ("question", "expected", "function"),
    [
        ("what is the gcd of 462 and 1071?", 21, "_gcd"),
        ("what is the area of a 3 by 4 rectangle", 12, "_rectangle_area"),
        ("how many digits are in 100 factorial?", 158, "_factorial_digits"),
    ],
)
def test_a_number_names_the_form_that_produced_it(question, expected, function):
    result = computable_result(question)

    assert result is not None
    assert result.value == expected
    assert result.function == function


def test_arithmetic_provenance_and_answer_come_from_one_traversal():
    """Two entry points that walk the branches separately will disagree."""
    for question in (
        "what is 47 * 89",
        "what is 15% of 200",
        "what is the gcd of 462 and 1071?",
    ):
        assert requested_arithmetic_result(question) is not None
        assert requested_arithmetic_provenance(question)

    assert requested_arithmetic_provenance("tell me a story") is None


def test_the_recorded_method_answers_the_next_turn():
    result = computed_text_result("reverse the word stressed")
    assert result is not None
    record_computation("reverse the word stressed", result.value, result.provenance())

    assert asks_how_it_was_computed("how did you do that?")
    block = how_it_was_computed_block("was that the model or code?")

    assert "computable_text._reversed" in block
    assert "desserts" in block
    assert "not produced by the language model" in block


def test_no_computation_means_no_account_invented():
    assert last_computation() is None
    assert how_it_was_computed_block("how did you do that?") == ""


def test_the_reading_serves_the_method_through_the_registry():
    """End to end through the observable, the way a turn reaches it."""
    from core.brain.observable_registry import (
        _matches_how_computed,
        _read_computed_text,
        _read_how_computed,
    )

    served = asyncio.run(_read_computed_text("reverse the word stressed"))
    assert "desserts" in served
    # The same reading that gives the answer states the mechanism.
    assert "computable_text._reversed" in served

    assert _matches_how_computed("how did you get that?")
    assert not _matches_how_computed("how are you doing")
    assert "_reversed" in asyncio.run(_read_how_computed("how did you get that?"))
