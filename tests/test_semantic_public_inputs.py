import hashlib

import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_public_inputs import (
    semantic_public_character_inputs,
    semantic_public_token_inputs,
)


def test_public_inputs_extract_top_level_closed_values_in_source_order():
    text = "Use [3, -14, 5], then combine 7 and -2; ignore 3.14 and v2."

    observed = semantic_public_character_inputs(text)

    assert observed.values == ((3, -14, 5), 7, -2)
    assert observed.source_text_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert observed.receipt()["family_router_present"] is False
    assert observed.receipt()["expected_answer_available"] is False


def test_public_inputs_support_empty_and_singleton_sequences():
    observed = semantic_public_character_inputs("Compare [] with (3,) and (4, -5).")

    assert observed.values == ((), (3,), (4, -5))


def test_public_inputs_bind_character_values_to_token_offsets():
    text = "Take [4, 8], then 3."
    offsets = ((0, 4), (5, 11), (11, 12), (13, 17), (18, 19), (19, 20))

    observed = semantic_public_token_inputs(text, offsets)

    assert observed.values == ((4, 8), 3)
    assert observed.token_spans == (TokenSpan(1, 2), TokenSpan(4, 5))


@pytest.mark.parametrize(
    ("text", "offsets"),
    (("", ((0, 0),)), ("3", ()), ("3", ((2, 3),))),
)
def test_public_inputs_reject_invalid_source_or_offsets(text, offsets):
    with pytest.raises(ValueError):
        semantic_public_token_inputs(text, offsets)
