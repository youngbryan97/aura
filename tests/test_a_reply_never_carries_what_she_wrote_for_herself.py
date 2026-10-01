"""What a model writes for itself never reaches what a person reads.

LIVE 2026-10-01: asked what a personality test would say about her, her
forecast reached the chat as "<internal_critique> First instinct: ..." and the
prediction itself was never shown. One rule in her assembled context asked
for the critique to be written; that rule is gone, and anything that slips
through is stripped where her words are read.
"""
from __future__ import annotations

import inspect

import pytest

from core.language.answer_surface import without_private_markup

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("<internal_critique>INTP, maybe.</internal_critique>I expect INTJ.", "I expect INTJ."),
        ("<internal_critique>First instinct: INTP.\n\nI expect INTJ.", "I expect INTJ."),
        ("<think>hmm</think>Answer.", "Answer."),
        ("No markup, and a < b.", "No markup, and a < b."),
    ],
)
def test_a_private_block_is_taken_out(written: str, said: str) -> None:
    assert without_private_markup(written) == said


def test_an_unclosed_block_with_no_break_keeps_the_answer_in_it() -> None:
    """Dropping everything after an unclosed tag would drop the answer too."""
    assert without_private_markup("<internal_critique>I expect INTJ.") == "I expect INTJ."


def test_no_rule_asks_her_to_write_a_critique_into_her_answer() -> None:
    from core.brain.llm import context_assembler

    assert "perform an <internal_critique>" not in inspect.getsource(context_assembler)
