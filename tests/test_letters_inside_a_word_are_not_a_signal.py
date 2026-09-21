"""What decides whether a turn gets the full prompt or a trimmed one.

`_is_casual_interaction` tested its signal lists with `in`, so "ok" matched
inside look, book, broke and token; "hi" inside this, which and hire; "real"
inside really, realize and unreal. A request to "tokenize this" carried a
casual signal and "it looks unreal" carried two, and major state, world and
research blocks were included or dropped on the strength of letters sitting
inside unrelated words.
"""
from __future__ import annotations

import pytest

from core.brain.llm.context_assembler import (
    _CASUAL_MAX_WORDS,
    _CASUAL_RE,
    _DELIBERATE_RE,
    ContextAssembler,
)


@pytest.mark.parametrize(
    "text",
    [
        "can you tokenize this string",
        "the book is on the shelf",
        "we broke the build",
        "which file did you mean",
    ],
)
def test_letters_inside_a_word_are_not_a_casual_signal(text):
    assert _CASUAL_RE.search(text) is None, text


@pytest.mark.parametrize(
    "text",
    [
        "it looks unreal",
        "he realized the mistake",
        "the realtor called",
        "unthinking machinery",
    ],
)
def test_letters_inside_a_word_are_not_a_deliberate_signal(text):
    assert _DELIBERATE_RE.search(text) is None, text


@pytest.mark.parametrize("text", ["ok", "thanks", "hey", "got it"])
def test_the_real_casual_words_still_match(text):
    assert _CASUAL_RE.search(text) is not None, text


@pytest.mark.parametrize(
    "text", ["do you feel anything", "explain the architecture", "what do you remember"]
)
def test_the_real_deliberate_words_still_match(text):
    assert _DELIBERATE_RE.search(text) is not None, text


def test_a_short_greeting_is_casual():
    assert ContextAssembler._is_casual_interaction("hey") is True


def test_a_question_about_consciousness_is_not_casual():
    assert ContextAssembler._is_casual_interaction("Is Aura conscious?") is False


def test_a_casual_word_in_a_long_message_is_a_word_not_a_register():
    long_message = "ok " + "and then we need to reconcile the two ledgers " * 3

    assert len(long_message.split()) > _CASUAL_MAX_WORDS
    assert ContextAssembler._is_casual_interaction(long_message) is False


def test_unrecognised_input_gets_the_full_prompt():
    """Another language, a code paste, anything adversarial. A full prompt for
    small talk costs context; a trimmed prompt for a real question costs the
    answer."""
    for text in (
        "¿puedes revisar el informe trimestral?",
        "def solve(n): return [i for i in range(n) if i % 7 == 0]",
        "現在の状態を教えてください",
    ):
        assert ContextAssembler._is_casual_interaction(text) is False, text


def test_the_tool_block_says_what_available_was_checked_against():
    """The list read as a guarantee. The catalog verifies the skill is enabled,
    validated, dependency-ready and past preflight; it never calls the tool, so
    nothing there proves a credential is current or a target answers."""
    from pathlib import Path as _Path

    from tests.source_contract import family_text_at

    # The tool block was lifted into `context_assembler_blocks` (2026-09-20).
    source = family_text_at(
        _Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "context_assembler.py"
    )

    assert "registered, validated and past preflight" in source
    assert "not proof the tool works right now" in source

    from core.brain.llm.context_assembler import ContextAssembler
    from core.container import ServiceContainer
    from core.state.aura_state import AuraState

    class Engine:
        @staticmethod
        def build_tool_affordance_block(**_kwargs):
            return "## LIVE TOOL OPTIONS\n- clock: Check time and date."

    ServiceContainer.clear()
    ServiceContainer.register_instance("capability_engine", Engine(), required=False)
    try:
        prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert "Treat the first use in a turn as the test" in prompt
