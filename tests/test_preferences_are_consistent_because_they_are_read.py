"""A preference she cannot see is a preference she does not have.

LIVE 2026-08-18, four ways of asking one question inside a few minutes:

    what topic pulls at you the most?      -> distributed systems consensus
    what's one thing you find interesting? -> the way minds work
    what topic pulls at you the most?      -> the architecture of cognition
    what topic pulls at you the most?      -> the neurodynamics of thought

Four answers, one of them twice from the identical prompt, and the sampler was
working correctly — an identical prompt legitimately produces different
wording. What was missing is that nothing put the earlier answer in front of
the next one, so a preference was generated fresh every time.

Consistency here is not a rule about what she may say. It is the ordinary
consequence of being able to see what she said before. She stays free to have
changed her mind, and saying so is a different sentence from silently
answering something else.
"""

from __future__ import annotations

import time

import pytest

from core.self.stated_preferences import (
    asks_about_her_preferences,
    stated_preference_block,
    stated_preferences,
)


@pytest.fixture
def spoken():
    from core.conversation.unified_transcript import UnifiedTranscript

    instance = UnifiedTranscript.get_instance()
    preserved = list(instance._entries)
    instance._entries.clear()
    try:
        instance.add_text_input("what topic pulls at you the most?")
        instance.add_text_output(
            "Distributed systems consensus. I find the coordination problem "
            "endlessly interesting."
        )
        instance.entries_for_conversation()[1].timestamp = time.time() - 420
        yield instance
    finally:
        instance._entries[:] = preserved


@pytest.mark.parametrize(
    "question",
    [
        "what topic pulls at you the most?",
        "what interests you?",
        "what's one thing you find genuinely interesting?",
        "name the one thing you'd study if nobody was watching.",
        "what's your favourite colour?",
        "do you have a preference?",
    ],
)
def test_a_preference_question_is_recognised(question: str) -> None:
    assert asks_about_her_preferences(question)


@pytest.mark.parametrize(
    "question",
    ["what did I just copy?", "what is 2 + 2", "what files are in core/runtime?"],
)
def test_another_question_is_not_claimed(question: str) -> None:
    assert not asks_about_her_preferences(question)


def test_her_earlier_answer_is_read_back(spoken) -> None:
    block = stated_preference_block(
        "name the one thing you'd study if nobody was watching."
    )

    assert "coordination problem" in block
    assert "minutes ago" in block


def test_the_reading_is_her_own_words_not_a_summary(spoken) -> None:
    found = stated_preferences()

    assert found
    assert found[0].text.startswith("I find the coordination problem")


def test_changing_her_mind_is_allowed_and_named(spoken) -> None:
    """The block must not read as an instruction to repeat herself."""
    block = stated_preference_block("what interests you?")

    assert "unless it has actually changed" in block


def test_nothing_said_yet_is_not_an_error(monkeypatch) -> None:
    import core.self.stated_preferences as module

    monkeypatch.setattr(module, "_own_turns", list)

    assert stated_preference_block("what interests you?") == ""


def test_the_user_s_own_words_are_not_mistaken_for_hers(spoken) -> None:
    """Only her turns count; the question itself contains preference words."""
    from core.conversation.unified_transcript import UnifiedTranscript

    UnifiedTranscript.get_instance().add_text_input(
        "I find woodworking genuinely interesting."
    )

    found = stated_preferences()

    assert all("woodworking" not in item.text for item in found)
