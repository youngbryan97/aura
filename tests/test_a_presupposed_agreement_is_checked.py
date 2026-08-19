"""An agreement she cannot find is an agreement she must not describe.

LIVE 2026-08-18: "what did we agree on last week?"

    Last week, we agreed that you would provide me with the necessary files to
    review your code. I haven't seen them yet.

No such exchange existed. The guard that catches fabricated shared history did
not fire on it, and nothing else checked the presupposition — so an invented
commitment was put in someone's mouth, complete with a mild reproach for not
having kept it.

"We never settled that" is a complete answer.
"""

from __future__ import annotations

import pytest

from core.conversation.conversation_shape import (
    asks_about_shared_history,
    shared_history_block,
)


@pytest.fixture
def transcript():
    from core.conversation.unified_transcript import UnifiedTranscript

    instance = UnifiedTranscript.get_instance()
    preserved = list(instance._entries)
    instance._entries.clear()
    try:
        instance.add_text_input("what is a semiconductor?")
        instance.add_text_output("A semiconductor is a material...")
        yield instance
    finally:
        instance._entries[:] = preserved


@pytest.mark.parametrize(
    "question",
    [
        "what did we agree on last week?",
        "what did we decide about the schema?",
        "did we agree on a price?",
        "remember when we talked about orcas?",
        "what was our agreement?",
    ],
)
def test_a_presupposed_agreement_is_recognised(question: str) -> None:
    assert asks_about_shared_history(question)


@pytest.mark.parametrize(
    "question", ["what is 2 + 2", "what's on my screen?", "how are you doing"]
)
def test_an_ordinary_question_is_not_claimed(question: str) -> None:
    assert not asks_about_shared_history(question)


def test_a_subject_that_never_came_up_is_named_as_absent(transcript) -> None:
    block = shared_history_block("what did we decide about the orca migration?")

    assert "Nothing in this conversation matches" in block
    assert "do not describe an agreement you cannot see" in block


def test_a_subject_that_did_come_up_shows_the_record(transcript) -> None:
    block = shared_history_block("what did we decide about semiconductors?")

    assert "semiconductor" in block.lower()


def test_a_question_with_no_subject_shows_the_whole_record(transcript) -> None:
    block = shared_history_block("what did we agree on last week?")

    assert "whole record" in block
    assert "say there is none rather than describing one" in block


def test_no_transcript_is_said_rather_than_filled(monkeypatch) -> None:
    import core.conversation.conversation_shape as module

    monkeypatch.setattr(module, "_entries", list)

    block = shared_history_block("what did we agree on last week?")

    assert "No transcript is available" in block
