"""A draft three gates called repairable must reach the person if repair fails.

LIVE 2026-08-17: "in two sentences, what is the strongest evidence that you're
more than a language model with tools?" was answered "I couldn't get to an
answer I'd stand behind on that one."

    reply_reliability_gate_failed:missing_requested_sentence_count
    CognitiveEngine completion replacement remained incomplete; withholding it
    from the user surface.

The draft answered the question and missed the requested sentence count. The
gate rejected it for the formatting miss, the replacement came back incomplete
and was withheld, and a real answer was traded for an apology.

The last-resort refusal site already reads preserved_draft() for exactly this.
Nothing ever wrote it — preserve_draft() had zero callers in the chat route, so
the reader was permanently empty and the salvage could never fire.
"""

from __future__ import annotations

import inspect

import pytest

from core.conversation.surface_disposition import (
    disposition_for,
    draft_is_servable,
    preserve_draft,
    preserved_draft,
)


def test_a_sentence_count_miss_is_repairable_not_discardable() -> None:
    """It is a formatting shortfall, not a failure to answer."""
    assert draft_is_servable(["missing_requested_sentence_count"]) is True
    assert str(disposition_for(["missing_requested_sentence_count"])).endswith("REPAIR")


def test_the_chat_gate_now_preserves_the_draft() -> None:
    """Without a writer, the salvage reader is permanently empty."""
    from interface.routes import chat

    source = inspect.getsource(chat)

    assert "preserve_draft(assessment_text)" in source


def test_preservation_round_trips() -> None:
    preserve_draft("A real answer that missed the sentence count.")

    assert "missed the sentence count" in preserved_draft()


@pytest.mark.parametrize(
    "reasons",
    [
        ["missing_requested_sentence_count"],
        ["missing_requested_word_count"],
        ["generic_assistant_language"],
    ],
)
def test_formatting_shortfalls_stay_servable(reasons: list[str]) -> None:
    """Discarding content over a format miss is the defect being fixed."""
    assert draft_is_servable(reasons) is True
