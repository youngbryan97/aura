"""A marker that is a fragment of an ordinary word will meet that word.

Live 2026-08-18. Asked "search the web and tell me what the latest Claude
model is, and cite your sources", Aura opened a browser, tried to hold an
eight-turn conversation with Claude, and answered:

    I routed the Claude conversation through the governed web_interlocutor
    skill, but I am not claiming a successful proof: no_visible_editable_field.
    Observed 0/8 turns; memory=none.

No search ran. The gate matched the target "claude" and the action "test" —
"test" inside "la-test". This is the third instance of the shape in this
codebase: "in your own words" launched Microsoft Word, and "notes.txt" opened
the Notes app. Markers are words.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_capability_inventory import (
    _looks_like_web_interlocutor_execution_request as asks_for_interlocutor,
)


@pytest.mark.parametrize(
    "message",
    [
        # The live miss, and its neighbours.
        "search the web and tell me what the latest Claude model is, and cite your sources",
        "what is the latest Claude model?",
        "the latest test results are in",
        "google the latest claude model for me",
        "search the web for news about fusion power",
        # A report about an action is not a request for one.
        "ChatGPT runs tests on Aura",
    ],
)
def test_an_ordinary_sentence_does_not_open_a_browser_conversation(
    message: str,
) -> None:
    assert not asks_for_interlocutor(message)


@pytest.mark.parametrize(
    "message",
    [
        "open ChatGPT and have a conversation about recursion",
        "ask Gemini what it thinks about recursion and report back",
        "talk to another AI and learn from it",
        "go to Claude and discuss consciousness with it",
    ],
)
def test_a_real_interlocutor_request_still_routes(message: str) -> None:
    assert asks_for_interlocutor(message)


def test_a_marker_inside_a_longer_word_does_not_count() -> None:
    """The exact mechanism: "test" is in "latest"."""
    assert not asks_for_interlocutor("tell me the latest claude model")
    assert asks_for_interlocutor("test claude and report back")
