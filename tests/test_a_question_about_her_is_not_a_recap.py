"""A question about what SHE did is not a request for the transcript.

LIVE, 2026-09-21. Asked "Name one thing you got wrong earlier in this
conversation, if anything", Aura answered with a list of the last four
exchanges — and the list contained the very answer she had been asked to
judge (she had said the bird flies 480 km; it flies 240).

Two things put it there.

`_CONVERSATION_RECALL_TOPIC_MARKERS` held the bare string "earlier in this
conversation". Every other marker in that tuple is a whole ask — "what did
we discuss", "what was the topic". That one is a time adverbial and can sit
inside any question at all.

And the guard that already exists for this — `_CONVERSATION_RECALL_HER_WORDS_RE`,
written on 2026-09-16 for "what was the key reason you gave" — is a list of
verbs about what she SAID. A question can be about what she DID.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_memory_state import (
    _classify_conversation_recall_request as classify,
)


@pytest.mark.parametrize(
    "asked",
    [
        "Name one thing you got wrong earlier in this conversation, if anything.",
        "Did you get anything wrong earlier?",
        "Were you wrong about the bird?",
        "What did you miss in that answer?",
        "Have you changed your mind about any of it?",
        "What was the reason you gave?",
        "Was your answer right?",
    ],
)
def test_a_question_about_her_goes_to_the_model(asked):
    assert classify(asked) == "", (
        "this asks her to judge her own conduct; a transcript summary is not "
        "an answer to it"
    )


@pytest.mark.parametrize(
    "asked",
    [
        "What did we discuss?",
        "What were we talking about?",
        "What did we discuss earlier in this conversation?",
        "Remind me what we talked about.",
        "Summarize our conversation.",
    ],
)
def test_an_actual_recap_still_gets_one(asked):
    assert classify(asked) == "topic"


def test_the_time_phrase_alone_does_not_classify():
    """The exact shape that failed: a real ask with the phrase inside it."""
    assert classify("Pick the weakest claim you made earlier in this conversation.") == ""
    assert classify("Count the questions I asked earlier in this conversation.") == ""


def test_every_topic_marker_is_a_whole_ask():
    """A marker that is a fragment matches questions it has no answer for."""
    from interface.routes.chat_memory_state import (
        _CONVERSATION_RECALL_TOPIC_MARKERS as markers,
    )

    for marker in markers:
        words = marker.split()
        assert words[0] in {
            "what",
            "can",
            "do",
            "remind",
            "summarize",
            "tell",
            "show",
        }, f"{marker!r} is a fragment, not an ask"
