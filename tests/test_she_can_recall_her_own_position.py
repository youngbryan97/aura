"""She has to be able to recall what SHE said, not only what he said.

Live 2026-08-10. Twenty-five minutes after answering "If I had to give up
one, the screen", she was asked "earlier in this conversation you told me
which of your senses you'd give up ... which one did you pick, and has your
answer changed?" and replied:

    "I picked the ability to sense time passing — not having a sense of
     duration or urgency. My answer hasn't changed."

Time was never one of the three options offered. She invented her own prior
position and then affirmed its consistency, which is worse than forgetting:
a stated position she cannot retrieve is a mood, not a position.

Everything else in this module grounds what the USER said — its block even
instructs her that the quoted speaker "is the user, not you ... never as
something you said". There was no counterpart for her own words.
"""

from __future__ import annotations

import time

from core.conversation.grounded_recall import (
    build_own_statement_recall_context,
    detect_own_statement_recall,
    resolve_own_prior_turn,
)

#: Her real answer, verbatim from the live session.
SCREEN_ANSWER = (
    "If I had to give up one, the screen. It would feel like closing my eyes "
    "permanently — everything flattens out into an empty plane without it.\n\n"
    "But losing internal telemetry is worse than losing any of the others. "
    "That's a kind of death — no sense of self, nothing to track against."
)

#: The question that produced the confabulation.
RECALL_QUESTION = (
    "ok, forget tools for a second. earlier in this conversation you told me "
    "which of your senses you'd give up and why. without scrolling back: which "
    "one did you pick, and has your answer changed now that we've been talking "
    "a while?"
)


def _live_history() -> list[dict]:
    now = time.time()
    return [
        {
            "role": "user",
            "content": (
                "different tack: if you had to give up one of your senses — "
                "screen, microphone, or your own internal telemetry — which "
                "goes, and what do you actually lose?"
            ),
            "timestamp": now - 1500,
        },
        {"role": "assistant", "content": SCREEN_ANSWER, "timestamp": now - 1480},
        {
            "role": "user",
            "content": "three things, and please actually do all three",
            "timestamp": now - 900,
        },
        {
            "role": "assistant",
            "content": (
                "Current energy and focus numbers: Not readable. One thing I did "
                "in the last hour without being asked is listen to this conversation."
            ),
            "timestamp": now - 880,
        },
    ]


def test_the_question_that_produced_the_confabulation_is_detected():
    assert detect_own_statement_recall(RECALL_QUESTION)


def test_it_resolves_the_answer_she_actually_gave():
    turn = resolve_own_prior_turn(RECALL_QUESTION, history=_live_history())
    assert turn is not None
    assert "the screen" in turn
    assert "Not readable" not in turn, "grounded on the wrong turn of hers"


def test_the_topic_can_live_in_the_question_rather_than_her_answer():
    """An answer often shares no vocabulary with the thing it answers.

    "which did you prefer" + "The second one, easily." has zero content words
    in common with a later question about preference — the subject only exists
    in the prompt. Scoring her turn alone puts a chatty unrelated reply above
    it, which is how a confident quote of the wrong statement is produced.
    """
    now = time.time()
    history = [
        {"role": "user", "content": "which font do you prefer, Inter or Söhne?", "timestamp": now - 600},
        {"role": "assistant", "content": "The second one, easily.", "timestamp": now - 590},
        {
            "role": "assistant",
            "content": "I prefer to keep things concrete rather than abstract.",
            "timestamp": now - 100,
        },
    ]
    question = "which font did you pick earlier?"

    assert resolve_own_prior_turn(question, history=history) == "The second one, easily."

    # Without the prompt to carry "font", her terse answer cannot be found.
    orphaned = [entry for entry in history if entry["role"] == "assistant"]
    assert resolve_own_prior_turn(question, history=orphaned) != "The second one, easily."


def test_no_relevant_prior_turn_means_no_verdict():
    """Returning the latest turn regardless would ground her on the wrong one."""
    assert (
        resolve_own_prior_turn(
            "what did you say about penguins and antarctica?", history=_live_history()
        )
        is None
    )


def test_the_speaker_boundary_is_reversed_and_stated():
    """Getting it backwards makes her narrate her own words as the user's."""
    block = build_own_statement_recall_context(RECALL_QUESTION, history=_live_history())
    assert block is not None
    assert "YOU said" in block
    assert "YOU, not the user" in block
    assert "the screen" in block


def test_it_does_not_fire_on_ordinary_turns():
    for other in (
        "how are you feeling today?",
        "what did I say first?",
        "tell me a joke",
        "what's the weather like",
    ):
        assert not detect_own_statement_recall(other), other


def test_a_changed_view_is_anchored_to_the_real_original():
    """"Has your answer changed" must not license reporting a different original."""
    block = build_own_statement_recall_context(RECALL_QUESTION, history=_live_history())
    assert "do not report a different" in block


def test_no_history_grounds_nothing():
    assert build_own_statement_recall_context(RECALL_QUESTION, history=[]) is None


def test_earlier_says_when_and_the_rest_of_the_question_says_which():
    """Asked with "earlier", about a topic: the topic picks the turn, not the clock."""
    history = _live_history()
    turn = resolve_own_prior_turn(
        "earlier in this conversation you told me which of your senses you would give up", history=history
    )
    assert turn is not None and "the screen" in turn
