"""Nothing said yet is a fact the runtime holds, not a question for the model.

LIVE 2026-09-07, in a fresh session: "what was the very first thing I said to
you in this conversation?" The positional-recall path resolves the earliest
completed turn — and only when there IS one. With an empty transcript it
returned None, the turn went to the cortex, which had no history to answer
from and emitted seven tokens twice. Both were refused by the text-integrity
check, the desktop contract then refused a lower-lane fallback, and the person
got "I couldn't get my full attention onto that one. Try me again in a moment."

The runtime was holding the answer the whole time. Since 15 November 2026 it
holds it as a reading beside her turn, and she answers from it.
"""

from __future__ import annotations

import pytest

import interface.routes.chat_memory_state as memory_state


@pytest.fixture
def no_exchanges(monkeypatch):
    async def _none(**_kwargs):
        return []

    monkeypatch.setattr(
        memory_state, "_recent_completed_conversation_exchanges", _none
    )


@pytest.fixture
def two_exchanges(monkeypatch):
    async def _two(**_kwargs):
        return [
            {"user": "tell me about yourself", "assistant": "I like mysteries."},
            {"user": "and what do you dislike?", "assistant": "Vagueness."},
        ]

    monkeypatch.setattr(
        memory_state, "_recent_completed_conversation_exchanges", _two
    )


@pytest.mark.asyncio
async def test_a_fresh_session_answers_from_its_own_emptiness(no_exchanges) -> None:
    reply = await memory_state._what_the_conversation_holds_for(
        "what was the very first thing I said to you in this conversation?",
        session_id="session-here",
    )
    assert reply == "This conversation has no completed turn before this one."


@pytest.mark.asyncio
async def test_the_same_for_the_last_thing(no_exchanges) -> None:
    reply = await memory_state._what_the_conversation_holds_for(
        "what was the last thing I said?", session_id="session-here"
    )
    assert reply == "This conversation has no completed turn before this one."


@pytest.mark.asyncio
async def test_a_real_transcript_is_quoted_verbatim(two_exchanges) -> None:
    first = await memory_state._what_the_conversation_holds_for(
        "what did I ask you first?"
    )
    assert first is not None and "tell me about yourself" in first


@pytest.mark.asyncio
async def test_a_populated_last_is_left_to_the_content_path(two_exchanges) -> None:
    """Only the EMPTY case was the gap.

    The first draft took "what did I just ask" over whenever it could, and
    turned a summary of the last two turns into a single quotation. Where
    there are exchanges, the content classifier below still owns it.
    """
    reply = await memory_state._what_the_conversation_holds_for(
        "what did I just ask?"
    )
    assert reply is None or "tell me about yourself" not in reply


@pytest.mark.asyncio
async def test_a_question_that_is_not_positional_is_left_alone(no_exchanges) -> None:
    """This path must not start answering questions that are not about position."""
    assert (
        await memory_state._what_the_conversation_holds_for(
            "what is 2 + 2?", session_id="session-here"
        )
        is None
    )


@pytest.mark.asyncio
async def test_without_a_session_it_cannot_claim_the_conversation_is_empty(
    no_exchanges,
) -> None:
    """With no session, the durable rows arrive through the cross-session scan.

    Refusing that removes recall rather than narrowing it, and claiming the
    conversation is empty over a persistence layer that is holding the turn is
    a guess dressed as a fact.
    """
    reply = await memory_state._what_the_conversation_holds_for(
        "what did I ask you first?"
    )
    assert reply is None or "no completed turn before this one" not in reply


@pytest.mark.asyncio
async def test_a_turn_that_asks_two_things_gets_the_reading_and_answers_both(no_exchanges) -> None:
    """A reading is not an answer, so a compound turn no longer has to be kept from it.

    When the runtime answered from here, "Answer directly in two sentences: what did I just ask you to do, and
    what mind/cognition path are you using right now?" was short-circuited on the recall half. She answers now,
    both halves, with the empty conversation read beside her turn.
    """
    reply = await memory_state._what_the_conversation_holds_for(
        "Answer directly in two sentences: what did I just ask you to do, and "
        "what mind/cognition path are you using right now?",
        session_id="session-here",
    )
    assert reply == "This conversation has no completed turn before this one."


@pytest.mark.asyncio
async def test_this_conversation_means_this_one(monkeypatch) -> None:
    """A question naming this conversation must not reach into another.

    The cross-session reach exists so a restart does not erase yesterday, and
    it is right for "what were we talking about". LIVE 2026-09-07: asked in a
    fresh session what the first thing said was, she quoted a turn from a
    different session entirely — accurately, and about a conversation the
    person had not had here.
    """
    seen: dict[str, object] = {}

    async def _record(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        memory_state, "_recent_completed_conversation_exchanges", _record
    )
    await memory_state._what_the_conversation_holds_for(
        "what did I ask you first?", session_id="session-here"
    )
    assert seen.get("allow_cross_session") is False
    assert seen.get("session_id") == "session-here"
