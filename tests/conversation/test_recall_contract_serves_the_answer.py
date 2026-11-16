"""A question about the conversation is hers to answer; what the transcript holds goes beside her turn.

The route used to compose the answer from the transcript and serve it in her place: when her draft failed a
gate, when her generation wrote nothing, and when her answer shared too few words with the composed one. On
15 November 2026 at 12:19, "What did I tell you about how I slept?" got 'Your last completed message before
this was: "Pick a number ..."': her cortex had answered in 24 characters, and a phrase list had picked the
wrong turn to quote. Bryan, that evening: the transcript goes to her turn as a reading, and she answers.
"""

from __future__ import annotations

import inspect

import pytest


def _route_sources() -> str:
    from interface.routes import chat, chat_memory_state, chat_reply_repair, chat_turn_recall

    return "\n".join(
        inspect.getsource(module) for module in (chat, chat_memory_state, chat_reply_repair, chat_turn_recall)
    )


def test_no_door_serves_the_composed_recall_as_her_reply() -> None:
    source = _route_sources()

    for gone in (
        "serving the transcript-composed recall",
        "Serving the transcript-composed recall",
        "repairing from canonical conversation log",
        "Replacing inadequate conversation recall reply",
        'response_path="conversation_recall_from_transcript"',
        "conversation_recall_log_repair_after_empty_engine",
        "cognitive_engine_recall_bounded_projection",
        "_conversation_recall_reply_is_inadequate",
        "_repair_conversation_recall_if_needed",
        "_build_conversation_recall_reply",
    ):
        assert gone not in source, gone


def test_the_reading_is_given_to_her_turn_beside_his_words() -> None:
    from interface.routes import chat

    source = inspect.getsource(chat)

    assert 'context["conversation_recall_evidence"] = conversation_recall_context' in source
    assert '"[CONVERSATION RECALL EVIDENCE]\\n"' not in source, "the reading went back inside his message"


@pytest.mark.asyncio
async def test_the_reading_speaks_of_the_person_and_of_her_not_to_him() -> None:
    from interface.routes import chat_memory_state as memory_state

    async with memory_state._get_convo_lock():
        memory_state._conversation_log.clear()
        memory_state._conversation_log.extend(
            [
                {"user": "Not bad, slept okay.", "aura": "Glad to hear you slept okay.", "status": "complete"},
                {"user": "Pick a number between 1 and 10.", "aura": "7", "status": "complete"},
            ]
        )

    reading = await memory_state._what_the_conversation_holds_for("What did I tell you about how I slept?")

    assert reading
    assert reading.startswith("The person"), reading
    assert not reading.startswith(("You", "Your", "I ", "My ")), reading
