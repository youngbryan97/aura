"""How long we have been talking is arithmetic, not recollection.

LIVE 2026-08-18, asked "how long have we been talking?":

    About an hour. Time flies when we're geeking out on distributed systems
    and mycelial networks, doesn't it? What's next? More Solaris analysis?

None of that had happened. The conversation was minutes old and had been about
a clipboard, a file, and a prime number.

Nothing reached the turn. The transcript observable answers "what did I ask
you first"; "how long", "how many" and "what about" are different questions,
so they arrived with no reading and the model supplied a number and some
topics, warmly. Every one of those facts is on the transcript, which
timestamps each entry.
"""

from __future__ import annotations

import time

import pytest

from core.conversation.conversation_shape import (
    asks_about_conversation_shape,
    conversation_shape_block,
)


@pytest.fixture
def transcript():
    """A transcript holding exactly this conversation.

    The transcript is a singleton, so without clearing it each test inherits
    the entries the previous one added and the counts drift — the same
    order-dependence the runner reports separately.
    """
    from core.conversation.unified_transcript import UnifiedTranscript

    instance = UnifiedTranscript.get_instance()
    preserved = list(instance._entries)
    instance._entries.clear()
    try:
        yield _populate(instance)
    finally:
        instance._entries[:] = preserved


def _populate(instance):
    instance.add_text_input("what is a semiconductor?")
    instance.add_text_output("A semiconductor is a material...")
    instance.add_text_input("what did I just copy?")
    instance.add_text_output("ORION-7 checkpoint alpha")
    entries = instance.entries_for_conversation()
    entries[0].timestamp = time.time() - 912  # 15.2 minutes
    return instance


@pytest.mark.parametrize(
    "question",
    [
        "how long have we been talking?",
        "what have we talked about so far?",
        "how many messages have I sent you?",
        "what did we cover earlier?",
        "how long has this conversation been going?",
    ],
)
def test_a_question_about_the_conversation_is_recognised(question: str) -> None:
    assert asks_about_conversation_shape(question)


@pytest.mark.parametrize(
    "question",
    [
        # Uptime is not conversation length.
        "how long have you been running?",
        "how long will it take to build?",
        "what is 2 + 2",
        "how are you doing",
    ],
)
def test_a_different_question_is_not_claimed(question: str) -> None:
    assert not asks_about_conversation_shape(question)


def test_the_duration_is_measured_from_the_transcript(transcript) -> None:
    block = conversation_shape_block("how long have we been talking?")

    assert "15 minutes ago" in block


def test_the_count_is_counted(transcript) -> None:
    block = conversation_shape_block("how many messages have I sent you?")

    assert "2 message(s) from the person" in block


def test_the_topics_are_the_ones_that_happened(transcript) -> None:
    block = conversation_shape_block("what have we talked about so far?")

    assert "what is a semiconductor?" in block
    assert "what did I just copy?" in block


def test_an_absent_transcript_is_named_not_invented(monkeypatch) -> None:
    import core.conversation.conversation_shape as module

    monkeypatch.setattr(module, "_entries", list)

    block = conversation_shape_block("how long have we been talking?")

    assert "No transcript is available" in block


def test_the_reading_reaches_the_grounding_channel(transcript) -> None:
    import asyncio

    import core.brain.observable_registry  # noqa: F401
    from core.brain.observable_grounding import observable_blocks

    blocks = asyncio.run(observable_blocks("how long have we been talking?"))
    text = "\n".join(blocks) if isinstance(blocks, list) else str(blocks)

    assert "THE SHAPE OF THIS CONVERSATION" in text
