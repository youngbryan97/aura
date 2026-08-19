"""A complaint that a question went unanswered names no subject.

LIVE 2026-08-19. The turn before was "if you had to pick between being useful
and being understood, which would you choose?", answered. Then "you didn't
answer my question" got "I'm sorry, I got distracted. You asked about my
neurochemistry." Nothing about neurochemistry had been said.

The complaint refers to a specific earlier turn and contains none of its
words, so no reading matched it and the reply guessed. An apology attached to
the wrong question is the worst available outcome: it sounds settled.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.unanswered_question import (
    complains_the_question_went_unanswered,
    last_exchange,
    unanswered_question_block,
)


@pytest.mark.parametrize(
    "message",
    [
        "you didn't answer my question",
        "you did not answer my question",
        "you never answered my question",
        "that's not what I asked",
        "that wasn't my question",
        "you dodged the question",
        "you ignored my question",
        "answer my question please",
        "you changed the subject",
    ],
)
def test_the_complaint_is_recognised_however_it_is_put(message: str):
    assert complains_the_question_went_unanswered(message)


@pytest.mark.parametrize(
    "message",
    [
        # A question ABOUT the transcript, owned by the transcript reading.
        "what did I ask you two messages ago?",
        "how are you doing",
        "what is 2 + 2",
        # Says what it wants; nothing to look up.
        "I asked you to reverse a string",
        "can you answer in one sentence",
    ],
)
def test_an_ordinary_turn_is_not_a_complaint(message: str):
    assert not complains_the_question_went_unanswered(message)


def test_the_reading_quotes_the_real_question(monkeypatch):
    import core.conversation.unanswered_question as module

    monkeypatch.setattr(
        module,
        "_pairs",
        lambda: [
            ("what's your favourite colour?", "Blue, if I have to pick one."),
            (
                "if you had to pick between being useful and being understood,"
                " which would you choose?",
                "Being understood.",
            ),
        ],
    )

    block = unanswered_question_block("you didn't answer my question")

    assert "being useful and being understood" in block
    assert "Being understood." in block
    # The one thing it must never do is name a subject nobody raised.
    assert "neurochemistry" not in block.lower()


def test_the_complaint_itself_is_never_offered_as_the_open_question(monkeypatch):
    import core.conversation.unanswered_question as module

    monkeypatch.setattr(
        module,
        "_pairs",
        lambda: [
            ("what is your favourite colour?", "Blue."),
            ("you didn't answer my question", "Sorry — which one?"),
        ],
    )

    exchange = last_exchange(before="you didn't answer my question")

    assert exchange is not None
    assert exchange.question == "what is your favourite colour?"


def test_no_record_means_asking_rather_than_naming(monkeypatch):
    import core.conversation.unanswered_question as module

    monkeypatch.setattr(module, "_pairs", list)

    block = unanswered_question_block("you didn't answer my question")

    assert "Ask which one they mean" in block


def test_the_reading_reaches_the_turn_through_the_registry(monkeypatch):
    from core.brain.observable_registry import _matches_unanswered, _read_unanswered

    assert _matches_unanswered("you didn't answer my question")
    assert not _matches_unanswered("how are you doing")

    import core.conversation.unanswered_question as module

    monkeypatch.setattr(
        module, "_pairs", lambda: [("what is 47 times 89?", "4183.")]
    )
    served = asyncio.run(_read_unanswered("that's not what I asked"))

    assert "47 times 89" in served
