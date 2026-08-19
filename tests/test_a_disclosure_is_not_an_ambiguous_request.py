"""Telling her something about yourself is not asking her an unclear question.

LIVE 2026-08-18: "I've been working on this project for two years and I still
don't know if it's real or if I'm fooling myself."

    How does it feel when you're working on it? Is there a difference in the
    quality of your attention between 'real' and 'fooling yourself'?

Two questions and nothing else. The turn matched the inquiry trigger "don't
know if", which exists for an ambiguous REQUEST — "should I", "help me decide",
"not sure" — where asking what someone means is the helpful move. A person
describing their own doubt has been perfectly clear, and asking them to specify
is the opposite of engaging with it.

The distinction is grammatical, not a phrase list: a first-person statement
about themselves, with no request in it, is a disclosure.
"""

from __future__ import annotations

import pytest

from core.cognition.cognitive_kernel import CognitiveKernel


@pytest.fixture
def kernel() -> CognitiveKernel:
    return object.__new__(CognitiveKernel)


@pytest.mark.parametrize(
    "message",
    [
        "I've been working on this project for two years and I still don't know if it's real or if I'm fooling myself.",
        "I'm not sure this is working.",
        "I don't know if I can keep doing this.",
        "I've felt off all week and I can't tell why.",
    ],
)
def test_a_disclosure_is_not_answered_with_a_clarifying_question(
    kernel: CognitiveKernel, message: str
) -> None:
    assert not CognitiveKernel._should_inquire(kernel, message, [], "simple")


@pytest.mark.parametrize(
    "message",
    [
        "should i use postgres or sqlite?",
        "help me decide between these two.",
        "what would you do about this?",
        "tell me about this",
    ],
)
def test_an_ambiguous_request_still_asks(
    kernel: CognitiveKernel, message: str
) -> None:
    assert CognitiveKernel._should_inquire(kernel, message, [], "simple")


def test_a_disclosure_that_also_asks_is_a_request(kernel: CognitiveKernel) -> None:
    """"I don't know if it's real — what would you do?" wants an answer."""
    assert CognitiveKernel._should_inquire(
        kernel, "I don't know if it's real. what would you do?", [], "simple"
    )
