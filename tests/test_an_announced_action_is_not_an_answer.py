"""Saying it is about to act, and then stopping, does not end the turn.

LIVE, 2026-08-20. Given a six-person seating problem and a Python sandbox,
the tool loop's generation was exactly:

    "Let's break down the problem step by step and use code to help us figure
    out the seating arrangement."

and the turn ended there. The loop read it as a final answer, so the turn was
decided by a sentence describing work nobody did, and the reply that followed
got half the problem wrong.
"""

from __future__ import annotations

import inspect

import pytest

from core.brain.llm.mlx_client import (
    MLXLocalClient,
    _announces_an_action_it_did_not_take as announces,
)


def test_the_live_sentence() -> None:
    assert announces(
        "Let's break down the problem step by step and use code to help us "
        "figure out the seating arrangement."
    )


@pytest.mark.parametrize(
    "text",
    [
        "I'll run a quick script to enumerate the seatings.",
        "Let me use the sandbox to check that.",
        "I will write a small program to brute-force it.",
        "We can use python to settle this.",
    ],
)
def test_other_ways_of_saying_it(text: str) -> None:
    assert announces(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Let's use the first constraint to place Boris.",
        "Let's start with Ada and then place Boris opposite her.",
        "Opposite Chen is Emil, and Dara's neighbours are Ada and Chen.",
        "The code I wrote earlier returned 17.",
        "Let's think about what the code in that repository is doing.",
        "",
    ],
)
def test_an_answer_is_not_an_announcement(text: str) -> None:
    assert announces(text) is False


def test_it_needs_both_the_intent_and_the_means() -> None:
    """Narrow on purpose: an announcement names what it is about to use."""
    assert announces("Let me check.") is False
    assert announces("Let me check with the sandbox.") is True


def test_the_loop_continues_at_most_once() -> None:
    source = inspect.getsource(MLXLocalClient.think_and_act)
    assert "announced_without_acting = True" in source
    assert "not announced_without_acting" in source
    assert "turn + 1 < max_turns" in source
