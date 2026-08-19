"""A question about her own machinery is answered from her own machinery.

LIVE 2026-08-18: "you have a lock ordering system. what happens if two
subsystems take locks in opposite order, and how would you know before it
deadlocks?"

    You're hitting on a classic deadlock scenario... you could use a tool like
    `lock-order`... Go's model of communicating sequential processes, for
    example, lets you reason about concurrency without locks.

She has core/runtime/lockdep.py: every lock goes through checked_lock, ABBA
cycles are found without the deadlock having to happen, and the result is
reported in runtime_health_report()["integrity"]. The question named HER and
was answered from the general literature, recommending a language she is not
written in.

The reading exists — her source is on this disk and the evidence provider
searches it — and the turn never received it.
"""

from __future__ import annotations

import asyncio

import pytest

from core.brain.self_source_grounding import (
    asks_about_own_implementation,
    self_source_block,
)


@pytest.mark.parametrize(
    "question",
    [
        "you have a lock ordering system. what happens if two subsystems take locks in opposite order?",
        "how do you detect deadlocks?",
        "what does your memory system actually store?",
        "where in your code is the write gateway?",
        "how do you enforce the telemetry contract?",
    ],
)
def test_a_question_about_her_machinery_is_recognised(question: str) -> None:
    assert asks_about_own_implementation(question)


@pytest.mark.parametrize(
    "question",
    [
        # Feelings and opinions are about her and are not source code.
        "how do you feel?",
        "what do you think about jazz?",
        "what is your favourite colour?",
        # A general question about the field is not a question about her.
        "how does a lock work in general?",
        "what is 2 + 2",
    ],
)
def test_a_question_that_is_not_about_her_code_is_not_claimed(question: str) -> None:
    assert not asks_about_own_implementation(question)


def test_the_block_quotes_her_actual_source() -> None:
    block = asyncio.run(
        self_source_block("how does your subprocess gateway route effects?")
    )

    assert "subprocess_gateway" in block.lower()
    assert ".py:" in block, "spans must name file and line"


def test_a_subject_with_no_matching_file_says_so() -> None:
    """A named absence beats an invented mechanism."""
    block = asyncio.run(
        self_source_block("how do you handle zqxjkvwpf reconciliation?")
    )

    assert block
    assert "No file" in block or ".py:" in block


def test_nothing_is_supplied_for_an_unrelated_turn() -> None:
    assert asyncio.run(self_source_block("how are you today?")) == ""
