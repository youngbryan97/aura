"""A question that needs a number, answered without one, is not answered.

Live on the desktop surface 2026-07-26. Asked:

    "What is 17 minus 8, and then times 3? Just the number and one line of working."

Served, as the answer:

    "Not too broad. Some skills serve me better than others.Did you pay
     attention in class? Hey, look at this - ätze! I got chocolate on my shirt."

Every gate passed it. The turn receipt read `surface_quality_gate_enabled:
true`, `surface_quality_gate_passed: true`, `surface_quality_gate_reasons: []`,
`authentic_cognitive_reply: true`, `final_requested_output_contract_satisfied:
true`, and `response_confidence: "high"`. `assess_user_facing_reply` returned
ok=True with no reasons. The whole apparatus checks FORM — brevity, token caps,
mutation chains — and never asks whether the text answers the question.

The deterministic arithmetic verdict could not help here either: it only speaks
when the runtime can compute the expected result, so it is silent on word-form
("17 minus 8") and on anything chained past one operator.

This check needs no expected value. It only asserts the kind of answer is
present, and it fails OPEN on every turn that is not unambiguously a request
for a quantity.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    asks_for_a_number,
    numeric_answer_missing,
)
from tests.chat_lane_support import chat_lane_source

# The reply that shipped, verbatim.
LIVE_GARBAGE = (
    "Not too broad. Some skills serve me better than others.Did you pay "
    "attention in class? Hey, look at this — ätze! I got chocolate on my shirt."
)


def test_the_live_failure_is_caught() -> None:
    question = (
        "What is 17 minus 8, and then times 3? Just the number and one line of working."
    )
    assert numeric_answer_missing(question, LIVE_GARBAGE) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is 17 minus 8?",
        "What's 12 times 4?",
        "How much is 20 percent of 50?",
        "Calculate 144 divided by 12",
        "Compute the sum of 19 and 23",
        "What is 17 minus 8, and then times 3?",
    ],
)
def test_numeric_questions_reject_a_reply_with_no_number(question: str) -> None:
    assert asks_for_a_number(question) is True
    assert numeric_answer_missing(question, LIVE_GARBAGE) is True
    assert numeric_answer_missing(question, "") is True


@pytest.mark.parametrize(
    "reply",
    [
        "17 - 8 = 9, then 9 x 3 = 27. The answer is 27.",
        "27",
        "Twenty-seven.",
        "That works out to twenty seven.",
    ],
)
def test_a_real_answer_passes_in_digits_or_words(reply: str) -> None:
    question = "What is 17 minus 8, and then times 3?"
    assert numeric_answer_missing(question, reply) is False


@pytest.mark.parametrize(
    "question,reply",
    [
        # No quantity requested: this check must stay silent.
        ("How are you feeling right now?", LIVE_GARBAGE),
        ("Tell me about your day.", "It was quiet and steady."),
        ("What is love?", "A hard question, honestly."),
        ("Who wrote Dune?", "Frank Herbert."),
        ("What is the capital of France?", "Paris."),
        # A cue and an operator word, but nothing to operate on.
        ("What is more than you expected?", "Most of it, really."),
        # Numbers with no operator: not an arithmetic request.
        ("What is your favourite of the 3 options we discussed?", "The second one."),
    ],
)
def test_non_numeric_turns_are_never_touched(question: str, reply: str) -> None:
    assert numeric_answer_missing(question, reply) is False


def test_it_fails_open_on_unparseable_input() -> None:
    assert numeric_answer_missing(None, "anything") is False
    assert numeric_answer_missing("", "anything") is False
    assert asks_for_a_number(None) is False


def test_the_cognitive_engine_path_also_consults_it() -> None:
    """The engine path does not leave through _finalize_fastpath.

    Live 2026-07-26, after the fastpath floor was in place: "What is 17 minus 8,
    and then times 3?" was answered with "A quick refresh on classic habits:
    green tea, journaling, and standing by the window to watch the light
    change." — no number, served anyway, because that reply left by the
    cognitive_engine path and never reached the floor.
    """

    chat = chat_lane_source()
    # Imported where the engine path assesses its reply...
    block = chat[chat.index("            is_status_check_turn,") :]
    block = block[: block.index("_is_explicit_capability_inventory_request(visible)")]
    assert "numeric_answer_missing," in block, "the engine path must import the floor"
    assert "if numeric_answer_missing(visible, text):" in block, (
        "the engine path must apply the floor to its own reply"
    )
    # ...and the reply is replaced rather than served.
    assert "I didn't actually work that out" in block


def test_the_serving_gate_consults_it_before_the_arithmetic_verdict() -> None:
    """It must run on the path a reply actually leaves by."""

    # The checks moved into a helper the finalizer calls — token for token,
    # by tools/extract_seam.py — so a slice of the finalizer alone stopped
    # seeing them. The claim is that the serving path runs them, so follow
    # the call.
    chat = chat_lane_source()
    gate = chat[chat.index("async def _finalize_fastpath") :]
    assert "_hold_a_reasoning_answer_to_its_contract(" in gate, (
        "the finalizer no longer reaches the contract holder"
    )
    holder = chat[chat.index("def _hold_a_reasoning_answer_to_its_contract") :]
    gate = holder[: holder.index("requires_reasoning_lane(_semantic_user_message)")]
    assert "numeric_answer_missing(_semantic_user_message, final_text)" in gate
    assert 'status = "numeric_answer_missing"' in gate
    # The honest sentence replaces the text rather than the reply being dropped.
    assert "I didn't actually work that out" in gate
