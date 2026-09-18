"""A turn that answers carries 131 keys. A turn that refuses carried none.

Measured live 2026-09-18. Asked "Which single file under core/consciousness
is the largest by bytes?", the runtime produced three drafts — 62, 83 and
77 tokens — discarded all of them, and served the canned line. The response
came back as:

    status='canonical_chat_no_reply'  confidence='failed'
    contract_keys=0                   delivery='failed'

against the answering turn's 131 contract keys. Nothing in either log sink
named why any draft went, and the one string that did say something —
"implicit_legacy_orchestrator_fallback_refused" — sat in an output receipt
the caller never sees.

The receipt existed for the case that did not need it and vanished for the
case that did.
"""

from __future__ import annotations

from interface.routes.chat_refusals import refusal_receipt

LANE = {
    "state": "failed",
    "last_failure_reason": "retryable_error_and_nothing_served",
}


def test_a_refusal_says_that_it_is_one():
    receipt = refusal_receipt(
        status="canonical_chat_no_reply", reason="drafts_exhausted", lane=LANE
    )
    assert receipt["refused"] is True
    assert receipt["status"] == "canonical_chat_no_reply"


def test_the_reason_reaches_the_caller():
    """It was in an output receipt nobody downstream reads."""

    receipt = refusal_receipt(
        status="canonical_chat_no_reply",
        reason="implicit_legacy_orchestrator_fallback_refused",
        lane=LANE,
    )
    assert receipt["refusal_reason"] == "implicit_legacy_orchestrator_fallback_refused"


def test_the_lane_failure_is_carried_not_summarised_away():
    receipt = refusal_receipt(status="x", reason="y", lane=LANE)
    assert receipt["lane_state"] == "failed"
    assert receipt["lane_last_failure_reason"] == "retryable_error_and_nothing_served"


def test_a_ladder_answer_is_distinguishable_from_a_canned_one():
    """Two very different turns that looked identical from outside."""

    canned = refusal_receipt(status="x", reason="y", lane=LANE)
    laddered = refusal_receipt(
        status="x", reason="y", lane=LANE, served_by_ladder=True
    )
    assert canned["fallback_ladder_answered"] is False
    assert laddered["fallback_ladder_answered"] is True


def test_a_missing_lane_does_not_raise():
    """A refusal must never fail while explaining itself."""

    for lane in (None, {}, "not a dict", 7):
        receipt = refusal_receipt(status="x", reason="y", lane=lane)
        assert receipt["lane_state"] == ""
        assert receipt["refused"] is True


def test_the_question_is_measured_not_quoted():
    """Length, not content: a receipt is not a place to copy the person."""

    receipt = refusal_receipt(
        status="x", reason="y", lane=LANE, question="how big is the largest file"
    )
    assert receipt["question_chars"] == 27
    assert "largest" not in str(receipt)


def test_both_refusal_responses_carry_it():
    import inspect

    from interface.routes import chat_refusals

    source = inspect.getsource(chat_refusals)
    assert source.count('"live_turn_contract": refusal_receipt(') == 2, (
        "a refusal path that returns without a receipt is the defect this fixes"
    )
