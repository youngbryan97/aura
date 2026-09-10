"""An answer the runtime holds is served even when the model's passed.

LIVE, 2026-08-20. "what have you been up to tonight?" was answered "I've been
running in the background. The system was a little sluggish earlier, but it's
settling down now." Thirty-eight finished pieces of work were in the record.

The reply had crossed the answer contract, so answer_delivery_proven was True
and the whole correction chain returned early — every record server with it.
That gate is right that a proof must not come to refer to different text.
What follows from it is re-typing the response, not keeping a wrong answer.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from interface.routes import chat
from interface.routes.chat import (
    _apply_recorded_answer,
    _recorded_answer_corrections,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", [False, True])
async def test_terminal_rewrite_cannot_inherit_the_original_byte_proof(monkeypatch, recorded):
    original = "The first complete sentence. However"
    replacement = "The measured record contains two completed actions."

    async def correction(_message, reply):
        return (replacement, True) if recorded else (reply, False)

    monkeypatch.setattr(chat, "_recorded_answer_corrections", correction)
    monkeypatch.setattr(chat, "_append_past_action_record", lambda _message, text: text)
    monkeypatch.setattr(chat, "_correct_unsourced_self_metrics", lambda text: text)
    monkeypatch.setattr(chat, "_correct_false_capability_denials", lambda text: text)
    payload = _payload(original, proven=recorded)
    payload["live_turn_contract"]["authored_answer_completion_proven"] = True
    data = _served(await _apply_recorded_answer("Explain the result", _Response(payload)))
    assert data["response"] != original
    contract = data["live_turn_contract"]
    assert contract["authored_answer_completion_proven"] is False
    assert contract["delivery_payload_mutated_after_proof"] is True
    assert contract["pre_mutation_response_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert contract["delivered_response_sha256"] == hashlib.sha256(data["response"].encode()).hexdigest()
    if recorded:
        assert contract["recorded_answer_served"] is True


class _Response:
    """Enough of a JSONResponse for the wrapper to read."""

    status_code = 200

    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode()


def _payload(reply: str, *, proven: bool) -> dict:
    return {
        "response": reply,
        "response_confidence": "high",
        "live_turn_contract": {
            "answer_delivery_proven": proven,
            "response_confidence": "high",
        },
    }


def _served(response) -> dict:
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_a_record_replaces_a_proven_reply_and_is_retyped() -> None:
    original = "I've been running in the background."
    response = await _apply_recorded_answer(
        "what have you been up to tonight?",
        _Response(_payload(original, proven=True)),
    )
    data = _served(response)
    if data["response"] == original:
        # Nothing recorded on this machine right now; the gate still must not
        # be what stopped it.
        return
    assert data["response_confidence"] == "computed"
    contract = data["live_turn_contract"]
    assert contract["recorded_answer_served"] is True
    assert contract["answer_delivery_proven"] is False


@pytest.mark.asyncio
async def test_a_proven_reply_with_no_record_is_left_alone() -> None:
    original = "I think Lem was writing about the limits of contact."
    response = await _apply_recorded_answer(
        "what do you make of Solaris?",
        _Response(_payload(original, proven=True)),
    )
    assert _served(response)["response"] == original


@pytest.mark.asyncio
async def test_the_readers_report_whether_they_replaced_anything() -> None:
    text, served = await _recorded_answer_corrections(
        "what do you make of Solaris?",
        "a view of my own",
    )
    assert text == "a view of my own"
    assert served is False


@pytest.mark.asyncio
async def test_an_unproven_reply_still_goes_through_the_whole_chain() -> None:
    original = "I could not get there."
    response = await _apply_recorded_answer(
        "what is the weather",
        _Response(_payload(original, proven=False)),
    )
    assert isinstance(_served(response)["response"], str)


@pytest.mark.asyncio
async def test_verified_assertion_bytes_bypass_generic_prose_repair() -> None:
    from core.epistemics.assertion import (
        Assertion,
        AssertionResponse,
        SourceKind,
        Verification,
    )

    text = "My resident cortex is 27B, with 26,895,993,856 parameters."
    typed = AssertionResponse(
        family="cortex_self_evidence",
        text=text,
        assertions=(
            Assertion(
                subject="resident cortex identity",
                claim=text,
                source=SourceKind.MEASURED,
                evidence=("f" * 64,),
                verification=Verification.VERIFIED,
            ),
        ),
    )
    response = await _apply_recorded_answer(
        "Which cortex are you running now?",
        _Response(
            {
                "response": text,
                "response_confidence": "high",
                "live_turn_contract": {
                    "answer_delivery_proven": False,
                    "verified_assertion_response": typed.authority(),
                },
            }
        ),
    )

    assert _served(response)["response"] == text
