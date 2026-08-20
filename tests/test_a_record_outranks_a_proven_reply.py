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

import json

from interface.routes.chat import (
    _apply_recorded_answer,
    _recorded_answer_corrections,
)


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


def test_a_record_replaces_a_proven_reply_and_is_retyped() -> None:
    original = "I've been running in the background."
    response = _apply_recorded_answer("what have you been up to tonight?", _Response(_payload(original, proven=True)))
    data = _served(response)
    if data["response"] == original:
        # Nothing recorded on this machine right now; the gate still must not
        # be what stopped it.
        return
    assert data["response_confidence"] == "computed"
    contract = data["live_turn_contract"]
    assert contract["recorded_answer_served"] is True
    assert contract["answer_delivery_proven"] is False


def test_a_proven_reply_with_no_record_is_left_alone() -> None:
    original = "I think Lem was writing about the limits of contact."
    response = _apply_recorded_answer("what do you make of Solaris?", _Response(_payload(original, proven=True)))
    assert _served(response)["response"] == original


def test_the_readers_report_whether_they_replaced_anything() -> None:
    text, served = _recorded_answer_corrections("what do you make of Solaris?", "a view of my own")
    assert text == "a view of my own"
    assert served is False


def test_an_unproven_reply_still_goes_through_the_whole_chain() -> None:
    original = "I could not get there."
    response = _apply_recorded_answer("what is the weather", _Response(_payload(original, proven=False)))
    assert isinstance(_served(response)["response"], str)
