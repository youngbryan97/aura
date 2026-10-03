"""A reply that was served is a delivered turn, whatever its status names.

LIVE 2026-10-02: every reply the smaller model wrote that evening appeared in
grey, centred, as a system notice. The ladder's rescue leaves through an exit
whose status is "desktop_cognitive_engine_unavailable"; the delivery journal
marked any status containing "unavailable" as a failed turn, and the page draws
a failed turn's text as a diagnostic, not as her reply.
"""

from __future__ import annotations

import pytest

from core.runtime.chat_delivery_journal import DeliveryState
from interface.routes.chat_delivery import _chat_delivery_state_for_response

RESCUED = {"status": "desktop_cognitive_engine_unavailable", "response": "Kobe's 81."}


@pytest.mark.parametrize("confidence", ["fallback", "computed"])
def test_a_served_answer_is_delivered(confidence: str) -> None:
    assert _chat_delivery_state_for_response({**RESCUED, "response_confidence": confidence}, 200) is (
        DeliveryState.COMPLETED
    )


def test_a_refusal_is_still_a_failed_turn() -> None:
    assert _chat_delivery_state_for_response({**RESCUED, "response_confidence": "failed"}, 200) is DeliveryState.FAILED
    assert _chat_delivery_state_for_response({"status": "guard_blocked", "response": "No."}, 200) is DeliveryState.FAILED


def test_a_strict_benchmark_status_code_still_fails() -> None:
    assert _chat_delivery_state_for_response({**RESCUED, "response_confidence": "fallback"}, 503) is DeliveryState.FAILED
