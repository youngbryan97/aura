"""A question about her runtime is answered by her, from what she already holds.

Until 16 November 2026 a phrase list (`_is_runtime_fact_status_request`) picked out questions such as "which
lane is active?" and the route answered them with a fixed card: "I am speaking through the launched desktop UI
into /api/chat, through CognitiveEngine, with ... as the active foreground lane. CognitiveEngine handled this
turn: yes. Governed tools available: yes, ...". The card was served without her on one path, replaced her reply
on seven more, and rode inside his message on the rest. A question the list missed got nothing, and a question it
caught got the card whatever she wrote.

Her live mind context carries the lane, the subsystems and tool governance on every turn, so she can answer
any such question in her own words. Nothing is routed by phrase.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

_ASKED = "Which lane is active right now, and are your governed tools available?"


def test_her_live_mind_context_carries_what_the_card_used_to_say() -> None:
    from interface.routes.chat_desktop_mode import _build_live_mind_context_payload

    payload = _build_live_mind_context_payload(
        user_message=_ASKED,
        lane={
            "desired_model": "Aura-Qwen3.8-27B",
            "foreground_endpoint": "Cortex",
            "state": "ready",
            "conversation_ready": True,
        },
        require_engine=True,
    )

    assert payload["lane"]["desired_model"] == "Aura-Qwen3.8-27B"
    assert payload["lane"]["foreground_endpoint"] == "Cortex"
    assert payload["lane"]["conversation_ready"] is True
    assert "cognitive_engine" in payload["required_subsystems"]
    assert "tool_governance_available" in payload["governance"]


def test_a_status_question_rides_into_her_turn_as_his_words_alone() -> None:
    from interface.routes.chat_reply_shaping import _compose_the_engine_message

    composed = _compose_the_engine_message(
        context={},
        context_challenge_context="",
        engine_user_message=_ASKED,
        require_engine=True,
        state_native_output_owner=False,
        visible=_ASKED,
    )

    assert composed == _ASKED


def test_a_status_question_is_given_room_by_its_shape_not_by_a_phrase_list() -> None:
    from interface.routes.chat_desktop_evidence import _desktop_live_reply_token_budget

    budget = _desktop_live_reply_token_budget(
        _ASKED,
        capability_inventory_contract=False,
        bounded_planning_contract=False,
        memory_state_contract=False,
    )

    assert budget > 384


@pytest.mark.parametrize(
    "name",
    [
        "_ground_runtime_fact_status_reply",
        "_build_runtime_fact_status_fastpath_reply",
        "_build_runtime_status_continuity_repair_reply",
    ],
)
def test_no_route_module_keeps_a_card_to_serve(name: str) -> None:
    import interface.routes.chat as chat_routes
    import interface.routes.chat_desktop_repair as desktop_repair
    import interface.routes.chat_reply_shaping as shaping

    for module in (chat_routes, desktop_repair, shaping):
        assert not hasattr(module, name)
