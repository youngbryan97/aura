from __future__ import annotations

import hashlib

import pytest

from core.conversation.answer_provenance import (
    AnswerProvenance,
    answer_provenance_from_turn,
    answer_provenance_reply,
    asks_for_prior_answer_provenance,
    select_prior_answer_provenance,
)
from core.conversation.session_scope import set_user_question
from core.conversation.surface_disposition import record_tool_receipt
from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    record_turn_grounding,
    record_turn_sensory_evidence,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_provenance_snapshots_only_exact_turn_evidence() -> None:
    answer = "Stanisław Lem."
    with bind_turn_evidence_custody(session_id="session-a", turn_id="turn-a"):
        set_user_question("Who wrote Solaris?")
        assert record_tool_receipt(
            "web_search",
            ok=True,
            action="search",
            object_ref="Solaris author",
        )
        assert record_turn_grounding("authenticated encyclopedia excerpt")
        assert record_turn_sensory_evidence(
            {
                "channel": "screen",
                "ok": True,
                "observation": "The page names Stanisław Lem.",
                "observed_at": 123.0,
            }
        )
        provenance = answer_provenance_from_turn(answer, response_path="protected_foreground")

    assert provenance.answer_sha256 == _digest(answer)
    assert provenance.session_id == "session-a"
    assert provenance.turn_id == "turn-a"
    assert provenance.response_path == "protected_foreground"
    assert provenance.model_native_inference is True
    assert provenance.tool_receipts[0]["tool"] == "web_search"
    assert provenance.tool_receipts[0]["turn_id"] == "turn-a"
    assert provenance.sensory_evidence[0]["channel"] == "screen"
    assert provenance.grounding_digests == (_digest("authenticated encyclopedia excerpt"),)


def test_generic_epistemic_followups_resolve_the_prior_answer() -> None:
    answer = "Stanisław Lem."
    provenance = AnswerProvenance(
        answer_sha256=_digest(answer),
        session_id="session-a",
        turn_id="turn-a",
        captured_at=10.0,
    )
    exchanges = [
        {
            "user": "Who wrote Solaris?",
            "aura": answer,
            "answer_provenance": provenance.to_dict(),
        }
    ]

    for followup in (
        "How'd you know that?",
        "How did you know?",
        "Well then how did you know it?",
        "Where did that answer come from?",
        "What's your source for that?",
        "What made you say that?",
        "Why did you believe that?",
    ):
        assert asks_for_prior_answer_provenance(followup), followup
        assert select_prior_answer_provenance(followup, exchanges) == provenance


def test_unrelated_how_question_does_not_claim_prior_answer_provenance() -> None:
    assert not asks_for_prior_answer_provenance("How does Dijkstra's algorithm work?")
    assert not asks_for_prior_answer_provenance("Where is San Jose?")


def test_model_native_provenance_cannot_invent_a_lookup() -> None:
    provenance = AnswerProvenance(
        answer_sha256=_digest("Stanisław Lem."),
        session_id="session-a",
        turn_id="turn-a",
        captured_at=10.0,
    )

    reply = answer_provenance_reply(provenance)

    assert "learned parameters" in reply
    assert "no tool lookup" in reply
    assert "hidden representation" in reply


def test_provenance_rejects_metadata_bound_to_different_answer() -> None:
    provenance = AnswerProvenance(
        answer_sha256=_digest("Different answer"),
        session_id="session-a",
        turn_id="turn-a",
        captured_at=10.0,
    )
    exchanges = [
        {
            "user": "Who wrote Solaris?",
            "aura": "Stanisław Lem.",
            "answer_provenance": provenance.to_dict(),
        }
    ]

    assert select_prior_answer_provenance("How did you know that?", exchanges) is None


def test_repeated_source_question_walks_through_a_provenance_projection() -> None:
    original = "Stanisław Lem."
    original_provenance = AnswerProvenance(
        answer_sha256=_digest(original),
        session_id="session-a",
        turn_id="turn-a",
        captured_at=10.0,
        response_path="protected_foreground",
    )
    explanation = answer_provenance_reply(original_provenance)
    explanation_provenance = AnswerProvenance(
        answer_sha256=_digest(explanation),
        session_id="session-a",
        turn_id="turn-b",
        captured_at=11.0,
        response_path="verified_answer_provenance",
        grounding_digests=(_digest("the original provenance"),),
        model_native_inference=False,
    )
    exchanges = [
        {
            "user": "Who wrote Solaris?",
            "aura": original,
            "answer_provenance": original_provenance.to_dict(),
        },
        {
            "user": "How did you know that?",
            "aura": explanation,
            "answer_provenance": explanation_provenance.to_dict(),
        },
    ]

    selected = select_prior_answer_provenance(
        "Well then how did you know?",
        exchanges,
    )

    assert selected == original_provenance


def test_lookup_language_still_requires_an_exact_turn_receipt() -> None:
    from core.conversation.response_reliability import _has_unfounded_tool_execution_claim

    assert _has_unfounded_tool_execution_claim("I looked it up when your question came in.")
    assert _has_unfounded_tool_execution_claim("I actually used web search for that.")
    assert not _has_unfounded_tool_execution_claim(
        "I looked it up when your question came in.",
        tool_receipts=({"tool": "web_search", "ok": True},),
    )


def test_unrelated_fact_turn_does_not_receive_historical_action_receipts(monkeypatch) -> None:
    import asyncio

    from core.brain.inference_gate import _attach_the_present_moment

    monkeypatch.setattr("core.brain.present_moment.present_moment_block", lambda: "present")
    monkeypatch.setattr(
        "core.brain.recent_actions.recent_actions_block",
        lambda: "old unrelated search",
    )

    ambient: list[str] = []
    asyncio.run(
        _attach_the_present_moment(
            ambient_grounding_blocks=ambient,
            isolated_generation_contract=False,
            recent_actions_already_grounded=False,
            task_grounding_blocks=[],
            visible_user_prompt="Who wrote Solaris?",
        )
    )

    assert ambient == ["present"]


@pytest.mark.asyncio
async def test_live_projection_reads_bound_prior_answer_without_model_narration(
    monkeypatch,
) -> None:
    from core.conversation.turn_evidence_custody import turn_grounding_evidence
    from interface.routes import chat, chat_memory_state

    answer = "Stanisław Lem."
    provenance = AnswerProvenance(
        answer_sha256=_digest(answer),
        session_id="prior-session",
        turn_id="prior-turn",
        captured_at=10.0,
    )

    async def _recent(**_kwargs):
        return [
            {
                "user": "Who wrote Solaris?",
                "aura": answer,
                "answer_provenance": provenance.to_dict(),
            }
        ]

    monkeypatch.setattr(chat_memory_state, "_recent_completed_conversation_exchanges", _recent)
    with bind_turn_evidence_custody(session_id="current-session", turn_id="current-turn"):
        reply = await chat._resolve_answer_provenance_projection(
            "Well then how did you know that?",
            session_id="current-session",
        )
        grounding = turn_grounding_evidence()

    assert "no tool lookup" in reply
    assert len(grounding) == 1
    assert "aura.answer_provenance.v1" in grounding[0]
