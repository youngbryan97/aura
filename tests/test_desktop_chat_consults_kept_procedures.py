"""Her desktop chat is answered before any phase runs, so it consults her kept procedures itself.

Live 2026-10-16: the five requests of the G09 live proof were answered by
CognitiveEngine._direct_desktop_quick_reply, no phase ran, and a greatest
common divisor her book keeps a procedure for went unconsulted.
"""

from __future__ import annotations

import pytest


def _planted(form: str) -> dict:
    return {
        "objective": "What's the greatest common divisor of 4862 and 7854?",
        "answer": "374",
        "family": "dm_numbers__gcd",
        "form": form,
        "agreed_known": 5000,
        "confidence": 1.0,
        "receipt": {},
        "generation_metadata": {},
        "summary": "computed by the procedure she keeps for problems of the kind 'dm_numbers__gcd'",
    }


async def _turn(monkeypatch, *, planted, draft: str):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.phases import kept_procedures_in_a_turn

    calls: list[dict] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return draft

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    async def kept_answer_for(objective, *, router, origin, tier):
        assert router is not None
        return planted

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", kept_answer_for)
    request = "What's the greatest common divisor of 4862 and 7854?"
    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        request,
        ThinkingMode.FAST,
        "user",
        {"desktop_quick_reply_contract": True, "visible_user_message": request, "max_tokens": 512},
        timeout_s=30.0,
    )
    shown = "\n".join(str(message.get("content") or "") for message in calls[-1]["messages"])
    return thought, shown


@pytest.mark.asyncio
async def test_the_kept_answer_is_in_front_of_her_before_she_drafts(monkeypatch):
    thought, shown = await _turn(monkeypatch, planted=_planted("restated"), draft="It comes out to 374.")

    assert "[SKILL RESULT: kept_procedure]" in shown and "Answer: 374" in shown
    assert thought.content == "It comes out to 374."
    assert thought.metadata["kept_procedure"]["family"] == "dm_numbers__gcd"


@pytest.mark.asyncio
async def test_a_restated_answer_is_evidence_and_her_words_stand(monkeypatch):
    thought, _shown = await _turn(monkeypatch, planted=_planted("restated"), draft="I make it 22.")

    assert thought.content == "I make it 22."


@pytest.mark.asyncio
async def test_an_answer_to_the_request_as_worded_is_served_when_her_draft_does_not_state_it(monkeypatch):
    thought, _shown = await _turn(monkeypatch, planted=_planted("words"), draft="I make it 22.")

    assert thought.content == "374"


@pytest.mark.asyncio
async def test_no_kept_answer_leaves_the_turn_as_it_was(monkeypatch):
    thought, shown = await _turn(monkeypatch, planted=None, draft="It comes out to 374.")

    assert "kept_procedure" not in shown
    assert thought.content == "It comes out to 374."
    assert thought.metadata["kept_procedure"] is None
