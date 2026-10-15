"""A kept procedure's answer is put in front of her before she drafts; her own words stand when they state it."""

from __future__ import annotations

import types

import pytest

from core.phases.response_generation import ResponseGenerationPhase, _states_the_answer
from core.state.aura_state import AuraState


def _kept(answer: str):
    from core.brain.reasoning_amplifier_v2 import AmplifiedAnswer, ReasoningReceipt

    operation = {"family": "bbeh_dyck_languages", "answer": answer, "agreed_known": 180, "confidence": 0.9945}
    receipt = ReasoningReceipt(mode="kept_procedure", strategy_used="kept_procedure", task_type="kept_procedure",
                               num_candidates=0, verifiers_run=[], valid_candidates=0, winning_candidate_id=None,
                               confidence=0.9945, agreement=0.0, epistemic_status="kept_procedure",
                               promotion_authority="kept_procedure", cognitive_operations=[operation])
    return AmplifiedAnswer(answer=answer, confidence=0.9945, verified=False, calibrated=False, receipt=receipt,
                           generation_metadata={"response_path": "kept_procedure"})


def test_her_closing_words_state_the_answer_or_they_do_not() -> None:
    assert _states_the_answer("Walking the stack through, it closes at step 18.\n\nThe answer is: 18", "18")
    assert _states_the_answer("I make it option B.", "(B)")
    assert _states_the_answer("So the answer is: (b)", "(B)")
    assert not _states_the_answer("I think it is a trick, and the answer is (C).", "(B)")
    assert not _states_the_answer("It closes at 18.\n\nBut on reflection the answer is 17.", "18")
    assert not _states_the_answer("", "18")


@pytest.mark.asyncio
async def test_the_kept_answer_reaches_her_working_memory_as_a_skill_result(monkeypatch) -> None:
    from core.utils.injected_blocks import is_stamped_grounding

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures",
                        lambda objective, *a, **k: _kept("18") if "Dyck" in objective else None)
    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    before = len(state.cognition.working_memory)
    assert not await phase._consult_kept_procedures(state=state, objective="What's a good name for a cat?")
    assert len(state.cognition.working_memory) == before and "kept_procedure" not in state.response_modifiers
    assert await phase._consult_kept_procedures(state=state, objective="A Dyck sequence ...")
    block = state.cognition.working_memory[-1]
    assert is_stamped_grounding(block) and block["metadata"]["skill"] == "kept_procedure"
    assert "Answer: 18" in block["content"] and "all 180 answers" in block["content"]
    assert state.response_modifiers["kept_procedure"]["answer"] == "18"


@pytest.mark.asyncio
async def test_her_reply_stands_when_it_states_the_kept_answer_and_the_answer_is_served_when_it_does_not(
        monkeypatch) -> None:
    async def no_search(*_a, **_k):
        raise AssertionError("a kept answer was in front of her; no search runs")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", no_search)
    common = {"objective": "A Dyck sequence ...", "router": None, "request_timeout": 20.0, "origin": "desktop",
              "tier": "primary", "runtime_context": {}, "is_user_facing": True, "is_background": False,
              "proof_or_benchmark": False}
    phase = ResponseGenerationPhase(types.SimpleNamespace())
    for draft, served, voice in (
        ("I traced every bracket; it closes at step 18.\n\nThe answer is: 18", None, "her_reply_states_it"),
        ("I traced it and it closes at 17.\n\nThe answer is: 17", "18", "computed_answer_served"),
    ):
        state = AuraState.default()
        state.response_modifiers["kept_procedure"] = {"objective": common["objective"], "answer": "18",
                                                      "confidence": 0.99,
                                                      "receipt": {"strategy_used": "kept_procedure"},
                                                      "generation_metadata": {"response_path": "kept_procedure"}}
        out = await phase._maybe_amplify_response(draft=draft, state=state, **common)
        assert str(out) == (served or draft)
        assert state.response_modifiers["reasoning_amplifier_v2_active_phase"]["voice"] == voice


@pytest.mark.asyncio
async def test_the_kept_answer_survives_to_what_her_model_is_given(monkeypatch) -> None:
    from core.brain.what_she_is_given import what_she_is_given
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures",
                        lambda objective, *a, **k: _kept("18"))
    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    assert await phase._consult_kept_procedures(state=state, objective="A Dyck sequence ...")
    block = state.cognition.working_memory[-1]
    turn = [{"role": "system", "content": "You are someone."}, block,
            {"role": "user", "content": "A Dyck sequence ..."}]
    given, system_prompt = what_she_is_given(turn, "", markers=())
    evidence = [m for m in given if m.get("role") == RUNTIME_EVIDENCE_ROLE]
    assert system_prompt == "" and len(evidence) == 1 and "Answer: 18" in evidence[0]["content"]
    assert all(m.get("role") != "system" for m in given)
    assert given.index(evidence[0]) == len(given) - 2 and given[-1]["role"] == "user"
