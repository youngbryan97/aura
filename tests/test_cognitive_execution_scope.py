from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.kernel.upgrades_10x import GodModeToolPhase
from core.phases.cognitive_routing import CognitiveRoutingPhase as LegacyRoutingPhase
from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.runtime.cognitive_execution_scope import (
    CognitiveExecutionScope,
    bind_cognitive_execution_scope,
    bound_cognitive_execution_scope,
    cognitive_request_allows_actions,
    resolve_cognitive_execution_scope,
)
from core.state.aura_state import AuraState, CognitiveMode

_JOURNAL_OBJECTIVE = (
    "Write an internal journal record from the supplied episode data and preserve "
    "the factual changes."
)


def test_internal_generation_is_reasoning_only_without_reading_prompt_words() -> None:
    assert resolve_cognitive_execution_scope(
        origin="system",
        context={"mode": "introspection", "tier": "journal"},
    ) is CognitiveExecutionScope.REASONING_ONLY


def test_user_and_governed_autonomous_requests_retain_action_eligibility() -> None:
    assert resolve_cognitive_execution_scope(
        origin="user",
        context={},
    ) is CognitiveExecutionScope.GOVERNED_ACTIONS
    assert resolve_cognitive_execution_scope(
        origin="overt_action_loop",
        context={
            "autonomous": True,
            "authorization": "governed_autonomous_overt_action",
            "requested_authority_scope": "overt_action_loop:action-1:web_search",
        },
    ) is CognitiveExecutionScope.GOVERNED_ACTIONS


def test_unknown_explicit_scope_fails_closed() -> None:
    assert resolve_cognitive_execution_scope(
        origin="user",
        context={"cognitive_execution_scope": "anything_goes"},
    ) is CognitiveExecutionScope.REASONING_ONLY


def test_scope_is_bound_to_one_objective_not_left_as_global_state() -> None:
    state = AuraState.default()
    bind_cognitive_execution_scope(
        state,
        _JOURNAL_OBJECTIVE,
        CognitiveExecutionScope.REASONING_ONLY,
        source="cognitive_engine:system",
    )

    assert bound_cognitive_execution_scope(
        state, _JOURNAL_OBJECTIVE
    ) is CognitiveExecutionScope.REASONING_ONLY
    assert not cognitive_request_allows_actions(state, _JOURNAL_OBJECTIVE)
    assert bound_cognitive_execution_scope(state, "Search for a new paper") is None
    assert cognitive_request_allows_actions(state, "Search for a new paper")


@pytest.mark.asyncio
async def test_unitary_routing_never_skill_matches_reasoning_only_generation() -> None:
    phase = CognitiveRoutingPhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "system"
    state.response_modifiers["matched_skills"] = ["desktop_task"]
    bind_cognitive_execution_scope(
        state,
        _JOURNAL_OBJECTIVE,
        CognitiveExecutionScope.REASONING_ONLY,
        source="cognitive_engine:system",
    )

    routed = await phase.execute(state, objective=_JOURNAL_OBJECTIVE)

    assert routed.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in routed.response_modifiers
    assert routed.cognition.current_mode is CognitiveMode.DELIBERATE


@pytest.mark.asyncio
async def test_legacy_routing_never_skill_matches_reasoning_only_generation() -> None:
    phase = LegacyRoutingPhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "system"
    state.cognition.current_objective = _JOURNAL_OBJECTIVE
    state.response_modifiers["matched_skills"] = ["desktop_task"]
    bind_cognitive_execution_scope(
        state,
        _JOURNAL_OBJECTIVE,
        CognitiveExecutionScope.REASONING_ONLY,
        source="cognitive_engine:system",
    )

    routed = await phase.execute(state, objective=_JOURNAL_OBJECTIVE)

    assert routed.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in routed.response_modifiers
    assert routed.cognition.current_mode is CognitiveMode.DELIBERATE


@pytest.mark.asyncio
async def test_godmode_backstop_blocks_both_skill_and_task_dispatch(monkeypatch) -> None:
    phase = GodModeToolPhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "system"
    state.response_modifiers.update(
        {"intent_type": "TASK", "matched_skills": ["desktop_task"]}
    )
    bind_cognitive_execution_scope(
        state,
        _JOURNAL_OBJECTIVE,
        CognitiveExecutionScope.REASONING_ONLY,
        source="cognitive_engine:system",
    )
    monkeypatch.setattr(
        phase,
        "_get_cap_engine",
        lambda: pytest.fail("reasoning-only request reached capability lookup"),
    )

    result = await phase.execute(state, objective=_JOURNAL_OBJECTIVE)

    assert result.response_modifiers["intent_type"] == "CHAT"
    assert "matched_skills" not in result.response_modifiers
    assert result.response_modifiers["execution_suppressed"]["reason"] == (
        "cognitive_request_reasoning_only"
    )


@pytest.mark.asyncio
async def test_godmode_does_not_block_a_different_autonomous_objective() -> None:
    phase = GodModeToolPhase(SimpleNamespace())
    state = AuraState.default()
    state.cognition.current_origin = "autonomous"
    state.response_modifiers.update(
        {"intent_type": "SKILL", "matched_skills": ["web_search"]}
    )
    bind_cognitive_execution_scope(
        state,
        _JOURNAL_OBJECTIVE,
        CognitiveExecutionScope.REASONING_ONLY,
        source="cognitive_engine:system",
    )

    result = await phase.execute(state, objective="Search for a new paper")

    assert result.response_modifiers["intent_type"] == "SKILL"
    assert "execution_suppressed" not in result.response_modifiers


@pytest.mark.asyncio
async def test_an_objective_bound_reasoning_only_stays_so_when_another_tick_runs_it() -> None:
    """LIVE 15-16 November 2026: her mind tick ran the dream journal's prompt, left as her current objective, through a
    kernel tick whose state carried no binding, and her skill phase dispatched build_document for a dream."""
    dream = "[SYSTEM ROLE: THE SUBCONSCIOUS]\nSynthesize these fragments into a single metaphor."
    bind_cognitive_execution_scope(
        AuraState.default(), dream, CognitiveExecutionScope.REASONING_ONLY, source="cognitive_engine:dream_journal",
    )
    rerun = AuraState.default()
    rerun.cognition.current_origin = "dream_journal"
    rerun.response_modifiers.update({"intent_type": "SKILL", "matched_skills": ["build_document"]})

    result = await GodModeToolPhase(SimpleNamespace()).execute(rerun, objective=dream)

    assert result.response_modifiers["intent_type"] == "CHAT"
    assert result.response_modifiers["execution_suppressed"]["reason"] == "cognitive_request_reasoning_only"


def test_a_later_action_binding_of_the_same_words_is_not_held_back() -> None:
    words = "Search for the paper on ternary weights"
    bind_cognitive_execution_scope(AuraState.default(), words, CognitiveExecutionScope.REASONING_ONLY, source="a")
    bind_cognitive_execution_scope(AuraState.default(), words, CognitiveExecutionScope.GOVERNED_ACTIONS, source="b")
    assert cognitive_request_allows_actions(AuraState.default(), words)
