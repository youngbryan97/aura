from types import SimpleNamespace

import pytest

from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.state.aura_state import AuraState, CognitiveMode


def test_user_facing_work_defaults_to_primary():
    assert CognitiveRoutingPhase._resolve_model_tier(True) == "primary"
    assert CognitiveRoutingPhase._resolve_model_tier(False) == "tertiary"


def test_prefixed_user_origin_stays_foreground():
    assert CognitiveRoutingPhase._is_user_facing_origin("routing_user") is True
    assert CognitiveRoutingPhase._is_user_facing_origin("routing_voice_command") is True
    assert CognitiveRoutingPhase._normalize_origin("routing_user") == "user"


def test_deep_handoff_is_enabled_for_explicit_heavy_reasoning():
    assert CognitiveRoutingPhase._should_allow_deep_handoff(
        "Do a flagship architecture deep dive and root cause analysis of this system",
        is_user_facing=True,
        intent_type="TASK",
    ) is True


def test_creative_foreground_stays_on_32b_without_forcing_72b():
    assert CognitiveRoutingPhase._should_allow_deep_handoff(
        "Write something creative and vivid about the ocean at night",
        is_user_facing=True,
        intent_type="CHAT",
    ) is False


def test_background_work_never_requests_deep_handoff():
    assert CognitiveRoutingPhase._should_allow_deep_handoff(
        "Do a flagship architecture deep dive and root cause analysis of this system",
        is_user_facing=False,
        intent_type="TASK",
    ) is False


@pytest.mark.asyncio
async def test_everyday_chat_fast_path_stays_reactive_on_primary():
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()

    new_state = await phase.execute(state, objective="With me?", priority=True)

    assert new_state.cognition.current_mode == CognitiveMode.REACTIVE
    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert new_state.response_modifiers["model_tier"] == "primary"
    assert new_state.response_modifiers["deep_handoff"] is False


@pytest.mark.asyncio
async def test_explicit_search_request_routes_to_skill_before_everyday_chat(monkeypatch):
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()

    capability_engine = SimpleNamespace(detect_intent=lambda _text: ["web_search"])
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: capability_engine if name == "capability_engine" else default),
    )

    new_state = await phase.execute(
        state,
        objective="Search the web for Beautiful Mind by Ciscero and Oddisee",
        priority=True,
    )

    assert new_state.cognition.current_mode == CognitiveMode.REACTIVE
    assert new_state.response_modifiers["intent_type"] == "SKILL"
    assert new_state.response_modifiers["matched_skills"] == ["web_search"]
    assert new_state.response_modifiers["model_tier"] == "primary"


@pytest.mark.asyncio
async def test_specific_fact_lookup_forces_grounded_search_even_without_pattern_match(monkeypatch):
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    new_state = await phase.execute(
        state,
        objective='Tell me who the author of "I Worked at a Top Secret Government Research Lab. I Need to Share My Journals" is.',
        priority=True,
    )

    assert new_state.response_modifiers["intent_type"] == "SKILL"
    assert new_state.response_modifiers["matched_skills"] == ["web_search"]
    assert new_state.response_modifiers["response_contract"]["requires_search"] is True


@pytest.mark.asyncio
async def test_state_reflection_routes_to_deliberate_with_affective_pressure():
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()
    state.affect.arousal = 0.82
    state.affect.valence = -0.12
    state.affect.emotions.update({
        "joy": 0.32,
        "trust": 0.28,
        "fear": 0.31,
        "sadness": 0.29,
    })

    new_state = await phase.execute(
        state,
        objective="How do I know you're an actual present mind and what do you feel right now?",
        priority=True,
    )

    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE
    assert new_state.response_modifiers["intent_type"] == "CHAT"
    assert new_state.response_modifiers["model_tier"] == "primary"
    assert new_state.response_modifiers["deep_handoff"] is True
