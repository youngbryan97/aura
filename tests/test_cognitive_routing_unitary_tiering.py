from types import SimpleNamespace

import pytest

from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
from core.runtime.turn_analysis import analyze_turn
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


def test_complex_coding_debug_turn_enables_deep_handoff():
    analysis = analyze_turn(
        "Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py."
    )
    route_meta = CognitiveRoutingPhase._build_coding_route_metadata(
        "Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py.",
        analysis=analysis,
        intent_type="TASK",
    )

    assert route_meta["coding_request"] is True
    assert route_meta["coding_complexity_score"] >= 0.65
    assert CognitiveRoutingPhase._should_allow_deep_handoff(
        "Debug the failing pytest in core/runtime/conversation_support.py and core/orchestrator/mixins/tool_execution.py.",
        is_user_facing=True,
        intent_type="TASK",
        analysis=analysis,
        route_meta=route_meta,
    ) is True


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


@pytest.mark.asyncio
async def test_followup_coding_turn_stays_on_engineering_lane(monkeypatch):
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()
    state.cognition.current_origin = "user"

    monkeypatch.setattr(
        "core.runtime.coding_session_memory.get_coding_route_hints",
        lambda _objective="": {
            "coding_request": False,
            "active_coding_thread": True,
            "recent_file_count": 3,
            "recent_command_count": 2,
            "has_test_failure": True,
            "has_runtime_error": False,
            "last_objective": "Fix the failing pytest in core/runtime/conversation_support.py",
        },
    )

    new_state = await phase.execute(state, objective="Let's keep it going")

    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE
    assert new_state.response_modifiers["coding_request"] is True
    assert new_state.response_modifiers["coding_route_hints"]["active_coding_thread"] is True


@pytest.mark.asyncio
async def test_short_lets_do_it_followup_stays_on_engineering_lane(monkeypatch):
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()
    state.cognition.current_origin = "user"

    monkeypatch.setattr(
        "core.runtime.coding_session_memory.get_coding_route_hints",
        lambda _objective="": {
            "coding_request": False,
            "active_coding_thread": True,
            "recent_file_count": 2,
            "recent_command_count": 1,
            "has_test_failure": True,
            "has_runtime_error": False,
            "last_objective": "Fix the failing pytest in core/runtime/conversation_support.py",
        },
    )

    new_state = await phase.execute(state, objective="Let's do it")

    assert new_state.cognition.current_mode == CognitiveMode.DELIBERATE
    assert new_state.response_modifiers["coding_request"] is True
    assert new_state.response_modifiers["coding_route_hints"]["active_coding_thread"] is True


@pytest.mark.asyncio
async def test_coding_route_hints_include_execution_loop_state(monkeypatch):
    kernel = SimpleNamespace(orchestrator=SimpleNamespace(cycle_count=100))
    phase = CognitiveRoutingPhase(kernel)
    state = AuraState.default()
    state.cognition.current_origin = "user"

    monkeypatch.setattr(
        "core.runtime.coding_session_memory.get_coding_route_hints",
        lambda _objective="": {
            "coding_request": True,
            "active_coding_thread": True,
            "recent_file_count": 2,
            "recent_command_count": 1,
            "has_test_failure": True,
            "has_runtime_error": False,
            "has_active_plan": True,
            "has_verification_failure": True,
            "repair_attempts": 2,
            "execution_phase": "repairing",
        },
    )

    new_state = await phase.execute(
        state,
        objective="Keep going on the failing pytest in core/runtime/conversation_support.py",
    )

    route_hints = new_state.response_modifiers["coding_route_hints"]
    assert route_hints["has_active_plan"] is True
    assert route_hints["has_verification_failure"] is True
    assert route_hints["repair_attempts"] == 2
    assert route_hints["execution_phase"] == "repairing"
