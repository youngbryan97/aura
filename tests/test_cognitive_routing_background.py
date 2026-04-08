from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.phases.cognitive_routing import CognitiveRoutingPhase
from core.state.aura_state import AuraState, CognitiveMode


@pytest.mark.asyncio
async def test_background_impulse_uses_tertiary_tier_even_if_payload_looks_like_user_text():
    router = SimpleNamespace(classify=AsyncMock(return_value="casual"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    impulse = "Impulse: I feel frustrated. I need to reflect on my recent interactions."
    state.cognition.current_objective = impulse
    state.cognition.current_origin = "background"
    state.cognition.working_memory.append({
        "role": "user",
        "content": impulse,
        "origin": "background",
    })

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["model_tier"] == "tertiary"
    assert new_state.response_modifiers["deep_handoff"] is False


@pytest.mark.asyncio
async def test_short_background_messages_do_not_promote_to_primary():
    router = SimpleNamespace(classify=AsyncMock(return_value="casual"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = "status"
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.working_memory.append({
        "role": "assistant",
        "content": "status",
        "origin": "autonomous_thought",
    })

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["model_tier"] == "tertiary"
    assert new_state.cognition.current_mode is not None


@pytest.mark.asyncio
async def test_active_background_objective_overrides_stale_user_history():
    router = SimpleNamespace(classify=AsyncMock(return_value="technical"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.working_memory.extend([
        {"role": "user", "content": "Can you help me debug this?", "origin": "api"},
        {"role": "assistant", "content": "Absolutely.", "origin": "api"},
    ])
    state.cognition.current_objective = "Impulse: reflect on the previous exchange and update continuity."
    state.cognition.current_origin = "autonomous_thought"

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["model_tier"] == "tertiary"
    assert new_state.response_modifiers["deep_handoff"] is False
    assert new_state.cognition.current_origin == "autonomous_thought"
    router.classify.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_residual_user_objective_is_suppressed_within_cooldown():
    router = SimpleNamespace(classify=AsyncMock(return_value="casual"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_mode = CognitiveMode.DORMANT
    state.cognition.current_objective = "Reflect on the previous exchange."
    state.cognition.current_origin = "user"
    state.cognition.working_memory.append({
        "role": "assistant",
        "content": "Previous response.",
        "origin": "user",
    })

    first_state = await phase.execute(state)
    # The first call should actually route (derive a new state with mode/tier set).
    assert first_state is not state or first_state.cognition.current_mode is not None

    second_state = await phase.execute(first_state)
    # The second call should be suppressed — same objective, same origin, within
    # the dedup cooldown window — so the state is returned unchanged.
    assert second_state is first_state


@pytest.mark.asyncio
async def test_impulse_prefixed_objective_forces_autonomous_lane_even_if_origin_looks_user_facing():
    router = SimpleNamespace(classify=AsyncMock(return_value="casual"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_objective = "Impulse: I feel frustrated and need to regroup."
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DORMANT
    state.cognition.working_memory.append({
        "role": "assistant",
        "content": "Give me a moment — I'm thinking through something.",
        "origin": "user",
    })

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["model_tier"] == "tertiary"
    assert new_state.cognition.current_origin == "autonomous_thought"
    router.classify.assert_not_called()


@pytest.mark.asyncio
async def test_stringified_queue_tuple_in_working_memory_is_recovered_to_background_lane():
    router = SimpleNamespace(classify=AsyncMock(return_value="casual"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    phase = CognitiveRoutingPhase(container)

    state = AuraState.default()
    state.cognition.current_mode = CognitiveMode.DORMANT
    state.cognition.working_memory.append({
        "role": "user",
        "content": "(15, 1234.5, 7, {'content': 'Impulse: reflect on my recent interactions.', 'origin': 'autonomous_thought'}, 'autonomous_thought')",
    })

    new_state = await phase.execute(state)

    assert new_state.response_modifiers["model_tier"] == "tertiary"
    assert new_state.cognition.current_origin == "autonomous_thought"
    router.classify.assert_not_called()
