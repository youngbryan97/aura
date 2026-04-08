from types import SimpleNamespace

import pytest

from core.phases.response_generation import ResponseGenerationPhase
from core.state.aura_state import AuraState, CognitiveMode


class _Container:
    def __init__(self, services):
        self.services = services

    def get(self, name, default=None):
        return self.services.get(name, default)


class _Router:
    def __init__(self):
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return "Thermal-safe response."


@pytest.mark.asyncio
async def test_response_generation_downshifts_on_thermal_pressure(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Perform a deep architectural audit"
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["model_tier"] = "secondary"
    state.response_modifiers["deep_handoff"] = True
    state.soma.hardware["temperature"] = 86.0
    state.soma.hardware["cpu_usage"] = 63.0

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(state)

    assert router.calls
    call = router.calls[0]
    assert call["prefer_tier"] == "tertiary"
    assert call["deep_handoff"] is False
    assert call["max_tokens"] < 6144
    assert new_state.response_modifiers["thermal_guard"] is True
    # Downstream voice shaping may add punctuation/styling, so verify content
    # presence rather than exact equality.
    assert "Thermal-safe response" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_identity_refresh_when_runtime_is_not_idle(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\nSummarize recent continuity."
    state.cognition.current_origin = "system"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_noise_objective(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Task exception: database is locked while background cognitive state retries."
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_treats_prefixed_user_origin_as_foreground(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Hello Aura."
    state.cognition.current_origin = "routing_user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    def _unexpected_background_gate(*_args, **_kwargs):
        raise AssertionError("foreground origins should not consult background gating")

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _unexpected_background_gate,
    )

    result = await phase.execute(state)

    # The router must have been called as a foreground request.
    # We don't assert the exact response text because downstream voice shaping
    # (SubstrateVoiceEngine) may legitimately restyle it — but the routing
    # decision (foreground vs background) is what this test validates.
    assert router.calls, "Router should have been called for a user-facing origin"
    assert router.calls[0]["is_background"] is False
    assert result.cognition.last_response, "A response should have been generated"
