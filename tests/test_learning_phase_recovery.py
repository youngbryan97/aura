from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.phases.learning_phase import LearningPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


class _BrokenCalibrator:
    def record_prediction(self, *, confidence: float, actual_correctness: float) -> None:
        raise RuntimeError("calibrator offline")


def _state_with_response() -> AuraState:
    state = AuraState()
    state.cognition.last_response = "This response is long enough to trigger learning."
    state.cognition.working_memory = [{"role": "user", "content": "that's wrong"}]
    return state


async def _passthrough(state: AuraState, objective: str) -> AuraState:
    return state


async def _noop(state: AuraState, objective: str) -> None:
    return None


@pytest.fixture(autouse=True)
def isolated_learning_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()


def test_learning_phase_marks_calibration_failure_in_state():
    ServiceContainer.register_instance("metacognitive_calibrator", _BrokenCalibrator())
    phase = LearningPhase(SimpleNamespace())
    phase._perform_standard_learning = _passthrough
    phase._map_cross_domain = _passthrough
    phase._wire_conversation_learning = _noop

    state = _state_with_response()
    result = asyncio.run(phase.execute(state, objective="explain it again"))

    assert result is state
    marker = state.response_modifiers["learning_phase"]
    assert marker["status"] == "degraded"
    assert marker["failures"][-1]["stage"] == "metacognitive_calibration"
    assert state.health["learning_phase"]["stage"] == "metacognitive_calibration"
    assert get_degradation_tracker().recent(subsystem="learning_phase")[-1].action


def test_learning_phase_preserves_state_when_standard_learning_fails():
    phase = LearningPhase(SimpleNamespace())

    async def fail_standard(state: AuraState, objective: str) -> AuraState:
        raise RuntimeError("learner database locked")

    phase._perform_standard_learning = fail_standard
    phase._map_cross_domain = _passthrough
    phase._wire_conversation_learning = _noop
    state = _state_with_response()

    result = asyncio.run(phase.execute(state, objective="remember this"))

    assert result is state
    assert state.response_modifiers["learning_phase_degraded"] is True
    assert (
        state.response_modifiers["learning_phase"]["failures"][-1]["stage"] == "standard_learning"
    )


def test_cross_domain_mapping_fails_closed_when_background_gate_errors(monkeypatch):
    phase = LearningPhase(SimpleNamespace())
    state = _state_with_response()
    state.affect.curiosity = 0.95

    def broken_background_gate(*args, **kwargs):
        raise RuntimeError("failure pressure unavailable")

    monkeypatch.setattr(
        "core.phases.learning_phase.background_activity_allowed",
        broken_background_gate,
    )

    result = asyncio.run(phase._map_cross_domain(state, objective="connect these ideas"))

    assert result is state
    assert state.cognition.pending_intents == []
    assert (
        state.response_modifiers["learning_phase"]["failures"][-1]["stage"]
        == "cross_domain_background_gate"
    )
