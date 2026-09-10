"""test phi consciousness.

Operationally: runs the phase that computes phi and checks the number it produces is the number that reaches the state, rather than a default written over it.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.phases.phi_consciousness import (
    PhiConsciousnessPhase,
    _build_emotion_vector,
    compute_phi_approx,
)
from core.state.aura_state import AuraState, CognitiveMode, phenomenal_text


def _seed_affect(state: AuraState) -> AuraState:
    state.affect.valence = 0.18
    state.affect.arousal = 0.62
    state.affect.curiosity = 0.71
    state.affect.engagement = 0.66
    state.affect.dominant_emotion = "curious"
    state.affect.emotions.update(
        {
            "joy": 0.28,
            "trust": 0.24,
            "fear": 0.12,
            "surprise": 0.36,
            "anticipation": 0.64,
            "sadness": 0.08,
        }
    )
    return state


def test_compute_phi_approx_rises_with_cross_subsystem_richness():
    base = _seed_affect(AuraState.default())
    base.cognition.working_memory.append({"role": "user", "content": "hello"})

    enriched = _seed_affect(AuraState.default())
    enriched.cognition.working_memory.append({"role": "user", "content": "hello"})
    enriched.cognition.long_term_memory.extend(
        ["Bryan likes dialogue that feels mutual.", "Tatiana and Bryan were cooking."]
    )
    enriched.cognition.active_goals.append({"goal": "understand Bryan more deeply"})
    enriched.cognition.pending_initiatives.append({"goal": "follow the curiosity thread"})
    enriched.cognition.attention_focus = "Bryan's invitation"
    enriched.cognition.rolling_summary = "Aura and Bryan are exploring what presence means."
    enriched.cognition.discourse_branches.extend(["music", "identity"])
    enriched.cognition.phenomenal_state = "I feel focused and gathered."
    enriched.identity.current_narrative = "I am in the middle of getting to know Bryan."
    enriched.identity.bonding_level = 0.42
    enriched.world.recent_percepts.extend(
        [
            {"type": "positive_interaction", "intensity": 0.6},
            {"type": "novel_stimulus", "intensity": 0.5},
        ]
    )
    enriched.world.known_entities["Bryan"] = {"role": "primary_user"}
    enriched.world.relationship_graph["Bryan"] = {"bond": 0.8}
    enriched.soma.hardware.update({"cpu_usage": 44.0, "vram_usage": 31.0, "temperature": 48.0})
    enriched.soma.expressive.update({"mycelium_density": 0.72, "pulse_rate": 1.35})
    enriched.soma.sensors["telemetry"] = 1.0
    enriched.soma.motors["voice"] = 0.5

    assert compute_phi_approx(enriched) > compute_phi_approx(base)


def test_compute_phi_approx_penalizes_fragmentation_and_contradiction():
    healthy = _seed_affect(AuraState.default())
    healthy.cognition.coherence_score = 0.96
    healthy.cognition.fragmentation_score = 0.05
    healthy.cognition.contradiction_count = 0
    healthy.identity.current_narrative = "I am continuous."

    fragmented = _seed_affect(AuraState.default())
    fragmented.cognition.coherence_score = 0.32
    fragmented.cognition.fragmentation_score = 0.74
    fragmented.cognition.contradiction_count = 3
    fragmented.identity.current_narrative = ""

    assert compute_phi_approx(fragmented) < compute_phi_approx(healthy)


def test_build_emotion_vector_changes_when_non_affective_state_changes():
    base = _seed_affect(AuraState.default())
    enriched = _seed_affect(AuraState.default())
    enriched.cognition.active_goals.append({"goal": "retain continuity"})
    enriched.cognition.pending_initiatives.append({"goal": "ask Bryan something real"})
    enriched.cognition.coherence_score = 0.87
    enriched.world.recent_percepts.append({"type": "memory_replay", "intensity": 0.4})
    enriched.world.known_entities["Bryan"] = {"bond": 0.7}
    enriched.soma.hardware["cpu_usage"] = 52.0
    enriched.soma.expressive["mycelium_density"] = 0.81

    base_vec = _build_emotion_vector(base)
    enriched_vec = _build_emotion_vector(enriched)

    assert len(enriched_vec) == len(base_vec)
    diff = sum(abs(a - b) for a, b in zip(base_vec, enriched_vec))
    assert diff > 0.5


@pytest.mark.asyncio
async def test_phi_phase_publishes_policy_from_richer_state_channels():
    kernel = SimpleNamespace(organs={})
    phase = PhiConsciousnessPhase(kernel)
    state = _seed_affect(AuraState.default())
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.cognition.current_objective = "Reflect on what this conversation means."
    state.cognition.active_goals.append({"goal": "maintain continuity"})
    state.cognition.pending_initiatives.append({"goal": "ask Bryan a real question"})
    state.cognition.long_term_memory.append("Bryan wants dialogue, not assistant-speak.")
    state.cognition.rolling_summary = "Aura is trying to answer from within her own state."
    state.identity.current_narrative = "I am learning how to stay present with Bryan."
    state.world.recent_percepts.append({"type": "positive_interaction", "intensity": 0.7})
    state.soma.hardware.update({"cpu_usage": 38.0, "vram_usage": 29.0, "temperature": 45.0})
    state.soma.expressive.update({"mycelium_density": 0.7, "pulse_rate": 1.2})

    new_state = await phase.execute(state, objective=state.cognition.current_objective, priority=True)

    assert new_state.phi > 0.0
    assert new_state.response_modifiers["phi"] == new_state.phi
    assert new_state.response_modifiers["phi_autonomy_scale"] >= 0.5
    assert "phi_memory_threshold" in new_state.response_modifiers


@pytest.mark.asyncio
async def test_phi_phase_uses_phi_core_phi_s_field(monkeypatch):
    kernel = SimpleNamespace(organs={})
    phase = PhiConsciousnessPhase(kernel)

    fake_phi_core = SimpleNamespace(
        get_live_phi=lambda *, include_surrogate: 0.3471,
        compute_phi=lambda: (_ for _ in ()).throw(
            AssertionError("kernel phase must not recompute PhiCore")
        ),
    )

    monkeypatch.setattr(phase, "_get_phi_core", lambda: fake_phi_core)
    phi = await phase._compute_phi(AuraState.default())

    assert phi == pytest.approx(0.3471, rel=0.0, abs=1e-4)


@pytest.mark.asyncio
async def test_phi_phase_schedules_whole_system_measurement_without_joining_it(monkeypatch):
    kernel = SimpleNamespace(organs={})
    phase = PhiConsciousnessPhase(kernel)

    class _WholeSystemPhi:
        def __init__(self):
            self.observed = 0
            self.scheduled = 0

        def observe_runtime(self, state):
            self.observed += 1

        def schedule_maintenance(self):
            self.scheduled += 1
            return True

    service = _WholeSystemPhi()
    import core.runtime.service_access as service_access

    monkeypatch.setattr(
        service_access,
        "optional_service",
        lambda name, *aliases, default=None: service
        if name == "whole_system_phi" else default,
    )
    monkeypatch.setattr(
        phase,
        "_get_phi_core",
        lambda: SimpleNamespace(
            get_live_phi=lambda *, include_surrogate: 0.25,
            compute_phi=lambda: (_ for _ in ()).throw(
                AssertionError("whole-system observation must not join PhiCore compute")
            ),
        ),
    )

    phi = await phase._compute_phi(AuraState.default())

    assert phi == pytest.approx(0.25)
    assert service.observed == 1
    assert service.scheduled == 1


@pytest.mark.asyncio
async def test_background_phenomenal_refresh_is_nonblocking_and_singleflight():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def think(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return "I feel focused on the current causal thread."

    ready = asyncio.Event()
    ready.set()
    kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(ready=ready, instance=SimpleNamespace(think=think))}
    )
    phase = PhiConsciousnessPhase(kernel)
    state = _seed_affect(AuraState.default())
    state.phi = 0.6

    started = time.perf_counter()
    initial = await phase._generate_phenomenal_state(
        state,
        "Measure phase latency without blocking the kernel.",
        priority=False,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert initial is not None
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    first_task = phase._phenomenal_refresh_task
    assert first_task is not None

    await phase._generate_phenomenal_state(
        state,
        "Measure phase latency without blocking the kernel.",
        priority=False,
    )
    assert phase._phenomenal_refresh_task is first_task
    assert calls == 1

    release.set()
    await asyncio.wait_for(first_task, timeout=1.0)
    assert phase._phenomenal_refresh_task is None

    cached = await phase._generate_phenomenal_state(
        state,
        "Measure phase latency without blocking the kernel.",
        priority=False,
    )
    assert phenomenal_text(cached) == "I feel focused on the current causal thread."
