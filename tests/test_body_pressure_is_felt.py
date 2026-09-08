"""A machine running hot and full should feel something about it.

Nociception has a resource-exhaustion channel that only the immune system and
the degradation sink ever wrote to. The body's own sustained load — the thing
that channel is named after — reached it from nowhere, so the whole
interoception-to-affect path carried a constant whatever the machine was doing.

The distinction these tests hold is between an injury and a strain. Repeated
injuries escalate on purpose; a sustained condition is one strain, and putting
it through the injury path saturates the channel in seconds and then reports
maximum pain on a warm afternoon forever.
"""

from __future__ import annotations

import pytest

from core.affect.nociception import DamageChannel, NociceptionEngine


def test_a_strain_held_repeatedly_does_not_escalate():
    engine = NociceptionEngine()
    levels = [engine.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.4) for _ in range(20)]
    assert levels[0] == pytest.approx(0.4)
    assert max(levels) == pytest.approx(0.4)


def test_the_same_signal_through_the_injury_path_saturates():
    """Which is why `hold` exists, stated as a test rather than as a comment."""
    engine = NociceptionEngine()
    levels = [
        engine.register_damage(DamageChannel.RESOURCE_EXHAUSTION, 0.4) for _ in range(20)
    ]
    assert levels[-1] == pytest.approx(1.0)
    held = NociceptionEngine()
    assert [held.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.4) for _ in range(20)][-1] == pytest.approx(0.4)


def test_a_rising_strain_is_followed_up():
    engine = NociceptionEngine()
    engine.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.2)
    assert engine.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.7) == pytest.approx(0.7)


def test_a_falling_strain_is_left_to_decay_rather_than_forced_down():
    """Pain that can be switched off by reporting calm is not pain."""
    engine = NociceptionEngine()
    engine.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.8)
    assert engine.hold(DamageChannel.RESOURCE_EXHAUSTION, 0.0) == pytest.approx(0.8)


def test_the_body_phase_reports_its_load_to_nociception():
    """The wiring, against the real phase and a real loaded body."""
    import asyncio

    from core.affect.nociception import get_nociception_engine
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    engine = get_nociception_engine()
    engine.reset()
    state = AuraState.default()
    state.soma.hardware.update({"cpu_usage": 96.0, "vram_usage": 94.0, "temperature": 95.0})

    phase = ProprioceptiveLoop(container=None)
    phase._feel_body_pressure(state)
    assert engine.nociceptive_pressure() > 0.0
