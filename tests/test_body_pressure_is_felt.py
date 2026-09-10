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


def test_the_body_phase_pushes_a_frame_into_the_substrate():
    """`inject_perceptual_frame` had no caller: the only thing in the tree with
    that name is a different class in the language layer. Perception and the
    body reached recurrent cognition through nothing at all."""
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.runtime.service_registry import register_runtime_service
    from core.state.aura_state import AuraState

    frames: list[dict] = []

    class _Substrate:
        def inject_perceptual_frame(self, frame_data):
            frames.append(dict(frame_data))

    from core.state.percepts import emit_percept

    register_runtime_service("liquid_substrate", _Substrate(), required=False)
    state = AuraState.default()
    state.soma.hardware.update({"cpu_usage": 71.0, "ram_usage": 64.0, "temperature": 80.0})
    state.affect.valence = -0.4
    # In the shape percepts arrive in. The first version of this test wrote a
    # `source` key, which the frame asked for and no producer in the tree has
    # ever written — so the screen channel read zero however much she was
    # looking at, and presence was a bare count where one percept and twenty
    # read the same.
    emit_percept(state.world, "vision", content="a window moved", intensity=0.8)

    ProprioceptiveLoop(container=None)._push_perceptual_frame(state)

    assert frames, "the substrate was not given a frame"
    frame = frames[-1]
    assert frame["cpu_percent"] == pytest.approx(71.0)
    assert frame["memory_percent"] == pytest.approx(64.0)
    assert frame["thermal"] == pytest.approx(0.8)
    assert frame["valence"] == pytest.approx(-0.4)
    assert frame["user_presence"] == pytest.approx(0.8)
    assert frame["screen_changed"] == pytest.approx(0.8)


def test_the_frame_grades_what_arrived_rather_than_counting_it():
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.runtime.service_registry import register_runtime_service
    from core.state.aura_state import AuraState
    from core.state.percepts import emit_percept

    frames: list[dict] = []

    class _Substrate:
        def inject_perceptual_frame(self, frame_data):
            frames.append(dict(frame_data))

    register_runtime_service("liquid_substrate", _Substrate(), required=False)
    state = AuraState.default()
    emit_percept(state.world, "interaction", content="a message", intensity=0.9)
    emit_percept(state.world, "resource_pressure", content="under load", intensity=0.6)
    ProprioceptiveLoop(container=None)._push_perceptual_frame(state)
    frame = frames[-1]
    assert frame["social"] == pytest.approx(0.9)
    assert frame["threat"] == pytest.approx(0.6)
    assert frame["screen_changed"] == 0.0


def test_a_channel_that_reported_nothing_is_left_out_of_the_frame():
    """The substrate treats an absent key as zero, which is the honest answer."""
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.runtime.service_registry import register_runtime_service
    from core.state.aura_state import AuraState

    frames: list[dict] = []

    class _Substrate:
        def inject_perceptual_frame(self, frame_data):
            frames.append(dict(frame_data))

    register_runtime_service("liquid_substrate", _Substrate(), required=False)
    ProprioceptiveLoop(container=None)._push_perceptual_frame(AuraState.default())
    assert "novelty" not in frames[-1]


def test_the_substrate_reaches_affect_and_not_only_the_modifiers():
    """`HomeostaticCoupling` says the substrate is the ground truth for felt
    state and blends it at thirty percent — into a local dictionary used to
    pick cognitive modifiers, never into the affect state. The push existed and
    the return did not."""
    import asyncio

    from core.consciousness.homeostatic_coupling import SUBSTRATE_SHARE
    from core.container import ServiceContainer
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    class _Substrate:
        def update(self, **_kwargs):
            return None

        def get_state_summary_nowait(self):
            return {"valence": -0.9, "arousal": 0.9, "snapshot_stale": False}

    ServiceContainer.register_instance("liquid_substrate", _Substrate())
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    state.affect.valence = 0.0
    state.affect.arousal = 0.0

    asyncio.run(AffectUpdatePhase(None).execute(state))

    assert state.affect.valence < 0.0
    assert state.affect.arousal > 0.0
    assert state.response_modifiers.get("substrate_share_of_affect") == SUBSTRATE_SHARE


def test_a_stale_substrate_snapshot_is_not_blended():
    """A felt state built from a reading older than the last thing that
    happened is worse than one built without it."""
    import asyncio

    from core.container import ServiceContainer
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    class _Stale:
        def update(self, **_kwargs):
            return None

        def get_state_summary_nowait(self):
            return {"valence": -0.9, "arousal": 0.9, "snapshot_stale": True}

    ServiceContainer.register_instance("liquid_substrate", _Stale())
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    asyncio.run(AffectUpdatePhase(None).execute(state))
    assert "substrate_share_of_affect" not in state.response_modifiers


def test_the_lifetime_blend_survives_the_rest_of_the_affect_phase():
    """It ran at the top of the phase and the derived-affect step three steps
    later recomputed curiosity from the emotions dictionary, overwriting it.
    Displacing the developmental state far enough to take novelty from 0.60 to
    1.00 moved curiosity by five ten-thousandths."""
    import asyncio
    import inspect

    from core.phases.affect_update import AffectUpdatePhase

    source = inspect.getsource(AffectUpdatePhase.execute)
    blend = source.index("_advance_lifetime")
    derived = source.index("_update_resonance")
    assert blend > derived, "the lifetime blend must run after affect is settled"

    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "hello"})
    asyncio.run(AffectUpdatePhase(None).execute(state))
    assert "ontogenetic_novelty" in state.response_modifiers
