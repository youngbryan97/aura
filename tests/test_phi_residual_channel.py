"""Grassmann states cross the process boundary, or Φ has no activation grounding.

The blocker this closes, stated exactly. ``_maybe_record_phi_residual`` resolved
PhiCore with ``ServiceContainer.has("phi_core")`` — an in-process lookup — while
generation runs in the MLX worker subprocess (``ctx.Process(target=
_mlx_worker_loop, ...)`` in mlx_client.py). The hook and the thing it feeds have
never shared a process, so the lookup was False on every token and the
activation-grounded complex read, on a boot with three hooks installed and seven
real generations:

    residual_stream_grassmann (insufficient_history:0/50 grassmann transitions)

Zero. Not "not enough yet" — none, ever, which is why no activation-grounded
live Φ has existed.

The fix mirrors ``_substrate_mem``, the shared array already passed
parent→worker for steering, in reverse. It is cheap because the Grassmann
encoder does the expensive part in the worker: ~5120 floats in, one byte out.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from core.consciousness.phi_residual_channel import (
    RING_SLOTS,
    create_channel,
    drain,
    publish_state,
)


@pytest.fixture
def channel():
    return create_channel(mp.get_context("spawn"))


class TestTheRingCarriesStates:
    def test_what_is_published_is_drained(self, channel):
        for value in (0, 7, 42, 255):
            publish_state(channel, value)
        states, cursor = drain(channel, 0)
        assert states == [0, 7, 42, 255]
        assert cursor == 4

    def test_draining_twice_yields_nothing_the_second_time(self, channel):
        publish_state(channel, 5)
        _, cursor = drain(channel, 0)
        assert drain(channel, cursor) == ([], cursor)

    def test_new_states_arrive_incrementally(self, channel):
        publish_state(channel, 1)
        _, cursor = drain(channel, 0)
        publish_state(channel, 2)
        states, _ = drain(channel, cursor)
        assert states == [2]

    def test_states_are_masked_to_a_byte(self, channel):
        publish_state(channel, 300)
        states, _ = drain(channel, 0)
        assert states == [300 & 0xFF]


class TestItFailsSoftly:
    """A telemetry sample is never worth a generation."""

    def test_no_channel_publishes_nothing_and_raises_nothing(self):
        assert publish_state(None, 5) is False
        assert drain(None, 0) == ([], 0)

    def test_a_broken_channel_does_not_raise_into_the_forward_pass(self):
        class _Hostile:
            def __getitem__(self, _index):
                raise OSError("shared memory gone")

            def __setitem__(self, _index, _value):
                raise OSError("shared memory gone")

        assert publish_state(_Hostile(), 5) is False
        assert drain(_Hostile(), 0) == ([], 0)


class TestWrapping:
    def test_a_wrapped_ring_keeps_the_most_recent_states(self, channel):
        for value in range(RING_SLOTS + 50):
            publish_state(channel, value % 256)
        states, cursor = drain(channel, 0)
        assert len(states) == RING_SLOTS
        assert cursor == RING_SLOTS + 50
        # The newest survive; a gap widens the interval on Φ rather than
        # invalidating it, and blocking decode to guarantee delivery would not
        # be a trade worth making.
        assert states[-1] == (RING_SLOTS + 49) % 256


class TestPhiCoreConsumesIt:
    def test_drained_states_become_grassmann_history(self, channel):
        from core.consciousness.phi_core import PhiCore

        core = PhiCore()
        assert len(core._grassmann_state_history) == 0
        for index in range(60):
            publish_state(channel, (index * 13) % 256)

        taken = core.drain_worker_residuals(channel)
        assert taken == 60
        assert len(core._grassmann_state_history) == 60
        assert core._grassmann_state_visits.sum() > 0

    def test_a_second_drain_does_not_double_count(self, channel):
        from core.consciousness.phi_core import PhiCore

        core = PhiCore()
        for index in range(10):
            publish_state(channel, index)
        core.drain_worker_residuals(channel)
        assert core.drain_worker_residuals(channel) == 0
        assert len(core._grassmann_state_history) == 10

    def test_no_channel_is_not_an_error(self):
        from core.consciousness.phi_core import PhiCore

        assert PhiCore().drain_worker_residuals(None) == 0

    def test_the_locator_is_quiet_when_no_worker_is_up(self):
        from core.consciousness.phi_core import PhiCore

        assert PhiCore()._locate_worker_residual_channel() is None


class TestTheWiringIsActuallyPresent:
    """A channel nothing passes across the fork changes nothing."""

    def test_the_client_allocates_the_ring(self):
        import inspect

        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        assert "_phi_residual_mem" in source
        assert "create_channel" in source

    def test_the_worker_accepts_it_and_hands_it_to_the_hooks(self):
        import inspect

        from core.brain.llm import mlx_worker

        # Follow the parameter rather than grepping one function body for the
        # attribute name. The attachment moved into its own helper once, and a
        # text match on the loop reported the channel unwired while it was
        # working.
        assert (
            "phi_residual_mem"
            in inspect.signature(mlx_worker._mlx_worker_loop).parameters
        )
        assert (
            "phi_residual_mem"
            in inspect.signature(mlx_worker._attach_affective_steering).parameters
        )
        assert (
            "phi_residual_mem"
            in inspect.signature(mlx_worker._finish_affective_attachment).parameters
        )

    def test_every_hook_gets_the_channel(self):
        """The attachment reaches all hooks, not just the first one."""
        from core.brain.llm import mlx_worker

        class _Hook:
            pass

        class _Engine:
            _model_attached = True

            def __init__(self):
                self._hooks = [_Hook(), _Hook(), _Hook()]

            def is_active(self):
                return True

            def start_substrate_sync(self, shared_state=None):
                pass

        engine = _Engine()
        channel = object()
        mlx_worker._finish_affective_attachment(
            engine, substrate_mem=None, phi_residual_mem=channel,
            steering_active_flag=None,
        )
        assert all(h._phi_residual_channel is channel for h in engine._hooks)

    def test_the_attach_actually_calls_the_finisher(self, monkeypatch):
        """The two halves are joined by a call, so make the call happen."""
        from core.brain.llm import mlx_worker

        seen = {}

        def _spy(engine, **kwargs):
            seen.update(kwargs)
            return True

        monkeypatch.setattr(mlx_worker, "_finish_affective_attachment", _spy)
        channel = object()

        class _Engine:
            _model_attached = True
            _hooks = [object()]
            _alpha = 1.0

            def attach(self, *args, **kwargs):
                pass

            def is_active(self):
                return True

        monkeypatch.setattr(
            "core.consciousness.affective_steering.get_steering_engine",
            lambda: _Engine(),
        )
        mlx_worker._attach_affective_steering(
            object(), object(), None, channel, None, model_path=None
        )
        assert seen.get("phi_residual_mem") is channel

    def test_the_hook_publishes_rather_than_looking_up_a_local_phi_core(self):
        import inspect

        from core.consciousness.affective_steering import AffectiveSteeringHook

        source = inspect.getsource(AffectiveSteeringHook._maybe_record_phi_residual)
        assert "publish_state" in source
        assert "_encode_grassmann_state" in source

    def test_compute_phi_drains_before_it_measures(self):
        import inspect

        from core.consciousness.phi_core import PhiCore

        assert "drain_worker_residuals" in inspect.getsource(PhiCore.compute_phi)
