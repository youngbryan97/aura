"""The Φ reverse channel had a writer and no reader.

``core/consciousness/phi_residual_channel.py`` exists because the steering
hook's ``ServiceContainer.has("phi_core")`` is an in-process lookup and
generation does not run in that process. The hook lives in the MLX worker;
PhiCore is registered in the main runtime. So the module was written, the
parent allocated the ring, the worker published an 8-bit Grassmann state per
sampled token — and nothing ever called ``drain()``.

The activation-grounded complex went on reporting
``insufficient_history:0/50``, which is the precise symptom the channel was
built to cure. A half-wired fix is indistinguishable from no fix, and this
one looked more convincing than most: three modules, a documented protocol,
and a live writer.

These tests hold both ends.
"""
from __future__ import annotations

from mlx_source import client_source, worker_source

import ast
import inspect
import multiprocessing as mp
from pathlib import Path

import pytest

from core.consciousness.phi_core import PhiCore
from core.consciousness.phi_residual_channel import (
    RING_SLOTS,
    create_channel,
    drain,
    publish_state,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def channel():
    return create_channel(mp.get_context("spawn"))


class TestTheChannelItself:
    def test_states_survive_the_round_trip(self, channel):
        for value in range(50):
            publish_state(channel, value)
        states, cursor = drain(channel, 0)
        assert states == list(range(50))
        assert cursor == 50

    def test_a_second_drain_returns_only_what_is_new(self, channel):
        for value in range(10):
            publish_state(channel, value)
        _, cursor = drain(channel, 0)
        for value in range(10, 15):
            publish_state(channel, value)
        states, _ = drain(channel, cursor)
        assert states == list(range(10, 15))

    def test_a_wrapped_ring_keeps_the_most_recent(self, channel):
        for value in range(RING_SLOTS + 100):
            publish_state(channel, value % 256)
        states, _ = drain(channel, 0)
        assert len(states) == RING_SLOTS


class TestPhiCoreAcceptsThem:
    def test_an_encoded_state_lands_in_the_history(self):
        core = PhiCore()
        before = core.grassmann_history_depth()
        core.record_grassmann_state(42)
        assert core.grassmann_history_depth() == before + 1

    def test_garbage_does_not_raise(self):
        core = PhiCore()
        core.record_grassmann_state("not an int")  # type: ignore[arg-type]
        core.record_grassmann_state(None)  # type: ignore[arg-type]


def test_the_whole_path_produces_an_activation_grounded_phi(channel):
    """Worker publishes, parent drains, PhiCore measures.

    Before the reader existed this was unreachable: the history stayed empty,
    so ``compute_grassmann_residual_phi()`` returned None on every call for
    the life of the process.
    """
    import random

    random.seed(7)
    state = 0
    for _ in range(400):
        state = ((state << 1) | (1 if random.random() < 0.5 else 0)) & 0xFF
        publish_state(channel, state)

    states, _ = drain(channel, 0)
    core = PhiCore()
    assert core.compute_grassmann_residual_phi() is None, (
        "an empty history must not produce a Φ — that would be the "
        "zero-over-zero failure in the measurement itself"
    )

    for value in states:
        core.record_grassmann_state(value)

    result = core.compute_grassmann_residual_phi()
    assert result is not None, "the complex still cannot fill"
    assert result.grounding == "activation_geometry"
    assert result.tpm_n_samples > 300
    assert result.phi_s >= 0.0


class TestTheReaderIsWired:
    def test_the_client_drains_the_ring(self):
        # The drain moved to the latent lane's module. Both halves still have
        # to be in the client's source; which file holds them is not what this
        # is about.
        source = client_source()
        assert "def _drain_phi_residual_ring" in source
        assert "self._drain_phi_residual_ring()" in source, (
            "the drain exists but nothing calls it — which is the state this "
            "whole channel was already in"
        )

    def test_the_drain_is_called_on_the_generation_path(self):
        """Not in a helper nobody reaches: the path that produces the states."""
        from core.brain.llm.mlx_client import MLXLocalClient

        source = inspect.getsource(MLXLocalClient.generate)
        assert "_drain_phi_residual_ring" in source

    def test_the_writer_is_still_there_too(self):
        """Both ends, or the channel is half a channel again."""
        steering = (ROOT / "core" / "consciousness" / "affective_steering.py").read_text(
            encoding="utf-8"
        )
        assert "publish_state(channel, state)" in steering

        worker = worker_source()
        assert "_phi_residual_channel = phi_residual_mem" in worker

    def test_the_parent_allocates_the_ring_before_the_fork(self):
        client = client_source()
        tree = ast.parse(client)
        creates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_channel"
        ]
        assert creates, "the parent no longer allocates the phi residual ring"
