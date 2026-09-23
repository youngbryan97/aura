"""Her substrate reaches the steering hooks in the model worker, on the scale they read.

The hooks run in the MLX worker and read a shared array as her state. Nothing
in the parent wrote it, so they read zeros, and centred on 0.5 each of the five
steered dimensions weighed -0.76: every token of every attached worker was
pushed towards low valence, arousal, frustration, curiosity and energy, the
same way whatever she felt. The fusion certificate had been measured by handing
states to the hooks directly, so nothing noticed.
"""

from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np
import pytest

from core.consciousness import steering_channel
from core.consciousness.mood_weight import NEUTRAL_MOOD, signed_weight

pytestmark = pytest.mark.unit

#: The substrate indices the steering library addresses, and the function each reads through.
STEERED = {0: "tanh", 1: "tanh", 3: "tanh", 4: "tanh", 5: "tanh"}


class _Substrate:
    """The substrate's layout and its non-blocking read, nothing else."""

    idx_valence, idx_arousal, idx_dominance = 0, 1, 2
    idx_frustration, idx_curiosity, idx_energy, idx_focus = 3, 4, 5, 6

    def __init__(self, x: list[float]) -> None:
        self.x = np.array(x + [0.0] * (64 - len(x)))

    def _state_snapshot_nowait(self) -> dict[str, Any]:
        return {"x": self.x.copy()}


def _weights(channel: Any) -> dict[int, float]:
    """What a hook computes from the channel: each dimension's weight against neutral."""
    state = np.asarray(channel, dtype=np.float32)
    return {i: signed_weight(state[i], fn) - signed_weight(NEUTRAL_MOOD, fn) for i, fn in STEERED.items()}


def _channel() -> Any:
    return steering_channel.create_channel(multiprocessing.get_context("spawn"))


def test_the_zeros_the_worker_used_to_read_steer_every_dimension_down() -> None:
    """The defect, pinned: an unwritten array is not neutral."""
    zeros = multiprocessing.get_context("spawn").Array("d", 16, lock=False)
    assert all(weight == pytest.approx(-0.7616, abs=1e-3) for weight in _weights(zeros).values())


def test_a_new_channel_stands_steering_down() -> None:
    channel = _channel()
    assert len(channel) == steering_channel.CHANNEL_LENGTH
    assert all(abs(weight) < 1e-9 for weight in _weights(channel).values())
    assert channel[-1] == 0.0


def test_her_resting_state_steers_nowhere() -> None:
    """Valence and arousal rest at 0 in the substrate; read raw, rest was fully negative."""
    channel = _channel()
    steering_channel.publish(channel, _Substrate([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]))
    assert all(abs(weight) < 1e-6 for weight in _weights(channel).values())


def test_opposite_feelings_steer_opposite_ways() -> None:
    high, low = _channel(), _channel()
    steering_channel.publish(high, _Substrate([0.9, 0.8, 0.0, 0.1, 0.9, 0.9, 0.5]))
    steering_channel.publish(low, _Substrate([-0.9, -0.8, 0.0, 0.9, 0.1, 0.1, 0.5]))
    up, down = _weights(high), _weights(low)
    for index in STEERED:
        assert np.sign(up[index]) == -np.sign(down[index]) != 0, index
    assert up[0] > 0 and down[0] < 0


def test_the_liveness_slot_is_the_workers() -> None:
    channel = _channel()
    channel[-1] = 1.0
    steering_channel.publish(channel, _Substrate([0.3] * 20))
    assert channel[-1] == 1.0


def test_no_substrate_leaves_the_channel_as_it_was(monkeypatch) -> None:
    monkeypatch.setattr(steering_channel, "her_substrate", lambda: None)
    channel = _channel()
    assert steering_channel.publish(channel) is None
    assert all(abs(weight) < 1e-9 for weight in _weights(channel).values())


def test_the_client_builds_its_channel_here_and_publishes_before_each_generation() -> None:
    """Read from the source, since a client cannot be built without a worker."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm"
    client = (root / "mlx_client.py").read_text(encoding="utf-8")
    assert "self._substrate_mem = self._create_steering_channel()" in client
    assert 'Array("d", 16' not in client
    drain = client.index("await self._drain_latent_readouts()")
    assert client.index("self._publish_steering_state()", drain) - drain < 80


def test_the_published_state_is_what_the_mixin_writes(monkeypatch) -> None:
    from core.brain.llm.mlx_latent_reasoning import _ReasonsInLatentSpace

    substrate = _Substrate([0.5, -0.5, 0.0, 0.2, 0.7, 0.4, 0.6])
    monkeypatch.setattr(steering_channel, "her_substrate", lambda: substrate)

    class Client(_ReasonsInLatentSpace):
        _mp_context = multiprocessing.get_context("spawn")

    client = Client()
    client._substrate_mem = client._create_steering_channel()
    client._publish_steering_state()
    written = np.asarray(client._substrate_mem, dtype=np.float64)
    assert written[0] == pytest.approx(0.75) and written[1] == pytest.approx(0.25)
    assert written[3:6] == pytest.approx([0.2, 0.7, 0.4])
    assert steering_channel.steering_now() == pytest.approx(list(written[:15]))


# ── The governor, and the path the sync thread takes in each process ───────────


class _Hook:
    """What the sync loop touches on a hook, recorded; stops the loop after one tick."""

    def __init__(self, thread_box: dict) -> None:
        self.box = thread_box
        self._alpha = None
        self.vectors: list[tuple[np.ndarray, str]] = []
        self.substrate_source = ""

    def update_substrate_vector(self, substrate_x, *, moods=None, source="") -> None:
        self.vectors.append((np.asarray(substrate_x, dtype=np.float64).copy(), source))
        self.box["thread"]._running = False

    @staticmethod
    def _neutral_reference_state() -> np.ndarray:
        from core.consciousness.affective_steering import AffectiveSteeringHook

        return AffectiveSteeringHook._neutral_reference_state()


def _engine() -> Any:
    import threading
    from types import SimpleNamespace

    from core.consciousness.affective_steering import DEFAULT_ALPHA, SteeringGovernor

    return SimpleNamespace(
        _state_control_lock=threading.Lock(),
        governor=SteeringGovernor(base_alpha=DEFAULT_ALPHA),
        telemetry=SimpleNamespace(alpha=None),
        _surface_alpha_override=None,
    )


def _one_tick(monkeypatch, *, shared_state=None, services=None) -> _Hook:
    """Run the real sync loop for one tick, with the container answering from `services`."""
    from core.consciousness import affective_steering
    from core.container import ServiceContainer

    services = services or {}
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: services.get(name, default)))
    monkeypatch.setattr(affective_steering, "SUBSTRATE_SYNC_INTERVAL_S", 0.0)
    box: dict = {}
    hook = _Hook(box)
    thread = affective_steering.SubstrateSyncThread([hook], engine=_engine(), shared_state=shared_state)
    box["thread"] = thread
    thread._running = True
    thread._loop()
    return hook


def test_the_arousal_slot_is_where_the_library_steers_arousal() -> None:
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    (arousal,) = [spec for spec in AFFECTIVE_DIMENSIONS if spec["key"] == "arousal"]
    assert arousal["substrate_idx"] == steering_channel.AROUSAL_SLOT


def test_the_governor_reads_arousal_from_the_state_the_hooks_steer_by() -> None:
    published = np.full(16, 0.5)
    published[steering_channel.AROUSAL_SLOT] = 0.8
    assert steering_channel.governor_inputs(published, {}) == (pytest.approx(0.8), 1.0)
    assert steering_channel.governor_inputs(published, {"arousal": 0.1, "coherence": 0.6}) == (
        pytest.approx(0.8),
        pytest.approx(0.6),
    )
    assert steering_channel.governor_inputs(None, {"arousal": 0.3}) == (pytest.approx(0.3), 1.0)


def test_in_the_worker_alpha_follows_her_arousal(monkeypatch) -> None:
    """The worker has no neurochemical system; alpha used to sit at 0.0013 of the stream."""
    import math

    from core.consciousness.affective_steering import DEFAULT_ALPHA

    def alpha_at(x_arousal: float) -> float:
        channel = _channel()
        steering_channel.publish(channel, _Substrate([0.0, x_arousal, 0.0, 0.5, 0.5, 0.5, 0.5]))
        return _one_tick(monkeypatch, shared_state=channel)._alpha

    resting, raised, lowered = alpha_at(0.0), alpha_at(0.6), alpha_at(-0.6)
    assert resting == pytest.approx(DEFAULT_ALPHA * 0.5)
    assert raised == pytest.approx(DEFAULT_ALPHA / (1.0 + math.exp(-10.0 * 0.3)))
    assert lowered < resting < raised
    assert resting > 50 * DEFAULT_ALPHA / (1.0 + math.exp(5.0))


def test_in_process_the_hooks_read_her_substrate_as_activations(monkeypatch) -> None:
    """Read raw, her resting valence of 0 was a fully negative feeling."""
    hook = _one_tick(monkeypatch, services={"liquid_substrate": _Substrate([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])})
    (vector, source), = hook.vectors
    assert source == "liquid_substrate"
    assert all(abs(weight) < 1e-6 for weight in _weights(vector[:16]).values())


def test_with_no_state_anywhere_the_hooks_are_given_neutral(monkeypatch) -> None:
    """The fallback used to hand them zeros, which they read as feeling low on every dimension."""
    hook = _one_tick(monkeypatch)
    (vector, source), = hook.vectors
    assert source == "neutral_fallback"
    assert all(abs(weight) < 1e-6 for weight in _weights(vector[:16]).values())
