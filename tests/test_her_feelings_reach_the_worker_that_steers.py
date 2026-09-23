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
