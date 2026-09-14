"""Measuring the mind must not stop the mind.

`AffectiveSteeringHook` patches the forward pass of all 64 transformer blocks,
so anything it does happens 64 times per forward — including every forward of
the prompt during prefill, before a first token can exist.

PhiCore's `record_residual_stream` begins with `np.asarray(hidden_state)`. On
MLX that is a blocking device sync *and* a full materialisation of whatever it
is handed. During prefill the hook is handed the whole sequence — [1, seq, 5120]
— so a single Φ sample copied tens of megabytes off the GPU and collapsed MLX's
lazy pipeline, repeatedly, on the one path where latency decides whether a turn
survives at all.

Measured live 2026-07-26: ~3k-token prompts took 58-82s to a first token —
about 50 tok/s, roughly twenty times slower than this model should prefill.
Turns 5 through 7 of one conversation died on that alone, while turns 1-4 had
answered well. Nothing about her degraded; the instrumentation was in the way.

Prefill is not a thought moment: the signal Φ wants is the per-token dynamics of
generation. So the hook samples only single-token decode steps, and hands over
one already-sliced position rather than a sequence.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.consciousness.phi_residual_sampler import PhiResidualSampler


class _FakeTensor:
    """Stands in for an MLX array; slicing it must be what gets recorded."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def __getitem__(self, key):  # noqa: ANN001 - test double
        return "sliced-position"


def _hook() -> PhiResidualSampler:
    # sample every call, so skips are the sampler's doing
    return PhiResidualSampler(layer_idx=7, report=lambda *a, **k: None, sample_every=1)


def _record(hook, tensor) -> list:
    seen: list = []
    phi = MagicMock()
    phi.record_residual_stream = lambda sample, **_kw: seen.append(sample)
    with patch("core.container.ServiceContainer.has", return_value=True), patch(
        "core.container.ServiceContainer.get", return_value=phi
    ):
        hook.maybe_record(tensor, inject_count=0)
    return seen


@pytest.mark.parametrize(
    "shape",
    [(1, 2048, 5120), (1, 3000, 5120), (1, 2, 5120), (512, 5120)],
)
def test_prefill_is_never_sampled(shape: tuple[int, ...]) -> None:
    """A multi-token hidden state is prefill; syncing there is the 20x cost."""
    assert _record(_hook(), _FakeTensor(shape)) == []


def test_a_decode_step_is_still_sampled() -> None:
    """The signal Φ actually wants — per-token generation dynamics — survives."""
    assert _record(_hook(), _FakeTensor((1, 1, 5120))) == ["sliced-position"]


def test_the_recorded_sample_is_one_position_not_a_sequence() -> None:
    """Hand over ~5120 floats, never a tensor for PhiCore to materialise."""
    seen = _record(_hook(), _FakeTensor((1, 1, 5120)))
    assert seen == ["sliced-position"], "the hook must slice before handing over"


def test_a_shapeless_value_still_records_as_before() -> None:
    """The skip is about sequences, not about types.

    Anything without a `.shape` is not a multi-token MLX tensor, so it cannot
    be the prefill copy this guards against. It keeps the prior behaviour and
    reaches PhiCore unchanged.
    """

    class _NoShape:
        pass

    value = _NoShape()
    assert _record(_hook(), value) == [value]


def test_the_env_kill_switch_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("AURA_PHI_RECORD_RESIDUALS", "0")
    assert _record(_hook(), _FakeTensor((1, 1, 5120))) == []
