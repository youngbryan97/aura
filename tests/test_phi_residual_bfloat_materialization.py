"""Residual telemetry accepts the resident model's native numerical dtype."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.consciousness.phi_residual_sampler import (
    PhiResidualSampler,
    _materialize_float32_vector,
)


class _BfloatLike:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("Item size 2 for PEP 3118 buffer format string B")

    def tolist(self):
        return [0.25, -0.5, 1.0]


def test_bfloat_residual_reaches_the_encoder_without_a_fault():
    sampler = PhiResidualSampler(layer_idx=3, report=lambda *args, **kwargs: None)
    seen = []
    sampler.encoder = SimpleNamespace(observe=lambda vector: seen.append(vector))
    assert sampler.encode(_BfloatLike()) is None
    np.testing.assert_array_equal(seen[0], np.array([0.25, -0.5, 1.0], dtype=np.float32))
    assert sampler.encode_errors == 0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_real_mlx_residual_values_survive_materialization(dtype):
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        sample = mx.array([0.25, -0.5, 1.0], dtype=getattr(mx, dtype))
        vector = _materialize_float32_vector(sample)
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, [0.25, -0.5, 1.0])


def test_invalid_residual_is_still_an_error():
    with pytest.raises((TypeError, ValueError)):
        _materialize_float32_vector(object())
