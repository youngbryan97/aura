from __future__ import annotations

import mlx.core as mx
import numpy as np

from core.runtime.tensor_bridge import as_float32_numpy


def test_numpy_float32_boundary_preserves_shape_and_values() -> None:
    source = np.array([[1.25, -2.5]], dtype=np.float16)

    converted = as_float32_numpy(source)

    assert converted.dtype == np.float32
    assert converted.shape == (1, 2)
    assert converted.tolist() == [[1.25, -2.5]]


def test_mlx_bfloat16_crosses_the_numpy_boundary_as_float32() -> None:
    source = mx.array([[1.25, -2.5]], dtype=mx.bfloat16)
    mx.eval(source)

    converted = as_float32_numpy(source)

    assert converted.dtype == np.float32
    assert converted.shape == (1, 2)
    assert converted.tolist() == [[1.25, -2.5]]
