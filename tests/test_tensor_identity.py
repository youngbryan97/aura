from __future__ import annotations

import hashlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from core.brain.llm.latent_cortex.runtime_integrity import (
    parameter_canary_fingerprint,
)
from core.brain.llm.latent_cortex.verified_best import tensor_sha256
from core.runtime.tensor_identity import tensor_identity_parts


def test_numpy_tensor_identity_preserves_dtype_shape_and_storage() -> None:
    value = np.array([[1.5, -2.0]], dtype=np.float16)

    dtype, shape, payload = tensor_identity_parts(value)

    assert dtype == "float16"
    assert shape == (1, 2)
    assert payload == np.ascontiguousarray(value).tobytes(order="C")


def test_mlx_bfloat16_identity_uses_exact_uint16_storage() -> None:
    value = mx.array([[1.5, -2.0]], dtype=mx.bfloat16)
    mx.eval(value)

    dtype, shape, payload = tensor_identity_parts(value)

    assert dtype == "bfloat16"
    assert shape == (1, 2)
    assert payload == np.asarray(value.view(mx.uint16)).tobytes(order="C")
    assert len(payload) == 4


def test_logical_dtype_distinguishes_equal_bfloat16_and_uint16_storage() -> None:
    bfloat = mx.array([1.5, -2.0], dtype=mx.bfloat16)
    storage = bfloat.view(mx.uint16)
    mx.eval(bfloat, storage)

    _, _, bfloat_payload = tensor_identity_parts(bfloat)
    _, _, storage_payload = tensor_identity_parts(storage)

    assert bfloat_payload == storage_payload
    assert tensor_sha256(bfloat) != tensor_sha256(storage)


def test_tensor_identity_is_deterministic_and_mutation_sensitive() -> None:
    first = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
    changed = mx.array([1.0, 2.0, 4.0], dtype=mx.bfloat16)
    mx.eval(first, changed)

    assert tensor_sha256(first) == tensor_sha256(first)
    assert tensor_sha256(first) != tensor_sha256(changed)


def test_parameter_canary_fingerprints_real_mlx_bfloat16_parameters() -> None:
    model = SimpleNamespace(
        parameters=lambda: {
            "weight": mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)
        }
    )

    first = parameter_canary_fingerprint(model, stride=1, elements_per_tensor=4)
    second = parameter_canary_fingerprint(model, stride=1, elements_per_tensor=4)

    assert first == second
    assert first["sampled_tensor_count"] == 1
    assert first["sampled_element_count"] == 4
    assert len(first["sha256"]) == hashlib.sha256().digest_size * 2


def test_tensor_identity_rejects_object_storage() -> None:
    with pytest.raises(TypeError, match="does not admit object storage"):
        tensor_identity_parts(np.array([object()], dtype=object))
