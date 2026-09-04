"""Stable host-array boundaries for accelerator tensors."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_float32_numpy(value: Any) -> np.ndarray:
    """Return a NumPy float32 view, including for MLX bfloat16 arrays.

    PEP 3118 has no bfloat16 format. MLX therefore exposes its two-byte
    bfloat16 storage through a one-byte ``B`` format that NumPy cannot cast
    directly. Casting on the device before crossing the host boundary keeps
    the values exact and gives NumPy a portable buffer contract.
    """

    source_module = type(value).__module__.partition(".")[0]
    if source_module == "mlx":
        import mlx.core as mx

        value = value.astype(mx.float32)
        mx.eval(value)
    return np.asarray(value, dtype=np.float32)


__all__ = ["as_float32_numpy"]
