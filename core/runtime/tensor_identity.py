"""Canonical logical identity and storage bytes for numeric tensors."""

from __future__ import annotations

import hashlib
from typing import Any


def tensor_identity_parts(value: Any) -> tuple[str, tuple[int, ...], bytes]:
    """Return logical dtype, shape, and exact contiguous storage bytes.

    NumPy cannot consume MLX ``bfloat16`` through the buffer protocol because
    PEP 3118 has no matching format. Viewing that tensor as ``uint16`` exposes
    its exact storage without changing the logical identity carried in the
    digest. Other numeric arrays retain their existing NumPy representation.
    """

    import numpy as np

    source_module = type(value).__module__.partition(".")[0]
    source_dtype = str(getattr(value, "dtype", "")).removeprefix("mlx.core.")
    if source_module == "mlx" and source_dtype == "bfloat16":
        import mlx.core as mx

        array = np.asarray(value.view(mx.uint16))
        logical_dtype = "bfloat16"
    else:
        array = np.asarray(value)
        logical_dtype = str(array.dtype)
    if array.dtype.hasobject:
        raise TypeError("tensor identity does not admit object storage")
    contiguous = np.ascontiguousarray(array)
    shape = tuple(int(dimension) for dimension in contiguous.shape)
    return logical_dtype, shape, contiguous.tobytes(order="C")


def tensor_identity_sha256(value: Any) -> str:
    """Hash a tensor's logical dtype, shape, and exact storage bytes."""

    dtype, shape, payload = tensor_identity_parts(value)
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(str(shape).encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


__all__ = ["tensor_identity_parts", "tensor_identity_sha256"]
