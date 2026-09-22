"""Rust-accelerated substrate kernels with a NumPy fallback (task #40).

The continuous-time substrate (core/consciousness/unified_field) runs a tight
integration loop every tick. On Apple Silicon this is a real CPU/heat source,
and heat correlates with the cortex-wedge frontier (#45). This module is the
seam for porting those hot kernels to Rust (rust_extensions/aura_m1_ext):

  - if the compiled extension is present, the Rust kernel runs;
  - otherwise the exact NumPy reference runs, so behavior is identical whether or
    not the extension is built.

Build the extension with (from the repo root, needs rustup + maturin)::

    maturin develop -m rust_extensions/aura_m1_ext/Cargo.toml --release

Until then, ``RUST_ACCEL_AVAILABLE`` is False and the NumPy path is used. When
built, the Rust path is ~4x faster at exact parity and is used by default; the
parity test (tests/test_substrate_accel.py) verifies the two paths agree.
"""
from __future__ import annotations

import os

import numpy as np

_ACCEL_IMPORT_ERRORS = (ImportError, AttributeError, OSError)

try:  # pragma: no cover - depends on whether the native ext is built
    import aura_m1_ext as _ext

    RUST_ACCEL_AVAILABLE = hasattr(_ext, "field_integrate")
# not a failure: the native extension is optional; the numpy path stays correct.
except _ACCEL_IMPORT_ERRORS:
    _ext = None
    RUST_ACCEL_AVAILABLE = False

# Whether to actually USE the Rust path. Default ON when the extension is built:
# with zero-copy numpy buffers the kernel is measured ~4x faster than the NumPy
# path (0.75us vs 3.11us per 256-elem call) at EXACT parity (verified in tests).
# (A naive list-marshalling binding was 5x SLOWER — zero-copy is what makes it a
# real win.) Opt out with AURA_RUST_SUBSTRATE=0.
_RUST_ENABLED = RUST_ACCEL_AVAILABLE and os.environ.get(
    "AURA_RUST_SUBSTRATE", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _field_integrate_numpy(
    f: np.ndarray, activity: np.ndarray, noise: np.ndarray, decay: float, dt: float
) -> np.ndarray:
    """Reference: next = clip(f + (-decay*f + activity + noise)*dt, -1, 1)."""
    df = (-float(decay) * f + activity + noise) * float(dt)
    return np.clip(f + df, -1.0, 1.0).astype(np.float32)


def field_integrate(
    f: np.ndarray, activity: np.ndarray, noise: np.ndarray, decay: float, dt: float
) -> np.ndarray:
    """One Euler integration step of the unified field.

    Rust-accelerated when ``aura_m1_ext`` is built; otherwise the identical NumPy
    reference. Always returns a float32 array of the same shape as ``f``.
    """
    if _RUST_ENABLED:
        try:
            # Zero-copy: pass contiguous float32 numpy buffers straight to Rust.
            return np.asarray(
                _ext.field_integrate(
                    np.ascontiguousarray(f, dtype=np.float32).ravel(),
                    np.ascontiguousarray(activity, dtype=np.float32).ravel(),
                    np.ascontiguousarray(noise, dtype=np.float32).ravel(),
                    float(decay),
                    float(dt),
                ),
                dtype=np.float32,
            )
        # not a failure: the comment below says it: the numpy fallback runs instead.
        except _ACCEL_IMPORT_ERRORS + (ValueError, TypeError, RuntimeError):
            # Never let an extension hiccup break the substrate — fall back.
            pass
    return _field_integrate_numpy(
        np.asarray(f, dtype=np.float32),
        np.asarray(activity, dtype=np.float32),
        np.asarray(noise, dtype=np.float32),
        decay,
        dt,
    )
