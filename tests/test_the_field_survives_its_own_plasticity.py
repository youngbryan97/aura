"""A fail-closed subsystem that died on the first turn of every campaign.

`csr_matrix` counts the non-zeros of an array, allocates for that count and
then copies. The plasticity pass writes the same array in place — pruning below
a threshold, zeroing the diagonal, rescaling — and it was the one path in the
class that did not take the lock the others take. A write landing between the
count and the copy makes scipy raise "number of non-zero array elements changed
during function execution", and unified_field is fail-closed, so the field task
died at turn one and every later turn of the run had no field in it.

Two halves: the sparse build gets a private array, and the in-place write takes
the lock.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest


def _field():
    from core.consciousness.unified_field import UnifiedField

    return UnifiedField.__new__(UnifiedField)


def test_the_sparse_build_cannot_see_a_write_land() -> None:
    """Scipy is handed a copy, so a concurrent writer cannot change it."""
    sp = pytest.importorskip("scipy.sparse")
    field = _field()
    weights = np.zeros((64, 64), dtype=np.float32)
    weights[0, 1] = 1.0
    stop = threading.Event()

    def churn() -> None:
        rng = np.random.default_rng(0)
        while not stop.is_set():
            # The same in-place mutation the plasticity pass makes.
            weights[rng.integers(0, 64), rng.integers(0, 64)] = rng.normal()
            weights[weights < 0.0] = 0.0

    writer = threading.Thread(target=churn, daemon=True)
    writer.start()
    try:
        for _ in range(200):
            out = field._to_sparse(weights)
            assert out.shape == (64, 64)
    finally:
        stop.set()
        writer.join(timeout=2.0)


def test_the_copy_is_private_to_the_sparse_matrix() -> None:
    sp = pytest.importorskip("scipy.sparse")
    field = _field()
    weights = np.zeros((8, 8), dtype=np.float32)
    weights[2, 3] = 5.0
    out = field._to_sparse(weights)
    weights[2, 3] = 0.0
    assert out[2, 3] == pytest.approx(5.0), "the sparse form shares the caller's array"


def test_without_scipy_the_dense_array_is_returned() -> None:
    import core.consciousness.unified_field as uf

    field = _field()
    saved = uf.sp
    uf.sp = None
    try:
        weights = np.eye(4, dtype=np.float32)
        assert field._to_sparse(weights) is weights
    finally:
        uf.sp = saved


def test_the_plasticity_write_takes_the_lock() -> None:
    """The one path that mutates the weights in place, under the same lock."""
    from core.consciousness.unified_field import UnifiedField

    field = _field()
    field._lock = threading.RLock()
    held: list[bool] = []

    def _locked(self=field) -> None:
        # The lock is an RLock, so a holder can acquire it again without
        # blocking; a non-holder on another thread cannot.
        got = threading.Thread(target=lambda: held.append(field._lock.acquire(timeout=0.2)))
        got.start()
        got.join()

    field._apply_plasticity_locked = _locked
    UnifiedField._apply_plasticity(field)
    assert held == [False], "the plasticity write ran without the lock"
