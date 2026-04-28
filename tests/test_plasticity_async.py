"""Tests for PlasticityMonitor async SVD offload.

Verifies that measure_async() produces identical results to measure()
while running off the event-loop thread.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest


def test_sync_measure_still_works():
    """The synchronous measure() path must remain functional."""
    from core.consciousness.plasticity_monitor import PlasticityMonitor

    mon = PlasticityMonitor()
    W = np.random.randn(64, 64).astype(np.float64)
    report = mon.measure(W)

    assert report is not None
    assert report.nominal_rank == 64
    assert report.stable_rank > 0
    assert report.stable_rank_ratio > 0
    assert report.measurement_count == 1


@pytest.mark.asyncio
async def test_async_measure_matches_sync():
    """measure_async() must produce the same report as measure() for the same matrix."""
    from core.consciousness.plasticity_monitor import PlasticityMonitor

    mon = PlasticityMonitor()
    W = np.random.randn(32, 32).astype(np.float64)

    sync_report = mon.measure(W.copy())
    async_report = await mon.measure_async(W.copy())

    assert async_report is not None
    assert sync_report is not None
    # Values should be identical (same matrix, same algorithm)
    assert abs(async_report.stable_rank - sync_report.stable_rank) < 1e-6
    assert abs(async_report.stable_rank_ratio - sync_report.stable_rank_ratio) < 1e-6
    # Measurement count incremented twice (once sync, once async)
    assert mon.measurement_count == 2


@pytest.mark.asyncio
async def test_async_measure_none_input():
    """measure_async(None) should return None without error."""
    from core.consciousness.plasticity_monitor import PlasticityMonitor

    mon = PlasticityMonitor()
    result = await mon.measure_async(None)
    assert result is None


@pytest.mark.asyncio
async def test_async_measure_oversized_matrix_refused():
    """Matrices exceeding MAX_MATRIX_DIM should return None."""
    from core.consciousness.plasticity_monitor import PlasticityMonitor, MAX_MATRIX_DIM

    mon = PlasticityMonitor()
    # Create a matrix just over the limit (don't actually allocate a huge one)
    W = np.zeros((MAX_MATRIX_DIM + 1, 2), dtype=np.float64)
    result = await mon.measure_async(W)
    assert result is None


def test_bounded_pool_created():
    """The SVD pool should be bounded (max 2 workers)."""
    from core.consciousness.plasticity_monitor import _get_svd_pool

    pool = _get_svd_pool()
    assert pool._max_workers == 2
