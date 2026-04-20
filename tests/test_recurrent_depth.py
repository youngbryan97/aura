"""Tests for the Mythos-inspired recurrent-depth patch.

Guards the load-bearing assumption: mlx_lm's KVCache state/meta_state
snapshot/restore correctly rewinds offset after a mutation. A silent
failure here would have the recurrent loop accumulate N copies of K/V
into the cache — far worse than leaving recurrent depth off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.llm.recurrent_depth import (  # noqa: E402
    CacheSnapshotError,
    _self_test_cache_snapshot,
    _snapshot_recurrent_caches,
    _restore_recurrent_caches,
    _get_lane_defaults,
)


def test_self_test_cache_snapshot_passes_on_installed_mlx_lm():
    """If this fails, mlx_lm's cache contract changed and we must not patch."""
    _self_test_cache_snapshot()


def test_snapshot_fails_loud_on_unsupported_cache():
    """Incompatible caches must raise, never silently no-op."""

    class _BadCache:
        """Neither state/meta_state nor keys/values/offset."""
        pass

    with pytest.raises(CacheSnapshotError):
        _snapshot_recurrent_caches([_BadCache()], 0, 1)


def test_lane_defaults_cover_real_model_sizes():
    """Qwen2.5-32B has 64 layers; Qwen2.5-72B has 80. Both must land in
    lanes with n_loops >= 2 — otherwise the entire feature is a no-op on
    the models Aura actually uses."""
    assert _get_lane_defaults(64)[0] >= 2, "32B (64 layers) must map to a looped lane"
    assert _get_lane_defaults(80)[0] >= 2, "72B (80 layers) must map to a looped lane"
    # And the small-model lanes must be standard-pass (no unnecessary cost).
    assert _get_lane_defaults(28)[0] == 1, "14B (28-40 layers) should be standard"
    assert _get_lane_defaults(12)[0] == 1, "7B class should be standard"


def test_restore_rewinds_mlx_cache():
    """Direct end-to-end proof the snapshot/restore actually works."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    c.update_and_fetch(mx.ones((1, 2, 8, 16)), mx.ones((1, 2, 8, 16)))
    pre_offset = c.offset
    snap = _snapshot_recurrent_caches([c], 0, 1)

    c.update_and_fetch(mx.ones((1, 2, 1, 16)) * 3, mx.ones((1, 2, 1, 16)) * 3)
    assert c.offset > pre_offset, "Mutation did not advance cache offset"

    _restore_recurrent_caches([c], 0, 1, snap)
    assert c.offset == pre_offset, f"Restore failed: {pre_offset} → {c.offset}"
