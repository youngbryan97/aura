"""A snapshot restored into two caches gives both the same buffers.

`_restore_recurrent_caches` assigns the snapshot's arrays straight onto the
cache — `c.keys = snap[1]` — which preserves the spare allocation and is
exactly right for the rewind it was written for, where one cache goes back to
its own earlier state.

Restoring ONE snapshot into TWO caches is a different operation and it aliases
them. The heterogeneous dual-lane decode did that, and its two "cache-isolated"
lanes wrote into each other: their persisted logits differed by 34.9 and their
bridged logits were byte-identical, so the Jensen-Shannon divergence between
them was exactly zero on every token. A measurement that could not vary,
reported as a measurement.

Nothing noticed because a separate defect upstream made every episode fall
back to a vanilla decode before reaching this code at all.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.recurrent_depth import (  # noqa: E402
    _restore_recurrent_caches,
    _snapshot_recurrent_caches,
    isolate_cache_buffers,
)


class _Cache:
    """The shape `_snapshot_recurrent_caches` treats as a live MLX cache."""

    def __init__(self, value: float) -> None:
        self.keys = mx.array([[value, value]])
        self.values = mx.array([[value, value]])
        self.offset = 2


def test_restoring_one_snapshot_into_two_caches_aliases_them():
    """The behaviour, named. This is not a bug in the restorer."""
    source = [_Cache(1.0)]
    snapshots = _snapshot_recurrent_caches(source, 0, 1)

    left, right = [_Cache(0.0)], [_Cache(0.0)]
    _restore_recurrent_caches(left, 0, 1, snapshots)
    _restore_recurrent_caches(right, 0, 1, snapshots)

    assert left[0].keys is right[0].keys


def test_isolating_gives_each_cache_its_own_storage():
    source = [_Cache(1.0)]
    snapshots = _snapshot_recurrent_caches(source, 0, 1)

    left, right = [_Cache(0.0)], [_Cache(0.0)]
    _restore_recurrent_caches(left, 0, 1, snapshots)
    _restore_recurrent_caches(right, 0, 1, snapshots)
    isolate_cache_buffers(left, 0, 1)
    isolate_cache_buffers(right, 0, 1)

    assert left[0].keys is not right[0].keys
    assert left[0].values is not right[0].values
    # Same contents, different storage: isolation is not a reset.
    assert float(mx.sum(mx.abs(left[0].keys - right[0].keys))) == 0.0


def test_a_write_to_one_isolated_cache_does_not_reach_the_other():
    """The property the dual-lane decode needed and did not have."""
    source = [_Cache(1.0)]
    snapshots = _snapshot_recurrent_caches(source, 0, 1)

    left, right = [_Cache(0.0)], [_Cache(0.0)]
    _restore_recurrent_caches(left, 0, 1, snapshots)
    _restore_recurrent_caches(right, 0, 1, snapshots)
    isolate_cache_buffers(left, 0, 1)
    isolate_cache_buffers(right, 0, 1)

    # In place, the way a persisted decode writes into its own cache.
    left[0].keys[:] = left[0].keys + 5.0
    mx.eval(left[0].keys)
    assert float(mx.sum(mx.abs(left[0].keys))) == 12.0
    assert float(mx.sum(mx.abs(right[0].keys))) == 2.0


def test_isolating_a_cache_with_no_buffers_is_harmless():
    class _Composite:
        def __init__(self) -> None:
            self.state = mx.array([1.0])
            self.meta_state = None

    caches = [_Composite(), None]
    isolate_cache_buffers(caches, 0, 2)
    assert caches[1] is None
