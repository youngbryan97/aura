"""Her live recall store can hold as many entries as recall needs before it will steer."""
from __future__ import annotations

from core.brain.nonparametric_memory import ENTRIES_TO_STEER_GENERATION, NonParametricMemory, get_nonparametric_memory
from core.brain.nonparametric_worker import _MIN_ENTRIES_TO_STEER_GENERATION


def test_the_live_store_holds_the_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    store = get_nonparametric_memory(64)
    assert store is not None and store._max >= _MIN_ENTRIES_TO_STEER_GENERATION == ENTRIES_TO_STEER_GENERATION


def test_capacity_allocates_nothing_until_used(tmp_path):
    store = NonParametricMemory(64, path=tmp_path / "s", max_entries=ENTRIES_TO_STEER_GENERATION)
    assert store._keys.shape[0] <= 64
