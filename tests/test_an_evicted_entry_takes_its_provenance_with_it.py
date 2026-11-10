"""When the store is full and drops an entry, the drop takes that entry's provenance and dedup slot with it.

Found 10 November 2026 reading the store's eviction for the F03 trial: the row arrays and four per-row lists moved
up over the dropped row, and the provenance list and the dedup index did not. After one eviction every later row
read as owned by the principal of the row before it, which is what principal isolation decides by, and a replay of
a known entry rewrote its neighbour.
"""
from __future__ import annotations

import numpy as np

from core.brain.nonparametric_memory import EntryProvenance, NonParametricMemory

DIM = 16


def _keys(count, seed=3):
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(count, DIM)).astype(np.float32)
    return keys / np.linalg.norm(keys, axis=1, keepdims=True)


def test_after_an_eviction_each_row_keeps_its_own_principal_and_dedup_slot(tmp_path):
    memory = NonParametricMemory(DIM, tmp_path / "full", max_entries=64)
    keys = _keys(65)
    owners = [f"person-{i % 5}" for i in range(65)]
    for i in range(64):
        memory.add(keys[i], i, f"t{i}", weight=1.0 if i != 10 else 0.01,
                   provenance=EntryProvenance(source_id=f"s{i}", principal=owners[i]))
    memory.add(keys[64], 64, "t64", provenance=EntryProvenance(source_id="s64", principal=owners[64]))
    assert memory._stats["evicted"] == 1
    assert len(memory._provenance) == len(memory) == 64
    # The entry with the least weight went, and every row still names whoever added it.
    for row in range(len(memory)):
        token = memory._token_ids[row]
        assert token != 10
        assert memory._provenance[row].principal == owners[token]
        assert memory._provenance[row].source_id == f"s{token}"
    # A replay of entry 40 refreshes entry 40, not the row after it.
    before = list(memory._token_ids)
    memory.add(keys[40], 40, "t40", weight=0.5, provenance=EntryProvenance(source_id="s40", principal=owners[40]))
    assert len(memory) == 64 and memory._token_ids == before
    row = before.index(40)
    assert memory._weights[row] == memory._clamped_weight(0.5)
    assert memory._provenance[row].source_id == "s40"
    assert memory._provenance[row + 1].source_id == f"s{memory._token_ids[row + 1]}"


def test_who_may_read_an_entry_survives_an_eviction(tmp_path):
    memory = NonParametricMemory(DIM, tmp_path / "isolation", max_entries=64)
    keys = _keys(65, seed=5)
    for i in range(64):
        memory.add(keys[i], i, f"t{i}", weight=1.0 if i else 0.01,
                   provenance=EntryProvenance(source_id="s", principal="alice" if i < 32 else "bob"))
    memory.add(keys[64], 64, "t64", provenance=EntryProvenance(source_id="s", principal="bob"))
    seen = memory.visible_to("alice")
    assert sorted(int(t) for t in seen["token_ids"]) == list(range(1, 32))
