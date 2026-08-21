"""Transactional and evidence contracts for long-term memory consolidation."""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from core.brain.cognitive.memory_management import MemoryConsolidator
from core.memory.black_hole_vault import BlackHoleVault
from core.memory.rag import compute_term_freq, tokenize
from core.memory.sqlite_vector_store import SQLiteVectorStore


def _meta(*, principal: str = "p1", namespace: str = "n1", **extra):
    return {
        "principal_id": principal,
        "memory_namespace": namespace,
        "memory_class": "episodic",
        "embedding_metric": "cosine",
        "embedding_version": "test-v1",
        "timestamp": 1.0,
        **extra,
    }


class _SQLiteMemory:
    def __init__(self, path: Path):
        self.collection_name = "test-memory"
        self._sqlite_vectors = SQLiteVectorStore(path, collection_name=self.collection_name)
        self._collection = None
        self._fallback_mode = False
        self._mutation_lock = threading.RLock()
        self.single_principal_collection = False
        self.embedding_metric = "cosine"
        self.embedding_version = "test-v1"
        self._store = []
        self._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]

    def put(self, record_id: str, content: str, vector, metadata=None):
        metadata = metadata or _meta()
        self._sqlite_vectors.upsert(record_id, content, vector, metadata=metadata)
        self._store.append({"id": record_id, "content": content, "metadata": metadata})


class _ChromaCollection:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self, records):
        self.records = {item["id"]: dict(item) for item in records}
        self.fail_delete_once = False

    def count(self):
        return len(self.records)

    def get(self, *, ids=None, limit=None, offset=0, include=None):
        keys = sorted(self.records)
        if ids is not None:
            requested = set(ids)
            keys = [key for key in keys if key in requested]
        elif limit is not None:
            keys = keys[offset : offset + limit]
        return {
            "ids": keys,
            "documents": [self.records[key]["content"] for key in keys],
            "metadatas": [dict(self.records[key]["metadata"]) for key in keys],
            "embeddings": [list(self.records[key]["embedding"]) for key in keys],
        }

    def update(self, *, ids, documents, metadatas, embeddings):
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "id": record_id,
                "content": documents[index],
                "metadata": dict(metadatas[index]),
                "embedding": list(embeddings[index]),
            }

    def delete(self, *, ids):
        if self.fail_delete_once:
            self.fail_delete_once = False
            raise OSError("simulated delete failure")
        for record_id in ids:
            self.records.pop(record_id, None)

    def upsert(self, *, ids, documents, metadatas, embeddings):
        self.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


class _ChromaMemory:
    def __init__(self, path: Path, records):
        self.collection_name = "chroma-test"
        self.persist_directory = path
        self._sqlite_vectors = None
        self._collection = _ChromaCollection(records)
        self._fallback_mode = False
        self._mutation_lock = threading.RLock()
        self.single_principal_collection = False
        self.embedding_metric = "cosine"
        self.embedding_version = "test-v1"
        self._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]


class _BlackHoleMemory:
    memory_consolidation_backend = "black_hole_vault"
    collection_name = "black_hole_vault"
    embedding_metric = "cosine"
    embedding_version = "black-hole-tfhash384-v1"
    single_principal_collection = True

    def __init__(self, records):
        self.memories = deepcopy(records)
        self._mutation_lock = threading.RLock()
        self._dirty = False
        self.saved = []
        self.fail_save_once = False

    _memory_id = staticmethod(BlackHoleVault._memory_id)

    def _ensure_memory_ids(self, *, persist=False):
        return BlackHoleVault._ensure_memory_ids(self, persist=persist)

    def _save_vault(self):
        if self.fail_save_once:
            self.fail_save_once = False
            raise OSError("simulated vault write failure")
        self.saved.append(deepcopy(self.memories))
        self._dirty = False


def _black_hole_record(record_id, content, metadata=None):
    return {
        "id": record_id,
        "text": content,
        "metadata": metadata or _meta(),
        "vec": compute_term_freq(tokenize(content)),
        "created": 1,
        "access_count": 0,
    }


def _record(record_id, content, vector, metadata=None):
    return {
        "id": record_id,
        "content": content,
        "metadata": metadata or _meta(),
        "embedding": vector,
    }


@pytest.mark.parametrize(
    ("threshold", "batch_size"),
    [(0.49, 10), (-0.1, 10), (1.1, 10), (float("nan"), 10), (0.9, 0), (0.9, 10_001)],
)
def test_constructor_rejects_unsafe_scan_parameters(threshold, batch_size):
    with pytest.raises(ValueError):
        MemoryConsolidator(
            object(), similarity_threshold=threshold, batch_size=batch_size
        )


@pytest.mark.asyncio
async def test_complete_scan_transitive_cluster_and_content_preservation(tmp_path):
    memory = _SQLiteMemory(tmp_path / "vectors.sqlite3")
    # A~B and B~C clear the threshold, while A~C does not. Connected-component
    # clustering must still produce one deterministic cluster.
    memory.put(
        "a",
        "alpha fact",
        [1.0, 0.0],
        _meta(importance=0.4, source_link="episode://alpha"),
    )
    memory.put("b", "beta fact", [math.cos(math.radians(30)), 0.5], _meta(importance=0.8))
    memory.put("c", "gamma fact", [0.5, math.sin(math.radians(60))], _meta(importance=0.6))
    memory.put("d", "unrelated", [-1.0, 0.0], _meta())

    report = await MemoryConsolidator(
        memory, similarity_threshold=0.85, batch_size=2
    ).consolidate()

    assert report.scan_complete is True
    assert report.memories_scanned == 4
    assert report.pages_scanned == 2
    assert report.clusters_found == 1
    assert report.duplicates_merged == 2
    assert report.transactions_committed == 1
    rows = memory._sqlite_vectors.list_records(collection=memory.collection_name)
    assert len(rows) == 2
    assert {row["id"] for row in rows} & {"a", "b", "c"}
    assert "d" in {row["id"] for row in rows}
    winner = next(row for row in rows if row["id"] != "d")
    assert all(text in winner["content"] for text in ("alpha fact", "beta fact", "gamma fact"))
    assert winner["metadata"]["merged_count"] == 2
    provenance = json.loads(winner["metadata"]["consolidation_provenance_json"])
    assert {item["id"] for item in provenance} == {"a", "b", "c"}
    alpha = next(item for item in provenance if item["id"] == "a")
    assert alpha["metadata"]["source_link"] == "episode://alpha"


@pytest.mark.asyncio
async def test_scope_and_memory_class_boundaries_prevent_cross_owner_merge(tmp_path):
    memory = _SQLiteMemory(tmp_path / "vectors.sqlite3")
    memory.put("p1", "same", [1.0, 0.0], _meta(principal="p1"))
    memory.put("p2", "same", [1.0, 0.0], _meta(principal="p2"))
    memory.put(
        "semantic",
        "same",
        [1.0, 0.0],
        {**_meta(principal="p1"), "memory_class": "semantic"},
    )

    report = await MemoryConsolidator(memory, similarity_threshold=0.99).consolidate()

    assert report.clusters_found == 0
    assert report.duplicates_merged == 0
    assert memory._sqlite_vectors.count(collection=memory.collection_name) == 3


@pytest.mark.asyncio
async def test_configured_scope_filters_mismatches_instead_of_relabeling_them(tmp_path):
    memory = _SQLiteMemory(tmp_path / "vectors.sqlite3")
    memory.put("p1", "same", [1.0, 0.0], _meta(principal="p1", namespace="n1"))
    memory.put("p2", "same", [1.0, 0.0], _meta(principal="p2", namespace="n1"))
    memory.put("n2", "same", [1.0, 0.0], _meta(principal="p1", namespace="n2"))

    report = await MemoryConsolidator(
        memory,
        principal_id="p1",
        namespace="n1",
        similarity_threshold=0.99,
    ).consolidate()

    assert report.skipped_reasons == {
        "namespace_scope_mismatch": 1,
        "principal_scope_mismatch": 1,
    }
    assert report.duplicates_merged == 0
    assert memory._sqlite_vectors.count(collection=memory.collection_name) == 3


@pytest.mark.asyncio
async def test_governance_holds_and_invalid_embeddings_are_excluded(tmp_path):
    memory = _SQLiteMemory(tmp_path / "vectors.sqlite3")
    memory.put("held", "held", [1.0, 0.0], _meta(legal_hold=True))
    memory.put("secret", "secret", [1.0, 0.0], _meta(sensitivity="secret"))
    memory.put("zero", "zero", [0.0, 0.0], _meta())
    memory.put("good", "good", [1.0, 0.0], _meta())

    report = await MemoryConsolidator(memory).consolidate()

    assert report.records_skipped == 3
    assert report.skipped_reasons == {
        "governance_hold": 1,
        "sensitive_memory_class": 1,
        "zero_norm_embedding": 1,
    }
    assert report.duplicates_merged == 0


def test_sqlite_atomic_merge_rolls_back_on_revision_conflict(tmp_path):
    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3", collection_name="c")
    store.upsert("a", "a", [1.0, 0.0], metadata=_meta())
    store.upsert("b", "b", [1.0, 0.0], metadata=_meta())
    rows = {row["id"]: row for row in store.iter_records(collection="c")}
    expected = {key: float(row["updated_at"]) for key, row in rows.items()}
    expected["b"] -= 1.0

    with pytest.raises(RuntimeError, match="revision changed"):
        store.merge_records_atomic(
            winner_id="a",
            loser_ids=["b"],
            content="merged",
            embedding=[1.0, 0.0],
            metadata=_meta(),
            expected_updated_at=expected,
            collection="c",
        )

    assert {row["id"] for row in store.list_records(collection="c")} == {"a", "b"}


def test_sqlite_atomic_delete_rolls_back_on_revision_conflict(tmp_path):
    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3", collection_name="c")
    store.upsert("a", "a", [1.0, 0.0], metadata=_meta())
    store.upsert("b", "b", [1.0, 0.0], metadata=_meta())
    rows = {row["id"]: row for row in store.iter_records(collection="c")}
    expected = {key: float(row["updated_at"]) for key, row in rows.items()}
    expected["b"] -= 1.0

    with pytest.raises(RuntimeError, match="revision changed"):
        store.delete_records_atomic(
            record_ids=["a", "b"],
            expected_updated_at=expected,
            collection="c",
        )

    assert {row["id"] for row in store.list_records(collection="c")} == {"a", "b"}


@pytest.mark.asyncio
async def test_chroma_failure_restores_every_record_and_clears_journal(tmp_path):
    records = [
        _record("a", "alpha", [1.0, 0.0]),
        _record("b", "beta", [1.0, 0.0]),
    ]
    memory = _ChromaMemory(tmp_path, records)
    memory._collection.fail_delete_once = True

    report = await MemoryConsolidator(memory, similarity_threshold=0.99).consolidate()

    assert report.transactions_committed == 0
    assert report.transactions_rolled_back == 1
    assert report.duplicates_merged == 0
    assert set(memory._collection.records) == {"a", "b"}
    assert memory._collection.records["a"]["content"] == "alpha"
    assert memory._collection.records["b"]["content"] == "beta"
    assert not list(tmp_path.glob("*.consolidation-rollback.json"))
    assert any("cluster_merge_failed" in error for error in report.errors)


@pytest.mark.asyncio
async def test_chroma_success_verifies_winner_and_removes_journal(tmp_path):
    records = [
        _record("a", "alpha", [1.0, 0.0], _meta(source_link="episode://a")),
        _record("b", "beta", [1.0, 0.0], _meta(source_link="episode://b")),
    ]
    memory = _ChromaMemory(tmp_path, records)

    report = await MemoryConsolidator(memory, similarity_threshold=0.99).consolidate()

    assert report.transactions_committed == 1
    assert report.duplicates_merged == 1
    assert len(memory._collection.records) == 1
    winner = next(iter(memory._collection.records.values()))
    assert "alpha" in winner["content"] and "beta" in winner["content"]
    provenance = json.loads(winner["metadata"]["consolidation_provenance_json"])
    assert {item["metadata"]["source_link"] for item in provenance} == {
        "episode://a",
        "episode://b",
    }
    assert not list(tmp_path.glob("*.consolidation-rollback.json"))


@pytest.mark.asyncio
async def test_pending_chroma_journal_is_recovered_before_scan(tmp_path):
    records = [
        _record("a", "alpha", [1.0, 0.0]),
        _record("b", "beta", [-1.0, 0.0]),
    ]
    memory = _ChromaMemory(tmp_path, records)
    consolidator = MemoryConsolidator(memory, similarity_threshold=0.99)
    originals = list(consolidator._get_chroma_records(["a", "b"]).values())
    path = consolidator._journal_path()
    assert path is not None
    consolidator._write_journal(path, "crash-receipt", originals)
    memory._collection.records.pop("b")

    report = await consolidator.consolidate()

    assert report.transactions_rolled_back == 1
    assert set(memory._collection.records) == {"a", "b"}
    assert report.duplicates_merged == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_unsupported_backend_is_not_reported_as_a_merge():
    report = await MemoryConsolidator(object(), allow_unscoped=True).consolidate()

    assert report.scan_complete is False
    assert report.duplicates_merged == 0
    assert report.transactions_committed == 0
    assert "unsupported_memory_backend" in report.errors


@pytest.mark.asyncio
async def test_black_hole_vault_uses_native_backend_not_chroma_alias():
    memory = _BlackHoleMemory(
        [
            _black_hole_record("a", "alpha shared fact"),
            _black_hole_record("b", "alpha shared fact"),
        ]
    )

    consolidator = MemoryConsolidator(memory, similarity_threshold=0.99)
    assert consolidator._backend_kind() == "black_hole_vault"

    report = await consolidator.consolidate()

    assert report.backend == "black_hole_vault"
    assert report.scan_complete is True
    assert report.transactions_committed == 1
    assert report.duplicates_merged == 1
    assert len(memory.memories) == 1
    assert memory.memories[0]["metadata"]["consolidation_receipt_id"]
    assert memory.saved


@pytest.mark.asyncio
async def test_black_hole_vault_merge_rolls_back_after_persistence_failure():
    records = [
        _black_hole_record("a", "alpha shared fact"),
        _black_hole_record("b", "alpha shared fact"),
    ]
    memory = _BlackHoleMemory(records)
    memory.fail_save_once = True

    report = await MemoryConsolidator(
        memory,
        similarity_threshold=0.99,
    ).consolidate()

    assert report.transactions_committed == 0
    assert report.transactions_rolled_back == 1
    assert memory.memories == records
    assert memory.saved[-1] == records


def test_collection_alias_without_complete_chroma_contract_is_unsupported():
    class CompatibilityAlias:
        _sqlite_vectors = None

        def __init__(self):
            self._collection = self

        def get(self, **_kwargs):
            return {}

    assert MemoryConsolidator(
        CompatibilityAlias(), allow_unscoped=True
    )._backend_kind() == "unsupported"


@pytest.mark.asyncio
async def test_consolidation_work_runs_off_the_event_loop(monkeypatch):
    consolidator = MemoryConsolidator(object(), allow_unscoped=True)
    caller_thread = threading.get_ident()
    observed = []

    def _probe():
        observed.append(threading.get_ident())
        from core.brain.cognitive.memory_management import ConsolidationReport

        return ConsolidationReport(scan_complete=True)

    monkeypatch.setattr(consolidator, "_consolidate_sync", _probe)
    await consolidator.consolidate()

    assert observed and observed[0] != caller_thread


def test_new_vector_memories_stamp_metric_version_and_namespace(monkeypatch, tmp_path):
    import core.memory.vector_memory as module

    monkeypatch.setattr(module, "_CHROMA_AVAILABLE", False)
    monkeypatch.setattr("core.utils.core_db.get_core_db", lambda: object())
    from core.memory.vector_memory import VectorMemory

    memory = VectorMemory(collection_name="scope", persist_directory=str(tmp_path))
    try:
        assert memory.add_memory("hello", {"principal_id": "p1"}, _id="id1") is True
        record = memory._sqlite_vectors.list_records(collection="scope")[0]
        assert record["metadata"]["memory_namespace"] == "scope"
        assert record["metadata"]["embedding_metric"] == "cosine"
        assert record["metadata"]["embedding_version"]
    finally:
        memory.close()


def test_fallback_add_does_not_expose_memory_when_persistence_fails(monkeypatch, tmp_path):
    import core.memory.vector_memory as module

    monkeypatch.setattr(module, "_CHROMA_AVAILABLE", False)
    monkeypatch.setattr("core.utils.core_db.get_core_db", lambda: object())
    from core.memory.vector_memory import VectorMemory

    memory = VectorMemory(collection_name="scope", persist_directory=str(tmp_path))
    try:
        monkeypatch.setattr(memory, "_upsert_fallback", lambda *_args, **_kwargs: False)
        assert memory.add_memory("not durable", _id="id1") is False
        assert memory._store == []
    finally:
        memory.close()


def test_chroma_prune_holds_mutation_lease_through_selection_and_delete():
    from core.memory.vector_memory import VectorMemory

    class GuardedCollection:
        def __init__(self, lock):
            self.lock = lock

        def get(self, *, include):
            assert self.lock._is_owned()
            return {"ids": ["stale"], "metadatas": [{"timestamp": 0.0, "valence": -1.0}]}

        def delete(self, *, ids):
            assert self.lock._is_owned()
            assert ids == ["stale"]

    memory = VectorMemory.__new__(VectorMemory)
    memory._fallback_mode = False
    memory._mutation_lock = threading.RLock()
    memory.collection_name = "scope"
    memory._collection = GuardedCollection(memory._mutation_lock)

    assert memory.prune_low_salience(threshold_days=1) == 1
