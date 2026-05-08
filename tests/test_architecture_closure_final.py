from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.collective.belief_sync import BeliefSync
from core.collective.delegator import AgentDelegator
from core.memory.sqlite_vector_store import SQLiteVectorStore
from core.memory.vector_memory import VectorMemory
from core.memory.vector_memory_engine import Memory, MemoryVault
from core.will import ActionDomain, UnifiedWill, WillOutcome


def test_sqlite_vector_store_migrates_legacy_json_without_plaintext_vectors(tmp_path):
    legacy = tmp_path / "long_term.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "id": "m1",
                    "text": "Bryan wants governed memory writes.",
                    "vector": [1.0, 0.0, 0.0],
                    "timestamp": 10.0,
                    "metadata": {"kind": "test"},
                },
                {
                    "id": "m2",
                    "text": "Aura should fail closed on unsafe effects.",
                    "vector": [0.0, 1.0, 0.0],
                    "timestamp": 11.0,
                    "metadata": {"kind": "test"},
                },
            ]
        )
    )
    db = tmp_path / "vectors.sqlite3"
    store = SQLiteVectorStore(db, collection_name="long_term")

    assert store.migrate_legacy_json(legacy) == 2
    assert store.count() == 2

    results = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), limit=1)
    assert results[0].id == "m1"
    raw = db.read_bytes()
    assert b"1.0, 0.0, 0.0" not in raw


def test_vector_memory_fallback_uses_persistent_sqlite_vectors(monkeypatch, tmp_path):
    monkeypatch.setattr("core.memory.vector_memory._CHROMA_AVAILABLE", False)
    memory = VectorMemory(collection_name="test", persist_directory=str(tmp_path))
    assert memory.add_memory("governance receipt memory", _id="receipt-memory")

    reloaded = VectorMemory(collection_name="test", persist_directory=str(tmp_path))
    results = reloaded.search_similar("governance receipt", limit=1)

    assert results
    assert results[0]["id"] == "receipt-memory"
    assert reloaded.get_stats()["engine"] == "sqlite_vector"


def test_memory_vault_sqlite_fallback_persists_across_instances(tmp_path):
    vault = MemoryVault(str(tmp_path / "vault"))
    vault._collection = None
    if vault._sqlite_vectors is None:
        vault._sqlite_vectors = SQLiteVectorStore(
            tmp_path / "vault" / "vectors.sqlite3",
            collection_name=vault.collection_name,
        )
    memory = Memory(
        id="m1",
        content="persistent vector vault memory",
        memory_type="episodic",
        timestamp=1.0,
        importance=0.7,
    )
    vault.store(memory, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    reloaded = MemoryVault(str(tmp_path / "vault"))
    reloaded._collection = None
    if reloaded._sqlite_vectors is None:
        reloaded._sqlite_vectors = SQLiteVectorStore(
            tmp_path / "vault" / "vectors.sqlite3",
            collection_name=reloaded.collection_name,
        )
    results = reloaded.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), n_results=1)

    assert results[0][0] == "m1"


@pytest.mark.asyncio
async def test_belief_sync_deduplicates_semantic_principle_paraphrases(tmp_path, service_container):
    class _Abs:
        def __init__(self):
            self.storage_path = tmp_path / "principles.json"
            self.storage_path.write_text(
                json.dumps(
                    {
                        "payload": [
                            {
                                "principle": "Memory writes must carry an authorized Will receipt."
                            }
                        ]
                    }
                )
            )
            self.commits = []

        async def _commit_principle(self, principle):
            self.commits.append(principle)

    abs_engine = _Abs()
    service_container.register_instance("abstraction_engine", abs_engine)
    sync = BeliefSync(SimpleNamespace(peers={}))

    await sync.handle_incoming_principles(
        {
            "origin": "peer",
            "principles": [
                {"principle": "Memory write operations should include authorized will receipts."}
            ],
        }
    )

    assert abs_engine.commits == []


def test_swarm_throttles_parallelism_from_integrity_pressure(service_container):
    report = SimpleNamespace(cpu_percent=91.0, memory_percent=40.0, thermal_level=0)
    monitor = SimpleNamespace(_last_report=report)
    service_container.register_instance("integrity_monitor", monitor)

    delegator = AgentDelegator(SimpleNamespace(cognitive_engine=None))

    assert delegator.effective_max_parallel() == 1
    assert "Deterministic swarm consensus" in delegator._deterministic_consensus(
        "topic",
        ['{"claim":"Use SQLite vectors","confidence":0.8,"flaws":["migration needed"]}'],
    )


def test_will_signs_receipts_and_allows_reserved_self_repair_under_catatonia():
    will = UnifiedWill()
    now = time.time()
    for idx in range(10):
        will._audit_trail.append(
            SimpleNamespace(
                outcome=WillOutcome.REFUSE,
                timestamp=now - idx,
            )
        )
    will._state.total_decisions = 10
    will._state.refuses = 10

    decision = will.decide(
        "repair the Will circuit breaker",
        source="self_repair",
        domain=ActionDomain.STATE_MUTATION,
        priority=0.9,
        context={
            "catatonia_relief": True,
            "unity_override": "repair_only",
        },
    )

    assert decision.outcome == WillOutcome.CONSTRAIN
    assert any("catatonia_relief" in constraint for constraint in decision.constraints)
    assert decision.signature
    assert will.verify_receipt_signature(decision.receipt_id) is True
