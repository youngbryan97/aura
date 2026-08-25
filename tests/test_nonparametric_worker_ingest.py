from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from core.brain import (
    nonparametric_generation,
    nonparametric_ingest,
    nonparametric_memory,
)
from core.brain.llm import mlx_client, mlx_worker
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.nonparametric_ingest import NonParametricIngestor
from core.brain.nonparametric_memory import NonParametricMemory
from core.utils.deadlines import get_deadline
from tests.nonparametric_support import entry_provenance


class _Process:
    def is_alive(self) -> bool:
        return True


class _BatchEncoder:
    dim = 8
    calls = 0

    def __init__(self, _model, _tokenizer) -> None:
        self.model = _model
        self.tokenizer = _tokenizer

    def encode_tokens(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def encode_hidden_sequence_ids(self, ids: list[int]) -> np.ndarray:
        type(self).calls += 1
        values = np.zeros((len(ids), self.dim), dtype=np.float32)
        for index in range(len(ids)):
            values[index, index % self.dim] = 1.0
        return values

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        raise AssertionError(f"quadratic prefix encoder used for {ids}")


def test_worker_ingest_reuses_resident_model_with_one_forward(tmp_path, monkeypatch):
    memory = NonParametricMemory(dim=8, path=tmp_path / "memory")
    _BatchEncoder.calls = 0
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", _BatchEncoder)
    monkeypatch.setattr(
        nonparametric_ingest,
        "collect_trusted_pairs_by_source",
        lambda *, limit: [("entries", [("keeper name", "Tessaly the great")])],
    )
    monkeypatch.setattr(
        nonparametric_ingest,
        "NonParametricIngestor",
        lambda target, *, provenance: NonParametricIngestor(
            target,
            provenance=provenance,
            dedup_path=tmp_path / "seen.json",
        ),
    )
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda dim: memory if dim == 8 else None,
    )

    language = SimpleNamespace(
        args=SimpleNamespace(hidden_size=8),
        model=lambda value: value,
    )
    result = mlx_worker._run_nonparametric_ingest_job(
        SimpleNamespace(
            args=SimpleNamespace(model_type="qwen3_5"),
            language_model=language,
        ),
        object(),
        {"seq": 7, "max_pairs": 1, "deadline_s": 5.0},
    )

    assert result["state"] == "complete"
    assert result["pairs_ingested"] == 1
    assert result["positions_ingested"] == 3
    assert _BatchEncoder.calls == 1
    assert len(memory) == 3
    # every entry names the store it came from, so a discredited trusted
    # store can be withdrawn without erasing the rest
    assert memory.revoke_source("trusted_store:entries") == 3


def test_worker_ingest_skips_seen_prefix_and_uses_next_pair(tmp_path, monkeypatch):
    memory = NonParametricMemory(dim=8, path=tmp_path / "memory")
    dedup_path = tmp_path / "seen.json"
    encoder = _BatchEncoder(object(), object())
    seed = NonParametricIngestor(memory, dedup_path=dedup_path, provenance=entry_provenance())
    assert seed.ingest_sequence("already known", "first answer", encoder) == 2
    _BatchEncoder.calls = 0
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", _BatchEncoder)
    monkeypatch.setattr(
        nonparametric_ingest,
        "collect_trusted_pairs_by_source",
        lambda *, limit: [
            (
                "entries",
                [
                    ("already known", "first answer"),
                    ("new keeper", "second answer"),
                ],
            )
        ],
    )
    monkeypatch.setattr(
        nonparametric_ingest,
        "NonParametricIngestor",
        lambda target, *, provenance: NonParametricIngestor(
            target, provenance=provenance, dedup_path=dedup_path
        ),
    )
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda _dim: memory,
    )

    result = mlx_worker._run_nonparametric_ingest_job(
        SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
        object(),
        {"seq": 8, "max_pairs": 1, "deadline_s": 5.0},
    )

    assert result["state"] == "complete"
    assert result["pairs_scanned"] == 1
    assert result["pairs_ingested"] == 1
    assert _BatchEncoder.calls == 1


def test_worker_ingest_skips_oversized_prefix_and_uses_eligible_pair(tmp_path, monkeypatch):
    memory = NonParametricMemory(dim=8, path=tmp_path / "memory")
    _BatchEncoder.calls = 0
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", _BatchEncoder)
    monkeypatch.setattr(
        nonparametric_ingest,
        "collect_trusted_pairs_by_source",
        lambda *, limit: [
            (
                "entries",
                [
                    ("oversized context", " ".join(["token"] * 40)),
                    ("eligible context", "short answer"),
                ],
            )
        ],
    )
    monkeypatch.setattr(
        nonparametric_ingest,
        "NonParametricIngestor",
        lambda target, *, provenance: NonParametricIngestor(
            target,
            provenance=provenance,
            dedup_path=tmp_path / "seen.json",
        ),
    )
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda _dim: memory,
    )

    result = mlx_worker._run_nonparametric_ingest_job(
        SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
        object(),
        {
            "seq": 9,
            "max_pairs": 1,
            "scan_limit": 1,
            "max_sequence_tokens": 8,
            "deadline_s": 5.0,
        },
    )

    assert result["state"] == "complete"
    assert result["pairs_considered"] == 2
    assert result["pairs_scanned"] == 1
    assert result["pairs_ingested"] == 1
    assert _BatchEncoder.calls == 1


def test_worker_does_not_publish_dedup_receipt_before_memory_persist(tmp_path, monkeypatch):
    memory = NonParametricMemory(dim=8, path=tmp_path / "memory")
    dedup_path = tmp_path / "seen.json"
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", _BatchEncoder)
    monkeypatch.setattr(
        nonparametric_ingest,
        "collect_trusted_pairs_by_source",
        lambda *, limit: [("entries", [("keeper name", "Tessaly")])],
    )
    monkeypatch.setattr(
        nonparametric_ingest,
        "NonParametricIngestor",
        lambda target, *, provenance: NonParametricIngestor(
            target, provenance=provenance, dedup_path=dedup_path
        ),
    )
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda _dim: memory,
    )
    monkeypatch.setattr(memory, "persist", lambda: False)

    with pytest.raises(RuntimeError, match="nonparametric_memory_persist_failed"):
        mlx_worker._run_nonparametric_ingest_job(
            SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
            object(),
            {"seq": 9, "max_pairs": 1, "deadline_s": 5.0},
        )

    assert dedup_path.exists() is False


def test_worker_ingest_observes_soft_cancel_before_model_forward(tmp_path, monkeypatch):
    memory = NonParametricMemory(dim=8, path=tmp_path / "memory")
    _BatchEncoder.calls = 0
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", _BatchEncoder)
    monkeypatch.setattr(
        nonparametric_ingest,
        "collect_trusted_pairs_by_source",
        lambda *, limit: [("entries", [("keeper name", "Tessaly")])],
    )
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda _dim: memory,
    )

    result = mlx_worker._run_nonparametric_ingest_job(
        SimpleNamespace(args=SimpleNamespace(hidden_size=8)),
        object(),
        {"seq": 9, "deadline_s": 5.0},
        cancel_seq=SimpleNamespace(value=9),
    )

    assert result["state"] == "soft_cancelled"
    assert result["positions_ingested"] == 0
    assert _BatchEncoder.calls == 0


@pytest.mark.asyncio
async def test_client_ingest_uses_resident_worker_and_releases_ownership(monkeypatch):
    client = MLXLocalClient(model_path="/models/test-32b")
    client._process = _Process()
    client._init_done = True
    client._req_q = queue.Queue()
    monkeypatch.setattr(
        mlx_client,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    task = asyncio.create_task(
        client.ingest_nonparametric_async(
            max_pairs=1,
            max_positions=32,
            max_sequence_tokens=64,
            timeout_s=5.0,
        )
    )
    request = await asyncio.to_thread(client._req_q.get, True, 2.0)
    future = client._pending_generations[request["id"]]
    mlx_client._set_shared_future_result(
        future,
        {
            "id": request["id"],
            "action": "nonparametric_ingest",
            "status": "ok",
            "state": "complete",
            "pairs_scanned": 1,
            "pairs_ingested": 1,
            "positions_ingested": 7,
        },
    )

    result = await task

    assert request["action"] == "nonparametric_ingest"
    assert request["max_sequence_tokens"] == 64
    assert result["status"] == "complete"
    assert result["positions_ingested"] == 7
    assert client._active_generations == 0
    assert client._current_request_id == ""
    assert client._request_lock.locked() is False


@pytest.mark.asyncio
async def test_foreground_preempts_nonparametric_maintenance():
    client = MLXLocalClient(model_path="/models/test-32b")
    assert client._request_lock.acquire(False) is True
    client._request_lock_owner_label = "reasoning_nonparametric_ingest"
    client._request_lock_acquired_at = 1.0
    client._mark_generation_started("maintenance", request_seq=17)

    acquire = asyncio.create_task(
        client._acquire_request_lock(
            owner_label="desktop_user",
            deadline=get_deadline(1.0),
            foreground_request=True,
        )
    )
    for _attempt in range(100):
        if int(client._cancel_seq.value) == 17:
            break
        await asyncio.sleep(0.01)
    assert int(client._cancel_seq.value) == 17
    client._release_request_lock()
    assert await acquire is True
    client._release_request_lock()
