"""Tests for the tamper-evident audit chain over receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runtime.audit_chain import (
    AuditChain,
    ChainEntry,
    GENESIS_PREV_HASH,
    canonical_json,
    hash_receipt_body,
)
from core.runtime.receipts import (
    GovernanceReceipt,
    OutputReceipt,
    TurnReceipt,
    get_receipt_store,
    reset_receipt_store,
)


def _fresh_store(tmp_path: Path):
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    return store


def _emit_three(store) -> list:
    out = []
    out.append(store.emit(TurnReceipt(cause="t1", origin="test")))
    out.append(
        store.emit(GovernanceReceipt(cause="t2", domain="test", action="x", approved=True))
    )
    out.append(store.emit(OutputReceipt(cause="t3", origin="o", target="t", digest="d")))
    return out


# ---------------------------------------------------------------------------
# canonical hashing
# ---------------------------------------------------------------------------
def test_canonical_json_is_deterministic():
    a = {"b": 1, "a": [3, 2, 1], "c": {"y": 2, "x": 1}}
    b = {"a": [3, 2, 1], "c": {"x": 1, "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_hash_receipt_body_is_stable():
    body = {"receipt_id": "r-1", "kind": "turn", "cause": "x", "created_at": 1.0}
    h1 = hash_receipt_body(body)
    h2 = hash_receipt_body(dict(body))
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_entry_hash_changes_with_any_field():
    base = dict(
        seq=0,
        receipt_id="r-1",
        kind="turn",
        content_hash="sha256:" + "0" * 64,
        timestamp=1.0,
        prev_hash=GENESIS_PREV_HASH,
    )
    h0 = ChainEntry.compute_entry_hash(**base)
    for field in base:
        mutated = dict(base)
        if isinstance(base[field], str):
            mutated[field] = base[field] + "x"
        elif isinstance(base[field], (int, float)):
            mutated[field] = base[field] + 1
        h = ChainEntry.compute_entry_hash(**mutated)
        assert h != h0, f"entry hash did not change when {field} changed"


# ---------------------------------------------------------------------------
# chain construction via ReceiptStore.emit
# ---------------------------------------------------------------------------
def test_emit_appends_one_entry_per_receipt(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)
    result = store.verify_chain()
    assert result["ok"] is True
    assert result["length"] == 3
    assert result["problems"] == []


def test_chain_head_advances_monotonically(tmp_path):
    store = _fresh_store(tmp_path)
    receipts = _emit_three(store)
    chain = store._chain
    entries = chain.entries()
    assert [e.seq for e in entries] == [0, 1, 2]
    assert [e.receipt_id for e in entries] == [r.receipt_id for r in receipts]
    # Genesis link
    assert entries[0].prev_hash == GENESIS_PREV_HASH
    # Each subsequent prev_hash equals the previous entry_hash
    for prev, curr in zip(entries, entries[1:]):
        assert curr.prev_hash == prev.entry_hash


def test_chain_persists_across_restart(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)
    head_a = store._chain.head_hash()

    # Simulate process restart: drop the singleton, reopen the chain.
    reset_receipt_store()
    store2 = get_receipt_store(tmp_path / "receipts")
    head_b = store2._chain.head_hash()
    assert head_a == head_b
    assert store2._chain.length() == 3
    # Emitting again continues the chain rather than forking.
    store2.emit(TurnReceipt(cause="t4", origin="restart"))
    result = store2.verify_chain()
    assert result["ok"] is True
    assert result["length"] == 4


# ---------------------------------------------------------------------------
# tamper detection
# ---------------------------------------------------------------------------
def test_detects_modified_receipt_body(tmp_path):
    store = _fresh_store(tmp_path)
    receipts = _emit_three(store)
    target = receipts[1]
    body_path = tmp_path / "receipts" / target.kind / f"{target.receipt_id}.json"
    env = json.loads(body_path.read_text(encoding="utf-8"))
    env["payload"]["action"] = "TAMPERED"
    body_path.write_text(json.dumps(env), encoding="utf-8")

    result = store.verify_chain()
    assert result["ok"] is False
    reasons = [p["reason"] for p in result["problems"]]
    assert any("content_hash mismatch" in r for r in reasons)


def test_detects_modified_chain_entry(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)
    chain_path = tmp_path / "receipts" / AuditChain.CHAIN_FILENAME
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    # Mutate the timestamp of the second entry without recomputing entry_hash.
    record = json.loads(lines[1])
    record["timestamp"] = record["timestamp"] + 999.0
    lines[1] = json.dumps(record, sort_keys=True)
    chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = store.verify_chain()
    assert result["ok"] is False
    reasons = [p["reason"] for p in result["problems"]]
    assert any("entry_hash mismatch" in r for r in reasons)


def test_detects_broken_link(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)
    chain_path = tmp_path / "receipts" / AuditChain.CHAIN_FILENAME
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    # Tamper the prev_hash of entry 2; entry_hash stays the old (now invalid)
    # value, which surfaces as both a broken link and an entry mismatch.
    record = json.loads(lines[2])
    record["prev_hash"] = "sha256:" + "1" * 64
    lines[2] = json.dumps(record, sort_keys=True)
    chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = store.verify_chain()
    assert result["ok"] is False
    reasons = [p["reason"] for p in result["problems"]]
    assert any("broken chain link" in r for r in reasons)


def test_detects_deleted_entry(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)
    chain_path = tmp_path / "receipts" / AuditChain.CHAIN_FILENAME
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    # Drop the middle entry to create a seq gap.
    chain_path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")

    result = store.verify_chain()
    assert result["ok"] is False
    reasons = [p["reason"] for p in result["problems"]]
    assert any("out-of-order or missing seq" in r for r in reasons)


def test_detects_missing_receipt_body(tmp_path):
    store = _fresh_store(tmp_path)
    receipts = _emit_three(store)
    target = receipts[0]
    body_path = tmp_path / "receipts" / target.kind / f"{target.receipt_id}.json"
    body_path.unlink()

    result = store.verify_chain()
    assert result["ok"] is False
    reasons = [p["reason"] for p in result["problems"]]
    assert any("body missing" in r for r in reasons)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def test_export_produces_portable_bundle(tmp_path):
    store = _fresh_store(tmp_path)
    _emit_three(store)

    dest = tmp_path / "audit_export"
    info = store.export_chain(dest)

    assert (dest / "chain.jsonl").exists()
    assert (dest / "MANIFEST.txt").exists()
    assert info["length"] == 3
    assert info["head_hash"] == store._chain.head_hash()

    # Manifest contains the head hash and length so an offline auditor
    # can verify by simply re-reading the chain.
    manifest = (dest / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "head_hash=" in manifest
    assert "length=3" in manifest

    # Exported chain.jsonl is byte-identical to the live chain.
    src = (tmp_path / "receipts" / AuditChain.CHAIN_FILENAME).read_bytes()
    dst = (dest / "chain.jsonl").read_bytes()
    assert src == dst


def test_exported_chain_can_be_independently_verified(tmp_path):
    """An auditor with only the exported file rebuilds an AuditChain
    pointing at it and runs verify().  No receipt bodies are available
    in this scenario, so we only verify the chain links themselves."""
    store = _fresh_store(tmp_path)
    _emit_three(store)
    dest = tmp_path / "audit_export"
    store.export_chain(dest)

    # Reuse AuditChain on a directory that contains only chain.jsonl
    # under the canonical name.
    audit_root = tmp_path / "audit_root"
    audit_root.mkdir()
    (audit_root / AuditChain.CHAIN_FILENAME).write_bytes(
        (dest / "chain.jsonl").read_bytes()
    )
    chain = AuditChain(audit_root)
    ok, problems = chain.verify()
    assert ok is True
    assert problems == []
    assert chain.length() == 3
