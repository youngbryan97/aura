from __future__ import annotations

import json
import sqlite3

import pytest

from core.runtime.receipts import (
    OutputReceipt,
    ReceiptStore,
    ResourceAdmissionReceipt,
    StateMutationReceipt,
    close_receipt_store,
    get_receipt_store,
    reset_receipt_store,
)


def _admission_receipt(index: int) -> ResourceAdmissionReceipt:
    return ResourceAdmissionReceipt(
        cause="test:background",
        request_id=f"request-{index}",
        owner="test.background",
        work_class="background",
        lane="maintenance",
        decision="deferred",
        reason="memory_pressure",
    )


def test_high_volume_receipts_use_compact_ledger_and_bounded_hot_index(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_RECEIPT_HOT_INDEX_LIMIT", "64")
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    emitted = [store.emit(_admission_receipt(index)) for index in range(80)]

    stats = store.storage_stats()
    assert stats["high_volume_ledger_available"] is True
    assert stats["ledger_by_kind"]["resource_admission"] == 80
    assert stats["hot_by_kind"]["resource_admission"] == 64
    assert not (root / "resource_admission").exists()

    oldest_id = emitted[0].receipt_id
    assert oldest_id not in store._index
    assert store.get(oldest_id).request_id == "request-0"
    assert store.storage_stats()["hot_by_kind"]["resource_admission"] == 64
    assert store.verify_chain()["ok"] is True
    store.close()

    reloaded = ReceiptStore(root)
    assert reloaded.reload_from_disk() == 64
    assert reloaded.get(oldest_id).request_id == "request-0"
    assert reloaded.coverage_stats()["resource_admission"] == 80
    assert reloaded.verify_chain()["ok"] is True
    reloaded.close()


def test_output_receipts_use_compact_ledger_and_survive_restart(tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    receipt = store.emit(
        OutputReceipt(
            cause="chat_response",
            origin="api",
            target="primary",
            digest="sha256:" + "a" * 64,
        )
    )

    assert store.storage_stats()["ledger_by_kind"]["output"] == 1
    assert not (root / "output").exists()
    assert store.get(receipt.receipt_id).digest == receipt.digest
    assert store.verify_chain()["ok"] is True
    store.close()

    reloaded = ReceiptStore(root)
    assert reloaded.reload_from_disk() == 1
    assert reloaded.get(receipt.receipt_id).digest == receipt.digest
    assert reloaded.coverage_stats()["output"] == 1
    assert reloaded.verify_chain()["ok"] is True
    reloaded.close()


def test_ordinary_receipts_keep_schema_envelope_files(tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    receipt = store.emit(
        StateMutationReceipt(
            cause="test",
            domain="runtime",
            key="mode",
        )
    )

    path = root / "state_mutation" / f"{receipt.receipt_id}.json"
    assert path.exists()
    assert not (root / "resource_admission").exists()
    assert store.verify_chain()["ok"] is True
    store.close()


def test_ordinary_receipts_use_bounded_hot_index_and_cold_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_RECEIPT_HOT_INDEX_LIMIT", "64")
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    emitted = [
        store.emit(
            StateMutationReceipt(
                cause="test",
                domain="runtime",
                key=f"key-{index}",
            )
        )
        for index in range(80)
    ]

    oldest_id = emitted[0].receipt_id
    assert store.storage_stats()["hot_by_kind"]["state_mutation"] == 64
    assert store.coverage_stats()["state_mutation"] == 80
    assert oldest_id not in store._index
    assert store.get(oldest_id).key == "key-0"
    assert store.storage_stats()["hot_by_kind"]["state_mutation"] == 64
    store.close()

    reloaded = ReceiptStore(root)
    assert reloaded.reload_from_disk() == 64
    assert reloaded.get(oldest_id).key == "key-0"
    assert reloaded.coverage_stats()["state_mutation"] == 80
    reloaded.close()


def test_receipt_ids_are_idempotent_but_immutable_after_emit(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    receipt = StateMutationReceipt(
        receipt_id="state-fixed",
        cause="test",
        domain="runtime",
        key="mode",
    )

    store.emit(receipt)
    chain_length = store.verify_chain()["length"]
    store.emit(receipt)
    assert store.verify_chain()["length"] == chain_length

    receipt.key = "mutated-after-emit"
    with pytest.raises(ValueError, match="immutable"):
        store.emit(receipt)
    assert store.get("state-fixed").key == "mode"
    store.close()


def test_high_volume_ledger_tampering_breaks_audit_chain(tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    receipt = store.emit(_admission_receipt(1))

    connection = sqlite3.connect(root / "_high_volume_receipts.sqlite3")
    try:
        body = receipt.to_dict()
        body["reason"] = "tampered"
        connection.execute(
            "UPDATE receipt_ledger SET body_json = ? WHERE receipt_id = ?",
            (
                json.dumps(body, sort_keys=True, separators=(",", ":")),
                receipt.receipt_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    result = store.verify_chain()
    assert result["ok"] is False
    assert any(
        str(problem.get("reason", "")).startswith("content_hash mismatch")
        for problem in result["problems"]
    )
    store.close()


def test_ledger_initialization_failure_falls_back_to_envelope_files(
    monkeypatch,
    tmp_path,
):
    from core.runtime import receipts

    def unavailable(*_args, **_kwargs):
        raise sqlite3.OperationalError("ledger unavailable")

    monkeypatch.setattr(receipts.sqlite3, "connect", unavailable)
    monkeypatch.setenv("AURA_RECEIPT_HOT_INDEX_LIMIT", "64")
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    emitted = [store.emit(_admission_receipt(index)) for index in range(70)]
    oldest = emitted[0]

    assert store.storage_stats()["high_volume_ledger_available"] is False
    assert (root / "resource_admission" / f"{oldest.receipt_id}.json").exists()
    assert oldest.receipt_id not in store._index
    assert store.get(oldest.receipt_id).request_id == "request-0"
    assert store.coverage_stats()["resource_admission"] == 70
    assert store.verify_chain()["ok"] is True
    store.close()


def test_runtime_ledger_write_failure_falls_back_to_envelope(monkeypatch, tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)

    def fail_write(_body):
        raise sqlite3.OperationalError("runtime ledger failure")

    monkeypatch.setattr(store, "_ledger_put_locked", fail_write)
    receipt = store.emit(_admission_receipt(12))

    assert store.storage_stats()["high_volume_ledger_available"] is False
    assert (root / "resource_admission" / f"{receipt.receipt_id}.json").exists()
    assert store.get(receipt.receipt_id).request_id == "request-12"
    assert store.verify_chain()["ok"] is True
    store.close()


def test_receipt_store_close_is_terminal_and_idempotent(tmp_path):
    store = ReceiptStore(tmp_path / "receipts")
    store.emit(_admission_receipt(1))

    store.close()
    store.close()

    assert store._ledger is None
    assert store._chain is None
    assert store._ledger_available is False
    with pytest.raises(RuntimeError, match="receipt store is closed"):
        store.emit(_admission_receipt(2))


def test_global_receipt_store_close_does_not_construct_store(tmp_path):
    reset_receipt_store()
    assert close_receipt_store() == {
        "clean": True,
        "closed": False,
        "reason": "not_initialized",
    }
    get_receipt_store(tmp_path / "receipts")
    assert close_receipt_store() == {"clean": True, "closed": True}
    reset_receipt_store()


def test_verify_chain_detects_persisted_cold_receipt_without_chain_entry(tmp_path):
    root = tmp_path / "receipts"
    store = ReceiptStore(root)
    store.emit(_admission_receipt(1))
    unchained = _admission_receipt(99)
    unchained.receipt_id = "resource_admission-unchained"
    body = unchained.to_dict()

    connection = sqlite3.connect(root / "_high_volume_receipts.sqlite3")
    try:
        connection.execute(
            """
            INSERT INTO receipt_ledger(receipt_id, kind, created_at, body_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                unchained.receipt_id,
                unchained.kind,
                unchained.created_at,
                json.dumps(body, sort_keys=True, separators=(",", ":")),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    result = store.verify_chain()

    assert result["ok"] is False
    assert unchained.receipt_id in result["missing_from_chain"]
    assert any(
        problem.get("receipt_id") == unchained.receipt_id
        and problem.get("reason") == "receipt missing from audit chain"
        for problem in result["problems"]
    )
    store.close()


def test_diagnostics_reads_recent_receipts_from_compact_ledger_after_restart(tmp_path):
    from core.runtime.diagnostics_bundle import collect_recent_receipts

    reset_receipt_store()
    try:
        root = tmp_path / "receipts"
        store = get_receipt_store(root)
        receipt = store.emit(_admission_receipt(7))
        store.close()
        reset_receipt_store()
        restarted = get_receipt_store(root)

        payload = collect_recent_receipts(per_kind_limit=5)

        assert payload["counts"]["resource_admission"] == 1
        assert payload["recent"]["resource_admission"][0]["receipt_id"] == (
            receipt.receipt_id
        )
        assert payload["storage"]["ledger_by_kind"]["resource_admission"] == 1
        restarted.close()
    finally:
        reset_receipt_store()
