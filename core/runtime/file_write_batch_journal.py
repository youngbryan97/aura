"""The crash-recovery journal for a directory-scoped multi-file write.

A directory batch stages every payload beside its target, records what it is
about to do, then renames. If the process dies between the first rename and
the last, the directory holds a half-applied batch, and this is what puts it
back: the journal names each target, whether it existed, its mode, and the
digest of its original bytes.

Recovery is a separate concern from writing — it runs when nobody is writing —
and it reads better on its own than buried in the gateway.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from core.runtime import file_write_primitives as _primitives
from core.runtime.file_write_primitives import (
    DIRECTORY_BATCH_INTERNAL_PREFIX,
    DIRECTORY_BATCH_JOURNAL_FILE,
    DIRECTORY_BATCH_JOURNAL_SCHEMA,
    MAX_DIRECTORY_BATCH_JOURNAL_BYTES,
    FileWriteTransactionError,
    read_regular_at,
    validated_flat_name,
)

__all__ = [
    "canonical_journal_bytes",
    "write_directory_batch_journal",
    "load_directory_batch_journal",
    "cleanup_abandoned_directory_batch_staging",
    "recover_directory_batch",
]


def canonical_journal_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def write_directory_batch_journal(
    directory_fd: int,
    *,
    transaction_id: str,
    entries: list[dict[str, Any]],
    state: str,
) -> None:
    if state not in {"rollback_required", "committed"}:
        raise ValueError("directory batch journal state is invalid")
    body = {
        "schema": DIRECTORY_BATCH_JOURNAL_SCHEMA,
        "transaction_id": transaction_id,
        "state": state,
        "entries": entries,
    }
    journal = {
        **body,
        "journal_sha256": hashlib.sha256(
            canonical_journal_bytes(body)
        ).hexdigest(),
    }
    payload = canonical_journal_bytes(journal)
    if not payload or len(payload) > MAX_DIRECTORY_BATCH_JOURNAL_BYTES:
        raise FileWriteTransactionError(
            "directory batch journal exceeds durable recovery bound"
        )
    temporary = (
        f"{DIRECTORY_BATCH_INTERNAL_PREFIX}{transaction_id}-journal.tmp"
    )
    try:
        _primitives.stage_bytes_at(
            directory_fd,
            temporary,
            payload,
            0o600,
        )
        os.replace(
            temporary,
            DIRECTORY_BATCH_JOURNAL_FILE,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        _primitives.unlink_private_regular_at(directory_fd, temporary)


def load_directory_batch_journal(
    directory_fd: int,
) -> dict[str, Any]:
    payload, mode = read_regular_at(
        directory_fd,
        DIRECTORY_BATCH_JOURNAL_FILE,
    )
    if (
        mode != 0o600
        or not payload
        or len(payload) > MAX_DIRECTORY_BATCH_JOURNAL_BYTES
    ):
        raise FileWriteTransactionError("directory batch journal invalid")
    try:
        journal = json.loads(payload)
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise FileWriteTransactionError(
            "directory batch journal is not valid JSON"
        ) from exc
    fields = {
        "schema",
        "transaction_id",
        "state",
        "entries",
        "journal_sha256",
    }
    if (
        not isinstance(journal, dict)
        or set(journal) != fields
        or journal.get("schema") != DIRECTORY_BATCH_JOURNAL_SCHEMA
        or not isinstance(journal.get("transaction_id"), str)
        or len(journal["transaction_id"]) != 32
        or any(
            character not in "0123456789abcdef"
            for character in journal["transaction_id"]
        )
        or journal.get("state") not in {"rollback_required", "committed"}
        or not isinstance(journal.get("entries"), list)
        or not journal["entries"]
        or len(journal["entries"]) > 10_000
    ):
        raise FileWriteTransactionError("directory batch journal invalid")
    body = dict(journal)
    observed_sha256 = body.pop("journal_sha256")
    if (
        not isinstance(observed_sha256, str)
        or hashlib.sha256(canonical_journal_bytes(body)).hexdigest()
        != observed_sha256
    ):
        raise FileWriteTransactionError(
            "directory batch journal commitment invalid"
        )
    transaction_id = journal["transaction_id"]
    seen_targets: set[str] = set()
    entry_fields = {
        "target",
        "temporary",
        "backup",
        "original_exists",
        "original_mode",
        "original_sha256",
    }
    for index, entry in enumerate(journal["entries"]):
        expected_temporary = (
            f"{DIRECTORY_BATCH_INTERNAL_PREFIX}{transaction_id}-{index}.tmp"
        )
        expected_backup = (
            f"{DIRECTORY_BATCH_INTERNAL_PREFIX}{transaction_id}-{index}.bak"
        )
        try:
            target = (
                validated_flat_name(entry.get("target"))
                if isinstance(entry, dict)
                else ""
            )
        except (TypeError, ValueError) as exc:
            raise FileWriteTransactionError(
                "directory batch journal entry invalid"
            ) from exc
        if (
            not isinstance(entry, dict)
            or set(entry) != entry_fields
            or target in seen_targets
            or entry.get("temporary") != expected_temporary
            or entry.get("backup") != expected_backup
            or type(entry.get("original_exists")) is not bool
            or (
                entry["original_exists"]
                and (
                    type(entry.get("original_mode")) is not int
                    or entry["original_mode"] & ~0o777
                    or not isinstance(entry.get("original_sha256"), str)
                    or len(entry["original_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in entry["original_sha256"]
                    )
                )
            )
            or (
                not entry["original_exists"]
                and (
                    entry.get("original_mode") is not None
                    or entry.get("original_sha256") is not None
                )
            )
        ):
            raise FileWriteTransactionError(
                "directory batch journal entry invalid"
            )
        seen_targets.add(target)
    return journal


def cleanup_abandoned_directory_batch_staging(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
        if name.startswith(DIRECTORY_BATCH_INTERNAL_PREFIX):
            _primitives.unlink_private_regular_at(directory_fd, name)
    os.fsync(directory_fd)


def recover_directory_batch(directory_fd: int) -> None:
    journal = load_directory_batch_journal(directory_fd)
    if journal["state"] == "committed":
        _primitives.unlink_private_regular_at(
            directory_fd,
            DIRECTORY_BATCH_JOURNAL_FILE,
        )
        os.fsync(directory_fd)
        cleanup_abandoned_directory_batch_staging(directory_fd)
        return
    transaction_id = journal["transaction_id"]
    for index, entry in enumerate(journal["entries"]):
        target = entry["target"]
        if entry["original_exists"]:
            backup_payload, backup_mode = read_regular_at(
                directory_fd,
                entry["backup"],
            )
            if backup_mode != entry["original_mode"]:
                raise FileWriteTransactionError(
                    "directory batch backup mode mismatch"
                )
            if (
                hashlib.sha256(backup_payload).hexdigest()
                != entry["original_sha256"]
            ):
                raise FileWriteTransactionError(
                    "directory batch backup commitment mismatch"
                )
            recovery_name = (
                f"{DIRECTORY_BATCH_INTERNAL_PREFIX}{transaction_id}-"
                f"recover-{index}.tmp"
            )
            _primitives.unlink_private_regular_at(directory_fd, recovery_name)
            _primitives.stage_bytes_at(
                directory_fd,
                recovery_name,
                backup_payload,
                backup_mode,
            )
            os.replace(
                recovery_name,
                target,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            _primitives.unlink_private_regular_at(directory_fd, target)
    os.fsync(directory_fd)
    _primitives.unlink_private_regular_at(
        directory_fd,
        DIRECTORY_BATCH_JOURNAL_FILE,
    )
    os.fsync(directory_fd)
    cleanup_abandoned_directory_batch_staging(directory_fd)
