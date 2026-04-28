"""Canonical persistence ownership helpers for Aura.

Goal: every durable runtime write should be atomic, schema-versioned when
possible, and easy to audit. This module wraps Aura's canonical AtomicWriter
when present and falls back to temp+replace when imported in test/minimal
contexts.
"""
from __future__ import annotations


import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass  # no-op: intentional


def atomic_write_text_owned(
    path: str | Path,
    text: str,
    *,
    schema_name: str = "text",
    schema_version: int = 1,
    encoding: str = "utf-8",
) -> None:
    """Durably write text via temp+fsync+replace.

    Use for non-JSON durable state. For JSON, prefer atomic_write_json_owned.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer canonical writer if it supports text; many Aura versions only
    # expose JSON, so fallback remains important.
    try:
        from core.runtime.atomic_writer import atomic_write_text  # type: ignore
        atomic_write_text(path, text, schema_version=schema_version, schema_name=schema_name)
        return
    except Exception:
        pass  # no-op: intentional

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass  # no-op: intentional


def atomic_write_json_owned(
    path: str | Path,
    payload: Any,
    *,
    schema_name: str,
    schema_version: int = 1,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
) -> None:
    """Durably write JSON with Aura's AtomicWriter when available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from core.runtime.atomic_writer import atomic_write_json
        atomic_write_json(
            path,
            payload,
            schema_version=schema_version,
            schema_name=schema_name,
        )
        return
    except Exception:
        pass  # no-op: intentional

    envelope = {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "payload": payload,
    }
    text = json.dumps(envelope, indent=indent, sort_keys=sort_keys, default=repr)
    atomic_write_text_owned(
        path,
        text,
        schema_name=schema_name,
        schema_version=schema_version,
    )


def emit_persistence_receipt(
    *,
    path: str | Path,
    cause: str,
    family: str = "runtime",
    record_id: str = "",
    bytes_written: Optional[int] = None,
    schema_version: int = 1,
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Emit a MemoryWriteReceipt if Aura's receipt store is live.

    Returns True when a receipt was emitted, False otherwise.
    """
    path = Path(path)
    try:
        from core.runtime.receipts import MemoryWriteReceipt, get_receipt_store
        import uuid

        if bytes_written is None:
            try:
                bytes_written = path.stat().st_size
            except Exception:
                bytes_written = 0

        get_receipt_store().emit(
            MemoryWriteReceipt(
                receipt_id=f"memwr-{uuid.uuid4()}",
                cause=cause,
                family=family,
                record_id=record_id or path.name,
                bytes_written=int(bytes_written or 0),
                schema_version=schema_version,
                metadata={"path": str(path), **dict(metadata or {})},
            )
        )
        return True
    except Exception:
        return False


def write_json_with_receipt(
    path: str | Path,
    payload: Any,
    *,
    schema_name: str,
    cause: str,
    family: str = "runtime",
    record_id: str = "",
    schema_version: int = 1,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Atomic JSON write + optional MemoryWriteReceipt."""
    path = Path(path)
    atomic_write_json_owned(
        path,
        payload,
        schema_name=schema_name,
        schema_version=schema_version,
    )
    emit_persistence_receipt(
        path=path,
        cause=cause,
        family=family,
        record_id=record_id or path.name,
        schema_version=schema_version,
        metadata=metadata,
    )
