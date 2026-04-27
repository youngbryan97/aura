"""Canonical AtomicWriter — single durable-write gateway.

Postgres-grade durability requires that every persistent write is:

    write to a temp file in the same directory
    flush + fsync the file
    fsync the parent directory
    atomic rename over the target

Crash points (between any of those steps) must leave the target either
unchanged (old committed state) or fully replaced (new committed state).

This module exposes:

- atomic_write_bytes(path, payload)
- atomic_write_text(path, text)
- atomic_write_json(path, obj, schema_version)

with explicit schema-version envelopes so loaders can detect ancient
records and refuse rather than silently misread.
"""
from __future__ import annotations


import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("Aura.AtomicWriter")

PathLike = Union[str, Path]

DEFAULT_TEMP_PREFIX = ".aura_atomic_"


class AtomicWriteError(RuntimeError):
    """Raised when an atomic write cannot complete."""


def _fsync_file(fd: int) -> None:
    try:
        os.fsync(fd)
    except (AttributeError, OSError):
        # Best-effort on platforms where fsync is unavailable.
        pass


def _fsync_dir(directory: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except (FileNotFoundError, PermissionError, OSError):
        return
    try:
        _fsync_file(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: PathLike, payload: bytes) -> None:
    """Atomically replace `path` with `payload`."""
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix=DEFAULT_TEMP_PREFIX, dir=str(parent))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            _fsync_file(fh.fileno())
        os.replace(tmp_path, target)
        _fsync_dir(parent)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    schema_version: int,
    schema_name: Optional[str] = None,
    indent: Optional[int] = 2,
) -> None:
    """Atomically write a JSON envelope `{schema, version, payload}`."""
    if not isinstance(schema_version, int) or schema_version < 1:
        raise AtomicWriteError("schema_version must be a positive int")
    envelope = {
        "schema": schema_name or path.__class__.__name__,
        "schema_version": schema_version,
        "payload": obj,
    }
    text = json.dumps(envelope, indent=indent, sort_keys=True, default=str)
    atomic_write_text(path, text)


def read_json_envelope(path: PathLike) -> dict:
    target = Path(path)
    raw = target.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "schema_version" not in data:
        raise AtomicWriteError(
            f"file at {target} is not a versioned envelope (missing schema_version)"
        )
    return data


def cleanup_partial_writes(directory: PathLike) -> int:
    """Remove leftover temp files from interrupted writes. Returns count."""
    parent = Path(directory)
    if not parent.exists():
        return 0
    removed = 0
    for child in parent.iterdir():
        if child.name.startswith(DEFAULT_TEMP_PREFIX):
            try:
                child.unlink()
                removed += 1
            except OSError:
                continue
    return removed
