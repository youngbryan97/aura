"""core/runtime/file_write_gateway.py — Canonical File Write Gateway.

All file writing operations should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import stat
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime import file_write_primitives as _primitives
from core.runtime.atomic_writer import (
    PathLike,
    atomic_append_text,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    durable_replace,
    durable_unlink,
    interprocess_file_lock,
)

# Extracted so this module owns one job. The primitives work against open
# directory descriptors and know nothing about governance; the journal is what
# puts a half-applied directory batch back after a crash. Aliased to the names
# the gateway already used, so no call site below changed.
from core.runtime.file_write_batch_journal import (
    cleanup_abandoned_directory_batch_staging as _cleanup_abandoned_directory_batch_staging,
)
from core.runtime.file_write_batch_journal import (
    load_directory_batch_journal as _load_directory_batch_journal,
)
from core.runtime.file_write_batch_journal import (
    recover_directory_batch as _recover_directory_batch,
)
from core.runtime.file_write_batch_journal import (
    write_directory_batch_journal as _write_directory_batch_journal,
)
from core.runtime.file_write_primitives import (
    DIRECTORY_BATCH_INTERNAL_PREFIX as _DIRECTORY_BATCH_INTERNAL_PREFIX,
)
from core.runtime.file_write_primitives import (
    DIRECTORY_BATCH_JOURNAL_FILE as _DIRECTORY_BATCH_JOURNAL_FILE,
)
from core.runtime.file_write_primitives import (
    DIRECTORY_BATCH_LOCK_FILE as _DIRECTORY_BATCH_LOCK_FILE,
)
from core.runtime.file_write_primitives import (
    DIRECTORY_BATCH_THREAD_LOCK as _DIRECTORY_BATCH_THREAD_LOCK,
)
from core.runtime.file_write_primitives import (
    FileWriteTransactionError,
)
from core.runtime.file_write_primitives import (
    assert_lock_binding as _assert_lock_binding,
)
from core.runtime.file_write_primitives import (
    exists_at as _exists_at,
)
from core.runtime.file_write_primitives import (
    open_directory_no_follow as _open_directory_no_follow,
)
from core.runtime.file_write_primitives import (
    read_regular_at as _read_regular_at,
)
from core.runtime.file_write_primitives import (
    replace_symlink_unchecked as _replace_symlink_unchecked,
)
from core.runtime.file_write_primitives import (
    required_no_follow_flag as _required_no_follow_flag,
)
from core.runtime.file_write_primitives import (
    validated_flat_name as _validated_flat_name,
)
from core.runtime.file_write_primitives import (
    validated_permissions as _validated_permissions,
)
from core.runtime.state_ownership import assert_state_path_allowed

logger = logging.getLogger("Aura.FileWriteGateway")
_BinaryAdapter = TypeVar("_BinaryAdapter")
_FILE_WRITE_DOMAINS = (
    "file_write",
    "memory_write",
    "state_mutation",
    "self_modification",
    "tool_execution",
)


@dataclass(frozen=True, slots=True)
class FileWriteBatchEntry:
    path: PathLike
    payload: bytes
    mode: int = 0o600


@dataclass(frozen=True, slots=True)
class DirectoryFileWriteBatchEntry:
    """One flat filename published relative to an opened directory inode."""

    name: str
    payload: bytes
    mode: int = 0o600


@dataclass(frozen=True, slots=True)
class FileWriteBatchReceipt:
    transaction_id: str
    paths: tuple[str, ...]
    sha256: tuple[tuple[str, str], ...]


@contextmanager
def _not_on_the_loop(what: str) -> Iterator[None]:
    """Record a synchronous disk write that happened on the loop's thread.

    The async lane below exists because an on-loop fsync froze the live loop
    for about twenty minutes under thrash. The rule was written in a guide and
    a ratchet test catches new sync writes inside `async def`; neither can see
    a sync writer called FROM async through two intermediate frames. This can:
    it asks which thread it is actually on.

    Recorded, never raised. A write that reaches here has already passed
    governance and the caller is expecting it to happen.
    """
    try:
        from core.runtime.which_thread_may_do_this import AKindOfWork, this_is
    except ImportError:  # pragma: no cover - foundation import order
        yield
        return
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, what):
        yield


class FileWriteGateway:
    """Single canonical owner for filesystem write operations."""

    def __init__(self) -> None:
        self._allowed_domains = _FILE_WRITE_DOMAINS

    def ensure_directory(self, path: PathLike, *, source: str = "unknown") -> str:
        """Create a private directory through the governed filesystem lane."""

        directory = Path(path).expanduser()
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.ensure_directory:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import ensure_private_directory

        return str(ensure_private_directory(directory))

    async def ensure_directory_async(self, path: PathLike, *, source: str = "unknown") -> str:
        """Create a private directory off the event loop after inline governance."""

        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.ensure_directory:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_ensure_private_directory

        return str(await async_ensure_private_directory(path))

    def write_bytes(self, path: PathLike, payload: bytes, *, source: str = "unknown") -> None:
        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_write_bytes(target, bytes(payload))

    def write_bytes_batch(
        self,
        entries: Sequence[FileWriteBatchEntry],
        *,
        source: str = "unknown",
    ) -> FileWriteBatchReceipt:
        """Commit a same-directory file set with exception rollback.

        Each target replacement is independently crash-atomic and durable. If
        an ordinary write fails, already replaced targets are restored before
        the error escapes. A process/power loss can still occur between file
        replacements, so consumers of a coupled set must validate their mutual
        consistency on read before using it.
        """

        batch = tuple(entries)
        if not batch:
            raise ValueError("file batch must contain at least one entry")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes_batch:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        normalized: list[tuple[Path, bytes, int]] = []
        seen: set[Path] = set()
        parent: Path | None = None
        for entry in batch:
            target = _coerce_target(entry.path)
            if target.is_symlink():
                raise FileWriteTransactionError(
                    f"refusing batch replacement through symlink: {target}"
                )
            if not isinstance(entry.payload, (bytes, bytearray, memoryview)):
                raise TypeError("batch payloads must be bytes-like")
            resolved_parent = target.parent.resolve()
            if parent is None:
                parent = resolved_parent
            elif parent != resolved_parent:
                raise ValueError("all batch targets must share one directory")
            absolute_target = resolved_parent / target.name
            if absolute_target in seen:
                raise ValueError(f"duplicate batch target: {target}")
            seen.add(absolute_target)
            normalized.append(
                (
                    absolute_target,
                    bytes(entry.payload),
                    _validated_permissions(entry.mode),
                )
            )

        assert parent is not None
        from core.runtime.atomic_writer import ensure_private_directory

        ensure_private_directory(parent)
        lock_path = parent / ".aura_file_write_batch.lock"
        transaction_id = uuid.uuid4().hex
        with interprocess_file_lock(lock_path):
            originals: dict[Path, tuple[bytes, int] | None] = {}
            for target, _payload, _mode in normalized:
                if target.exists():
                    if not target.is_file():
                        raise FileWriteTransactionError(
                            f"batch target is not a regular file: {target}"
                        )
                    originals[target] = (
                        target.read_bytes(),
                        stat.S_IMODE(target.stat().st_mode),
                    )
                else:
                    originals[target] = None

            committed: list[Path] = []
            try:
                for target, payload, mode in normalized:
                    committed.append(target)
                    atomic_write_bytes(target, payload, mode=mode)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                rollback_failures: list[str] = []
                for target in reversed(committed):
                    try:
                        original = originals[target]
                        if original is None:
                            durable_unlink(target, missing_ok=True)
                        else:
                            atomic_write_bytes(
                                target,
                                original[0],
                                mode=original[1],
                            )
                    except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                        rollback_failures.append(
                            f"{target}:{type(rollback_exc).__name__}:{rollback_exc}"
                        )
                detail = (
                    f"; rollback failures={rollback_failures}"
                    if rollback_failures
                    else "; prior targets restored"
                )
                raise FileWriteTransactionError(
                    f"file batch {transaction_id} did not commit{detail}"
                ) from exc

        paths = tuple(str(target) for target, _payload, _mode in normalized)
        hashes = tuple(
            (str(target), hashlib.sha256(payload).hexdigest())
            for target, payload, _mode in normalized
        )
        return FileWriteBatchReceipt(
            transaction_id=transaction_id,
            paths=paths,
            sha256=hashes,
        )

    def write_bytes_batch_in_directory(
        self,
        directory: PathLike,
        entries: Sequence[DirectoryFileWriteBatchEntry],
        *,
        allowed_existing_names: Collection[str],
        commit_marker: str,
        source: str = "unknown",
    ) -> FileWriteBatchReceipt:
        """Publish a flat file set through one no-follow directory descriptor."""

        batch = tuple(entries)
        if not batch:
            raise ValueError("directory file batch must not be empty")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes_batch_in_directory:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        normalized: list[tuple[str, bytes, int]] = []
        seen: set[str] = set()
        for entry in batch:
            if not isinstance(entry, DirectoryFileWriteBatchEntry):
                raise TypeError(
                    "directory batch entries must be "
                    "DirectoryFileWriteBatchEntry"
                )
            name = _validated_flat_name(entry.name)
            if name in seen:
                raise ValueError(f"duplicate directory batch target: {name}")
            if not isinstance(entry.payload, (bytes, bytearray, memoryview)):
                raise TypeError("directory batch payloads must be bytes-like")
            seen.add(name)
            normalized.append(
                (
                    name,
                    bytes(entry.payload),
                    _validated_permissions(entry.mode),
                )
            )
        marker = _validated_flat_name(commit_marker)
        if marker not in seen:
            raise ValueError("commit marker must be one of the batch entries")
        allowed = {
            _validated_flat_name(name)
            for name in allowed_existing_names
        }
        if allowed != seen:
            raise ValueError(
                "allowed existing names must exactly match batch names"
            )
        ordered = [
            *[row for row in normalized if row[0] != marker],
            *[row for row in normalized if row[0] == marker],
        ]
        directory_fd, lexical = _open_directory_no_follow(directory)
        bound = os.fstat(directory_fd)
        transaction_id = uuid.uuid4().hex
        transaction_names: set[str] = set()
        originals: dict[str, tuple[bytes, int] | None] = {}
        lock_fd: int | None = None
        commit_boundary_crossed = False
        try:
            with _DIRECTORY_BATCH_THREAD_LOCK:
                lock_fd = os.open(
                    _DIRECTORY_BATCH_LOCK_FILE,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_CLOEXEC
                    | _required_no_follow_flag(),
                    0o600,
                    dir_fd=directory_fd,
                )
                _assert_lock_binding(directory_fd, lock_fd)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                _assert_lock_binding(directory_fd, lock_fd)
                if _exists_at(
                    directory_fd,
                    _DIRECTORY_BATCH_JOURNAL_FILE,
                ):
                    _recover_directory_batch(directory_fd)
                else:
                    _cleanup_abandoned_directory_batch_staging(
                        directory_fd
                    )
                inventory = set(os.listdir(directory_fd))
                unexpected = inventory - allowed - {
                    _DIRECTORY_BATCH_LOCK_FILE
                }
                if unexpected:
                    raise FileWriteTransactionError(
                        "directory batch contains unexpected entries: "
                        + ",".join(sorted(unexpected))
                    )
                for name in inventory - {_DIRECTORY_BATCH_LOCK_FILE}:
                    metadata = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_nlink != 1
                        or metadata.st_uid != os.getuid()
                    ):
                        raise FileWriteTransactionError(
                            "directory batch contains non-regular or "
                            f"non-private entry: {name}"
                        )
                journal_entries: list[dict[str, Any]] = []
                for name, _payload, _mode in ordered:
                    if name in inventory:
                        original_payload, original_mode = _read_regular_at(
                            directory_fd,
                            name,
                        )
                        original_exists = True
                        originals[name] = (
                            original_payload,
                            original_mode,
                        )
                    else:
                        original_payload = b""
                        original_mode = None
                        original_exists = False
                        originals[name] = None
                    index = len(journal_entries)
                    temporary = (
                        f"{_DIRECTORY_BATCH_INTERNAL_PREFIX}"
                        f"{transaction_id}-{index}.tmp"
                    )
                    backup = (
                        f"{_DIRECTORY_BATCH_INTERNAL_PREFIX}"
                        f"{transaction_id}-{index}.bak"
                    )
                    journal_entries.append(
                        {
                            "target": name,
                            "temporary": temporary,
                            "backup": backup,
                            "original_exists": original_exists,
                            "original_mode": original_mode,
                            "original_sha256": (
                                hashlib.sha256(original_payload).hexdigest()
                                if original_exists
                                else None
                            ),
                        }
                    )
                try:
                    for journal_entry, (_name, payload, mode) in zip(
                        journal_entries,
                        ordered,
                        strict=True,
                    ):
                        temporary = journal_entry["temporary"]
                        transaction_names.add(temporary)
                        _primitives.stage_bytes_at(
                            directory_fd,
                            temporary,
                            payload,
                            mode,
                        )
                        if journal_entry["original_exists"]:
                            original = originals[journal_entry["target"]]
                            if original is None:
                                raise FileWriteTransactionError(
                                    "directory batch original disappeared"
                                )
                            original_payload, original_mode = original
                            backup = journal_entry["backup"]
                            transaction_names.add(backup)
                            _primitives.stage_bytes_at(
                                directory_fd,
                                backup,
                                original_payload,
                                original_mode,
                            )
                    _write_directory_batch_journal(
                        directory_fd,
                        transaction_id=transaction_id,
                        entries=journal_entries,
                        state="rollback_required",
                    )
                    for journal_entry in journal_entries:
                        os.replace(
                            journal_entry["temporary"],
                            journal_entry["target"],
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                        )
                        transaction_names.discard(journal_entry["temporary"])
                    os.fsync(directory_fd)
                    final_inventory = set(os.listdir(directory_fd))
                    expected_internal = {
                        _DIRECTORY_BATCH_LOCK_FILE,
                        _DIRECTORY_BATCH_JOURNAL_FILE,
                        *(
                            journal_entry["backup"]
                            for journal_entry in journal_entries
                            if journal_entry["original_exists"]
                        ),
                    }
                    if final_inventory != allowed | expected_internal:
                        raise FileWriteTransactionError(
                            "directory batch final inventory mismatch"
                        )
                    for name, payload, mode in ordered:
                        observed, observed_mode = _read_regular_at(
                            directory_fd,
                            name,
                        )
                        if observed != payload or observed_mode != mode:
                            raise FileWriteTransactionError(
                                f"directory batch verification failed: {name}"
                            )
                    reopened_fd, _reopened_path = _open_directory_no_follow(
                        lexical
                    )
                    try:
                        reopened = os.fstat(reopened_fd)
                        if (
                            reopened.st_dev,
                            reopened.st_ino,
                        ) != (bound.st_dev, bound.st_ino):
                            raise FileWriteTransactionError(
                                "directory path changed during batch commit"
                            )
                    finally:
                        os.close(reopened_fd)
                    _assert_lock_binding(directory_fd, lock_fd)
                    _write_directory_batch_journal(
                        directory_fd,
                        transaction_id=transaction_id,
                        entries=journal_entries,
                        state="committed",
                    )
                    commit_boundary_crossed = True
                    _recover_directory_batch(directory_fd)
                    transaction_names.clear()
                except BaseException as exc:
                    rollback_failures: list[str] = []
                    try:
                        if _exists_at(
                            directory_fd,
                            _DIRECTORY_BATCH_JOURNAL_FILE,
                        ):
                            durable_state = _load_directory_batch_journal(
                                directory_fd
                            )["state"]
                            if durable_state == "committed":
                                commit_boundary_crossed = True
                            _recover_directory_batch(directory_fd)
                        else:
                            for name in tuple(transaction_names):
                                _primitives.unlink_private_regular_at(
                                    directory_fd,
                                    name,
                                )
                            os.fsync(directory_fd)
                    except BaseException as rollback_exc:  # noqa: BLE001 - aggregated into rollback_failures, which raises below
                        rollback_failures.append(
                            f"{type(rollback_exc).__name__}:{rollback_exc}"
                        )
                    try:
                        _assert_lock_binding(directory_fd, lock_fd)
                    except BaseException as lock_exc:  # noqa: BLE001 - aggregated into rollback_failures, which raises below
                        rollback_failures.append(
                            f"lock:{type(lock_exc).__name__}:{lock_exc}"
                        )
                    if rollback_failures:
                        if commit_boundary_crossed:
                            raise FileWriteTransactionError(
                                f"directory file batch {transaction_id} "
                                "committed but durable cleanup is incomplete: "
                                + ";".join(rollback_failures)
                            ) from exc
                        raise FileWriteTransactionError(
                            "directory batch rollback incomplete: "
                            + ";".join(rollback_failures)
                        ) from exc
                    if commit_boundary_crossed:
                        raise FileWriteTransactionError(
                            f"directory file batch {transaction_id} committed "
                            "but durable cleanup did not complete"
                        ) from exc
                    if isinstance(exc, Exception):
                        raise FileWriteTransactionError(
                            f"directory file batch {transaction_id} "
                            "did not commit; originals restored"
                        ) from exc
                    raise
                finally:
                    for name in tuple(transaction_names):
                        try:
                            _primitives.unlink_private_regular_at(
                                directory_fd,
                                name,
                            )
                        except FileWriteTransactionError:
                            pass
                    if lock_fd is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(directory_fd)
        paths = tuple(str(lexical / name) for name, _payload, _mode in normalized)
        hashes = tuple(
            (
                str(lexical / name),
                hashlib.sha256(payload).hexdigest(),
            )
            for name, payload, _mode in normalized
        )
        return FileWriteBatchReceipt(
            transaction_id=transaction_id,
            paths=paths,
            sha256=hashes,
        )

    def write_text(
        self,
        path: PathLike,
        text: str,
        *,
        encoding: str = "utf-8",
        source: str = "unknown",
        durable: bool = True,
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        with _not_on_the_loop(f"write_text:{source}"):
            atomic_write_text(target, text, encoding=encoding, durable=durable)

    def append_text(self, path: PathLike, text: str, *, encoding: str = "utf-8", source: str = "unknown") -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        with _not_on_the_loop(f"append_text:{source}"):
            atomic_append_text(target, text, encoding=encoding)

    # ── Event-loop-safe lane ─────────────────────────────────────────
    # Governance is checked inline (fail fast, caller's context); only the
    # blocking disk write is offloaded. Async callers must use these — an
    # on-loop fsync froze the live event loop for ~20 minutes under thrash.

    async def write_text_async(
        self,
        path: PathLike,
        text: str,
        *,
        encoding: str = "utf-8",
        source: str = "unknown",
        durable: bool = True,
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_text

        await async_atomic_write_text(target, text, encoding=encoding, durable=durable)

    async def write_bytes_async(
        self, path: PathLike, payload: bytes, *, source: str = "unknown", durable: bool = True
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_bytes

        await async_atomic_write_bytes(target, bytes(payload), durable=durable)

    def write_bytes_if_absent(
        self,
        path: PathLike,
        payload: bytes,
        *,
        mode: int = 0o600,
        source: str = "unknown",
        durable: bool = True,
    ) -> bool:
        """Atomically publish bytes once without replacing an existing target."""

        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes_if_absent:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        return atomic_write_bytes_if_absent(
            target,
            bytes(payload),
            durable=durable,
            mode=mode,
        )

    def provision_private_bytes(
        self,
        path: PathLike,
        candidate: bytes,
        *,
        expected_size: int,
        mode: int = 0o600,
        source: str = "unknown",
    ) -> bytes:
        """Create or adopt one stable, owner-only byte identity.

        The winner is read through an owner-private directory descriptor with
        ``O_NOFOLLOW`` and stable inode checks.  A competing process may win the
        create race, but a symlink, hardlink, replacement, wrong mode, or wrong
        size can never become accepted key material.
        """

        target = _coerce_target(path)
        if not isinstance(candidate, (bytes, bytearray, memoryview)):
            raise TypeError("candidate must be bytes-like")
        payload = bytes(candidate)
        if type(expected_size) is not int or expected_size <= 0:
            raise ValueError("expected_size must be a positive integer")
        if len(payload) != expected_size:
            raise ValueError("candidate length does not match expected_size")
        permissions = _validated_permissions(mode)
        if permissions & 0o077:
            raise ValueError("private byte identities must be owner-only")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.provision_private_bytes:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import (
            atomic_write_bytes_if_absent,
            ensure_private_directory,
        )

        ensure_private_directory(target.parent)
        atomic_write_bytes_if_absent(
            target,
            payload,
            durable=True,
            mode=permissions,
        )
        directory_fd, _lexical = _open_directory_no_follow(target.parent)
        try:
            authoritative, observed_mode = _read_regular_at(
                directory_fd,
                target.name,
                max_bytes=expected_size,
            )
        finally:
            os.close(directory_fd)
        if len(authoritative) != expected_size:
            raise FileWriteTransactionError(
                f"private byte identity has invalid size: {target}"
            )
        if observed_mode != permissions:
            raise FileWriteTransactionError(
                f"private byte identity has invalid permissions: {target}"
            )
        return authoritative

    async def write_bytes_if_absent_async(
        self,
        path: PathLike,
        payload: bytes,
        *,
        mode: int = 0o600,
        source: str = "unknown",
        durable: bool = True,
    ) -> bool:
        """Atomically publish bytes without replacing an existing target."""

        target = _coerce_target(path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_bytes_if_absent:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_bytes_if_absent

        return await async_atomic_write_bytes_if_absent(
            target,
            bytes(payload),
            durable=durable,
            mode=mode,
        )

    def open_owned_binary(
        self,
        path: PathLike,
        *,
        mode: str,
        permissions: int = 0o600,
        source: str = "unknown",
    ) -> BinaryIO:
        """Open a process-owned mutable binary file through governance.

        This narrow primitive exists for `mmap` rings and process-lifetime lock
        files that cannot use replace-on-write persistence. Symlinks and
        arbitrary mode strings are rejected.
        """

        target = _coerce_target(path)
        if target.is_symlink():
            raise OSError(f"refusing owned binary open through symlink: {target}")
        flag_by_mode = {
            "a+b": os.O_RDWR | os.O_CREAT | os.O_APPEND,
            "w+b": os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            "r+b": os.O_RDWR,
        }
        if mode not in flag_by_mode:
            raise ValueError(f"unsupported owned binary mode: {mode!r}")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.open_owned_binary:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import ensure_private_directory

        ensure_private_directory(target.parent)
        flags = flag_by_mode[mode]
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(target), flags, _validated_permissions(permissions))
        try:
            os.fchmod(fd, _validated_permissions(permissions))
            return cast("BinaryIO", os.fdopen(fd, mode))
        except (OSError, ValueError):
            os.close(fd)
            raise

    @contextmanager
    def open_owned_binary_adapter(
        self,
        path: PathLike,
        *,
        mode: str,
        adapter: Callable[..., _BinaryAdapter],
        adapter_kwargs: Mapping[str, Any] | None = None,
        permissions: int = 0o600,
        source: str = "unknown",
    ) -> Iterator[_BinaryAdapter]:
        """Own a write-capable library adapter and its underlying file.

        Media/database libraries often need a mutable file object and expose
        their own ``open`` function. Letting each caller invoke that second
        opener makes effect ownership ambiguous and scatters flush/close
        ordering. This lane gives the adapter only an already-governed handle,
        closes the adapter first, then durably flushes the underlying file.
        """

        if not callable(adapter):
            raise TypeError("binary adapter must be callable")
        kwargs = dict(adapter_kwargs or {})
        with self.open_owned_binary(
            path,
            mode=mode,
            permissions=permissions,
            source=source,
        ) as handle:
            wrapped = adapter(handle, **kwargs)
            try:
                yield wrapped
            finally:
                close = getattr(wrapped, "close", None)
                if callable(close):
                    close()
                if not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())

    def replace_file(
        self,
        path: PathLike,
        destination: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Durably replace a file through the governed synchronous lane."""

        src = _coerce_target(path)
        dst = _coerce_target(destination)
        if src.is_symlink() or dst.is_symlink():
            raise OSError("refusing durable replacement involving a symlink")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_file:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        durable_replace(src, dst)
        return str(dst)

    def move_path(
        self,
        path: PathLike,
        destination: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Durably move a file or directory through the synchronous lane."""

        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if src.is_symlink() or dst.is_symlink():
            raise OSError("refusing durable move involving a symlink")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.move_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        durable_replace(src, dst)
        return str(dst)

    def copy_path(
        self,
        path: PathLike,
        destination: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Copy a file or directory through the governed synchronous lane.

        This counterpart to :meth:`copy_path_async` is for offline tooling and
        detached workers. Runtime coroutines must use the async form so a large
        artifact copy cannot block the event loop.
        """

        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if src.is_symlink() or dst.is_symlink():
            raise OSError("refusing governed copy involving a symlink")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.copy_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        if not os.path.lexists(src):
            raise FileNotFoundError(f"copy source does not exist: {src}")
        import shutil

        if src.is_dir():
            return str(shutil.copytree(str(src), str(dst), symlinks=True))
        return str(shutil.copy2(str(src), str(dst), follow_symlinks=False))

    async def append_text_async(
        self, path: PathLike, text: str, *, encoding: str = "utf-8", source: str = "unknown"
    ) -> None:
        target = _coerce_target(path)
        if not isinstance(text, str):
            raise TypeError("text payload must be a string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_append_text

        await async_atomic_append_text(target, text, encoding=encoding)

    def replace_symlink(
        self,
        path: PathLike,
        target_path: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Atomically create or replace a symlink through the file-write lane."""
        link = _coerce_path_allow_dir(path)
        target = _coerce_path_allow_dir(target_path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_symlink:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        return _replace_symlink_unchecked(link, target)

    async def replace_symlink_async(
        self,
        path: PathLike,
        target_path: PathLike,
        *,
        source: str = "unknown",
    ) -> str:
        """Atomically replace a symlink off the event loop after governance."""
        link = _coerce_path_allow_dir(path)
        target = _coerce_path_allow_dir(target_path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.replace_symlink:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        import asyncio

        return await asyncio.to_thread(
            _replace_symlink_unchecked,
            link,
            target,
        )

    def delete_file(self, path: PathLike, *, source: str = "unknown") -> bool:
        """Delete a single file through the same governance lane as writes."""
        target = _coerce_target(path)
        if target.exists() and target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(f"target path is a directory: {target}")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_file:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        try:
            return durable_unlink(target, missing_ok=True)
        except FileNotFoundError:
            return False

    def delete_path(
        self, path: PathLike, *, recursive: bool = False, source: str = "unknown"
    ) -> bool:
        """Synchronous counterpart of :meth:`delete_path_async`.

        `delete_file` refuses directories and `delete_path_async` is a
        coroutine, so a SYNCHRONOUS caller that needed to remove a tree had
        no governed option and reached for `shutil.rmtree` directly —
        `disk_budget.prune_superseded_artifacts` did exactly that, deleting
        artifact directories outside the write gateway entirely.

        Same rules as the async form: governance is required, and a
        directory needs `recursive=True` stated explicitly, because
        "remove this file" and "remove everything under here" must not be
        the same call.

        Use the async form on the event loop — this one blocks on the
        filesystem, and an on-loop tree delete is the shape that froze the
        runtime for twenty minutes once already.
        """
        target = _coerce_path_allow_dir(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        if not os.path.lexists(target):
            return False
        if target.is_symlink():
            target.unlink()
            return True
        if target.is_dir():
            if not recursive:
                raise IsADirectoryError(
                    f"refusing to delete directory without recursive=True: {target}"
                )
            import shutil

            shutil.rmtree(target)
            return True
        target.unlink()
        return True

    def delete_owned_readonly_tree(
        self,
        path: PathLike,
        *,
        source: str = "unknown",
        expected_device: int | None = None,
        expected_inode: int | None = None,
        expected_mtime_ns: int | None = None,
        quarantine_directory: PathLike | None = None,
    ) -> bool:
        """Remove one owner-private read-only artifact tree under governance.

        Immutable training generations deliberately use ``0500`` directories
        and ``0400`` files. A normal recursive delete cannot unlink children
        from those directories. This operation is intentionally narrower than
        ``delete_path``: every entry must be owned by this user, no symlink may
        occur anywhere in the tree, and no group/other permission bit may be
        set. Only then are directory write bits restored for removal.
        """

        target = _coerce_path_allow_dir(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_owned_readonly_tree:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        if not os.path.lexists(target):
            return False
        initial = target.lstat()
        if stat.S_ISLNK(initial.st_mode) or not stat.S_ISDIR(initial.st_mode):
            raise FileWriteTransactionError(
                "owned read-only tree target must be a real directory"
            )

        expected_identity = (
            int(initial.st_dev) if expected_device is None else int(expected_device),
            int(initial.st_ino) if expected_inode is None else int(expected_inode),
            int(initial.st_mtime_ns)
            if expected_mtime_ns is None
            else int(expected_mtime_ns),
        )
        expected_uid = os.geteuid()

        def _validated_directories(root: Path) -> list[Path]:
            directories: list[Path] = []
            for current, child_directories, child_files in os.walk(
                root,
                topdown=True,
                followlinks=False,
            ):
                current_path = Path(current)
                directories.append(current_path)
                for candidate in (
                    current_path,
                    *(current_path / name for name in child_directories),
                    *(current_path / name for name in child_files),
                ):
                    metadata = candidate.lstat()
                    if (
                        stat.S_ISLNK(metadata.st_mode)
                        or metadata.st_uid != expected_uid
                        or stat.S_IMODE(metadata.st_mode) & 0o077
                    ):
                        raise FileWriteTransactionError(
                            "owned read-only tree custody differs"
                        )
            return directories

        # Reject a known-bad tree without moving it. This first traversal is
        # not the race boundary; the exact root inode is checked and renamed
        # atomically below, and the quarantined tree is validated again.
        _validated_directories(target)
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | _required_no_follow_flag(),
        )
        quarantine_descriptor: int | None = None
        target_descriptor: int | None = None
        quarantine_moved = False
        quarantine_path: Path
        quarantine_name = f".aura-delete-{target.name}-{uuid.uuid4().hex}"
        try:
            parent_metadata = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or parent_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(parent_metadata.st_mode) & 0o077
            ):
                raise FileWriteTransactionError(
                    "owned read-only tree parent custody differs"
                )
            observed = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or (observed.st_dev, observed.st_ino, observed.st_mtime_ns)
                != expected_identity
            ):
                raise FileWriteTransactionError(
                    "owned read-only tree changed before quarantine"
                )
            target_descriptor = os.open(
                target.name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | _required_no_follow_flag(),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(target_descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mtime_ns,
            ) != expected_identity:
                raise FileWriteTransactionError(
                    "owned read-only tree changed before quarantine"
                )
            # Darwin refuses to rename a 0500 directory even when its parent is
            # writable. Change permissions through the verified descriptor,
            # then recheck the pathname before the atomic namespace move.
            os.fchmod(target_descriptor, 0o700)

            if quarantine_directory is None:
                quarantine_descriptor = os.dup(parent_descriptor)
                quarantine_path = target.parent / quarantine_name
            else:
                quarantine_root = _coerce_path_allow_dir(quarantine_directory)
                quarantine_descriptor = os.open(
                    quarantine_root,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | _required_no_follow_flag(),
                )
                quarantine_metadata = os.fstat(quarantine_descriptor)
                if (
                    not stat.S_ISDIR(quarantine_metadata.st_mode)
                    or quarantine_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(quarantine_metadata.st_mode) != 0o700
                    or quarantine_metadata.st_dev != observed.st_dev
                ):
                    raise FileWriteTransactionError(
                        "owned read-only tree quarantine custody differs"
                    )
                quarantine_path = quarantine_root / quarantine_name

            os.rename(
                target.name,
                quarantine_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=quarantine_descriptor,
            )
            quarantine_moved = True
            quarantined_metadata = os.stat(
                quarantine_name,
                dir_fd=quarantine_descriptor,
                follow_symlinks=False,
            )
            if (
                quarantined_metadata.st_dev,
                quarantined_metadata.st_ino,
                quarantined_metadata.st_mtime_ns,
            ) != expected_identity:
                raise FileWriteTransactionError(
                    "owned read-only tree quarantine identity differs"
                )
            os.fsync(parent_descriptor)
            if quarantine_descriptor != parent_descriptor:
                os.fsync(quarantine_descriptor)
        finally:
            if target_descriptor is not None:
                if not quarantine_moved:
                    try:
                        os.fchmod(target_descriptor, stat.S_IMODE(initial.st_mode))
                    except OSError:
                        pass
                os.close(target_descriptor)
            os.close(parent_descriptor)
            if quarantine_descriptor is not None:
                os.close(quarantine_descriptor)

        directories = _validated_directories(quarantine_path)

        for directory in directories:
            descriptor = os.open(
                directory,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | _required_no_follow_flag(),
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != expected_uid
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                ):
                    raise FileWriteTransactionError(
                        "owned read-only tree changed before removal"
                    )
                os.fchmod(descriptor, 0o700)
            finally:
                os.close(descriptor)

        import shutil

        shutil.rmtree(quarantine_path)
        return True

    async def delete_path_async(
        self, path: PathLike, *, recursive: bool = False, source: str = "unknown"
    ) -> bool:
        """Delete a file or directory tree under governance, off the event loop.

        Directories require ``recursive=True`` — refusing an implicit tree
        delete is the difference between "remove this file" and "remove
        everything under here", and callers must state which they mean.

        Returns True if something was deleted, False if the path was absent.
        """
        target = _coerce_path_allow_dir(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.delete_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _delete() -> bool:
            if not os.path.lexists(target):
                return False
            if target.is_symlink():
                target.unlink()
                return True
            if target.is_dir():
                if not recursive:
                    raise IsADirectoryError(
                        f"refusing to delete directory without recursive=True: {target}"
                    )
                import shutil

                shutil.rmtree(target)
                return True
            target.unlink()
            return True

        import asyncio

        return await asyncio.to_thread(_delete)

    async def move_path_async(
        self, path: PathLike, destination: PathLike, *, source: str = "unknown"
    ) -> str:
        """Move a file or directory under governance, off the event loop.

        Returns the final destination path as a string.
        """
        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.move_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _move() -> str:
            if not os.path.lexists(src):
                raise FileNotFoundError(f"move source does not exist: {src}")
            import shutil

            return str(shutil.move(str(src), str(dst)))

        import asyncio

        return await asyncio.to_thread(_move)

    async def copy_path_async(
        self, path: PathLike, destination: PathLike, *, source: str = "unknown"
    ) -> str:
        """Copy a file or directory tree under governance, off the event loop.

        Returns the final destination path as a string.
        """
        src = _coerce_path_allow_dir(path)
        dst = _coerce_path_allow_dir(destination)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.copy_path:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        def _copy() -> str:
            if not os.path.lexists(src):
                raise FileNotFoundError(f"copy source does not exist: {src}")
            import shutil

            if src.is_dir():
                return str(shutil.copytree(str(src), str(dst), symlinks=True))
            return str(shutil.copy2(str(src), str(dst), follow_symlinks=False))

        import asyncio

        return await asyncio.to_thread(_copy)

    def drain_text(self, path: PathLike, *, encoding: str = "utf-8", source: str = "unknown") -> str:
        """Atomically drain a text queue file and return its previous contents.

        The target is first moved aside, then read and deleted. Writers that
        append during the drain create a fresh target file, so entries are not
        lost by a read-then-clear race.
        """
        target = _coerce_target(path)
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.drain_text:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        if not target.exists():
            return ""
        drain_path = target.with_name(
            f".aura_drain_{target.name}_{os.getpid()}_{time.time_ns()}"
        )
        try:
            target.replace(drain_path)
        except FileNotFoundError:
            return ""
        try:
            return drain_path.read_text(encoding=encoding)
        finally:
            try:
                drain_path.unlink()
            except FileNotFoundError:
                pass

    def write_json(
        self,
        path: PathLike,
        obj: Any,
        *,
        schema_version: int,
        schema_name: str | None = None,
        indent: int | None = 2,
        source: str = "unknown",
    ) -> None:
        target = _coerce_target(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_json:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        atomic_write_json(
            target,
            obj,
            schema_version=schema_version,
            schema_name=schema_name,
            indent=indent,
        )

    async def write_json_async(
        self,
        path: PathLike,
        obj: Any,
        *,
        schema_version: int,
        schema_name: str | None = None,
        indent: int | None = 2,
        source: str = "unknown",
    ) -> None:
        target = _coerce_target(path)
        if governance_runtime_active():
            require_governance(
                f"file_write_gateway.write_json:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )
        from core.runtime.atomic_writer import async_atomic_write_json

        await async_atomic_write_json(
            target,
            obj,
            schema_version=schema_version,
            schema_name=schema_name,
            indent=indent,
        )


def _coerce_target(path: PathLike) -> Path:
    if path is None:
        raise ValueError("target path is required")
    target = Path(path).expanduser()
    if target.exists() and target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(f"target path is a directory: {target}")
    assert_state_path_allowed(target, source="file_write_gateway")
    return target


def _coerce_path_allow_dir(path: PathLike) -> Path:
    """Coerce a path argument for operations that legitimately act on directories."""
    if path is None:
        raise ValueError("target path is required")
    target = Path(path).expanduser()
    assert_state_path_allowed(target, source="file_write_gateway")
    return target


_gateway: FileWriteGateway | None = None


def get_file_write_gateway() -> FileWriteGateway:
    global _gateway
    if _gateway is None:
        _gateway = FileWriteGateway()
    return _gateway


def _registry_write_bytes(path: PathLike, payload: bytes, source: str) -> None:
    get_file_write_gateway().write_bytes(path, payload, source=source)


def _registry_write_text(path: PathLike, text: str, encoding: str, source: str) -> None:
    get_file_write_gateway().write_text(path, text, encoding=encoding, source=source)


try:
    from core.runtime.service_registry import install_file_write_sinks

    install_file_write_sinks(
        write_bytes=_registry_write_bytes,
        write_text=_registry_write_text,
    )
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
    logger.debug("Runtime file-write bridge unavailable: %s", exc)


__all__ = [
    "DirectoryFileWriteBatchEntry",
    "FileWriteBatchEntry",
    "FileWriteBatchReceipt",
    "FileWriteGateway",
    "FileWriteTransactionError",
    "get_file_write_gateway",
]
