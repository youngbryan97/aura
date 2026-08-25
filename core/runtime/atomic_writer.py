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
- atomic_append_text(path, text)
- atomic_write_json(path, obj, schema_version)
- atomic_hardlink_replace(source, target)
- durable_replace(source, target)
- durable_unlink(path)

with explicit schema-version envelopes so loaders can detect ancient
records and refuse rather than silently misread.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.AtomicWriter")

PathLike = str | Path

DEFAULT_TEMP_PREFIX = ".aura_atomic_"
_append_locks: dict[Path, threading.Lock] = {}
_append_locks_guard = threading.Lock()
_interprocess_locks: dict[Path, _ReentrantFileLockState] = {}
_interprocess_locks_guard = threading.Lock()


class _ReentrantFileLockState:
    """One process-local owner for a path-backed advisory lock."""

    def __init__(self) -> None:
        self.thread_lock = threading.RLock()
        self.depth = 0
        self.fd: int | None = None


class AtomicWriteError(RuntimeError):
    """Raised when an atomic write cannot complete."""


#: True when this platform's fsync() leaves data in the drive's own write
#: cache, so that only F_FULLFSYNC actually reaches stable storage.
#:
#: This is macOS behaviour and it is documented, not suspected: fsync(2) there
#: pushes to the device and returns, and Apple's guidance is to use
#: fcntl(F_FULLFSYNC) when the write must survive power loss. Aura runs on
#: macOS, so until now every "durable" write in the runtime — mind-state
#: backups and hash-chained ledgers included — was durable against process
#: death but not against power loss, and nothing said so.
_FSYNC_NEEDS_FULLSYNC = sys.platform == "darwin" and hasattr(fcntl, "F_FULLFSYNC")

#: Set once when F_FULLFSYNC is asked for and the filesystem refuses it.
#: Network and some virtual filesystems return EINVAL/ENOTSUP; retrying per
#: call would pay the failed syscall forever.
_fullsync_unsupported = False


def _fsync_file(fd: int, *, full: bool = False) -> None:
    # Every durable write in the runtime funnels through here, which makes
    # it the one place worth instrumenting: fsync is the blocking call that
    # froze the live event loop for 20 minutes, and it is IO stall by
    # definition. PSI accounts the wait; lockdep reports (once per call
    # site) if we are about to block while holding a lock.
    #
    # `full` requests F_FULLFSYNC — a real flush of the drive cache rather
    # than a handoff to it. It is opt-in rather than the default precisely
    # because of that 20-minute freeze: F_FULLFSYNC is far slower, and making
    # every write in the runtime pay for power-loss durability would trade a
    # silent correctness gap for a loud liveness one. Callers that hold
    # something they cannot reconstruct ask for it; everyone else does not.
    from core.runtime.lockdep import assert_no_locks_held
    from core.runtime.pressure_stall import Resource, stall

    global _fullsync_unsupported

    assert_no_locks_held("fsync")
    want_full = full and _FSYNC_NEEDS_FULLSYNC and not _fullsync_unsupported
    used_full = False
    started = time.perf_counter()
    with stall(Resource.IO):
        if want_full:
            try:
                fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
                used_full = True
            except OSError as exc:
                # The filesystem does not implement it. Fall through to plain
                # fsync rather than losing the write, and stop asking.
                _fullsync_unsupported = True
                logger.warning(
                    "F_FULLFSYNC unsupported on this filesystem (%s); durable "
                    "writes fall back to fsync and are no longer power-loss safe",
                    exc,
                )
        if not used_full:
            try:
                os.fsync(fd)
            except (AttributeError, OSError):
                # Best-effort on platforms where fsync is unavailable.
                pass  # no-op: intentional
    try:
        from core.observability.histograms import record

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record("Aura.Fsync.DurationMs", elapsed_ms)
        if used_full:
            # Separate channel: the whole point of opting in is that the cost
            # is different, and a merged histogram would hide it.
            record("Aura.Fsync.FullDurationMs", elapsed_ms)
    except Exception:  # noqa: BLE001 — telemetry never blocks durability
        logger.debug("fsync histogram recording failed", exc_info=True)


def _fsync_dir(directory: Path, *, full: bool = False) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except (FileNotFoundError, PermissionError, OSError):
        return
    try:
        _fsync_file(dir_fd, full=full)
    finally:
        os.close(dir_fd)


def ensure_private_directory(path: PathLike, *, durable: bool = True) -> Path:
    """Create a durability directory and restrict it to the current user.

    ``durable=False`` skips the fsync of the parent. Use it when the directory
    holds nothing worth recovering — a lock file, a probe, a cache — because
    the fsync is a blocking disk operation and callers reach this while
    holding locks.
    """

    directory = Path(path)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    if durable:
        _fsync_dir(directory.parent)
    return directory


async def async_ensure_private_directory(path: PathLike) -> Path:
    import asyncio

    return await asyncio.to_thread(ensure_private_directory, path)


@contextmanager
def interprocess_file_lock(path: PathLike) -> Iterator[None]:
    """Serialize a multi-file transaction across threads and processes.

    ``flock`` semantics for separately opened descriptors vary by platform and
    are not a substitute for a process-local mutex. The registry closes that
    gap while preserving re-entrant use by the same thread.
    """

    requested = Path(path).expanduser()
    target = requested.parent.resolve(strict=False) / requested.name
    with _interprocess_locks_guard:
        state = _interprocess_locks.setdefault(target, _ReentrantFileLockState())
    state.thread_lock.acquire()
    try:
        if state.depth == 0:
            # durable=False: this directory exists to hold a lock file, which
            # is worthless after a crash. Fsyncing its parent to take a lock
            # put a blocking disk operation inside every first acquisition —
            # and callers reach here already holding their own locks, which is
            # how lockdep found it.
            ensure_private_directory(target.parent, durable=False)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(str(target), flags, 0o600)
            try:
                os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            except BaseException:  # noqa: BLE001 - never leak fd on interruption
                os.close(fd)
                raise
            state.fd = fd
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if state.depth == 0:
                held_fd = state.fd
                state.fd = None
                if held_fd is not None:
                    try:
                        fcntl.flock(held_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(held_fd)
    finally:
        state.thread_lock.release()


def _validated_file_mode(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("file mode must be an integer")
    if mode < 0 or mode & ~0o777:
        raise ValueError("file mode must contain only rwx permission bits")
    return mode


def atomic_write_bytes(
    path: PathLike,
    payload: bytes,
    *,
    durable: bool = True,
    power_safe: bool = False,
    mode: int = 0o600,
) -> None:
    """Atomically replace `path` with `payload`.

    ``durable=False`` keeps the write atomic (readers never observe a torn
    file) but skips both fsyncs. Probe files, caches, and other content that
    is worthless after a crash must use it: under memory-pressure thrash a
    single fsync has blocked the live event loop for 20 minutes.

    ``power_safe=True`` additionally requests F_FULLFSYNC, which is the only
    thing on macOS that flushes the drive's own write cache. Plain ``durable``
    survives process death; only ``power_safe`` survives power loss. It is
    opt-in because it is markedly slower, and the same 20-minute freeze is the
    reason not to make it the default for every write in the runtime. Reach for
    it when losing the write would lose something unreconstructable — mind
    state, ledger commitments, identity material.
    """
    target = Path(path)
    file_mode = _validated_file_mode(mode)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix=DEFAULT_TEMP_PREFIX, dir=str(parent))
    tmp_path = Path(tmp_path_str)
    try:
        os.fchmod(fd, file_mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            if durable:
                _fsync_file(fh.fileno(), full=power_safe)
        os.replace(tmp_path, target)
        if durable:
            # The rename must reach stable storage too, or a power loss can
            # leave the old name pointing at nothing on some filesystems.
            _fsync_dir(parent, full=power_safe)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass  # no-op: intentional
        raise


def atomic_write_bytes_if_absent(
    path: PathLike,
    payload: bytes,
    *,
    durable: bool = True,
    mode: int = 0o600,
) -> bool:
    """Publish complete bytes exactly once without replacing an existing path.

    A fully written temporary file is hard-linked into place. ``link(2)`` is
    atomic and fails with ``EEXIST``, so concurrent publishers can verify the
    winning content without any reader observing a partial file.
    """

    target = Path(path)
    file_mode = _validated_file_mode(mode)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(prefix=DEFAULT_TEMP_PREFIX, dir=str(parent))
    tmp_path = Path(tmp_path_str)
    published = False
    try:
        os.fchmod(fd, file_mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            if durable:
                _fsync_file(fh.fileno())
        try:
            os.link(tmp_path, target, follow_symlinks=False)
            published = True
        except FileExistsError:
            published = False
        if durable:
            _fsync_dir(parent)
        return published
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = True,
    power_safe: bool = False,
    mode: int = 0o600,
) -> None:
    atomic_write_bytes(
        path,
        text.encode(encoding),
        durable=durable,
        power_safe=power_safe,
        mode=mode,
    )


def atomic_hardlink_replace(
    source: PathLike,
    target: PathLike,
    *,
    durable: bool = True,
) -> bool:
    """Atomically make ``target`` another name for a regular ``source`` file.

    Immutable multi-gigabyte artifacts often need a compatibility filename.
    Rewriting their bytes doubles storage and I/O; a hard link preserves the
    exact inode while still giving legacy readers a stable path. Readers see
    either the prior target or the complete source inode.

    Returns ``True`` when the target changed and ``False`` when it already
    names the source inode.
    """

    source_path = Path(source)
    target_path = Path(target)
    source_identity = source_path.lstat()
    if source_path.is_symlink() or not stat.S_ISREG(source_identity.st_mode):
        raise AtomicWriteError("hard-link source must be a regular file")
    if target_path.exists() and not target_path.is_symlink():
        try:
            if os.path.samefile(source_path, target_path):
                return False
        except OSError:
            pass

    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=DEFAULT_TEMP_PREFIX, dir=str(parent))
    temporary = Path(temporary_name)
    os.close(fd)
    temporary.unlink()
    try:
        os.link(source_path, temporary, follow_symlinks=False)
        linked_identity = temporary.lstat()
        if (
            not stat.S_ISREG(linked_identity.st_mode)
            or linked_identity.st_dev != source_identity.st_dev
            or linked_identity.st_ino != source_identity.st_ino
        ):
            raise AtomicWriteError("hard-link source changed during publication")
        os.replace(temporary, target_path)
        if durable:
            _fsync_dir(parent)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


async def async_atomic_write_bytes(
    path: PathLike,
    payload: bytes,
    *,
    durable: bool = True,
    power_safe: bool = False,
    mode: int = 0o600,
) -> None:
    """Event-loop-safe atomic write: the fsync happens on a worker thread.

    Under memory-pressure thrash a single on-loop fsync has frozen the live
    event loop for ~20 minutes (12 recorded crashes). Async callers must use
    this lane; the sync functions are for threads and sync bootstrap only.

    ``power_safe`` is available here for the same reason it exists at all, and
    matters MORE on this lane: F_FULLFSYNC is the slower call, so it is exactly
    the one that must never run on the event loop. Going through the thread is
    what makes power-loss durability affordable for async callers.
    """
    import asyncio

    await asyncio.to_thread(
        atomic_write_bytes,
        path,
        payload,
        durable=durable,
        power_safe=power_safe,
        mode=mode,
    )


async def async_atomic_write_bytes_if_absent(
    path: PathLike,
    payload: bytes,
    *,
    durable: bool = True,
    mode: int = 0o600,
) -> bool:
    import asyncio

    return await asyncio.to_thread(
        atomic_write_bytes_if_absent,
        path,
        payload,
        durable=durable,
        mode=mode,
    )


async def async_atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = True,
    power_safe: bool = False,
    mode: int = 0o600,
) -> None:
    await async_atomic_write_bytes(
        path,
        text.encode(encoding),
        durable=durable,
        power_safe=power_safe,
        mode=mode,
    )


async def async_atomic_append_text(
    path: PathLike, text: str, *, encoding: str = "utf-8"
) -> None:
    import asyncio

    await asyncio.to_thread(atomic_append_text, path, text, encoding=encoding)


def atomic_append_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Durably append text without reading or rewriting the existing target."""
    target = Path(path)
    with _append_locks_guard:
        lock = _append_locks.setdefault(target.resolve(), threading.Lock())
    with lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(str(target), flags, 0o600)
        except OSError as exc:
            raise AtomicWriteError(f"cannot open append target: {target}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            payload = text.encode(encoding)
            written = 0
            while written < len(payload):
                written += os.write(fd, payload[written:])
            _fsync_file(fd)
        except OSError as exc:
            raise AtomicWriteError(f"cannot append to target: {target}") from exc
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)
        _fsync_dir(target.parent)


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    schema_version: int,
    schema_name: str | None = None,
    indent: int | None = 2,
    power_safe: bool = False,
) -> None:
    """Atomically write a JSON envelope `{schema, version, payload}`.

    ``power_safe=True`` requests F_FULLFSYNC for records that cannot be
    reconstructed if the machine loses power — identity, commitments, ledger
    state. See :func:`atomic_write_bytes` for why it is not the default.
    """
    if not isinstance(schema_version, int) or schema_version < 1:
        raise AtomicWriteError("schema_version must be a positive int")
    target = Path(path)
    inferred_schema = schema_name or target.stem or target.name or "atomic_json"
    envelope = {
        "schema": inferred_schema,
        "schema_name": inferred_schema,
        "schema_version": schema_version,
        "payload": obj,
    }
    text = json.dumps(envelope, indent=indent, sort_keys=True, default=str)
    atomic_write_text(path, text, power_safe=power_safe)


async def async_atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    schema_version: int,
    schema_name: str | None = None,
    indent: int | None = 2,
    power_safe: bool = False,
) -> None:
    """Event-loop-safe atomic_write_json: serialization + fsync on a worker thread."""
    import asyncio

    await asyncio.to_thread(
        atomic_write_json,
        path,
        obj,
        schema_version=schema_version,
        schema_name=schema_name,
        indent=indent,
        power_safe=power_safe,
    )


def durable_replace(source: PathLike, target: PathLike) -> None:
    """Atomically move ``source`` over ``target`` and fsync both directories."""

    source_path = Path(source)
    target_path = Path(target)
    if not os.path.lexists(source_path):
        raise FileNotFoundError(f"durable replace source does not exist: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source_path, target_path)
    _fsync_dir(source_path.parent)
    if target_path.parent != source_path.parent:
        _fsync_dir(target_path.parent)


async def async_durable_replace(source: PathLike, target: PathLike) -> None:
    import asyncio

    await asyncio.to_thread(durable_replace, source, target)


def durable_unlink(path: PathLike, *, missing_ok: bool = False) -> bool:
    """Delete a file/symlink and fsync its parent directory."""

    target = Path(path)
    if not os.path.lexists(target):
        if missing_ok:
            return False
        raise FileNotFoundError(target)
    if target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(target)
    os.unlink(target)
    _fsync_dir(target.parent)
    return True


async def async_durable_unlink(path: PathLike, *, missing_ok: bool = False) -> bool:
    import asyncio

    return await asyncio.to_thread(durable_unlink, path, missing_ok=missing_ok)


def read_json_envelope(path: PathLike) -> dict[str, Any]:
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
