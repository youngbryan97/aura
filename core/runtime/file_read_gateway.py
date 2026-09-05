"""Canonical no-follow boundary for stable reads of evidence and state files."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

PathLike = str | Path
_DIRECTORY_BATCH_LOCK_FILE = ".aura_file_write_batch.lock"


class StableFileReadError(OSError):
    """A file could not be proven regular, bounded, and stable while read."""

    def __init__(self, code: str, path: PathLike) -> None:
        super().__init__(f"{code}: {path}")
        self.code = code


def _required_no_follow_flag(path: PathLike) -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(value, int) or value <= 0:
        raise StableFileReadError("nofollow_unsupported", path)
    return value


@dataclass(frozen=True)
class StableFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> StableFileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


@contextmanager
def open_stable_readonly_binary(
    path: PathLike,
    *,
    max_bytes: int,
) -> Iterator[tuple[BinaryIO, StableFileIdentity]]:
    """Open a regular file without following its final symlink.

    The descriptor identity is checked after a successful read so replacement,
    truncation, or mutation cannot silently produce an accepted receipt.
    """

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    target = Path(path).expanduser()
    if target.is_symlink():
        raise StableFileReadError("symlink_rejected", target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_no_follow_flag(target)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StableFileReadError("symlink_rejected", target) from exc
        raise
    handle: BinaryIO | None = None
    try:
        before_stat = os.fstat(fd)
        if not stat.S_ISREG(before_stat.st_mode):
            raise StableFileReadError("not_regular_file", target)
        before = StableFileIdentity.from_stat(before_stat)
        if before.size < 0 or before.size > max_bytes:
            raise StableFileReadError("size_exceeds_bound", target)
        handle = os.fdopen(fd, "rb", closefd=False)
        try:
            yield handle, before
        except BaseException:  # noqa: BLE001 - preserve the caller's validation failure
            raise
        else:
            after = StableFileIdentity.from_stat(os.fstat(fd))
            if after != before:
                raise StableFileReadError("changed_during_read", target)
    finally:
        try:
            if handle is not None:
                handle.close()
        finally:
            os.close(fd)


def _mark_file_provenance(path: "PathLike") -> None:
    """Record that this turn read a file whose contents Aura did not write.

    Never raises: provenance is a note about the turn, and failing to take the
    note must not fail the read. It is recorded as a degradation instead,
    because a missing note means an action gate answers "trusted" for a turn
    that read someone else's text.
    """
    try:
        from core.security.content_provenance import ProvenanceClass, record_ingest

        record_ingest(ProvenanceClass.OWNER_FILE, f"read {str(path)[:120]}")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "file_read_gateway",
            exc,
            severity="warning",
            action="file content provenance not recorded; action gates will treat this turn as untainted",
            enforce_failure_policy=False,
        )


def read_stable_bytes(path: PathLike, *, max_bytes: int) -> bytes:
    """Read one stable regular file under an explicit byte bound."""

    with open_stable_readonly_binary(path, max_bytes=max_bytes) as (handle, identity):
        payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes or len(payload) != identity.size:
            raise StableFileReadError("bounded_read_length_mismatch", path)
        if handle.read(1):
            raise StableFileReadError("grew_beyond_bound", path)
        # Recorded as OWNER_FILE, which sits BELOW the untrusted floor and so
        # does not by itself disarm an action gate. That is a policy choice,
        # not an oversight: the owner pointed at this file deliberately, and
        # treating every file read as untrusted would disarm desktop control on
        # almost every turn, which makes a control nobody keeps. The residual
        # risk is real and named — a README inside a cloned repository was
        # authored by a stranger and is currently trusted at this level.
        _mark_file_provenance(path)
        return payload


def _open_directory_no_follow(path: PathLike) -> tuple[int, Path]:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in lexical.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | _required_no_follow_flag(lexical),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise StableFileReadError(
                "directory_not_owner_private",
                lexical,
            )
        return descriptor, lexical
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_stable_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
) -> tuple[int, os.stat_result]:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _required_no_follow_flag(name)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise StableFileReadError("not_regular_file", name)
        if before.st_size < 0 or before.st_size > max_bytes:
            raise StableFileReadError("size_exceeds_bound", name)
        return descriptor, before
    except BaseException:
        os.close(descriptor)
        raise


def _read_open_stable_at(
    descriptor: int,
    before: os.stat_result,
    *,
    name: str,
    max_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    after = os.fstat(descriptor)
    if StableFileIdentity.from_stat(before) != StableFileIdentity.from_stat(
        after
    ):
        raise StableFileReadError("changed_during_read", name)
    if len(payload) != before.st_size or len(payload) > max_bytes:
        raise StableFileReadError("bounded_read_length_mismatch", name)
    return payload


def _assert_lock_binding(
    directory_fd: int,
    lock_fd: int,
    lexical: Path,
) -> None:
    opened = os.fstat(lock_fd)
    try:
        entry = os.stat(
            _DIRECTORY_BATCH_LOCK_FILE,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise StableFileReadError(
            "directory_lock_path_changed",
            lexical,
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) & 0o077
        or (opened.st_dev, opened.st_ino)
        != (entry.st_dev, entry.st_ino)
    ):
        raise StableFileReadError(
            "directory_lock_path_changed",
            lexical,
        )


def read_stable_directory_files(
    directory: PathLike,
    *,
    names: Collection[str],
    max_bytes_per_file: int,
) -> dict[str, bytes]:
    """Read an exact package generation under its directory transaction lock."""

    if type(max_bytes_per_file) is not int or max_bytes_per_file < 0:
        raise ValueError("max_bytes_per_file must be a non-negative integer")
    required = set(names)
    if (
        not required
        or any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\0" in name
            for name in required
        )
    ):
        raise ValueError("directory read names must be flat safe filenames")
    directory_fd, lexical = _open_directory_no_follow(directory)
    lock_fd: int | None = None
    opened_files: dict[str, tuple[int, os.stat_result]] = {}
    try:
        lock_fd = os.open(
            _DIRECTORY_BATCH_LOCK_FILE,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | _required_no_follow_flag(lexical),
            dir_fd=directory_fd,
        )
        _assert_lock_binding(directory_fd, lock_fd, lexical)
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        _assert_lock_binding(directory_fd, lock_fd, lexical)
        inventory = set(os.listdir(directory_fd))
        if inventory != required | {_DIRECTORY_BATCH_LOCK_FILE}:
            raise StableFileReadError(
                "directory_inventory_mismatch",
                lexical,
            )
        for name in sorted(required):
            opened_files[name] = _open_stable_at(
                directory_fd,
                name,
                max_bytes=max_bytes_per_file,
            )
        payloads = {
            name: _read_open_stable_at(
                descriptor,
                before,
                name=name,
                max_bytes=max_bytes_per_file,
            )
            for name, (descriptor, before) in opened_files.items()
        }
        if set(os.listdir(directory_fd)) != required | {
            _DIRECTORY_BATCH_LOCK_FILE
        }:
            raise StableFileReadError(
                "directory_inventory_changed_during_read",
                lexical,
            )
        for name, (descriptor, before) in opened_files.items():
            after = os.fstat(descriptor)
            entry = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                StableFileIdentity.from_stat(before)
                != StableFileIdentity.from_stat(after)
                or (after.st_dev, after.st_ino)
                != (entry.st_dev, entry.st_ino)
                or entry.st_nlink != 1
                or entry.st_uid != os.getuid()
            ):
                raise StableFileReadError(
                    "directory_artifact_identity_changed",
                    name,
                )
        _assert_lock_binding(directory_fd, lock_fd, lexical)
        reopened_fd, _ = _open_directory_no_follow(lexical)
        try:
            before = os.fstat(directory_fd)
            after = os.fstat(reopened_fd)
            if (before.st_dev, before.st_ino) != (
                after.st_dev,
                after.st_ino,
            ):
                raise StableFileReadError(
                    "directory_path_changed_during_read",
                    lexical,
                )
        finally:
            os.close(reopened_fd)
        return payloads
    finally:
        for descriptor, _before in opened_files.values():
            os.close(descriptor)
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        os.close(directory_fd)


__all__ = [
    "StableFileIdentity",
    "StableFileReadError",
    "open_stable_readonly_binary",
    "read_stable_directory_files",
    "read_stable_bytes",
]
