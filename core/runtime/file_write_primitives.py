"""Descriptor-level file operations, below governance.

Every function here works against an OPEN DIRECTORY DESCRIPTOR rather than a
path, because a path can be swapped between the check and the use. Nothing
here knows about governance, logging or the write gateway's API — it is the
layer the gateway and the batch journal both stand on, and it was extracted
from file_write_gateway.py when that module reached the size ceiling that
exists to stop one file owning three jobs.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import PathLike

__all__ = [
    "FileWriteTransactionError",
    "DIRECTORY_BATCH_THREAD_LOCK",
    "DIRECTORY_BATCH_LOCK_FILE",
    "DIRECTORY_BATCH_JOURNAL_FILE",
    "DIRECTORY_BATCH_INTERNAL_PREFIX",
    "DIRECTORY_BATCH_JOURNAL_SCHEMA",
    "MAX_DIRECTORY_BATCH_JOURNAL_BYTES",
    "validated_permissions",
    "validated_flat_name",
    "required_no_follow_flag",
    "assert_private_owned_directory",
    "open_directory_no_follow",
    "write_all",
    "read_regular_at",
    "stage_bytes_at",
    "directory_entry_identity",
    "assert_lock_binding",
    "exists_at",
    "unlink_private_regular_at",
    "replace_symlink_unchecked",
]


class FileWriteTransactionError(RuntimeError):
    """A multi-file gateway commit failed or could not be rolled back."""



from core.runtime.lockdep import checked_lock

DIRECTORY_BATCH_THREAD_LOCK = checked_lock(
    "runtime.file_write_primitives.directory_batch"
)
DIRECTORY_BATCH_LOCK_FILE = ".aura_file_write_batch.lock"
DIRECTORY_BATCH_JOURNAL_FILE = ".aura_file_write_batch.journal"
DIRECTORY_BATCH_INTERNAL_PREFIX = ".aura-batch-"
DIRECTORY_BATCH_JOURNAL_SCHEMA = "aura.file_write.directory_batch_journal.v2"
MAX_DIRECTORY_BATCH_JOURNAL_BYTES = 1024 * 1024


def validated_permissions(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("permissions must be an integer")
    if mode < 0 or mode & ~0o777:
        raise ValueError("permissions must contain only rwx permission bits")
    return mode


def validated_flat_name(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\0" in name
        or name.startswith(DIRECTORY_BATCH_INTERNAL_PREFIX)
        or name in {
            DIRECTORY_BATCH_LOCK_FILE,
            DIRECTORY_BATCH_JOURNAL_FILE,
        }
    ):
        raise ValueError("directory batch names must be flat safe filenames")
    return name


def required_no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(value, int) or value <= 0:
        raise FileWriteTransactionError(
            "directory batch requires O_NOFOLLOW support"
        )
    return value


def assert_private_owned_directory(
    metadata: os.stat_result,
    lexical: Path,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise FileWriteTransactionError(
            f"directory batch target is not owner-private: {lexical}"
        )


def open_directory_no_follow(path: PathLike) -> tuple[int, Path]:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = required_no_follow_flag()
    descriptor = os.open("/", flags)
    try:
        for component in lexical.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        assert_private_owned_directory(metadata, lexical)
        return descriptor, lexical
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short directory-relative batch write")
        offset += written


def read_regular_at(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int | None = None,
) -> tuple[bytes, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | required_no_follow_flag()
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
        ):
            raise FileWriteTransactionError(
                f"directory batch target is not a private regular file: {name}"
            )
        if max_bytes is not None and metadata.st_size > max_bytes:
            raise FileWriteTransactionError(
                f"directory batch target exceeds read bound: {name}"
            )
        chunks: list[bytes] = []
        remaining = None if max_bytes is None else max_bytes + 1
        while True:
            read_size = 1024 * 1024
            if remaining is not None:
                if remaining <= 0:
                    break
                read_size = min(read_size, remaining)
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise FileWriteTransactionError(
                f"directory batch target disappeared during read: {name}"
            ) from exc
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(getattr(metadata, field) != getattr(after, field) for field in stable_fields)
            or (after.st_dev, after.st_ino, after.st_nlink)
            != (entry.st_dev, entry.st_ino, entry.st_nlink)
            or len(payload) != metadata.st_size
            or (max_bytes is not None and len(payload) > max_bytes)
        ):
            raise FileWriteTransactionError(
                f"directory batch target changed during read: {name}"
            )
        return payload, stat.S_IMODE(metadata.st_mode)
    finally:
        os.close(descriptor)


def stage_bytes_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    mode: int,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | required_no_follow_flag()
    )
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        write_all(descriptor, payload)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def directory_entry_identity(
    directory_fd: int,
    name: str,
) -> tuple[int, int, int]:
    metadata = os.stat(
        name,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    return metadata.st_dev, metadata.st_ino, metadata.st_nlink


def assert_lock_binding(directory_fd: int, lock_fd: int) -> None:
    opened = os.fstat(lock_fd)
    try:
        entry = os.stat(
            DIRECTORY_BATCH_LOCK_FILE,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise FileWriteTransactionError(
            "directory batch lock path disappeared"
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) & 0o077
        or (opened.st_dev, opened.st_ino)
        != (entry.st_dev, entry.st_ino)
    ):
        raise FileWriteTransactionError(
            "directory batch lock path no longer binds the held lock"
        )


def exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def unlink_private_regular_at(directory_fd: int, name: str) -> None:
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise FileWriteTransactionError(
            f"refusing to remove unsafe transaction artifact: {name}"
        )
    os.unlink(name, dir_fd=directory_fd)


def replace_symlink_unchecked(link: Path, target: Path) -> str:
    """Point a symlink at a target, atomically, with no governance check.

    A static method on the gateway until the class hit its method ceiling. It
    never took `self` and never asked the Will anything: the caller does that
    and then calls this. It belongs with the other descriptor-level moves.
    """
    from core.runtime.atomic_writer import ensure_private_directory

    if not target.exists():
        raise FileNotFoundError(f"symlink target does not exist: {target}")
    if link.exists() and link.is_dir() and not link.is_symlink():
        raise IsADirectoryError(f"refusing to replace directory with symlink: {link}")
    ensure_private_directory(link.parent)
    temporary = link.with_name(
        f".{link.name}.{os.getpid()}.{time.time_ns()}.symlink.tmp"
    )
    try:
        temporary.symlink_to(
            target.resolve(),
            target_is_directory=target.is_dir(),
        )
        os.replace(temporary, link)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    return str(target.resolve())
