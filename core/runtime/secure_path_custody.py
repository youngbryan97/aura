"""Descriptor-bound custody for security-sensitive directory trees.

Path validation alone is vulnerable to time-of-check/time-of-use exchanges.
This module walks directory components with ``openat`` semantics, refuses
symlinks, retains the final directory descriptor, and performs writes relative
to that descriptor. Device/inode bindings make later pathname replacement
detectable without trusting the pathname that was originally checked.

The security boundary is the effective operating-system user. This module
protects against stale paths, pre-existing links, accidental namespace drift,
and non-cooperating unprivileged processes under other UIDs. It cannot isolate
a process from a malicious process running as the same UID or from a privileged
actor: those peers can signal or trace the process, bypass discretionary access
controls, and race any userspace pathname verification. Campaigns bind this
exact threat model into their signed configuration.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any


class SecurePathCustodyError(RuntimeError):
    """A directory or child path violated descriptor custody."""


PATH_CUSTODY_THREAT_MODEL_SCHEMA = "aura.secure_path_custody_threat_model.v1"


def path_custody_threat_model() -> dict[str, Any]:
    """Return the exact, self-bound OS trust boundary for path custody."""

    body = {
        "schema": PATH_CUSTODY_THREAT_MODEL_SCHEMA,
        "security_boundary": "exclusive_effective_os_user",
        "trusted_principal": {"effective_uid": os.geteuid()},
        "guarantees": [
            "descriptor_bound_root_identity",
            "nofollow_component_walk",
            "owner_only_private_roots",
            "regular_single_link_file_admission",
            "descriptor_relative_atomic_publication",
            "namespace_drift_detection_within_trust_boundary",
        ],
        "excluded_adversary": "malicious_same_uid_or_privileged_process",
        "excluded_capabilities": [
            "privileged_discretionary_access_control_bypass",
            "same_uid_debug_or_process_control",
            "same_uid_race_after_userspace_namespace_verification",
        ],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {**body, "threat_model_sha256": digest}


def validate_path_custody_threat_model(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = path_custody_threat_model()
    if dict(value) != expected:
        raise SecurePathCustodyError("secure_path_threat_model_mismatch")
    return expected


def _directory_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _identity(fd: int) -> dict[str, int]:
    observed = os.fstat(fd)
    if not stat.S_ISDIR(observed.st_mode):
        raise SecurePathCustodyError("secure_path_not_directory")
    return {"st_dev": int(observed.st_dev), "st_ino": int(observed.st_ino)}


def validate_directory_identity(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != {"st_dev", "st_ino"}:
        raise SecurePathCustodyError("secure_path_identity_schema_invalid")
    normalized: dict[str, int] = {}
    for field in ("st_dev", "st_ino"):
        item = value.get(field)
        if type(item) is not int or item < 0:
            raise SecurePathCustodyError("secure_path_identity_invalid")
        normalized[field] = item
    return normalized


def _relative_parts(value: str | Path) -> tuple[str, ...]:
    text = value.as_posix() if isinstance(value, Path) else value
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or str(pure) != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SecurePathCustodyError("secure_path_relative_path_invalid")
    return pure.parts


def _open_absolute_directory(path: Path, *, create: bool) -> int:
    absolute = path.expanduser().absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise SecurePathCustodyError("secure_path_absolute_path_required")
    flags = _directory_flags()
    current_fd = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o700, dir_fd=current_fd)
                next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


class DirectoryCustody:
    """Hold and operate beneath one immutable directory identity."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path.expanduser().absolute()
        self._fd = fd
        self._identity = _identity(fd)
        self._closed = False

    @classmethod
    def acquire(
        cls,
        path: str | Path,
        *,
        create: bool = False,
        expected_identity: Mapping[str, Any] | None = None,
        private: bool = False,
    ) -> DirectoryCustody:
        absolute = Path(path).expanduser().absolute()
        try:
            fd = _open_absolute_directory(absolute, create=create)
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_directory_open_failed") from exc
        custody = cls(absolute, fd)
        try:
            if expected_identity is not None:
                expected = validate_directory_identity(expected_identity)
                if custody.identity != expected:
                    raise SecurePathCustodyError("secure_path_identity_mismatch")
            if private:
                os.fchmod(fd, 0o700)
                observed = os.fstat(fd)
                if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) != 0o700:
                    raise SecurePathCustodyError("secure_path_private_root_invalid")
            custody.verify()
            return custody
        except BaseException:
            custody.close()
            raise

    @property
    def identity(self) -> dict[str, int]:
        return dict(self._identity)

    def fileno(self) -> int:
        """Return the held descriptor after proving its identity is still current."""

        self.verify()
        return self._fd

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> DirectoryCustody:
        self.verify()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            # not a failure: a finalizer cannot raise anywhere useful, and
            # the descriptor is being reclaimed by the interpreter either
            # way. __exit__ above is where a close is reported.
            pass

    def verify(self) -> None:
        if self._closed:
            raise SecurePathCustodyError("secure_path_custody_closed")
        if _identity(self._fd) != self._identity:
            raise SecurePathCustodyError("secure_path_held_identity_drift")
        try:
            observed_fd = _open_absolute_directory(self.path, create=False)
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_pathname_replaced") from exc
        try:
            if _identity(observed_fd) != self._identity:
                raise SecurePathCustodyError("secure_path_pathname_replaced")
        finally:
            os.close(observed_fd)

    def fsync(self) -> None:
        self.verify()
        os.fsync(self._fd)
        self.verify()

    def _open_parent(self, relative: str | Path, *, create: bool) -> tuple[int, str]:
        parts = _relative_parts(relative)
        parent_fd = os.dup(self._fd)
        flags = _directory_flags()
        try:
            for component in parts[:-1]:
                try:
                    next_fd = os.open(component, flags, dir_fd=parent_fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                    next_fd = os.open(component, flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            self._verify_open_parent(relative, parent_fd)
            return parent_fd, parts[-1]
        except OSError as exc:
            os.close(parent_fd)
            raise SecurePathCustodyError("secure_path_parent_open_failed") from exc
        except BaseException:
            os.close(parent_fd)
            raise

    def _verify_open_parent(self, relative: str | Path, parent_fd: int) -> None:
        parts = _relative_parts(relative)
        expected = self.path.joinpath(*parts[:-1]) if len(parts) > 1 else self.path
        try:
            observed_fd = _open_absolute_directory(expected, create=False)
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_descendant_parent_replaced") from exc
        try:
            if _identity(observed_fd) != _identity(parent_fd):
                raise SecurePathCustodyError("secure_path_descendant_parent_replaced")
        finally:
            os.close(observed_fd)
        self.verify()

    def open_directory(self, relative: str | Path, *, create: bool = False) -> int:
        self.verify()
        parent_fd, name = self._open_parent(relative, create=create)
        try:
            try:
                fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            self._verify_open_parent(relative, parent_fd)
            expected_child = self.path.joinpath(*_relative_parts(relative))
            observed_child_fd = _open_absolute_directory(expected_child, create=False)
            try:
                if _identity(observed_child_fd) != _identity(fd):
                    raise SecurePathCustodyError("secure_path_descendant_directory_replaced")
            finally:
                os.close(observed_child_fd)
            return fd
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_directory_open_failed") from exc
        finally:
            os.close(parent_fd)
            self.verify()

    def open_file(
        self,
        relative: str | Path,
        flags: int,
        *,
        mode: int = 0o600,
        create_parents: bool = False,
        regular: bool = True,
    ) -> int:
        self.verify()
        parent_fd, name = self._open_parent(relative, create=create_parents)
        try:
            fd = os.open(
                name,
                flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
            observed = os.fstat(fd)
            if regular and (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or observed.st_nlink != 1
            ):
                os.close(fd)
                raise SecurePathCustodyError("secure_path_file_identity_unsafe")
            self._verify_open_parent(relative, parent_fd)
            return fd
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_file_open_failed") from exc
        finally:
            os.close(parent_fd)
            self.verify()

    def file_exists(self, relative: str | Path) -> bool:
        try:
            fd = self.open_file(relative, os.O_RDONLY)
        except SecurePathCustodyError as exc:
            if exc.__cause__ is not None and isinstance(exc.__cause__, FileNotFoundError):
                return False
            raise
        else:
            os.close(fd)
            return True

    @contextmanager
    def file_lock(self, relative: str | Path) -> Iterator[int]:
        fd = self.open_file(
            relative,
            os.O_RDWR | os.O_CREAT,
            mode=0o600,
            create_parents=True,
        )
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            self.verify()

    def ensure_directory(self, relative: str | Path) -> Path:
        self.verify()
        parent_fd, name = self._open_parent(relative, create=True)
        try:
            try:
                child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            except FileNotFoundError:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            try:
                os.fchmod(child_fd, 0o700)
            finally:
                os.close(child_fd)
            self._verify_open_parent(relative, parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_directory_create_failed") from exc
        finally:
            os.close(parent_fd)
        self.verify()
        return self.path.joinpath(*_relative_parts(relative))

    def _publish_temp(self, parent_fd: int, payload: bytes, mode: int) -> str:
        temp_name = f".aura-custody-{os.getpid()}-{secrets.token_hex(12)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp_name, flags, mode, dir_fd=parent_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise SecurePathCustodyError("secure_path_short_write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        return temp_name

    def write_bytes_once(self, relative: str | Path, payload: bytes, *, mode: int = 0o600) -> bool:
        self.verify()
        parent_fd, name = self._open_parent(relative, create=True)
        temp_name = ""
        published = False
        try:
            temp_name = self._publish_temp(parent_fd, payload, mode)
            self._verify_open_parent(relative, parent_fd)
            try:
                os.link(
                    temp_name,
                    name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                published = True
            except FileExistsError:
                # not a failure: write-once means the name was taken, and
                # False is how the caller hears that.
                published = False
            self._verify_open_parent(relative, parent_fd)
            os.fsync(parent_fd)
            return published
        except SecurePathCustodyError:
            if published:
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass  # not a failure: the publish this is undoing did not land
            raise
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_write_once_failed") from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass  # not a failure: the rename above consumed it
            os.close(parent_fd)
            self.verify()

    def atomic_write_bytes(
        self, relative: str | Path, payload: bytes, *, mode: int = 0o600
    ) -> None:
        self.verify()
        parent_fd, name = self._open_parent(relative, create=True)
        temp_name = ""
        published = False
        try:
            temp_name = self._publish_temp(parent_fd, payload, mode)
            self._verify_open_parent(relative, parent_fd)
            os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_name = ""
            published = True
            self._verify_open_parent(relative, parent_fd)
            os.fsync(parent_fd)
        except SecurePathCustodyError:
            if published:
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass  # not a failure: the publish this is undoing did not land
            raise
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_atomic_write_failed") from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass  # not a failure: the rename above consumed it
            os.close(parent_fd)
            self.verify()

    def read_bytes(self, relative: str | Path, *, max_bytes: int) -> bytes:
        if type(max_bytes) is not int or max_bytes < 1:
            raise SecurePathCustodyError("secure_path_read_limit_invalid")
        self.verify()
        parent_fd, name = self._open_parent(relative, create=False)
        try:
            self._verify_open_parent(relative, parent_fd)
            fd = os.open(name, _file_flags(), dir_fd=parent_fd)
            try:
                observed = os.fstat(fd)
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_uid != os.geteuid()
                    or observed.st_nlink != 1
                    or observed.st_size > max_bytes
                ):
                    raise SecurePathCustodyError("secure_path_file_invalid")
                chunks: list[bytes] = []
                remaining = max_bytes + 1
                while remaining > 0:
                    chunk = os.read(fd, min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if len(payload) > max_bytes:
                    raise SecurePathCustodyError("secure_path_file_too_large")
                return payload
            finally:
                os.close(fd)
        except OSError as exc:
            raise SecurePathCustodyError("secure_path_read_failed") from exc
        finally:
            os.close(parent_fd)
            self.verify()


__all__ = [
    "DirectoryCustody",
    "SecurePathCustodyError",
    "validate_directory_identity",
]
