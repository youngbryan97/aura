"""Durable, fail-closed selection for non-serving recurrent shadow tissue."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Never

from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.state_ownership import state_root

POINTER_SCHEMA: Final = "aura.unified_intrinsic.shadow_pointer.v1"
MAX_POINTER_BYTES: Final = 64 * 1024
_FIELDS: Final = {
    "schema",
    "package_path",
    "package_id",
    "manifest_sha256",
    "pointer_sha256",
}


class UnifiedRecurrentShadowPointerError(RuntimeError):
    """The durable shadow selection is missing custody or identity evidence."""


def _fail(message: str) -> Never:
    raise UnifiedRecurrentShadowPointerError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"unified shadow pointer {role} is invalid")
    return value


def default_shadow_activation_paths() -> tuple[Path, Path]:
    """Return the durable pointer and its only admitted package directory."""

    root = state_root() / "data/adapters/unified-recurrent-shadow"
    return root / "active.json", root / "releases"


def _reject_symlink_chain(path: Path, *, must_exist: bool) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    parts = lexical.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not must_exist and index == len(parts) - 1:
                return lexical
            raise UnifiedRecurrentShadowPointerError(
                "unified shadow pointer path is unavailable"
            ) from None
        except OSError as exc:
            raise UnifiedRecurrentShadowPointerError(
                "unified shadow pointer path is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail("unified shadow pointer path contains a symlink")
    return lexical


def _private_directory(path: Path, *, create: bool = False) -> Path:
    lexical = path.expanduser().absolute()
    if create:
        lexical.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = _reject_symlink_chain(lexical, must_exist=True)
    try:
        metadata = resolved.stat()
    except OSError as exc:
        raise UnifiedRecurrentShadowPointerError(
            "unified shadow pointer directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("unified shadow pointer directory custody differs")
    return resolved


def _strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    try:
        decoded = json.loads(payload.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, ValueError) as exc:
        raise UnifiedRecurrentShadowPointerError(
            "unified shadow pointer JSON is invalid"
        ) from exc
    if not isinstance(decoded, dict) or _canonical_bytes(decoded) != payload:
        _fail("unified shadow pointer is not canonical")
    return decoded


def _read_pointer(path: Path) -> tuple[dict[str, Any], bytes]:
    lexical = _reject_symlink_chain(path, must_exist=True)
    try:
        before = lexical.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= MAX_POINTER_BYTES
        ):
            _fail("unified shadow pointer custody differs")
        payload = read_stable_bytes(lexical, max_bytes=MAX_POINTER_BYTES)
        after = lexical.stat()
    except (OSError, ValueError) as exc:
        raise UnifiedRecurrentShadowPointerError(
            "unified shadow pointer is unreadable"
        ) from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        _fail("unified shadow pointer changed while reading")
    return validate_shadow_pointer(_strict_json(payload)), payload


def read_shadow_pointer(pointer_path: Path) -> dict[str, Any]:
    """Reopen one pointer under custody checks without resolving its package."""

    pointer, _payload = _read_pointer(pointer_path)
    return pointer


def validate_shadow_pointer(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pointer without trusting any path it names."""

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        _fail("unified shadow pointer fields differ")
    pointer = dict(value)
    body = {key: item for key, item in pointer.items() if key != "pointer_sha256"}
    package_path = pointer.get("package_path")
    package_id = pointer.get("package_id")
    if (
        pointer.get("schema") != POINTER_SCHEMA
        or not isinstance(package_path, str)
        or not package_path
        or not Path(package_path).is_absolute()
        or os.path.normpath(package_path) != package_path
        or not isinstance(package_id, str)
        or not package_id
        or len(package_id) > 120
        or pointer.get("pointer_sha256") != _canonical_sha256(body)
    ):
        _fail("unified shadow pointer identity differs")
    _sha256(pointer.get("manifest_sha256"), role="manifest digest")
    return pointer


def build_shadow_pointer(
    package: Path,
    *,
    package_id: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Build a canonical identity pointer for an already verified package."""

    body = {
        "schema": POINTER_SCHEMA,
        "package_path": str(package.expanduser().absolute()),
        "package_id": package_id,
        "manifest_sha256": manifest_sha256,
    }
    return validate_shadow_pointer(
        {**body, "pointer_sha256": _canonical_sha256(body)}
    )


def resolve_shadow_pointer(pointer_path: Path, *, releases_root: Path) -> Path:
    """Resolve a stable pointer and reopen the selected package identity."""

    root = _private_directory(releases_root, create=False)
    pointer, _payload = _read_pointer(pointer_path)
    package = _reject_symlink_chain(Path(pointer["package_path"]), must_exist=True)
    try:
        relative = package.relative_to(root)
    except ValueError:
        _fail("unified shadow pointer package is outside the release root")
    if not relative.parts:
        _fail("unified shadow pointer package must be a strict release child")

    from core.brain.llm.unified_recurrent_shadow import inspect_shadow_package

    verified = inspect_shadow_package(package)
    manifest = verified.get("manifest")
    if not (
        isinstance(manifest, Mapping)
        and manifest.get("package_id") == pointer["package_id"]
        and manifest.get("manifest_sha256") == pointer["manifest_sha256"]
    ):
        _fail("unified shadow pointer package identity differs")
    return package


def shadow_pointer_publication_lock_path(pointer_path: Path) -> Path:
    """Return the cross-document lock shared with qualified authority."""

    return pointer_path.expanduser().absolute().parent / ".shadow-pointer-publication.lock"


def publish_shadow_pointer(
    package: Path,
    *,
    pointer_path: Path,
    releases_root: Path,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    activation_root = _private_directory(
        pointer_path.expanduser().absolute().parent,
        create=True,
    )
    with interprocess_file_lock(
        shadow_pointer_publication_lock_path(activation_root / pointer_path.name)
    ):
        return _publish_shadow_pointer_locked(
            package,
            pointer_path=pointer_path,
            releases_root=releases_root,
            expected_current_sha256=expected_current_sha256,
        )


def _publish_shadow_pointer_locked(
    package: Path,
    *,
    pointer_path: Path,
    releases_root: Path,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a verified package with compare-and-swap replacement."""

    root = _private_directory(releases_root, create=True)
    activation_root = _private_directory(pointer_path.expanduser().absolute().parent, create=True)
    if root.parent != activation_root:
        _fail("unified shadow pointer and release roots are unrelated")
    package = _reject_symlink_chain(package, must_exist=True)
    try:
        relative = package.relative_to(root)
    except ValueError:
        _fail("unified shadow package is outside the release root")
    if not relative.parts:
        _fail("unified shadow package must be a strict release child")

    from core.brain.llm.unified_recurrent_shadow import inspect_shadow_package

    verified = inspect_shadow_package(package)
    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        _fail("unified shadow package manifest is unavailable")
    pointer = build_shadow_pointer(
        package,
        package_id=str(manifest.get("package_id") or ""),
        manifest_sha256=str(manifest.get("manifest_sha256") or ""),
    )
    payload = _canonical_bytes(pointer)

    if pointer_path.exists() or pointer_path.is_symlink():
        current, current_payload = _read_pointer(pointer_path)
        if current_payload == payload:
            return current
        if (
            expected_current_sha256 is None
            or current["pointer_sha256"] != expected_current_sha256
        ):
            _fail("unified shadow pointer replacement lost compare-and-swap")
    elif expected_current_sha256 is not None:
        _fail("unified shadow pointer expected current value is absent")
    qualified = activation_root / "qualified-active.json"
    if qualified.exists() or qualified.is_symlink():
        _fail("qualified activation must be revoked before shadow pointer mutation")

    candidate = activation_root / f".{pointer_path.name}.{os.getpid()}.candidate"
    descriptor = -1
    try:
        descriptor = os.open(
            candidate,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        written = os.write(descriptor, payload)
        if written != len(payload):
            _fail("unified shadow pointer write was short")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(candidate, pointer_path)
        directory_fd = os.open(activation_root, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise UnifiedRecurrentShadowPointerError(
            "unified shadow pointer publication failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            candidate.unlink()
        # not a failure: a file that is already gone is the state this is reaching.
        except FileNotFoundError:
            pass
    resolved = resolve_shadow_pointer(pointer_path, releases_root=root)
    if resolved != package:
        _fail("unified shadow pointer publication reopened a different package")
    return pointer


def deactivate_shadow_pointer(
    *,
    pointer_path: Path,
    releases_root: Path,
    expected_current_sha256: str,
) -> dict[str, Any]:
    activation_root = _private_directory(pointer_path.expanduser().absolute().parent)
    with interprocess_file_lock(
        shadow_pointer_publication_lock_path(activation_root / pointer_path.name)
    ):
        return _deactivate_shadow_pointer_locked(
            pointer_path=pointer_path,
            releases_root=releases_root,
            expected_current_sha256=expected_current_sha256,
        )


def _deactivate_shadow_pointer_locked(
    *,
    pointer_path: Path,
    releases_root: Path,
    expected_current_sha256: str,
) -> dict[str, Any]:
    """Atomically remove active selection while retaining an immutable receipt."""

    current, payload = _read_pointer(pointer_path)
    _sha256(expected_current_sha256, role="expected current pointer")
    if current["pointer_sha256"] != expected_current_sha256:
        _fail("unified shadow pointer deactivation lost compare-and-swap")
    resolve_shadow_pointer(pointer_path, releases_root=releases_root)
    activation_root = _private_directory(pointer_path.expanduser().absolute().parent)
    qualified = activation_root / "qualified-active.json"
    if qualified.exists() or qualified.is_symlink():
        _fail("qualified activation must be revoked before shadow pointer mutation")
    retired = _private_directory(activation_root / "retired", create=True)
    destination = retired / f"{current['pointer_sha256']}.json"
    if destination.exists() or destination.is_symlink():
        archived, archived_payload = _read_pointer(destination)
        if archived != current or archived_payload != payload:
            _fail("unified shadow pointer retirement receipt differs")
        pointer_path.unlink()
    else:
        os.replace(pointer_path, destination)
    directory_fd = os.open(activation_root, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return current


__all__ = [
    "POINTER_SCHEMA",
    "UnifiedRecurrentShadowPointerError",
    "build_shadow_pointer",
    "deactivate_shadow_pointer",
    "default_shadow_activation_paths",
    "publish_shadow_pointer",
    "read_shadow_pointer",
    "resolve_shadow_pointer",
    "shadow_pointer_publication_lock_path",
    "validate_shadow_pointer",
]
