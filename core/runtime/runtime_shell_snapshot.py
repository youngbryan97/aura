"""Immutable in-process browser-shell snapshots keyed by signed runtime revision."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from core.runtime.launch_provenance import (
    RUNTIME_SHELL_ASSETS,
    runtime_shell_assets_digest,
)

_MAX_RETAINED_REVISIONS = 4
_REVISION_PATTERN = frozenset("0123456789abcdef")
_PATH_TO_ASSET = {
    "/" if relative == "interface/static/index.html" else "/" + relative.removeprefix("interface/"): relative
    for relative in RUNTIME_SHELL_ASSETS
}
_PATH_TO_ASSET["/static/index.html"] = "interface/static/index.html"


@dataclass(frozen=True, slots=True)
class RuntimeShellSnapshot:
    revision_token: str
    shell_assets_sha256: str
    captured_at_unix: float
    assets: Mapping[str, bytes]


_LOCK = threading.RLock()
_SNAPSHOTS: OrderedDict[str, RuntimeShellSnapshot] = OrderedDict()


def _valid_revision(value: str) -> bool:
    return len(value) == 64 and all(character in _REVISION_PATTERN for character in value)


def runtime_shell_request_path(path: str) -> bool:
    return str(path or "") in _PATH_TO_ASSET


def publish_runtime_shell_snapshot(
    *,
    revision_token: str,
    shell_assets_sha256: str,
    assets: Mapping[str, bytes],
) -> RuntimeShellSnapshot:
    """Publish only a complete snapshot whose bytes match its signed shell digest."""

    revision = str(revision_token or "").strip().lower()
    expected_digest = str(shell_assets_sha256 or "").strip().lower()
    if not _valid_revision(revision) or not _valid_revision(expected_digest):
        raise RuntimeError("runtime shell snapshot identity is malformed")
    frozen_assets = {name: bytes(content) for name, content in assets.items()}
    actual_digest = runtime_shell_assets_digest(frozen_assets)
    if actual_digest != expected_digest:
        raise RuntimeError("runtime shell snapshot bytes do not match signed digest")
    snapshot = RuntimeShellSnapshot(
        revision_token=revision,
        shell_assets_sha256=actual_digest,
        captured_at_unix=time.time(),
        assets=MappingProxyType(frozen_assets),
    )
    with _LOCK:
        _SNAPSHOTS[revision] = snapshot
        _SNAPSHOTS.move_to_end(revision)
        while len(_SNAPSHOTS) > _MAX_RETAINED_REVISIONS:
            _SNAPSHOTS.popitem(last=False)
    return snapshot


def runtime_shell_snapshot_asset(revision_token: str, request_path: str) -> bytes | None:
    revision = str(revision_token or "").strip().lower()
    relative = _PATH_TO_ASSET.get(str(request_path or ""))
    if not _valid_revision(revision) or relative is None:
        return None
    with _LOCK:
        snapshot = _SNAPSHOTS.get(revision)
        if snapshot is None:
            return None
        return snapshot.assets.get(relative)


def runtime_shell_snapshot_known(revision_token: str) -> bool:
    revision = str(revision_token or "").strip().lower()
    with _LOCK:
        return _valid_revision(revision) and revision in _SNAPSHOTS


def clear_runtime_shell_snapshots() -> None:
    with _LOCK:
        _SNAPSHOTS.clear()


__all__ = [
    "RuntimeShellSnapshot",
    "clear_runtime_shell_snapshots",
    "publish_runtime_shell_snapshot",
    "runtime_shell_request_path",
    "runtime_shell_snapshot_asset",
    "runtime_shell_snapshot_known",
]
