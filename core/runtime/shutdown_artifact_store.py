"""Canonical low-level owner for shutdown forensic artifacts.

Shutdown evidence must remain writable after governance and ordinary runtime
services have begun teardown. This module owns only that control-plane storage;
user and application file effects continue through ``FileWriteGateway``.
"""

from __future__ import annotations

from pathlib import Path

from core.runtime.atomic_writer import (
    PathLike,
    atomic_write_text,
    ensure_private_directory,
)
from core.runtime.state_ownership import state_root


def write_shutdown_artifact(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write one shutdown artifact in a private directory."""

    target = Path(path).expanduser()
    parent = target.parent
    if not parent.exists():
        ensure_private_directory(parent)
    else:
        aura_root = state_root()
        if parent == aura_root or parent.is_relative_to(aura_root):
            parent.chmod(0o700)
    atomic_write_text(target, str(text), encoding=encoding)
    return target


def delete_shutdown_artifact(path: PathLike) -> bool:
    """Delete one bounded-history shutdown artifact if it still exists."""

    target = Path(path).expanduser()
    try:
        target.unlink()
    # not a failure: the artifact is already gone, and False is this
    # function's word for "there was nothing to delete".
    except FileNotFoundError:
        return False
    return True
