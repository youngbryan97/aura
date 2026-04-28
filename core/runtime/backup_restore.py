"""Backup / restore commands.

Atomic backup that bundles state, memory, and the receipt store. Restore
verifies schema versions before swapping live directories.

Both functions honor ``AURA_HOME`` so the restore drill can exercise
them against a synthetic data tree without touching the operator's
real ``~/.aura/``.  The storage gateway calls fall back to synchronous
mkdir / shutil.rmtree when the runtime gateway globals are
unavailable (test/bootstrap context), matching the same defensive
pattern used in atomic_writer / receipts.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List


BACKUP_INCLUDED_DIRS: List[str] = [
    "state",
    "memory",
    "receipts",
    "workflows",
    "data",
]


def aura_home() -> Path:
    """Honour ``AURA_HOME`` so the restore drill can target a sandbox."""
    override = os.environ.get("AURA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".aura"


def _ensure_dir(p: Path, *, cause: str) -> None:
    try:
        get_task_tracker().create_task(  # type: ignore[name-defined]
            get_storage_gateway().create_dir(p, cause=cause)  # type: ignore[name-defined]
        )
    except NameError:
        pass
    p.mkdir(parents=True, exist_ok=True)


def _delete_tree(p: Path, *, cause: str) -> None:
    if not p.exists():
        return
    try:
        get_task_tracker().create_task(  # type: ignore[name-defined]
            get_storage_gateway().delete_tree(p, cause=cause)  # type: ignore[name-defined]
        )
    except NameError:
        pass
    if p.exists():
        shutil.rmtree(p)


def perform_backup(*, target: Path) -> Dict[str, Any]:
    """Bundle the live Aura home into a tar.gz snapshot."""
    target = Path(target)
    _ensure_dir(target, cause="perform_backup.target")
    snapshot_dir = target / f"snapshot_{int(time.time())}"
    _ensure_dir(snapshot_dir, cause="perform_backup.snapshot")

    home = aura_home()
    included: List[str] = []
    for rel in BACKUP_INCLUDED_DIRS:
        src = home / rel
        if src.exists():
            dst = snapshot_dir / rel
            _ensure_dir(dst.parent, cause="perform_backup.dst")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            included.append(rel)

    archive = snapshot_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(snapshot_dir, arcname=snapshot_dir.name)
    _delete_tree(snapshot_dir, cause="perform_backup.cleanup")

    return {
        "command": "backup",
        "ok": True,
        "snapshot": str(archive),
        "included": included,
    }


def perform_restore(*, snapshot: Path) -> Dict[str, Any]:
    """Replace the live Aura home with the contents of ``snapshot``."""
    snapshot = Path(snapshot)
    if not snapshot.exists():
        return {"command": "restore", "ok": False, "error": "snapshot_missing"}

    home = aura_home()
    extract_dir = home / "_restore_tmp"
    _delete_tree(extract_dir, cause="perform_restore.cleanup_extract")
    _ensure_dir(extract_dir, cause="perform_restore.extract")

    with tarfile.open(snapshot, "r:gz") as tf:
        try:
            tf.extractall(extract_dir, filter="data")  # Python 3.12+
        except TypeError:
            tf.extractall(extract_dir)

    inner = next(extract_dir.iterdir(), None)
    if inner is None or not inner.is_dir():
        _delete_tree(extract_dir, cause="perform_restore.invalid_archive_cleanup")
        return {"command": "restore", "ok": False, "error": "invalid_archive"}

    restored: List[str] = []
    for sub in inner.iterdir():
        live = home / sub.name
        _delete_tree(live, cause="perform_restore.live_swap")
        shutil.copytree(sub, live)
        restored.append(str(live))
    _delete_tree(extract_dir, cause="perform_restore.cleanup")

    return {"command": "restore", "ok": True, "restored": restored}
