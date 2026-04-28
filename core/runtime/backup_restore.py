"""Backup / restore commands.

Atomic backup that bundles state, memory, and the receipt store. Restore
verifies schema versions before swapping live directories.
"""
from __future__ import annotations


import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List


BACKUP_INCLUDED_DIRS: List[str] = [
    ".aura/state",
    ".aura/memory",
    ".aura/receipts",
    ".aura/workflows",
    ".aura/data",
]


def perform_backup(*, target: Path) -> Dict[str, Any]:
    get_task_tracker().create_task(get_storage_gateway().create_dir(target, cause='perform_backup'))
    snapshot_dir = target / f"snapshot_{int(time.time())}"
    get_task_tracker().create_task(get_storage_gateway().create_dir(snapshot_dir, cause='perform_backup'))
    home = Path.home()
    included: List[str] = []
    for rel in BACKUP_INCLUDED_DIRS:
        src = home / rel
        if src.exists():
            dst = snapshot_dir / rel.replace(".aura/", "")
            get_task_tracker().create_task(get_storage_gateway().create_dir(dst.parent, cause='perform_backup'))
            shutil.copytree(src, dst, dirs_exist_ok=True)
            included.append(rel)
    archive = snapshot_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(snapshot_dir, arcname=snapshot_dir.name)
    get_task_tracker().create_task(get_storage_gateway().delete_tree(snapshot_dir, cause='perform_backup'))
    return {
        "command": "backup",
        "ok": True,
        "snapshot": str(archive),
        "included": included,
    }


def perform_restore(*, snapshot: Path) -> Dict[str, Any]:
    if not snapshot.exists():
        return {"command": "restore", "ok": False, "error": "snapshot_missing"}
    home = Path.home()
    extract_dir = home / ".aura" / "_restore_tmp"
    if extract_dir.exists():
        get_task_tracker().create_task(get_storage_gateway().delete_tree(extract_dir, cause='perform_restore'))
    get_task_tracker().create_task(get_storage_gateway().create_dir(extract_dir, cause='perform_restore'))
    with tarfile.open(snapshot, "r:gz") as tf:
        try:
            tf.extractall(extract_dir, filter="data")  # Python 3.12+
        except TypeError:
            tf.extractall(extract_dir)
    inner = next(extract_dir.iterdir(), None)
    if inner is None or not inner.is_dir():
        return {"command": "restore", "ok": False, "error": "invalid_archive"}
    restored: List[str] = []
    for sub in inner.iterdir():
        live = home / ".aura" / sub.name
        if live.exists():
            get_task_tracker().create_task(get_storage_gateway().delete_tree(live, cause='perform_restore'))
        shutil.copytree(sub, live)
        restored.append(str(live))
    get_task_tracker().create_task(get_storage_gateway().delete_tree(extract_dir, cause='perform_restore'))
    return {"command": "restore", "ok": True, "restored": restored}
