"""Tested backup -> wipe -> restore -> verify round-trip.

The platform decision for disaster recovery is "manual backups + a
*tested* restore drill."  This module is the test, runnable both
inside pytest and from the operator CLI:

    perform_drill(home_override=Path("/tmp/aura_drill"))

The drill:

  1. seeds the home with a known synthetic state,
  2. takes a backup,
  3. wipes the live tree,
  4. restores from the backup,
  5. recursively diffs the restored tree against the recorded
     fingerprint,
  6. returns a structured DrillReport with ``ok``, the file count,
     the fingerprint hash, and any mismatches.

A non-empty mismatch list fails the drill — it means the backup or
restore lost data.  The fingerprint is SHA-256 over a sorted
``rel_path:sha256`` listing so it's deterministic across runs.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.backup_restore import (
    BACKUP_INCLUDED_DIRS,
    perform_backup,
    perform_restore,
)


@dataclass
class DrillReport:
    ok: bool
    home: str
    snapshot: str
    file_count: int
    fingerprint_before: str
    fingerprint_after: str
    mismatches: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_tree(root: Path) -> Dict[str, str]:
    """Return ``{rel_path: sha256}`` for every regular file under root."""
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            out[rel] = _sha256_file(p)
    return out


def _fingerprint_hash(fp: Dict[str, str]) -> str:
    listing = "\n".join(f"{rel}:{sha}" for rel, sha in sorted(fp.items()))
    return "sha256:" + hashlib.sha256(listing.encode("utf-8")).hexdigest()


def _seed_synthetic_home(home: Path) -> Dict[str, str]:
    """Create a small but realistic Aura-shaped data tree.

    Returns the per-file fingerprint of the seeded tree so callers can
    compare it to the post-restore fingerprint.
    """
    home.mkdir(parents=True, exist_ok=True)
    layout = {
        "state/snapshot.json": '{"vitality": 0.8, "ts": 1}',
        "state/identity.json": '{"name": "aura"}',
        "memory/episodic/2026-04-28.jsonl": (
            '{"id": "ep1", "content": "first memory"}\n'
            '{"id": "ep2", "content": "second memory"}\n'
        ),
        "memory/semantic/concepts.jsonl": '{"id": "c1", "label": "smooth"}\n',
        "receipts/turn/turn-aaa.json": '{"kind": "turn", "id": "aaa"}',
        "receipts/governance/gov-bbb.json": '{"kind": "governance", "id": "bbb"}',
        "workflows/q1.jsonl": '{"step": 1}\n{"step": 2}\n',
        "data/prediction_ledger.db": "restore-drill ledger fixture",
        "data/curriculum_lessons.db": "restore-drill lesson fixture",
    }
    for rel, body in layout.items():
        target = home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body.encode("utf-8"))
    return _fingerprint_tree(home)


def perform_drill(
    *,
    home_override: Optional[Path] = None,
    backup_target_override: Optional[Path] = None,
    seed: bool = True,
) -> DrillReport:
    """Run the full backup -> wipe -> restore -> verify cycle.

    ``home_override`` runs the drill against a sandbox dir instead of
    the operator's real ~/.aura; the AURA_HOME env var is set+restored
    around the drill so backup_restore.py picks it up.

    When ``seed`` is True the function lays down a known synthetic
    tree first, which is what tests want.  Operators who want to
    verify their *current* live data set ``seed=False``.
    """
    started = time.monotonic()
    notes: List[str] = []

    if home_override is None:
        home_override = Path(tempfile.mkdtemp(prefix="aura_drill_home_"))
        notes.append("created ephemeral home_override")
    if backup_target_override is None:
        backup_target_override = Path(tempfile.mkdtemp(prefix="aura_drill_backup_"))
        notes.append("created ephemeral backup_target_override")

    home_override = Path(home_override).resolve()
    backup_target_override = Path(backup_target_override).resolve()

    prev_home = os.environ.get("AURA_HOME")
    os.environ["AURA_HOME"] = str(home_override)
    try:
        if seed:
            _seed_synthetic_home(home_override)
            notes.append("seeded synthetic home")

        before_fp = _fingerprint_tree(home_override)
        before_hash = _fingerprint_hash(before_fp)

        backup_info = perform_backup(target=backup_target_override)
        if not backup_info.get("ok"):
            return DrillReport(
                ok=False,
                home=str(home_override),
                snapshot="",
                file_count=len(before_fp),
                fingerprint_before=before_hash,
                fingerprint_after="",
                mismatches=["backup_failed"],
                elapsed_seconds=time.monotonic() - started,
                notes=notes,
            )
        snapshot_path = Path(backup_info["snapshot"])
        notes.append(f"snapshot at {snapshot_path}")

        # Wipe the live tree: only the directories that were backed up.
        for rel in BACKUP_INCLUDED_DIRS:
            target = home_override / rel
            if target.exists():
                shutil.rmtree(target)
        notes.append("wiped backup-included dirs")

        restore_info = perform_restore(snapshot=snapshot_path)
        if not restore_info.get("ok"):
            return DrillReport(
                ok=False,
                home=str(home_override),
                snapshot=str(snapshot_path),
                file_count=len(before_fp),
                fingerprint_before=before_hash,
                fingerprint_after="",
                mismatches=[
                    f"restore_failed:{restore_info.get('error', 'unknown')}"
                ],
                elapsed_seconds=time.monotonic() - started,
                notes=notes,
            )

        after_fp = _fingerprint_tree(home_override)
        after_hash = _fingerprint_hash(after_fp)

        mismatches: List[str] = []
        before_keys = set(before_fp.keys())
        after_keys = set(after_fp.keys())
        for missing in sorted(before_keys - after_keys):
            mismatches.append(f"missing_after_restore:{missing}")
        for extra in sorted(after_keys - before_keys):
            mismatches.append(f"extra_after_restore:{extra}")
        for shared in sorted(before_keys & after_keys):
            if before_fp[shared] != after_fp[shared]:
                mismatches.append(f"content_mismatch:{shared}")

        return DrillReport(
            ok=len(mismatches) == 0,
            home=str(home_override),
            snapshot=str(snapshot_path),
            file_count=len(before_fp),
            fingerprint_before=before_hash,
            fingerprint_after=after_hash,
            mismatches=mismatches,
            elapsed_seconds=time.monotonic() - started,
            notes=notes,
        )
    finally:
        if prev_home is None:
            os.environ.pop("AURA_HOME", None)
        else:
            os.environ["AURA_HOME"] = prev_home
