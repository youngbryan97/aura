#!/usr/bin/env python3
"""What holds the disk, whether anything bounds it, and what nothing reads.

Q02 asks for retention that bounds logs, exports, checkpoints and caches while
keeping live dependencies, unique source work, state and scientific evidence.
The first half of that is knowing which is which. This walks the places Aura
writes and reports each as one of:

  bounded      a mechanism holds it (rotation, a row cap with pruning)
  live         something running or configured reads it
  unreferenced nothing names it; listed with its size and provenance
  unknown      the tool cannot tell, and says so rather than guessing

It deletes nothing. `--prune-orphans` removes only entries the tool classified
unreferenced AND that the owner names on the command line, one at a time — a
checkpoint is the owner's to remove.

    python tools/disk_retention.py            # the report
    python tools/disk_retention.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core.runtime.sqlite_support import connecting

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AURA_HOME = Path(os.environ.get("AURA_HOME", "~/.aura")).expanduser()
GIB = 1024**3


def _du(path: Path) -> int:
    """Bytes on disk, from `du`: block counts, hardlinks once, sparse files as
    stored. Walking a 150GB tree file by file took longer than the report is
    worth, and apparent size is not what the disk is out of."""
    try:
        done = subprocess.run(
            ["du", "-sk", str(path)], capture_output=True, text=True, timeout=600,
        )
        return int(done.stdout.split()[0]) * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _gb(n: int) -> float:
    return round(n / GIB, 1)


def _age_days(path: Path) -> float:
    try:
        return round((time.time() - path.stat().st_mtime) / 86400, 1)
    except OSError:
        return -1.0


def logs() -> dict[str, Any]:
    """Rotation is the bound: N backups of maxBytes each."""
    folder = AURA_HOME / "logs"
    size = _du(folder) if folder.is_dir() else 0
    try:
        from core.observability import logging_config

        max_bytes = int(getattr(logging_config, "DEFAULT_MAX_BYTES", 0) or 0)
        backups = int(getattr(logging_config, "DEFAULT_BACKUP_COUNT", 0) or 0)
    except Exception:  # noqa: BLE001 — the report says unknown rather than stopping
        max_bytes, backups = 0, 0
    unrotated = []
    if folder.is_dir():
        for child in folder.iterdir():
            if child.is_file() and child.stat().st_size > 200 * 1024 * 1024:
                unrotated.append({"path": str(child), "gb": _gb(child.stat().st_size)})
    return {
        "class": "logs",
        "path": str(folder),
        "gb": _gb(size),
        "status": "bounded" if not unrotated else "unbounded_file",
        "bound": f"RotatingFileHandler maxBytes={max_bytes} backupCount={backups}" if max_bytes else "rotation (bounds not read)",
        "unrotated": unrotated,
    }


def state_stores() -> list[dict[str, Any]]:
    """The vault's store is pruned and capped; anything else with a state_log is an orphan."""
    rows: list[dict[str, Any]] = []
    live = AURA_HOME / "data" / "aura_state.db"
    for candidate in sorted((AURA_HOME / "data").rglob("aura_state.db")):
        size = candidate.stat().st_size
        row: dict[str, Any] = {"class": "state_store", "path": str(candidate), "gb": _gb(size)}
        try:
            with connecting(
                sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=2.0)
            ) as conn:
                count, newest = conn.execute(
                    "SELECT COUNT(*), MAX(timestamp) FROM state_log"
                ).fetchone()
                row["rows"] = int(count or 0)
                row["last_write_days_ago"] = round((time.time() - float(newest)) / 86400, 1) if newest else None
        except sqlite3.Error as exc:
            row["error"] = str(exc)
        if candidate.resolve() == live.resolve():
            row["status"] = "live"
            row["bound"] = "StateRepository: DB_PAYLOAD_MAX_BYTES per row, prune every 100 commits, VACUUM every 1000"
        else:
            row["status"] = "unreferenced"
            row["note"] = (
                "the vault writes data/aura_state.db; this path was the proxy's standalone "
                "fallback, unbounded until 2026-09-13, and nothing reads it"
            )
        rows.append(row)
    return rows


def worktrees() -> list[dict[str, Any]]:
    """Agent worktrees under .claude/worktrees: size, age, and untracked bulk."""
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / ".claude" / "worktrees"
    if not base.is_dir():
        return rows
    for tree in sorted(base.iterdir()):
        if not tree.is_dir() or not (tree / ".git").exists():
            continue
        size = _du(tree)
        try:
            head = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-C", str(tree), "log", "-1", "--format=%ci"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            untracked = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-C", str(tree), "status", "--porcelain", "--untracked-files=normal"],
                capture_output=True, text=True, timeout=120,
            ).stdout
            untracked_dirs = [
                line[3:] for line in untracked.splitlines() if line.startswith("?? ")
            ]
        except (OSError, subprocess.SubprocessError):
            head, untracked_dirs = "", []
        biggest_untracked = []
        for rel in untracked_dirs:
            target = tree / rel
            if target.exists():
                n = _du(target)
                if n > GIB:
                    biggest_untracked.append({"path": rel, "gb": _gb(n)})
        rows.append({
            "class": "worktree",
            "path": str(tree),
            "gb": _gb(size),
            "last_commit": head,
            "age_days": _age_days(tree),
            "untracked_over_1gb": sorted(biggest_untracked, key=lambda r: -r["gb"]),
            "status": "unknown",
            "note": "a worktree is a session's; whether it is done is the session owner's to say",
        })
    return sorted(rows, key=lambda r: -r["gb"])


def _named_anywhere(needle: str) -> int:
    """Production code, tools, config, and every manifest beside the active one."""
    hits = 0
    try:
        done = subprocess.run(
            # The manifests beside the active one are `active.json.rollback`,
            # `active.json.staged`, `active.json.identity-backup`: not *.json,
            # and exactly the files that say what a rollback would restore.
            ["grep", "-rlF", "--include=*.py", "--include=*.json", "--include=active.json*",
             "--include=*.jsonl", "--include=*.toml", "--include=*.yaml", "--include=*.yml",
             "--include=*.jinja", "--include=*.mk",
             needle, "core", "interface", "config", "tools", "training/fused-model", "Makefile"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        hits = len([line for line in done.stdout.splitlines() if not line.endswith(".pyc")])
    except (OSError, subprocess.SubprocessError):
        return -1
    return hits


def fusions_and_adapters() -> list[dict[str, Any]]:
    """Each fused model and adapter directory: named by a manifest, code, or nothing.

    The active manifest names the live cortex; the rollback and identity-backup
    manifests name what a rollback would restore. A fusion none of them and no
    code names is the 32B era's, kept by nobody.
    """
    rows: list[dict[str, Any]] = []
    for base in (REPO_ROOT / "training" / "fused-model", REPO_ROOT / "training" / "adapters"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            size = _du(child)
            if size < GIB // 2:
                continue
            hits = _named_anywhere(child.name)
            rows.append({
                "class": "fusion" if base.name == "fused-model" else "adapter",
                "path": str(child),
                "gb": _gb(size),
                "age_days": _age_days(child),
                "named_by": hits,
                "status": "live" if hits > 0 else ("unknown" if hits < 0 else "unreferenced"),
                "note": "" if hits > 0 else "no manifest, code, config or tool names this directory",
            })
    return sorted(rows, key=lambda r: -r["gb"])


def big_directories() -> list[dict[str, Any]]:
    """The other large roots, with what bounds them where something does."""
    rows: list[dict[str, Any]] = []
    for path, bound in (
        (REPO_ROOT / "artifacts", "sealed evidence; canaries pin what they read by hash"),
        (AURA_HOME / "training-campaigns", "campaign outputs; nothing prunes them"),
        (AURA_HOME / "knowledge", "knowledge stores; nothing prunes them"),
        (AURA_HOME / "data" / "stream_of_being", "append-only record"),
        (AURA_HOME / "data" / "outcome_ledger.db", "append-only ledger"),
    ):
        if path.exists():
            rows.append({"class": "root", "path": str(path), "gb": _gb(_du(path)),
                         "status": "unknown", "bound": bound})
    return rows


def unreferenced_checkpoints() -> list[dict[str, Any]]:
    try:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import model_inventory

        record = model_inventory.inventory()
        return [
            {"class": "checkpoint", "path": r["path"], "gb": r["size_gb"], "status": "unreferenced",
             "name": r["name"]}
            for r in record["unreferenced_on_disk"] if r["size_gb"] > 0.05
        ]
    except Exception as exc:  # noqa: BLE001
        return [{"class": "checkpoint", "status": "unknown", "error": repr(exc)}]


def report() -> dict[str, Any]:
    entries = [logs(), *state_stores(), *worktrees(), *fusions_and_adapters(), *big_directories(), *unreferenced_checkpoints()]
    reclaimable = [e for e in entries if e.get("status") == "unreferenced"]
    return {
        "schema": "aura.disk_retention.v1",
        "at": time.time(),
        "entries": entries,
        "unreferenced_gb": round(sum(float(e.get("gb", 0) or 0) for e in reclaimable), 1),
        "unreferenced": reclaimable,
    }


def _table(record: dict[str, Any]) -> str:
    lines = [f"{'CLASS':<12} {'GB':>8}  {'STATUS':<13} PATH"]
    for e in record["entries"]:
        age = e.get("last_write_days_ago", e.get("age_days"))
        tail = f"  (last written {age}d ago)" if age is not None and e.get("status") == "unreferenced" else ""
        lines.append(f"{e['class']:<12} {str(e.get('gb', '')):>8}  {e.get('status', ''):<13} {e.get('path', e.get('error', ''))}{tail}")
        for u in e.get("untracked_over_1gb", []):
            lines.append(f"{'':<12} {str(u['gb']):>8}  {'untracked':<13}   └ {u['path']}")
        if e.get("unrotated"):
            for u in e["unrotated"]:
                lines.append(f"{'':<12} {str(u['gb']):>8}  {'unrotated':<13}   └ {u['path']}")
    lines.append("")
    lines.append(f"unreferenced by anything: {record['unreferenced_gb']}GB across {len(record['unreferenced'])} entries — listed, not removed")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    record = report()
    print(json.dumps(record, indent=2, default=str) if args.json else _table(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
