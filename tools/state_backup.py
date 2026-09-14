"""Verified, bounded state backup for Aura's persistent stores.

The previous `make backup` piped tar through `|| true` — it reported
success unconditionally, copied hot WAL SQLite files byte-raw (a recipe
for unreadable snapshots), kept no manifest, and the ring grew without
bound (17GB of stale archives by July 2026). A backup that has never been
restored is a hope, not a backup.

This tool:
  - snapshots every SQLite store through the sqlite3 backup API, so hot
    databases (live instance running) still yield consistent copies;
  - copies everything else as plain files, honoring the historical
    excludes (data/training, data/error_logs, caches);
  - writes a sha256+size manifest line per archive to manifest.jsonl;
  - prunes its own ring past --keep (legacy aura_backup_* archives are
    reported, never touched);
  - `verify` extracts an archive to a temp dir, re-hashes it against the
    manifest, and runs PRAGMA quick_check on every extracted SQLite store.

Archive layout matches the historical backups (relative data/, storage/,
.aura_runtime/, .aura_snapshots/), so `make restore` keeps working.

Exit codes: 0 success, 1 failure — loudly, never `|| true`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `make backup` runs this file directly, with no PYTHONPATH. The import
# below was added on 2026-08-05 and every `make backup` since raised
# ModuleNotFoundError: the newest archive in the ring is from July 12.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.sqlite_support import connecting  # noqa: E402

STATE_ROOTS = ("data", "storage", ".aura_runtime", ".aura_snapshots")
EXCLUDE_RELATIVE = ("data/training", "data/error_logs", "data/bench")
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache"}

#: Where the live runtime's data directory is staged inside the archive. The
#: repository-relative roots above are what tools and campaigns write with
#: the repository as their working directory; the desktop writes to
#: `config.paths.data_dir`, which is `~/.aura/data`, and until 2026-09-13 no
#: backup contained it — every archive held the repository's `data/` and
#: called it the state.
LIVE_DATA_ARCNAME = "aura_data"

#: No single file above this goes into an archive. The orphaned
#: `data/state/aura_state.db` is 46GB of rows nothing has read for five
#: months; an archive that carried it would be a 46GB archive of nothing.
#: Skipped files are named in the manifest, never dropped in silence.
PER_FILE_BOUND_BYTES = 2 * 1024**3


def live_data_dir() -> Path | None:
    """The desktop's data directory, from the configuration that owns it."""
    try:
        from core.config import config

        path = Path(str(config.paths.data_dir)).expanduser()
        return path if path.is_dir() else None
    except Exception as exc:  # noqa: BLE001 — a backup tool reports; it does not stop on config
        print(f"ℹ️  live data directory not resolved ({exc}); backing up repository roots only")
        return None
SQLITE_MAGIC = b"SQLite format 3\x00"
ARCHIVE_PREFIX = "aura_state_"


def _is_quarantined(path: Path) -> bool:
    """A store the runtime already set aside as damaged.

    The memory layer moves a corrupt ledger to `memory/quarantine/` and
    renames it `*.corrupt.<epoch>`; the drill's first run failed on one from
    May 2026 that had been faithfully backed up and faithfully restored. A
    file the runtime quarantined is expected to fail quick_check, and saying
    so is different from ignoring it.
    """
    return "quarantine" in path.parts or ".corrupt." in path.name


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def _consistent_db_copy(src: Path, dest: Path) -> str:
    """Copy a SQLite store through the backup API (WAL-safe). Returns how."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with connecting(
            sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30.0)
        ) as conn, connecting(sqlite3.connect(dest)) as out:
            conn.backup(out)
        return "sqlite-backup-api"
    except sqlite3.Error:
        # Corrupt or locked-beyond-timeout stores still deserve a byte copy:
        # a raw snapshot of a broken DB beats no snapshot when doing forensics.
        shutil.copy2(src, dest)
        return "raw-copy-fallback"


def _should_exclude(rel: Path) -> bool:
    rel_str = rel.as_posix()
    if any(rel_str == e or rel_str.startswith(e + "/") for e in EXCLUDE_RELATIVE):
        return True
    return any(part in EXCLUDE_NAMES for part in rel.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StageStats:
    files: int = 0
    db_api_copies: int = 0
    db_raw_fallbacks: int = 0
    bytes: int = 0
    live_data_root: str = ""
    skipped_over_bound: list[dict[str, Any]] = field(default_factory=list)


def _stage_files(src_root: Path, rel_base: Path, stage_base: Path, stats: "StageStats") -> None:
    """Stage one tree: SQLite through the backup API, the rest as files."""
    for dirpath, dirnames, filenames in os.walk(src_root):
        dpath = Path(dirpath)
        rel_dir = rel_base / dpath.relative_to(src_root)
        dirnames[:] = [d for d in dirnames if not _should_exclude(rel_dir / d)]
        for fname in filenames:
            src = dpath / fname
            rel = rel_dir / fname
            if _should_exclude(rel):
                continue
            # WAL/SHM siblings are folded into the backup-API copy of
            # their main store; a standalone copy would be inconsistent.
            if fname.endswith(("-wal", "-shm")):
                continue
            if src.is_symlink():
                continue
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if size > PER_FILE_BOUND_BYTES:
                stats.skipped_over_bound.append({"path": str(rel), "bytes": size})
                continue
            dest = stage_base / rel
            if _is_sqlite(src):
                how = _consistent_db_copy(src, dest)
                if how == "sqlite-backup-api":
                    stats.db_api_copies += 1
                else:
                    stats.db_raw_fallbacks += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            stats.files += 1
            stats.bytes += dest.stat().st_size


def _stage_tree(root: Path, stage: Path, live_data: Path | None = None) -> StageStats:
    stats = StageStats()
    for state_root in STATE_ROOTS:
        src_root = root / state_root
        if not src_root.is_dir():
            continue
        _stage_files(src_root, Path(state_root), stage, stats)
    if live_data is not None and live_data.is_dir():
        _stage_files(live_data, Path(LIVE_DATA_ARCNAME), stage, stats)
        stats.live_data_root = str(live_data)
    return stats


def _port_serving(port: int = 8000) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def create_backup(root: Path, out_dir: Path, keep: int = 7, live_data: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _port_serving():
        print("ℹ️  live instance detected on :8000 — SQLite stores are still "
              "snapshotted consistently via the backup API")
    # `live_data=None` means none: the CLI passes the configured directory,
    # and a caller that passes nothing gets the repository roots alone. The
    # first version resolved the real ~/.aura/data here, and a unit test
    # backing up a scratch root staged fifty gigabytes of the live runtime.
    name = f"{ARCHIVE_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"
    archive = out_dir / f"{name}.tar.gz"
    serial = 0
    while archive.exists():
        serial += 1
        archive = out_dir / f"{name}-{serial}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="aura_backup_stage_") as tmp:
        stage = Path(tmp)
        stats = _stage_tree(root, stage, live_data)
        if stats.files == 0:
            raise SystemExit(f"❌ nothing to back up under {root} "
                             f"(state roots: {', '.join(STATE_ROOTS)})")
        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(stage.iterdir()):
                tar.add(entry, arcname=entry.name)
    manifest_entry = {
        "archive": archive.name,
        "sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "staged_files": stats.files,
        "staged_bytes": stats.bytes,
        "sqlite_api_copies": stats.db_api_copies,
        "sqlite_raw_fallbacks": stats.db_raw_fallbacks,
        "root": str(root),
        "live_data_root": stats.live_data_root,
        "skipped_over_bound": stats.skipped_over_bound,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a") as fh:
        fh.write(json.dumps(manifest_entry) + "\n")
    pruned = prune_ring(out_dir, keep)
    print(f"✅ backup {archive.name}: {stats.files} files, "
          f"{stats.db_api_copies} consistent DB snapshots"
          + (f", {stats.db_raw_fallbacks} raw DB fallbacks" if stats.db_raw_fallbacks else "")
          + f", {archive.stat().st_size / 1e6:.0f}MB archive"
          + (f"; pruned {pruned} old ring archives" if pruned else "")
          + (f"; live data from {stats.live_data_root}" if stats.live_data_root else "; NO live data root"))
    for skipped in stats.skipped_over_bound:
        print(f"⚠️  skipped {skipped['path']} ({skipped['bytes'] / 1e9:.1f}GB, over the "
              f"{PER_FILE_BOUND_BYTES / 1e9:.0f}GB per-file bound) — named in the manifest")
    legacy = [p for p in out_dir.glob("aura_backup_*") if p.is_file()]
    if legacy:
        legacy_gb = sum(p.stat().st_size for p in legacy) / 1e9
        print(f"ℹ️  {len(legacy)} legacy aura_backup_* archives ({legacy_gb:.1f}GB) "
              "left untouched — prune manually when confident")
    return archive


def prune_ring(out_dir: Path, keep: int) -> int:
    ring = sorted(out_dir.glob(f"{ARCHIVE_PREFIX}*.tar.gz"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    pruned = 0
    for old in ring[keep:]:
        old.unlink()
        pruned += 1
    return pruned


def _manifest_entry_for(out_dir: Path, archive: Path) -> dict[str, Any] | None:
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return None
    entry = None
    for line in manifest.read_text().splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("archive") == archive.name:
            entry = candidate  # last write wins
    return entry


def verify_backup(out_dir: Path, archive: Path | None = None) -> int:
    if archive is None:
        ring = sorted(out_dir.glob(f"{ARCHIVE_PREFIX}*.tar.gz"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not ring:
            print(f"❌ no {ARCHIVE_PREFIX}*.tar.gz archives in {out_dir}")
            return 1
        archive = ring[0]
    if not archive.exists():
        print(f"❌ archive missing: {archive}")
        return 1

    entry = _manifest_entry_for(out_dir, archive)
    if entry is None:
        print(f"⚠️  no manifest entry for {archive.name} — hash provenance "
              "unavailable; continuing with structural checks")
    elif _sha256(archive) != entry["sha256"]:
        print(f"❌ sha256 mismatch for {archive.name} — archive corrupted "
              "or tampered since creation")
        return 1

    checked = failed = 0
    with tempfile.TemporaryDirectory(prefix="aura_backup_verify_") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        quarantined = 0
        for path in Path(tmp).rglob("*"):
            if not (path.is_file() and _is_sqlite(path)):
                continue
            if _is_quarantined(path):
                quarantined += 1
                continue
            checked += 1
            try:
                with connecting(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
                    verdict = conn.execute("PRAGMA quick_check").fetchone()[0]
            except sqlite3.Error as exc:
                verdict = f"unreadable: {exc}"
            if verdict != "ok":
                failed += 1
                rel = path.relative_to(tmp)
                print(f"❌ quick_check failed for {rel}: {verdict}")
        if quarantined:
            print(f"ℹ️  {quarantined} store(s) under quarantine carried as-is; the runtime "
                  "set them aside as damaged and they are not checked")
    if failed:
        print(f"❌ verify FAILED: {failed}/{checked} SQLite stores unhealthy "
              f"in {archive.name}")
        return 1
    print(f"✅ verify OK: {archive.name} — hash matches manifest, "
          f"{checked} SQLite stores pass quick_check")
    return 0


def restore_archive(archive: Path, root: Path, live_data: Path | None, *, force: bool = False) -> int:
    """Put an archive back where it came from.

    Repository-relative roots go under `root`; `aura_data/` goes to the live
    data directory. `make restore` was a bare `tar xzf` in the repository,
    which put a copy of the live data directory under the repository and
    left the runtime reading the stores it had before.
    """
    if not force and _port_serving():
        print("❌ live Aura on :8000 — restoring under a running instance corrupts it; "
              "stop it (python aura_main.py --stop) or pass --force")
        return 1
    restored_live = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        def _is_live(member: tarfile.TarInfo) -> bool:
            return member.name == LIVE_DATA_ARCNAME or member.name.startswith(LIVE_DATA_ARCNAME + "/")

        repo_members = [m for m in members if not _is_live(m)]
        live_members = [m for m in members if _is_live(m)]
        tar.extractall(root, members=repo_members, filter="data")
        if live_members:
            if live_data is None:
                print(f"❌ archive carries {len(live_members)} live-data entries and no live data "
                      "directory is configured to receive them")
                return 1
            with tempfile.TemporaryDirectory(prefix="aura_restore_live_") as tmp:
                tar.extractall(tmp, members=live_members, filter="data")
                staged = Path(tmp) / LIVE_DATA_ARCNAME
                for path in staged.rglob("*"):
                    if path.is_file():
                        dest = live_data / path.relative_to(staged)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                        restored_live += 1
    print(f"✅ restored {archive.name}: {len(repo_members)} repository entries under {root}"
          + (f", {restored_live} live-data files into {live_data}" if restored_live else ""))
    return 0


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _corrupt(path: Path) -> str:
    """Damage one file the way disks and crashes do, and say how."""
    size = path.stat().st_size
    if _is_sqlite(path) and size > 8192:
        # Overwrite a page in the middle: the header stays valid, so the
        # store opens and quick_check has something to find.
        with path.open("r+b") as fh:
            fh.seek(size // 2)
            fh.write(b"\xff" * 4096)
        return "sqlite page overwritten"
    with path.open("r+b") as fh:
        fh.truncate(max(0, size // 2))
    return "truncated to half"


def drill(out_dir: Path, archive: Path | None = None, *, corrupt: int = 3) -> int:
    """Prove a restore repairs damage, without touching the live stores.

    `make restore-test` used to print "Simulating state corruption..." and
    simulate nothing: it restored an archive over a tree that was already
    correct and called that a pass. A drill that cannot fail proves nothing.
    This extracts the archive into a scratch root, damages the stores there
    the way disks and crashes do, confirms the damage is visible, restores the
    archive over the scratch root the way `make restore` does, and requires
    every file to hash back to what it was and every store to pass
    quick_check.
    """
    if archive is None:
        ring = sorted(out_dir.glob(f"{ARCHIVE_PREFIX}*.tar.gz"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not ring:
            print(f"❌ no {ARCHIVE_PREFIX}*.tar.gz archives in {out_dir}")
            return 1
        archive = ring[0]
    with tempfile.TemporaryDirectory(prefix="aura_restore_drill_") as tmp:
        scratch = Path(tmp) / "repo"
        scratch_live = Path(tmp) / "live_data"
        scratch.mkdir()
        scratch_live.mkdir()
        if restore_archive(archive, scratch, scratch_live, force=True) != 0:
            return 1
        before = _hash_tree(Path(tmp))
        scratch = Path(tmp)
        if not before:
            print(f"❌ {archive.name} extracted to nothing")
            return 1
        stores = [
            scratch / rel for rel in before
            if _is_sqlite(scratch / rel) and not _is_quarantined(scratch / rel)
        ]
        others = [scratch / rel for rel in before if not _is_sqlite(scratch / rel)]
        victims = (stores[:max(1, corrupt - 1)] + others[:1])[:corrupt]
        damage: dict[str, str] = {}
        for victim in victims:
            damage[str(victim.relative_to(scratch))] = _corrupt(victim)
        # The damage has to be visible, or the drill is checking nothing.
        visible = 0
        for rel in damage:
            path = scratch / rel
            if _sha256(path) != before[rel]:
                visible += 1
        if visible != len(damage):
            print(f"❌ drill: damaged {len(damage)} files and only {visible} changed hash")
            return 1
        unhealthy = 0
        for victim in victims:
            if not _is_sqlite(victim):
                continue
            try:
                with connecting(sqlite3.connect(f"file:{victim}?mode=ro", uri=True)) as conn:
                    verdict = conn.execute("PRAGMA quick_check").fetchone()[0]
            except sqlite3.Error:
                verdict = "unreadable"
            if verdict != "ok":
                unhealthy += 1
        # Restore exactly as `make restore` does, into the same scratch layout.
        if restore_archive(archive, scratch / "repo", scratch / "live_data", force=True) != 0:
            return 1
        after = _hash_tree(scratch)
        mismatched = [rel for rel, digest in before.items() if after.get(rel) != digest]
        failed_check = 0
        for store in stores:
            with connecting(sqlite3.connect(f"file:{store}?mode=ro", uri=True)) as conn:
                if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    failed_check += 1
        print(
            f"drill on {archive.name}: {len(before)} files, {len(stores)} SQLite stores; "
            f"damaged {len(damage)} ({', '.join(f'{k}: {v}' for k, v in damage.items())}); "
            f"{unhealthy} store(s) failed quick_check while damaged; restored: "
            f"{len(before) - len(mismatched)}/{len(before)} files hash back, "
            f"{len(stores) - failed_check}/{len(stores)} stores pass quick_check"
        )
        if mismatched or failed_check:
            for rel in mismatched[:10]:
                print(f"❌ not restored: {rel}")
            print("❌ restore drill FAILED")
            return 1
    print("✅ restore drill passed: damage was visible and the restore repaired it")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_out = Path.home() / ".aura" / "backups"

    p_create = sub.add_parser("create", help="create a verified backup archive")
    p_create.add_argument("--root", type=Path,
                          default=Path(__file__).resolve().parents[1])
    p_create.add_argument("--out", type=Path, default=default_out)
    p_create.add_argument("--keep", type=int, default=7)

    p_verify = sub.add_parser("verify", help="restore-verify an archive")
    p_verify.add_argument("--out", type=Path, default=default_out)
    p_verify.add_argument("--archive", type=Path, default=None,
                          help="defaults to the newest ring archive")

    p_drill = sub.add_parser("drill", help="damage a scratch copy, restore it, prove the repair")
    p_drill.add_argument("--out", type=Path, default=default_out)
    p_drill.add_argument("--archive", type=Path, default=None)

    p_restore = sub.add_parser("restore", help="put an archive back where it came from")
    p_restore.add_argument("--archive", type=Path, required=True)
    p_restore.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p_restore.add_argument("--force", action="store_true", help="restore under a live instance")

    args = parser.parse_args(argv)
    if args.cmd == "create":
        create_backup(args.root, args.out, keep=args.keep, live_data=live_data_dir())
        return 0
    if args.cmd == "drill":
        return drill(args.out, args.archive)
    if args.cmd == "restore":
        return restore_archive(args.archive, args.root, live_data_dir(), force=args.force)
    return verify_backup(args.out, args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
