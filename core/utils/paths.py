from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from core.runtime.flags import aura_log_dir_override, aura_root_override
from core.runtime.state_ownership import state_root

logger = logging.getLogger("core.utils.paths")

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CORE_DIR.parent


def _ensure_dir(p: Path, *, cause: str) -> Path:
    """Create ``p`` durably when the storage gateway is available; fall
    back to a synchronous mkdir for test/bootstrap contexts.

    Mirrors the same defensive pattern used by core.runtime.atomic_writer
    and core.runtime.receipts.
    """
    try:
        get_task_tracker().create_task(  # type: ignore[name-defined]
            get_storage_gateway().create_dir(p, cause=cause)  # type: ignore[name-defined]
        )
    except NameError as _exc:
        logger.debug("Suppressed %s in core.utils.paths: %s", type(_exc).__name__, _exc)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_paths() -> Dict[str, Path]:
    """Returns a dictionary of all core Aura paths for subsystems."""
    from core.config import config
    return {
        "root": aura_root(),
        "data": aura_data_dir(),
        "logs": aura_logs_dir(),
        "backups": aura_backups_dir(),
        "error_logs": aura_error_logs_dir(),
        "vault": aura_vault_dir(),
        "project_root": PROJECT_ROOT
    }


def aura_root() -> Path:
    """Returns the root directory for Aura data/logs, defaulting to ~/.aura"""
    override = aura_root_override()
    return Path(override or state_root()).expanduser().resolve()


def aura_data_dir() -> Path:
    """Returns the data directory, creating it if it doesn't exist."""
    return _ensure_dir(aura_root() / "data", cause="aura_data_dir")


def aura_logs_dir() -> Path:
    """Returns the logs directory, creating it if it doesn't exist."""
    return _ensure_dir(aura_root() / "logs", cause="aura_logs_dir")


def aura_backups_dir() -> Path:
    """Returns the backups directory, creating it if it doesn't exist."""
    return _ensure_dir(aura_root() / "backups", cause="aura_backups_dir")


def aura_error_logs_dir() -> Path:
    """Returns the error logs directory, creating it if it doesn't exist."""
    return _ensure_dir(aura_data_dir() / "error_logs", cause="aura_error_logs_dir")


def aura_vault_dir() -> Path:
    return _ensure_dir(aura_root() / "vault", cause="aura_vault_dir")


# ── Forensics roots ───────────────────────────────────────────────────────
#
# Two conventions grew side by side and nobody noticed they disagreed:
#
#   canonical   config.paths.data_dir / "error_logs"   ->  ~/.aura/data/error_logs
#   cwd-relative Path("data/error_logs")               ->  $CWD/data/error_logs
#
# The live runtime is launched from the source checkout, so every cwd-relative
# WRITER (aura_main's faulthandler, the flight recorder, the incident narrator)
# landed its artifacts in <checkout>/data/error_logs, while every canonical
# READER — including source_body's crash correlation — looked in
# ~/.aura/data/error_logs and found an empty directory. Measured 2026-08-06:
# 502 stall dumps, 5 crash dumps and 25 flight reports on the writer side; zero
# on the reader side. The crash-correlation path had been reporting "no crashes"
# because it was reading somewhere nothing was ever written. That is an absence
# of evidence rendered as evidence of absence, which is the failure mode this
# codebase is least allowed to have.
#
# One resolver now answers "where do forensics go", and readers walk every root
# that exists so artifacts written under the old convention stay visible instead
# of being silently orphaned. Nothing is moved: the live instance owns those
# files while it is running.
_LEGACY_FORENSICS_RELATIVE = Path("data/error_logs")


def forensics_root() -> Path:
    """The one directory forensics artifacts are written to.

    ``AURA_LOG_DIR`` wins, because that is the switch a hermetic run already
    sets and forensics written into the live record by a test are worse than no
    forensics at all — 58 test-driver dumps once polluted a triage ranking.
    Everything else lands under the canonical data dir.
    """
    override = aura_log_dir_override()
    if override:
        return _ensure_dir(Path(override) / "error_logs", cause="forensics_root:override")
    return aura_error_logs_dir()


def forensics_dir(kind: str) -> Path:
    """Canonical directory for one class of forensics artifact.

    ``kind`` is the subdirectory name — "crash", "stalls", "memory", "flight",
    "bags", "traces". Created on demand, because a forensics writer that fails
    on a missing directory loses exactly the evidence it exists to keep.
    """
    return _ensure_dir(forensics_root() / kind, cause=f"forensics_dir:{kind}")


def forensics_search_roots() -> list[Path]:
    """Every root a reader must consult, canonical first.

    The legacy cwd-relative root is included only when it exists and is not the
    canonical one, so a reader on a clean machine sees a single root and a
    reader on this machine still finds the artifacts already on disk.
    """
    roots: list[Path] = [forensics_root()]
    # An explicit override is EXCLUSIVE. AURA_LOG_DIR means "this is the
    # record" — for a hermetic run, for a replay, for a sandboxed audit — and
    # continuing to search the machine's real trees would let a test read 502
    # live stall dumps it never wrote. The legacy roots exist to stop history
    # being orphaned by the canonicalisation, which is a concern only when
    # nobody has said where the record is.
    if aura_log_dir_override():
        return roots
    try:
        legacy = _LEGACY_FORENSICS_RELATIVE.resolve()
    except OSError:
        return roots
    if legacy.is_dir() and legacy not in roots:
        roots.append(legacy)
    project_legacy = (PROJECT_ROOT / _LEGACY_FORENSICS_RELATIVE).resolve()
    if project_legacy.is_dir() and project_legacy not in roots:
        roots.append(project_legacy)
    return roots


def forensics_search_dirs(kind: str) -> list[Path]:
    """Every directory that may hold ``kind`` artifacts, canonical first.

    The canonical directory is always included even when it does not exist yet.
    Filtering it out for being empty is how a reader built before the first
    crash ends up permanently watching only the legacy tree — the same
    reader/writer split this resolver exists to close, reintroduced one level
    down. Callers skip directories that are not there; that is cheap, and it is
    the caller's business rather than the resolver's.
    """
    # forensics_dir, not forensics_root()/kind: it creates the directory.
    # With an exclusive override the canonical directory may not exist yet, and
    # a reader pointed at a path that is not there reports "no evidence" for
    # the same reason the original defect did. Creating an empty directory is
    # harmless and makes the reader and the writer agree by construction.
    canonical = forensics_dir(kind)
    dirs = [canonical]
    dirs.extend(
        root / kind
        for root in forensics_search_roots()
        if (root / kind).is_dir() and root / kind != canonical
    )
    return dirs


# v1.0.1: Moved to end of file to prevent circular import issues during early boot
DATA_DIR = aura_data_dir()
