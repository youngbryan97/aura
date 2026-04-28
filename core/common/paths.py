from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

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
    except NameError:
        pass
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
    return Path(os.getenv("AURA_ROOT", Path.home() / ".aura")).expanduser().resolve()


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


# v1.0.1: Moved to end of file to prevent circular import issues during early boot
DATA_DIR = aura_data_dir()
