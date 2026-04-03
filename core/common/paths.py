from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CORE_DIR.parent

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
    p = aura_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p

def aura_logs_dir() -> Path:
    """Returns the logs directory, creating it if it doesn't exist."""
    p = aura_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def aura_backups_dir() -> Path:
    """Returns the backups directory, creating it if it doesn't exist."""
    p = aura_root() / "backups"
    p.mkdir(parents=True, exist_ok=True)
    return p

def aura_error_logs_dir() -> Path:
    """Returns the error logs directory, creating it if it doesn't exist."""
    p = aura_data_dir() / "error_logs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def aura_vault_dir() -> Path:
    p = aura_root() / "vault"
    p.mkdir(parents=True, exist_ok=True)
    return p

# v1.0.1: Moved to end of file to prevent circular import issues during early boot
DATA_DIR = aura_data_dir()