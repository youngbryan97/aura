# core/config_paths.py
from dataclasses import dataclass
from pathlib import Path
from core.common.paths import aura_root, aura_data_dir, aura_logs_dir, aura_backups_dir

@dataclass(frozen=True)
class AuraPaths:
    root: Path = aura_root()
    data_dir: Path = aura_data_dir()
    logs_dir: Path = aura_logs_dir()
    backups_dir: Path = aura_backups_dir()
