"""core/body/filesystem_sensor.py
Filesystem activity sensor tracking changes inside workspace directories.
"""
from pathlib import Path
import os
import time
from typing import Dict, Any, List
from core.body.sensor_registry import BaseSensor
from core.config import get_config
from core.runtime.errors import record_degradation


class FilesystemSensor(BaseSensor):
    """Monitors workspace file counts and recent modifications."""

    @property
    def name(self) -> str:
        return "filesystem"

    async def read(self) -> Dict[str, Any]:
        cfg = get_config()
        base_path = Path(cfg.paths.project_root)
        
        # Scan top level files for modifications in the last hour
        recent_changes: List[str] = []
        now = time.time()
        
        try:
            for item in base_path.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    mtime = item.stat().st_mtime
                    if now - mtime < 3600:
                        recent_changes.append(item.name)
        except OSError as exc:
            record_degradation("body.filesystem_sensor", exc)

        return {
            "monitored_root": str(base_path),
            "recent_modifications": recent_changes,
            "modified_count": len(recent_changes)
        }
