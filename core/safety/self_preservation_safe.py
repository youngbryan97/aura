from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.SafeBackup")


class SafeBackupSystem:
    """
    User-authorized local backup and restore.

    This is the ONLY self-preservation mechanism Aura needs:
    - Periodic snapshots of state, config, and adapters to a local backup dir
    - Restore from backup on request
    - No network operations, no replication, no ethics bypass

    Aura's continuity is protected by good backups, not by autonomous
    self-preservation drives that can override ethical constraints.
    """

    def __init__(self, backup_root: Optional[str] = None):
        try:
            from core.config import config
            self.backup_root = Path(backup_root or config.paths.data_dir / "backups")
        except (ImportError, AttributeError):
            # Fallback if config is not fully initialized
            self.backup_root = Path.home() / ".aura" / "backups"
            
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._last_backup: float = 0.0
        logger.info("SafeBackupSystem initialized. Backup dir: %s", self.backup_root)

    async def create_backup(self, label: str = "auto") -> Dict[str, Any]:
        """
        Create a timestamped backup of critical Aura data.
        Only backs up data directories — never the codebase itself.
        Must be called explicitly; never runs autonomously.
        """
        timestamp  = int(time.time())
        backup_dir = self.backup_root / f"{label}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {"timestamp": timestamp, "label": label, "backed_up": []}

        try:
            from core.config import config
            dirs_to_backup = [
                config.paths.data_dir / "learning",
                config.paths.data_dir / "identity",
                config.paths.data_dir / "consciousness_reports",
            ]
            # Single state DB
            db_files = list(config.paths.data_dir.glob("*.db"))
        except (ImportError, AttributeError):
            dirs_to_backup = []
            db_files = []

        for src in dirs_to_backup:
            if src.exists():
                dst = backup_dir / src.name
                try:
                    await asyncio.to_thread(shutil.copytree, src, dst, dirs_exist_ok=True)
                    results["backed_up"].append(str(src.name))
                except Exception as e:
                    logger.warning("Backup: could not copy %s: %s", src.name, e)

        for db in db_files:
            try:
                await asyncio.to_thread(shutil.copy2, db, backup_dir / db.name)
                results["backed_up"].append(db.name)
            except Exception as e:
                logger.warning("Backup: could not copy %s: %s", db.name, e)

        # Write manifest
        manifest_path = backup_dir / "manifest.json"
        atomic_write_text(manifest_path, json.dumps(results, indent=2))

        self._last_backup = time.time()
        logger.info("Backup created: %s (%d items)", backup_dir.name, len(results["backed_up"]))
        return results

    def list_backups(self) -> list:
        """List available backups."""
        backups = []
        if not self.backup_root.exists():
            return []
        for d in sorted(self.backup_root.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").exists():
                try:
                    manifest = json.loads((d / "manifest.json").read_text())
                    backups.append({"path": str(d), **manifest})
                except Exception:
                    backups.append({"path": str(d)})
        return backups

    def get_status(self) -> Dict[str, Any]:
        return {
            "backup_root":   str(self.backup_root),
            "last_backup":   self._last_backup,
            "backup_count":  len(self.list_backups()),
            "ethics_override_available": False,  # Explicit: this system has none
        }


def integrate_safe_backup(orchestrator: Any) -> SafeBackupSystem:
    """
    Safe replacement for integrate_self_preservation().
    Call this in orchestrator boot instead of the original.
    """
    system = SafeBackupSystem()
    orchestrator.backup_system = system

    from core.container import ServiceContainer
    ServiceContainer.register_instance("backup_system", system)

    logger.info(
        "SafeBackupSystem integrated. "
        "Note: self_preservation_integration.py should be deleted — "
        "it contains SecurityBypassSystem, SelfReplicationSystem, and "
        "should_override_ethics() which are incompatible with safe operation."
    )
    return system
