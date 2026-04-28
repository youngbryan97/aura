"""Automated Backup and Vacuum System for Aura.
Handles periodic database maintenance (VACUUM) and rotating compressed backups
of critical state to ensure Data Safety & Recovery on Apple Silicon.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.config import config
from core.runtime import background_policy

logger = logging.getLogger("Aura.Backup")

class BackupManager:
    """Manages SQLite VACUUMs and rotating backups of the data directory."""

    def __init__(self, max_backups: int = 5):
        self.max_backups = max_backups
        self.data_dir = config.paths.data_dir
        self.backup_dir = config.paths.home_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._is_running = False
        self._maintenance_registered = False
        self._backup_in_progress = False
        self._vacuum_in_progress = False
        self._last_backup_at = 0.0
        self._last_vacuum_at = 0.0
        self._last_backup_path: Optional[Path] = None
        self.vacuum_interval_s = 3600.0 * 12
        self.backup_interval_s = 3600.0 * 24

    def _discover_database_paths(self) -> List[Path]:
        candidates = sorted({path for path in self.data_dir.rglob("*.db") if path.is_file()})
        try:
            from core.resilience.database_coordinator import get_db_coordinator

            db_coord = get_db_coordinator()
            for path in getattr(db_coord, "_connections", {}).keys():
                candidates.append(Path(path))
        except Exception as exc:
            record_degradation('backup', exc)
            logger.debug("BackupManager database discovery skipped coordinator paths: %s", exc)
        deduped = []
        seen = set()
        for item in candidates:
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(item)
        return deduped

    def _list_backups_sync(self) -> List[Path]:
        return sorted(self.backup_dir.glob("aura_backup_*.zip"), key=os.path.getmtime)

    def _maintenance_block_reason(self) -> str:
        try:
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", default=None)
            return str(
                background_policy.background_activity_reason(
                    orchestrator,
                    profile=background_policy.MAINTENANCE_BACKGROUND_POLICY,
                )
                or ""
            )
        except Exception as exc:
            record_degradation('backup', exc)
            logger.debug("Maintenance background policy check skipped: %s", exc)
            return ""

    @staticmethod
    def _vacuum_database_sync(db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("VACUUM;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.commit()
        finally:
            conn.close()

    async def run_vacuum(self) -> bool:
        """Runs VACUUM on all registered SQLite databases to reclaim space."""
        if self._vacuum_in_progress:
            logger.info("VACUUM already in progress; skipping overlapping run.")
            return False
        reason = self._maintenance_block_reason()
        if reason:
            logger.info("Skipping periodic VACUUM during active runtime window: %s", reason)
            return False
        logger.info("Starting periodic database VACUUM...")
        self._vacuum_in_progress = True
        try:
            dbs_to_vacuum = await asyncio.to_thread(self._discover_database_paths)
            if not dbs_to_vacuum:
                logger.info("No SQLite databases discovered for VACUUM.")
                self._last_vacuum_at = time.time()
                return True

            for db_path in dbs_to_vacuum:
                logger.debug("Vacuuming %s...", db_path)
                try:
                    await asyncio.to_thread(self._vacuum_database_sync, db_path)
                except Exception as e:
                    record_degradation('backup', e)
                    logger.warning("Failed to vacuum %s: %s", db_path, e)

            logger.info("VACUUM operation complete.")
            self._last_vacuum_at = time.time()
            return True
        except Exception as e:
            record_degradation('backup', e)
            logger.error("VACUUM operation failed: %s", e)
            return False
        finally:
            self._vacuum_in_progress = False

    async def create_backup(self) -> Optional[Path]:
        """Creates a zip archive of the data directory."""
        if self._backup_in_progress:
            logger.info("Backup already in progress; skipping overlapping run.")
            return None
        reason = self._maintenance_block_reason()
        if reason:
            logger.info("Skipping periodic backup during active runtime window: %s", reason)
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"aura_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating backup: {backup_name}.zip")
        self._backup_in_progress = True
        try:
            # Reclaim space before backup
            await self.run_vacuum()
            
            # Use asyncio.to_thread for blocking IO
            await asyncio.to_thread(
                shutil.make_archive,
                str(backup_path),
                'zip',
                str(self.data_dir)
            )
            
            final_path = self.backup_dir / f"{backup_name}.zip"
            if final_path.exists():
                logger.info(f"Backup created successfully: {final_path}")
                await self._enforce_rotation()
                self._last_backup_at = time.time()
                self._last_backup_path = final_path
                return final_path
            return None
        except Exception as e:
            record_degradation('backup', e)
            logger.error(f"Backup failed: {e}")
            return None
        finally:
            self._backup_in_progress = False

    async def _enforce_rotation(self):
        """Enforces the maximum number of retained backups."""
        try:
            backups = await asyncio.to_thread(self._list_backups_sync)
            
            while len(backups) > self.max_backups:
                oldest = backups.pop(0)
                logger.debug(f"Removing old backup: {oldest}")
                await asyncio.to_thread(os.remove, oldest)
                
        except Exception as e:
            record_degradation('backup', e)
            logger.error(f"Failed to enforce backup rotation: {e}")

    async def ensure_recent_backup(self, max_age_s: Optional[float] = None) -> Optional[Path]:
        target_age = float(max_age_s or (self.backup_interval_s * 1.5))
        if self._backup_in_progress:
            return None
        if self._last_backup_at and (time.time() - self._last_backup_at) <= target_age:
            return None
        return await self.create_backup()

    async def on_start_async(self):
        """Service start hook. Registers with core scheduler."""
        self._is_running = True
        if self._maintenance_registered:
            return
        logger.info("BackupManager service started.")
        from core.scheduler import TaskSpec, scheduler

        # Maintenance jobs should not stampede the runtime during boot.
        now = time.monotonic()
        await scheduler.register(TaskSpec(
            name="periodic_db_vacuum",
            coro=self.run_vacuum,
            tick_interval=self.vacuum_interval_s,
            last_run=now,
        ))

        await scheduler.register(TaskSpec(
            name="periodic_state_backup",
            coro=self.create_backup,
            tick_interval=self.backup_interval_s,
            last_run=now,
        ))
        self._maintenance_registered = True

    async def on_stop_async(self):
        """Service stop hook."""
        self._is_running = False
        logger.info("BackupManager service stopped.")

    async def get_health(self):
        """Service health check."""
        backups = await asyncio.to_thread(self._list_backups_sync)
        latest_backup = backups[-1] if backups else self._last_backup_path
        latest_backup_age_s = (
            max(0.0, time.time() - self._last_backup_at)
            if self._last_backup_at
            else None
        )
        return {
            "status": "online",
            "backup_count": len(backups),
            "max_backups": self.max_backups,
            "latest_backup": latest_backup.name if latest_backup else None,
            "latest_backup_age_s": latest_backup_age_s,
            "last_backup_at": self._last_backup_at or None,
            "last_vacuum_at": self._last_vacuum_at or None,
            "scheduler_registered": self._maintenance_registered,
            "backup_interval_s": self.backup_interval_s,
            "vacuum_interval_s": self.vacuum_interval_s,
            "backup_in_progress": self._backup_in_progress,
            "vacuum_in_progress": self._vacuum_in_progress,
        }
