"""
core/ops/lymphatic_reaper.py
Enterprise Maintenance: Cleans uporphaned processes, stale file handles, and fragments.
Inspired by the biological lymphatic system.
"""
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import os
import psutil
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.container import ServiceContainer
from core.observability.metrics import get_metrics

logger = logging.getLogger("Aura.Reaper")
metrics = get_metrics()

class LymphaticReaper:
    def __init__(self, interval_s: float = 300.0):
        self._interval = interval_s
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._data_dir = Path(os.environ.get("AURA_DATA_DIR", "~/.aura/data")).expanduser()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._run_loop())
        logger.info("🛡️ Lymphatic Reaper active (Interval: %.1fs)", self._interval)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in lymphatic_reaper.py: %s', _e)
        logger.info("🛡️ Lymphatic Reaper shutdown.")

    async def _run_loop(self):
        while self._running:
            try:
                await self.sweep()
            except Exception as e:
                logger.error("Reaper sweep failed: %s", e)
            await asyncio.sleep(self._interval)

    async def sweep(self):
        """Execute all maintenance tasks."""
        start_time = time.time()
        logger.debug("🧼 Starting lymphatic sweep...")
        
        proc_cleaned = self._hunt_orphans()
        fs_cleaned = self._filesystem_sweep()
        mem_freed = self._defragment_memory()
        
        duration = time.time() - start_time
        logger.info(
            "🧼 Sweep complete: %d procs reaped, %.1fMB storage reclaimed. (Duration: %.2fs)",
            proc_cleaned, fs_cleaned / (1024*1024), duration
        )
        
        metrics.gauge("reaper.sweep_duration_s", duration)
        metrics.increment("reaper.sweeps_total")

    def _hunt_orphans(self) -> int:
        """Find and terminate orphaned child processes."""
        count = 0
        current_proc = psutil.Process()
        for child in current_proc.children(recursive=True):
            try:
                # Check if process is a zombie or hanging
                if child.status() == psutil.STATUS_ZOMBIE:
                    child.wait()
                    count += 1
                elif time.time() - child.create_time() > 3600: # 1 hour limit for anonymous children
                    child.terminate()
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.debug('Ignored Exception in lymphatic_reaper.py: %s', "unknown_error")
        return count

    def _filesystem_sweep(self) -> int:
        """Clean temporary files, stale locks, and logs."""
        reclaimed = 0
        tmp_dir = self._data_dir / "tmp"
        if tmp_dir.exists():
            for f in tmp_dir.glob("*"):
                try:
                    # Remove files older than 24 hours
                    if time.time() - f.stat().st_mtime > 86400:
                        if f.is_file():
                            reclaimed += f.stat().st_size
                            f.unlink()
                        elif f.is_dir():
                            shutil.rmtree(f)
                except Exception as e:
                    logger.debug("Reaper: failed to clean %s: %s", f, e)
        return reclaimed

    def _defragment_memory(self) -> bool:
        """Clear internal caches and trigger GC."""
        import gc
        gc.collect()
        
        # If we have a vector store or MLX model, trigger their specific clears if possible
        # For now, just a generic log
        return True

_reaper: Optional[LymphaticReaper] = None

def get_reaper() -> LymphaticReaper:
    global _reaper
    if _reaper is None:
        _reaper = LymphaticReaper()
    return _reaper
