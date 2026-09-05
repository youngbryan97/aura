"""Metabolism Engine — Digital Homeostasis

Periodically scans the project tree and purges:
  - Temp files (.tmp, .cache, .pyc, __pycache__)
  - Stale log files older than days_threshold

Returns a report of bytes reclaimed and files removed.
Runs as the first maintenance step in DreamerV2.engage_sleep_cycle().
"""
import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.runtime.errors import record_degradation

# The backward-compatibility shim that re-exported MetabolismService from
# here is gone: nothing imported it from this path. What callers actually
# want from this module is MetabolismEngine, and the shim was one package
# reaching into another for a name nobody asked it for.

logger = logging.getLogger("Kernel.Metabolism")

WASTE_EXTENSIONS = {".tmp", ".cache", ".pyc"}
WASTE_DIRS = {"__pycache__"}
LOG_EXTENSION = ".log"


@dataclass
class PurgeReport:
    files_removed: int = 0
    dirs_removed: int = 0
    bytes_reclaimed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __str__(self) -> str:
        mb = self.bytes_reclaimed / (1024 * 1024)
        return (
            f"Metabolism: {self.files_removed} files, {self.dirs_removed} dirs removed "
            f"({mb:.2f} MB reclaimed) in {self.duration_s:.1f}s"
        )


class MetabolismEngine:
    """Biological waste-removal system.
    Scans root_dir for temp artifacts and stale logs, purges them safely.
    """

    def __init__(
        self,
        root_dir: Path | None = None,
        days_threshold: int = 7,
        protected_dirs: set | None = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.days_threshold = days_threshold
        self.protected_dirs = protected_dirs or {
            ".git", "node_modules", "venv", ".venv", "backups", "dist", ".tox",
        }

    async def scan_and_purge(self) -> PurgeReport:
        return await asyncio.to_thread(self._scan_and_purge_sync)

    def _scan_and_purge_sync(self) -> PurgeReport:
        report = PurgeReport()
        t0 = time.monotonic()
        logger.info("🫀 Metabolism sweep starting at %s", self.root_dir)
        try:
            self._purge_waste(report)
            self._purge_stale_logs(report)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('metabolism', exc)
            msg = f"Metabolism sweep error: {exc}"
            logger.error(msg, exc_info=True)
            report.errors.append(msg)
        report.duration_s = time.monotonic() - t0
        logger.info("🫀 %s", report)
        return report

    def _purge_waste(self, report: PurgeReport) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root_dir, topdown=True):
            dp = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in self.protected_dirs]
            for dname in list(dirnames):
                if dname in WASTE_DIRS:
                    target = dp / dname
                    removed, size = self._remove_waste_dir(target)
                    if removed:
                        report.dirs_removed += 1
                        report.bytes_reclaimed += size
                        dirnames.remove(dname)
            for fname in filenames:
                fpath = dp / fname
                if fpath.suffix in WASTE_EXTENSIONS:
                    try:
                        size = fpath.stat().st_size
                        fpath.unlink(missing_ok=True)
                        report.files_removed += 1
                        report.bytes_reclaimed += size
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        logger.debug("Metabolism left volatile waste file in place: %s: %s", fpath, exc)

    def _purge_stale_logs(self, report: PurgeReport) -> None:
        cutoff = time.time() - (self.days_threshold * 86400)
        for dirpath, dirnames, filenames in os.walk(self.root_dir, topdown=True):
            dp = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in self.protected_dirs]
            for fname in filenames:
                fpath = dp / fname
                if fpath.suffix == LOG_EXTENSION:
                    try:
                        if fpath.stat().st_mtime < cutoff:
                            size = fpath.stat().st_size
                            fpath.unlink()
                            report.files_removed += 1
                            report.bytes_reclaimed += size
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation('metabolism', exc)
                        report.errors.append(f"stale log {fpath}: {exc}")

    def _remove_waste_dir(self, target: Path) -> tuple[bool, int]:
        """Best-effort removal for volatile cache directories.

        ``__pycache__`` directories are actively recreated while Aura imports
        modules. A race here is housekeeping noise, not a runtime degradation.
        """
        size = self._dir_size(target)
        for _ in range(2):
            if not target.exists():
                return False, 0
            shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                return True, size
            time.sleep(0.02)
        logger.debug("Metabolism left live cache directory in place: %s", target)
        return False, 0

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("Suppressed: %s", exc)
        return total
