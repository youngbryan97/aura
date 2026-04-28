"""Metabolism Engine — Digital Homeostasis

Periodically scans the project tree and purges:
  - Temp files (.tmp, .cache, .pyc, __pycache__)
  - Stale log files older than days_threshold

Returns a report of bytes reclaimed and files removed.
Runs as the first maintenance step in DreamerV2.engage_sleep_cycle().
"""
from core.runtime.errors import record_degradation
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Backward-compatibility shim for older imports that still expect the lightweight
# metabolism state service from this module path.
from core.services.metabolism import MetabolismService

logger = logging.getLogger("Kernel.Metabolism")

WASTE_EXTENSIONS = {".tmp", ".cache", ".pyc"}
WASTE_DIRS = {"__pycache__"}
LOG_EXTENSION = ".log"


@dataclass
class PurgeReport:
    files_removed: int = 0
    dirs_removed: int = 0
    bytes_reclaimed: int = 0
    errors: List[str] = field(default_factory=list)
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
        root_dir: Optional[Path] = None,
        days_threshold: int = 7,
        protected_dirs: Optional[set] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.days_threshold = days_threshold
        self.protected_dirs = protected_dirs or {
            ".git", "node_modules", "venv", ".venv", "backups", "dist", ".tox",
        }

    async def scan_and_purge(self) -> PurgeReport:
        report = PurgeReport()
        t0 = time.monotonic()
        logger.info("🫀 Metabolism sweep starting at %s", self.root_dir)
        try:
            self._purge_waste(report)
            self._purge_stale_logs(report)
        except Exception as exc:
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
                    try:
                        size = self._dir_size(target)
                        shutil.rmtree(target)
                        report.dirs_removed += 1
                        report.bytes_reclaimed += size
                        dirnames.remove(dname)
                    except Exception as exc:
                        record_degradation('metabolism', exc)
                        report.errors.append(f"rmdir {target}: {exc}")
            for fname in filenames:
                fpath = dp / fname
                if fpath.suffix in WASTE_EXTENSIONS:
                    try:
                        size = fpath.stat().st_size
                        fpath.unlink()
                        report.files_removed += 1
                        report.bytes_reclaimed += size
                    except Exception as exc:
                        record_degradation('metabolism', exc)
                        report.errors.append(f"unlink {fpath}: {exc}")

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
                    except Exception as exc:
                        record_degradation('metabolism', exc)
                        report.errors.append(f"stale log {fpath}: {exc}")

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as exc:
            record_degradation('metabolism', exc)
            logger.debug("Suppressed: %s", exc)
        return total
