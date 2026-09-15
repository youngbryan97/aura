"""Metabolism Engine — Digital Homeostasis

Periodically scans the project tree and purges temp files (.tmp, .cache,
.pyc, __pycache__). Returns a report of bytes reclaimed and files removed.
Runs as the first maintenance step in DreamerV2.engage_sleep_cycle().

It does not touch .log files. It used to delete any older than seven days
anywhere under the tree, and the tree holds artifacts/closeout/ — the
runner and detached logs of every sealed campaign, tracked in git. On
2026-09-15 it deleted 103 of them, 21 tracked. The runtime's own logs live
under the state root and the Archiver keeps those.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# The backward-compatibility shim that re-exported MetabolismService from
# here is gone: nothing imported it from this path. What callers actually
# want from this module is MetabolismEngine, and the shim was one package
# reaching into another for a name nobody asked it for.

logger = logging.getLogger("Kernel.Metabolism")

WASTE_EXTENSIONS = {".tmp", ".cache", ".pyc"}
WASTE_DIRS = {"__pycache__"}


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
        # `.claude` holds other sessions' worktrees — thirty-six of them on
        # the live host — and a directory with its own .git is somebody
        # else's checkout whatever it is called. The sweep walked all of them
        # (2026-09-15: 2,368 directories removed in 80.2s, during shutdown,
        # under three running campaigns) and deleted their caches and any
        # .tmp file they were writing.
        self.protected_dirs = protected_dirs or {
            ".git", "node_modules", "venv", ".venv", "backups", "dist", ".tox",
            ".claude", ".worktrees",
        }

    async def scan_and_purge(self) -> PurgeReport:
        return await asyncio.to_thread(self._scan_and_purge_sync)

    def _scan_and_purge_sync(self) -> PurgeReport:
        report = PurgeReport()
        t0 = time.monotonic()
        logger.info("🫀 Metabolism sweep starting at %s", self.root_dir)
        try:
            self._purge_waste(report)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('metabolism', exc)
            msg = f"Metabolism sweep error: {exc}"
            logger.error(msg, exc_info=True)
            report.errors.append(msg)
        report.duration_s = time.monotonic() - t0
        logger.info("🫀 %s", report)
        return report

    def _own_subdirectories(self, dp: Path, dirnames: list[str]) -> list[str]:
        """The subdirectories this sweep may enter: not protected, not a checkout."""
        return [
            d for d in dirnames
            if d not in self.protected_dirs and not (dp / d / ".git").exists()
        ]

    def _purge_waste(self, report: PurgeReport) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root_dir, topdown=True):
            dp = Path(dirpath)
            dirnames[:] = self._own_subdirectories(dp, dirnames)
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
