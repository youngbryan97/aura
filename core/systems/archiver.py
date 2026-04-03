"""Archive Engine — Vital-Log Preservation

Compresses designated log files into timestamped ZIP archives
*before* Metabolism purges them, ensuring a permanent history.
Runs as the very first step in the Dreamer sleep cycle.
"""
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("Kernel.Archiver")

DEFAULT_VITAL_LOGS = [
    "aura_errors.log",
    "integrity_audit.log",
    "modification_audit.log",
    "metabolism_report.log",
    "dreamer.log",
]


class ArchiveEngine:
    """Compresses vital log files into ZIP archives for long-term storage.
    Auto-prunes old archives beyond max_archives.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        archive_dir: Optional[Path] = None,
        vital_logs: Optional[List[str]] = None,
        max_archives: int = 30,
    ):
        from core.config import config
        self.log_dir = Path(log_dir) if log_dir else config.paths.log_dir
        self.archive_dir = Path(archive_dir) if archive_dir else config.paths.data_dir / "archives"
        self.vital_logs = vital_logs or DEFAULT_VITAL_LOGS
        self.max_archives = max_archives

    async def archive_vital_logs(self) -> Dict:
        """Compress vital logs into a timestamped ZIP. Returns summary dict."""
        logger.info("📦 Archive sweep starting in %s", self.log_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        found: List[Path] = []
        for name in self.vital_logs:
            candidate = self.log_dir / name
            if candidate.exists() and candidate.stat().st_size > 0:
                found.append(candidate)

        if not found:
            logger.info("📦 No vital logs to archive.")
            return {"archived": 0, "path": None}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.archive_dir / f"vital_logs_{ts}.zip"
        archived_count = 0
        total_bytes = 0

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for log_path in found:
                    try:
                        total_bytes += log_path.stat().st_size
                        zf.write(log_path, arcname=log_path.name)
                        archived_count += 1
                    except Exception as exc:
                        logger.warning("Failed to archive %s: %s", log_path.name, exc)
        except Exception as exc:
            logger.error("Failed to create archive %s: %s", zip_path, exc)
            return {"archived": 0, "path": None, "error": str(exc)}

        self._prune_old_archives()
        mb = total_bytes / (1024 * 1024)
        logger.info("📦 Archived %d logs (%.2f MB) → %s", archived_count, mb, zip_path.name)
        return {"archived": archived_count, "path": str(zip_path), "bytes_original": total_bytes}

    def _prune_old_archives(self) -> None:
        try:
            archives = sorted(
                self.archive_dir.glob("vital_logs_*.zip"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(archives) > self.max_archives:
                oldest = archives.pop(0)
                oldest.unlink()
                logger.debug("Pruned old archive: %s", oldest.name)
        except Exception as exc:
            logger.warning("Archive pruning failed: %s", exc)