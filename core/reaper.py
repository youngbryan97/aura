"""
The Reaper: Aura's cross-process lifecycle manager.
Spawned before the Kernel. Survives SIGKILL of the Kernel.
Performs post-mortem cleanup when the Kernel disappears.
"""

import json
import logging
import multiprocessing.shared_memory as shm_lib
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

REAPER_MANIFEST_ENV = "AURA_REAPER_MANIFEST"
DEFAULT_REAPER_MANIFEST = Path(tempfile.gettempdir()) / "aura_reaper_manifest.json"
POLL_INTERVAL = 1.0  # seconds

logger = logging.getLogger("Aura.Reaper")


def _record_reaper_degradation(
    error: BaseException,
    *,
    stage: str,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage, "repair_requested": True}
    if extra:
        payload.update(extra)
    record_degradation(
        "reaper",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        extra=payload,
    )

def resolve_reaper_manifest_path() -> Path:
    """Resolve a single canonical manifest path for every runtime surface."""
    raw_path = os.environ.get(REAPER_MANIFEST_ENV, "").strip()
    if raw_path:
        return Path(raw_path).expanduser()
    return DEFAULT_REAPER_MANIFEST


REAPER_MANIFEST = resolve_reaper_manifest_path()


class ReaperManifest:
    """Tracks all resources the Reaper must clean up."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else resolve_reaper_manifest_path()
        self._data: dict[str, Any] = {"shm_names": [], "child_pids": [], "pipe_fds": []}
        self._load()

    def register_shm(self, name: str):
        if name not in self._data["shm_names"]:
            self._data["shm_names"].append(name)
        self._save()

    def register_pid(self, pid: int):
        if pid not in self._data["child_pids"]:
            self._data["child_pids"].append(pid)
        self._save()

    def deregister_shm(self, name: str):
        self._data["shm_names"] = [n for n in self._data["shm_names"] if n != name]
        self._save()

    def deregister_pid(self, pid: int):
        self._data["child_pids"] = [p for p in self._data["child_pids"] if p != pid]
        self._save()

    def _save(self):
        try:
            # Atomic write to prevent corruption (Windows compatible)
            import tempfile
            fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.path))
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self._data, f)
                os.replace(temp_path, self.path)
            except (RuntimeError, AttributeError, TypeError, ValueError):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_reaper_degradation(
                e,
                stage="manifest_save",
                action="kept in-memory reaper manifest after manifest save failed",
                severity="degraded",
                extra={"path": str(self.path)},
            )
            logger.error("[REAPER] Manifest save failed: %s", e)

    def _load(self):
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, TypeError, ValueError) as _e:
            _record_reaper_degradation(
                _e,
                stage="manifest_load",
                action="started fresh reaper manifest after persisted manifest could not be decoded",
                severity="degraded",
                extra={"path": str(self.path)},
            )
            # If corrupt or missing, start fresh
            logger.debug('Ignored Exception in reaper.py: %s', _e)

def reaper_loop(kernel_pid: int, manifest_path: Path):
    """
    The Reaper process main loop.
    Polls kernel liveness. On death, executes cleanup in strict order.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, RuntimeError, ValueError) as exc:
        _record_reaper_degradation(
            exc,
            stage="signal_setup",
            action="continued reaper loop without overriding SIGINT handling",
            severity="warning",
        )
        logger.debug("[REAPER] Unable to ignore SIGINT: %s", exc)
    # Configure logging for the detached process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    manifest = ReaperManifest(manifest_path)

    logger.info("[REAPER] Watching Kernel PID %d", kernel_pid)

    # Poll until kernel dies
    is_kernel_alive = True
    while is_kernel_alive:
        try:
            os.kill(kernel_pid, 0)  # Signal 0: existence check only
        except ProcessLookupError:
            logger.warning("[REAPER] Kernel PID %d is GONE. Initiating cleanup.", kernel_pid)
            is_kernel_alive = False
            _execute_cleanup(manifest)
            break
        except PermissionError as _e:
            logger.debug('Ignored PermissionError in reaper.py: %s', _e)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_reaper_degradation(
                e,
                stage="kernel_liveness_probe",
                action="continued polling after non-fatal kernel liveness probe failure",
                severity="warning",
                extra={"kernel_pid": kernel_pid},
            )
            logger.debug("[REAPER] Existence check failed (non-fatal): %s", e)
            
        if is_kernel_alive:
            time.sleep(POLL_INTERVAL)

def _execute_cleanup(manifest: ReaperManifest) -> dict[str, Any]:
    """Execute cleanup in order: children first, then shared memory."""
    summary: dict[str, Any] = {
        "terminated_pids": [],
        "missing_pids": [],
        "failed_pids": [],
        "unlinked_shm": [],
        "missing_shm": [],
        "failed_shm": [],
        "manifest_removed": False,
    }

    # 1. Terminate orphaned child processes
    child_pids: list[int] = manifest._data.get("child_pids", [])
    for pid in list(child_pids):
        cleaned_pid = False
        try:
            logger.info("[REAPER] Cleaning up PID %d", pid)
            os.kill(pid, signal.SIGTERM)
            # Short grace period
            for _ in range(5):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                # Force kill if still alive
                os.kill(pid, signal.SIGKILL)
                logger.warning("[REAPER] Force-killed orphan PID %d", pid)
            cleaned_pid = True
            summary["terminated_pids"].append(pid)
        except ProcessLookupError as _e:
            cleaned_pid = True
            summary["missing_pids"].append(pid)
            logger.debug('Ignored ProcessLookupError in reaper.py: %s', _e)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            summary["failed_pids"].append(pid)
            _record_reaper_degradation(
                e,
                stage="pid_cleanup",
                action="kept PID in reaper manifest for a future cleanup attempt",
                severity="degraded",
                extra={"pid": pid},
            )
            logger.error("[REAPER] Failed to kill PID %d: %s", pid, e)
        if cleaned_pid:
            manifest.deregister_pid(pid)

    # 2. Unlink named shared memory segments
    shm_names: list[str] = manifest._data.get("shm_names", [])
    for name in list(shm_names):
        cleaned_shm = False
        try:
            # We must attach before we can unlink in some versions,
            # or use the internal shm_unlink if available.
            # Python's SharedMemory makes this easy with unlink()
            try:
                segment = shm_lib.SharedMemory(name=name)
                segment.close()
                segment.unlink()
                logger.info("[REAPER] Unlinked SHM segment: %s", name)
                cleaned_shm = True
                summary["unlinked_shm"].append(name)
            except FileNotFoundError as _e:
                cleaned_shm = True
                summary["missing_shm"].append(name)
                logger.debug('Ignored FileNotFoundError in reaper.py: %s', _e)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            summary["failed_shm"].append(name)
            _record_reaper_degradation(
                e,
                stage="shm_cleanup",
                action="kept shared-memory name in reaper manifest for a future cleanup attempt",
                severity="degraded",
                extra={"shm_name": name},
            )
            logger.error("[REAPER] Failed to unlink SHM %s: %s", name, e)
        if cleaned_shm:
            manifest.deregister_shm(name)

    # 3. Clean up the manifest file itself
    unresolved = bool(manifest._data.get("child_pids") or manifest._data.get("shm_names"))
    if unresolved:
        manifest._save()
        logger.warning("[REAPER] Cleanup incomplete; manifest retained for retry.")
    else:
        try:
            manifest.path.unlink(missing_ok=True)
            summary["manifest_removed"] = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
            _record_reaper_degradation(
                _e,
                stage="manifest_remove",
                action="completed resource cleanup but left manifest file after unlink failed",
                severity="warning",
                extra={"path": str(manifest.path)},
            )
            logger.debug('Ignored Exception in reaper.py: %s', _e)

    logger.info("[REAPER] Cleanup complete.")
    return summary

def register_reaper_pid(pid: int):
    """Convenience helper for kernel components."""
    ReaperManifest().register_pid(pid)

def register_reaper_shm(name: str):
    """Convenience helper for kernel components."""
    ReaperManifest().register_shm(name)
