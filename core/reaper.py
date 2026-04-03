"""
The Reaper: Aura's cross-process lifecycle manager.
Spawned before the Kernel. Survives SIGKILL of the Kernel.
Performs post-mortem cleanup when the Kernel disappears.
"""

import os
import signal
import time
import json
import logging
import atexit
import multiprocessing.shared_memory as shm_lib
from pathlib import Path
from typing import List, Dict, Any

import tempfile
# Standardized path for the reaper manifest
REAPER_MANIFEST = Path(tempfile.gettempdir()) / "aura_reaper_manifest.json"
POLL_INTERVAL = 1.0  # seconds

logger = logging.getLogger("Aura.Reaper")

class ReaperManifest:
    """Tracks all resources the Reaper must clean up."""

    def __init__(self, path: Path = REAPER_MANIFEST):
        self.path = path
        self._data: Dict[str, Any] = {"shm_names": [], "child_pids": [], "pipe_fds": []}
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
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        except Exception as e:
            logger.error(f"[REAPER] Manifest save failed: {e}")

    def _load(self):
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text())
        except Exception as _e:
            # If corrupt or missing, start fresh
            logger.debug('Ignored Exception in reaper.py: %s', _e)

def reaper_loop(kernel_pid: int, manifest_path: Path):
    """
    The Reaper process main loop.
    Polls kernel liveness. On death, executes cleanup in strict order.
    """
    # Configure logging for the detached process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    manifest = ReaperManifest(manifest_path)

    logger.info("[REAPER] Watching Kernel PID %d", kernel_pid)

    # Poll until kernel dies
    while True:
        try:
            os.kill(kernel_pid, 0)  # Signal 0: existence check only
        except ProcessLookupError:
            logger.warning("[REAPER] Kernel PID %d is GONE. Initiating cleanup.", kernel_pid)
            _execute_cleanup(manifest)
            return
        except PermissionError as _e:
            logger.debug('Ignored PermissionError in reaper.py: %s', _e)
        except Exception as e:
            logger.debug(f"[REAPER] Existence check failed (non-fatal): {e}")
            
        time.sleep(POLL_INTERVAL)

def _execute_cleanup(manifest: ReaperManifest):
    """Execute cleanup in order: children first, then shared memory."""

    # 1. Terminate orphaned child processes
    child_pids: List[int] = manifest._data.get("child_pids", [])
    for pid in list(child_pids):
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
        except ProcessLookupError as _e:
            logger.debug('Ignored ProcessLookupError in reaper.py: %s', _e)
        except Exception as e:
            logger.error("[REAPER] Failed to kill PID %d: %s", pid, e)
        manifest.deregister_pid(pid)

    # 2. Unlink named shared memory segments
    shm_names: List[str] = manifest._data.get("shm_names", [])
    for name in list(shm_names):
        try:
            # We must attach before we can unlink in some versions,
            # or use the internal shm_unlink if available.
            # Python's SharedMemory makes this easy with unlink()
            try:
                segment = shm_lib.SharedMemory(name=name)
                segment.close()
                segment.unlink()
                logger.info("[REAPER] Unlinked SHM segment: %s", name)
            except FileNotFoundError as _e:
                logger.debug('Ignored FileNotFoundError in reaper.py: %s', _e)
        except Exception as e:
            logger.error("[REAPER] Failed to unlink SHM %s: %s", name, e)
        manifest.deregister_shm(name)

    # 3. Clean up the manifest file itself
    try:
        manifest.path.unlink(missing_ok=True)
    except Exception as _e:
        logger.debug('Ignored Exception in reaper.py: %s', _e)

    logger.info("[REAPER] Cleanup complete.")

def register_reaper_pid(pid: int):
    """Convenience helper for kernel components."""
    ReaperManifest().register_pid(pid)

def register_reaper_shm(name: str):
    """Convenience helper for kernel components."""
    ReaperManifest().register_shm(name)
