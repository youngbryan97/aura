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
from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import state_root

try:
    import psutil
# not a failure: the module is optional here, and its absence is the answer.
except ImportError:  # pragma: no cover - exercised on minimal installs
    psutil = None

REAPER_MANIFEST_ENV = "AURA_REAPER_MANIFEST"
LEGACY_REAPER_MANIFEST = Path(tempfile.gettempdir()) / "aura_reaper_manifest.json"
DEFAULT_REAPER_MANIFEST_DIR = state_root() / "run" / "reaper"
POLL_INTERVAL = 1.0  # seconds

logger = logging.getLogger("Aura.Reaper")
_PSUTIL_ERRORS = (psutil.Error,) if psutil is not None else ()
_REAPER_PROCESS_ERRORS = (
    *_PSUTIL_ERRORS,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


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
    if raw_path and Path(raw_path).expanduser() != LEGACY_REAPER_MANIFEST:
        return Path(raw_path).expanduser()
    runtime_id = os.environ.get("AURA_RUNTIME_ID", "").strip() or f"{int(time.time())}-{os.getpid()}"
    os.environ.setdefault("AURA_RUNTIME_ID", runtime_id)
    path = DEFAULT_REAPER_MANIFEST_DIR / f"manifest-{runtime_id}.json"
    os.environ.setdefault(REAPER_MANIFEST_ENV, str(path))
    return path


REAPER_MANIFEST = resolve_reaper_manifest_path()


class ReaperManifest:
    """Tracks all resources the Reaper must clean up."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else resolve_reaper_manifest_path()
        self._data: dict[str, Any] = {
            "schema": "aura.reaper_manifest.v2",
            "shm_names": [],
            "child_pids": [],
            "child_pid_records": [],
            "pipe_fds": [],
        }
        self._load()

    def register_shm(self, name: str):
        if name not in self._data["shm_names"]:
            self._data["shm_names"].append(name)
        self._save()

    def register_pid(self, pid: int):
        record = self._pid_record(pid)
        records = [
            existing
            for existing in self._data.get("child_pid_records", [])
            if int(existing.get("pid", -1)) != int(pid)
        ]
        records.append(record)
        self._data["child_pid_records"] = records
        self._save()

    def deregister_shm(self, name: str):
        self._data["shm_names"] = [n for n in self._data["shm_names"] if n != name]
        self._save()

    def deregister_pid(self, pid: int):
        self._data["child_pids"] = [p for p in self._data["child_pids"] if p != pid]
        self._data["child_pid_records"] = [
            p
            for p in self._data.get("child_pid_records", [])
            if int(p.get("pid", -1)) != int(pid)
        ]
        self._save()

    @staticmethod
    def _pid_record(pid: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "pid": int(pid),
            "registered_at": time.time(),
            "registered_by": os.getpid(),
        }
        try:
            process = get_resource_observer().process(int(pid))
            if process is None:
                raise RuntimeError("process_identity_unavailable")
            record.update(
                {
                    "create_time": process.create_time,
                    "ppid": process.ppid,
                    "cmdline": list(process.cmdline),
                    "cwd": process.cwd,
                    "observation_source": process.provenance.source.value,
                    "observation_scenario_id": process.provenance.scenario_id,
                }
            )
        except _REAPER_PROCESS_ERRORS as exc:  # pragma: no cover - defensive identity metadata path
            record["identity_error"] = f"{type(exc).__name__}: {exc}"
        return record

    def _save(self):
        try:
            # Atomic write to prevent corruption (Windows compatible)
            import tempfile
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(dir=str(self.path.parent))
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
                raw = json.loads(self.path.read_text())
                self._data = self._normalize(raw)
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

    @staticmethod
    def _normalize(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "schema": "aura.reaper_manifest.v2",
                "shm_names": [],
                "child_pids": [],
                "child_pid_records": [],
                "pipe_fds": [],
            }
        legacy_pids: list[int] = []
        records: list[dict[str, Any]] = []
        for item in raw.get("child_pid_records", []):
            if isinstance(item, dict) and "pid" in item:
                records.append(dict(item))
        for item in raw.get("child_pids", []):
            if isinstance(item, dict) and "pid" in item:
                records.append(dict(item))
            else:
                try:
                    legacy_pids.append(int(item))
                except (TypeError, ValueError):
                    continue
        return {
            "schema": "aura.reaper_manifest.v2",
            "shm_names": list(dict.fromkeys(str(v) for v in raw.get("shm_names", []) if v)),
            "child_pids": list(dict.fromkeys(legacy_pids)),
            "child_pid_records": records,
            "pipe_fds": list(raw.get("pipe_fds", []) or []),
        }

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
            _execute_cleanup(manifest, kernel_pid=kernel_pid)
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

def _pid_cleanup_authorized(
    entry: Any,
    *,
    kernel_pid: int | None = None,
) -> tuple[bool, int | None, str]:
    """Return whether a manifest PID entry is safe to signal.

    Bare legacy PIDs are not enough evidence. macOS and Linux reuse PIDs, and a
    static manifest can outlive the runtime that wrote it. Signaling a reused
    PID can terminate a healthy live Aura process, so cleanup requires stored
    process identity metadata from register_pid().
    """
    if isinstance(entry, dict):
        try:
            pid = int(entry.get("pid"))
        except (TypeError, ValueError):
            return False, None, "invalid_pid_record"
        expected_create_time = entry.get("create_time")
    else:
        try:
            pid = int(entry)
        except (TypeError, ValueError):
            return False, None, "invalid_legacy_pid"
        expected_create_time = None

    if not expected_create_time:
        if os.environ.get("AURA_REAPER_ALLOW_LEGACY_PID_CLEANUP") == "1":
            return True, pid, "legacy_cleanup_explicitly_allowed"
        return False, pid, "legacy_pid_without_identity"

    try:
        process = get_resource_observer().process(pid)
        if process is None:
            return False, pid, "missing_pid"
        actual_create_time = process.create_time
        if abs(actual_create_time - float(expected_create_time)) > 0.05:
            return False, pid, "pid_reused_or_stale"
        expected_cwd = str(entry.get("cwd") or "")
        if expected_cwd and process.cwd != expected_cwd:
            return False, pid, "process_identity_cwd_mismatch"
        if kernel_pid is not None and pid == int(kernel_pid):
            return False, pid, "refusing_to_signal_kernel_pid"
        return True, pid, "identity_verified"
    except ProcessLookupError:
        return False, pid, "missing_pid"
    except _REAPER_PROCESS_ERRORS as exc:
        if type(exc).__name__ in {"NoSuchProcess", "ZombieProcess"}:
            return False, pid, "missing_pid"
        return False, pid, f"identity_check_failed:{type(exc).__name__}"


def _execute_cleanup(manifest: ReaperManifest, kernel_pid: int | None = None) -> dict[str, Any]:
    """Execute cleanup in order: children first, then shared memory."""
    summary: dict[str, Any] = {
        "terminated_pids": [],
        "missing_pids": [],
        "skipped_pids": [],
        "skipped_pid_details": [],
        "failed_pids": [],
        "unlinked_shm": [],
        "missing_shm": [],
        "failed_shm": [],
        "manifest_removed": False,
    }

    # 1. Terminate orphaned child processes
    child_entries: list[Any] = [
        *list(manifest._data.get("child_pid_records", []) or []),
        *list(manifest._data.get("child_pids", []) or []),
    ]
    for entry in list(child_entries):
        authorized, pid, identity_reason = _pid_cleanup_authorized(entry, kernel_pid=kernel_pid)
        if pid is None:
            continue
        if not authorized:
            if identity_reason == "missing_pid":
                summary["missing_pids"].append(pid)
            else:
                summary["skipped_pids"].append(pid)
                summary["skipped_pid_details"].append(
                    {"pid": pid, "reason": identity_reason}
                )
                logger.warning(
                    "[REAPER] Skipping PID %d cleanup: %s",
                    pid,
                    identity_reason,
                )
            manifest.deregister_pid(pid)
            continue

        cleaned_pid = False
        try:
            logger.info("[REAPER] Cleaning up PID %d", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except PermissionError as e:
                summary["skipped_pids"].append(pid)
                _record_reaper_degradation(
                    e,
                    stage="pid_cleanup",
                    action="deregistered non-owned or reused PID from reaper manifest without signaling it",
                    severity="warning",
                    extra={"pid": pid},
                )
                logger.warning(
                    "[REAPER] PID %d not signalable (PermissionError on SIGTERM); skipping.",
                    pid,
                )
                manifest.deregister_pid(pid)
                continue
            # Short grace period
            for _ in range(5):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError as e:
                    summary["skipped_pids"].append(pid)
                    _record_reaper_degradation(
                        e,
                        stage="pid_cleanup",
                        action="stopped PID cleanup after liveness probe identified a non-owned or reused PID",
                        severity="warning",
                        extra={"pid": pid},
                    )
                    logger.warning(
                        "[REAPER] PID %d exists but is not signalable; treating as non-owned/reused.",
                        pid,
                    )
                    break
            else:
                # Force kill if still alive
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.warning("[REAPER] Force-killed orphan PID %d", pid)
                except PermissionError as e:
                    summary["skipped_pids"].append(pid)
                    _record_reaper_degradation(
                        e,
                        stage="pid_cleanup",
                        action="skipped SIGKILL on non-owned or reused PID",
                        severity="warning",
                        extra={"pid": pid},
                    )
                    logger.warning(
                        "[REAPER] PID %d not signalable (PermissionError on SIGKILL); skipping.",
                        pid,
                    )
            cleaned_pid = True
            if pid not in summary["skipped_pids"]:
                summary["terminated_pids"].append(pid)
        except ProcessLookupError as _e:
            cleaned_pid = True
            summary["missing_pids"].append(pid)
            logger.debug('Ignored ProcessLookupError in reaper.py: %s', _e)
        except PermissionError as e:
            cleaned_pid = True
            summary["skipped_pids"].append(pid)
            _record_reaper_degradation(
                e,
                stage="pid_cleanup",
                action="deregistered non-owned or reused PID from reaper manifest without signaling it",
                severity="warning",
                extra={"pid": pid},
            )
            logger.warning(
                "[REAPER] Cannot signal PID %d; assuming non-owned or PID-reused process and skipping.",
                pid,
            )
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
    unresolved = bool(
        manifest._data.get("child_pids")
        or manifest._data.get("child_pid_records")
        or manifest._data.get("shm_names")
    )
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
