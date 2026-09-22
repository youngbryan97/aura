import fcntl
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Utils.Singleton")
_PROCESS_METADATA_ERRORS = (AttributeError, ImportError, OSError, RuntimeError, ValueError)

_LOCK_FD: int | None = None
_LOCK_NAME: str | None = None


def singleton[T](cls: type[T]) -> Callable[..., T]:
    """
    Decorator to make a class a singleton.
    Usage:
    @singleton
    class MyClass: ...
    """
    instances: dict[type[T], T] = {}

    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def instance_lock_path(lock_name: str = "singleton") -> Path:
    return state_root() / "locks" / f"{lock_name}.lock"


def boot_blocked_path() -> Path:
    """Where a refused start publishes WHY, for the launcher/GUI to surface."""
    return state_root() / "run" / "boot_blocked.json"


def clear_boot_blocked() -> None:
    """Drop any stale blocked-boot notice (called once a lock is acquired)."""
    try:
        with local_internal_governed_scope(
            "singleton.clear_boot_blocked",
            domain="file_write",
        ):
            get_file_write_gateway().delete_file(
                boot_blocked_path(),
                source="singleton.clear_boot_blocked",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
        logger.debug("Could not clear boot-blocked notice: %s", exc)


def read_boot_blocked() -> dict[str, Any]:
    """The last refusal notice, or {} when the last start was not blocked."""
    try:
        payload = json.loads(boot_blocked_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    holder_pid = payload.get("holder_pid")
    # A notice about a process that has since exited is not a live blocker.
    try:
        os.kill(int(holder_pid), 0)
    except (OSError, TypeError, ValueError):
        return {}
    return payload


def _publish_boot_blocked(lock_name: str, holder_pid: int) -> None:
    """Record WHY this start was refused, so the GUI never just spins.

    The refusal used to exist only as a print() on a stdout nobody reads: the
    runtime exited immediately with EX_TEMPFAIL while the desktop boot monitor
    sat on "Aura is waking up… waiting for boot health" forever. A start that
    was positively refused must say so.
    """
    holder = read_instance_lock_metadata(lock_name)
    cmdline = holder.get("cmdline") or []
    cwd = str(holder.get("cwd") or "")
    is_foreign = ".claude/worktrees" in cwd or "--headless" in cmdline
    notice = {
        "schema": "aura.boot_blocked.v1",
        "blocked_at_unix": time.time(),
        "lock_name": lock_name,
        "holder_pid": int(holder_pid),
        "holder_cmdline": cmdline,
        "holder_cwd": cwd,
        "holder_started_unix": holder.get("create_time"),
        "holder_is_background_instance": bool(is_foreign),
        "reason": (
            f"Another Aura runtime (PID {holder_pid}) already holds the "
            f"'{lock_name}' instance lock. Only one runtime may hold it — a second "
            "would load a second copy of the resident model and exhaust host memory."
        ),
        "remedy": (
            f"Stop the other instance first: kill {holder_pid}"
            + (
                "  (it is a background/headless instance, not your desktop app)"
                if is_foreign
                else "  (quit the other Aura window)"
            )
            + ", then relaunch. `python aura_cleanup.py` also reclaims a stuck runtime."
        ),
    }
    try:
        path = boot_blocked_path()
        with local_internal_governed_scope(
            "singleton.publish_boot_blocked",
            domain="file_write",
        ):
            get_file_write_gateway().write_text(
                path,
                json.dumps(notice, indent=2, sort_keys=True),
                encoding="utf-8",
                source="singleton.publish_boot_blocked",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
        logger.debug("Could not publish boot-blocked notice: %s", exc)


def instance_lock_metadata_path(lock_name: str = "singleton") -> Path:
    return state_root() / "locks" / f"{lock_name}.lock.meta.json"


def parse_instance_lock_pid(raw: str) -> int | None:
    """Parse both legacy PID-only locks and future structured lock payloads."""
    text = str(raw or "").strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    try:
        return int(first_line)
    except ValueError as _exc:
        logger.debug("Suppressed %s in core.utils.singleton: %s", type(_exc).__name__, _exc)
    try:
        payload = json.loads(text)
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    try:
        return int(payload.get("pid"))
    # not a failure: a reading that cannot be taken, or that is not a
    # number when it is, leaves this at the value below.
    except (AttributeError, TypeError, ValueError):
        return None


def read_instance_lock_pid(lock_name: str = "singleton") -> int | None:
    try:
        return parse_instance_lock_pid(instance_lock_path(lock_name).read_text(encoding="utf-8"))
    except OSError:
        return None


def read_instance_lock_metadata(lock_name: str = "singleton") -> dict[str, Any]:
    try:
        payload = json.loads(instance_lock_metadata_path(lock_name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_process_metadata(lock_name: str, pid: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema": "aura.instance_lock.v1",
        "lock_name": lock_name,
        "pid": int(pid),
        "written_at": time.time(),
        "cwd": os.getcwd(),
        "cmdline": list(sys.argv),
    }
    try:
        process = get_resource_observer().process(pid)
        if process is None:
            raise RuntimeError("process_identity_unavailable")
        metadata.update(
            {
                "create_time": process.create_time,
                "ppid": process.ppid,
                "username": process.username,
                "observation_source": process.provenance.source.value,
                "observation_scenario_id": process.provenance.scenario_id,
            }
        )
    except _PROCESS_METADATA_ERRORS as exc:  # pragma: no cover - optional diagnostic metadata
        metadata["identity_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _write_instance_lock_metadata(lock_name: str, pid: int) -> None:
    path = instance_lock_metadata_path(lock_name)
    try:
        atomic_write_text(
            path,
            json.dumps(_current_process_metadata(lock_name, pid), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        record_degradation("singleton", exc)
        logger.warning("Failed to write instance lock metadata for '%s': %s", lock_name, exc)


def acquire_instance_lock(lock_name: str = "singleton", skip_lock: bool = False) -> None:
    """
    Ensure only one instance of a specific Aura component/process is running.
    Uses a file lock in ~/.aura/locks/.
    
    Args:
        lock_name: Name of the lock (e.g., "orchestrator", "genesis").
        skip_lock: If True, bypass the lock check.
    """
    if skip_lock:
        return

    global _LOCK_FD, _LOCK_NAME
    if _LOCK_FD is not None and _LOCK_NAME == lock_name:
        return

    # Standardize lock path
    lock_file = instance_lock_path(lock_name)
    lock_dir = lock_file.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open with O_RDWR to allow writing the PID.
        # Add O_CLOEXEC to prevent FD inheritance by child processes.
        # This ensures the lock is freed on main process crash even if children (Reaper/GUI) survive.
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
            
        _LOCK_FD = os.open(str(lock_file), flags, 0o600)
        
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Lock is held by another process. Read the PID.
            try:
                os.lseek(_LOCK_FD, 0, os.SEEK_SET)
                pid_bytes = os.read(_LOCK_FD, 4096)
                pid = parse_instance_lock_pid(pid_bytes.decode()) or 0
                if pid <= 0:
                    raise ValueError("invalid lock pid")
                
                # Check if the process is actually running
                try:
                    os.kill(pid, 0)
                    # Publish the refusal where the launcher/GUI can see it, so a
                    # positively-refused start is never rendered as "still waking
                    # up". This is the difference between a 20-minute debug and a
                    # one-line remedy.
                    _publish_boot_blocked(lock_name, pid)
                    holder = read_instance_lock_metadata(lock_name)
                    message = (
                        f"⚠️  Aura ({lock_name}) is already running (PID: {pid})"
                        + (f" — {' '.join(holder.get('cmdline') or [])}" if holder.get("cmdline") else "")
                        + f". Stop it first: kill {pid}"
                    )
                    logger.error(message)
                    print(message)
                    # Exit 0 only under the launchd supervisor (it sets
                    # AURA_SUPERVISED=1): a non-zero exit there would
                    # restart-loop against the live instance every 15s.
                    # Every other caller — proof batteries especially —
                    # must see "couldn't run" as failure, never success.
                    if os.environ.get("AURA_SUPERVISED") == "1":
                        raise SystemExit(0)
                    raise SystemExit(75)  # EX_TEMPFAIL: instance busy
                except OSError:
                    # Process is dead. Reclaim the stale lock.
                    logger.warning(
                        "🔓 Stale lock found for dead PID %d. Reclaiming lock for %s.",
                        pid, lock_name,
                    )
                    # Close and reopen to get a fresh FD
                    try:
                        os.close(_LOCK_FD)
                    except OSError as _exc:
                        logger.debug("Suppressed %s in core.utils.singleton: %s", type(_exc).__name__, _exc)
                    flags = os.O_CREAT | os.O_RDWR
                    if hasattr(os, "O_CLOEXEC"):
                        flags |= os.O_CLOEXEC
                    _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                    try:
                        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        # If we still can't get it, a child process holds the FD.
                        # Last resort: unlink and recreate.
                        try:
                            lock_file.unlink()
                            os.close(_LOCK_FD)
                            _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                            fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            logger.info("🔓 Stale lock reclaimed via unlink+recreate.")
                        except OSError as reclaim_exc:
                            message = f"⚠️  Failed to reclaim stale lock for {lock_name}: {reclaim_exc}"
                            logger.error(message)
                            print(message)
                            raise SystemExit(1) from None
            except (ValueError, UnicodeDecodeError):
                # PID file is corrupt. Reclaim unconditionally.
                logger.warning("🔓 Corrupt lock file for %s. Reclaiming.", lock_name)
                try:
                    os.close(_LOCK_FD)
                    lock_file.unlink(missing_ok=True)
                    flags = os.O_CREAT | os.O_RDWR
                    if hasattr(os, "O_CLOEXEC"):
                        flags |= os.O_CLOEXEC
                    _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                    fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    message = f"⚠️  Aura ({lock_name}) is already running in another window."
                    logger.error(message)
                    print(message)
                    raise SystemExit(0) from None
        
        # Lock acquired. Write current PID.
        os.ftruncate(_LOCK_FD, 0)
        os.write(_LOCK_FD, f"{os.getpid()}\n".encode())
        os.fsync(_LOCK_FD)
        _LOCK_NAME = lock_name
        _write_instance_lock_metadata(lock_name, os.getpid())
        # This start was NOT blocked — retire any previous refusal notice so the
        # launcher never shows a stale blocker.
        clear_boot_blocked()

        logger.info("🔒 Instance lock acquired: %s (PID: %d)", lock_name, os.getpid())
        
    except OSError as e:
        record_degradation('singleton', e)
        logger.warning("Failed to acquire single-instance lock for '%s': %s", lock_name, e)


def release_instance_lock() -> None:
    """Explicitly release the lock (usually handled by process exit)."""
    global _LOCK_FD, _LOCK_NAME
    if _LOCK_FD is not None:
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            os.close(_LOCK_FD)
            _LOCK_FD = None
            _LOCK_NAME = None
        except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
            record_degradation('singleton', _e)
            logger.debug('Ignored Exception in singleton.py: %s', _e)
