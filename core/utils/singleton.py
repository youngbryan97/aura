from core.runtime.errors import record_degradation
import fcntl
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger("Aura.Utils.Singleton")

_LOCK_FD: int | None = None
T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
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

    global _LOCK_FD
    # Standardize lock path
    lock_dir = Path.home() / ".aura" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{lock_name}.lock"
    
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
                pid_bytes = os.read(_LOCK_FD, 16)
                pid = int(pid_bytes.decode().strip())
                
                # Check if the process is actually running
                try:
                    os.kill(pid, 0)
                    message = f"⚠️  Aura ({lock_name}) is already running (PID: {pid})."
                    logger.error(message)
                    print(message)
                    raise SystemExit(0)
                except OSError:
                    # Process is dead. On macOS, flock is associated with the FD.
                    # If flock failed, SOMEONE has it. But if kill -0 failed, the PID is dead.
                    # This happens if a child inherited the FD. O_CLOEXEC prevents this.
                    # For now, we exit to avoid corruption, but tell the user why.
                    message = f"⚠️  Stale lock found for dead PID {pid} (likely held by child process)."
                    logger.error(message)
                    print(message)
                    raise SystemExit(1) from None
            except Exception:
                message = f"⚠️  Aura ({lock_name}) is already running in another window."
                logger.error(message)
                print(message)
                raise SystemExit(0) from None
        
        # Lock acquired. Write current PID.
        os.ftruncate(_LOCK_FD, 0)
        os.write(_LOCK_FD, str(os.getpid()).encode())
        os.fsync(_LOCK_FD)
        
        logger.info("🔒 Instance lock acquired: %s (PID: %d)", lock_name, os.getpid())
        
    except Exception as e:
        record_degradation('singleton', e)
        logger.warning("Failed to acquire single-instance lock for '%s': %s", lock_name, e)


def release_instance_lock() -> None:
    """Explicitly release the lock (usually handled by process exit)."""
    global _LOCK_FD
    if _LOCK_FD is not None:
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            os.close(_LOCK_FD)
            _LOCK_FD = None
        except Exception as _e:
            record_degradation('singleton', _e)
            logger.debug('Ignored Exception in singleton.py: %s', _e)
