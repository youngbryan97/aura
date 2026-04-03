#!/usr/bin/env python3
"""
Aura Cognitive Cleanup Utility (macOS)
--------------------------------------
Identifies and terminates orphaned Aura processes, clears VRAM lock files,
and prepares the system for a clean restart.
"""

import os
import signal
import subprocess
import time
import logging
from typing import Iterable, Set

try:
    import psutil
except ImportError:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Aura.Cleanup")

PROCESS_PATTERNS = (
    "aura_main.py",
    "interface/gui_actor.py",
    "core.brain.llm.mlx_worker",
    "forkserver",
)
PORTS = (8000, 10003, 11435, 11436, 11437, 11438)
LOCK_PATHS = (
    "/tmp/aura_vram.lock",
    os.path.expanduser("~/.aura/locks/orchestrator.lock"),
    os.path.expanduser("~/.aura/singleton.lock"),
)

CLEAR_ONLY_PATHS = (
    os.path.expanduser("~/.aura/locks/desktop-app-launch.marker"),
    os.path.expanduser("~/.aura/locks/desktop-app-launch.lock"),
)
SUBPROCESS_TIMEOUT_S = 2.0
GRACEFUL_SHUTDOWN_S = 2.0


def _recent_process_grace_s() -> float:
    try:
        return max(0.0, float(os.environ.get("AURA_CLEANUP_RECENT_GRACE_S", "0") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _process_age_s(pid: int) -> float | None:
    if psutil is None:
        return None
    try:
        return max(0.0, time.time() - float(psutil.Process(pid).create_time()))
    except Exception:
        return None


def _preserve_recent_pid(pid: int) -> bool:
    grace_s = _recent_process_grace_s()
    if grace_s <= 0.0:
        return False
    age_s = _process_age_s(pid)
    return age_s is not None and age_s < grace_s


def _run_capture(args: list[str], *, timeout: float = SUBPROCESS_TIMEOUT_S) -> list[str]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out while running cleanup probe: %s", " ".join(args))
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _pids_from_values(values: Iterable[str]) -> Set[int]:
    pids: Set[int] = set()
    for value in values:
        try:
            pids.add(int(value))
        except (TypeError, ValueError):
            continue
    return pids


def _collect_pattern_pids() -> Set[int]:
    pids: Set[int] = set()
    for pattern in PROCESS_PATTERNS:
        pids.update(_pids_from_values(_run_capture(["pgrep", "-f", pattern])))
    return pids


def _collect_port_pids() -> Set[int]:
    pids: Set[int] = set()
    for port in PORTS:
        pids.update(
            _pids_from_values(
                _run_capture(["lsof", "-nP", f"-iTCP:{port}", "-t"])
            )
        )
    return pids


def _collect_lockfile_pids() -> Set[int]:
    pids: Set[int] = set()
    for lock_path in LOCK_PATHS:
        if not os.path.exists(lock_path):
            continue
        pids.update(_pids_from_values(_run_capture(["lsof", "-t", lock_path])))
    return pids


def get_aura_pids() -> Set[int]:
    """Find Aura-related pids without letting cleanup probes hang indefinitely."""
    pids = set()
    pids.update(_collect_pattern_pids())
    pids.update(_collect_port_pids())
    pids.update(_collect_lockfile_pids())
    return pids


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for_exit(pid: int, *, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)
    return not _pid_exists(pid)

def cleanup():
    logger.info("🧹 Starting Aura Cognitive Cleanup...")

    # 1. Kill processes
    pids = get_aura_pids()
    my_pid = os.getpid()
    pids.discard(my_pid)

    if pids:
        preserved = {pid for pid in pids if _preserve_recent_pid(pid)}
        if preserved:
            logger.info("Preserving fresh Aura process(es): %s", preserved)
            pids -= preserved

    if pids:
        logger.info(f"Found {len(pids)} Aura-related processes: {pids}")
        for pid in pids:
            try:
                logger.info(f"Terminating PID {pid}...")
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                logger.error(f"Permission denied to kill {pid}")

        remaining = {pid for pid in pids if not _wait_for_exit(pid, timeout=GRACEFUL_SHUTDOWN_S)}
        for pid in remaining:
            try:
                logger.info(f"Force-killing PID {pid}...")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
    else:
        logger.info("No active Aura processes found.")

    # 2. Clear lock files
    for lock_path in (*LOCK_PATHS, *CLEAR_ONLY_PATHS):
        if not os.path.exists(lock_path):
            continue
        try:
            os.remove(lock_path)
            logger.info("✅ Cleared lock file: %s", lock_path)
        except Exception as e:
            logger.error(f"Failed to clear lock file {lock_path}: {e}")

    logger.info("✨ Cleanup complete. You can now restart Aura safely.")

if __name__ == "__main__":
    cleanup()
