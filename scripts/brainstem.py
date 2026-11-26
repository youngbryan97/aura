#!/usr/bin/env python3
"""
Lazarus Protocol - Minimalist Life Support Watchdog
--------------------------------------------------
This script monitors the main Aura process and reboots it if it crashes
or stops pulsing its heartbeat.
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

# --- Lazarus Version Lock ---------------------------------------
# The brainstem must use the same interpreter as the cognitive mind
# to ensure consistent environment variables and library access.
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 12
if sys.version_info.major != REQUIRED_MAJOR or sys.version_info.minor != REQUIRED_MINOR:
    # Attempt to locate the venv python if we're not running it
    VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    if VENV_PYTHON.exists():
        print(f"[BRAINSTEM] Version mismatch (running on {sys.version.split()[0]}). Re-executing with 3.12 venv...")
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)
    else:
        print(f"[BRAINSTEM] Warning: Running on {sys.version.split()[0]}. Expected 3.12. Native AI cores may fail.")

# ---------------------------------------------------------------

# --- Configuration for the Brainstem ---
PROJECT_ROOT = ROOT
MIND_SCRIPT = str(PROJECT_ROOT / "aura_main.py")
HEARTBEAT_FILE = Path(tempfile.gettempdir()) / "aura_heartbeat.pulse"
HEARTBEAT_TIMEOUT = 60 # seconds (Extended slightly for heavy M1 Pro boot)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
COGNITIVE_LOG_FILE = LOG_DIR / "aura_cognitive.log"
BRAINSTEM_LOG_FILE = LOG_DIR / "aura_brainstem.log"

def log_brainstem(message: str):
    """Minimal, safe logger for the brainstem."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} [BRAINSTEM]: {message}\n"
    try:
        with open(BRAINSTEM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
    except OSError as exc:
        print(f"{timestamp} [BRAINSTEM_LOG_ERROR]: {exc}")
    print(log_line.strip())

_BRAINSTEM_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutExpired, ValueError)


def launch_mind_process() -> Any | None:
    """Launches the cognitive core in a new, separate process."""
    log_brainstem("Igniting cognitive core process...")
    
    # The gateway still spawns an independent child process, but gives launch
    # ownership, cwd/env validation, and strict/live governance fail-closed behavior.
    try:
        with open(COGNITIVE_LOG_FILE, "a", encoding="utf-8") as log_file:
            process = get_subprocess_gateway().spawn(
                [sys.executable, MIND_SCRIPT],
                stdout=log_file,
                stderr=log_file,
                env=os.environ.copy(),
                cwd=str(PROJECT_ROOT),
                offline_tooling=True,
                source="maintenance_tooling:brainstem_launch_mind",
            )
        log_brainstem(f"Mind process launched with PID: {process.pid}")
        return process
    except _BRAINSTEM_RECOVERABLE_ERRORS as exc:
        log_brainstem(f"Failed to launch Mind process: {type(exc).__name__}: {exc}")
        return None

def main():
    """The main life-support loop."""
    log_brainstem(" Lazarus Protocol Active. Awaiting cognitive ignition.")
    
    # Ensure heartbeat file is clean on start
    if HEARTBEAT_FILE.exists():
        try:
            HEARTBEAT_FILE.unlink()
        except OSError as exc:
            log_brainstem(f"Failed to remove stale heartbeat file: {exc}")
        
    mind_process = launch_mind_process()
    if not mind_process:
        log_brainstem("FATAL: Could not start Aura. Exiting.")
        sys.exit(1)
    
    boot_grace_period = time.time() + 30 # Give 30s before checking heartbeat
    
    running = True
    while running:
        try:
            time.sleep(5) # Check every 5 seconds
            
            # 1. Check if process is still running
            return_code = mind_process.poll()
            if return_code is not None:
                log_brainstem(f" CRITICAL: Mind process terminated unexpectedly (Exit Code: {return_code}).")
                log_brainstem("Rebooting cognitive core...")
                time.sleep(2)
                mind_process = launch_mind_process()
                boot_grace_period = time.time() + 30
                continue
            
            # 2. Check for hung process via heartbeat
            if not HEARTBEAT_FILE.exists():
                if time.time() > boot_grace_period:
                    log_brainstem(" WARNING: No heartbeat file detected after grace period.")
                continue
                
            last_pulse_age = time.time() - HEARTBEAT_FILE.stat().st_mtime
            if last_pulse_age > HEARTBEAT_TIMEOUT:
                log_brainstem(f"CRITICAL: Cognitive heartbeat timed out (last pulse {last_pulse_age:.1f}s ago).")
                log_brainstem("Assuming hang. Terminating and rebooting...")
                
                mind_process.terminate()
                try:
                    mind_process.wait(timeout=5)
                except TimeoutExpired:
                    log_brainstem("Stubborn process: Escalating to SIGKILL.")
                    mind_process.kill()
                
                try:
                    HEARTBEAT_FILE.unlink()
                except OSError as exc:
                    log_brainstem(f"Failed to remove heartbeat before reboot: {exc}")
                    
                time.sleep(2)
                mind_process = launch_mind_process()
                boot_grace_period = time.time() + 30
                
        except KeyboardInterrupt:
            running = False
            log_brainstem("Lazarus Protocol disabled by user. Shutting down Mind...")
            mind_process.terminate()
            try:
                mind_process.wait(timeout=5)
            except _BRAINSTEM_RECOVERABLE_ERRORS:
                mind_process.kill()
            sys.exit(0)
        except _BRAINSTEM_RECOVERABLE_ERRORS as exc:
            log_brainstem(f"‼️ FATAL BRAINSTEM ERROR: {type(exc).__name__}: {exc}")
            time.sleep(5)

if __name__ == "__main__":
    main()
