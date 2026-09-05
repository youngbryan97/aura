"""Sovereign Resilience: Supervisor & Watchdog
-------------------------------------------
The SovereignSupervisor is responsible for keeping the Aura core alive.
It uses strictly local monitoring (psutil) and implements exponential backoff
to prevent rapid crash loops from consuming resources.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

try:
    import psutil
except ImportError:
    psutil = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("Sovereign.Supervisor")

_SUPERVISOR_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_CRASH_WINDOW_SECONDS = 60.0
_MAX_RESTART_BACKOFF_SECONDS = 300.0
_GRACE_FLAG_TTL_SECONDS = 300.0


def _record_supervisor_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict | None = None,
) -> None:
    record_degradation(
        "supervisor",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


class SovereignSupervisor:
    def __init__(self, target_script: str, args: list[str] | None = None):
        self.target_script = Path(target_script)
        self.args = args or []
        self.process: asyncio.subprocess.Process | None = None
        self.should_run = True
        self.crash_count = 0
        self.last_crash_time = 0
        self._shutdown_event = threading.Event()

    async def start(self):
        """Main loop: launches and watches the target process."""
        if not self.target_script.exists():
            logger.critical("Target script missing: %s", self.target_script)
            return

        logger.info("🛡️  Sovereign Supervisor active. Guarding: %s", self.target_script.name)
        
        while self.should_run:
            try:
                await self._launch_process()
                await self._monitor_process()
            except KeyboardInterrupt:
                await self.stop()
            except _SUPERVISOR_ERRORS as e:
                self.crash_count += 1
                delay_s = min(_MAX_RESTART_BACKOFF_SECONDS, 5.0 * (2 ** min(self.crash_count - 1, 5)))
                _record_supervisor_degradation(
                    e,
                    action="backed off supervisor launch loop after monitor failure",
                    severity="warning",
                    extra={
                        "target": str(self.target_script),
                        "crash_count": self.crash_count,
                        "backoff_s": delay_s,
                    },
                )
                logger.error("Supervisor loop error: %s", e)
                await asyncio.sleep(delay_s)

    async def stop(self):
        """Gracefully stops the supervisor and child process."""
        self.should_run = False
        self._shutdown_event.set()
        if self.process:
            logger.info("Stopping monitored process...")
            self._kill_process_tree(self.process.pid)

    async def _launch_process(self):
        """Launches the target script as a subprocess."""
        cmd = [sys.executable, str(self.target_script)] + self.args
        logger.info("🚀 Launching %s...", self.target_script.name)
        
        self.process = await get_subprocess_gateway().spawn_async(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.cwd()),
            source="environment_action:resilience_supervisor.launch",
            accelerator_capability="auto",
        )

        if self.process.stdout is not None:
            get_task_tracker().create_task(
                self._pipe_logger_async(self.process.stdout, logging.INFO, "stdout"),
                name="sovereign_supervisor.pipe.stdout",
            )
        if self.process.stderr is not None:
            get_task_tracker().create_task(
                self._pipe_logger_async(self.process.stderr, logging.ERROR, "stderr"),
                name="sovereign_supervisor.pipe.stderr",
            )

    async def _pipe_logger_async(self, pipe: asyncio.StreamReader, level: int, label: str):
        """Reads from an asyncio stream and logs each line."""
        try:
            while self.should_run:
                line_bytes = await pipe.readline()
                if not line_bytes:
                    break
                try:
                    line = line_bytes.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line = line_bytes.decode('latin-1', errors='replace').strip()
                if line:
                    logger.log(level, "[Sub] %s", line)
        except _SUPERVISOR_ERRORS as e:
            _record_supervisor_degradation(
                e,
                action="kept supervisor alive after subprocess pipe logger failed",
                severity="warning",
                extra={"pipe": label},
            )
            logger.error("Error reading pipe %s: %s", label, e)

    async def _monitor_process(self):
        """Blocks while monitoring the process. Returns when process exits."""
        # Implement a 5-second poll() timeout check
        while self.process and self.process.returncode is None:
            if not self.should_run:
                try:
                    self.process.terminate()
                except ProcessLookupError as _exc:
                    logger.debug("Suppressed %s in core.resilience.supervisor: %s", type(_exc).__name__, _exc)
                return

            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                # 5-second poll timeout passed, process still alive
                continue
            except ProcessLookupError:
                break

        return_code = self.process.returncode
        await self._handle_exit(return_code)

    async def _handle_exit(self, return_code: int | None):
        """Decide whether/how quickly to restart based on exit code."""
        if not self.should_run:
            logger.info("Process exited (code %s). Supervisor stopping.", return_code)
            return

        grace_file = state_root() / "run" / "grace_exit.flag"
        graceful = return_code == 0 or self._grace_flag_matches_child(grace_file)
        if grace_file.exists():
            grace_file.unlink(missing_ok=True)

        if graceful:
            logger.info("Process exited cleanly/gracefully (code %s). Restarting in 5s...", return_code)
            await asyncio.sleep(5)
            return

        # ── Crash path ─────────────────────────────────────────────────────
        logger.warning("Process crashed/exited without grace flag (code %s)", return_code)
        
        now = time.time()
        if now - self.last_crash_time < _CRASH_WINDOW_SECONDS:
            self.crash_count += 1
        else:
            self.crash_count = 1  # Reset window
        self.last_crash_time = now

        backoff_s = min(_MAX_RESTART_BACKOFF_SECONDS, 2.0 ** min(self.crash_count - 1, 8))
        logger.info("Restarting after %.1fs crash backoff (crash #%d in current window)", backoff_s, self.crash_count)
        await asyncio.sleep(backoff_s)
        return

    def _grace_flag_matches_child(self, grace_file: Path) -> bool:
        """Accept a graceful-exit flag only from the child that just exited."""
        if not grace_file.exists() or self.process is None:
            return False
        try:
            payload = json.loads(grace_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        try:
            flag_pid = int(payload.get("pid"))
            created_at = float(payload.get("created_at_unix"))
        except (TypeError, ValueError):
            return False
        if flag_pid != int(self.process.pid):
            return False
        return 0.0 <= time.time() - created_at <= _GRACE_FLAG_TTL_SECONDS

    def _kill_process_tree(self, pid):
        """Kills the process and its children using psutil."""
        if not psutil:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                _record_supervisor_degradation(
                    exc,
                    action="ignored missing process while stopping supervised tree without psutil",
                    severity="debug",
                    extra={"pid": pid},
                )
            return

        try:
            table = get_resource_observer().process_tree(pid)
            target_pids = (
                [
                    process.pid
                    for process in table.processes
                ]
                if table.available
                else [pid]
            )
            handles = []
            for target_pid in reversed(target_pids):
                try:
                    handles.append(psutil.Process(target_pid))
                except psutil.NoSuchProcess:
                    continue
            for handle in handles:
                handle.terminate()

            _gone, alive = psutil.wait_procs(handles, timeout=3)
            for p in alive:
                p.kill()
        except psutil.NoSuchProcess as exc:
            _record_supervisor_degradation(
                exc,
                action="ignored missing process while stopping supervised tree",
                severity="debug",
                extra={"pid": pid},
            )

if __name__ == "__main__":
    # Example usage: Watch run_aura.py
    supervisor = SovereignSupervisor("run_aura.py", ["--server"])
    
    async def main():
        # Setup loop-based signal handling correctly if possible
        # For simplicity in __main__:
        await supervisor.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug("Exception caught during execution", exc_info=True)
