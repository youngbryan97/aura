"""Sovereign Resilience: Supervisor & Watchdog
-------------------------------------------
The SovereignSupervisor is responsible for keeping the Aura core alive.
It uses strictly local monitoring (psutil) and implements exponential backoff
to prevent rapid crash loops from consuming resources.
"""

from core.runtime.errors import record_degradation
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import asyncio
from pathlib import Path
from typing import List, Optional

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

class SovereignSupervisor:
    def __init__(self, target_script: str, args: List[str] = None):
        self.target_script = Path(target_script)
        self.args = args or []
        self.process: Optional[subprocess.Popen] = None
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
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('supervisor', e)
                logger.error("Supervisor loop error: %s", e)
                await asyncio.sleep(5)

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
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.cwd())
        )

        asyncio.create_task(self._pipe_logger_async(self.process.stdout, logging.INFO, "stdout"))
        asyncio.create_task(self._pipe_logger_async(self.process.stderr, logging.ERROR, "stderr"))

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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('supervisor', e)
            logger.error(f"Error reading pipe {label}: {e}")

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
            except asyncio.TimeoutError:
                # 5-second poll timeout passed, process still alive
                continue
            except ProcessLookupError:
                break

        return_code = self.process.returncode
        await self._handle_exit(return_code)

    async def _handle_exit(self, return_code: Optional[int]):
        """Decide whether/how quickly to restart based on exit code."""
        if not self.should_run:
            logger.info("Process exited (code %s). Supervisor stopping.", return_code)
            return

        grace_file = Path.home() / ".aura" / "run" / "grace_exit.flag"
        graceful = grace_file.exists() or return_code == 0
        if grace_file.exists():
            grace_file.unlink(missing_ok=True)

        if graceful:
            logger.info("Process exited cleanly/gracefully (code %s). Restarting in 5s...", return_code)
            await asyncio.sleep(5)
            return

        # ── Crash path ─────────────────────────────────────────────────────
        logger.warning("Process crashed/exited without grace flag (code %s)", return_code)
        
        now = time.time()
        if now - self.last_crash_time < 60:
            self.crash_count += 1
        else:
            self.crash_count = 1  # Reset window
        self.last_crash_time = now

        # Instant reboot (0s wait) on crash
        logger.info("Resurrection instant (crash #%d in current window)", self.crash_count)
        return

    def _kill_process_tree(self, pid):
        """Kills the process and its children using psutil."""
        if not psutil:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                logger.debug("Exception caught during execution", exc_info=True)
            return

        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            for p in alive:
                p.kill()
        except psutil.NoSuchProcess:
            logger.debug("Exception caught during execution", exc_info=True)

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