"""StallWatchdog: Async Event Loop Monitoring
Part of Aura's Neural Neuro-Surgeon (Phase 29).
"""

import asyncio
import logging
import time
import threading
import traceback
import sys
from typing import Optional

logger = logging.getLogger("Aura.Resilience.Watchdog")

class StallWatchdog(threading.Thread):
    """Monitor thread that tracks event loop responsiveness."""
    
    def __init__(self, loop: asyncio.AbstractEventLoop, threshold: float = 5.0):
        super().__init__(daemon=True, name="AuraStallWatchdog")
        self.loop = loop
        self.threshold = threshold
        self._last_heartbeat = time.time()
        self._running = False
        self._stop_event = threading.Event()

    def run(self):
        logger.info("🛡️ StallWatchdog: Monitoring loop (Threshold: %.1fs)", self.threshold)
        self._running = True
        
        while not self._stop_event.is_set():
            # Schedule a heartbeat on the loop
            try:
                if self.loop.is_closed():
                    logger.debug("StallWatchdog: event loop closed, exiting.")
                    break
                self.loop.call_soon_threadsafe(lambda: self._heartbeat())
            except RuntimeError:
                # Event loop closed during shutdown — exit silently
                break
            except Exception as e:
                logger.debug("Watchdog heartbeat schedule issue: %s", e)

            time.sleep(1.0) # Check every second
            
            # Check for stall
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.threshold:
                self._report_stall(elapsed)
                # Reset to avoid spamming
                self._last_heartbeat = time.time()

    def stop(self):
        self._stop_event.set()

    def _heartbeat(self):
        self._last_heartbeat = time.time()

    def _report_stall(self, elapsed: float):
        logger.error("🚨 [WATCHDOG] EVENT LOOP STALL DETECTED! (Elapsed: %.1fs)", elapsed)
        
        # Dump tracebacks of all threads
        import os
        from pathlib import Path
        dump_dir = Path("data/error_logs/stalls")
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_file = dump_dir / f"stall_{int(time.time())}.txt"
        
        with open(dump_file, "w") as f:
            f.write(f"STALL DETECTED: {elapsed:.1f}s\n")
            f.write("="*40 + "\n")
            for thread_id, frame in sys._current_frames().items():
                f.write(f"\nThread ID: {thread_id}\n")
                traceback.print_stack(frame, file=f)
        
        logger.info("💉 [IMMUNE] Stall traceback dumped to: %s", dump_file)
        
        # Proactively trigger Neuro-Surgeon analysis
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            hub = get_diagnostic_hub()
            # Future: trigger auto-repair or circuit break
        except Exception as _e:
            logger.debug('Ignored Exception in stall_watchdog.py: %s', _e)

def start_watchdog(loop: Optional[asyncio.AbstractEventLoop] = None, threshold: float = 5.0):
    """Convenience helper to start the watchdog."""
    try:
        target_loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        target_loop = asyncio.new_event_loop()
    dog = StallWatchdog(target_loop, threshold=threshold)
    dog.start()
    return dog
