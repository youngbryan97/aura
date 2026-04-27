"""StallWatchdog: Async Event Loop Monitoring + Active Recovery
Part of Aura's Neural Neuro-Surgeon (Phase 29).

Design notes:
- The watchdog runs in its own daemon thread so it survives even if the
  asyncio loop is wedged.
- A heartbeat is scheduled via call_soon_threadsafe every second. If the
  loop fails to run that callback within `threshold` seconds, we record a
  stall.
- On a long stall we now do more than log: we (a) dump task state, (b)
  cancel asyncio tasks that look hung, and (c) signal subsystems to
  recycle. This is what turns "we noticed the freeze" into "we ended
  the freeze."
"""

import asyncio
import logging
import os
import time
import threading
import traceback
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Aura.Resilience.Watchdog")

# How long an asyncio task can be pending (not done) before the watchdog
# considers it suspect during a stall and cancels it. Conservative — only
# fires after a confirmed stall, not as routine cleanup.
_TASK_HUNG_SECONDS = 90.0

# Minimum stall length to trigger active recovery. Below this, we just log.
_ACTIVE_RECOVERY_THRESHOLD = 30.0


class StallWatchdog(threading.Thread):
    """Monitor thread that tracks event loop responsiveness."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold: float = 5.0):
        super().__init__(daemon=True, name="AuraStallWatchdog")
        self.loop = loop
        self.threshold = threshold
        self._last_heartbeat = time.time()
        self._running = False
        self._stop_event = threading.Event()
        self._task_birth: dict[int, float] = {}
        self._consecutive_long_stalls: int = 0

    def run(self):
        logger.info("🛡️ StallWatchdog: Monitoring loop (Threshold: %.1fs)", self.threshold)
        self._running = True

        while not self._stop_event.is_set():
            # Schedule a heartbeat on the loop
            try:
                if self.loop.is_closed():
                    logger.debug("StallWatchdog: event loop closed, exiting.")
                    break
                self.loop.call_soon_threadsafe(self._heartbeat)
            except RuntimeError:
                # Event loop closed during shutdown — exit silently
                break
            except Exception as e:
                logger.debug("Watchdog heartbeat schedule issue: %s", e)

            time.sleep(1.0)  # Check every second

            # Check for stall
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.threshold:
                self._report_stall(elapsed)
                if elapsed >= _ACTIVE_RECOVERY_THRESHOLD:
                    self._consecutive_long_stalls += 1
                    self._attempt_active_recovery(elapsed)
                else:
                    self._consecutive_long_stalls = 0
                # Reset so the next stall measurement is fresh.
                self._last_heartbeat = time.time()
            else:
                self._consecutive_long_stalls = 0

    def stop(self):
        self._stop_event.set()

    def _heartbeat(self):
        self._last_heartbeat = time.time()
        # Track task ages so a future stall can pick out which ones look hung.
        # This runs on the loop thread — cheap and safe.
        try:
            now = time.time()
            tasks = asyncio.all_tasks(self.loop)
            seen = set()
            for task in tasks:
                tid = id(task)
                seen.add(tid)
                if tid not in self._task_birth:
                    self._task_birth[tid] = now
            # Drop dead bookkeeping
            for tid in list(self._task_birth.keys()):
                if tid not in seen:
                    self._task_birth.pop(tid, None)
        except Exception as exc:
            logger.debug("Task age bookkeeping failed: %s", exc)

    def _report_stall(self, elapsed: float):
        logger.error("🚨 [WATCHDOG] EVENT LOOP STALL DETECTED! (Elapsed: %.1fs)", elapsed)

        # Dump tracebacks of all threads
        dump_dir = Path("data/error_logs/stalls")
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_file = dump_dir / f"stall_{int(time.time())}.txt"

        with open(dump_file, "w") as f:
            f.write(f"STALL DETECTED: {elapsed:.1f}s\n")
            f.write("=" * 40 + "\n")
            for thread_id, frame in sys._current_frames().items():
                f.write(f"\nThread ID: {thread_id}\n")
                traceback.print_stack(frame, file=f)

        logger.info("💉 [IMMUNE] Stall traceback dumped to: %s", dump_file)

        # Proactively trigger Neuro-Surgeon analysis
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            get_diagnostic_hub()
            # Future: trigger auto-repair or circuit break
        except Exception as _e:
            logger.debug('Ignored Exception in stall_watchdog.py: %s', _e)

    def _attempt_active_recovery(self, elapsed: float) -> None:
        """Don't just log a stall — try to break it.

        We schedule a recovery coroutine onto the (possibly wedged) loop.
        If the loop is truly frozen, the coroutine will queue and run when
        the loop wakes up — and it will then prevent the next stall by
        cancelling the hung tasks. If the loop is partially responsive, the
        coroutine runs immediately.
        """
        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._recover_on_loop(elapsed), loop=self.loop
                )
            )
        except RuntimeError:
            return
        except Exception as exc:
            logger.debug("Stall recovery scheduling failed: %s", exc)

    async def _recover_on_loop(self, elapsed: float) -> None:
        """Cancel hung tasks and ask known subsystems to recycle.

        Runs on the asyncio loop. Conservative — we only cancel tasks that
        have been alive for far longer than this stall, and we never touch
        the kernel main loop / watchdog / orchestrator coordinator tasks.
        """
        protected_substrings = (
            "AuraKernel",
            "OrchestratorMainLoop",
            "AuraStallWatchdog",
            "ConsciousnessLoopMonitor",
            "Server.Chat",
            "uvicorn",
        )
        cutoff = time.time() - max(_TASK_HUNG_SECONDS, elapsed * 1.5)
        cancelled = 0
        try:
            tasks = asyncio.all_tasks(self.loop)
        except RuntimeError:
            return

        for task in tasks:
            if task.done():
                continue
            name = getattr(task, "get_name", lambda: "")() or repr(task)
            if any(p in name for p in protected_substrings):
                continue
            birth = self._task_birth.get(id(task))
            if birth is None or birth >= cutoff:
                continue
            try:
                task.cancel()
                cancelled += 1
            except Exception as exc:
                logger.debug("Stall recovery: failed to cancel %s: %s", name, exc)

        if cancelled:
            logger.warning(
                "💉 [IMMUNE] Stall recovery cancelled %d hung asyncio tasks (stall=%.0fs).",
                cancelled,
                elapsed,
            )

        # Ask the brainstem and cortex MLX clients to self-check; the
        # stale-handshake path in mlx_client._ensure_worker_alive will
        # recycle anyone that's been wedged.
        try:
            from core.brain.llm.mlx_client import _LIVE_MLX_CLIENTS  # type: ignore
        except Exception:
            _LIVE_MLX_CLIENTS = None  # type: ignore

        if _LIVE_MLX_CLIENTS:
            for client in list(_LIVE_MLX_CLIENTS):
                try:
                    if hasattr(client, "_lane_state") and client._lane_state == "handshaking":
                        # Schedule a no-op alive probe so the stale-handshake
                        # branch fires on next entry.
                        self.loop.call_soon(client._mark_progress)
                except Exception as exc:
                    logger.debug("Stall recovery MLX poke failed: %s", exc)

        # If we've taken many long stalls in a row, ask the orchestrator's
        # state vault to flush so we don't lose continuity.
        if self._consecutive_long_stalls >= 3:
            try:
                from core.container import ServiceContainer
                state_repo = ServiceContainer.get("state_repository", default=None)
                if state_repo and hasattr(state_repo, "request_flush"):
                    state_repo.request_flush()
                    logger.info("💉 [IMMUNE] Requested state vault flush after %d consecutive stalls.", self._consecutive_long_stalls)
            except Exception as exc:
                logger.debug("Stall recovery state-flush request failed: %s", exc)

def start_watchdog(loop: Optional[asyncio.AbstractEventLoop] = None, threshold: float = 5.0):
    """Convenience helper to start the watchdog."""
    try:
        target_loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        target_loop = asyncio.new_event_loop()
    dog = StallWatchdog(target_loop, threshold=threshold)
    dog.start()
    return dog
