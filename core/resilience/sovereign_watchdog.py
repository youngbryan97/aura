"""core/resilience/sovereign_watchdog.py — Enterprise Uptime Guardian

The Sovereign Watchdog is the final layer of Aura's resilience. It monitors 
the Orchestrator's heartbeat and the availability of critical services. 
If a deadlock or cognitive stall is detected, it triggers a recovery sequence.
"""
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("Aura.SovereignWatchdog")

class SovereignWatchdog:
    """Guarantees system availability through active monitoring and recovery."""

    def __init__(self, orchestrator, interval: float = 15.0, timeout: float = 120.0):
        self._orchestrator = orchestrator
        self._interval = interval
        self._timeout = timeout
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_heartbeat = time.monotonic()
        self._recovery_count = 0
        self._last_recovery_time = 0.0  # Cooldown: prevent repeated recovery spam
        self._recovery_cooldown = 300.0  # 5 minutes between recoveries

    async def start(self):
        """Start the watchdog loop."""
        if self._running:
            return
        self._running = True
        self._last_heartbeat = time.monotonic()
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("🛡️ Sovereign Watchdog ACTIVE (Timeout: %.1fs)", self._timeout)

    async def stop(self):
        """Stop the watchdog."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in sovereign_watchdog.py: %s', _e)

    def heartbeat(self, component: str = "orchestrator"):
        """Called by the Orchestrator to signal life."""
        self._last_heartbeat = time.monotonic()

    async def _watch_loop(self):
        """Periodic check for system vitality."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                
                elapsed = time.monotonic() - self._last_heartbeat
                if elapsed > self._timeout:
                    # Cooldown: don't spam recovery if we just did one
                    now = time.monotonic()
                    if now - self._last_recovery_time < self._recovery_cooldown:
                        logger.debug("Watchdog: stall detected but cooling down (last recovery %.0fs ago).",
                                     now - self._last_recovery_time)
                        continue
                    logger.critical("🚨 SOVEREIGN WATCHDOG: HEARTBEAT STALL DETECTED (%.1fs elapsed)", elapsed)
                    await self._execute_recovery()
                
                # Also check if the event loop is heavily blocked
                loop_start = time.monotonic()
                await asyncio.sleep(0.01)
                loop_elapsed = time.monotonic() - loop_start
                if loop_elapsed > 5.0:  # Only fire on genuine stalls, not GIL contention during model loads
                    logger.info("⏱️ SOVEREIGN WATCHDOG: Event loop lag detected (%.2fs block)", loop_elapsed)
                    try:
                        from core.event_bus import get_event_bus
                        get_event_bus().publish_threadsafe("telemetry", {
                            "type": "event_loop_lag",
                            "lag_seconds": round(loop_elapsed, 2),
                            "metadata": {"system": True},
                        })
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Watchdog loop error: %s", e)

    async def _execute_recovery(self):
        """Attempt to recover from a stall."""
        self._recovery_count += 1
        self._last_recovery_time = time.monotonic()
        logger.warning("🛠️ Initiating Recovery Sequence #%d...", self._recovery_count)
        
        # 1. Clear GPU Sentinel (common cause of deadlocks)
        try:
            from core.utils.gpu_sentinel import get_gpu_sentinel
            sentinel = get_gpu_sentinel()
            # RLock does not have .locked(). We try to acquire without blocking to check if it's held.
            if not sentinel._lock.acquire(blocking=False):
                logger.warning("🛠️ GPU Sentinel lock is HELD. Force-releasing.")
                # Since it's an RLock, we might need to release multiple times if it's recursive,
                # but for recovery, we just want to break the deadlock.
                sentinel.release()
            else:
                # If we acquired it, it wasn't deadlocked (at least not by this lock), so release it back.
                sentinel._lock.release()
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # 2. Reset Heartbeat to give time for recovery
        self.heartbeat()

        # 3. Notification via Telemetry
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            bus.publish_threadsafe("telemetry", {
                "type": "aura_message",
                "message": "⚠️ I've detected a cognitive stall in my primary reasoning loop. I'm auto-retrying with a cleared context window now. Please wait...",
                "metadata": {
                    "system": True, 
                    "recovery": True,
                    "actionable": True,
                    "recovery_count": self._recovery_count
                }
            })
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # 4. Signal Orchestrator to reset internal state if supported
        if hasattr(self._orchestrator, "reset_internal_state"):
            try:
                await self._orchestrator.reset_internal_state()
            except Exception as e:
                logger.error("Failed to reset orchestrator state: %s", e)

        logger.info("🛠️ Recovery sequence complete. Monitoring for stabilization.")
