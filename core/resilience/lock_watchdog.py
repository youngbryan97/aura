import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional
from core.utils.singleton import singleton

logger = logging.getLogger("Aura.LockWatchdog")


@dataclass
class _TrackedLock:
    start_time: float
    name: str
    on_stall: Optional[Callable[[], Any]] = None
    interventions: int = 0
    last_alert_at: float = 0.0
    last_intervention_at: float = 0.0

@singleton
class LockWatchdog:
    """
    Centralized monitor for all system locks.
    Prevents deadlocks by tracking lock duration and triggering force-releases
    if they persist beyond safe thresholds.
    """
    def __init__(self, check_interval: float = 10.0, threshold: float = 180.0):
        self._active_locks: Dict[str, _TrackedLock] = {}
        self._check_interval = check_interval
        self._threshold = threshold
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._intervention_cooldown = max(check_interval, threshold / 2.0)

    def start(self):
        """Starts the background monitoring task."""
        if self._running:
            return
        self._running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self._monitor_loop(),
                name="aura.lock_watchdog",
            )
        except Exception:
            self._task = asyncio.create_task(self._monitor_loop(), name="aura.lock_watchdog")
        logger.info(f"🛡️ LockWatchdog ACTIVE (Threshold: {self._threshold}s).")

    async def stop(self):
        """Stops the monitoring task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        self._task = None

    def report_acquire_start(self, lock_id: str, name: str, on_stall: Optional[Callable[[], Any]] = None):
        """Called when a lock acquisition begins."""
        existing = self._active_locks.get(lock_id)
        self._active_locks[lock_id] = _TrackedLock(
            start_time=time.monotonic(),
            name=name,
            on_stall=on_stall or (existing.on_stall if existing else None),
            interventions=existing.interventions if existing else 0,
            last_alert_at=existing.last_alert_at if existing else 0.0,
            last_intervention_at=existing.last_intervention_at if existing else 0.0,
        )

    def report_acquire_success(self, lock_id: str):
        """Called when a lock is successfully acquired."""
        # We update the time to the actual hold start
        tracked = self._active_locks.get(lock_id)
        if tracked is not None:
            tracked.start_time = time.monotonic()

    def report_release(self, lock_id: str):
        """Called when a lock is released."""
        self._active_locks.pop(lock_id, None)

    def get_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        locks = []
        for lock_id, tracked in self._active_locks.items():
            locks.append(
                {
                    "lock_id": lock_id,
                    "name": tracked.name,
                    "held_duration_s": round(max(0.0, now - tracked.start_time), 3),
                    "interventions": tracked.interventions,
                    "last_alert_at": tracked.last_alert_at,
                    "last_intervention_at": tracked.last_intervention_at,
                }
            )
        locks.sort(key=lambda item: item["held_duration_s"], reverse=True)
        return {
            "threshold_s": self._threshold,
            "check_interval_s": self._check_interval,
            "active_count": len(locks),
            "locks": locks,
        }

    async def _attempt_recovery(self, lock_id: str, tracked: _TrackedLock) -> bool:
        callback = tracked.on_stall
        if callback is None:
            return False
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
            refreshed = self._active_locks.get(lock_id)
            if refreshed is not None:
                refreshed.interventions += 1
                refreshed.last_intervention_at = time.monotonic()
            logger.critical(
                "🛠️ LockWatchdog intervention executed for '%s' (ID: %s).",
                tracked.name,
                lock_id,
            )
            return True
        except Exception as exc:
            logger.error("LockWatchdog recovery failed for '%s': %s", tracked.name, exc)
            return False

    async def _monitor_loop(self):
        """Background loop to check for stalled locks."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                now = time.monotonic()
                for lock_id, tracked in list(self._active_locks.items()):
                    held_duration = now - tracked.start_time
                    if held_duration > self._threshold:
                        if (now - tracked.last_alert_at) >= self._check_interval:
                            tracked.last_alert_at = now
                            logger.critical(
                                "🚨 DEADLOCK ALERT: Lock '%s' (ID: %s) held for %.1fs!",
                                tracked.name,
                                lock_id,
                                held_duration,
                            )
                            try:
                                from core.health.degraded_events import record_degraded_event

                                record_degraded_event(
                                    "lock_watchdog",
                                    "stalled_lock",
                                    detail=f"{tracked.name}:{held_duration:.1f}s",
                                    severity="critical",
                                    classification="background_degraded",
                                    context={
                                        "lock_id": lock_id,
                                        "held_duration_s": round(held_duration, 3),
                                        "interventions": tracked.interventions,
                                    },
                                )
                            except Exception as exc:
                                logger.debug("LockWatchdog degraded event emit failed: %s", exc)
                        if tracked.on_stall and (
                            tracked.interventions == 0
                            or (now - tracked.last_intervention_at) >= self._intervention_cooldown
                        ):
                            await self._attempt_recovery(lock_id, tracked)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in LockWatchdog loop: {e}")

def get_lock_watchdog() -> LockWatchdog:
    return LockWatchdog()
