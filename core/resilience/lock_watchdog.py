import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.concurrency import cancel_and_join
from core.utils.singleton import singleton

logger = logging.getLogger("Aura.LockWatchdog")


@dataclass
class _TrackedLock:
    start_time: float
    name: str
    on_stall: Callable[[], Any] | None = None
    threshold_s: float | None = None
    interventions: int = 0
    last_alert_at: float = 0.0
    last_intervention_at: float = 0.0
    #: Whether this hold has ever been excused by observed progress. Counted
    #: so the log can say how long a legitimate hold actually ran.
    working_looks: int = 0


def _the_holder_is_working() -> bool:
    """Whether the turn this lock is part of is still producing tokens.

    A lock held while real work is being done is not a deadlock, and this
    watchdog had no way to tell the two apart. LIVE, 2026-09-21: twenty-nine
    ``DEADLOCK ALERT: Lock 'AuraKernel.StateLock' held for 438.8s`` while the
    kernel tick that held it was mid-generation on the Brainstem — each one
    a CRITICAL log line, a degraded event at critical severity, and a
    StabilityGuardian DEGRADED card, for a turn that was working.

    The hold is real and it is long, which is worth knowing. What it is not
    is a deadlock, and the difference decides whether a recovery callback
    force-releases a lock out from under live work.

    ``still_producing`` is the same reading the cognitive engine renews its
    own cycle from, the mlx client tells a slow decode from a wedged one
    with, and the stall watchdog's starvation carve-out uses. This is the
    fourth caller and the first that could have force-released something.
    """
    try:
        from core.brain.llm.thinking_reserve import seconds_to_decode
        from core.runtime.turn_progress import normal_gap_between_tokens, still_producing

        quiet_for = normal_gap_between_tokens(float(seconds_to_decode(64)))
        return bool(still_producing(within_s=quiet_for))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Unmeasurable is not working: an unreadable progress signal must not
        # excuse a hold forever.
        logger.debug(
            "lock watchdog could not read turn progress (%s: %s); "
            "judging the hold on its duration alone",
            type(exc).__name__,
            exc,
        )
        return False

@singleton
class LockWatchdog:
    """
    Centralized monitor for all system locks.
    Prevents deadlocks by tracking lock duration and triggering force-releases
    if they persist beyond safe thresholds.
    """
    def __init__(self, check_interval: float = 10.0, threshold: float = 180.0):
        self._active_locks: dict[str, _TrackedLock] = {}
        self._active_locks_guard = threading.RLock()
        self._check_interval = check_interval
        self._threshold = threshold
        self._running = False
        self._task: asyncio.Task | None = None
        self._intervention_cooldown = max(check_interval, threshold / 2.0)
        self.last_start_error: str = ""

    def start(self) -> bool:
        """Starts the background monitoring task."""
        if self._running:
            return True
        self._running = True
        self.last_start_error = ""
        monitor = self._monitor_loop()
        try:
            from core.runtime.task_ownership import close_awaitable, create_tracked_task

            self._task = create_tracked_task(
                monitor,
                name="aura.lock_watchdog",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            self._running = False
            self._task = None
            try:
                close_awaitable(monitor)
            except NameError:
                if inspect.iscoroutine(monitor):
                    monitor.close()
            except (RuntimeError, AttributeError, TypeError) as close_exc:
                record_degradation(
                    "lock_watchdog",
                    close_exc,
                    severity="critical",
                    action="marked watchdog startup failed after coroutine cleanup failed",
                    classification=FallbackClassification.AUDIT_GAP,
                )
            self.last_start_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "lock_watchdog",
                exc,
                severity="critical",
                action="left watchdog stopped and exposed last_start_error for health checks",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
            )
            logger.error("LockWatchdog failed to start under tracked task ownership: %s", exc)
            return False
        logger.info("🛡️ LockWatchdog ACTIVE (Threshold: %ss).", self._threshold)
        return True

    async def stop(self):
        """Stops the monitoring task."""
        self._running = False
        if self._task and not self._task.done():
            await cancel_and_join(self._task, owner="core.resilience.lock_watchdog")
        self._task = None

    def report_acquire_start(
        self,
        lock_id: str,
        name: str,
        on_stall: Callable[[], Any] | None = None,
        threshold_s: float | None = None,
    ):
        """Called when a lock acquisition begins."""
        with self._active_locks_guard:
            existing = self._active_locks.get(lock_id)
            self._active_locks[lock_id] = _TrackedLock(
                start_time=time.monotonic(),
                name=name,
                on_stall=on_stall or (existing.on_stall if existing else None),
                threshold_s=threshold_s if threshold_s is not None else (existing.threshold_s if existing else None),
                interventions=existing.interventions if existing else 0,
                last_alert_at=existing.last_alert_at if existing else 0.0,
                last_intervention_at=existing.last_intervention_at if existing else 0.0,
            )

    def report_acquire_success(self, lock_id: str):
        """Called when a lock is successfully acquired."""
        # We update the time to the actual hold start
        with self._active_locks_guard:
            tracked = self._active_locks.get(lock_id)
            if tracked is not None:
                tracked.start_time = time.monotonic()

    def report_wait_progress(self, lock_id: str):
        """Called to indicate a lock is actively waiting, preventing false stalls."""
        with self._active_locks_guard:
            tracked = self._active_locks.get(lock_id)
            if tracked is not None:
                tracked.start_time = time.monotonic()

    def report_release(self, lock_id: str):
        """Called when a lock is released."""
        with self._active_locks_guard:
            self._active_locks.pop(lock_id, None)

    def get_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        locks = []
        with self._active_locks_guard:
            active_items = list(self._active_locks.items())
        for lock_id, tracked in active_items:
            locks.append(
                {
                    "lock_id": lock_id,
                    "name": tracked.name,
                    "held_duration_s": round(max(0.0, now - tracked.start_time), 3),
                    "threshold_s": tracked.threshold_s if tracked.threshold_s is not None else self._threshold,
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
            "running": self._running,
            "last_start_error": self.last_start_error,
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
            with self._active_locks_guard:
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
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('lock_watchdog', exc)
            logger.error("LockWatchdog recovery failed for '%s': %s", tracked.name, exc)
            return False

    async def _monitor_loop(self):
        """Background loop to check for stalled locks."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                now = time.monotonic()
                with self._active_locks_guard:
                    active_items = list(self._active_locks.items())
                for lock_id, tracked in active_items:
                    held_duration = now - tracked.start_time
                    threshold_s = tracked.threshold_s if tracked.threshold_s is not None else self._threshold
                    if held_duration > threshold_s:
                        if _the_holder_is_working():
                            with self._active_locks_guard:
                                refreshed = self._active_locks.get(lock_id)
                                if refreshed is not None:
                                    refreshed.working_looks += 1
                                    refreshed.last_alert_at = now
                            if (now - tracked.last_alert_at) >= self._check_interval:
                                logger.info(
                                    "⏳ LockWatchdog: '%s' has been held %.1fs and the "
                                    "turn is still producing; not a deadlock (%d look(s)).",
                                    tracked.name,
                                    held_duration,
                                    tracked.working_looks + 1,
                                )
                            continue
                        if (now - tracked.last_alert_at) >= self._check_interval:
                            with self._active_locks_guard:
                                refreshed = self._active_locks.get(lock_id)
                                if refreshed is not None:
                                    refreshed.last_alert_at = now
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
                            except (ImportError, AttributeError, RuntimeError) as exc:
                                record_degradation('lock_watchdog', exc)
                                logger.debug("LockWatchdog degraded event emit failed: %s", exc)
                        if tracked.on_stall and (
                            tracked.interventions == 0
                            or (now - tracked.last_intervention_at) >= self._intervention_cooldown
                        ):
                            await self._attempt_recovery(lock_id, tracked)
                
            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('lock_watchdog', e)
                logger.error("Error in LockWatchdog loop: %s", e)

def get_lock_watchdog() -> LockWatchdog:
    return LockWatchdog()
