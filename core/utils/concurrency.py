import asyncio
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LOOP_BLOCKED_CEILING_FRACTION, LOOP_HOLD_STARVED_FRACTION
from core.utils.task_tracker import get_task_tracker, mark_task_protected

# Use the centralized enhanced logger
try:
    from core.utils.aura_logging import core_logger as logger
except ImportError:
    logger = logging.getLogger("Aura.Core")

# Sentinels
LOCK_SENTINEL = "LOCK_ACQUIRED"


async def cancel_and_join(
    task: asyncio.Task[Any] | None,
    *,
    owner: str,
    timeout: float | None = None,  # noqa: ASYNC109 - a teardown budget, not a wait
) -> None:
    """Cancel a task and wait for it, without eating our own cancellation.

    `task.cancel()` then `await task` is how every teardown in this tree
    stops a background loop, and a hundred and fifty of them catch
    `asyncio.CancelledError` and carry on. That is right for the CHILD's
    cancellation and wrong for ours: while we are awaiting, this task can be
    cancelled too, the same `except` catches that as well, and the caller
    keeps running after it was told to stop. It is the swallowed
    cancellation FAULT-001 names, and it is invisible — the teardown looks
    like it worked and reports success.

    `Task.cancelling()` counts the cancellations aimed at THIS task, so the
    two are separated by asking rather than by guessing. The child's own
    exception is carried to a degradation instead of dropped, because a
    background loop that died of something real is a fact somebody wants.
    """
    if task is None or (task.done() and task.cancelled()):
        return
    task.cancel()
    mine = asyncio.current_task()
    try:
        if timeout is None:
            await task
        else:
            await asyncio.wait_for(asyncio.shield(task), timeout)
    except asyncio.CancelledError:
        if mine is not None and mine.cancelling() > 0:
            raise
    except TimeoutError:
        record_degradation(
            owner,
            TimeoutError(f"{owner}: a cancelled task did not finish in {timeout}s"),
            severity="warning",
            action="left the task cancelled and carried on with teardown",
        )
    except Exception as exc:  # noqa: BLE001 - the child's death, carried not dropped
        record_degradation(
            owner,
            exc,
            severity="warning",
            action="teardown continued after the cancelled task raised",
        )


async def run_io_bound(func, *args, **kwargs):
    """
    Runs a blocking I/O bound function in a separate thread to avoid blocking the event loop.
    """
    import functools

    if hasattr(asyncio, "to_thread"):
        # Python 3.9+
        return await asyncio.to_thread(func, *args, **kwargs)
    else:
        # Fallback for older versions
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


class RobustLock:
    """
    A loop-agnostic version of a lock that implements the 'Timeout & Retry' protocol.
    Uses an internal threading.Lock to allow sharing across multiple asyncio event loops.
    ZENITH LOCKDOWN: Adaptive timeouts and GPU load scaling.
    """

    def __init__(
        self,
        name: str = "UnnamedLock",
        *,
        watchdog_threshold_s: float | None = None,
        force_release_on_stall: bool = False,
        timeout_s: float | None = None,
    ):
        import uuid

        self.name = name
        full_id = str(uuid.uuid4())
        self.id = full_id[:8]
        self._lock = threading.Lock()
        self.timeout = float(timeout_s) if timeout_s is not None else 30.0  # Base timeout (Zenith default)
        self.adaptive = True
        self.last_acquire_start = 0.0
        self.watchdog_threshold_s = watchdog_threshold_s
        self.force_release_on_stall = force_release_on_stall

    @staticmethod
    def _watchdog_report_acquire_start(
        watchdog: Any, lock_id: str, name: str, callback: Any, threshold_s: float | None = None
    ) -> None:
        try:
            watchdog.report_acquire_start(
                lock_id,
                name,
                on_stall=callback,
                threshold_s=threshold_s,
            )
        except TypeError:
            watchdog.report_acquire_start(lock_id, name)

    async def acquire_robust(  # noqa: ASYNC109
        self,
        timeout: float | None = None,  # noqa: ASYNC109
        max_retries: int = 3,
    ) -> bool:
        """
        Attempts to acquire the lock with a timeout and retries.
        """
        wait_time = timeout or self.timeout

        # 0. Register with Watchdog
        from core.resilience.lock_watchdog import get_lock_watchdog

        watchdog = get_lock_watchdog()
        stall_callback = self.force_release if self.force_release_on_stall else None
        self._watchdog_report_acquire_start(
            watchdog,
            self.id,
            self.name,
            stall_callback,
            self.watchdog_threshold_s,
        )

        # Adaptive Scaling: if GPU is saturated, extend timeout
        if self.adaptive:
            # We check the metrics collector for GPU load if possible
            try:
                from core.runtime.service_registry import get_runtime_service

                metrics = get_runtime_service("metrics", default=None)
                m = getattr(metrics, "_custom_gauges", {}).get("gpu_utilization", 0)
                if m > 0.8:
                    wait_time = max(wait_time, 180.0)
                    logger.debug(
                        "🛡️ [ADAPTIVE] GPU Saturated (%s). Extending '%s' timeout to %ss",
                        f"{m:.2f}",
                        self.name,
                        wait_time,
                    )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                record_degradation("concurrency", _exc)
                logger.debug("Suppressed Exception: %s", _exc)

        async def _await_threaded_acquire(acquire_timeout: float) -> bool:
            acquire_task = get_task_tracker().create_task(
                asyncio.to_thread(self._lock.acquire, timeout=acquire_timeout),
                name=f"lock_acquire:{self.name}",
            )
            try:
                # Pulse the watchdog while waiting so it doesn't trigger a stall on long adaptive timeouts
                start_wait = time.monotonic()
                while not acquire_task.done():
                    elapsed = time.monotonic() - start_wait
                    if elapsed > 0:
                        watchdog.report_wait_progress(
                            self.id
                        )  # Notify watchdog we are actively waiting
                    done, pending = await asyncio.wait([acquire_task], timeout=1.0)
                    if done:
                        break
                return acquire_task.result()
            except asyncio.CancelledError:
                # asyncio.to_thread cannot be cancelled safely; wait for the worker to
                # finish and immediately release any lock it may have acquired so we
                # don't strand the mutex or its watchdog entry forever.
                try:
                    acquired = await asyncio.wait_for(acquire_task, timeout=1.0)
                except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError):
                    acquired = False
                if acquired:
                    try:
                        self._lock.release()
                    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                        record_degradation("concurrency", _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
                watchdog.report_release(self.id)
                raise

        for attempt in range(max_retries):
            self.last_acquire_start = time.monotonic()
            self._watchdog_report_acquire_start(
                watchdog,
                self.id,
                self.name,
                stall_callback,
                self.watchdog_threshold_s,
            )
            try:
                success = await _await_threaded_acquire(wait_time)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("concurrency", e)
                watchdog.report_release(self.id)
                logger.error("Unexpected error acquiring lock '%s': %s", self.name, e)
                break

            if success:
                watchdog.report_acquire_success(self.id)
                logger.debug("Successfully locked: '%s'", self.name)
                return True

            watchdog.report_release(self.id)
            logger.warning(
                "Attempt %s/%s: Timeout waiting for '%s'.", attempt + 1, max_retries, self.name
            )
            await asyncio.sleep(random.uniform(0.1, 0.5))

        # Safety valve: force-release the EXISTING lock only for locks that
        # explicitly permit corruption-risk recovery. Boot and promotion locks
        # prefer fail-closed acquisition over replacing a live mutex.
        if not self.force_release_on_stall:
            watchdog.report_release(self.id)
            return False

        if not self.force_release():
            watchdog.report_release(self.id)
            return False
        watchdog.report_release(self.id)

        self._watchdog_report_acquire_start(
            watchdog,
            self.id,
            self.name,
            stall_callback,
            self.watchdog_threshold_s,
        )
        try:
            success = await _await_threaded_acquire(10.0)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, AttributeError, TypeError, ValueError):
            watchdog.report_release(self.id)
            raise
        if success:
            watchdog.report_acquire_success(self.id)
            return True
        watchdog.report_release(self.id)
        return False

    async def acquire(self) -> bool:
        """Standard async acquire with adaptive timeout."""
        return await self.acquire_robust()

    def release(self):
        """Release the lock and reset timing."""
        try:
            if self._lock.locked():
                self._lock.release()
                from core.resilience.lock_watchdog import get_lock_watchdog

                get_lock_watchdog().report_release(self.id)
                logger.debug("Released lock: '%s'", self.name)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("concurrency", e)
            logger.debug("RobustLock.release() error for '%s': %s", self.name, e)

    def force_release(self) -> bool:
        """CRITICAL: Force release the lock to break a detected deadlock.

        Replaces the lock entirely so that the blocked thread can proceed.
        The thread holding the old lock will release the old lock safely.
        """
        if not self.force_release_on_stall:
            exc = RuntimeError(f"force release disabled for lock '{self.name}'")
            record_degradation(
                "concurrency",
                exc,
                severity="critical",
                action="blocked_lock_force_release_without_explicit_opt_in",
                extra={"lock_id": self.id, "lock_name": self.name},
            )
            logger.error(
                "Force release blocked for lock '%s'; lock was configured fail-closed.",
                self.name,
            )
            return False
        logger.critical("⚠️ FORCE RELEASING LOCK '%s' due to deadlock watchdog!", self.name)
        try:
            self._lock = threading.Lock()
        except RuntimeError:
            # release() on an unlocked lock — harmless
            pass  # no-op: intentional
        except (AttributeError, TypeError, ValueError) as _exc:
            record_degradation("concurrency", _exc)
            logger.debug("Suppressed Exception: %s", _exc)
            return False
        return True

    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def is_locked(self) -> bool:
        """Compatibility property for Watchdog."""
        return self.locked()

    @property
    def held_duration(self) -> float:
        if not self.locked():
            return 0.0
        return time.monotonic() - self.last_acquire_start

    async def __aenter__(self):
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(f"failed to acquire robust lock '{self.name}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


@dataclass(frozen=True, order=True)
class LockableResource:
    """A resource identifier for strict ordering."""

    name: str
    lock: RobustLock


class DeadlockPrevention:
    """
    Implements the 'Strict Lock Ordering + Timeout Fallback' protocol.
    """

    @staticmethod
    async def acquire_multiple(  # noqa: ASYNC109
        resources: list[LockableResource],
        timeout: float = 2.0,  # noqa: ASYNC109
        max_retries: int = 5,
    ) -> bool:
        """
        Acquires multiple locks in a strict alphabetical order with timeout/retry logic.
        """
        # Step 1: Strict Ordering (Sort the requested resources by name)
        sorted_resources = sorted(resources, key=lambda r: r.name)
        resource_names = [r.name for r in sorted_resources]
        logger.debug("Starting multi-lock acquisition for: %s", resource_names)

        for attempt in range(max_retries):
            acquired_locks: list[LockableResource] = []
            success = True

            for res in sorted_resources:
                # Inner acquisition uses no retries here because the outer loop handles it
                # But we use the robust method for the timeout logic
                if await res.lock.acquire_robust(timeout=timeout, max_retries=1):
                    acquired_locks.append(res)
                else:
                    success = False
                    break

            if success:
                logger.info("All locks acquired for: %s", resource_names)
                return True
            else:
                # Phase 4a: Release what we managed to grab
                for res in reversed(acquired_locks):
                    res.lock.release()

                logger.debug(
                    "Backing off after failed attempt %s for %s", attempt + 1, resource_names
                )
                # Phase 4b: Randomized backoff before next attempt
                await asyncio.sleep(random.uniform(0.1, 0.5))

        # CRITICAL: Failed to resolve contention
        logger.error(
            f"MULTI-LOCK TRANSACTION FAILED: Max retries ({max_retries}) reached "
            f"for resources {resource_names}. Possible systemic deadlock."
        )
        return False

    @staticmethod
    def release_multiple(resources: list[LockableResource]):
        """Releases multiple locks in reverse order."""
        # Sorting is not strictly necessary for release but good for consistency
        sorted_resources = sorted(resources, key=lambda r: r.name, reverse=True)
        for res in sorted_resources:
            if res.lock.locked():
                res.lock.release()


# Global Lock Registry to enforce unique names
_LOCK_REGISTRY: dict[str, RobustLock] = {}


def get_robust_lock(name: str) -> RobustLock:
    """Returns a named RobustLock instance, creating it if necessary."""
    if name not in _LOCK_REGISTRY:
        _LOCK_REGISTRY[name] = RobustLock(name)
    return _LOCK_REGISTRY[name]


#: The defaults in EventLoopMonitor's own signature. A declared flag has one
#: spec per knob, so these cannot come from a caller's argument.
_SIGNATURE_DEFAULTS = {
    "AURA_EVENT_LOOP_MONITOR_THRESHOLD_S": 0.75,
    "AURA_EVENT_LOOP_MONITOR_STARTUP_GRACE_S": 300.0,
}


class EventLoopMonitor:
    """
    Monitors the asyncio event loop for blocking operations.
    If the loop is delayed by more than 'threshold' seconds beyond its
    intended sleep interval, it logs a warning.
    """

    def __init__(
        self, threshold: float = 0.75, interval: float = 1.0, startup_grace: float = 300.0
    ):
        # Declared, so the knob has one owner, one spelling and one way of
        # being parsed. Seven raw reads in this constructor each had their own
        # try/except and their own idea of what "true" looks like.
        from core.runtime.flags import FlagKind, declare, env_bool, env_float

        _owner = "core.utils.concurrency.EventLoopMonitor"

        def _seconds(name: str, fallback: float, description: str) -> float:
            """The knob when it is set, otherwise what this caller asked for.

            A declaration is one spec per knob, and these two defaults are
            constructor arguments — so declaring them with the caller's value
            made the second monitor with a different threshold a conflicting
            declaration. The declared default is the signature's; the caller's
            argument is used only where nothing set the knob at all, which is
            what the raw `os.getenv(name, str(threshold))` meant.
            """
            resolved, source = declare(
                name,
                kind=FlagKind.FLOAT,
                default=_SIGNATURE_DEFAULTS[name],
                description=description,
                owner=_owner,
            ).value_with_source()
            if source.startswith("default"):
                return float(fallback)
            try:
                return float(resolved)
            except (TypeError, ValueError):
                return float(fallback)

        self.threshold = _seconds(
            "AURA_EVENT_LOOP_MONITOR_THRESHOLD_S",
            threshold,
            "Loop lag, in seconds, that counts as a breach while idle.",
        )
        self.active_threshold = max(
            self.threshold,
            env_float(
                "AURA_EVENT_LOOP_MONITOR_ACTIVE_THRESHOLD_S",
                default=5.0,
                description="Loop lag tolerated while a turn is being served.",
                owner=_owner,
            ),
        )
        self.interval = interval
        self.startup_grace = _seconds(
            "AURA_EVENT_LOOP_MONITOR_STARTUP_GRACE_S",
            startup_grace,
            "Seconds after start before loop lag is reported at all.",
        )
        self.log_transient_lag = env_bool(
            "AURA_EVENT_LOOP_LOG_TRANSIENTS",
            default=False,
            description="Log every transient lag sample, not only sustained breaches.",
            owner=_owner,
        )
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_lag: float = 0.0
        self._last_sample_at: float = 0.0
        self._last_sample_monotonic: float = 0.0
        self._peak_lag: float = 0.0
        #: (wall time, lag seconds) for the last five minutes at one sample a
        #: second. The stability guardian reads this window rather than
        #: keeping a sleep of its own on the same loop.
        self._lag_samples: deque[tuple[float, float]] = deque(maxlen=300)
        self._last_breach_lag: float = 0.0
        self._last_breach_at: float = 0.0
        self._consecutive_breaches: int = 0
        #: Lag samples the host caused, and when that was last said.
        self._starved_samples: int = 0
        self._last_starvation_log_at: float = 0.0
        self._started_at: float = 0.0
        self._last_failure_at: float = 0.0
        self._last_failure_reason: str = ""
        self._healthy_lag_samples_after_failure: int = 0
        self._last_incident_at: float = 0.0
        self._last_incident_reason: str = ""
        self._last_recovered_at: float = 0.0
        self._incident_count: int = 0
        from core.runtime.flags import env_int

        self.hard_failure_threshold = env_float(
            "AURA_EVENT_LOOP_MONITOR_HARD_FAILURE_S",
            default=5.0,
            description="Loop lag, in seconds, that is recorded as a hard failure.",
            owner=_owner,
        )
        self.failure_recovery_window_s = env_float(
            "AURA_EVENT_LOOP_MONITOR_FAILURE_RECOVERY_S",
            default=15.0,
            description="Seconds of healthy samples before a failure counts as recovered.",
            owner=_owner,
        )
        self.failure_recovery_samples = max(
            1,
            env_int(
                "AURA_EVENT_LOOP_MONITOR_RECOVERY_SAMPLES",
                default=3,
                description="Healthy samples required before declaring recovery.",
                owner=_owner,
            ),
        )

    def _active_runtime_reason(self) -> str | None:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active():
                return "proof_run_active"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        try:
            from core.runtime.service_registry import get_runtime_service

            gate = get_runtime_service("inference_gate", default=None)
            status_getter = getattr(gate, "get_conversation_status", None)
            if callable(status_getter):
                status = status_getter()
                if isinstance(status, dict):
                    lane_state = str(status.get("state") or "").strip().lower()
                    if (
                        bool(status.get("active"))
                        or bool(status.get("foreground_owned"))
                        or bool(status.get("warmup_in_flight"))
                        or int(status.get("active_generations", 0) or 0) > 0
                        or float(status.get("current_request_started_at", 0.0) or 0.0) > 0.0
                        or lane_state in {"spawning", "handshaking", "warming", "recovering"}
                    ):
                        return "foreground_generation"
                elif bool(getattr(status, "active", False)):
                    return "foreground_generation"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        try:
            from core.runtime import foreground_guard

            reason = foreground_guard.foreground_activity_reason()
            if reason:
                return str(reason)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        return None

    def _lag_threshold_for_context(self) -> tuple[float, str]:
        reason = self._active_runtime_reason()
        if reason:
            return self.active_threshold, reason
        return self.threshold, "idle"

    def start(self):
        """Starts the monitor in a background task."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        # A restarted monitor must not publish the previous task's final sample
        # as current scheduling pressure while its first new tick is pending.
        self._last_lag = 0.0
        self._last_sample_at = 0.0
        self._last_sample_monotonic = 0.0
        self._task = get_task_tracker().create_task(self._run())
        mark_task_protected(self._task, owner="event_loop_monitor")
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._owner_loop = None
        logger.info(
            "🕒 EventLoopMonitor started (threshold=%.2fs, interval=%.1fs)",
            self.threshold,
            self.interval,
        )

    def _task_running(self) -> bool:
        return bool(
            self._task is not None
            and not self._task.done()
            and not self._stop_event.is_set()
        )

    def is_running(self) -> bool:
        """Return lifecycle liveness without conflating it with lag health.

        The desired-state control plane uses this probe to decide whether the
        monitor task needs a restart. ``is_alive`` remains the stricter health
        contract and stays false while a hard-lag incident is recovering.
        Restarting a running monitor cannot heal historical lag and previously
        trapped the control plane in start/probe/stop churn.
        """
        return self._task_running()

    def ensure_running(self) -> bool:
        """Restart the monitor task if supervision finds it stopped.

        Health checks must not turn a dead monitor into a healthy signal, but
        the runtime should also not remain degraded forever after a monitor task
        exits during model warmup or shutdown/restart races. This returns the
        post-restart task state; hard-lag recovery is still judged by
        ``is_alive`` after healthy samples accrue.
        """
        if self._task_running():
            return True
        try:
            self.start()
        except RuntimeError as exc:
            # Health checks run on plain threads, where task creation raises —
            # so a dead monitor could never be revived by the very pulse that
            # detected it, and the runtime stayed DEGRADED for 84 minutes with
            # working restart machinery (observed live 2026-07-05). Hand the
            # restart to the owning loop instead.
            owner_loop = getattr(self, "_owner_loop", None)
            if owner_loop is not None and not owner_loop.is_closed():
                owner_loop.call_soon_threadsafe(self.start)
                logger.info("EventLoopMonitor restart scheduled onto owning loop from thread.")
            else:
                record_degradation(
                    "event_loop_monitor",
                    exc,
                    severity="degraded",
                    action="event-loop monitor restart failed; no live owning loop",
                    enforce_failure_policy=False,
                )
                logger.warning("EventLoopMonitor restart failed: %s", exc)
            return False
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "event_loop_monitor",
                exc,
                severity="degraded",
                action="event-loop monitor restart failed; runtime remains unhealthy",
                enforce_failure_policy=False,
            )
            logger.warning("EventLoopMonitor restart failed: %s", exc)
            return False
        return self._task_running()

    async def stop(self):
        """Stops the monitor gracefully."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("🕒 EventLoopMonitor stopped.")

    def is_alive(self) -> bool:
        """Return lifecycle and sampling liveness, independent of incident state.

        A hard-lag incident must remain visible until recovery is proved, but it
        does not mean the monitor task died. Conflating those states made model
        admission wait for the entire incident-stability window even while the
        monitor was publishing fresh healthy samples, creating a boot recovery
        deadlock.
        """
        if not self._task_running():
            self.ensure_running()
            return False
        freshness_budget_s = self._sample_freshness_budget_s()
        now = time.perf_counter()
        if self._last_sample_monotonic > 0.0:
            if now - self._last_sample_monotonic > freshness_budget_s:
                return False
        elif self._started_at > 0.0 and now - self._started_at > freshness_budget_s:
            return False
        return True

    def is_healthy(self) -> bool:
        """Return current health after consecutive-sample incident recovery."""
        if not self.is_alive():
            return False
        if self._last_failure_at:
            stable_for = time.time() - self._last_failure_at
            if (
                self._healthy_lag_samples_after_failure < self.failure_recovery_samples
                or stable_for < self.failure_recovery_window_s
            ):
                return False
        return True

    def _sample_freshness_budget_s(self) -> float:
        """Maximum age at which a lag sample still describes current pressure."""
        return max(2.0, float(self.interval) * 3.0)

    def _capture_lag_sample(
        self,
        lag: float,
        *,
        sampled_at: float | None = None,
        sampled_monotonic: float | None = None,
    ) -> None:
        current = max(0.0, float(lag))
        self._last_lag = current
        self._last_sample_at = float(sampled_at if sampled_at is not None else time.time())
        self._last_sample_monotonic = float(
            sampled_monotonic if sampled_monotonic is not None else time.perf_counter()
        )
        self._peak_lag = max(self._peak_lag, current)
        self._lag_samples.append((self._last_sample_at, current))

    def lag_samples(self, window_s: float) -> list[tuple[float, float]]:
        """(wall time, lag seconds) samples from the last ``window_s`` seconds."""
        cutoff = time.time() - max(0.0, float(window_s))
        return [(at, lag) for at, lag in self._lag_samples if at >= cutoff]

    def last_lag_sample(self) -> tuple[float, float] | None:
        """The most recent lag reading and its age in seconds, or None before one."""
        if self._last_sample_monotonic <= 0.0:
            return None
        return self._last_lag, max(0.0, time.perf_counter() - self._last_sample_monotonic)

    def get_status(self) -> dict[str, Any]:
        alive = self.is_alive()
        healthy = self.is_healthy() if alive else False
        sample_age_s = (
            max(0.0, time.perf_counter() - self._last_sample_monotonic)
            if self._last_sample_monotonic > 0.0
            else None
        )
        sample_fresh = bool(
            sample_age_s is not None
            and sample_age_s <= self._sample_freshness_budget_s()
        )
        return {
            "alive": alive,
            "healthy": healthy,
            "running": self._task_running(),
            "last_lag_s": self._last_lag,
            "last_sample_at_unix": self._last_sample_at,
            "sample_age_s": round(sample_age_s, 4) if sample_age_s is not None else None,
            "sample_fresh": sample_fresh,
            "sample_freshness_budget_s": self._sample_freshness_budget_s(),
            "peak_lag_s": self._peak_lag,
            "last_breach_lag_s": self._last_breach_lag,
            "last_breach_at_unix": self._last_breach_at,
            "consecutive_breaches": self._consecutive_breaches,
            "starved_samples": self._starved_samples,
            "last_failure_at": self._last_failure_at,
            "last_failure_reason": self._last_failure_reason,
            "incident_active": bool(self._last_failure_at),
            "incident_count": self._incident_count,
            "last_incident_at": self._last_incident_at,
            "last_incident_reason": self._last_incident_reason,
            "last_recovered_at": self._last_recovered_at,
            "healthy_recovery_samples": self._healthy_lag_samples_after_failure,
            "required_recovery_samples": self.failure_recovery_samples,
            "recovery_window_s": self.failure_recovery_window_s,
        }

    def _note_starvation(self, lag: float, loop_share: float) -> None:
        """Lag the host caused: counted, said once a minute, not a breach."""
        self._starved_samples += 1
        now = time.monotonic()
        if now - self._last_starvation_log_at >= 60.0:
            self._last_starvation_log_at = now
            logger.warning(
                "Event loop starved, not blocked: %.2fs of lag on %.0f%% of a core "
                "(%d such samples this run). The host is not scheduling it.",
                lag,
                100.0 * loop_share,
                self._starved_samples,
            )

    async def _run(self):
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            cpu_start = time.thread_time()
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

            end_time = time.perf_counter()
            actual_elapsed = end_time - start_time
            lag = actual_elapsed - self.interval
            sampled_at = time.time()
            # This runs on the loop thread, so its own CPU clock over the
            # sample says what the lag was: a loop thread that got a few
            # percent of a core was runnable and not scheduled, which is the
            # host's doing and nothing in this process can repair. LIVE
            # 2026-09-16, load 34 on 18 cores: 52 CRITICAL degradations in
            # 22 minutes for lag the loop was not causing.
            loop_share = (time.thread_time() - cpu_start) / max(actual_elapsed, 1e-9)
            starved = LOOP_BLOCKED_CEILING_FRACTION <= loop_share < LOOP_HOLD_STARVED_FRACTION
            self._capture_lag_sample(
                lag,
                sampled_at=sampled_at,
                sampled_monotonic=end_time,
            )
            threshold, context = self._lag_threshold_for_context()
            in_startup_grace = (
                self.startup_grace > 0
                and self._started_at > 0
                and (end_time - self._started_at) < self.startup_grace
            )

            if lag > threshold and not in_startup_grace and starved:
                self._note_starvation(lag, loop_share)
            elif lag > threshold and not in_startup_grace:
                self._last_breach_lag = max(0.0, lag)
                self._last_breach_at = sampled_at
                self._consecutive_breaches += 1
                if lag >= self.hard_failure_threshold:
                    if not self._last_failure_at:
                        self._incident_count += 1
                    self._last_failure_at = sampled_at
                    self._last_failure_reason = (
                        f"hard event-loop lag {lag:.4f}s exceeded {self.hard_failure_threshold:.2f}s"
                    )
                    self._last_incident_at = sampled_at
                    self._last_incident_reason = self._last_failure_reason
                    self._healthy_lag_samples_after_failure = 0
                    record_degradation(
                        "event_loop_monitor",
                        RuntimeError(self._last_failure_reason),
                        severity="critical",
                        action="marked event loop monitor unhealthy until healthy lag samples confirm recovery",
                        enforce_failure_policy=False,
                    )
                severe = lag >= max(threshold * 3.0, 0.50)
                if severe or self._consecutive_breaches >= 5:
                    logger.warning(
                        "🚨 EVENT LOOP LAG DETECTED: %.4fs (context=%s threshold=%.2fs, streak=%d). "
                        "Something is blocking the event loop!",
                        lag,
                        context,
                        threshold,
                        self._consecutive_breaches,
                    )
                    try:
                        from core.resilience.omni_tracer import write_trace

                        write_trace(
                            "event_loop_monitor",
                            "EventLoopLag",
                            (
                                f"lag={lag:.4f}s context={context} threshold={threshold:.2f}s "
                                f"streak={self._consecutive_breaches}"
                            ),
                        )
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        logger.debug(
                            "Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc
                        )
                elif self.log_transient_lag:
                    logger.debug(
                        "EventLoopMonitor: transient lag %.4fs observed (context=%s threshold=%.2fs).",
                        lag,
                        context,
                        threshold,
                    )
            else:
                self._consecutive_breaches = 0
                if self._last_failure_at:
                    self._healthy_lag_samples_after_failure += 1
                    stable_for = time.time() - self._last_failure_at
                    if (
                        self._healthy_lag_samples_after_failure >= self.failure_recovery_samples
                        and stable_for >= self.failure_recovery_window_s
                    ):
                        logger.info(
                            "EventLoopMonitor recovered after %d healthy lag samples over %.1fs.",
                            self._healthy_lag_samples_after_failure,
                            stable_for,
                        )
                        self._last_recovered_at = time.time()
                        self._last_failure_at = 0.0
                        self._last_failure_reason = ""


class RobustSemaphore:
    """
    A loop-agnostic version of a semaphore that uses threading.Semaphore
    to bridge across multiple event loops.
    """

    def __init__(self, value: int = 1, name: str = "UnnamedSemaphore"):
        self.name = name
        self._sem = threading.BoundedSemaphore(value)

    async def acquire(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Acquire the semaphore asynchronously using to_thread.

        When a timeout is provided, the timeout is enforced inside the backing
        thread primitive itself. That avoids orphaned background threads that
        can still acquire the permit after asyncio-side cancellation/timeout.
        """
        logger.debug("Attempting to acquire semaphore: '%s'", self.name)
        if timeout is None:
            acquired = await asyncio.to_thread(self._sem.acquire)
        else:
            acquired = await asyncio.to_thread(self._sem.acquire, True, max(0.0, float(timeout)))
        if acquired:
            logger.debug("Acquired semaphore: '%s'", self.name)
        else:
            logger.debug("Semaphore acquire timed out: '%s'", self.name)
        return bool(acquired)

    def release(self):
        """Release the underlying threading.Semaphore."""
        self._sem.release()
        logger.debug("Released semaphore: '%s'", self.name)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


_SEM_REGISTRY: dict[str, RobustSemaphore] = {}


def get_robust_semaphore(name: str, value: int = 1) -> RobustSemaphore:
    """Returns a named RobustSemaphore instance, creating it if necessary."""
    if name not in _SEM_REGISTRY:
        _SEM_REGISTRY[name] = RobustSemaphore(value, name)
    return _SEM_REGISTRY[name]
