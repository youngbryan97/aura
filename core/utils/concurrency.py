import asyncio
import logging
import random
import time
from typing import List, Optional, Any, Set, Dict
from dataclasses import dataclass

# Use the centralized enhanced logger
try:
    from core.utils.aura_logging import core_logger as logger
except ImportError:
    logger = logging.getLogger("Aura.Core")

# Sentinels
LOCK_SENTINEL = "LOCK_ACQUIRED"

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

import threading

class RobustLock:
    """
    A loop-agnostic version of a lock that implements the 'Timeout & Retry' protocol.
    Uses an internal threading.Lock to allow sharing across multiple asyncio event loops.
    ZENITH LOCKDOWN: Adaptive timeouts and GPU load scaling.
    """
    def __init__(self, name: str = "UnnamedLock"):
        import uuid
        self.name = name
        full_id = str(uuid.uuid4())
        self.id = full_id[:8]
        self._lock = threading.Lock()
        self.timeout = 30.0  # Base timeout (Zenith default)
        self.adaptive = True
        self.last_acquire_start = 0.0

    @staticmethod
    def _watchdog_report_acquire_start(watchdog: Any, lock_id: str, name: str, callback: Any) -> None:
        try:
            watchdog.report_acquire_start(lock_id, name, on_stall=callback)
        except TypeError:
            watchdog.report_acquire_start(lock_id, name)

    async def acquire_robust(self, timeout: Optional[float] = None, max_retries: int = 3) -> bool:
        """
        Attempts to acquire the lock with a timeout and retries.
        """
        wait_time = timeout or self.timeout
        
        # 0. Register with Watchdog
        from core.resilience.lock_watchdog import get_lock_watchdog
        watchdog = get_lock_watchdog()
        self._watchdog_report_acquire_start(watchdog, self.id, self.name, self.force_release)
        
        # Adaptive Scaling: if GPU is saturated, extend timeout
        if self.adaptive:
            # We check the metrics collector for GPU load if possible
            try:
                from core.observability.metrics import get_metrics
                m = get_metrics()._gauges.get("gpu_utilization", 0)
                if m > 0.8:
                    wait_time = max(wait_time, 180.0)
                    logger.debug(f"🛡️ [ADAPTIVE] GPU Saturated ({m:.2f}). Extending '{self.name}' timeout to {wait_time}s")
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        async def _await_threaded_acquire(acquire_timeout: float) -> bool:
            acquire_task = asyncio.create_task(
                asyncio.to_thread(self._lock.acquire, timeout=acquire_timeout),
                name=f"lock_acquire:{self.name}",
            )
            try:
                return await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                # asyncio.to_thread cannot be cancelled safely; wait for the worker to
                # finish and immediately release any lock it may have acquired so we
                # don't strand the mutex or its watchdog entry forever.
                try:
                    acquired = await acquire_task
                except Exception:
                    acquired = False
                if acquired:
                    try:
                        self._lock.release()
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                watchdog.report_release(self.id)
                raise

        for attempt in range(max_retries):
            self.last_acquire_start = time.monotonic()
            self._watchdog_report_acquire_start(watchdog, self.id, self.name, self.force_release)
            try:
                success = await _await_threaded_acquire(wait_time)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                watchdog.report_release(self.id)
                logger.error(f"Unexpected error acquiring lock '{self.name}': {e}")
                break

            if success:
                watchdog.report_acquire_success(self.id)
                logger.debug(f"Successfully locked: '{self.name}'")
                return True

            watchdog.report_release(self.id)
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: Timeout waiting for '{self.name}'.")
            await asyncio.sleep(random.uniform(0.1, 0.5))

        # Safety valve: force-release the EXISTING lock (don't reinitialize)
        self.force_release()
        watchdog.report_release(self.id)

        self._watchdog_report_acquire_start(watchdog, self.id, self.name, self.force_release)
        try:
            success = await _await_threaded_acquire(10.0)
        except asyncio.CancelledError:
            raise
        except Exception:
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
                logger.debug(f"Released lock: '{self.name}'")
        except Exception as e:
            logger.debug(f"RobustLock.release() error for '{self.name}': {e}")

    def force_release(self):
        """CRITICAL: Force release the lock to break a detected deadlock.

        Releases the EXISTING lock object instead of reinitializing it.
        Reinitializing would orphan any thread still holding the old reference,
        causing a permanent deadlock when the finally block tries to release
        the wrong object.
        """
        logger.critical(f"⚠️ FORCE RELEASING LOCK '{self.name}' due to deadlock watchdog!")
        try:
            if self._lock.locked():
                self._lock.release()
        except RuntimeError:
            # release() on an unlocked lock — harmless
            pass
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def is_locked(self) -> bool:
        """Compatibility property for Watchdog."""
        return self.locked()

    @property
    def held_duration(self) -> float:
        if not self.locked(): return 0.0
        return time.monotonic() - self.last_acquire_start

    async def __aenter__(self):
        await self.acquire()
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
    async def acquire_multiple(resources: List[LockableResource], timeout: float = 2.0, max_retries: int = 5) -> bool:
        """
        Acquires multiple locks in a strict alphabetical order with timeout/retry logic.
        """
        # Step 1: Strict Ordering (Sort the requested resources by name)
        sorted_resources = sorted(resources, key=lambda r: r.name)
        resource_names = [r.name for r in sorted_resources]
        logger.debug(f"Starting multi-lock acquisition for: {resource_names}")
        
        for attempt in range(max_retries):
            acquired_locks: List[LockableResource] = []
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
                logger.info(f"All locks acquired for: {resource_names}")
                return True
            else:
                # Phase 4a: Release what we managed to grab
                for res in reversed(acquired_locks):
                    res.lock.release()
                
                logger.debug(f"Backing off after failed attempt {attempt + 1} for {resource_names}")
                # Phase 4b: Randomized backoff before next attempt
                await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # CRITICAL: Failed to resolve contention
        logger.error(
            f"MULTI-LOCK TRANSACTION FAILED: Max retries ({max_retries}) reached "
            f"for resources {resource_names}. Possible systemic deadlock."
        )
        return False

    @staticmethod
    def release_multiple(resources: List[LockableResource]):
        """Releases multiple locks in reverse order."""
        # Sorting is not strictly necessary for release but good for consistency
        sorted_resources = sorted(resources, key=lambda r: r.name, reverse=True)
        for res in sorted_resources:
            if res.lock.locked():
                res.lock.release()

# Global Lock Registry to enforce unique names
_LOCK_REGISTRY: Dict[str, RobustLock] = {}

def get_robust_lock(name: str) -> RobustLock:
    """Returns a named RobustLock instance, creating it if necessary."""
    if name not in _LOCK_REGISTRY:
        _LOCK_REGISTRY[name] = RobustLock(name)
    return _LOCK_REGISTRY[name]

class EventLoopMonitor:
    """
    Monitors the asyncio event loop for blocking operations.
    If the loop is delayed by more than 'threshold' seconds beyond its
    intended sleep interval, it logs a warning.
    """
    def __init__(self, threshold: float = 0.1, interval: float = 1.0, startup_grace: float = 15.0):
        self.threshold = threshold
        self.interval = interval
        self.startup_grace = startup_grace
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_lag: float = 0.0
        self._consecutive_breaches: int = 0
        self._started_at: float = 0.0

    def start(self):
        """Starts the monitor in a background task."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._task = asyncio.create_task(self._run())
        logger.info("🕒 EventLoopMonitor started (threshold=%.2fs, interval=%.1fs)", 
                    self.threshold, self.interval)

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

    async def _run(self):
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            
            end_time = time.perf_counter()
            actual_elapsed = end_time - start_time
            lag = actual_elapsed - self.interval
            in_startup_grace = (
                self.startup_grace > 0
                and self._started_at > 0
                and (end_time - self._started_at) < self.startup_grace
            )

            if lag > self.threshold and not in_startup_grace:
                self._last_lag = lag
                self._consecutive_breaches += 1
                severe = lag >= max(self.threshold * 3.0, 0.50)
                if severe or self._consecutive_breaches >= 5:
                    logger.warning(
                        "🚨 EVENT LOOP LAG DETECTED: %.4fs (threshold=%.2fs, streak=%d). "
                        "Something is blocking the event loop!",
                        lag,
                        self.threshold,
                        self._consecutive_breaches,
                    )
                else:
                    logger.debug(
                        "EventLoopMonitor: transient lag %.4fs observed (threshold=%.2fs).",
                        lag,
                        self.threshold,
                    )
            else:
                self._consecutive_breaches = 0

class RobustSemaphore:
    """
    A loop-agnostic version of a semaphore that uses threading.Semaphore 
    to bridge across multiple event loops.
    """
    def __init__(self, value: int = 1, name: str = "UnnamedSemaphore"):
        self.name = name
        self._sem = threading.BoundedSemaphore(value)

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire the semaphore asynchronously using to_thread.

        When a timeout is provided, the timeout is enforced inside the backing
        thread primitive itself. That avoids orphaned background threads that
        can still acquire the permit after asyncio-side cancellation/timeout.
        """
        logger.debug(f"Attempting to acquire semaphore: '{self.name}'")
        if timeout is None:
            acquired = await asyncio.to_thread(self._sem.acquire)
        else:
            acquired = await asyncio.to_thread(self._sem.acquire, True, max(0.0, float(timeout)))
        if acquired:
            logger.debug(f"Acquired semaphore: '{self.name}'")
        else:
            logger.debug(f"Semaphore acquire timed out: '{self.name}'")
        return bool(acquired)

    def release(self):
        """Release the underlying threading.Semaphore."""
        self._sem.release()
        logger.debug(f"Released semaphore: '{self.name}'")

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

_SEM_REGISTRY: Dict[str, RobustSemaphore] = {}

def get_robust_semaphore(name: str, value: int = 1) -> RobustSemaphore:
    """Returns a named RobustSemaphore instance, creating it if necessary."""
    if name not in _SEM_REGISTRY:
        _SEM_REGISTRY[name] = RobustSemaphore(value, name)
    return _SEM_REGISTRY[name]
