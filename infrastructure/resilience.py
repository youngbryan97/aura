"""infrastructure/resilience.py
────────────────────────────
Resilience primitives: retry, circuit-breaker, and the @resilient decorator.

v3.0 — Thread-safe, async-first rewrite with proper locking.

C-06 FIX: CircuitBreaker now uses threading.Lock for thread-safe state
transitions. All state mutations are protected against TOCTOU races.
Retry uses exponential backoff with full jitter.
"""
from __future__ import annotations


import asyncio
import functools
import logging
import random
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger("Infra.Resilience")


# ── Exceptions ────────────────────────────────────────────────

class RetryExhausted(Exception):
    """All retry attempts failed."""

    def __init__(self, message: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.last_error = last_error


# ── Circuit Breaker ───────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Thread-safe circuit breaker with proper locking.

    C-06 FIX: All state mutations are now protected by a threading.Lock
    to prevent TOCTOU races under concurrent access. The half-open state
    correctly allows only one probe call through.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._total_opens = 0
        self._half_open_probe_active = False

    @property
    def state(self) -> CircuitState:
        """Get current state with automatic OPEN → HALF_OPEN transition."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._opened_at is not None
                and (time.time() - self._opened_at) >= self._recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_active = False
                logger.info(
                    "Circuit '%s' transitioning OPEN → HALF_OPEN", self.name
                )
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        In HALF_OPEN state, only one probe request is allowed.
        """
        with self._lock:
            # Check for OPEN → HALF_OPEN transition
            if (
                self._state == CircuitState.OPEN
                and self._opened_at is not None
                and (time.time() - self._opened_at) >= self._recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_active = False

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if not self._half_open_probe_active:
                    self._half_open_probe_active = True
                    return True
                return False
            else:  # OPEN
                return False

    def record_failure(self) -> None:
        """Record a failure. Trip to OPEN if threshold exceeded."""
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._half_open_probe_active = False
                logger.warning(
                    "Circuit '%s' HALF_OPEN probe failed → OPEN", self.name
                )
            elif self._failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._total_opens += 1
                logger.warning(
                    "Circuit '%s' TRIPPED (%d failures) → OPEN",
                    self.name,
                    self._failures,
                )

    def record_success(self) -> None:
        """Record a success. Reset to CLOSED if in HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_probe_active = False
                logger.info("Circuit '%s' recovered → CLOSED", self.name)
            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success
                self._failures = max(0, self._failures - 1)


# ── Async Circuit Breaker ─────────────────────────────────────

class AsyncCircuitBreaker:
    """Async-friendly circuit breaker wrapping the thread-safe CircuitBreaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self._cb = CircuitBreaker(name, failure_threshold, recovery_timeout)

    @property
    def name(self) -> str:
        return self._cb.name

    @property
    def state(self) -> CircuitState:
        return self._cb.state

    def allow_request(self) -> bool:
        return self._cb.allow_request()

    def record_failure(self) -> None:
        self._cb.record_failure()

    def record_success(self) -> None:
        self._cb.record_success()


class _SafeDatabaseLockGuard:
    def __init__(self, lock: threading.RLock, *, name: str, timeout: Optional[float]) -> None:
        self._lock = lock
        self._name = name
        self._timeout = timeout

    def __enter__(self) -> threading.RLock:
        if self._timeout is None:
            acquired = self._lock.acquire()
        else:
            acquired = self._lock.acquire(timeout=max(0.0, float(self._timeout)))
        if not acquired:
            raise TimeoutError(f"Timed out acquiring database lock for {self._name}")
        return self._lock

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._lock.release()
        return False


class SafeDatabaseLock:
    """Small context-managed lock with timeout semantics for DB critical sections."""

    def __init__(self, name: str, timeout_s: float = 5.0) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self._lock = threading.RLock()

    def acquire(self, timeout: Optional[float] = None) -> _SafeDatabaseLockGuard:
        return _SafeDatabaseLockGuard(
            self._lock,
            name=self.name,
            timeout=self.timeout_s if timeout is None else timeout,
        )


# ── Retry Logic ───────────────────────────────────────────────


async def retry_async(
    fn: Callable,
    *args,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
) -> Any:
    """Retry an async function with exponential backoff and full jitter.

    Uses the "Full Jitter" algorithm from AWS Architecture Blog:
    sleep = random_between(0, min(cap, base * 2 ** attempt))
    """
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except retryable as exc:
            last_error = exc
            if attempt < attempts - 1:
                # Full jitter: random(0, min(max_delay, base * 2^attempt))
                delay = random.uniform(
                    0, min(max_delay, base_delay * (2**attempt))
                )
                logger.debug(
                    "Retry %d/%d for %s after %.2fs: %s",
                    attempt + 1,
                    attempts,
                    getattr(fn, "__name__", "unknown"),
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
    raise RetryExhausted(
        f"All {attempts} attempts failed for {getattr(fn, '__name__', 'unknown')}",
        last_error=last_error,
    )


# ── @resilient Decorator ──────────────────────────────────────

_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def _get_or_create_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Thread-safe breaker registry."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name, failure_threshold, recovery_timeout
            )
        return _breakers[name]


def resilient(
    component: str,
    retry_attempts: int = 3,
    base_delay: float = 1.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    fallback: Any = None,
):
    """Decorator combining retry + circuit-breaker for resilient calls.

    Works on both sync and async functions.
    """

    def decorator(fn: Callable) -> Callable:
        breaker = _get_or_create_breaker(
            component, failure_threshold, recovery_timeout
        )

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            if not breaker.allow_request():
                logger.warning(
                    "Circuit '%s' is OPEN — returning fallback", component
                )
                if callable(fallback):
                    return fallback()
                return fallback

            try:
                result = await retry_async(
                    fn,
                    *args,
                    attempts=retry_attempts,
                    base_delay=base_delay,
                    **kwargs,
                )
                breaker.record_success()
                return result
            except RetryExhausted as exc:
                breaker.record_failure()
                raise exc

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            if not breaker.allow_request():
                logger.warning(
                    "Circuit '%s' is OPEN — returning fallback", component
                )
                if callable(fallback):
                    return fallback()
                return fallback

            last_error = None
            for attempt in range(retry_attempts):
                try:
                    result = fn(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as exc:
                    last_error = exc
                    if attempt < retry_attempts - 1:
                        delay = random.uniform(
                            0, min(30.0, base_delay * (2**attempt))
                        )
                        # Avoid blocking event loop if we are accidentally in one
                        try:
                            asyncio.get_running_loop()
                            logger.error("CRITICAL: Sync resilient function '%s' called from ASYNC context. This blocks the event loop!", fn.__name__)
                            # We still sleep because the function is sync and MUST return, 
                            # but this log alerts us to a major architectural debt.
                            time.sleep(delay)
                        except RuntimeError:
                            time.sleep(delay)

            breaker.record_failure()
            raise RetryExhausted(
                f"All {retry_attempts} attempts failed for {fn.__name__}",
                last_error=last_error,
            )

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


# ── Infrastructure Hardening System ───────────────────────────

class InfrastructureHardeningSystem:
    """Centralized resilience management with thread-safe breaker registry."""

    _instance: Optional["InfrastructureHardeningSystem"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("InfrastructureHardeningSystem initialized")

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Get or create a named circuit breaker (thread-safe)."""
        return _get_or_create_breaker(name, failure_threshold, recovery_timeout)

    def get_health(self) -> Dict[str, Any]:
        """Return health status of all circuit breakers."""
        with _breakers_lock:
            return {
                name: {
                    "state": cb.state.name,
                    "failures": cb._failures,
                    "total_opens": cb._total_opens,
                }
                for name, cb in _breakers.items()
            }


_system: Optional[InfrastructureHardeningSystem] = None


def get_resilience_system() -> InfrastructureHardeningSystem:
    global _system
    if _system is None:
        _system = InfrastructureHardeningSystem()
    return _system
