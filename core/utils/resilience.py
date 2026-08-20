from core.runtime.errors import record_degradation
import time
import logging
import asyncio
from enum import Enum
from typing import Optional, Callable, Any

logger = logging.getLogger("Utils.Resilience")

class CircuitState(Enum):
    CLOSED = "closed"   # Normal operation
    OPEN = "open"       # Failed, bypassing
    HALF_OPEN = "half-open" # Testing for recovery

class CircuitBreaker:
    """Implementation of the Circuit Breaker pattern with exponential backoff."""
    
    def __init__(self, name: str, max_failures: int = 3, reset_timeout: float = 30.0):
        self.name = name
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "reset_timeout": self.reset_timeout
        }

    def from_dict(self, data: dict):
        if not data: return
        self.state = CircuitState(data.get("state", "closed"))
        self.failure_count = data.get("failure_count", 0)
        self.last_failure_time = data.get("last_failure_time")
        self.reset_timeout = data.get("reset_timeout", 30.0)
        
    @property
    def is_available(self) -> bool:
        """Check if the service can be called."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if cooldown period has passed
            elapsed = time.time() - (self.last_failure_time or 0)
            if elapsed > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("⚡ Circuit [%s] Transition: OPEN -> HALF_OPEN (Cooldown finished)", self.name)
                return True
            return False
            
        if self.state == CircuitState.HALF_OPEN:
            # Only allow one request at a time to test
            return True
            
        return False

    #: A probe may shorten the cooldown, never below this. Zero would mean an
    #: OPEN circuit is retried instantly, which is the thundering retry the
    #: breaker exists to stop.
    MIN_PROBE_RESET_TIMEOUT = 5.0

    def request_probe(self, *, reason: str, requested_timeout: Optional[float] = None) -> bool:
        """Ask a tripped circuit to allow ONE trial call.

        CP126 39211fcc: recovery paths reached in and assigned ``state`` and
        ``reset_timeout`` directly. That can half-open a breaker while the
        underlying failure is still active, leaves the failure counter saying
        something the state contradicts, and skips every log line that makes a
        transition auditable. Worse, the assignment could LENGTHEN nothing and
        shorten anything — including to zero.

        This is the transition API those callers were missing. It refuses on a
        CLOSED circuit (there is nothing to reopen), never touches the failure
        count — so a failing probe trips again exactly as it should — and can
        only shorten the cooldown, down to a floor.

        Returns whether the circuit is now allowing a trial call.
        """
        if self.state == CircuitState.CLOSED:
            return False
        if requested_timeout is not None:
            try:
                requested = float(requested_timeout)
            except (TypeError, ValueError):
                requested = self.reset_timeout
            if requested == requested and requested not in (float("inf"), float("-inf")):
                self.reset_timeout = max(
                    self.MIN_PROBE_RESET_TIMEOUT, min(self.reset_timeout, requested)
                )
        previous = self.state
        self.state = CircuitState.HALF_OPEN
        logger.info(
            "⚡ Circuit [%s] Transition: %s -> HALF_OPEN (probe requested: %s, cooldown %.0fs)",
            self.name,
            previous.value,
            str(reason or "unspecified"),
            self.reset_timeout,
        )
        return True

    def record_success(self):
        """Reset failures on successful call."""
        if self.state != CircuitState.CLOSED:
            logger.info("✅ Circuit [%s] Transition: %s -> CLOSED (Success detected)", self.name, self.state.value)
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None

    def record_failure(self, error: Any = None):
        """Track failures and trip the circuit if threshold met."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        # User-facing response phases get gentler backoff (max 30s)
        # to prevent long periods where Aura can't speak
        is_response_critical = "response" in self.name.lower() or "unitary" in self.name.lower()
        max_backoff = 30.0 if is_response_critical else 300.0

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.reset_timeout = min(self.reset_timeout * 1.5, max_backoff)
            logger.warning("Circuit [%s] Half-Open Failure. Back to OPEN. Timeout: %ds", self.name, self.reset_timeout)

        elif self.failure_count >= self.max_failures:
            self.state = CircuitState.OPEN
            # Don't let response-critical circuits stay open long
            if is_response_critical:
                self.reset_timeout = min(self.reset_timeout, 15.0)
            logger.critical("Circuit [%s] TRIPPED! OPEN. Failures: %d.", self.name, self.failure_count)

async def run_with_watchdog(task_name: str, coro_or_fn: Any, timeout: float = 5.0, fallback: Any = None):
    """Executes a task with a strict timeout and logging."""
    try:
        if asyncio.iscoroutine(coro_or_fn):
            return await asyncio.wait_for(coro_or_fn, timeout=timeout)
        elif callable(coro_or_fn):
            # If it's a synchronous function, run it in a thread to prevent blocking
            return await asyncio.wait_for(asyncio.to_thread(coro_or_fn), timeout=timeout)
        else:
            return await asyncio.wait_for(coro_or_fn, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("⏰ Task [%s] Timed Out (>%ds)", task_name, timeout)
        return fallback
    except (RuntimeError, TimeoutError, AttributeError) as e:
        record_degradation('resilience', e)
        logger.error("💥 Task [%s] Failed: %s", task_name, e)
        return fallback