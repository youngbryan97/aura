import asyncio
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional, Dict
from core.exceptions import CircuitOpenError
from .circuit_breaker_state import CircuitState

logger = logging.getLogger("Aura.Resilience.CircuitBreaker")

try:
    from prometheus_client import Gauge, Counter
    CIRCUIT_STATE = Gauge('aura_circuit_breaker_state', 'State of circuit breaker (0=OPEN, 1=HALF_OPEN, 2=CLOSED)', ['name'])
    CIRCUIT_FAILURES = Counter('aura_circuit_breaker_failures_total', 'Total failures for circuit breaker', ['name'])
    CIRCUIT_CALLS = Counter('aura_circuit_breaker_calls_total', 'Total calls for circuit breaker', ['name'])
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class CircuitBreaker:
    """Manages the state of a circuit to prevent cascading failures.
    
    Implements a state machine (CLOSED, OPEN, HALF_OPEN) with exponential backoff.
    Canonical version replacing redundant implementations in core/ and core/utils/.
    """
    def __init__(self, 
                 name: str = "Default",
                 failure_threshold: int = 5, 
                 base_recovery_timeout: float = 30.0, 
                 max_recovery_timeout: float = 300):
        self.name = name
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = base_recovery_timeout
        self.max_recovery_timeout = max_recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.total_failure_streak = 0
        self.last_failure_time = 0.0
        self.total_calls = 0
        
        self._lock = asyncio.Lock()
        self._update_metrics()

    def _update_metrics(self):
        if not PROMETHEUS_AVAILABLE:
            return
        state_map = {CircuitState.OPEN: 0, CircuitState.HALF_OPEN: 1, CircuitState.CLOSED: 2}
        CIRCUIT_STATE.labels(name=self.name).set(state_map.get(self.state, 2))

    @property
    def current_recovery_timeout(self) -> float:
        """Calculates recovery timeout with exponential backoff."""
        if self.total_failure_streak <= 1:
            return float(self.base_recovery_timeout)
        backoff = min(self.max_recovery_timeout, self.base_recovery_timeout * (2 ** (self.total_failure_streak - 1)))
        return float(backoff)

    def get_cycle_delay(self) -> float:
        """Returns the appropriate delay based on circuit state (legacy support)."""
        if self.state == CircuitState.OPEN:
            return 5.0
        elif self.state == CircuitState.HALF_OPEN:
            return 2.0
        return 0.1 # Normal operation delay

    async def __aenter__(self):
        """Context manager support."""
        async with self._lock:
            self.total_calls += 1
            if PROMETHEUS_AVAILABLE:
                CIRCUIT_CALLS.labels(name=self.name).inc()

            if self.state == CircuitState.OPEN:
                elapsed = time.monotonic() - self.last_failure_time
                if elapsed > self.current_recovery_timeout:
                    logger.info(f"⚡ Circuit '{self.name}' trial (Streak: {self.total_failure_streak}, Cooldown: {self.current_recovery_timeout}s).")
                    self.state = CircuitState.HALF_OPEN
                    self._update_metrics()
                else:
                    raise CircuitOpenError(f"Circuit '{self.name}' is OPEN. Cooldown: {self.current_recovery_timeout - elapsed:.1f}s remaining.")
            return self

    async def __aexit__(self, exc_type, exc_val, traceback):
        """Context manager support."""
        async with self._lock:
            if exc_type is not None:
                self._record_failure()
            else:
                self._record_success()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Wrap a function call (legacy and functional support)."""
        async with self:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

    def _record_success(self):
        """Resets the circuit upon a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"✅ Circuit '{self.name}' recovered (state: CLOSED).")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.total_failure_streak = 0
        self._update_metrics()

    def _record_failure(self):
        """Records a failure and trips the circuit if the threshold is met."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        
        if PROMETHEUS_AVAILABLE:
            CIRCUIT_FAILURES.labels(name=self.name).inc()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.total_failure_streak += 1
            logger.warning(f"🚨 Circuit '{self.name}' trial failed (Streak: {self.total_failure_streak}). Re-opening.")
        elif self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.total_failure_streak += 1
                logger.critical(f"🚨 Circuit '{self.name}' tripped (Streak: {self.total_failure_streak})!")
        self._update_metrics()

    # --- Legacy & Compatibility Methods ---
    def record_success(self): self._record_success()
    def record_error(self, error: Any = None): self._record_failure()
    def _on_success(self): self._record_success()
    def _on_failure(self, error: Any = None): self._record_failure()
    def should_skip_non_essential(self) -> bool: return self.state != CircuitState.CLOSED
    
    def wrap_async(self, func: Callable):
        """Decorator support for legacy utils."""
        @wraps(func)
        async def _wrapped(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return _wrapped

# Singleton support for global access
_breaker_instance = None
def get_circuit_breaker() -> CircuitBreaker:
    global _breaker_instance
    if _breaker_instance is None:
        _breaker_instance = CircuitBreaker(failure_threshold=5, base_recovery_timeout=30)
    return _breaker_instance