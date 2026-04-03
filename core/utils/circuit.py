import pybreaker
from functools import wraps
import asyncio
import logging

logger = logging.getLogger("Aura.Circuit")

def circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: int = 30):
    """
    Wraps asynchronous network or heavy peripheral calls in a Circuit Breaker.
    If the decorated function fails `failure_threshold` times consecutively,
    the circuit opens and subsequent calls immediately fail for `recovery_timeout` seconds.
    """
    breaker = pybreaker.CircuitBreaker(
        fail_max=failure_threshold,
        reset_timeout=recovery_timeout,
        name=name
    )
    
    # Optional listeners for logging
    class BreakerLogListener(pybreaker.CircuitBreakerListener):
        def state_change(self, cb, old_state, new_state):
            msg = f"Circuit '{cb.name}' changed state from {old_state.name} to {new_state.name}"
            if new_state.name == "OPEN":
                logger.error("🚨 %s (Shield activated to prevent cascading failure)", msg)
            elif new_state.name == "HALF_OPEN":
                logger.warning("🛡️ %s (Testing waters...)", msg)
            else:
                logger.info("✅ %s (Shield lowered, normal operation resumed)", msg)
                
    breaker.add_listeners(BreakerLogListener())

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call_async(func, *args, **kwargs)
            except pybreaker.CircuitBreakerError:
                raise Exception(f"Circuit '{name}' is OPEN. Rejecting call to prevent saturation.")
        return wrapper
    return decorator
