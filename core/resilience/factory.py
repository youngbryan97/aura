import asyncio
import logging
from functools import wraps
from typing import Dict, Callable
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger("Aura.Resilience.Factory")

# A global registry to hold one breaker per service_name.
_breakers: Dict[str, CircuitBreaker] = {}
_factory_lock = asyncio.Lock()

async def get_breaker(service_name: str, failure_threshold: int = 3, recovery_timeout: int = 60) -> CircuitBreaker:
    """Factory function to get or create a circuit breaker for a service."""
    async with _factory_lock:
        if service_name not in _breakers:
            logger.info(f"🌿 Establishing new resilience root for '{service_name}'.")
            _breakers[service_name] = CircuitBreaker(failure_threshold, recovery_timeout)
        return _breakers[service_name]

import inspect

def circuit_breaker(service_name: str, failure_threshold: int = 3, recovery_timeout: int = 60):
    """
    A decorator that wraps an async function or async generator with a circuit breaker.
    It will use the service_name to get a shared CircuitBreaker instance.
    """
    def decorator(func: Callable):
        if inspect.isasyncgenfunction(func):
            @wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                breaker = await get_breaker(service_name, failure_threshold, recovery_timeout)
                async with breaker:
                    async for item in func(*args, **kwargs):
                        yield item
            return async_gen_wrapper
        else:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                breaker = await get_breaker(service_name, failure_threshold, recovery_timeout)
                async with breaker:
                    return await func(*args, **kwargs)
            return wrapper
    return decorator
