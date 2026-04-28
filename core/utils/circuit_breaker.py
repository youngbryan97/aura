"""Shim for core.resilience.circuit_breaker. Legacy location."""
from core.resilience.circuit_breaker import CircuitBreaker as _CB

class CircuitOpen(Exception):
    pass  # no-op: intentional

class CircuitBreaker(_CB):
    """Legacy wrapper for canonical CircuitBreaker."""
    async def call(self, coro, *args, **kwargs):
        try:
            return await super().call(coro, *args, **kwargs)
        except ConnectionAbortedError:
            raise CircuitOpen("circuit is open")
        except Exception:
            raise
