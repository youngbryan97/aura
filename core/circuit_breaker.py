"""Shim for core.resilience.circuit_breaker. Legacy location."""
from core.resilience.circuit_breaker import CircuitBreaker as _CB, CircuitState

class State:
    CLOSED = CircuitState.CLOSED
    OPEN = CircuitState.OPEN
    HALF_OPEN = CircuitState.HALF_OPEN

class CircuitBreaker(_CB):
    """Legacy wrapper for canonical CircuitBreaker."""
    pass  # no-op: intentional

class CircuitManager:
    """Registry for circuit breakers (Legacy)."""
    _circuits = {}

    @classmethod
    def get_circuit(cls, key: str, **kwargs) -> CircuitBreaker:
        if key not in cls._circuits:
            cls._circuits[key] = CircuitBreaker(name=key, **kwargs)
        return cls._circuits[key]
