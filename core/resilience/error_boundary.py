from __future__ import annotations

import asyncio
import functools
import logging
import inspect
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger("Aura.ErrorBoundary")


from .circuit_breaker_state import CircuitState


class CircuitBreaker:
    """
    Prevents cascading failures by "tripping" when a threshold of errors is hit.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_s: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            logger.info("CircuitBreaker [%s]: recovered (CLOSED).", self.name)
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self, error: Exception) -> bool:
        """Returns True if the circuit tripped as a result."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.error("CircuitBreaker [%s]: tripped (OPEN) after %d failures. Error: %s",
                             self.name, self.failure_count, error)
            self.state = CircuitState.OPEN
            return True
        return False

    def can_proceed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                logger.info("CircuitBreaker [%s]: testing recovery (HALF_OPEN).", self.name)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Only allow one test call through
            return False

        return True


class CircuitRegistry:
    """Central registry of all circuit breakers in the system."""
    _instance: Optional[CircuitRegistry] = None

    def __init__(self):
        self.circuits: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_instance(cls) -> CircuitRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_breaker(self, name: str) -> CircuitBreaker:
        if name not in self.circuits:
            self.circuits[name] = CircuitBreaker(name)
        return self.circuits[name]

    def get_all_status(self) -> Dict[str, str]:
        return {name: c.state.value for name, c in self.circuits.items()}


def _is_user_facing_response_phase(phase_name: str, state: Any) -> bool:
    if phase_name not in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
        return False
    origin = str(
        getattr(getattr(state, "cognition", None), "current_origin", "") or ""
    ).strip().lower().replace("-", "_")
    if not origin:
        return False
    if origin in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}:
        return True
    tokens = {token for token in origin.split("_") if token}
    return bool(tokens & {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"})


# ── Boundary Decorators ───────────────────────────────────────────────────────

def error_boundary(
    name: Optional[str] = None,
    fallback_value: Any = None,
    reraise: bool = False,
):
    """
    Decorator for critical functions (especially LLM or external calls).
    Wraps execution in a circuit breaker and catches exceptions.
    """

    def decorator(func: Callable):
        breaker_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = CircuitRegistry.get_instance().get_breaker(breaker_name)

            if not breaker.can_proceed():
                logger.warning("Circuit [%s] is OPEN. Returning fallback.", breaker_name)
                return fallback_value

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                logger.warning("Error in boundary [%s]: %s", breaker_name, e)
                if reraise:
                    raise
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = CircuitRegistry.get_instance().get_breaker(breaker_name)

            if not breaker.can_proceed():
                return fallback_value

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                logger.warning("Error in boundary [%s]: %s", breaker_name, e)
                if reraise:
                    raise
                return fallback_value

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ── Phase Wrapper ─────────────────────────────────────────────────────────────

async def wrap_phase(
    phase_name: str,
    phase_fn: Callable,
    state: Any,
    objective: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Wraps a kernel phase execution with circuit-breaker protection.
    If the phase fails, returns state unchanged — the kernel keeps running.

    FIX: was calling phase_fn() with no arguments, meaning state was never
    passed into execute(). Every phase ran against an empty call signature.
    """
    breaker = CircuitRegistry.get_instance().get_breaker(f"phase:{phase_name}")

    if not breaker.can_proceed():
        if _is_user_facing_response_phase(phase_name, state):
            breaker.state = CircuitState.HALF_OPEN
            logger.warning(
                "Phase [%s] circuit is OPEN. Allowing a user-facing recovery probe.",
                phase_name,
            )
        else:
            logger.warning(
                "Phase [%s] circuit is OPEN. Skipping.", phase_name
            )
            return state

    try:
        # FIX: pass state (and optional objective + kwargs) into the phase
        if asyncio.iscoroutinefunction(phase_fn):
            result = await phase_fn(state, objective=objective, **kwargs)
        else:
            result = phase_fn(state, objective=objective, **kwargs)

        # Awaitable guard (in case phase_fn returns a coroutine object)
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            result = await result

        breaker.record_success()
        return result if result is not None else state

    except Exception as e:
        breaker.record_failure(e)
        logger.error(
            "ERROR BOUNDARY: Phase [%s] failed. State preserved.\n%s",
            phase_name,
            traceback.format_exc(),
        )
        if hasattr(state, "world") and hasattr(state.world, "recent_percepts"):
            state.world.recent_percepts.append({
                "type": "internal_error",
                "severity": "critical",
                "payload": {"phase": phase_name, "error": str(e)},
            })
        return state


# ── Convenience getters ───────────────────────────────────────────────────────

def get_circuit_registry() -> CircuitRegistry:
    return CircuitRegistry.get_instance()
