"""core/utils/exceptions.py — Unified Exception Hierarchy for Aura
==================================================================
Standardizes error handling across the entire engine.
"""
from core.runtime.errors import record_degradation
import logging
import functools
from typing import Any, Optional

logger = logging.getLogger("Aura.Exceptions")


def _record_exception_degraded_event(error: Exception, context: Optional[dict] = None) -> None:
    try:
        from core.health.degraded_events import record_degraded_event

        ctx = context if isinstance(context, dict) else {}
        record_degraded_event(
            ctx.get("module") or ctx.get("context") or "exceptions",
            type(error).__name__,
            detail=str(error),
            severity="error",
            classification=str(ctx.get("classification") or "background_degraded"),
            context=ctx,
            exc=error,
        )
    except Exception as degraded_exc:
        record_degradation('exceptions', degraded_exc)
        logger.debug("Degraded event capture failed: %s", degraded_exc)

class AuraError(Exception):
    """Base class for all Aura-related exceptions."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class CognitiveError(AuraError):
    """Errors occurring during thought, belief revision, or context assembly."""
    pass  # no-op: intentional

class AgencyError(AuraError):
    """Errors occurring during tool execution or skill management."""
    pass  # no-op: intentional

class PersistenceError(AuraError):
    """Errors occurring during state saving or database operations."""
    pass  # no-op: intentional

class SecurityError(AuraError):
    """Errors occurring during safety checks or identity guarding."""
    pass  # no-op: intentional

class InfrastructureError(AuraError):
    """Errors occurring in hardware interfaces, audio, or network."""
    pass  # no-op: intentional

def capture_and_log(error_or_func=None, context=None):
    """
    Utility to capture and log errors. 
    Can be used as a decorator or a direct function call.
    """
    
    if callable(error_or_func) and not isinstance(error_or_func, Exception):
        # Decorator usage
        func = error_or_func
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(func):
                    async def async_run():
                        try:
                            return await func(*args, **kwargs)
                        except Exception as e:
                            record_degradation('exceptions', e)
                            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                            _record_exception_degraded_event(
                                e,
                                {
                                    "module": getattr(func, "__module__", "exceptions"),
                                    "context": getattr(func, "__qualname__", func.__name__),
                                },
                            )
                            return None
                    return async_run()
                return func(*args, **kwargs)
            except Exception as e:
                record_degradation('exceptions', e)
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                _record_exception_degraded_event(
                    e,
                    {
                        "module": getattr(func, "__module__", "exceptions"),
                        "context": getattr(func, "__qualname__", func.__name__),
                    },
                )
                return None
        return wrapper
    
    # Direct call usage
    if isinstance(error_or_func, Exception):
        logger.error(f"Captured Error: {error_or_func} | Context: {context}", exc_info=True)
        _record_exception_degraded_event(error_or_func, context if isinstance(context, dict) else {})
    elif error_or_func is not None:
         logger.error(f"Captured Issue: {error_or_func} | Context: {context}")
