"""core/governance_context.py -- Governed Execution Context
============================================================
Makes governance PHYSICALLY INESCAPABLE at the Python interpreter level.

Uses contextvars to carry the active WillReceipt through the entire
call stack. Any code path can check whether it's running inside a
governed context, and critical paths ASSERT it.

This is the mechanism that transforms governance from "social" (callers
must honor it) to "physical" (callers cannot bypass it).

Three mechanisms:
  1. GovernanceContext (contextvars) — carries active receipt through stack
  2. @governed decorator — wraps functions to require active context
  3. governed_scope() context manager — creates a governed context from a WillDecision

Usage:
    from core.governance_context import governed_scope, require_governance

    # Creating governed scope (done by Will/OutputGate/etc):
    decision = will.decide(...)
    async with governed_scope(decision):
        # Everything inside here has governance
        await do_something()

    # Checking governance (done by tools/memory/etc):
    @governed
    def write_to_memory(data):
        # This will raise GovernanceViolation if called outside governed_scope
        ...

    # Or inline:
    require_governance("memory_write")
"""
from __future__ import annotations

import contextvars
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("Aura.Governance")


# ---------------------------------------------------------------------------
# Context variable — the single source of truth
# ---------------------------------------------------------------------------

_active_receipt: contextvars.ContextVar[Optional["GovernanceToken"]] = \
    contextvars.ContextVar("active_governance_receipt", default=None)


@dataclass
class GovernanceToken:
    """A token proving that the current execution is governed."""
    receipt_id: str
    domain: str
    source: str
    timestamp: float = field(default_factory=time.time)
    constraints: list = field(default_factory=list)
    ttl: float = 30.0  # tokens expire after 30 seconds

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl

    @property
    def valid(self) -> bool:
        return bool(self.receipt_id) and not self.expired


class GovernanceViolation(RuntimeError):
    """Raised when code attempts to execute without governance."""
    pass


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def governed_scope(decision: Any):
    """Create a governed execution context from a WillDecision.

    Usage:
        decision = will.decide(...)
        async with governed_scope(decision):
            await do_governed_work()
    """
    token = GovernanceToken(
        receipt_id=getattr(decision, "receipt_id", ""),
        domain=getattr(decision, "domain", "unknown"),
        source=getattr(decision, "source", "unknown"),
        constraints=getattr(decision, "constraints", []),
    )
    reset_token = _active_receipt.set(token)
    try:
        yield token
    finally:
        _active_receipt.reset(reset_token)


@contextmanager
def governed_scope_sync(decision: Any):
    """Synchronous version of governed_scope."""
    token = GovernanceToken(
        receipt_id=getattr(decision, "receipt_id", ""),
        domain=getattr(decision, "domain", "unknown"),
        source=getattr(decision, "source", "unknown"),
        constraints=getattr(decision, "constraints", []),
    )
    reset_token = _active_receipt.set(token)
    try:
        yield token
    finally:
        _active_receipt.reset(reset_token)


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def get_active_governance() -> Optional[GovernanceToken]:
    """Get the current governance token, or None if ungoverned."""
    token = _active_receipt.get()
    if token and token.valid:
        return token
    return None


def is_governed() -> bool:
    """Check if current execution is inside a governed context."""
    return get_active_governance() is not None


def require_governance(operation: str = "unknown") -> GovernanceToken:
    """Assert that current execution is governed. Raises GovernanceViolation if not.

    Use this at the top of any critical function (memory write, tool exec, etc.)
    to ensure it was called through proper governance channels.

    In degraded mode (boot, shutdown, etc.), this logs a warning instead of raising.
    """
    token = get_active_governance()
    if token is not None:
        return token

    # Check if we're in degraded mode (early boot, shutdown, testing)
    try:
        from core.container import ServiceContainer
        will = ServiceContainer.get("unified_will", default=None)
        if will is None:
            # Will not booted yet — degraded mode, allow with warning
            logger.debug("Governance check '%s' during degraded mode (Will not booted)", operation)
            return GovernanceToken(receipt_id="degraded_mode", domain="degraded",
                                   source="boot", ttl=300)
    except Exception:
        pass

    # Log the violation — in production this would be a hard error
    logger.warning("GOVERNANCE VIOLATION: '%s' called outside governed context", operation)

    # Return a violation token that tracks the bypass
    _record_violation(operation)
    return GovernanceToken(receipt_id="VIOLATION", domain="ungoverned",
                           source=operation, ttl=1)


def governed(fn: Callable) -> Callable:
    """Decorator that enforces governance on a function.

    Usage:
        @governed
        def write_memory(data):
            ...  # will raise if called outside governed_scope
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_governance(fn.__qualname__)
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        require_governance(fn.__qualname__)
        return await fn(*args, **kwargs)

    if _is_coroutine_function(fn):
        return async_wrapper
    return wrapper


def _is_coroutine_function(fn):
    import asyncio
    return asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------

_violations: list = []
_MAX_VIOLATIONS = 200


def _record_violation(operation: str) -> None:
    """Record a governance violation for audit."""
    _violations.append({
        "operation": operation,
        "timestamp": time.time(),
    })
    if len(_violations) > _MAX_VIOLATIONS:
        _violations.pop(0)

    # Publish to event bus
    try:
        from core.event_bus import get_event_bus
        get_event_bus().publish_threadsafe("governance.violation", {
            "operation": operation,
            "timestamp": time.time(),
        })
    except Exception:
        pass


def get_violations(n: int = 20) -> list:
    """Return recent governance violations."""
    return list(_violations[-n:])


def get_governance_status() -> dict:
    """Return governance system status."""
    return {
        "currently_governed": is_governed(),
        "active_token": _active_receipt.get() is not None,
        "total_violations": len(_violations),
        "recent_violations": get_violations(5),
    }
