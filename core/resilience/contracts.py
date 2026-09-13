"""core/resilience/contracts.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design-by-Contract (DbC) runtime assertion framework.

Provides decorators for preconditions, postconditions, and invariants.
Inspired by Eiffel's DbC and production-grade formal verification practices.

Mode: log+continue (recommended for production).
- Violations are logged and recorded in the fault taxonomy
- Function execution continues (no crash)
- Set AURA_CONTRACTS_ENFORCE=1 to fail-closed (fail-closed mode)

Usage:
    from core.resilience.contracts import precondition, postcondition, invariant

    @precondition(lambda self: self.state != "SHUTDOWN",
                  message="Cannot operate during shutdown")
    @postcondition(lambda result: result is not None,
                   message="Must return a non-None result")
    async def process_request(self, request):
        ...

    @invariant(lambda self: len(self.queue) <= self.max_size,
               message="Queue size invariant violated")
    class BoundedQueue:
        ...
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger("Aura.Contracts")

T = TypeVar("T")

def _resolve_enforce() -> bool:
    # Canonical definition lives in core.runtime.mode (the single strictness
    # resolver); fall back to the raw env read if mode is unavailable at import.
    try:
        from core.runtime.mode import contracts_enforced

        return contracts_enforced()
    except (ImportError, AttributeError, RuntimeError):
        return os.environ.get("AURA_CONTRACTS_ENFORCE", "0") == "1"


_ENFORCE = _resolve_enforce()

# Guarded-callable failure envelope: user-supplied callables (checks, guards,
# actions, hooks, channels) may raise anything. House discipline forbids broad
# `except Exception`; this tuple names the realistic failure universe
# explicitly. Exotic escapes — custom Exception subtypes outside these bases,
# SystemExit, KeyboardInterrupt — propagate loudly by design.
_GUARDED_CALLABLE_ERRORS = (
    RuntimeError, AttributeError, TypeError, ValueError,
    LookupError, ArithmeticError, OSError, ImportError,
)


class ContractViolation(RuntimeError):
    """Raised when a contract is violated and enforcement is enabled."""

    def __init__(self, kind: str, message: str, function: str) -> None:
        self.kind = kind
        self.message = message
        self.function = function
        super().__init__(f"{kind} violation in {function}: {message}")


@dataclass
class ViolationRecord:
    """Record of a contract violation."""
    kind: str  # "precondition", "postcondition", "invariant"
    function: str
    message: str
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "function": self.function,
            "message": self.message[:200],
            "timestamp": self.timestamp,
        }


class ContractTracker:
    """Tracks all contract violations for monitoring and dashboards."""

    def __init__(self, max_records: int = 500) -> None:
        self._lock = threading.Lock()
        self._violations: deque[ViolationRecord] = deque(maxlen=max_records)
        self._counts: dict[str, int] = {"precondition": 0, "postcondition": 0, "invariant": 0}

    def record(self, violation: ViolationRecord) -> None:
        with self._lock:
            self._violations.append(violation)
            self._counts[violation.kind] = self._counts.get(violation.kind, 0) + 1

    def recent(self, limit: int = 20) -> list[ViolationRecord]:
        with self._lock:
            items = list(self._violations)
            return items[-limit:]

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind:
                return self._counts.get(kind, 0)
            return sum(self._counts.values())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_violations": sum(self._counts.values()),
                "by_kind": dict(self._counts),
                "enforcement_mode": "fail_closed" if _ENFORCE else "log_continue",
                "recent": [v.to_dict() for v in list(self._violations)[-5:]],
            }


_tracker = ContractTracker()


def get_contract_tracker() -> ContractTracker:
    return _tracker


def _handle_violation(kind: str, message: str, func_name: str) -> None:
    """Central violation handler."""
    violation = ViolationRecord(kind=kind, function=func_name, message=message)
    _tracker.record(violation)

    logger.warning(
        "CONTRACT VIOLATION [%s] in %s: %s (mode=%s)",
        kind.upper(), func_name, message,
        "ENFORCE" if _ENFORCE else "LOG",
    )

    # Record in fault taxonomy
    try:
        from core.resilience.fault_taxonomy import get_fault_registry
        get_fault_registry().record_fault(
            "F20", subsystem=f"contract.{func_name}",
            details=f"{kind}: {message}",
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Fault registry unavailable for contract violation: %s", exc)

    if _ENFORCE:
        raise ContractViolation(kind, message, func_name)


def precondition(
    check: Callable[..., bool],
    message: str = "Precondition failed",
) -> Callable:
    """Decorator that checks a precondition before function execution.

    The check callable receives the same arguments as the decorated function.
    For methods, `self` is the first argument.
    """
    def decorator(func: Callable) -> Callable:
        func_name = getattr(func, "__qualname__", func.__name__)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    ok = check(*args, **kwargs) if args else check(**kwargs)
                except _GUARDED_CALLABLE_ERRORS:
                    ok = False
                if not ok:
                    _handle_violation("precondition", message, func_name)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    ok = check(*args, **kwargs) if args else check(**kwargs)
                except _GUARDED_CALLABLE_ERRORS:
                    ok = False
                if not ok:
                    _handle_violation("precondition", message, func_name)
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def postcondition(
    check: Callable[[Any], bool],
    message: str = "Postcondition failed",
) -> Callable:
    """Decorator that checks a postcondition on the return value.

    The check callable receives the return value as its only argument.
    """
    def decorator(func: Callable) -> Callable:
        func_name = getattr(func, "__qualname__", func.__name__)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await func(*args, **kwargs)
                try:
                    ok = check(result)
                except _GUARDED_CALLABLE_ERRORS:
                    ok = False
                if not ok:
                    _handle_violation("postcondition", message, func_name)
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                try:
                    ok = check(result)
                except _GUARDED_CALLABLE_ERRORS:
                    ok = False
                if not ok:
                    _handle_violation("postcondition", message, func_name)
                return result
            return sync_wrapper
    return decorator


def invariant(
    check: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable:
    """Class decorator that checks an invariant before and after every
    public method call.

    The check callable receives `self` as its only argument.
    """
    def decorator(cls: type) -> type:
        for attr_name in list(vars(cls)):
            if attr_name.startswith("_"):
                continue
            # Only wrap plain instance methods. getattr() unwraps descriptors,
            # so a classmethod/staticmethod would look like a bare callable and
            # get rebound with `self` as its first argument; nested classes are
            # callable too. All of those must be left untouched.
            static_attr = inspect.getattr_static(cls, attr_name)
            if isinstance(static_attr, (staticmethod, classmethod)):
                continue
            attr = getattr(cls, attr_name)
            if not (inspect.isfunction(attr) or inspect.iscoroutinefunction(attr)):
                continue

            func_name = f"{cls.__name__}.{attr_name}"

            if inspect.iscoroutinefunction(attr):
                @functools.wraps(attr)
                async def async_method(self: Any, *args: Any,
                                       _orig: Any = attr,
                                       _fn: str = func_name,
                                       **kwargs: Any) -> Any:
                    try:
                        ok = check(self)
                    except _GUARDED_CALLABLE_ERRORS:
                        ok = False
                    if not ok:
                        _handle_violation("invariant", f"pre-{message}", _fn)
                    result = await _orig(self, *args, **kwargs)
                    try:
                        ok = check(self)
                    except _GUARDED_CALLABLE_ERRORS:
                        ok = False
                    if not ok:
                        _handle_violation("invariant", f"post-{message}", _fn)
                    return result
                setattr(cls, attr_name, async_method)
            else:
                @functools.wraps(attr)
                def sync_method(self: Any, *args: Any,
                                _orig: Any = attr,
                                _fn: str = func_name,
                                **kwargs: Any) -> Any:
                    try:
                        ok = check(self)
                    except _GUARDED_CALLABLE_ERRORS:
                        ok = False
                    if not ok:
                        _handle_violation("invariant", f"pre-{message}", _fn)
                    result = _orig(self, *args, **kwargs)
                    try:
                        ok = check(self)
                    except _GUARDED_CALLABLE_ERRORS:
                        ok = False
                    if not ok:
                        _handle_violation("invariant", f"post-{message}", _fn)
                    return result
                setattr(cls, attr_name, sync_method)

        return cls
    return decorator
