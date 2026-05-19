"""core/adapters/typed_result.py
================================
Typed result wrapper for adapter operations.

PROBLEM: Many adapters return `[]` on failure, which causes the agent to
interpret "error" as "nothing happened in the world". This is a critical
lie that breaks the agent's world model.

SOLUTION: Every adapter operation returns a WorldResult that distinguishes
between "success with empty data" and "failure with error info".

Usage:
    from core.adapters.typed_result import WorldResult, AdapterError

    class WebSearchAdapter:
        def search(self, query: str) -> WorldResult:
            try:
                results = self._do_search(query)
                return WorldResult.ok(results)
            except TimeoutError as e:
                return WorldResult.fail("timeout", str(e))
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                return WorldResult.fail("adapter_failure", str(e))
"""
from __future__ import annotations
import inspect

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar("T")


class AdapterErrorKind(str, Enum):
    """Categorized adapter failure modes."""
    TIMEOUT = "timeout"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    PARSE_ERROR = "parse_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AdapterError:
    """Structured error from an adapter operation."""
    kind: AdapterErrorKind
    message: str
    retryable: bool = True
    retry_after_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.message}"


@dataclass
class WorldResult:
    """Typed result from an adapter/world interaction.

    NEVER returns `[]` for errors. Instead:
    - success=True, data=[]  → "We looked, nothing was there"
    - success=False, error_info=AdapterError  → "We couldn't look at all"

    This distinction is CRITICAL for the agent's world model.
    """
    success: bool
    data: Any = None
    error_info: Optional[AdapterError] = None
    timestamp: float = field(default_factory=time.time)
    adapter_name: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def ok(
        data: Any = None,
        *,
        adapter_name: str = "",
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldResult:
        """Create a successful result."""
        return WorldResult(
            success=True,
            data=data,
            adapter_name=adapter_name,
            latency_ms=latency_ms,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def fail(
        kind: Union[str, AdapterErrorKind],
        message: str,
        *,
        retryable: bool = True,
        retry_after_s: float = 0.0,
        adapter_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldResult:
        """Create an error result."""
        if isinstance(kind, str):
            try:
                error_kind = AdapterErrorKind(kind)
            except ValueError:
                error_kind = AdapterErrorKind.UNKNOWN
        else:
            error_kind = kind

        return WorldResult(
            success=False,
            error_info=AdapterError(
                kind=error_kind,
                message=message,
                retryable=retryable,
                retry_after_s=retry_after_s,
                metadata=dict(metadata or {}),
            ),
            adapter_name=adapter_name,
        )

    @property
    def failed(self) -> bool:
        return not self.success

    @property
    def is_empty(self) -> bool:
        """True only if successful but data is empty/None."""
        if not self.success:
            return False  # Failures are NOT empty, they're failures
        if self.data is None:
            return True
        if isinstance(self.data, (list, tuple, dict, set)):
            return len(self.data) == 0
        if isinstance(self.data, str):
            return not self.data.strip()
        return False

    def unwrap(self, default: Any = None) -> Any:
        """Get data, returning default on failure."""
        if self.success:
            return self.data
        return default

    def unwrap_or_raise(self) -> Any:
        """Get data or raise on failure."""
        if self.success:
            return self.data
        raise RuntimeError(
            f"Adapter {self.adapter_name} failed: {self.error_info}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/telemetry."""
        result: Dict[str, Any] = {
            "success": self.success,
            "adapter_name": self.adapter_name,
            "latency_ms": round(self.latency_ms, 1),
            "timestamp": self.timestamp,
        }
        if self.error_info:
            result["error"] = {
                "kind": self.error_info.kind.value,
                "message": self.error_info.message[:200],
                "retryable": self.error_info.retryable,
            }
        if self.success and self.data is not None:
            if isinstance(self.data, (list, tuple)):
                result["data_count"] = len(self.data)
            elif isinstance(self.data, dict):
                result["data_keys"] = list(self.data.keys())[:10]
        return result


def wrap_adapter_call(func):
    """Decorator to wrap legacy adapter methods that return [] on failure.

    Catches exceptions and returns WorldResult instead.
    """
    import asyncio
    import functools

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                result = await func(*args, **kwargs)
                return WorldResult.ok(
                    result,
                    adapter_name=func.__qualname__,
                    latency_ms=(time.time() - t0) * 1000,
                )
            except TimeoutError as e:
                return WorldResult.fail(
                    "timeout", str(e),
                    adapter_name=func.__qualname__,
                )
            except ConnectionError as e:
                return WorldResult.fail(
                    "network_error", str(e),
                    adapter_name=func.__qualname__,
                )
            except PermissionError as e:
                return WorldResult.fail(
                    "permission_denied", str(e),
                    retryable=False,
                    adapter_name=func.__qualname__,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                return WorldResult.fail(
                    "internal_error", str(e),
                    adapter_name=func.__qualname__,
                )
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                return WorldResult.ok(
                    result,
                    adapter_name=func.__qualname__,
                    latency_ms=(time.time() - t0) * 1000,
                )
            except TimeoutError as e:
                return WorldResult.fail(
                    "timeout", str(e),
                    adapter_name=func.__qualname__,
                )
            except ConnectionError as e:
                return WorldResult.fail(
                    "network_error", str(e),
                    adapter_name=func.__qualname__,
                )
            except PermissionError as e:
                return WorldResult.fail(
                    "permission_denied", str(e),
                    retryable=False,
                    adapter_name=func.__qualname__,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                return WorldResult.fail(
                    "internal_error", str(e),
                    adapter_name=func.__qualname__,
                )
        return sync_wrapper
