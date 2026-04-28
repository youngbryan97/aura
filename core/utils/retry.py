"""
Standardized Retry Utility — Ported from gemini-cli patterns

Provides exponential backoff with jitter, error classification,
and telemetry integration. Replaces all ad-hoc retry loops.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("Aura.Retry")

T = TypeVar("T")

# ── Error Classification ────────────────────────────────────────────────────

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# httpx-specific exceptions (imported lazily to avoid hard dependency)
_HTTPX_RETRYABLE = None

def _get_httpx_retryable():
    global _HTTPX_RETRYABLE
    if _HTTPX_RETRYABLE is None:
        try:
            import httpx
            _HTTPX_RETRYABLE = (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.PoolTimeout,
            )
        except ImportError:
            _HTTPX_RETRYABLE = ()
    return _HTTPX_RETRYABLE


def is_retryable(error: Exception) -> bool:
    """Classify whether an error is retryable."""
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    httpx_retryable = _get_httpx_retryable()
    if httpx_retryable and isinstance(error, httpx_retryable):
        return True
    # HTTP 429 (rate limit) and 5xx are retryable
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status and (status == 429 or status >= 500):
        return True
    return False


def get_retry_error_type(error: Exception) -> str:
    """Get a human-readable error type for telemetry."""
    name = type(error).__name__
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status:
        return f"{name}({status})"
    return name


# ── Core Retry Function ─────────────────────────────────────────────────────

async def retry_with_backoff(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_filter: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with exponential backoff retry.

    Args:
        fn: Async function to call
        max_retries: Maximum number of retry attempts (total calls = max_retries + 1)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Add random jitter to prevent thundering herd
        retry_filter: Custom function to determine if error is retryable
        on_retry: Callback(attempt, error, delay_seconds) on each retry

    Returns:
        The result of fn(*args, **kwargs)

    Raises:
        The last exception if all retries are exhausted
    """
    last_error = None
    filter_fn = retry_filter or is_retryable

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            record_degradation('retry', e)
            last_error = e

            if attempt >= max_retries:
                break

            if not filter_fn(e):
                logger.debug("Non-retryable error: %s", e)
                break

            # Calculate backoff with optional jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            if on_retry:
                try:
                    on_retry(attempt + 1, e, delay)
                except Exception:
                    pass

            logger.info(
                "Retry %d/%d in %.1fs: %s",
                attempt + 1, max_retries, delay, get_retry_error_type(e)
            )
            await asyncio.sleep(delay)

    raise last_error


def retry_with_backoff_sync(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Synchronous version of retry_with_backoff."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            record_degradation('retry', e)
            last_error = e
            if attempt >= max_retries or not is_retryable(e):
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay *= (0.5 + random.random())
            logger.info("Sync retry %d/%d in %.1fs: %s", attempt + 1, max_retries, delay, type(e).__name__)
            time.sleep(delay)

    raise last_error
