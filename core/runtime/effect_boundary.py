from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Tuple

from core.governance_context import normalize_governance_domain, require_governance


@dataclass(frozen=True)
class EffectSinkSpec:
    sink_id: str
    fn_qualname: str
    allowed_domains: Tuple[str, ...]


_REGISTERED_SINKS: Dict[str, EffectSinkSpec] = {}


def _register_sink(sink_id: str, fn: Callable[..., Any], allowed_domains: Iterable[str]) -> None:
    normalized_domains = tuple(
        sorted(
            {
                normalize_governance_domain(domain)
                for domain in allowed_domains
                if domain is not None
            }
        )
    )
    _REGISTERED_SINKS[sink_id] = EffectSinkSpec(
        sink_id=sink_id,
        fn_qualname=getattr(fn, "__qualname__", getattr(fn, "__name__", sink_id)),
        allowed_domains=normalized_domains,
    )


def get_registered_effect_sinks() -> Dict[str, EffectSinkSpec]:
    return dict(_REGISTERED_SINKS)


def _record_sink_commit(sink_id: str, token: Any) -> None:
    try:
        from core.unified_action_log import get_action_log

        get_action_log().record(
            action=sink_id,
            source=str(getattr(token, "source", "unknown") or "unknown"),
            generation="effect_kernel",
            gate_status="committed",
            outcome="receipt_verified",
            metadata={
                "receipt_id": getattr(token, "receipt_id", ""),
                "domain": getattr(token, "domain", ""),
            },
        )
    except Exception:
        pass  # no-op: intentional


def effect_sink(
    sink_id: str,
    *,
    allowed_domains: Iterable[str] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a consequential sink and fail closed when called ungoverned."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _register_sink(sink_id, fn, allowed_domains)
        normalized_domains = tuple(
            sorted(
                {
                    normalize_governance_domain(domain)
                    for domain in allowed_domains
                    if domain is not None
                }
            )
        )

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = require_governance(
                    f"sink:{sink_id}",
                    strict=True,
                    allowed_domains=normalized_domains,
                )
                _record_sink_commit(sink_id, token)
                return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            token = require_governance(
                f"sink:{sink_id}",
                strict=True,
                allowed_domains=normalized_domains,
            )
            _record_sink_commit(sink_id, token)
            return fn(*args, **kwargs)

        return sync_wrapper

    return decorator
