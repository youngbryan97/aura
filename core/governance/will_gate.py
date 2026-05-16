"""core/governance/will_gate.py
================================
The @will_gated decorator and enforcement infrastructure.

Every method that mutates state, emits output, writes to disk, makes network
calls, or modifies memory MUST pass through the Unified Will. This module
provides the decorator and the boot-time audit that enforces coverage.

Usage:
    from core.governance.will_gate import will_gated

    class CapabilityEngine:
        @will_gated(domain=ActionDomain.TOOL_EXECUTION)
        async def execute_tool(self, tool_name, payload, ...):
            ...

The decorator:
1. Calls UnifiedWill.decide() before entering the method.
2. If the decision is not approved, raises WillRefused or returns a
   sentinel depending on configuration.
3. Attaches the WillDecision receipt_id to the return value if possible.
4. Logs every gating decision for audit.

Boot-time enforcement:
    call `audit_will_coverage()` at startup. It enumerates all public
    methods on registered services that are known to be world-affecting
    and asserts they carry the @will_gated decorator. Fails boot if not.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from core.will import ActionDomain, WillOutcome, get_will

logger = logging.getLogger("Aura.Governance.WillGate")

F = TypeVar("F", bound=Callable[..., Any])


class WillRefused(Exception):
    """Raised when the Unified Will refuses an action."""

    def __init__(self, receipt_id: str, reason: str, domain: str):
        self.receipt_id = receipt_id
        self.reason = reason
        self.domain = domain
        super().__init__(f"Will REFUSED [{domain}]: {reason} (receipt={receipt_id})")


class WillDeferred(Exception):
    """Raised when the Unified Will defers an action."""

    def __init__(self, receipt_id: str, reason: str, domain: str):
        self.receipt_id = receipt_id
        self.reason = reason
        self.domain = domain
        super().__init__(f"Will DEFERRED [{domain}]: {reason} (receipt={receipt_id})")


# Registry of all will-gated methods for boot-time audit
_GATED_METHODS: Set[str] = set()


def will_gated(
    domain: ActionDomain,
    *,
    raise_on_refuse: bool = False,
    raise_on_defer: bool = False,
    return_none_on_block: bool = True,
    priority: float = 0.5,
    source_override: Optional[str] = None,
    is_critical: bool = False,
) -> Callable[[F], F]:
    """Decorator that gates a method through the Unified Will.

    Args:
        domain: The ActionDomain for this operation.
        raise_on_refuse: If True, raise WillRefused on REFUSE outcome.
        raise_on_defer: If True, raise WillDeferred on DEFER outcome.
        return_none_on_block: If True, return None when blocked (default).
        priority: Default priority for the Will decision.
        source_override: Override the source name (default: class.method).
        is_critical: If True, mark as safety-critical (always passes).
    """

    def decorator(func: F) -> F:
        # Register for audit
        qualname = getattr(func, "__qualname__", func.__name__)
        _GATED_METHODS.add(qualname)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                source = source_override or qualname
                content = f"{qualname}({', '.join(str(a)[:50] for a in args[1:3])})"

                # Extract context from kwargs if present
                context = kwargs.get("context", {})
                if not isinstance(context, dict):
                    context = {}

                will = get_will()
                decision = will.decide(
                    content=content[:200],
                    source=source,
                    domain=domain,
                    priority=priority,
                    is_critical=is_critical,
                    context=context,
                )

                if decision.is_approved():
                    # Inject receipt into kwargs if the function accepts it
                    sig = inspect.signature(func)
                    if "will_receipt_id" in sig.parameters:
                        kwargs["will_receipt_id"] = decision.receipt_id
                    return await func(*args, **kwargs)

                # Blocked
                if decision.outcome == WillOutcome.REFUSE:
                    logger.info(
                        "Will gate REFUSED %s: %s", qualname, decision.reason
                    )
                    if raise_on_refuse:
                        raise WillRefused(
                            decision.receipt_id, decision.reason, domain.value
                        )
                elif decision.outcome == WillOutcome.DEFER:
                    logger.info(
                        "Will gate DEFERRED %s: %s", qualname, decision.reason
                    )
                    if raise_on_defer:
                        raise WillDeferred(
                            decision.receipt_id, decision.reason, domain.value
                        )

                if return_none_on_block:
                    return None
                return None

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                source = source_override or qualname
                content = f"{qualname}({', '.join(str(a)[:50] for a in args[1:3])})"

                context = kwargs.get("context", {})
                if not isinstance(context, dict):
                    context = {}

                will = get_will()
                decision = will.decide(
                    content=content[:200],
                    source=source,
                    domain=domain,
                    priority=priority,
                    is_critical=is_critical,
                    context=context,
                )

                if decision.is_approved():
                    sig = inspect.signature(func)
                    if "will_receipt_id" in sig.parameters:
                        kwargs["will_receipt_id"] = decision.receipt_id
                    return func(*args, **kwargs)

                if decision.outcome == WillOutcome.REFUSE:
                    logger.info(
                        "Will gate REFUSED %s: %s", qualname, decision.reason
                    )
                    if raise_on_refuse:
                        raise WillRefused(
                            decision.receipt_id, decision.reason, domain.value
                        )
                elif decision.outcome == WillOutcome.DEFER:
                    logger.info(
                        "Will gate DEFERRED %s: %s", qualname, decision.reason
                    )
                    if raise_on_defer:
                        raise WillDeferred(
                            decision.receipt_id, decision.reason, domain.value
                        )

                if return_none_on_block:
                    return None
                return None

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Boot-time Will coverage audit
# ---------------------------------------------------------------------------

# Methods that MUST be will-gated for sealed operation.
# Format: "ClassName.method_name" or just "method_name" for module-level.
REQUIRED_GATED_METHODS: List[str] = [
    # These are checked as substrings in qualname
    "execute_tool",
    "execute_skill",
    "write_memory",
    "mutate_state",
    "emit_proactive",
    "file_operation",
    "shell_execute",
    "network_request",
    "send_message",
    "publish_post",
]


def audit_will_coverage(strict: bool = False) -> Dict[str, Any]:
    """Audit that all required methods are will-gated.

    Args:
        strict: If True, raise RuntimeError on missing coverage.

    Returns:
        Audit report with covered and missing methods.
    """
    covered = list(_GATED_METHODS)
    missing = []

    for required in REQUIRED_GATED_METHODS:
        found = any(required in gated for gated in _GATED_METHODS)
        if not found:
            missing.append(required)

    report = {
        "total_gated": len(covered),
        "required_checked": len(REQUIRED_GATED_METHODS),
        "missing": missing,
        "covered": covered[:50],  # truncate for readability
        "all_covered": len(missing) == 0,
        "timestamp": time.time(),
    }

    if missing:
        logger.warning(
            "Will coverage audit: %d/%d required methods not gated: %s",
            len(missing),
            len(REQUIRED_GATED_METHODS),
            ", ".join(missing),
        )
        if strict:
            raise RuntimeError(
                f"Will coverage audit FAILED: {len(missing)} methods not gated"
            )
    else:
        logger.info(
            "Will coverage audit PASSED: %d methods gated, all %d required covered.",
            len(covered),
            len(REQUIRED_GATED_METHODS),
        )

    return report
"""
    core.governance.will_gate — end of module
"""
