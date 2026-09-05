"""core/governance_context.py -- Governed Execution Context
============================================================
Enforces governed execution boundaries. When active or strict mode is enabled,
governance is inescapable at the Python interpreter level.

Uses contextvars to carry the active WillReceipt through the entire
call stack. Any code path can check whether it's running inside a
governed context, and critical paths ASSERT it.

Three mechanisms:
  1. GovernanceContext (contextvars) — carries active receipt through stack
  2. @governed decorator — wraps functions to require active context
  3. governed_scope() context manager — creates a governed context from a WillDecision

Execution modes:
  - Active/Strict Mode: Consequential operations fail closed if not governed.
  - Degraded/Boot Mode: Returns a degraded_mode token for early initialization,
    preflight checks, and local test runs.

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
import inspect
import logging
import os
import secrets
import threading as _threading
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Governance")


# ---------------------------------------------------------------------------
# Context variable — the single source of truth
# ---------------------------------------------------------------------------

_active_receipt: contextvars.ContextVar[GovernanceToken | None] = \
    contextvars.ContextVar("active_governance_receipt", default=None)


@dataclass
class GovernanceToken:
    """A token proving that the current execution is governed."""
    receipt_id: str
    domain: str
    source: str
    timestamp: float = field(default_factory=time.time)
    # Monotonic issue time — expiry is measured against this so a wall-clock
    # rollback cannot extend (or a wall-clock jump prematurely revoke) a
    # token's authority window.
    mono_timestamp: float = field(default_factory=time.monotonic)
    constraints: list = field(default_factory=list)
    ttl: float = 30.0  # tokens expire after 30 seconds
    # A receipt can legitimately be reused for more than one lexical scope.
    # The lease id identifies this exact installation so a copied ContextVar
    # cannot remain authoritative after the creating scope exits.
    lease_id: str = field(default_factory=lambda: secrets.token_urlsafe(18))

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.mono_timestamp) > self.ttl

    @property
    def valid(self) -> bool:
        """Structurally well-formed and unexpired. NOT the same as authority.

        A `VIOLATION` token and a `degraded_mode` token are both valid by
        this test — they have a receipt_id and have not timed out. Sinks
        that gate on `valid` (or merely on `token is not None`) therefore
        accept the two objects that exist to record the ABSENCE of
        governance. Use `authorizes` for that question.
        """
        return bool(self.receipt_id) and not self.expired

    @property
    def authorizes(self) -> bool:
        """Whether this token actually grants permission.

        `require_governance` returns a token on three different paths and
        only one of them is a grant:

        * a real governed scope — authority;
        * `degraded_mode`, meaning the governance runtime is not up yet, so
          the boundary could not be applied;
        * `VIOLATION`, meaning the call was made OUTSIDE a governed context
          and the bypass was recorded.

        The last two are records of absence. Returning them as tokens is
        deliberate — boot has to proceed, and a recorded violation is more
        useful than a silent one — but a downstream effect check that reads
        "I got a token" as "I am permitted" converts the absence of the
        security boundary into permission, which is precisely what the
        boundary exists to prevent.
        """
        return (
            self.valid
            and self.domain not in _NON_AUTHORITY_DOMAINS
            and self.receipt_id not in _NON_AUTHORITY_RECEIPTS
        )


#: Domains and receipt ids that record the ABSENCE of governance rather than
#: a grant of it. Named once so `authorizes` and every gate agree.
_NON_AUTHORITY_DOMAINS = frozenset({"degraded", "ungoverned"})
_NON_AUTHORITY_RECEIPTS = frozenset({"degraded_mode", "VIOLATION"})


@dataclass(frozen=True, slots=True)
class _GovernanceLease:
    token_identity: int
    installed_thread_id: int


_governance_leases: dict[str, _GovernanceLease] = {}
_governance_leases_lock = _threading.RLock()


def _register_governance_lease(token: GovernanceToken) -> None:
    with _governance_leases_lock:
        _governance_leases[token.lease_id] = _GovernanceLease(
            token_identity=id(token),
            installed_thread_id=_threading.get_ident(),
        )


def _revoke_governance_lease(token: GovernanceToken) -> None:
    with _governance_leases_lock:
        lease = _governance_leases.get(token.lease_id)
        if lease is not None and lease.token_identity == id(token):
            _governance_leases.pop(token.lease_id, None)


def governance_token_is_live(token: GovernanceToken | None) -> bool:
    """Return whether this exact lexical lease is still installed.

    Context variables are copied into child asyncio tasks. Resetting the
    parent's ContextVar therefore does not revoke the copied value. The shared
    registry is the revocation authority every sink checks, so a child that
    wakes after scope exit sees an inert receipt rather than a lingering grant.
    """

    if token is None or not token.valid:
        return False
    with _governance_leases_lock:
        lease = _governance_leases.get(token.lease_id)
        return bool(lease is not None and lease.token_identity == id(token))


@dataclass(frozen=True)
class LocalGovernanceDecision:
    """Runtime-owned decision for internal maintenance work.

    This is intentionally narrow: it exists for local durability and telemetry
    operations that must be governed even when they are not directly initiated
    by a user-facing Will decision.
    """

    receipt_id: str
    domain: str
    source: str
    constraints: dict[str, Any] = field(default_factory=dict)


class GovernanceViolationError(RuntimeError):
    """Raised when code attempts to execute without governance."""


GovernanceViolation = GovernanceViolationError


def _governance_production_active() -> bool:
    # Canonical definition lives in core.runtime.mode (the single strictness
    # resolver); the raw env read is the fallback if mode is unavailable.
    try:
        from core.runtime.mode import governance_production_active

        return governance_production_active()
    except (ImportError, AttributeError, RuntimeError):
        return os.getenv("AURA_GOVERNANCE_MODE", "").strip().lower() == "production"


def governance_runtime_active() -> bool:
    """Return True when the runtime should enforce hard governance.

    Operator overrides let CI, release canaries, and adversarial audits force
    governance checks to fail closed before the service container has completed
    boot. That closes the early-boot/test bypass where critical sinks could
    otherwise receive degraded tokens.
    """
    if os.getenv("AURA_GOVERNANCE_MODE", "").strip().lower() in {
        "strict",
        "enforce",
        "production",
    }:
        return True
    if os.getenv("AURA_REQUIRE_GOVERNANCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "strict",
    }:
        return True
    try:
        from core.runtime.service_registry import (
            has_runtime_service,
            is_runtime_registration_locked,
        )

        runtime_services_present = (
            has_runtime_service("executive_core")
            or has_runtime_service("aura_kernel")
            or has_runtime_service("kernel_interface")
        )
        registration_locked = is_runtime_registration_locked()
        if not registration_locked:
            # Early boot can use degraded governance only before canonical
            # runtime services exist. Once kernel/executive/interface is
            # registered, consequential sinks must fail closed even if the
            # container has not finished locking registration.
            return runtime_services_present
        return runtime_services_present or registration_locked
    except ImportError as exc:
        # The registry module is not importable at all. That is genuinely
        # early boot or a partial environment (a tool importing one module),
        # and there is no runtime to enforce against yet.
        record_degradation("governance_context", exc, severity="info")
        logger.debug("Strict governance mode lookup unavailable: %s", exc)
        return False
    except (AttributeError, RuntimeError) as exc:
        # The registry EXISTS and failed to answer. That is not the same
        # thing, and it used to be treated as if it were: any failure
        # returned False, which selects degraded authorization at precisely
        # the moment readiness is uncertain — the absence of the security
        # boundary converted into permission.
        #
        # Enforce instead. A false "governance is active" costs a refused
        # action that can be retried; a false "governance is inactive" is an
        # ungoverned effect nobody can undo.
        record_degradation(
            "governance_context",
            exc,
            severity="critical",
            action="enforced governance because runtime readiness could not be determined",
        )
        logger.warning(
            "Governance readiness could not be determined (%s); enforcing", exc
        )
        return True


def normalize_governance_domain(value: Any) -> str:
    """Normalize Will/constitutional/effect domains into one runtime vocabulary."""
    if hasattr(value, "value"):
        value = value.value
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    aliases = {
        "tool": "tool_execution",
        "memory_mutation": "memory_write",
        "belief_mutation": "memory_write",
        "state_mutation": "state_mutation",
        "expression": "expression",
        "response": "response",
        "initiative": "initiative",
        "continuity": "state_mutation",
    }
    return aliases.get(text, text)


_LOCAL_INTERNAL_DOMAINS = {
    "environment_action",
    "file_write",
    "memory_write",
    "state_mutation",
    "self_modification",
    "tool_execution",
}


def _receipt_component(value: str) -> str:
    text = str(value or "local").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:96] or "local"


def local_internal_decision(
    source: str,
    *,
    domain: str = "state_mutation",
    receipt_prefix: str | None = None,
    constraints: Mapping[str, Any] | None = None,
) -> LocalGovernanceDecision:
    """Build a governed decision for local runtime maintenance.

    Use this for internal durability writes, trace persistence, and health
    artifacts. User/tool actions still need their upstream Will/Authority
    receipts; this helper is not a substitute for consequential action
    authorization.
    """
    source_text = str(source or "").strip()
    if not source_text:
        raise ValueError("local governance source is required")
    normalized_domain = normalize_governance_domain(domain)
    if normalized_domain not in _LOCAL_INTERNAL_DOMAINS:
        raise ValueError(f"unsupported local internal governance domain: {domain}")
    receipt_root = receipt_prefix or f"local-internal-{_receipt_component(source_text)}"
    constraint_payload = {
        "governance_origin": "local_internal",
        "runtime_generated": True,
    }
    if constraints:
        constraint_payload.update(dict(constraints))
    return LocalGovernanceDecision(
        receipt_id=f"{receipt_root}:{time.time_ns()}",
        domain=normalized_domain,
        source=source_text,
        constraints=constraint_payload,
    )


def _decision_constraints(decision: Any) -> dict[str, Any]:
    raw = getattr(decision, "constraints", {})
    if isinstance(raw, dict):
        return dict(raw or {})
    return {}


def _extract_receipt_id(decision: Any) -> str:
    direct = (
        getattr(decision, "receipt_id", "")
        or getattr(decision, "will_receipt_id", "")
        or getattr(decision, "authority_receipt_id", "")
    )
    if direct:
        return str(direct)
    constraints = _decision_constraints(decision)
    for key in ("will_receipt_id", "receipt_id", "authority_receipt_id"):
        value = constraints.get(key)
        if value:
            return str(value)
    return ""


def _extract_domain(decision: Any) -> str:
    constraints = _decision_constraints(decision)
    raw_domain = (
        getattr(decision, "domain", None)
        or getattr(decision, "kind", None)
        or constraints.get("governance_domain")
        or constraints.get("domain")
    )
    return normalize_governance_domain(raw_domain)


def _extract_source(decision: Any) -> str:
    return str(getattr(decision, "source", "unknown") or "unknown")


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def governed_scope(decision: Any, *, ttl: float | None = None):
    """Create a governed execution context from a WillDecision.

    Usage:
        decision = will.decide(...)
        async with governed_scope(decision):
            await do_governed_work()

    ``ttl`` sizes the lease to the operation being authorized. The default is
    GovernanceToken's 30s, which is right for a short tool call and wrong for
    anything a person watches happen: live 2026-07-29, "open Notes and write a
    note" ran its sanctioned 101s budget and lost governance at exactly 30s
    mid-write — "run_applescript:computer_use called outside governed context"
    — after the document had already been created and filled. A lease that
    expires while the work it authorized is still running does not make the
    system safer; it makes a governed operation fail halfway and leaves the
    half behind. Callers that know the budget pass it.
    """
    token = GovernanceToken(
        receipt_id=_extract_receipt_id(decision) or ("degraded_mode" if not governance_runtime_active() else ""),
        domain=_extract_domain(decision),
        source=_extract_source(decision),
        constraints=list(_decision_constraints(decision).items()),
        **_ttl_kwargs(ttl),
    )
    _register_governance_lease(token)
    reset_token = _active_receipt.set(token)
    try:
        yield token
    finally:
        try:
            _active_receipt.reset(reset_token)
        finally:
            _revoke_governance_lease(token)


def _ttl_kwargs(ttl: float | None) -> dict[str, float]:
    """A lease length, when the caller knows one. Never shorter than default."""
    if ttl is None:
        return {}
    try:
        requested = float(ttl)
    except (TypeError, ValueError):
        return {}
    if requested <= 0.0:
        return {}
    # Only ever lengthen: a caller must not be able to shorten the window and
    # strand a sibling operation sharing this scope.
    return {"ttl": max(requested, GovernanceToken.ttl)}


@contextmanager
def governed_scope_sync(decision: Any, *, ttl: float | None = None):
    """Synchronous version of governed_scope."""
    token = GovernanceToken(
        receipt_id=_extract_receipt_id(decision) or ("degraded_mode" if not governance_runtime_active() else ""),
        domain=_extract_domain(decision),
        source=_extract_source(decision),
        constraints=list(_decision_constraints(decision).items()),
        **_ttl_kwargs(ttl),
    )
    _register_governance_lease(token)
    reset_token = _active_receipt.set(token)
    try:
        yield token
    finally:
        try:
            _active_receipt.reset(reset_token)
        finally:
            _revoke_governance_lease(token)


@contextmanager
def local_internal_governed_scope(
    source: str,
    *,
    domain: str = "state_mutation",
    receipt_prefix: str | None = None,
    constraints: Mapping[str, Any] | None = None,
    ttl: float | None = None,
):
    """Create a governed scope for local runtime maintenance work."""
    decision = local_internal_decision(
        source,
        domain=domain,
        receipt_prefix=receipt_prefix,
        constraints=constraints,
    )
    with governed_scope_sync(decision, ttl=ttl) as token:
        yield token


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

def get_active_governance() -> GovernanceToken | None:
    """Get the current governance token, or None if ungoverned."""
    token = _active_receipt.get()
    if governance_token_is_live(token):
        return token
    return None


def is_governed() -> bool:
    """Check if current execution is inside a governed context."""
    return get_active_governance() is not None


def require_governance(
    operation: str = "unknown",
    *,
    strict: bool = False,
    allowed_domains: list[str] | tuple[str, ...] | set[str] | None = None,
) -> GovernanceToken:
    """Assert that current execution is governed. Raises GovernanceViolation if not.

    Use this at the top of any critical function (memory write, tool exec, etc.)
    to ensure it was called through proper governance channels.

    In degraded mode (boot, shutdown, etc.), this logs a warning instead of raising.
    """
    token = get_active_governance()
    if token is not None:
        if allowed_domains:
            normalized_allowed = {
                normalize_governance_domain(domain)
                for domain in allowed_domains
                if domain is not None
            }
            if normalized_allowed and token.domain not in normalized_allowed and token.domain != "degraded":
                logger.warning(
                    "GOVERNANCE DOMAIN VIOLATION: '%s' expected %s but got %s",
                    operation,
                    sorted(normalized_allowed),
                    token.domain,
                )
                _record_violation(f"{operation}:domain:{token.domain}")
                if strict or _governance_production_active():
                    raise GovernanceViolation(
                        f"{operation} requires governance domain {sorted(normalized_allowed)} "
                        f"but active token domain is {token.domain}"
                    )
        return token

    # Check if we're in degraded mode (early boot, shutdown, testing)
    if not governance_runtime_active():
        logger.debug("Governance check '%s' during degraded mode", operation)
        return GovernanceToken(receipt_id="degraded_mode", domain="degraded", source="boot", ttl=300)

    # Log the violation — in production this would be a hard error
    logger.warning("GOVERNANCE VIOLATION: '%s' called outside governed context", operation)

    # Return a violation token that tracks the bypass
    _record_violation(operation)
    if strict or os.getenv("AURA_GOVERNANCE_MODE", "").strip().lower() == "production":
        raise GovernanceViolation(f"{operation} called outside governed context")
    return GovernanceToken(receipt_id="VIOLATION", domain="ungoverned", source=operation, ttl=1)


def governed(fn: Callable) -> Callable:
    """Decorator that enforces governance on a function.

    Usage:
        @governed
        def write_memory(data):
            ...  # will raise if called outside governed_scope
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_governance(fn.__qualname__, strict=True)
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        require_governance(fn.__qualname__, strict=True)
        return await fn(*args, **kwargs)

    if _is_coroutine_function(fn):
        return async_wrapper
    return wrapper


def _is_coroutine_function(fn):
    return inspect.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------

_violations: list = []
_MAX_VIOLATIONS = 200
_violations_lock = _threading.Lock()


def _record_violation(operation: str) -> None:
    """Record a governance violation for audit."""
    with _violations_lock:
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
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("governance_context", exc)
        logger.debug("Governance violation event publish failed for %s: %s", operation, exc)


def get_violations(n: int = 20) -> list:
    """Return recent governance violations."""
    with _violations_lock:
        return list(_violations[-n:])


def get_governance_status() -> dict:
    """Return governance system status."""
    context_token = _active_receipt.get()
    return {
        "currently_governed": is_governed(),
        "active_token": governance_token_is_live(context_token),
        "context_token_present": context_token is not None,
        "total_violations": len(_violations),
        "recent_violations": get_violations(5),
    }
