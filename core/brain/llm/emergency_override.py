"""Time-bounded, receipted overrides for hard safety refusals.

CP126 ``4e0cae60`` and ``1229b07f``. Two of the MLX client's memory guards
could be disabled by a bare process environment value:

* ``AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE`` skipped every spawn-admission
  check before a 20-40 GB worker was launched, and
* ``AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION`` disabled the refusal issued
  *after* critical memory pressure had been positively observed — the last
  guard before the model process can push macOS into swap or jetsam.

Making the bypass loud (a critical degradation record) was necessary but not
sufficient. An environment variable has no expiry, so a flag exported once
for a recovery session keeps disabling the guard for the life of the
process — and, if it lives in a launch profile, for the life of the
deployment. Nothing bounded how many times it could fire, and nothing
durable recorded that the risk had been accepted.

This module makes an emergency override behave like what it is: a decision,
not a setting.

* **Time-bounded.** An override is live for a bounded window from its first
  observation. After that it is EXPIRED and the guard it was disabling comes
  back on by itself. The window may be shortened per call; it can never be
  extended past the ceiling.
* **Use-bounded.** A window still caps the number of times the guard may be
  bypassed, so one flag cannot authorize an unbounded stream of unsafe
  admissions.
* **Receipted.** Every consumption emits a durable ``GovernanceReceipt``
  naming the flag, the guard, the observed condition and the remaining
  budget, so accepting the risk leaves evidence rather than a log line.

Fail-safe direction: any failure to evaluate an override resolves to
INACTIVE — the guard stays on. An override that cannot be proven valid is
not an override.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.runtime.errors import record_degradation

_SUBSYSTEM = "mlx_emergency_override"

# An emergency is measured in minutes, not deployments. A flag left exported
# stops working well before it can become permanent policy.
DEFAULT_WINDOW_S = 900.0
# No single override may authorize more than this many bypasses of a guard
# that stands between the host and jetsam.
DEFAULT_MAX_USES = 8

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OverrideDecision:
    """Whether a guard may be bypassed right now, and on what evidence."""

    active: bool
    flag: str
    guard: str
    reason: str
    first_seen_unix: Optional[float] = None
    expires_at_unix: Optional[float] = None
    uses: int = 0
    uses_remaining: Optional[int] = None
    receipt_id: Optional[str] = None

    def as_detail(self) -> str:
        """Compact form for degradation details and log lines."""
        if not self.active:
            return f"{self.flag}:{self.reason}"
        remaining = "?" if self.uses_remaining is None else str(self.uses_remaining)
        return f"{self.flag}:active:uses_left={remaining}"


@dataclass
class _OverrideState:
    first_seen_unix: float
    uses: int = 0
    expired_reported: bool = False
    exhausted_reported: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_states: dict[str, _OverrideState] = {}


def _flag_is_set(flag: str) -> bool:
    return str(os.environ.get(flag, "")).strip().lower() in _TRUE


def consume_override(
    flag: str,
    *,
    guard: str,
    observed: str = "",
    window_s: float = DEFAULT_WINDOW_S,
    max_uses: int = DEFAULT_MAX_USES,
    now: Optional[float] = None,
) -> OverrideDecision:
    """Decide whether ``flag`` may bypass ``guard`` on this call.

    Consuming is the act of deciding: a call that returns ``active=True``
    spends one use of the override's budget and emits a receipt. Callers must
    call this exactly once per guarded decision.

    ``observed`` describes the condition being overridden (for example the
    memory-pressure reason), so the receipt records what risk was accepted
    rather than merely that a flag was set.
    """
    at = time.time() if now is None else float(now)
    if not _flag_is_set(flag):
        _forget(flag)
        return OverrideDecision(False, flag, guard, "not_set")

    window = max(0.0, float(window_s))
    ceiling = min(window, DEFAULT_WINDOW_S)
    uses_cap = max(0, min(int(max_uses), DEFAULT_MAX_USES))

    with _lock:
        state = _states.get(flag)
        if state is None:
            state = _OverrideState(first_seen_unix=at)
            _states[flag] = state
        expires_at = state.first_seen_unix + ceiling
        expired = at >= expires_at
        exhausted = state.uses >= uses_cap
        if not expired and not exhausted:
            state.uses += 1
        uses = state.uses
        report_expired = expired and not state.expired_reported
        report_exhausted = (not expired) and exhausted and not state.exhausted_reported
        if report_expired:
            state.expired_reported = True
        if report_exhausted:
            state.exhausted_reported = True

    if expired:
        if report_expired:
            record_degradation(
                _SUBSYSTEM,
                RuntimeError(
                    f"{flag} expired {at - expires_at:.0f}s ago; {guard} is enforced again"
                ),
                action="emergency override expired; safety guard re-armed",
                severity="warning",
            )
        return OverrideDecision(
            False, flag, guard, "expired",
            first_seen_unix=state.first_seen_unix,
            expires_at_unix=expires_at,
            uses=uses,
            uses_remaining=0,
        )

    if exhausted:
        if report_exhausted:
            record_degradation(
                _SUBSYSTEM,
                RuntimeError(
                    f"{flag} exhausted its budget of {uses_cap} bypasses; "
                    f"{guard} is enforced again"
                ),
                action="emergency override exhausted; safety guard re-armed",
                severity="warning",
            )
        return OverrideDecision(
            False, flag, guard, "exhausted",
            first_seen_unix=state.first_seen_unix,
            expires_at_unix=expires_at,
            uses=uses,
            uses_remaining=0,
        )

    receipt_id = _emit_risk_receipt(
        flag=flag, guard=guard, observed=observed,
        expires_at=expires_at, uses=uses, uses_cap=uses_cap, at=at,
    )
    return OverrideDecision(
        True, flag, guard, "active",
        first_seen_unix=state.first_seen_unix,
        expires_at_unix=expires_at,
        uses=uses,
        uses_remaining=max(0, uses_cap - uses),
        receipt_id=receipt_id,
    )


def _emit_risk_receipt(
    *,
    flag: str,
    guard: str,
    observed: str,
    expires_at: float,
    uses: int,
    uses_cap: int,
    at: float,
) -> Optional[str]:
    """Record that a named risk was accepted, durably.

    Best-effort by design: the receipt store must never be able to turn a
    recovery override into a crash. A failure to record is itself recorded.
    """
    try:
        from core.runtime.receipts import GovernanceReceipt, get_receipt_store

        receipt = GovernanceReceipt(
            cause=f"emergency_override:{guard}",
            domain="mlx_memory_admission",
            action=f"bypass:{guard}",
            approved=True,
            reason=(
                f"{flag} accepted the risk of {observed or 'an unobserved condition'}"
            ),
            metadata={
                "flag": flag,
                "guard": guard,
                "observed": observed,
                "expires_at_unix": round(expires_at, 3),
                "seconds_remaining": round(max(0.0, expires_at - at), 3),
                "use": uses,
                "use_budget": uses_cap,
            },
        )
        return getattr(get_receipt_store().emit(receipt), "receipt_id", None)
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
        record_degradation(
            _SUBSYSTEM, exc,
            action="emergency-override risk receipt not recorded",
            severity="warning",
        )
        return None


def _forget(flag: str) -> None:
    """Drop state for a flag that is no longer set.

    Unsetting and re-setting a flag is a NEW decision and gets a new window;
    it must not inherit an exhausted budget, and it must not let an operator
    reset the clock without touching the environment.
    """
    with _lock:
        _states.pop(flag, None)


def override_status(flag: str, *, now: Optional[float] = None) -> dict[str, Any]:
    """Inspect an override without consuming a use (for health surfaces)."""
    at = time.time() if now is None else float(now)
    with _lock:
        state = _states.get(flag)
        set_now = _flag_is_set(flag)
        if state is None:
            return {"flag": flag, "set": set_now, "seen": False}
        expires_at = state.first_seen_unix + DEFAULT_WINDOW_S
        return {
            "flag": flag,
            "set": set_now,
            "seen": True,
            "first_seen_unix": round(state.first_seen_unix, 3),
            "expires_at_unix": round(expires_at, 3),
            "expired": at >= expires_at,
            "uses": state.uses,
            "uses_remaining": max(0, DEFAULT_MAX_USES - state.uses),
        }


def reset_overrides_for_test() -> None:
    with _lock:
        _states.clear()


__all__ = [
    "DEFAULT_MAX_USES",
    "DEFAULT_WINDOW_S",
    "OverrideDecision",
    "consume_override",
    "override_status",
    "reset_overrides_for_test",
]
