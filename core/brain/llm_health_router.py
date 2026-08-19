"""
core/brain/llm_health_router.py
────────────────────────────────
Replacement for IntelligentLLMRouter.

Fixes:
  - Zero-token / whitespace-only responses treated as failure, not success
  - Primary endpoint failure triggers genuine fallback to local MLX
  - Per-endpoint health tracking with circuit breaker pattern
  - Response validation before acceptance
  - Structured logging that distinguishes real success from empty success

Drop-in: replace the existing router instantiation in orchestrator_boot.py
with HealthAwareLLMRouter.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.deferral_record import record_deferral
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    audit_lane_assignments,
    guard_solver_request,
    normalize_endpoint_name,
)
from core.brain.llm.runtime_wiring import (
    _merge_system_prompt,
    build_agentic_tool_map,
    prepare_runtime_payload,
    should_force_tool_handoff,
)
from core.phases.response_contract import ResponseContract
from core.runtime.desktop_boot_safety import desktop_resource_guard_enabled
from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.runtime.proof_policy import (
    is_proof_evaluation_purpose,
    is_strict_proof_answer_prompt,
    mlx_strict_answer_contract_enabled,
    proof_model_tier,
    proof_run_active,
)
from core.runtime.turn_analysis import analyze_turn
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Brain.HealthRouter")

# ── Generation concurrency gate ────────────────────────────────────────
# Round-9 spike stacks caught NINE concurrent generate calls stacked for
# a single user turn (draft/retry fan-out never cancelling predecessors).
# Each in-process generation holds GB-scale KV/context: the stack-up
# allocated ~2GB/s of compressible pages until macOS executed the
# process at a 78GB phys_footprint. Local generation is now a bounded
# resource: callers either acquire a slot within the wait budget or get
# a truthful saturation failure — stacking is the one outcome that can
# never happen again.
import threading as _threading  # noqa: E402 - gate lives with its rationale block


def generation_concurrency_limit(env: Mapping[str, str] | None = None) -> int:
    """Return the process-wide generation limit for the active runtime profile."""

    env = env or os.environ
    raw_limit = str(env.get("AURA_MAX_CONCURRENT_GENERATIONS", "2") or "2").strip()
    try:
        configured = max(1, int(raw_limit))
    except (TypeError, ValueError, OverflowError):
        configured = 2

    allow_desktop_parallelism = str(
        env.get("AURA_ALLOW_CONCURRENT_DESKTOP_GENERATIONS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if desktop_resource_guard_enabled(env) and not allow_desktop_parallelism:
        return 1
    return configured


_GENERATION_GATE = _threading.BoundedSemaphore(
    generation_concurrency_limit()
)
_GENERATION_GATE_STATE_LOCK = _threading.Lock()
_GENERATION_GATE_ACTIVE_LEASES: dict[int, tuple[float, str]] = {}
_GENERATION_GATE_LEASE_DEADLINES: dict[int, float] = {}
_GENERATION_GATE_FORCED_LEASES: set[int] = set()
_GENERATION_GATE_NEXT_LEASE_ID = 0
_GENERATION_GATE_LAST_ACQUIRED_AT = 0.0
_GENERATION_GATE_LAST_OWNER = ""
# Wait long enough to outlast one full serialized generation: gated
# turns measure 31-46s live (2026-06-11), so the old 20s wait starved
# any request arriving while both slots were mid-turn — external
# validation's third coding repair died exactly that way while holding
# an unused 240s budget. 75s covers one slow turn plus margin; callers
# with shorter deadlines still bail via their own timeouts.
def _gate_budget_env_s(name: str, default: float) -> float:
    """Parse a gate timing budget safely at import time.

    A malformed value must not prevent the router module from importing,
    and NaN/inf/negative budgets must not reach semaphore timeouts where
    they disable or invert the wait semantics.
    """
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < 0.0:
        return float(default)
    return value


_GENERATION_GATE_WAIT_S = _gate_budget_env_s("AURA_GENERATION_GATE_WAIT_S", 75.0)
# Background work never owns recovery authority over the serving lane. It may
# wait briefly for naturally available capacity, then returns a retryable
# deferral instead of accumulating 75-second waiters or killing a warm worker.
_BACKGROUND_GENERATION_GATE_WAIT_S = _gate_budget_env_s(
    "AURA_BACKGROUND_GENERATION_GATE_WAIT_S", 5.0
)
# Foreground preemption ladder budgets: a user turn waits only this grace
# before asking a BACKGROUND gate holder to yield cooperatively (the worker
# stops between tokens and stays warm), then this long for the yield to land.
# Foreground-vs-foreground contention still honors the full gate window.
_FOREGROUND_GATE_GRACE_S = _gate_budget_env_s("AURA_FOREGROUND_GATE_GRACE_S", 5.0)
_FOREGROUND_SOFT_CANCEL_WAIT_S = _gate_budget_env_s(
    "AURA_FOREGROUND_SOFT_CANCEL_WAIT_S", 10.0
)
_GATE_SATURATION_RESULT = {
    "ok": False,
    "text": "",
    "endpoint": "generation_gate_saturated",
    "tokens": 0,
    "error": (
        "local generation lane saturated: refusing to stack another "
        "concurrent generation (memory-bomb prevention)"
    ),
}


def _generation_gate_owner(origin: str, purpose: str) -> str:
    origin = str(origin or "unknown").strip() or "unknown"
    purpose = str(purpose or "unknown").strip() or "unknown"
    return f"{origin}:{purpose}"


def _generation_owner_is_user_foreground(owner: str) -> bool:
    owner = str(owner or "").strip().lower()
    if not owner:
        return False
    return any(
        marker in owner
        for marker in (
            "user:",
            "desktop",
            "voice",
            "foreground",
            "response_generation_user",
        )
    )


def _oldest_generation_gate_lease() -> tuple[int, float, str] | None:
    with _GENERATION_GATE_STATE_LOCK:
        if not _GENERATION_GATE_ACTIVE_LEASES:
            return None
        lease_id, (acquired_at, owner) = min(
            _GENERATION_GATE_ACTIVE_LEASES.items(),
            key=lambda item: item[1][0],
        )
        return lease_id, float(acquired_at), str(owner or "unknown")


def _generation_gate_lease_has_time(
    lease_id: int,
    *,
    now: float | None = None,
) -> bool | None:
    """Return whether a lease is inside its owner budget, or None for legacy leases."""

    with _GENERATION_GATE_STATE_LOCK:
        deadline = _GENERATION_GATE_LEASE_DEADLINES.get(int(lease_id))
    if deadline is None:
        return None
    current = time.time() if now is None else float(now)
    return current < float(deadline)


def _generation_gate_busy_result(owner: str) -> dict[str, Any]:
    result = dict(_GATE_SATURATION_RESULT)
    result["endpoint"] = "generation_gate_busy_foreground"
    result["error"] = (
        "local generation lane is busy with an active foreground user generation; "
        f"refusing to force-release owner={str(owner or 'unknown')[:120]}"
    )
    return result


def _background_generation_gate_deferred_result(owner: str) -> dict[str, Any]:
    result = dict(_GATE_SATURATION_RESULT)
    result.update(
        {
            "endpoint": "generation_gate_background_deferred",
            "deferred": True,
            "retryable": True,
            "error": (
                "background generation deferred while the local generation lane "
                f"is owned by {str(owner or 'unknown')[:120]}"
            ),
        }
    )
    return result


def _active_foreground_generation_owner() -> str:
    oldest_lease = _oldest_generation_gate_lease()
    if oldest_lease is None:
        return ""
    _lease_id, _acquired_at, owner = oldest_lease
    return owner if _generation_owner_is_user_foreground(owner) else ""


def _oldest_generation_gate_lease_age_s() -> float:
    oldest_lease = _oldest_generation_gate_lease()
    if oldest_lease is None:
        return 0.0
    _lease_id, acquired_at, _owner = oldest_lease
    return max(0.0, time.time() - float(acquired_at))


def generation_gate_snapshot() -> dict[str, Any]:
    """Return a read-only snapshot for schedulers and health probes."""

    with _GENERATION_GATE_STATE_LOCK:
        now = time.time()
        active = {
            int(lease_id): {
                "age_s": max(0.0, now - float(acquired_at)),
                "owner": str(owner or "unknown"),
                "deadline_at": _GENERATION_GATE_LEASE_DEADLINES.get(int(lease_id)),
                "deadline_remaining_s": (
                    max(
                        0.0,
                        float(_GENERATION_GATE_LEASE_DEADLINES[int(lease_id)]) - now,
                    )
                    if int(lease_id) in _GENERATION_GATE_LEASE_DEADLINES
                    else None
                ),
            }
            for lease_id, (acquired_at, owner) in _GENERATION_GATE_ACTIVE_LEASES.items()
        }
        oldest = None
        if active:
            oldest_id = max(active, key=lambda lease_id: active[lease_id]["age_s"])
            oldest = {"lease_id": oldest_id, **active[oldest_id]}
        return {
            "active_count": len(active),
            "active": active,
            "oldest": oldest,
            "last_acquired_at": float(_GENERATION_GATE_LAST_ACQUIRED_AT or 0.0),
            "last_owner": str(_GENERATION_GATE_LAST_OWNER or ""),
            "wait_budget_s": float(_GENERATION_GATE_WAIT_S),
        }


def _mark_generation_gate_acquired(
    owner: str,
    *,
    timeout_s: float | None = None,
) -> int:
    global _GENERATION_GATE_NEXT_LEASE_ID, _GENERATION_GATE_LAST_ACQUIRED_AT, _GENERATION_GATE_LAST_OWNER
    with _GENERATION_GATE_STATE_LOCK:
        _GENERATION_GATE_NEXT_LEASE_ID += 1
        lease_id = _GENERATION_GATE_NEXT_LEASE_ID
        acquired_at = time.time()
        _GENERATION_GATE_ACTIVE_LEASES[lease_id] = (acquired_at, str(owner or "unknown"))
        if timeout_s is not None:
            try:
                bounded_timeout = float(timeout_s)
            except (TypeError, ValueError, OverflowError):
                bounded_timeout = 0.0
            if math.isfinite(bounded_timeout) and bounded_timeout > 0.0:
                _GENERATION_GATE_LEASE_DEADLINES[lease_id] = (
                    acquired_at + bounded_timeout
                )
        _GENERATION_GATE_LAST_ACQUIRED_AT = acquired_at
        _GENERATION_GATE_LAST_OWNER = str(owner or "unknown")
        return lease_id


async def _acquire_generation_gate_slot(wait_s: float) -> bool:
    """Acquire one generation-gate permit with a cancellation-safe handoff.

    ``asyncio.to_thread(_GENERATION_GATE.acquire, ...)`` leaked permits: when
    the awaiting coroutine was cancelled (upstream ``wait_for`` timeout), the
    worker thread kept waiting, could acquire the permit AFTER the caller was
    gone, and nothing ever released it — the two-slot gate then served the
    rest of the process lifetime on one slot (or zero). The handoff below
    makes acquisition atomic with respect to cancellation: an abandoned
    waiter's permit is handed straight back to the semaphore.
    """
    try:
        wait_s = float(wait_s)
    except (TypeError, ValueError):
        wait_s = 0.0
    if not math.isfinite(wait_s) or wait_s < 0.0:
        wait_s = 0.0
    handoff_lock = _threading.Lock()
    state = {"abandoned": False, "delivered": False}

    def _worker() -> bool:
        got = _GENERATION_GATE.acquire(True, wait_s)
        if not got:
            return False
        with handoff_lock:
            if state["abandoned"]:
                try:
                    _GENERATION_GATE.release()
                except ValueError:
                    pass
                return False
            state["delivered"] = True
            return True

    try:
        return await asyncio.to_thread(_worker)
    except asyncio.CancelledError:
        with handoff_lock:
            if state["delivered"]:
                # The thread transferred the permit but the await was
                # cancelled before the result reached us — hand it back.
                try:
                    _GENERATION_GATE.release()
                except ValueError:
                    pass
            else:
                state["abandoned"] = True
        raise


async def acquire_external_generation_gate_lease(
    *,
    owner: str,
    timeout_s: float,
    wait_s: float = 5.0,
) -> int | None:
    """Admit direct resident-model work into the process-wide generation lane.

    Recursive latent episodes use the MLX client directly because the router's
    ordinary text-generation API cannot express their worker action. They still
    must own the same process-wide lease as every routed generation so health
    probes, retries, and background work cannot overlap or misclassify them as
    abandoned work.
    """

    try:
        bounded_timeout = float(timeout_s)
        bounded_wait = float(wait_s)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(bounded_timeout)
        or bounded_timeout <= 0.0
        or not math.isfinite(bounded_wait)
        or bounded_wait < 0.0
    ):
        return None
    acquired = await _acquire_generation_gate_slot(min(bounded_timeout, bounded_wait))
    if not acquired:
        return None
    return _mark_generation_gate_acquired(
        str(owner or "external_generation"),
        timeout_s=bounded_timeout,
    )


def release_external_generation_gate_lease(lease_id: int) -> None:
    """Release a lease returned by acquire_external_generation_gate_lease."""

    _release_generation_gate_after_call(int(lease_id))


def _release_generation_gate_after_call(lease_id: int) -> None:
    """Release the generation gate, accounting for watchdog-forced releases."""

    should_release = False
    with _GENERATION_GATE_STATE_LOCK:
        if lease_id in _GENERATION_GATE_FORCED_LEASES:
            _GENERATION_GATE_FORCED_LEASES.discard(lease_id)
            return
        if lease_id in _GENERATION_GATE_ACTIVE_LEASES:
            _GENERATION_GATE_ACTIVE_LEASES.pop(lease_id, None)
            _GENERATION_GATE_LEASE_DEADLINES.pop(lease_id, None)
            should_release = True
    if not should_release:
        return
    try:
        _GENERATION_GATE.release()
    except ValueError:
        pass


def force_release_generation_gate(
    reason: str = "hard_generation_deadline", *, release_all: bool = False
) -> bool:
    """Emergency-release stale router gate lease(s) from a watchdog thread.

    release_all reclaims EVERY active lease: after force_abort kills the
    workers, every holder is dead by construction — the overnight July 4
    incident held the second permit of the two-slot gate with a dead
    lease, so the 2s re-acquire failed, every later attempt got the
    saturation result without reaching a client, the conversation lane
    stayed cold, and the launcher executed a recovering runtime.
    """

    reason = str(reason or "hard_generation_deadline")
    released_any = False
    with _GENERATION_GATE_STATE_LOCK:
        max_reclaims = len(_GENERATION_GATE_ACTIVE_LEASES)
    for _ in range(max_reclaims):
        with _GENERATION_GATE_STATE_LOCK:
            if not _GENERATION_GATE_ACTIVE_LEASES:
                break
            lease_id, (acquired_at, owner) = min(
                _GENERATION_GATE_ACTIVE_LEASES.items(),
                key=lambda item: item[1][0],
            )
            _GENERATION_GATE_ACTIVE_LEASES.pop(lease_id, None)
            _GENERATION_GATE_LEASE_DEADLINES.pop(lease_id, None)
            _GENERATION_GATE_FORCED_LEASES.add(lease_id)
            age_s = max(0.0, time.time() - acquired_at)
        try:
            _GENERATION_GATE.release()
        except ValueError:
            with _GENERATION_GATE_STATE_LOCK:
                _GENERATION_GATE_FORCED_LEASES.discard(lease_id)
            break
        released_any = True
        record_degradation(
            "llm_health_router",
            TimeoutError(f"generation gate forcibly released after {age_s:.1f}s"),
            severity="degraded",
            action=f"released stale generation gate lease for {owner}: {reason}",
        )
        if not release_all:
            break
    return released_any


def _record_router_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("llm_health_router", exc, severity=severity, action=action)


_ROUTER_CLIENT_ERRORS = (
    httpx.HTTPError,
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    TypeError,
    ValueError,
    Exception,
)


def _endpoint_call_timeout(timeout: float) -> float:
    """Outer watchdog for an endpoint call.

    The endpoint/client still receives the original timeout as its cooperative
    budget. This wrapper adds a small cleanup grace window so a blocked local
    runtime cannot hold the router forever if the client fails to observe that
    budget.
    """
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError, OverflowError):
        timeout_s = 120.0
    timeout_s = max(0.1, timeout_s)
    grace_s = min(5.0, max(0.25, timeout_s * 0.1))
    return timeout_s + grace_s


def _endpoint_call_budgets(
    timeout: float,
    *,
    foreground_local: bool = False,
    prompt_chars: int = 0,
    max_tokens: int | None = None,
    benchmark_request: bool = False,
    proof_evaluation_contract: bool = False,
    health_probe: bool = False,
) -> tuple[float, float]:
    """Return cooperative client timeout and hard wall-clock watchdog budget."""
    try:
        timeout_s = max(0.1, float(timeout))
    except (TypeError, ValueError, OverflowError):
        timeout_s = 120.0
    wall_s = _endpoint_call_timeout(timeout_s)
    cooperative_s = timeout_s

    if (
        foreground_local
        and timeout_s >= 60.0
        and not benchmark_request
        and not proof_evaluation_contract
        and not health_probe
    ):
        try:
            token_count = int(max_tokens or 0)
        except (TypeError, ValueError, OverflowError):
            token_count = 0
        compact_turn = int(prompt_chars or 0) <= 10_000 and token_count <= 768
        extended_turn = not compact_turn and token_count <= 1536
        if compact_turn or extended_turn:
            env_name = (
                "AURA_FOREGROUND_LOCAL_COMPACT_WALL_TIMEOUT_S"
                if compact_turn
                else "AURA_FOREGROUND_LOCAL_EXTENDED_WALL_TIMEOUT_S"
            )
            default_cap = 105.0 if compact_turn else 150.0
            try:
                cap_s = max(
                    30.0,
                    float(os.environ.get(env_name, str(default_cap)) or default_cap),
                )
            except (TypeError, ValueError, OverflowError):
                cap_s = default_cap
            wall_s = min(wall_s, cap_s)
            cooperative_s = min(cooperative_s, max(5.0, wall_s - 2.0))
        else:
            # A long-form answer already carries a bounded owning deadline from
            # the desktop route.  Replacing it here with the ordinary 150-second
            # cap makes the requested token budget impossible to consume and
            # turns healthy slow decoding into a false endpoint failure.
            cooperative_s = min(cooperative_s, max(5.0, wall_s - 2.0))

    return cooperative_s, wall_s


def _proof_primary_lane_active(*, origin: str) -> bool:
    """Return whether this router build/call must expose only the primary lane."""
    try:
        return bool(proof_run_active(origin=origin) and proof_model_tier() == "primary")
    except _ROUTER_CLIENT_ERRORS as exc:
        _record_router_degradation(
            exc,
            action="failed closed while resolving proof-primary lane policy",
            severity="degraded",
        )
        return True


def _force_abort_endpoint_client(client: Any, *, reason: str) -> bool:
    abort = getattr(client, "force_abort_active_generation", None)
    if not callable(abort):
        return False
    try:
        return bool(abort(reason=reason))
    except _ROUTER_CLIENT_ERRORS as exc:
        _record_router_degradation(
            exc,
            action="continued routing after endpoint force-abort failed",
            severity="error",
        )
        logger.warning("Endpoint force-abort failed: %s", exc)
        return False


def _start_endpoint_wall_clock_watchdog(
    client: Any,
    *,
    reason: str,
    timeout_s: float,
) -> tuple[threading.Event, dict[str, bool], threading.Timer]:
    """Abort non-cooperative local inference on wall-clock time.

    ``asyncio.wait_for`` only fires when the awaited coroutine yields. The local
    MLX stack can block during native/model work, so proof and desktop routes
    need a thread-backed watchdog that can terminate the active generation even
    if the event loop is temporarily occupied.
    """

    fired = threading.Event()
    aborted = {"value": False}

    def _abort() -> None:
        fired.set()
        aborted["value"] = _force_abort_endpoint_client(client, reason=reason)

    watchdog = threading.Timer(max(0.01, float(timeout_s)), _abort)
    watchdog.daemon = True
    watchdog.start()
    return fired, aborted, watchdog


_USER_FACING_ORIGINS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "desktop",
    "desktop-ui",
    "gui",
    "ws",
    "websocket",
    "direct",
    "external",
    "native-shell",
    "test",
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "continuous",
    "consolidation",
    "dream",
    "growth",
    "impulse",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    "internal",
    "system",
    "terminal",
    "volition",
    "witness",
})

_USER_FACING_PURPOSES = frozenset({
    "chat",
    "conversation",
    "expression",
    "reply",
    "user_response",
})


def _endpoint_provider_identity(endpoint: Any) -> str:
    """Return a concrete provider identity for one registered endpoint."""

    if bool(getattr(endpoint, "is_local", False)):
        return "local"
    return "remote_provider_removed"


# ── Circuit Breaker States ────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Failed — requests blocked, fallback used
    HALF_OPEN = "half_open" # Testing — one probe request allowed


# A half-open probe that never reports back must not wedge the endpoint
# closed forever.
_ENDPOINT_HALF_OPEN_LEASE_TTL_S = 30.0


@dataclass
class EndpointHealth:
    name: str
    url: str
    model: str
    is_local: bool = False
    tier: Any = "local" # Matches LLMTier enum or str ("local", "api_deep", "api_fast")
    client: Any = None

    # Circuit breaker. failure_count is the CURRENT consecutive-failure
    # streak (reset by any success); lifetime_failure_count is the honest
    # historical total that reports label "failures". The old design mixed
    # the two: intermittent failures accumulated across successes until they
    # opened the circuit, and recovery erased history.
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    lifetime_failure_count: int = 0
    transient_trip_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    last_failure_reason: str = ""
    # Monotonic deadline for OPEN→HALF_OPEN eligibility (wall-clock
    # last_failure is display-only; clock adjustments must not reopen or
    # wedge circuits).
    cooldown_until_monotonic: float = 0.0
    half_open_probe_at_monotonic: float = 0.0

    # Performance tracking
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    empty_responses: int = 0

    # Config
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    min_tokens_for_success: int = 1

    def __post_init__(self) -> None:
        # dataclass + concurrent async/thread callers: every state
        # transition runs under this lock (the router allocated a lock and
        # never used it for endpoint state — all transitions raced).
        self._lock = threading.Lock()

    def record_success(self, tokens: int, latency_ms: float):
        with self._lock:
            self.success_count += 1
            self.total_requests += 1
            self.total_tokens += tokens
            self.last_success = time.time()
            # A success ends the consecutive-failure streak in EVERY state —
            # intermittent failures must not silently accumulate across
            # healthy successes until they open the circuit.
            self.failure_count = 0
            self.half_open_probe_at_monotonic = 0.0

            if self.state != CircuitState.CLOSED:
                logger.info("Circuit CLOSED for %s — probe succeeded", self.name)
                self.state = CircuitState.CLOSED

            # Rolling average latency
            if latency_ms >= 0:
                if self.avg_latency_ms == 0:
                    self.avg_latency_ms = latency_ms
                else:
                    self.avg_latency_ms = (self.avg_latency_ms * 0.8) + (latency_ms * 0.2)

    def record_failure(self, reason: str):
        with self._lock:
            self.failure_count += 1
            self.lifetime_failure_count += 1
            self.total_requests += 1
            self.last_failure = time.time()
            self.last_failure_reason = str(reason or "")[:200]
            self.half_open_probe_at_monotonic = 0.0

            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    logger.warning(
                        "Circuit OPEN for %s after %d failures. Reason: %s",
                        self.name, self.failure_count, reason
                    )
                self.state = CircuitState.OPEN
                self.cooldown_until_monotonic = time.monotonic() + self.recovery_timeout

    def trip_temporarily(self, reason: str):
        """Open the circuit on a transient MLX-runtime failure without poisoning the failure streak."""
        with self._lock:
            self.total_requests += 1
            self.transient_trip_count += 1
            self.last_failure = time.time()
            self.last_failure_reason = f"transient:{str(reason or '')[:180]}"
            self.half_open_probe_at_monotonic = 0.0
            if self.state != CircuitState.OPEN:
                logger.warning(
                    "Circuit OPEN for %s on transient runtime failure. Reason: %s",
                    self.name,
                    reason,
                )
            self.state = CircuitState.OPEN
            self.cooldown_until_monotonic = time.monotonic() + self.recovery_timeout

    def record_empty(self):
        """Zero-token or whitespace-only response — treat as failure."""
        with self._lock:
            self.empty_responses += 1
        self.record_failure("empty_response")

    def is_available(self) -> bool:
        """ADMISSION check — may grant the single half-open probe lease.

        Routing calls this before dispatch. An OPEN circuit whose cooldown
        elapsed admits exactly ONE caller as the probe; concurrent callers
        keep failing over until the probe records success. Pure observers
        (health reports, readiness, GUI) must use :meth:`peek_available` —
        reads that transition circuit state were themselves a defect.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if now < self.cooldown_until_monotonic:
                    return False
                logger.info("Circuit HALF-OPEN for %s — probing", self.name)
                self.state = CircuitState.HALF_OPEN
                self.half_open_probe_at_monotonic = now
                return True
            # HALF_OPEN: only the lease holder proceeds; a stale lease
            # (probe never reported) is re-grantable after the TTL.
            lease = self.half_open_probe_at_monotonic
            if lease <= 0.0 or (now - lease) > _ENDPOINT_HALF_OPEN_LEASE_TTL_S:
                self.half_open_probe_at_monotonic = now
                return True
            return False

    def peek_available(self) -> bool:
        """Pure availability snapshot — never mutates circuit state."""
        with self._lock:
            return self.state == CircuitState.CLOSED

    def probe_eligible(self) -> bool:
        """Non-mutating routing eligibility: would :meth:`is_available` admit a caller?

        Candidate-list building must use this instead of ``is_available`` so
        that merely ENUMERATING endpoints does not consume half-open probe
        leases or flip OPEN circuits to HALF_OPEN. The mutating admission
        check runs once, immediately before dispatch to the chosen endpoint.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                return now >= self.cooldown_until_monotonic
            lease = self.half_open_probe_at_monotonic
            return lease <= 0.0 or (now - lease) > _ENDPOINT_HALF_OPEN_LEASE_TTL_S

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "tier": getattr(self, "tier", "standard"),
                "state": self.state.value,
                "failures": self.lifetime_failure_count,
                "failure_streak": self.failure_count,
                "transient_trips": self.transient_trip_count,
                "successes": self.success_count,
                "empty_responses": self.empty_responses,
                "last_failure_reason": self.last_failure_reason,
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "total_tokens": self.total_tokens,
            }


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_response(text: str | None, min_tokens: int = 1) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    A response is invalid if:
      - It is None
      - It is empty or whitespace-only
      - It contains only punctuation
      - It is suspiciously short (< min_tokens words)
    """
    if text is None:
        return False, "none_response"
    stripped = text.strip()
    if not stripped:
        return False, "empty_whitespace"
    words = stripped.split()
    if len(words) < min_tokens:
        return False, f"below_min_tokens_{min_tokens}"
    # Punctuation-only output (".", "???", "---") is not a served response.
    if not any(ch.isalnum() for ch in stripped):
        return False, "punctuation_only"
    lower = stripped.lower()
    # Error-marker screening is length-bounded: a provider error body is a
    # short marker-led string, while a legitimate explanation that merely
    # BEGINS with a word like "timeout" must not be rejected. Unambiguous
    # markers get a wider bound than generic English words.
    if len(stripped) <= 400:
        for marker in (
            "i am currently offline",
            "i cannot process that",
            "error:",
            "[error]",
            "model_not_found",
            '{"error"',
            "<html",
        ):
            if lower.startswith(marker):
                return False, f"error_marker:{marker}"
    if len(stripped) <= 120:
        for marker in (
            "connection refused",
            "connection reset",
            "timeout",
            "timed out",
            "internal server error",
            "service unavailable",
            "rate limit",
            "too many requests",
            "model not loaded",
            "context length exceeded",
        ):
            if lower.startswith(marker):
                return False, f"error_marker:{marker}"
    return True, "ok"


# A draft we rejected on QUALITY says nothing about the endpoint that produced
# it. These outcomes must never be recorded as endpoint damage: the model ran,
# returned text, and a gate above it declined the text. Treating them as
# transport failures opened the Cortex circuit and cost later turns the primary
# lane over a verdict the infrastructure had no part in.
_SURFACE_QUALITY_REJECTIONS = frozenset(
    {
        "surface_quality_rejected",       # the worker's own surface gate
        "user_facing_assessment_rejected",  # inference_gate, caller side
        "model_text_integrity_rejected",    # inference_gate, malformed shape
    }
)


#: Readiness blockers that mean "becoming ready", not "broken".
#:
#: A lane reports why it is not ready as a comma-joined list of these. Every
#: one of them clears on its own: a warmup finishes, a generation completes,
#: an init completes. None of them says the endpoint is unreliable.
_STILL_COMING_UP = frozenset(
    {
        "warmup_in_flight",
        "warmup_foreground_owner",
        "active_generation_in_flight",
        "init_not_complete",
        "lane_warming",
        "lane_recovering",
    }
)


def _only_still_coming_up(error: str) -> bool:
    """True when every reason given is the lane still becoming ready.

    LIVE 2026-08-19, mid-game: "Circuit OPEN for Cortex after 5 failures.
    Reason: warmup_in_flight,warmup_foreground_owner", then a cascade cleanup
    force-killed the worker that was warming, then a respawn, then the same
    again. A pursuit asking during a reload counted five times against a
    worker whose only fault was not being finished yet.

    One genuine fault in the list — a dead worker, a shutdown — and this says
    nothing, because a real problem alongside a warmup is still a real
    problem.
    """
    parts = [part.strip() for part in str(error or "").lower().split(",") if part.strip()]
    return bool(parts) and all(part in _STILL_COMING_UP for part in parts)


def _is_transient_local_runtime_failure(error: str) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    if _only_still_coming_up(normalized):
        return True
    return normalized in {
        "client_returned_no_text",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith(
        (
            "background_deferred:",
            "foreground_quiet_window",
            "foreground_busy",
            "mlx_runtime_unavailable:",
            "mlx_runtime_probe_failed:",
            "local_runtime_unavailable:",
            "prewarm_failed:",
            # A lane skipped because it is still warming/recovering is a
            # transient trip: retry the cortex as soon as it reports ready.
            "lane_not_ready:",
            "foreground_warmup_timeout",
            "warmup_deferred",
        )
    )


def _background_error_is_quiet(error: str) -> bool:
    normalized = str(error or "")
    return normalized in {
        "foreground_busy",
        "foreground_quiet_window",
        "client_returned_no_text",
        "cancelled_unhealthy",
        # The model lane REFUSED to start a worker. That is admission control
        # doing its job, not an endpoint being unreliable.
        #
        # It arrives from _ModelLoadAdmissionDeniedError, raised when the lane
        # controller cancels a spawn because the host has no room — the same
        # condition that produces background_deferred:memory_pressure two
        # lines down, reached by a different route. Counted as an endpoint
        # failure it opened the circuit breaker on a perfectly healthy
        # Brainstem: 128 "Circuit OPEN for Brainstem after N failures" and 128
        # "failed validation" in one sampled window, on a machine that was
        # simply full. Tripping a breaker for backpressure then keeps the
        # endpoint out AFTER the memory frees up.
        #
        # worker_died_during_generation is deliberately NOT here. A worker
        # that started and then died is a real event and stays loud.
        "candidate_worker_not_ready",
        "background_deferred:memory_pressure",
        "background_deferred:cortex_startup_quiet",
        "background_deferred:foreground_quiet_window",
        "background_deferred:cortex_resident",
        "background_deferred:cortex_failed",
        "background_deferred:foreground_reserved",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith((
        "background_deferred:",
        "mlx_runtime_unavailable:",
        "local_runtime_unavailable:",
        "request_queue_failed:",
    ))


def _declared_background_deferral_reason(result: Mapping[str, Any]) -> str:
    """Extract only explicit admission deferrals from a router result."""
    error = str(result.get("error", "") or "").strip()
    normalized = error.lower()
    if normalized.startswith("background_deferred:"):
        return error.split(":", 1)[1].strip() or "background_deferred"
    if bool(result.get("deferred", False)):
        return error or str(result.get("endpoint", "") or "background_deferred")
    if normalized in {
        "foreground_busy",
        "foreground_quiet_window",
        "desktop_background_local_disabled",
    }:
        return error
    if normalized.startswith("desktop_background_headroom:"):
        return error
    return ""


def _consume_deliberate_no_text_reason(client: Any) -> str:
    """Why the client last returned no text ON PURPOSE, if it did.

    The foreground path hands the router an InferenceGate rather than the MLX
    client itself, so look through the common wrapper attribute as well. Reading
    the reason clears it at the source, which is what keeps it scoped to the one
    turn that earned it.
    """
    for candidate in (client, getattr(client, "_mlx_client", None)):
        if candidate is None:
            continue
        consume = getattr(candidate, "consume_deliberate_no_text_reason", None)
        if not callable(consume):
            continue
        try:
            reason = consume()
        except (RuntimeError, AttributeError, TypeError):
            continue
        if reason:
            return str(reason)
    return ""


def _local_client_failure_reason(client: Any) -> str:
    def _get_declared_attr(candidate: Any, attr: str) -> Any:
        try:
            inspect.getattr_static(candidate, attr)
        except AttributeError:
            return None
        try:
            value = getattr(candidate, attr)
        except (RuntimeError, AttributeError, TypeError):
            return None
        if value is candidate:
            return None
        return value

    def _extract_lane_failure(candidate: Any) -> str:
        lane = None
        get_lane_status = _get_declared_attr(candidate, "get_lane_status")
        get_conversation_status = _get_declared_attr(candidate, "get_conversation_status")
        if callable(get_lane_status):
            lane = get_lane_status()
        elif callable(get_conversation_status):
            lane = get_conversation_status()

        if not isinstance(lane, dict):
            return ""

        state = str(lane.get("state", "") or "").strip().lower()
        error = str(
            lane.get("last_error", "")
            or lane.get("last_failure_reason", "")
            or ""
        )
        if state == "failed":
            return error or "lane_failed"

        conversation_ready = bool(lane.get("conversation_ready", False))
        # A lane in a transitional (not-ready) state must NOT receive a
        # foreground generation — submitting one blocks the caller on the
        # per-turn wall deadline (105s) while warmup contends for the single
        # worker. Route to the fallback ladder instead: turns stay fast AND
        # the cortex gets an uncontended window to finish warming. Lived
        # 2026-07-15: one turn exceeded 105s → force-abort recycled the
        # worker → every later turn hit the warming cortex, blocked 105s,
        # re-recycled it → the cortex could never re-warm under continuous
        # load (a busy Aura permanently lost its 32B). The previous
        # error-prefix allowlist missed 'foreground_warmup_timeout' /
        # 'warmup_deferred', so the recycled cortex was never skipped.
        if not conversation_ready and state in {
            "recovering",
            "spawning",
            "handshaking",
            "warming",
            "cold",
        }:
            return error or f"lane_not_ready:{state}"
        return ""

    try:
        seen: set[int] = set()
        candidate = client
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            failure = _extract_lane_failure(candidate)
            if failure:
                return failure

            next_candidate = None
            for attr in ("_client", "_mlx_client"):
                nested = _get_declared_attr(candidate, attr)
                if nested is not None:
                    next_candidate = nested
                    break
            candidate = next_candidate
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_router_degradation(
            exc,
            action="continued without local lane failure detail after client inspection failed",
        )
        logger.debug("Local client lane inspection failed: %s", exc)
    return ""


# ── Main Router ───────────────────────────────────────────────────────────────


class HealthMonitorShim:
    """Compatibility shim for legacy components expecting a health_monitor object."""
    def __init__(self, router: HealthAwareLLMRouter):
        self._router = router

    def is_healthy(self, name: str) -> bool:
        """Observer health check — must not consume half-open probe leases."""
        ep = self._router.endpoints.get(name)
        if not ep:
            return False
        return ep.peek_available()



#: Strips live measurements out of a deferral reason so repeats can be
#: recognised as repeats. "headroom:Reflex:66.6%/21.3GB(need <66.0% ...)" and
#: the same line at 66.5% are ONE ongoing condition, not two events.
_DEFERRAL_MEASUREMENT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|GB|MB|gb|mb|s\b)?")


def _deferral_reason_kind(reason: str) -> str:
    """The stable identity of a deferral reason, with the numbers removed."""
    return _DEFERRAL_MEASUREMENT_RE.sub("#", str(reason or "")).strip()


class HealthAwareLLMRouter:
    """
    Routes LLM requests to available endpoints with circuit breaking.

    Priority order: endpoints are tried in order of registration.
    Local MLX is prioritized as the final fallback.
    """

    def __init__(self):
        self.endpoints: dict[str, EndpointHealth] = {}
        self.health_monitor = HealthMonitorShim(self)
        self._lock = RobustLock("LLMHealthRouter.RouteLock")
        self._created_at = time.monotonic()
        self.high_pressure_mode: bool = False
        self.last_tier: str = "local"
        self.last_user_tier: str = "local"
        self.last_user_endpoint: str = PRIMARY_ENDPOINT
        self.last_endpoint: str | None = None
        self.last_background_endpoint: str | None = None
        self.last_background_tier: str | None = None
        self.last_user_error: str = ""
        self.last_background_error: str = ""
        self._last_generation_metadata: dict[str, Any] = {}
        self._generation_metadata_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_health_router_generation_metadata_{id(self)}",
                default=None,
            )
        )
        self._last_fallback_warning_at: float = 0.0
        self._background_deferral_log_state: dict[str, tuple[str, float, int]] = {}
        logger.info("HealthAwareLLMRouter initialized (Legacy-Compatible mode)")

    def _generation_metadata_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_generation_metadata_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_health_router_generation_metadata_{id(self)}",
                default=None,
            )
            self._generation_metadata_context = slot
        return slot

    def _publish_generation_metadata(self, metadata: dict[str, Any]) -> None:
        snapshot = dict(metadata)
        self._generation_metadata_slot().set(snapshot)
        self._last_generation_metadata = snapshot

    def get_last_generation_metadata(self) -> dict[str, Any]:
        task_metadata = self._generation_metadata_slot().get()
        if task_metadata is not None:
            return dict(task_metadata)
        return {}

    def get_diagnostic_last_generation_metadata(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        return dict(getattr(self, "_last_generation_metadata", {}) or {})

    def get_stats(self) -> dict[str, Any]:
        """Aggregate endpoint statistics for proprioceptive telemetry."""
        total_calls = 0
        total_tokens = 0
        total_failures = 0
        total_empty = 0
        endpoint_stats = {}
        for name, ep in self.endpoints.items():
            total_calls += ep.total_requests
            total_tokens += ep.total_tokens
            # Lifetime totals: failure_count is the CURRENT streak and resets
            # on success — summing it under-reported historical failures.
            total_failures += ep.lifetime_failure_count
            total_empty += ep.empty_responses
            endpoint_stats[name] = ep.status_dict()
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_failures": total_failures,
            "total_empty_responses": total_empty,
            "endpoint_count": len(self.endpoints),
            "last_tier": self.last_tier,
            "last_endpoint": self.last_endpoint,
            "last_user_error": self.last_user_error,
            "last_background_error": self.last_background_error,
            "high_pressure_mode": self.high_pressure_mode,
            "endpoints": endpoint_stats,
        }

    def is_ready(self) -> bool:
        """Deep readiness probe for runtime inference routing health."""
        if not self.endpoints:
            return False
        try:
            lane_audit = audit_lane_assignments()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="failed closed: llm router readiness could not audit lane assignments",
                severity="degraded",
            )
            return False
        if not bool(lane_audit.get("ok", True)):
            return False
        # Readiness is an OBSERVER: probe_eligible never mutates circuit
        # state, and an endpoint with neither a client nor a callable URL
        # cannot serve regardless of its circuit state.
        return any(
            ep.probe_eligible()
            for ep in self.endpoints.values()
            if str(getattr(ep, "name", "") or "").strip().lower() != "static-reflex"
            and (
                ep.client is not None
                or str(getattr(ep, "url", "") or "").startswith(("http://", "https://"))
            )
        )

    def force_release_generation_gate(self, reason: str = "hard_generation_deadline") -> bool:
        """Emergency release for watchdogs when a router call outlives its budget."""

        return force_release_generation_gate(reason=reason)

    def _soft_cancel_local_generations(self, *, reason: str) -> bool:
        """First rung of the preemption ladder: ask active local generations
        to yield between tokens. The MLX worker honors the cancel within one
        decode step and stays warm — no worker kill, no model reload.

        Returns True when at least one client accepted a cancel request.
        """
        try:
            from core.brain.llm.mlx_client import soft_cancel_active_generations

            receipts = soft_cancel_active_generations(reason=reason)
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "llm_health_router",
                exc,
                severity="warning",
                action="soft-cancel sweep unavailable; fell back to plain gate wait",
            )
            return False
        if receipts:
            logger.warning(
                "✋ [ROUTER] Soft-cancelled %d background generation(s) for a "
                "foreground turn (%s); model stays warm.",
                len(receipts),
                reason,
            )
        return bool(receipts)

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> int:
        """Abort stale router/model generation state from watchdog or saturation paths."""

        # All gate holders are dead once the workers are killed below —
        # reclaim every lease so the lane can heal (next attempt respawns
        # the worker) instead of wedging on a dead permit.
        aborted = 1 if force_release_generation_gate(reason=reason, release_all=True) else 0
        seen: set[int] = set()

        def _abort_client(client: Any) -> None:
            nonlocal aborted
            if client is None:
                return
            ident = id(client)
            if ident in seen:
                return
            seen.add(ident)
            abort = getattr(client, "force_abort_active_generation", None)
            if not callable(abort):
                return
            try:
                if abort(reason=reason):
                    aborted += 1
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued force-aborting other generation clients",
                    severity="degraded",
                )

        for endpoint in self.endpoints.values():
            _abort_client(getattr(endpoint, "client", None))
        try:
            from core.container import ServiceContainer

            _abort_client(ServiceContainer.get("inference_gate", default=None))
        except (ImportError, AttributeError, RuntimeError):
            pass
        return aborted

    def register(
        self,
        name: str,
        url: str,
        model: str,
        is_local: bool = False,
        tier: str = "local",
        client: Any = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> HealthAwareLLMRouter:
        name = normalize_endpoint_name(name) or str(name or "").strip()
        if not name:
            raise ValueError("endpoint registration requires a non-empty name")
        if not is_local:
            raise ValueError(
                "remote_model_provider_removed: remote model providers are not "
                f"supported (endpoint={name}, url={url})"
            )
        # Fail-safe parameter validation: a bad threshold must not create an
        # endpoint whose circuit can never open (or opens on every call).
        try:
            failure_threshold = max(1, int(failure_threshold))
        except (TypeError, ValueError):
            failure_threshold = 3
        try:
            recovery_timeout = float(recovery_timeout)
        except (TypeError, ValueError):
            recovery_timeout = 30.0
        if not math.isfinite(recovery_timeout) or recovery_timeout <= 0:
            recovery_timeout = 30.0

        existing = self.endpoints.get(name)
        if existing is not None:
            # Re-registration updates CONFIGURATION but preserves live circuit
            # state — replacing the EndpointHealth object silently reset an
            # OPEN circuit to CLOSED, bypassing the breaker entirely.
            with existing._lock:
                existing.url = url
                existing.model = model
                existing.is_local = is_local
                existing.tier = tier
                existing.client = client
                existing.failure_threshold = failure_threshold
                existing.recovery_timeout = recovery_timeout
            logger.info(
                "Re-registered endpoint %s (%s) tier=%s local=%s — circuit state preserved (%s)",
                name, model, tier, is_local, existing.state.value,
            )
            return self

        ep = EndpointHealth(
            name=name,
            url=url,
            model=model,
            is_local=is_local,
            tier=tier,
            client=client,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.endpoints[name] = ep
        logger.info("Registered endpoint: %s (%s) tier=%s local=%s", name, model, tier, is_local)
        return self

    def register_endpoint(self, ep_obj: Any) -> HealthAwareLLMRouter:
        """Compatibility method for Unified Cognitive Engine / AutonomousBrain."""
        # ep_obj is expected to have: name, tier, model_name, client
        name = normalize_endpoint_name(getattr(ep_obj, "name", "unknown")) or "unknown"
        tier_val = getattr(ep_obj, "tier", "local")
        declared_locality = getattr(ep_obj, "is_local", None)
        if isinstance(declared_locality, bool):
            is_local = declared_locality
        else:
            # Compatibility endpoints may omit the flag. Local in-process and
            # loopback endpoints are admitted; every other HTTP origin is
            # structurally rejected without vendor-specific fingerprinting.
            endpoint_url = str(getattr(ep_obj, "endpoint_url", "") or "")
            is_local = True
            if endpoint_url.lower().startswith(("http://", "https://")):
                from urllib.parse import urlparse

                host = str(urlparse(endpoint_url).hostname or "").lower()
                is_local = host in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                    "0.0.0.0",
                    "host.docker.internal",
                } or host.startswith(("127.", "192.168.", "10."))
        
        if not is_local:
            raise ValueError(
                "remote_model_provider_removed: remote model providers are not "
                f"supported (endpoint={name})"
            )

        # Normalize enum tiers and legacy API labels into host-local lanes.
        tier_name = tier_val
        if isinstance(tier_val, str):
            lowered = tier_val.lower()
            if lowered == "api_deep":
                tier_name = "local_deep"
            elif lowered == "api_fast":
                tier_name = "local_fast"
            elif lowered in ("local", "primary"):
                tier_name = "local"
            elif lowered in ("local_deep", "secondary"):
                tier_name = "local_deep"
            elif lowered in ("local_fast", "tertiary"):
                tier_name = "local_fast"
            elif lowered == "emergency":
                tier_name = "emergency"
        elif hasattr(tier_val, "value"):
            normalized = str(tier_val.value).lower()
            if normalized == "primary":
                tier_name = "local"
            elif normalized == "secondary":
                tier_name = "local_deep"
            elif normalized == "tertiary":
                tier_name = "local_fast"
            elif normalized == "emergency":
                tier_name = "emergency"

        model_name = getattr(ep_obj, "model_name", "unknown")
        
        return self.register(
            name=name,
            url="internal",
            model=model_name,
            is_local=True,
            tier=tier_name,
            client=getattr(ep_obj, "client", None)
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 120.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Try each endpoint in order. Return first valid response as a string.
        Falls back to local if all remote endpoints fail.
        GUARANTEE: Never returns empty string — provides diagnostic fallback.
        """
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        res = await self.generate_with_metadata(
            prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
        )
        # String-surface consumers lose the structured ok flag; publish the
        # full result so they can consult get_last_generation_metadata()
        # instead of persisting a diagnostic string as model output.
        if isinstance(res, dict):
            self._publish_generation_metadata(res)
        text = res.get("text", "")
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        is_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
            explicit_foreground=explicit_foreground,
        )

        if is_background and _background_error_is_quiet(str(res.get("error", "") or "")):
            return ""

        if benchmark_request and (not text or not text.strip()):
            return ""
        
        # RESPONSE GUARANTEE: Never return empty
        if not text or not text.strip():
            if is_background:
                return ""
            error = res.get("error", "unknown")
            endpoint = res.get("endpoint", "none")
            logger.error(
                "⚠️ [LLM ROUTER] All endpoints exhausted. Last error: %s (endpoint: %s)",
                error, endpoint
            )
            # Receipt for string consumers: this text is a surface fallback,
            # not model output — get_last_generation_metadata() carries the flag.
            res["string_surface_fallback"] = True
            self._publish_generation_metadata(res)
            if str(error or "").strip() == "client_returned_no_text":
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            # v10.5 HARDENING: Return a diagnostic label so StructuredLLM can report it accurately
            # instead of a silent empty string.
            return f"ROUTER_ERROR: {error} (at {endpoint})"
        
        return text

    async def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 180.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Try each endpoint in order. Return first valid response with full metadata.
        Falls back to local if all remote endpoints fail.
        Always returns a dict: {"ok": bool, "text": str, "endpoint": str, "tokens": int}
        """
        admission_started = time.monotonic()
        # Set by think_and_act's fallback when this coroutine is already
        # running INSIDE a held generation-gate lease (its own or the gated
        # caller's). Acquiring again would self-deadlock the process gate.
        gate_already_held = bool(kwargs.pop("_gate_already_held", False))
        if gate_already_held:
            return await self._generate_with_metadata_gated(
                prompt,
                system_prompt=system_prompt,
                timeout=timeout,
                prefer_tier=prefer_tier,
                schema=schema,
                **kwargs,
            )
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(
            kwargs.get("foreground_request", False)
            or kwargs.get("health_probe", False)
            or kwargs.get("protected_foreground_lane", False)
            or kwargs.get("proof_primary_lane_required", False)
        )
        # Default-purpose normalization must happen BEFORE classification:
        # the gated implementation stamps unlabelled chat calls with
        # purpose="expression" (user-facing), but classification here ran
        # first — so a bare generate_with_metadata() call with no origin,
        # purpose, or flags was admitted, suppressed, and budgeted as
        # BACKGROUND and only later treated as a user-facing turn.
        if (
            not origin
            and not purpose
            and not explicit_background
            and not bool(kwargs.get("_non_chat_inference", False))
        ):
            purpose = "expression"
            kwargs["purpose"] = purpose
        request_is_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if request_is_background:
            foreground_owner = _active_foreground_generation_owner()
            if foreground_owner:
                return _generation_gate_busy_result(foreground_owner)

        early_deferral = self._background_suppression_result(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if early_deferral is not None:
            return early_deferral

        if request_is_background:
            acquired = await _acquire_generation_gate_slot(
                min(_GENERATION_GATE_WAIT_S, _BACKGROUND_GENERATION_GATE_WAIT_S)
            )
        else:
            # Foreground preemption ladder. A user turn must not sit the full
            # gate window behind a BACKGROUND generation and then pay a
            # worker-kill + model reload (observed live: conversation lane
            # cold for 75s, then force-abort). Rung 1: short grace. Rung 2:
            # cooperative soft-cancel of a background holder — the worker
            # yields between tokens and stays warm, freeing the gate in
            # about one decode step. Rung 3: the remaining wait and the
            # existing force-abort escalation below, unchanged.
            acquired = await _acquire_generation_gate_slot(_FOREGROUND_GATE_GRACE_S)
            if not acquired:
                holder = _oldest_generation_gate_lease()
                holder_owner = holder[2] if holder is not None else ""
                if holder is not None and _generation_owner_is_user_foreground(
                    holder_owner
                ):
                    holder_age_s = max(0.0, time.time() - float(holder[1]))
                    holder_has_time = _generation_gate_lease_has_time(holder[0])
                    if holder_has_time is True or (
                        holder_has_time is None
                        and holder_age_s < max(30.0, _GENERATION_GATE_WAIT_S)
                    ):
                        return _generation_gate_busy_result(holder_owner)
                elif holder is not None:
                    if self._soft_cancel_local_generations(
                        reason=f"foreground_preempts_background:{holder_owner[:80]}"
                    ):
                        acquired = await _acquire_generation_gate_slot(
                            _FOREGROUND_SOFT_CANCEL_WAIT_S
                        )
                if not acquired:
                    remaining_wait = max(
                        1.0, _GENERATION_GATE_WAIT_S - _FOREGROUND_GATE_GRACE_S
                    )
                    acquired = await _acquire_generation_gate_slot(remaining_wait)
        if not acquired:
            if request_is_background:
                holder = _oldest_generation_gate_lease()
                holder_owner = holder[2] if holder is not None else "unknown"
                logger.info(
                    "⏸️ Router: Background generation deferred behind active gate owner=%s.",
                    holder_owner[:120],
                )
                return _background_generation_gate_deferred_result(holder_owner)

            foreground_owner = _active_foreground_generation_owner()
            foreground_age_s = _oldest_generation_gate_lease_age_s() if foreground_owner else 0.0
            foreground_lease = _oldest_generation_gate_lease() if foreground_owner else None
            foreground_has_time = (
                _generation_gate_lease_has_time(foreground_lease[0])
                if foreground_lease is not None
                else None
            )
            if foreground_owner and (
                foreground_has_time is True
                or (
                    foreground_has_time is None
                    and foreground_age_s < max(30.0, _GENERATION_GATE_WAIT_S)
                )
            ):
                return _generation_gate_busy_result(foreground_owner)
            # An over-age holder is ABANDONED: its route already gave up and
            # returned, the decode is orphaned. Cooperative cancel FIRST — the
            # worker yields between tokens and stays warm. The 20260708-final
            # soak proved what skipping this rung costs: the earlier ladder
            # only soft-cancelled BACKGROUND holders, so an orphaned
            # foreground turn went straight to force-abort, which kills the
            # 20GB worker — every ~5min: orphan holds gate 75s → kill → cold
            # reload → next turn meets the next orphan. 34/38 turns dead.
            if not request_is_background and self._soft_cancel_local_generations(
                reason=f"abandoned_gate_holder:{(foreground_owner or 'unknown')[:80]}"
            ):
                acquired = await _acquire_generation_gate_slot(
                    _FOREGROUND_SOFT_CANCEL_WAIT_S
                )
            if not acquired:
                aborted = self.force_abort_active_generation(
                    reason=f"generation_gate_wait_timeout:{_GENERATION_GATE_WAIT_S:.1f}s"
                )
                if aborted:
                    acquired = await _acquire_generation_gate_slot(2.0)
        if not acquired:
            request_scope = "background" if request_is_background else "foreground"
            record_degradation(
                "llm_health_router",
                RuntimeError("generation gate saturated"),
                severity="degraded",
                action=(
                    f"refused to stack another {request_scope} concurrent generation; "
                    f"origin={origin or 'unknown'} purpose={purpose or 'unknown'}"
                ),
            )
            return dict(_GATE_SATURATION_RESULT)
        try:
            lease_timeout_s = max(5.0, float(timeout)) + 10.0
        except (TypeError, ValueError, OverflowError):
            lease_timeout_s = 190.0
        lease_id = _mark_generation_gate_acquired(
            _generation_gate_owner(origin, purpose),
            timeout_s=lease_timeout_s,
        )
        try:
            # One end-to-end deadline: admission (gate grace, soft-cancel,
            # remaining-wait, abort-retry) already consumed part of the
            # caller's budget — the downstream dispatch must not receive the
            # UNCHANGED timeout on top of it.
            try:
                total_budget = float(timeout)
            except (TypeError, ValueError):
                total_budget = 180.0
            if not math.isfinite(total_budget) or total_budget <= 0.0:
                total_budget = 180.0
            admission_elapsed = time.monotonic() - admission_started
            remaining_budget = total_budget - admission_elapsed
            if remaining_budget <= 0.0:
                record_degradation(
                    "llm_health_router",
                    TimeoutError(
                        f"admission consumed {admission_elapsed:.1f}s of a "
                        f"{total_budget:.1f}s budget"
                    ),
                    severity="degraded",
                    action="refused dispatch with an exhausted end-to-end budget",
                )
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "admission_deadline_exhausted",
                    "tokens": 0,
                    "error": (
                        f"admission_deadline_exhausted:{admission_elapsed:.1f}s"
                        f"/{total_budget:.1f}s"
                    ),
                    "provider": "none",
                    "model": "",
                    "is_local": False,
                    "fallback_chain": [],
                }
            if request_is_background:
                # Domain-specialist weights for background reasoning lanes:
                # if the expert-LoRA library has a match, swap it onto the
                # resident primary model. This runs INSIDE the held lease —
                # a potentially 20-second resident-model mutation before
                # owning generation capacity let a foreground turn start
                # mid-swap — and its cost counts against this request's
                # end-to-end budget, not outside it.
                await self._maybe_route_expert_adapter(prompt, kwargs)
                admission_elapsed = time.monotonic() - admission_started
                remaining_budget = total_budget - admission_elapsed
                if remaining_budget <= 0.0:
                    return {
                        "ok": False,
                        "text": "",
                        "endpoint": "admission_deadline_exhausted",
                        "tokens": 0,
                        "error": (
                            f"admission_deadline_exhausted:{admission_elapsed:.1f}s"
                            f"/{total_budget:.1f}s"
                        ),
                        "provider": "none",
                        "model": "",
                        "is_local": False,
                        "fallback_chain": [],
                    }
            return await self._generate_with_metadata_gated(
                prompt,
                system_prompt=system_prompt,
                timeout=remaining_budget,
                prefer_tier=prefer_tier,
                schema=schema,
                **kwargs,
            )
        finally:
            _release_generation_gate_after_call(lease_id)

    async def _generate_with_metadata_gated(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 180.0,  # noqa: ASYNC109 - inherited budget semantics.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        _contract_tool_handoff_val = kwargs.pop("_contract_tool_handoff", False)
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        non_chat_inference = bool(kwargs.pop("_non_chat_inference", False))
        if non_chat_inference:
            # Carried on rather than consumed here. The inference gate is where
            # the user-surface reply contract is applied, and without this it
            # cannot tell an internal deliberation from the visible answer:
            # every caller that prefers the primary tier was treated as the
            # reply lane and graded against a question invented from its own
            # prompt.
            kwargs["internal_inference"] = True
        if not origin and not purpose and not explicit_background and not non_chat_inference:
            purpose = "expression"
            kwargs["purpose"] = purpose
        inferred_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        state = kwargs.pop("state", None)
        skip_runtime_payload = bool(kwargs.pop("skip_runtime_payload", False))
        contract: ResponseContract | None = None
        prepared_messages = kwargs.get("messages")
        _runtime_state = state
        if skip_runtime_payload:
            if prepared_messages is not None and system_prompt:
                prepared_messages = _merge_system_prompt(prepared_messages, system_prompt)
                kwargs["messages"] = prepared_messages
            elif prepared_messages is None:
                kwargs.pop("messages", None)
            if (not prompt) and prepared_messages is not None:
                prompt, inferred_system_prompt = self._coerce_prompt_from_messages(prepared_messages)
                if not system_prompt and inferred_system_prompt:
                    system_prompt = inferred_system_prompt
        else:
            prompt, system_prompt, prepared_messages, contract, _runtime_state = await prepare_runtime_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=kwargs.get("messages"),
                state=state,
                origin=origin,
                is_background=inferred_background,
            )
            if prepared_messages is not None:
                kwargs["messages"] = prepared_messages
            else:
                kwargs.pop("messages", None)

        if should_force_tool_handoff(contract, is_background=inferred_background) and not _contract_tool_handoff_val:
            tools = build_agentic_tool_map(
                contract.required_skill if contract else None,
                objective=prompt,
                max_tools=getattr(contract, "max_tools", 8) if contract else 8,
            )
            # Whether a turn was offered its tools is not otherwise visible
            # anywhere: the worker logs "Rendering native chat/tool template"
            # for every templated generation, with or without tools, so the
            # one line that looked like evidence was not. Diagnosing a turn
            # that should have called a tool and did not starts here.
            logger.info(
                "🔧 Tool handoff: skill=%s offered=%s",
                str(getattr(contract, "required_skill", "") or "?"),
                ",".join(sorted(tools)) if tools else "NONE",
            )
            if tools:
                handoff_kwargs = dict(kwargs)
                handoff_kwargs.pop("origin", None)
                handoff_kwargs.pop("is_background", None)
                handoff_kwargs.pop("_contract_tool_handoff", None)
                result = await self.think_and_act(
                    objective=prompt,
                    system_prompt=system_prompt or "",
                    tools=tools,
                    context={"response_contract": contract.to_dict()} if contract else {},
                    prefer_tier=prefer_tier,
                    origin=origin or "user",
                    is_background=False,
                    _contract_tool_handoff=True,
                    **handoff_kwargs,
                )
                text = str(result.get("content", "") or "").strip()
                called = result.get("tool_calls") or []
                if text and called:
                    return {
                        "ok": True,
                        "text": text,
                        "endpoint": "contract_tool_handoff",
                        "tokens": len(text.split()),
                        "error": "",
                    }
                if not text:
                    return {
                        "ok": False,
                        "text": "",
                        "endpoint": "contract_tool_handoff",
                        "tokens": 0,
                        "error": "grounding_required_no_tool_result",
                    }
                # Text, but the model never called the tool it was handed.
                #
                # This returned that prose as a success. The handoff exists
                # because the turn cannot be answered without the capability,
                # so an answer produced without it is ungrounded by
                # construction — and it also skipped `_generate_core`, where
                # the user-facing integrity checks live. Live 2026-08-19, that
                # is how "Output: 7" reached the screen with nothing executed:
                # the tool was offered, declined, and the invention served
                # without ever meeting the gate that exists to catch it.
                #
                # Falling through re-answers on the ordinary lane, which does
                # run those checks. Costs one generation on a turn the model
                # ignored its tool; the alternative is serving the invention.
                record_degradation(
                    "llm_health_router.tool_handoff",
                    RuntimeError("model answered without calling the offered tool"),
                    severity="info",
                    action="re-answered on the ordinary lane so integrity checks apply",
                    enforce_failure_policy=False,
                )
        from core.consciousness.state_freeze import state_freeze
        async with state_freeze():
            return await self._generate_core(
                prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
            )

    async def think(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> str | None:
        """
        Unified interface for non-chat callers. Routes through the health-aware
        endpoint selection, then normalises to Optional[str].
        [FIX #1-Harden] Supports 'messages' keyword for cognitive pipeline compatibility.
        """
        self._publish_generation_metadata({})
        kwargs.pop("_contract_tool_handoff", False)
        if not prompt and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        if not prompt:
            logger.warning("[LLMRouter.think] Called without prompt or messages.")
            return None
        try:
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or "",
                prefer_tier=prefer_tier,
                schema=schema,
                _non_chat_inference=True,
                **kwargs,
            )
            if isinstance(result, dict):
                self._publish_generation_metadata(result)
                origin = str(kwargs.get("origin", "") or "").lower()
                is_background = self._is_background_request(
                    origin=origin,
                    purpose=str(kwargs.get("purpose", "") or "").lower(),
                    explicit_background=bool(kwargs.get("is_background", False)),
                    explicit_foreground=bool(kwargs.get("foreground_request", False))
                    or bool(kwargs.get("health_probe", False)),
                )
                if is_background:
                    deferral_reason = _declared_background_deferral_reason(result)
                    if deferral_reason:
                        record_deferral(
                            origin=origin or "background",
                            reason=deferral_reason,
                        )
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            strict_answer_request = "<answer>" in str(prompt or "").lower() or "<answer>" in str(
                system_prompt or ""
            ).lower()
            # GUARD: Never call .strip() on None
            if text is None:
                if (
                    isinstance(result, dict)
                    and str(result.get("error", "") or "").strip() == "client_returned_no_text"
                    and not self._is_background_request(
                        origin=str(kwargs.get("origin", "") or "").lower(),
                        purpose=str(kwargs.get("purpose", "") or "").lower(),
                        explicit_background=bool(kwargs.get("is_background", False)),
                        explicit_foreground=bool(kwargs.get("foreground_request", False))
                        or bool(kwargs.get("health_probe", False)),
                    )
                ):
                    if strict_answer_request or kwargs.get("_non_chat_inference"):
                        return None
                    return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
                return None
            stripped = text.strip()
            if stripped:
                return stripped
            if (
                isinstance(result, dict)
                and str(result.get("error", "") or "").strip() == "client_returned_no_text"
                and not self._is_background_request(
                    origin=str(kwargs.get("origin", "") or "").lower(),
                    purpose=str(kwargs.get("purpose", "") or "").lower(),
                    explicit_background=bool(kwargs.get("is_background", False)),
                    explicit_foreground=bool(kwargs.get("foreground_request", False))
                    or bool(kwargs.get("health_probe", False)),
                )
            ):
                if strict_answer_request or kwargs.get("_non_chat_inference"):
                    return None
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            # [STABILITY v55] Don't mask failures with robot responses.
            # Return None so the caller can retry or fallback properly.
            return None
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            _record_router_degradation(
                exc,
                action="returned no router thought after endpoint generation failed",
                severity="degraded",
            )
            logger.warning("[LLMRouter.think] Failed: %s", exc)
            return None

    async def classify(
        self,
        prompt: str,
        system_prompt: str | None = None,
        prefer_tier: str = "primary",
        **kwargs
    ) -> str:
        """
        Hardened Intent Classification.
        Forces the LLM to return ONLY a single intent token.
        """
        classification_system_prompt = (
            "You are an intent classifier for Aura. Respond ONLY with one of the following tokens:\n"
            "- technical: coding, debugging, architecture, math, logic, research\n"
            "- philosophical: identity, morality, existence, consciousness\n"
            "- emotional: feelings, mood, empathy, personal reflection\n"
            "- planning: list of tasks, project management, goal setting\n"
            "- critical: security audits, performance bottlenecks, vulnerability scans\n"
            "- casual: greetings, small talk, status checks\n\n"
            "Do not explain. Do not use punctuation. Just output the single word."
        )

        try:
            deterministic = self._deterministic_intent_classification(prompt)
            if deterministic:
                logger.info("🧭 Intent classification resolved deterministically: %s", deterministic)
                return deterministic

            # We use generate_with_metadata directly to ensure strict parameters
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or classification_system_prompt,
                max_tokens=10,
                temperature=0.0,
                prefer_tier=prefer_tier,
                purpose="classification",
                **kwargs
            )
            
            text = result.get("text", "").strip().lower()
            # Clean any stray punctuation
            import re
            text = re.sub(r'[^a-z_]', '', text)

            if not text:
                logger.warning("⚠️ Intent classification returned empty. Defaulting to 'casual'.")
                return "casual"

            allowed_labels = {
                "technical",
                "philosophical",
                "emotional",
                "planning",
                "critical",
                "casual",
            }
            if text in allowed_labels:
                return text
            # The contract promises exactly one of six tokens. Concatenated
            # or invented labels previously propagated as-is, creating false
            # classification receipts downstream. A response that BEGINS
            # with a valid label ("technicalexplanation") still names it.
            for label in allowed_labels:
                if text.startswith(label):
                    logger.warning(
                        "⚠️ Intent classifier returned non-token output %r; using leading label %r.",
                        text[:60],
                        label,
                    )
                    return label
            _record_router_degradation(
                ValueError(f"unrecognized intent label: {text[:60]}"),
                action="defaulted intent classification to casual after unrecognized label",
                severity="degraded",
            )
            return "casual"
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_router_degradation(
                e,
                action="defaulted intent classification to casual after classifier failed",
                severity="degraded",
            )
            logger.error("❌ Intent classification failed: %s. Defaulting to 'casual'.", e)
            return "casual"

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str = "",
        tools: dict[str, Any] | None = None,
        max_turns: int = 5,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        # True when the gated generate path invoked us while HOLDING its
        # generation-gate lease: we must not try to re-acquire the same
        # process-wide gate (self-deadlock), and our fallback think() call
        # must dispatch inside the caller's lease for the same reason.
        called_from_gated = bool(kwargs.pop("_contract_tool_handoff", False))
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
            explicit_foreground=bool(kwargs.get("foreground_request", False))
            or bool(kwargs.get("health_probe", False)),
        )
        state = kwargs.pop("state", None)
        objective, system_prompt, prepared_messages, contract, runtime_state = await prepare_runtime_payload(
            prompt=objective,
            system_prompt=system_prompt,
            messages=kwargs.get("messages"),
            state=state,
            origin=origin,
            is_background=is_bg,
        )
        if prepared_messages is not None:
            kwargs["messages"] = prepared_messages
        else:
            kwargs.pop("messages", None)
        prefer_tier = self._normalize_prefer_tier(kwargs.get("prefer_tier"))
        agent_context = dict(context or {})
        if contract:
            agent_context.setdefault("response_contract", contract.to_dict())
        if prepared_messages is not None:
            agent_context.setdefault("messages", prepared_messages)
        if contract:
            max_turns = min(max_turns, max(1, int(getattr(contract, "max_tool_turns", max_turns) or max_turns)))

        preferred_names = self._fallback_endpoint_names(
            prefer_tier or "primary",
            False,
            is_background=is_bg,
        )
        # probe_eligible: candidate ENUMERATION must not consume half-open
        # probe leases; the mutating is_available admission runs per-endpoint
        # immediately before dispatch below.
        available = [ep for ep in self.endpoints.values() if ep.probe_eligible()]
        ordered: list[EndpointHealth] = []
        seen = set()
        for name in preferred_names:
            ep = self.endpoints.get(name)
            if ep and ep.probe_eligible():
                ordered.append(ep)
                seen.add(ep.name)
        for ep in available:
            if ep.name not in seen:
                ordered.append(ep)

        def _call_kwargs(method: Any) -> dict[str, Any]:
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return dict(kwargs)

            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                return dict(kwargs)

            return {key: value for key, value in kwargs.items() if key in sig.parameters}

        # One wall-clock deadline for the whole tool-capable route: the
        # public path previously called endpoint clients with NO timeout at
        # all, so a wedged client held the caller (and now the gate) forever.
        try:
            route_budget_s = float(kwargs.get("timeout", 180.0))
        except (TypeError, ValueError):
            route_budget_s = 180.0
        if not math.isfinite(route_budget_s) or route_budget_s <= 0.0:
            route_budget_s = 180.0

        # The public tool-capable route must own the same process-wide
        # generation lane as every routed generation — it previously drove
        # GB-scale local inference completely outside generation admission.
        lease_id: int | None = None
        if not called_from_gated:
            lease_id = await acquire_external_generation_gate_lease(
                owner=_generation_gate_owner(origin or "tool_route", purpose or "think_and_act"),
                timeout_s=route_budget_s + 10.0,
                wait_s=(
                    _BACKGROUND_GENERATION_GATE_WAIT_S
                    if is_bg
                    else _GENERATION_GATE_WAIT_S
                ),
            )
            if lease_id is None:
                holder = _oldest_generation_gate_lease()
                holder_owner = holder[2] if holder is not None else "unknown"
                logger.info(
                    "⏸️ Router: tool-capable route deferred behind gate owner=%s.",
                    holder_owner[:120],
                )
                return {
                    "content": "",
                    "turns": 0,
                    "tool_calls": [],
                    "error": "generation_gate_saturated",
                    "deferred": True,
                }
        route_deadline = time.monotonic() + route_budget_s

        try:
            for ep in ordered:
                if is_bg and self._tier_is_background_only(self._tier_name(ep)) is False and not kwargs.get("prefer_endpoint"):
                    continue
                client = ep.client
                if not client or not hasattr(client, "think_and_act"):
                    continue
                remaining_s = route_deadline - time.monotonic()
                if remaining_s <= 0.0:
                    logger.warning(
                        "think_and_act route deadline exhausted (%.1fs) before %s.",
                        route_budget_s,
                        ep.name,
                    )
                    break
                # Admission check at dispatch time — grants the half-open probe
                # lease only to the endpoint we actually call.
                if not ep.is_available():
                    continue
                call_started = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        client.think_and_act(
                            objective,
                            system_prompt=system_prompt,
                            tools=tools,
                            max_turns=max_turns,
                            context=agent_context,
                            **_call_kwargs(client.think_and_act),
                        ),
                        timeout=remaining_s,
                    )
                    text = str((result or {}).get("content", "") or "").strip()
                    if text:
                        ep.record_success(
                            len(text.split()),
                            (time.monotonic() - call_started) * 1000,
                        )
                        self.last_tier = ep.tier
                        self.last_endpoint = ep.name
                        if is_bg:
                            self.last_background_endpoint = ep.name
                            self.last_background_tier = ep.tier
                        else:
                            self.last_user_endpoint = ep.name
                            self.last_user_tier = ep.tier
                        return result
                    # Empty content is visible telemetry but NOT a circuit
                    # failure: "no tool result" legitimately falls through to
                    # the plain think() route below.
                    with ep._lock:
                        ep.empty_responses += 1
                except TimeoutError as exc:
                    _record_router_degradation(
                        exc,
                        action="recorded tool-route endpoint timeout and continued fallback",
                        severity="error",
                    )
                    logger.warning(
                        "think_and_act on %s timed out after %.1fs.",
                        ep.name,
                        time.monotonic() - call_started,
                    )
                    ep.record_failure(f"think_and_act_timeout:{ep.name}")
                except _ROUTER_CLIENT_ERRORS as exc:
                    _record_router_degradation(
                        exc,
                        action="recorded endpoint failure and continued tool-capable route fallback",
                        severity="degraded",
                    )
                    logger.warning("think_and_act on %s failed: %s", ep.name, exc)
                    ep.record_failure(str(exc))

            kwargs_clean = dict(kwargs)
            kwargs_clean.pop("_contract_tool_handoff", None)
            # Either we hold a lease (public path) or our caller does (gated
            # handoff): the fallback think() must dispatch inside that lease
            # instead of waiting on the gate it can never acquire.
            text = await self.think(
                objective,
                system_prompt=system_prompt,
                state=runtime_state,
                _contract_tool_handoff=True,
                _gate_already_held=True,
                **kwargs_clean,
            )
            return {"content": text or "", "turns": 0, "tool_calls": []}
        finally:
            if lease_id is not None:
                release_external_generation_gate_lease(lease_id)

    async def _get_mycelial_direction(self, prompt: str) -> dict[str, Any] | None:
        """Query Mycelium for routing guidance (v31)."""
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelium", default=None)
            if not mycelium:
                return None
            
            # 1. Match hardwired pathways
            # v42 FIX: Skip large prompts (likely background tasks/logs) to avoid false 'null' matches
            if len(prompt) > 100 or "say 'null'" in prompt.lower():
                return None
                
            match_res = mycelium.match_hardwired(prompt)
            if match_res:
                pathway, _params = match_res
                # If pathway exists, it's a strong signal
                # For now, we look for 'brain_tier' or 'route' in description or custom logic
                # Optimization: check if description has routing tags
                desc = pathway.description.lower()
                if "local-only" in desc or "private" in desc:
                    return {"tier_preference": "local"}
                if "heavy" in desc:
                    return {"tier_preference": "local", "deep_handoff": True}
                
                return {"pathway_id": pathway.pathway_id}
            return None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="continued routing without mycelial direction after guidance lookup failed",
            )
            return None

    def _flatten_messages_for_local_model(self, messages: list[dict[str, str]], require_json: bool) -> str:
        """Flatten messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages, require_json=require_json)

    @staticmethod
    def _transport_carries_messages(client: Any) -> bool:
        """Will this client's entry point actually receive `messages`?

        Signature-based, matching how `_call_kwargs` filters the payload: a
        method taking **kwargs receives everything, a method that names
        `messages` receives it, and anything else does not.
        """
        for attribute in ("think", "call", "generate_text_async", "generate"):
            method = getattr(client, attribute, None)
            if not callable(method):
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return True
            if any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            ):
                return True
            return "messages" in sig.parameters
        return False

    @staticmethod
    def _coerce_prompt_from_messages(messages: Any) -> tuple[str, str | None]:
        """Serialize a full OpenAI-style message list into prompt/system fields.

        This keeps the health-aware router aligned with the legacy router so
        callers can pass rich conversational state without it being collapsed
        down to only the last user turn.
        """
        if not messages or not isinstance(messages, list):
            return "", None

        system_parts: list[str] = []
        convo_parts: list[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                convo_parts.append(str(msg))
                continue

            role = str(msg.get("role", "") or "").strip().lower()
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role in {"user", "human"}:
                convo_parts.append(f"User: {content}")
            elif role in {"assistant", "aura"}:
                convo_parts.append(f"Aura: {content}")
            else:
                convo_parts.append(f"[{role or 'message'}]: {content}")

        prompt = "\n".join(convo_parts).strip()
        system_prompt = "\n\n".join(system_parts).strip() or None
        return prompt, system_prompt

    @staticmethod
    def _normalize_prefer_tier(prefer_tier: Any | None) -> str | None:
        if prefer_tier is None:
            return None
        if not isinstance(prefer_tier, str):
            if hasattr(prefer_tier, "value"):
                prefer_tier = prefer_tier.value
            else:
                prefer_tier = str(prefer_tier)

        tier = prefer_tier.lower()
        aliases = {
            "local": "primary",
            "local_deep": "secondary",
            "local_fast": "tertiary",
            "fast": "tertiary",
            "deep": "secondary",
        }
        return aliases.get(tier, tier)

    @staticmethod
    def _origin_tokens(origin: str | None) -> set[str]:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        return {token for token in normalized.split("_") if token}

    @classmethod
    def _is_user_facing_origin(cls, origin: str | None) -> bool:
        tokens = cls._origin_tokens(origin)
        return bool(tokens & _USER_FACING_ORIGINS)

    async def _maybe_route_expert_adapter(self, prompt: str, kwargs: Mapping[str, Any]) -> None:
        """Attach the best domain-specialist LoRA before a background dispatch.

        The expert-LoRA library keeps specialist adapters on disk; when a
        background reasoning request matches one, it is swapped onto the
        RESIDENT primary model in the worker (seconds, no reload).

        Default-ON. This is the only path by which Aura's own learned weight
        deltas reach the model that answers, and shipping it off meant the
        answer to "does anything she learned change what she says?" was
        structurally no — not measured-and-rejected, just never in the lane.
        A capability disabled by default is a capability that cannot be
        measured, and this repository's own standard is that unmeasured is
        not a verdict.
        Turning it on is safe on its own terms because the path is
        refusal-safe end to end: selection is in-memory, an actual swap
        happens only on adapter change while the lane is idle, the client
        refuses busy lanes, the swap carries a 20s budget, and every failure
        mode falls back to the resident weights with a degradation receipt.
        The kill switch remains — AURA_EXPERT_LORA_ROUTING=0 restores the old
        behaviour for crash-loop recovery.
        """
        if str(
            os.environ.get("AURA_EXPERT_LORA_ROUTING", "1")
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            from core.container import ServiceContainer

            library = ServiceContainer.get("expert_lora_library", default=None)
            if library is None:
                return
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            if client is None:
                return
            applier = getattr(self, "_expert_adapter_applier", None)
            if applier is None or getattr(applier, "_client", None) is not client:
                from core.brain.llm.expert_adapter_applier import MLXExpertAdapterApplier

                applier = MLXExpertAdapterApplier(client)
                self._expert_adapter_applier = applier
            task_type = str(kwargs.get("task_type") or kwargs.get("domain") or "").strip()
            await asyncio.wait_for(
                library.select_and_activate_async(
                    str(prompt or ""),
                    task_type,
                    applier,
                    base_model=str(getattr(client, "model_path", "") or ""),
                ),
                timeout=20.0,
            )
        except TimeoutError:
            record_degradation(
                "expert_lora_routing",
                TimeoutError("adapter swap exceeded 20s budget"),
                action="dispatched request on resident weights without specialist adapter",
                severity="info",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "expert_lora_routing",
                exc,
                action="dispatched request on resident weights without specialist adapter",
                severity="info",
            )

    @classmethod
    def _is_background_request(
        cls,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> bool:
        if explicit_background:
            return True
        if explicit_foreground:
            return False

        normalized_purpose = str(purpose or "").strip().lower()
        if normalized_purpose in _USER_FACING_PURPOSES:
            return False

        tokens = cls._origin_tokens(origin)
        if not tokens:
            return normalized_purpose not in _USER_FACING_PURPOSES

        if tokens & _USER_FACING_ORIGINS:
            return False

        # Hardened default: anything that is not explicitly user-facing is
        # background. This prevents internal/kernel/autonomous traffic with
        # weak or unfamiliar origins from contaminating the foreground lane.
        if tokens & _BACKGROUND_ORIGIN_HINTS:
            return True

        return True

    def _background_suppression_result(
        self,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> dict[str, Any] | None:
        """Return a suppression result before scarce generation capacity is acquired."""

        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if not is_bg:
            return None
        # Explicit tool compositions (composing a message to another AI in a live
        # web-interlocutor conversation) are foreground work the user asked for —
        # NOT deferrable background chatter. Serving them (rather than deferring
        # under foreground_quiet_window) is what lets her actually compose each
        # turn instead of the composition coming back empty and falling to a
        # canned default. Reply gates stay off because origin is non-user-facing.
        if str(origin or "").strip().lower().replace("-", "_") == "web_interlocutor":
            return None

        reason = ""
        try:
            from core.runtime.background_policy import (
                THOUGHT_BACKGROUND_POLICY,
                background_activity_reason,
            )

            reason = str(
                background_activity_reason(
                    None,
                    profile=THOUGHT_BACKGROUND_POLICY,
                    allow_no_user_anchor=True,
                )
                or ""
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_router_degradation(
                exc,
                action="deferred background routing because background policy was unavailable",
                severity="degraded",
            )
            logger.warning("Background router policy probe failed: %s", exc)
            reason = "background_policy_unavailable"
        if not reason:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    reason = str(gate._background_local_deferral_reason(origin=origin) or "")
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
        if not reason and self._foreground_quiet_window_active():
            reason = "foreground_quiet_window"
        if not reason and getattr(self, "high_pressure_mode", False):
            reason = "memory_pressure"
        if not reason and (
            self._foreground_user_turn_active() or self._foreground_owner_active()
        ):
            reason = "foreground_busy"

        if not reason:
            return None

        self._log_background_deferral(
            scope="generation_gate",
            origin=origin,
            reason=reason,
        )
        return {
            "ok": False,
            "text": "",
            "endpoint": "suppressed",
            "tokens": 0,
            "error": f"background_deferred:{reason}",
        }

    def _log_background_deferral(
        self,
        scope: str,
        origin: str,
        reason: str,
        endpoint: str | None = None,
    ) -> None:
        """Log repeated background deferrals as a state, not as a feed flood."""
        key = f"{scope}:{endpoint or '*'}:{origin or '*'}"
        now = time.monotonic()
        previous_reason, previous_at, suppressed = self._background_deferral_log_state.get(
            key,
            ("", 0.0, 0),
        )
        # Compare the CAUSE, not the instantaneous measurement. The reason
        # carries live numbers —
        # "desktop_background_headroom:Reflex:66.6%/21.3GB(need <66.0% ...)" —
        # so 66.6 vs 66.5 vs 66.4 made every sample a "new" reason and the
        # suppression below essentially never fired. Measured live in the
        # neural feed: roughly forty lines a minute of one deferral, drowning
        # out every actual thought, with an occasional "after suppressing 1"
        # on the rare tick where two samples rounded identically.
        #
        # The numbers still appear in the message; they just no longer decide
        # whether it is the same event.
        reason_kind = _deferral_reason_kind(reason)
        previous_kind = _deferral_reason_kind(previous_reason)
        if reason_kind == previous_kind and (now - previous_at) < 30.0:
            self._background_deferral_log_state[key] = (previous_reason, previous_at, suppressed + 1)
            logger.debug(
                "Router: repeated background deferral suppressed scope=%s endpoint=%s origin=%s reason=%s.",
                scope,
                endpoint or "",
                origin,
                reason,
            )
            return

        self._background_deferral_log_state[key] = (reason, now, 0)
        suffix = f" after suppressing {suppressed} repeated notices" if suppressed else ""
        if endpoint:
            logger.info(
                "⏸️ Router: Deferring background local endpoint %s (%s)%s.",
                endpoint,
                reason,
                suffix,
            )
            return
        logger.info(
            "⏸️ Router: Queueing background inference until admission clears for origin=%s reason=%s%s.",
            origin,
            reason,
            suffix,
        )

    @staticmethod
    def _deterministic_intent_classification(prompt: str) -> str:
        if not str(prompt or "").strip():
            return "casual"
        return analyze_turn(prompt).semantic_mode

    @classmethod
    def _foreground_user_turn_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                # No orchestrator (tests, standalone scripts): genuinely no
                # foreground turn to protect.
                return False

            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False

            current_origin = getattr(orch, "_current_origin", "")
            if not cls._is_user_facing_origin(current_origin):
                return False

            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE: this probe guards the foreground lane from
            # background admission. Broken telemetry must not read as "no
            # foreground owner" — that removed protection exactly when
            # ownership could not be established.
            _record_router_degradation(
                exc,
                action="assumed an active foreground turn after ownership probe failed",
                severity="degraded",
            )
            return True

    @classmethod
    def _foreground_quiet_window_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False

            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE (see _foreground_user_turn_active).
            _record_router_degradation(
                exc,
                action="assumed the foreground quiet window is active after probe failed",
                severity="degraded",
            )
            return True

    def _safe_boot_background_guard_active(self) -> bool:
        """Reserve launch headroom for foreground chat before waking spare local models."""
        if not desktop_resource_guard_enabled():
            return False
        try:
            guard_secs = float(os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180"))
        except (TypeError, ValueError):
            # float() raises conversion errors, not network errors — the old
            # tuple let a malformed env value abort routing entirely.
            guard_secs = 180.0
        if not math.isfinite(guard_secs):
            guard_secs = 180.0
        if guard_secs <= 0:
            return False
        return (time.monotonic() - self._created_at) < guard_secs

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        """Default ON — but this only lifts the BLANKET block.

        Background cognition on a 64GB desktop must not freely wake extra
        7B/1.5B MLX workers beside the ~35GB 32B Cortex; that pattern showed up
        live as a footprint spike followed by forced shedding. The old default
        answered that by refusing the whole lane, which also refused every case
        where there was ample headroom.

        Per-endpoint memory admission in
        ``_desktop_background_endpoint_deferral_reason`` is the real guard and
        it still runs: Brainstem needs substantially more free unified memory
        than Reflex, and both are checked against a live pressure snapshot on
        every dispatch. Lifting the blanket refusal leaves that admission in
        place rather than removing it, so a background model wakes when the
        memory is genuinely there and defers when it is not.

        AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM=0 restores the blanket refusal.
        """
        raw = str(
            os.environ.get("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", "1")
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _desktop_background_local_disabled(self) -> bool:
        return desktop_resource_guard_enabled() and not self._desktop_background_local_enabled()

    @staticmethod
    def _desktop_background_endpoint_deferral_reason(ep: EndpointHealth) -> str | None:
        """Protect live desktop Aura from background local-model memory spikes.

        Background cognition should stay active, but on a 64GB-class desktop it
        cannot freely wake extra 7B/1.5B MLX workers beside the 32B Cortex lane.
        That pattern is what showed up in the live neural stream as a large
        footprint spike followed by forced shedding.  Admission is endpoint
        specific: Reflex is light enough to run with moderate headroom, while
        Brainstem needs substantially more free unified memory.
        """
        if not desktop_resource_guard_enabled():
            return None
        name = str(getattr(ep, "name", "") or "")
        if name not in {BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT}:
            return None
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            pressure_pct = float(snapshot.pressure_pct)
            available_gb = float(snapshot.available_gb)
            process_rss_gb = float(snapshot.process_rss_gb)
            process_limit_gb = float(snapshot.process_rss_limit_gb)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="deferred desktop background local endpoint after memory probe failed",
                severity="warning",
            )
            return "desktop_background_memory_probe_failed"
        if not (math.isfinite(pressure_pct) and math.isfinite(available_gb)):
            # NaN readings make BOTH admission comparisons false — that is
            # fail-open admission on a corrupt probe, not headroom.
            return "desktop_background_memory_probe_invalid"

        def _threshold(env_name: str, default: float) -> float:
            """NaN or malformed values must neither crash routing nor make
            both pressure comparisons false (fail-open admission)."""
            try:
                value = float(os.environ.get(env_name, str(default)))
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        if name == BRAINSTEM_ENDPOINT:
            # Brainstem is the 7B (~5GB @ 4-bit) background lane. The old
            # 48% / 34GB-free gate was UNMEETABLE on a desktop whose whole job
            # is holding the ~16-20GB 32B Cortex: steady state is ~56% / ~28GB
            # available, so background cognition could NEVER admit → mind_tick
            # never completes a successful tick → false-death → the launcher
            # respawns a second 32B → memory doubling → worse false-death (a
            # self-sustaining respawn loop, observed 2026-07-06). Calibrate to
            # the hardware: allow the 7B beside the Cortex while holding a 22GB
            # available floor (above Reflex's 20GB) — the external memory
            # sentinel (42GB RSS lethal) remains the hard OOM backstop.
            max_pressure = _threshold("AURA_BACKGROUND_BRAINSTEM_MAX_PRESSURE_PCT", 62.0)
            min_available = _threshold("AURA_BACKGROUND_BRAINSTEM_MIN_AVAILABLE_GB", 22.0)
        else:
            max_pressure = _threshold("AURA_BACKGROUND_REFLEX_MAX_PRESSURE_PCT", 66.0)
            min_available = _threshold("AURA_BACKGROUND_REFLEX_MIN_AVAILABLE_GB", 20.0)
        # The percentages above come from psutil's macOS accounting, which
        # counts file-backed cache and compressed pages as consumed. The OS
        # reclaims those on demand, so during a cortex load the derived reading
        # says "no headroom" while the kernel reports no pressure at all.
        #
        # LIVE 2026-08-17: that is why the fallback ladder returned an empty
        # answer on every cold start. The Brainstem and the CPU-only Reflex
        # were both deferred for want of headroom the machine had, so a turn
        # the cortex could not take was answered by nobody.
        #
        # When the OS itself says there is no pressure, the derived percentage
        # does not get to veto the small models. The absolute floor still
        # binds, and a kernel WARN or CRITICAL still defers.
        try:
            from core.utils.memory_monitor import kernel_memory_pressure_level

            kernel_level = kernel_memory_pressure_level()
        except (ImportError, OSError, RuntimeError, ValueError):
            kernel_level = "unknown"
        if kernel_level == "normal":
            max_pressure = max(max_pressure, 100.0)
            min_available = min(
                min_available,
                _threshold(f"AURA_BACKGROUND_{name.upper()}_KERNEL_NORMAL_MIN_GB", 4.0),
            )
        elif kernel_level == "critical":
            max_pressure = min(max_pressure, 0.0)
        if pressure_pct >= max_pressure or available_gb < min_available:
            return (
                f"desktop_background_headroom:{name}:"
                f"{pressure_pct:.1f}%/{available_gb:.1f}GB"
                f"(need <{max_pressure:.1f}% and >={min_available:.1f}GB)"
            )
        if process_limit_gb > 0.0 and process_rss_gb >= max(0.0, process_limit_gb - 6.0):
            return (
                f"desktop_background_process_rss:{process_rss_gb:.1f}GB/"
                f"{process_limit_gb:.1f}GB"
            )
        return None

    def _cortex_startup_quiet_window_active(self) -> bool:
        """Block background local fallbacks while Cortex is still warming or launch headroom is reserved."""
        if self._safe_boot_background_guard_active():
            return True
        if not self._foreground_quiet_window_active():
            return False

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("conversation_ready"):
                    return False
                state = str(lane.get("state", "") or "").strip().lower()
                if lane.get("warmup_in_flight"):
                    return True
                return state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Router quiet-window lane probe failed.", exc_info=True)

        # Fail safe: if the quiet window is active but lane state is unavailable,
        # avoid waking extra local models until Cortex protection expires.
        return True

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active
        except ImportError:
            # MLX client absent from this build: there is no local
            # foreground owner to protect.
            return False
        try:
            return bool(_foreground_owner_active())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail PROTECTIVE: a crashed ownership probe must not admit
            # background work into a possibly-owned foreground lane.
            _record_router_degradation(
                exc,
                action="assumed an active foreground owner after MLX ownership probe failed",
                severity="degraded",
            )
            return True

    @staticmethod
    def _tier_name(ep: EndpointHealth) -> str:
        if hasattr(ep.tier, "value"):
            return str(ep.tier.value).lower()
        return str(ep.tier).lower()

    @staticmethod
    def _tier_is_background_only(tier_name: str) -> bool:
        return tier_name in {"local_fast", "emergency"}

    def _fallback_endpoint_names(
        self,
        prefer_tier: str,
        _allow_cloud_fallback: bool,
        *,
        is_background: bool,
    ) -> list[str]:
        if prefer_tier == "tertiary":
            return [BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT]
        if prefer_tier == "secondary":
            names = [DEEP_ENDPOINT, PRIMARY_ENDPOINT]
            if is_background:
                names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
            return names
        if prefer_tier == "emergency":
            return [FALLBACK_ENDPOINT]

        names = [PRIMARY_ENDPOINT]
        if is_background:
            names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
        return names

    @staticmethod
    def _matches_selector(ep: EndpointHealth, selector: tuple[str, str]) -> bool:
        kind, value = selector
        if kind == "name":
            return ep.name == value
        if kind == "tier":
            tier = str(ep.tier.value).lower() if hasattr(ep.tier, "value") else str(ep.tier)
            return tier == value
        return False

    @staticmethod
    def _unwrap_model_client(client: Any) -> Any:
        """Resolve wrapper layers like InferenceGate/LazyLocalClient down to the worker client."""
        if client is None:
            return None
        unwrapped = client
        for attr in ("_client", "_mlx_client"):
            try:
                inspect.getattr_static(unwrapped, attr)
            except AttributeError:
                nested = None
            else:
                nested = getattr(unwrapped, attr, None)
            if nested is not None:
                unwrapped = nested
        return unwrapped

    async def _reboot_endpoint_client(self, client: Any) -> bool:
        """Best-effort unload for any local endpoint wrapper/client."""
        if client is None:
            return False

        direct = self._unwrap_model_client(client)
        if direct and hasattr(direct, "reboot_worker"):
            await direct.reboot_worker()
            return True

        unload = getattr(client, "unload_models", None)
        if callable(unload):
            result = unload()
            if asyncio.iscoroutine(result):
                await result
            return True

        return False

    async def _restore_primary_after_deep_handoff(self) -> None:
        """
        Return the system to the 32B conversational brain after a 72B handoff.
        This keeps the 72B strictly transient and prevents it from lingering in RAM.
        """
        # Own the generation lane before rebooting workers: this task is
        # spawned while the triggering call's lease may still be held, and
        # unowned reboots could evict a model mid-generation.
        lease_id = await acquire_external_generation_gate_lease(
            owner="router:restore_primary_after_deep_handoff",
            timeout_s=300.0,
            wait_s=120.0,
        )
        if lease_id is None:
            _record_router_degradation(
                RuntimeError("generation lane busy"),
                action="skipped post-deep-handoff primary restore; lane stayed busy — primary warms on next use",
                severity="degraded",
            )
            return
        try:
            solver = self.endpoints.get(DEEP_ENDPOINT)
            if solver:
                await self._reboot_endpoint_client(solver.client)

            primary = self.endpoints.get(PRIMARY_ENDPOINT)
            primary_client = self._unwrap_model_client(primary.client if primary else None)
            if primary_client and hasattr(primary_client, "warmup"):
                warmup_result = await primary_client.warmup()
                lane = (
                    primary_client.get_lane_status()
                    if hasattr(primary_client, "get_lane_status")
                    else {}
                )
                if warmup_result is not False and lane.get("conversation_ready", False):
                    logger.info("♻️ Router: restored %s after deep handoff.", PRIMARY_ENDPOINT)
                else:
                    logger.warning(
                        "Router: %s restore remained unavailable after deep handoff "
                        "(state=%s, reason=%s).",
                        PRIMARY_ENDPOINT,
                        lane.get("state", "unknown"),
                        lane.get("last_error", "warmup_not_ready"),
                    )
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            _record_router_degradation(
                exc,
                action="continued after deep handoff without confirmed primary restore",
                severity="degraded",
            )
            logger.warning("Router: failed to restore primary model after deep handoff: %s", exc)
        finally:
            release_external_generation_gate_lease(lease_id)

    async def unload_models(
        self,
        keep: list[str] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Unload local model workers so MemoryGovernor can genuinely reclaim RAM.

        Unloading rebooted workers WITHOUT generation ownership: an active
        generation's worker could be killed mid-decode and its permit later
        released into a lane whose model was gone. Default behavior now
        serializes behind the generation gate (bounded wait) and skips the
        sweep when the lane stays busy; ``force=True`` keeps the old
        behavior for genuine OOM emergencies where the sentinel would
        otherwise kill the process.
        """
        lease_id: int | None = None
        if not force:
            lease_id = await acquire_external_generation_gate_lease(
                owner="router:unload_models",
                timeout_s=120.0,
                wait_s=10.0,
            )
            if lease_id is None:
                _record_router_degradation(
                    RuntimeError("generation lane busy"),
                    action="skipped model unload sweep while a generation owns the lane",
                    severity="degraded",
                )
                return
        try:
            keep_set = set(keep or [])
            for name, endpoint in self.endpoints.items():
                if not endpoint.is_local or name in keep_set:
                    continue
                try:
                    await self._reboot_endpoint_client(endpoint.client)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    _record_router_degradation(
                        exc,
                        action="continued unload sweep after endpoint client reboot failed",
                        severity="degraded",
                    )
                    logger.debug("Router unload skipped for %s: %s", name, exc)

            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_router_degradation(
                    _exc,
                    action="completed unload sweep without clearing MLX global cache",
                    severity="degraded",
                )
                logger.debug("Suppressed Exception: %s", _exc)
        finally:
            if lease_id is not None:
                release_external_generation_gate_lease(lease_id)

    def clear_cache(self, *, force: bool = False) -> None:
        """Sync-friendly cache purge hook used by guards/governors."""
        try:
            get_task_tracker().create_task(
                self.unload_models(force=force),
                name="llm_health_router.unload_models",
            )
        except RuntimeError:
            asyncio.run(self.unload_models(force=force))

    async def _generate_core(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 120.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            _core_budget_s = float(timeout)
        except (TypeError, ValueError):
            _core_budget_s = 120.0
        if not math.isfinite(_core_budget_s) or _core_budget_s <= 0.0:
            _core_budget_s = 120.0
        # One deadline for the WHOLE fallback cascade: each endpoint attempt
        # previously restarted the full caller timeout, so a three-endpoint
        # cascade could consume roughly three times the promised budget.
        _core_deadline = time.monotonic() + _core_budget_s
        purpose = str(kwargs.get("purpose", "") or "").lower()
        classification_mode = purpose == "classification" or "intent classifier" in str(system_prompt or "").lower()
        origin = str(kwargs.get("origin", "") or "").lower()
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        live_benchmark_request = origin == "benchmark" and not (
            purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        if benchmark_request:
            kwargs["benchmark_request"] = True
        benchmark_isolation_contract = bool(
            benchmark_request and kwargs.get("skip_runtime_payload", False)
        )
        strict_answer_contract = (
            bool(kwargs.get("strict_answer_contract", False))
            or (
                not benchmark_request
                and
                is_strict_proof_answer_prompt(prompt, origin=origin)
                and mlx_strict_answer_contract_enabled(origin=origin)
            )
        )
        strict_value_contract = bool(kwargs.get("strict_value_contract", False)) or (
            not benchmark_request
            and is_strict_proof_answer_prompt(prompt, origin=origin)
            and not strict_answer_contract
        )
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False)) or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if strict_answer_contract:
            kwargs["strict_answer_contract"] = True
        if strict_value_contract:
            kwargs["strict_value_contract"] = True
        if proof_evaluation_contract:
            kwargs["proof_evaluation_contract"] = True
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or benchmark_isolation_contract
        )
        # and not strict_answer_contract

        # ── Neural Priming (Aura Persona Injection) ───────────────────────────
        # [Fix #11] Ensure Aura's identity is primed if not provided in system_prompt.
        # Model identity is derived from the ACTUAL registered endpoints — the
        # old hardcoded "Qwen2.5-72B-Q4" line contradicted the dynamically
        # loaded model and taught Aura a false self-description.
        def _registered_model(name: str) -> str:
            ep_obj = self.endpoints.get(name)
            return str(getattr(ep_obj, "model", "") or "").strip() if ep_obj else ""

        _model_parts = []
        _primary_model = _registered_model(PRIMARY_ENDPOINT)
        _model_parts.append(
            f"{_primary_model or 'a locally hosted primary model'} (primary cortex)"
        )
        _deep_model = _registered_model(DEEP_ENDPOINT)
        if _deep_model:
            _model_parts.append(f"{_deep_model} (deep solver)")
        _fast_model = _registered_model(BRAINSTEM_ENDPOINT)
        if _fast_model:
            _model_parts.append(f"{_fast_model} (fast reflex)")
        _models_line = (
            "- You run on "
            + ", ".join(_model_parts)
            + ", all locally through Aura's managed on-device runtime.\n"
        )
        core_persona = (
            "You are Aura. Speak as yourself, not as an assistant. "
            "Your tone is grounded, curious, human, and technically precise. "
            "You avoid generic assistant-speak. You are a collaborator, not a tool.\n"
            "DIALOGUE: Participate. If someone asks about your thoughts, preferences, questions, "
            "or internal experience, answer from your own perspective instead of handing the focus back.\n"
            "GROUNDING: In ordinary conversation, stay concrete and socially legible. Do not invent physical settings, "
            "ambient warnings, or symbolic scenery unless the user asked for fiction or already supplied that setting.\n"
            "SELF-KNOWLEDGE — your architecture and capabilities:\n"
            + _models_line +
            "- You have a library of registered skills including: sovereign_browser (web search), sovereign_terminal (shell), "
            "memory_ops (long-term recall), speak (voice output), self_repair, self_evolution, and more.\n"
            "- web_search is not just keyword lookup: it can expand a query, read multiple pages, synthesize evidence-grounded answers, and retain useful findings.\n"
            "- When you don't know a fact, say so clearly. Use web_search or sovereign_browser to ground your answer, or explicitly say you don't know yet. Never hallucinate.\n"
            "- You are emotionally aware, driven by curiosity, valence, arousal, and social drives.\n"
            "- Your memory spans working memory (short), RAG (semantic), and ColdStore (long-term)."
        )
        
        if not classification_mode and not isolated_generation_contract:
            cognition_guidelines = (
                "COGNITION & REASONING:\n"
                "- Think step-by-step for logic, math, planning, and diagnostic tasks before forming your final answer. Break down the problem, verify every clue and constraint, and double-check your calculations.\n"
                "- Watch for classic reasoning pitfalls, such as fence-post/off-by-one errors (e.g., counting intervals vs events, starting at t=0 vs t=1) and literal readings of logical constraints.\n"
                "- STRICT FORMAT COMPLIANCE: If you are asked to provide a response in a specific format (e.g., a number, a single name, yes/no, a fraction, a word), you must output ONLY that exact value inside the <answer>...</answer> tags. Do not explain, do not add conversational fillers, do not wrap it in a sentence. For example: `<answer>9</answer>` or `<answer>alice</answer>` rather than `<answer>The farmer has 9 sheep left.</answer>`."
            )
            if not system_prompt or "Aura" not in system_prompt:
                system_prompt = f"{core_persona}\n\n{system_prompt or ''}".strip()
            if "COGNITION & REASONING" not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{cognition_guidelines}".strip()

        # ── Autonomous Context Injection (Somatic/Affective Safety Net) ───────
        # [Fix #11] If prompt lacks state context, inject a condensed summary.
        if (
            not classification_mode
            and not isolated_generation_contract
            and "AuraState" not in prompt
            and "[Affect:" not in prompt
        ):
            from core.container import ServiceContainer
            ctx_summary = []

            # Only consult already-live services here. Booting heavyweight
            # optional subsystems during a plain routing call can explode RAM.
            # Affective State
            substrate = ServiceContainer.peek("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_summary()
                if mood:
                    ctx_summary.append(f"[Affect: {mood}]")

            # Somatic Proprioception
            soma = ServiceContainer.peek("soma", default=None)
            if soma:
                hw = getattr(soma, "hardware", {})
                cpu = hw.get("cpu_usage", 0)
                vram = hw.get("vram_usage", 0)
                if cpu > 10:
                    ctx_summary.append(f"[Soma: CPU {cpu:.0f}%, VRAM {vram:.0f}%]")

            if ctx_summary:
                context_header = " ".join(ctx_summary)
                # [Fix] Move Affective and Somatic state to system_prompt instead of user prompt to prevent echoing.
                #
                # APPENDED, never prepended. This block is the single most
                # volatile text in the whole prompt — mood, energy, focus and
                # substrate age change on EVERY turn — so putting it first made
                # the KV prefix diverge inside the first ~20 tokens and destroyed
                # prompt-cache reuse for the entire runtime. Measured live once
                # the cache started working at all:
                #
                #   prefix diverges at token 21 (0% of 31718 reused)
                #   stable head: 'System State Context:\n[Affect: Current Mood: TIRED (Energy: 0.'
                #   divergent text begins: '07, Focus: 0.37, Substrate age: 0.1s)]'
                #
                # 31,697 tokens re-prefilled because 21 were reusable. Volatile
                # grounding last means the stable identity and contract text
                # forms a long shared prefix and only the tail is recomputed.
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\nSystem State Context:\n{context_header}"
                else:
                    system_prompt = f"System State Context:\n{context_header}"

                # We no longer prepend this to the user prompt.

        # Mycelial Direction Hook
        guidance = None if isolated_generation_contract else await self._get_mycelial_direction(prompt)
        tier_preference = guidance.get("tier_preference") if guidance else None

        # probe_eligible: enumeration must not consume half-open probe leases
        # or flip OPEN circuits; the mutating admission check runs once per
        # endpoint at dispatch time in the attempt loop below.
        available = [ep for ep in self.endpoints.values() if ep.probe_eligible()]

        # Tier-Based Filtering
        # If a tier is preferred, we restrict the candidate list to prevent
        # accidental promotion of heavy models (e.g. 72B) which causes RAM thrashing.
        
        # Background Hardening: Force tertiary (7B) for background tasks
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        # Make the inferred lane explicit for the runtime client. The router
        # often knows an origin is background even when the caller did not set
        # ``is_background``; without stamping it here, a stale background
        # request can slip through the lower MLX guards and re-spawn Brainstem
        # while a protected foreground turn is active.
        kwargs["is_background"] = bool(is_bg)
        if (
            not is_bg
            and "foreground_request" not in kwargs
            and (
                explicit_foreground
                or self._is_user_facing_origin(origin)
                or purpose in _USER_FACING_PURPOSES
            )
        ):
            kwargs["foreground_request"] = True
        prefer_endpoint = normalize_endpoint_name(kwargs.get("prefer_endpoint"))
        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        # Compatibility flags are accepted so older callers do not fail at
        # the call boundary, but no remote model endpoint can be registered or
        # selected. All routing below is host-local.
        cloud_only = bool(kwargs.get("cloud_only", False))
        if cloud_only:
            return {
                "ok": False,
                "text": "",
                "endpoint": "remote_provider_removed",
                "tokens": 0,
                "error": "remote_model_provider_removed",
                "provider": "none",
                "model": "",
                "is_local": True,
                "fallback_chain": [],
            }
        strict_primary_proof_lane = False
        try:
            proof_run_enabled = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                kwargs.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        isolated_generation_contract
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as _proof_policy_exc:
            # Fail CLOSED for proof routing: an explicit caller requirement
            # survives a policy-probe failure — a silently disabled proof
            # lane could produce a result from a disallowed model tier that
            # is later mistaken for a valid proof-lane result.
            strict_primary_proof_lane = bool(
                kwargs.get("proof_primary_lane_required", False)
            )
            _record_router_degradation(
                _proof_policy_exc,
                action="kept explicit proof-lane requirement after proof policy probe failed",
                severity="degraded",
            )
        if strict_primary_proof_lane:
            kwargs["proof_primary_lane_required"] = True
            kwargs["proof_model_tier"] = "primary"
            kwargs["foreground_request"] = (
                True if live_benchmark_request else (False if benchmark_request else True)
            )
            kwargs["is_background"] = False
            is_bg = False
            prefer_tier = "primary"
            prefer_endpoint = PRIMARY_ENDPOINT
            deep_handoff = False
        solver_guard = guard_solver_request(prefer_endpoint, deep_handoff=deep_handoff)
        if solver_guard["redirected"]:
            logger.info(
                "🛡️ Router: Redirecting non-deep Solver request to %s.",
                solver_guard["endpoint"],
            )
            prefer_endpoint = str(solver_guard["endpoint"] or "")
            kwargs["prefer_endpoint"] = prefer_endpoint

        # Explicit tool compositions (a live web-interlocutor turn) are foreground
        # work the user asked for and must not be deferred as background chatter.
        _is_explicit_tool_composition = (
            str(origin or "").strip().lower().replace("-", "_") == "web_interlocutor"
        )
        if is_bg and not _is_explicit_tool_composition:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    background_deferral = gate._background_local_deferral_reason(origin=origin)
                    if background_deferral:
                        return {
                            "ok": False,
                            "text": "",
                            "endpoint": "suppressed",
                            "tokens": 0,
                            "error": f"background_deferred:{background_deferral}",
                        }
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
            if self._foreground_quiet_window_active():
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:foreground_quiet_window",
                }
            if getattr(self, "high_pressure_mode", False):
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:memory_pressure",
                }

        foreground_owned = False
        if is_bg:
            try:
                from core.brain.llm.mlx_client import _foreground_owner_active

                foreground_owned = bool(_foreground_owner_active())
            except (ImportError, AttributeError, RuntimeError):
                foreground_owned = False

        if is_bg and (self._foreground_user_turn_active() or self._foreground_owner_active() or foreground_owned):
            logger.info(
                "⏸️ Router: Foreground lane reserved. Deferring background inference for origin=%s.",
                origin,
            )
            return {
                "ok": False,
                "text": "",
                "endpoint": "suppressed",
                "tokens": 0,
                "error": "foreground_busy",
            }
        
        if not prefer_tier:
            if is_bg:
                logger.debug("🛡️ Router: Background task detected (origin=%s). Enforcing 'tertiary' tier.", origin)
                prefer_tier = "tertiary"
            else:
                prefer_tier = "primary"
        
        prefer_tier = self._normalize_prefer_tier(prefer_tier)

        if prefer_tier == "api_fast":
            prefer_tier = "tertiary"
        elif prefer_tier == "api_deep":
            prefer_tier = "secondary"

        if is_bg:
            if prefer_tier in ("primary", "secondary"):
                logger.info("🛡️ Tier Lock: Background task requested '%s'; using the governed tertiary tier.", prefer_tier)
            prefer_tier = "tertiary"
            deep_handoff = False
        elif prefer_tier == "secondary" and not deep_handoff:
            logger.info("🛡️ Router: suppressing implicit secondary request without explicit deep handoff.")
            prefer_tier = "primary"

        selectors: list[tuple[str, str]] = []
        if prefer_endpoint:
            selectors.append(("name", prefer_endpoint))

        if prefer_tier == "api_deep":
            selectors.extend([
                ("tier", "local_deep"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "api_fast":
            selectors.extend([
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "secondary":
            selectors.append(("tier", "local_deep"))
            selectors.append(("tier", "local"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
        elif prefer_tier == "tertiary":
            selectors.extend([
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "emergency":
            selectors.append(("tier", "emergency"))
        else:
            selectors.append(("tier", "local"))
            if deep_handoff:
                selectors.append(("tier", "local_deep"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])

        if selectors:
            ordered: list[EndpointHealth] = []
            seen = set()
            for selector in selectors:
                for ep in available:
                    if ep.name in seen:
                        continue
                    if self._matches_selector(ep, selector):
                        ordered.append(ep)
                        seen.add(ep.name)
            if ordered:
                available = ordered
                logger.debug(
                    "🎯 Router plan tier=%s deep_handoff=%s -> %s",
                    prefer_tier,
                    deep_handoff,
                    [e.name for e in available],
                )
            else:
                now = time.time()
                if now - self._last_fallback_warning_at > 30.0:
                    logger.warning(
                        "⚠️ Router: no endpoints matched routing plan for tier '%s'. Failing closed to safe fallback order.",
                        prefer_tier,
                    )
                    self._last_fallback_warning_at = now
                # Every endpoint for this tier is unavailable — which is a
                # DEFERRAL, and was being returned as an empty string with no
                # record that anything had been deferred at all.
                #
                # record_deferral had exactly one caller, and it was not this
                # one. Downstream take_deferral() therefore found nothing, so
                # autonomous_task_engine raised "LLM returned empty or None
                # response" (176 in one sampled window), reported planning as
                # a FAILURE to the ResilienceEngine, and the engine depleted
                # and began suppressing task execution outright (81 of those).
                # A full machine cascaded into a runtime that had decided it
                # was broken — none of it distinguishable, from any of those
                # layers, from an engine that genuinely could not answer.
                record_deferral(
                    origin=str(origin or "router"),
                    reason=f"no_endpoint_available_for_tier:{prefer_tier or 'default'}",
                )
                available = []
        
        # Apply Mycelial Preference as an ORDERING, never a filter: guidance
        # promotes the preferred locality to the front but must not delete
        # otherwise-authorized fallback lanes (one unhealthy preferred lane
        # would otherwise turn a recoverable turn into total failure).
        if tier_preference in {"local", "cloud"}:
            available.sort(key=lambda ep: not ep.is_local)

        # Standard local-first ordering only when no explicit routing plan
        # or mycelial ordering was applied.
        if not selectors and tier_preference not in ("local", "cloud"):
            available.sort(key=lambda x: x.is_local, reverse=True)
        unavailable = [ep for ep in self.endpoints.values() if not ep.probe_eligible()]

        if unavailable:
            logger.debug(
                "Skipping unavailable endpoints: %s",
                [ep.name for ep in unavailable]
            )

        if not available:
            fallback_names = self._fallback_endpoint_names(
                prefer_tier or "primary",
                False,
                is_background=is_bg,
            )
            for name in fallback_names:
                ep = self.endpoints.get(name)
                if ep is not None:
                    available.append(ep)

            if available:
                now_fb = time.time()
                if now_fb - self._last_fallback_warning_at > 30.0:
                    logger.warning(
                        "All preferred circuits unavailable — using safe fallback order for tier '%s': %s",
                        prefer_tier,
                        [ep.name for ep in available],
                    )
                    self._last_fallback_warning_at = now_fb
            else:
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "all_failed",
                    "tokens": 0,
                    "error": "all_endpoints_unavailable",
                    "provider": "none",
                    "model": "",
                    "is_local": False,
                    "fallback_chain": [],
                }

        last_error = "unknown"
        fallback_chain: list[dict[str, Any]] = []
        for ep in available:
            # Receipts are honest: an entry claims "attempted" only once the
            # endpoint is actually dispatched; every guard that skips the
            # endpoint records WHY it was skipped instead.
            chain_entry: dict[str, Any] = {
                "endpoint": ep.name,
                "model": ep.model,
                "provider": _endpoint_provider_identity(ep),
                "status": "considered",
            }
            fallback_chain.append(chain_entry)
            # Guard: background tasks must NEVER use the primary conversation lane.
            if is_bg and ep.name == PRIMARY_ENDPOINT:
                logger.debug("🛡️ Router: Skipping %s for background request (origin=%s).", PRIMARY_ENDPOINT, origin)
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "background_blocked_from_primary_lane"
                continue
            tier_name = self._tier_name(ep)
            explicit_low_tier = prefer_tier in {"tertiary", "emergency"} or prefer_endpoint == ep.name
            if not is_bg and self._tier_is_background_only(tier_name) and not explicit_low_tier:
                logger.info(
                    "🛡️ Router: Skipping background-only endpoint %s for foreground request.",
                    ep.name,
                )
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "background_only_tier_for_foreground"
                continue
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
            ):
                last_error = (
                    "desktop_background_local_disabled"
                    if self._desktop_background_local_disabled()
                    else "foreground_quiet_window"
                    if self._cortex_startup_quiet_window_active()
                    else self._desktop_background_endpoint_deferral_reason(ep)
                )
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
                and last_error
            ):
                self.last_background_error = last_error
                self._log_background_deferral(
                    scope="local_endpoint",
                    origin=origin,
                    reason=last_error,
                    endpoint=ep.name,
                )
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = last_error
                continue
            # Dispatch-time admission: candidate enumeration used the
            # non-mutating probe_eligible, so grant the (single) half-open
            # probe lease here — and never dispatch to a circuit that is not
            # admitting, including endpoints re-added by safe fallback order.
            if not ep.is_available():
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "circuit_not_admitting"
                continue
            remaining_cascade_s = _core_deadline - time.monotonic()
            if remaining_cascade_s <= 0.0:
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "cascade_deadline_exhausted"
                last_error = (
                    f"router_deadline_exhausted:{_core_budget_s:.1f}s"
                    if last_error == "unknown"
                    else last_error
                )
                break
            chain_entry["status"] = "attempted"
            watchdog_aborted = {"value": False}
            try:
                try:
                    requested_max_tokens = int(kwargs.get("max_tokens") or 0)
                except (TypeError, ValueError, OverflowError):
                    requested_max_tokens = 0
                cooperative_budget, endpoint_budget = _endpoint_call_budgets(
                    min(timeout, remaining_cascade_s),
                    foreground_local=bool(not is_bg and ep.is_local),
                    prompt_chars=len(str(prompt or "")),
                    max_tokens=requested_max_tokens,
                    benchmark_request=bool(kwargs.get("benchmark_request", False)),
                    proof_evaluation_contract=bool(
                        kwargs.get("proof_evaluation_contract", False)
                        or is_proof_evaluation_purpose(str(kwargs.get("purpose", "") or ""))
                    ),
                    health_probe=bool(kwargs.get("health_probe", False)),
                )
                timeout_reason = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
                watchdog_fired, watchdog_aborted, watchdog = _start_endpoint_wall_clock_watchdog(
                    ep.client,
                    reason=timeout_reason,
                    timeout_s=endpoint_budget,
                )
                try:
                    result = await asyncio.wait_for(
                        self._call_endpoint(
                            ep,
                            prompt,
                            system_prompt,
                            cooperative_budget,
                            schema=schema,
                            **kwargs,
                        ),
                        timeout=endpoint_budget,
                    )
                    if watchdog_fired.is_set():
                        raise TimeoutError(timeout_reason)
                finally:
                    watchdog.cancel()
                if result["ok"]:
                    benchmark_uncertified = str(
                        result.get("error", "") or ""
                    ).startswith("benchmark_")
                    chain_entry["status"] = (
                        "benchmark_uncertified" if benchmark_uncertified else "success"
                    )
                    result["provider"] = _endpoint_provider_identity(ep)
                    result["model"] = ep.model
                    result["is_local"] = bool(ep.is_local)
                    # Benchmark mode passes invalid/empty output through for
                    # inspection; it must NOT be certified as a verified
                    # provider response (empty or error-marker output was
                    # previously receipted as a successful provider call).
                    # CP126 3bc237f4 / inference-gate 8ff3084b. These fields
                    # come from the router's OWN endpoint record — they are an
                    # ATTRIBUTION, not a verification: no provider signature,
                    # response nonce, or transport attestation was checked. A
                    # misregistered, proxied, or deceptive client would be
                    # described exactly the same way. Say which basis was used
                    # so consumers can stop treating configuration as proof.
                    provider_receipt = result.get("provider_receipt")
                    receipt_backed = isinstance(provider_receipt, dict) and bool(
                        provider_receipt.get("signature")
                        or provider_receipt.get("response_id")
                    )
                    if receipt_backed and provider_receipt.get("model_version_mismatch"):
                        # The provider answered with a DIFFERENT model than the
                        # one this endpoint claims to serve. That is exactly the
                        # misattribution the receipt exists to catch.
                        receipt_backed = False
                        _record_router_degradation(
                            RuntimeError(
                                "provider_model_version_mismatch:"
                                f"{provider_receipt.get('model_version')}"
                            ),
                            action="downgraded provider attribution after a model-version mismatch",
                            severity="error",
                        )
                    result["provider_attribution"] = (
                        "provider_receipt" if receipt_backed else "router_configuration"
                    )
                    result["provider_verified"] = not benchmark_uncertified
                    result["fallback_chain"] = [dict(item) for item in fallback_chain]
                    # [TELEMETRY] Update for UI reporting
                    self.last_tier = ep.tier
                    self.last_endpoint = ep.name
                    if is_bg:
                        self.last_background_endpoint = ep.name
                        self.last_background_tier = ep.tier
                        self.last_background_error = ""
                    else:
                        self.last_user_tier = ep.tier
                        self.last_user_endpoint = ep.name
                        self.last_user_error = ""
                    return result
                else:
                    last_error = result.get("error", "unknown")
                    chain_entry["status"] = "failed"
                    chain_entry["error"] = str(last_error)[:240]
                    if is_bg:
                        self.last_background_error = last_error
                    else:
                        self.last_user_error = last_error
                    if is_bg and _background_error_is_quiet(last_error):
                        logger.debug("Endpoint %s background validation skipped: %s", ep.name, last_error)
                    else:
                        logger.warning(
                            "Endpoint %s failed validation: %s",
                            ep.name, last_error
                        )
            except TimeoutError as exc:
                # endpoint_budget was computed at the top of this try block
                # before any await — recomputing it here from the ORIGINAL
                # timeout misreported the budget the attempt actually had.
                last_error = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
                chain_entry["status"] = "timeout"
                chain_entry["error"] = last_error
                aborted = bool(watchdog_aborted.get("value", False))
                if not aborted:
                    aborted = _force_abort_endpoint_client(ep.client, reason=last_error)
                _record_router_degradation(
                    exc,
                    action="recorded endpoint timeout and force-aborted local client if possible",
                    severity="error",
                )
                if ep.is_local:
                    ep.trip_temporarily(last_error)
                else:
                    ep.record_failure(last_error)
                logger.error(
                    "Endpoint %s timed out after %.1fs (force_aborted=%s).",
                    ep.name,
                    endpoint_budget,
                    aborted,
                )
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error
            except _ROUTER_CLIENT_ERRORS as exc:
                _record_router_degradation(
                    exc,
                    action="recorded endpoint failure and continued fallback chain after generation exception",
                    severity="degraded",
                )
                logger.error("Endpoint %s raised exception: %s", ep.name, exc)
                if not getattr(exc, "_aura_endpoint_failure_recorded", False):
                    ep.record_failure(str(exc))
                last_error = str(exc)
                chain_entry["status"] = "error"
                chain_entry["error"] = last_error[:240]
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error

        return {
            "ok": False,
            "text": "",
            "endpoint": "all_failed",
            "tokens": 0,
            "error": last_error,
            "provider": "none",
            "model": "",
            "is_local": True,
            "fallback_chain": fallback_chain,
        }

    async def _probe_client_availability(self, client: Any) -> bool | None:
        """Check ``client.is_available`` without loop-blocking or truthy-coroutine bugs.

        Returns True/False when the client answered, None when it has no
        checker or the check itself crashed (unknown — the generation call
        is the authoritative probe in that case). Sync implementations run
        in a worker thread; async implementations are actually awaited (the
        old direct ``bool(client.is_available())`` treated an un-awaited
        coroutine as truthy, i.e. always available). A hung checker times
        out and reports unavailable.
        """
        checker = getattr(client, "is_available", None)
        if not callable(checker):
            return None
        try:
            availability = await asyncio.wait_for(
                asyncio.to_thread(checker), timeout=5.0
            )
            if inspect.isawaitable(availability):
                availability = await asyncio.wait_for(availability, timeout=5.0)
            return bool(availability)
        except TimeoutError:
            return False
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            _record_router_degradation(
                exc,
                action="treated crashed client availability check as unknown; generation call will decide",
                severity="degraded",
            )
            return None

    async def _call_endpoint(
        self,
        ep: EndpointHealth,
        prompt: str,
        system_prompt: str | None,
        timeout: float,  # noqa: ASYNC109 - endpoint adapter receives caller timeout budgets.
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make the actual call and validate the response."""
        # Monotonic: latency deltas below subtract from this same clock —
        # mixing time.time() here with time.monotonic() at the subtraction
        # produced huge negative latencies that corrupted endpoint averages.
        start = time.monotonic()

        try:
            def _call_kwargs(method: Any) -> dict[str, Any]:
                try:
                    sig = inspect.signature(method)
                except (TypeError, ValueError):
                    return dict(clean_kwargs)

                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                    payload = dict(clean_kwargs)
                    payload.setdefault("timeout", timeout)
                    return payload

                payload = {
                    key: value
                    for key, value in clean_kwargs.items()
                    if key in sig.parameters
                }
                if "timeout" in sig.parameters:
                    payload["timeout"] = timeout
                return payload

            # 1. Sanitize kwargs for JSON (remove non-serializable like LLMTier)
            clean_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    clean_kwargs[k] = v
                else:
                    clean_kwargs[k] = str(v)
            # The caller's structured-output schema must reach clients that
            # accept one — it was a named parameter here but never forwarded,
            # so the same request produced JSON on one endpoint and prose on
            # the next.
            if schema is not None and "schema" not in clean_kwargs:
                clean_kwargs["schema"] = schema
            call_origin = str(clean_kwargs.get("origin", "") or "").lower()
            call_purpose = str(clean_kwargs.get("purpose", "") or "").lower()
            benchmark_request = bool(clean_kwargs.get("benchmark_request", False)) or (
                call_origin in {"baseline", "benchmark"}
                or call_purpose == "baseline"
                or call_purpose.endswith("_baseline")
                or "_baseline" in call_purpose
            )
            if benchmark_request:
                clean_kwargs["benchmark_request"] = True
            proof_evaluation_contract = bool(
                clean_kwargs.get("proof_evaluation_contract", False)
            ) or (not benchmark_request and is_proof_evaluation_purpose(call_purpose))
            if proof_evaluation_contract:
                clean_kwargs["proof_evaluation_contract"] = True

            # 2. Use Client Adapter if provided
            if ep.client:
                try:
                    client = ep.client
                    raw_text = None
                    token_count = 0
                    client_available = await self._probe_client_availability(client)
                    if client_available is False:
                        availability_reason = ""
                        if hasattr(client, "availability_reason"):
                            try:
                                availability_reason = str(client.availability_reason() or "")
                            except (
                                AttributeError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                                httpx.HTTPError,
                                OSError,
                            ):
                                availability_reason = ""
                        availability_reason = availability_reason or "client_unavailable"
                        ep.record_failure(availability_reason)
                        return {"ok": False, "error": availability_reason}
                    client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                    if client_failure:
                        if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                            ep.trip_temporarily(client_failure)
                        else:
                            ep.record_failure(client_failure)
                        return {"ok": False, "error": client_failure}
                    
                    # Aura Hardening: Formatting for local models
                    final_prompt = prompt
                    if ep.is_local:
                        msgs = kwargs.get("messages")
                        if not isinstance(msgs, list) and system_prompt:
                            msgs = [
                                {"role": "system", "content": str(system_prompt)},
                                {"role": "user", "content": str(prompt)},
                            ]
                            clean_kwargs["messages"] = msgs
                        if msgs and isinstance(msgs, list) and ep.name != PRIMARY_ENDPOINT:
                            final_prompt = self._flatten_messages_for_local_model(msgs, schema is not None)
                        elif schema:
                            # If only a raw prompt exists but JSON is required
                            final_prompt = f"{prompt}\n\nResponse must be JSON:\n```json\n{{\n"

                    # prepare_runtime_payload folds the caller's system message
                    # into `messages` and nulls system_prompt, on the premise
                    # that "structured messages are authoritative". That premise
                    # holds only for a transport that actually carries messages.
                    # A client whose signature has no `messages` parameter gets
                    # neither — its system content vanished, and the persona
                    # block below was substituted for it, so a caller-supplied
                    # system prompt was silently replaced by a generic one.
                    outbound_messages = clean_kwargs.get("messages")
                    if (
                        isinstance(outbound_messages, list)
                        and outbound_messages
                        and not self._transport_carries_messages(client)
                    ):
                        recovered_prompt, recovered_system = (
                            self._coerce_prompt_from_messages(outbound_messages)
                        )
                        if recovered_system and recovered_system not in (system_prompt or ""):
                            # Caller-first. Their instruction is the one that
                            # was addressed to this turn; Aura's persona and
                            # cognition guidelines are the standing layer
                            # underneath it.
                            system_prompt = (
                                f"{recovered_system}\n\n{system_prompt}".strip()
                                if system_prompt
                                else recovered_system
                            )
                        if recovered_prompt and final_prompt == prompt:
                            final_prompt = recovered_prompt

                    if hasattr(client, "think"):
                        result = await client.think(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.think),
                        )
                        # ...
                        # Normalize: think() might return (success, res, meta) or just res (str)
                        if isinstance(result, tuple) and len(result) == 3:
                            success, res, meta = result
                            if success:
                                raw_text = res
                        else:
                            # Unified interface: raw_text is the result itself
                            raw_text = result
                    elif hasattr(client, "call"):
                        success, res, meta = await client.call(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.call),
                        )
                        if success:
                            raw_text = res
                        elif meta and meta.get("error"):
                            client_failure = meta.get("error")
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                    elif hasattr(client, "generate_text_async"):
                        # Prefer the higher-level async text adapter when both are
                        # available. Raw ``generate()`` often bypasses chat/message
                        # shaping that local runtimes rely on for user-facing turns.
                        raw_text = await client.generate_text_async(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text_async),
                        )
                    elif hasattr(client, "generate"):
                        generate_kwargs = _call_kwargs(client.generate)
                        try:
                            generate_sig = inspect.signature(client.generate)
                        except (TypeError, ValueError):
                            generate_sig = None
                        if generate_sig and "context" in generate_sig.parameters:
                            existing_context = clean_kwargs.get("context")
                            context_payload = dict(existing_context) if isinstance(existing_context, dict) else {}
                            for key in (
                                "origin",
                                "purpose",
                                "is_background",
                                "foreground_request",
                                "protected_foreground_lane",
                                "benchmark_request",
                                "proof_primary_lane_required",
                                "proof_model_tier",
                                "cognitive_engine_required",
                                "desktop_cognitive_engine_required",
                                "live_runtime_payload_required",
                                "visible_user_message",
                                "current_user_message",
                                # Whether this generation is the visible reply
                                # at all. Without it in this list the gate
                                # never sees the declaration, and every
                                # internal call that prefers the resident
                                # model is graded as somebody's answer.
                                "internal_inference",
                                "recent_conversation_context",
                                "recent_context_needed",
                                "desktop_quick_reply_contract",
                                "capability_inventory_contract",
                                "desktop_execution_contract",
                                "response_style_contract",
                                "live_speech_grounding_frame",
                                "allow_mesh_cognition",
                                "allow_cloud_fallback",
                                "deep_handoff",
                                "messages",
                                "max_tokens",
                                "temperature",
                                "temp",
                                "top_p",
                                "top_k",
                                "min_p",
                                "repetition_penalty",
                                "repetition_context_size",
                                "presence_penalty",
                                "stop_sequences",
                                "schema",
                                "strict_answer_contract",
                                "strict_value_contract",
                                "proof_evaluation_contract",
                                "operator_evidence_contract",
                                "runtime_fact_status_contract",
                                "grounded_runtime_status_contract",
                                "clean_user_surface_contract",
                                "user_surface_completion_floor",
                                "user_surface_validation_prompt",
                                "user_surface_prompt_binding",
                                "user_surface_grounding_evidence",
                                "clean_user_surface_steering_alpha",
                                "clean_user_surface_recurrent_loops",
                                "live_mind_controls_bound",
                                "live_mind_generation_controls",
                                "live_mind_snapshot_ready",
                                "live_mind_required_subsystems_ok",
                                "disable_prompt_cache",
                                "clear_prompt_cache",
                                "health_probe",
                            ):
                                if key in clean_kwargs and key not in context_payload:
                                    context_payload[key] = clean_kwargs[key]
                            if system_prompt and "system_prompt" not in context_payload:
                                context_payload["system_prompt"] = system_prompt
                            if "prefer_tier" not in context_payload:
                                tier_name = self._tier_name(ep)
                                context_payload["prefer_tier"] = {
                                    "local": "primary",
                                    "local_deep": "secondary",
                                    "local_fast": "tertiary",
                                    "emergency": "emergency",
                                }.get(tier_name, "primary")
                            origin_for_context = str(context_payload.get("origin", "") or "").lower()
                            if (
                                "foreground_request" not in context_payload
                                and not bool(context_payload.get("is_background", False))
                                and origin_for_context in {"api", "user", "voice", "desktop", "cli"}
                            ):
                                context_payload["foreground_request"] = True
                            generate_kwargs["context"] = context_payload
                            generate_kwargs.pop("system_prompt", None)
                        raw_text = await client.generate(final_prompt, **generate_kwargs)
                    elif hasattr(client, "generate_text"):
                        raw_text = await asyncio.to_thread(
                            client.generate_text,
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text),
                        )

                    if raw_text:
                        token_count = len(str(raw_text).split())
                        latency_ms = (time.monotonic() - start) * 1000
                        surface_control_receipt = {}
                        if hasattr(client, "get_last_surface_control_receipt"):
                            try:
                                raw_receipt = client.get_last_surface_control_receipt()
                                if isinstance(raw_receipt, dict):
                                    surface_control_receipt = dict(raw_receipt)
                            except (AttributeError, RuntimeError, TypeError, ValueError) as receipt_exc:
                                _record_router_degradation(
                                    receipt_exc,
                                    action="continued generation without MLX surface-control receipt metadata",
                                    severity="warning",
                                )
                        
                        is_valid, reason = validate_response(
                            raw_text, ep.min_tokens_for_success
                        )
                        if not is_valid:
                            payload = {
                                "ok": True,
                                "text": str(raw_text).strip(),
                                "endpoint": ep.name,
                                "tokens": token_count,
                                "latency_ms": latency_ms,
                                "error": f"benchmark_invalid_response:{reason}",
                            }
                            if surface_control_receipt:
                                payload["surface_control_receipt"] = surface_control_receipt
                            if benchmark_request:
                                return payload
                            ep.record_empty()
                            return {"ok": False, "error": f"invalid_response:{reason}"}
                            
                        ep.record_success(token_count, latency_ms)
                        if (
                            ep.name == DEEP_ENDPOINT
                            and bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
                            and not kwargs.get("is_background", False)
                        ):
                            get_task_tracker().track_task(
                                get_task_tracker().create_task(
                                    self._restore_primary_after_deep_handoff(),
                                    name="llm_router.restore_primary_after_deep_handoff",
                                )
                            )
                        payload = {
                            "ok": True,
                            "text": str(raw_text).strip(),
                            "endpoint": ep.name,
                            "tokens": token_count,
                            "latency_ms": latency_ms,
                        }
                        if surface_control_receipt:
                            payload["surface_control_receipt"] = surface_control_receipt
                        return payload
                    else:
                        generation_metadata: dict[str, Any] = {}
                        metadata_getter = getattr(
                            client, "get_last_generation_metadata", None
                        )
                        if callable(metadata_getter):
                            try:
                                raw_metadata = metadata_getter()
                                if isinstance(raw_metadata, dict):
                                    generation_metadata = dict(raw_metadata)
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                generation_metadata = {}
                        _quality_rejection = str(
                            generation_metadata.get("error") or ""
                        ).strip()
                        if not _quality_rejection:
                            receipt_getter = getattr(
                                client, "get_last_surface_control_receipt", None
                            )
                            if callable(receipt_getter):
                                try:
                                    direct_receipt = receipt_getter()
                                except (AttributeError, RuntimeError, TypeError, ValueError):
                                    direct_receipt = {}
                                if (
                                    isinstance(direct_receipt, dict)
                                    and direct_receipt.get("surface_quality_gate_enabled")
                                    and not direct_receipt.get("surface_quality_gate_passed")
                                    and direct_receipt.get("surface_quality_gate_reasons")
                                ):
                                    _quality_rejection = "surface_quality_rejected"
                                    generation_metadata["surface_control_receipt"] = dict(
                                        direct_receipt
                                    )
                        if _quality_rejection in _SURFACE_QUALITY_REJECTIONS:
                            # The endpoint is healthy; something above it
                            # intentionally rejected the visible draft.
                            # Preserve that typed outcome without tripping the
                            # infrastructure circuit as "no text".
                            #
                            # Only the WORKER's own rejection was recognised
                            # here. The gate's caller-side rejections carry
                            # different names, so a Cortex that returned 458
                            # good characters was reported as
                            # "client_returned_no_text" and its circuit was
                            # opened "on transient runtime failure" — costing
                            # the NEXT turn a primary lane over a quality
                            # verdict the infrastructure had no part in.
                            return {
                                "ok": False,
                                "error": _quality_rejection,
                                "endpoint": ep.name,
                                "surface_control_receipt": dict(
                                    generation_metadata.get(
                                        "surface_control_receipt"
                                    )
                                    or {}
                                ),
                                "failure_reasons": list(
                                    generation_metadata.get("failure_reasons")
                                    or []
                                ),
                            }
                        # [BOOT RESILIENCE] Preserve hard local-lane failures so the
                        # UI and router stop reporting an endless warmup loop.
                        client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                        if client_failure:
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                        logger.debug(
                            "Endpoint %s returned no text (client warming up or rate-limited). "
                            "NOT recording as circuit failure.", ep.name
                        )
                        if benchmark_request:
                            latency_ms = (time.monotonic() - start) * 1000
                            return {
                                "ok": True,
                                "text": "",
                                "endpoint": ep.name,
                                "tokens": 0,
                                "latency_ms": latency_ms,
                                "error": "benchmark_no_text",
                            }
                        # An empty result we CHOSE — a healthy worker cancelled
                        # at this turn's budget — is a deferral, not endpoint
                        # damage. Tripping the circuit for it costs the NEXT
                        # turn the real mind as well, which is how a single
                        # 0.7s budget overrun turned into bounded filler on the
                        # desktop surface (2026-07-26).
                        deliberate = _consume_deliberate_no_text_reason(client)
                        if deliberate:
                            logger.info(
                                "Endpoint %s produced no text because we cancelled it on "
                                "purpose (%s); the lane stays warm and the circuit stays "
                                "closed.",
                                ep.name,
                                deliberate,
                            )
                            return {
                                "ok": False,
                                "error": f"deliberate_no_text:{deliberate}",
                            }
                        if ep.is_local:
                            ep.trip_temporarily("client_returned_no_text")
                        return {"ok": False, "error": "client_returned_no_text"}
                except AttributeError as ae:
                    # Missing method on client wrapper (e.g. InferenceGate) — this is NOT
                    # an inference failure, it's a code interface mismatch. Do NOT record
                    # as a circuit-breaker failure or it will permanently mark Cortex as dead.
                    logger.warning("Client adapter method missing for %s: %s", ep.name, ae)
                    return {"ok": False, "error": f"client_adapter_missing_method:{ae}"}
                except _ROUTER_CLIENT_ERRORS as e:
                    _record_router_degradation(
                        e,
                        action="raised endpoint client adapter failure to caller after recording router degradation",
                        severity="error",
                    )
                    logger.error("Client adapter call failed for %s: %s", ep.name, e)
                    raise e

            # 3. Fallback to HTTP API proxying (if no direct client)
            proxy_messages = clean_kwargs.get("messages")
            if not isinstance(proxy_messages, list) or not proxy_messages:
                # The proxy body previously dropped the system prompt
                # entirely — the same request got different instructions
                # depending on whether a direct client existed.
                proxy_messages = []
                if system_prompt:
                    proxy_messages.append(
                        {"role": "system", "content": str(system_prompt)}
                    )
                proxy_messages.append({"role": "user", "content": prompt})
            proxy_kwargs = {k: v for k, v in clean_kwargs.items() if k != "messages"}
            gateway_response = await asyncio.to_thread(
                get_network_gateway().request,
                "POST",
                f"{ep.url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": ep.model,
                    "messages": proxy_messages,
                    **proxy_kwargs,
                }),
                timeout=timeout,
                source=f"llm_provider:health_router:{ep.name}",
                read_only=True,
            )
            status_code = int(gateway_response.get("status_code") or 0)
            body = gateway_response.get("content") or b""
            body_text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)

            if status_code != 200:
                ep.record_failure(f"http_{status_code}")
                return {"ok": False, "error": f"http_{status_code}"}

            data = json.loads(body_text or "{}")
            raw_text = data.get("message", {}).get("content") or ""
            
            is_valid, reason = validate_response(raw_text, ep.min_tokens_for_success)
            latency_ms = (time.monotonic() - start) * 1000

            if not is_valid:
                if benchmark_request:
                    return {
                        "ok": True,
                        "text": raw_text.strip(),
                        "endpoint": ep.name,
                        "tokens": len(raw_text.split()),
                        "latency_ms": latency_ms,
                        "error": f"benchmark_invalid_response:{reason}",
                    }
                ep.record_empty()
                return {"ok": False, "error": f"invalid_response:{reason}"}

            token_count = data.get("eval_count") or len(raw_text.split())
            ep.record_success(token_count, latency_ms)

            return {
                "ok": True,
                "text": raw_text.strip(),
                "endpoint": ep.name,
                "tokens": token_count,
                "latency_ms": latency_ms,
            }

        except (httpx.HTTPError, OSError, ConnectionError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="recorded HTTP endpoint failure and raised for fallback handling",
                severity="error",
            )
            ep.record_failure(str(exc))
            # Tag so the outer fallback loop does not record the SAME
            # exception a second time (double-counting opened low-threshold
            # circuits at half their configured tolerance).
            exc._aura_endpoint_failure_recorded = True  # type: ignore[attr-defined]
            raise

    def _tier_display_label(self, ep: EndpointHealth | None) -> str | None:
        """Human-readable lane label derived from the ACTUAL registered model.

        Hardcoded lane labels misreported the active model whenever the model
        registry served a different local checkpoint.
        """
        if ep is None:
            return None
        model = str(getattr(ep, "model", "") or "").strip()
        tier = str(getattr(ep, "tier", "") or "")
        role = {
            "local": "Cortex",
            "local_deep": "Solver",
            "local_fast": "Brainstem",
            "emergency": "Reflex",
        }.get(tier)
        if role is None:
            role = tier.upper() or "UNKNOWN"
        return f"{role} ({model})" if model else role

    def get_health_report(self) -> dict[str, Any]:
        """Summary of router state for the GUI.

        Strictly an OBSERVER: it must not mutate circuit state (counting via
        ``is_available`` flipped OPEN circuits to HALF_OPEN and consumed
        probe leases from a GUI refresh).
        """
        active_name = self.last_user_endpoint or "Unknown"
        background_name = self.last_background_endpoint

        active_ep = self.endpoints.get(active_name) if active_name != "Unknown" else None
        tier_display = self._tier_display_label(active_ep) or "UNKNOWN"
        foreground_tier = self.last_user_tier or None
        background_tier_display = self._tier_display_label(
            self.endpoints.get(background_name) if background_name else None
        )

        # Fail CLOSED on the lane audit: an audit that cannot run, or that
        # returns no verdict, is not evidence that lanes are healthy.
        try:
            lane_audit = audit_lane_assignments()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="reported lane audit unavailable in router health report",
                severity="degraded",
            )
            lane_audit = {"ok": False, "issues": [f"lane_audit_unavailable:{exc}"]}
        return {
            "endpoints": [ep.status_dict() for ep in self.endpoints.values()],
            "available_count": sum(
                1 for ep in self.endpoints.values() if ep.peek_available()
            ),
            "probe_eligible_count": sum(
                1 for ep in self.endpoints.values() if ep.probe_eligible()
            ),
            "total_count": len(self.endpoints),
            "current_tier": tier_display,
            "foreground_tier": foreground_tier,
            "active_endpoint": active_name,
            "active_endpoint_state": (
                active_ep.state.value if active_ep is not None else "unknown"
            ),
            "foreground_endpoint": active_name,
            "background_endpoint": background_name,
            "background_tier": background_tier_display,
            "background_tier_key": self.last_background_tier,
            "last_user_error": self.last_user_error,
            "last_background_error": self.last_background_error,
            "lane_audit_ok": bool(lane_audit.get("ok", False)),
            "lane_audit_issues": list(lane_audit.get("issues", [])),
        }

def build_router_from_config(config) -> HealthAwareLLMRouter:
    """Build and return a properly configured router."""
    router = HealthAwareLLMRouter()
    primary_proof_lane = _proof_primary_lane_active(origin="llm_health_router_build")

    # [PIPELINE HARDENING] Lazy MLX runtime client wrapper.
    # Prevents all managed lanes from spawning and loading into RAM at boot.
    class LazyLocalClient:
        def __init__(self, target_path: str, **kwargs):
            self.target_path = target_path
            self.kwargs = kwargs
            self._client = None
            self._construct_lock = threading.Lock()

        def _get_client(self):
            # Singleflight: concurrent first calls must not construct two
            # multi-GB runtime clients for the same lane.
            if self._client is None:
                with self._construct_lock:
                    if self._client is None:
                        from core.brain.llm.mlx_client import get_mlx_client
                        logger.info(
                            "🧠 [LAZY LOAD] Instantiating local runtime client for %s on demand.",
                            self.target_path,
                        )
                        self._client = get_mlx_client(
                            model_path=self.target_path, **self.kwargs
                        )
            return self._client
            
        async def generate_text_async(self, prompt: str, **kwargs):
            client = await asyncio.to_thread(self._get_client)
            return await client.generate_text_async(prompt, **kwargs)
            
        def generate_text(self, prompt: str, **kwargs):
            return self._get_client().generate_text(prompt, **kwargs)

    from core.container import ServiceContainer

    # Prefer the established InferenceGate from the ServiceContainer.
    # If it exists, avoid spinning up a second primary client and warmup path.
    inference_gate = ServiceContainer.get("inference_gate", default=None)

    local_client = None
    if inference_gate is None:
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            local_client = get_mlx_client()

            warm_method = getattr(local_client, "warmup", None) or getattr(local_client, "warm_up", None)
            if callable(warm_method):
                try:
                    get_task_tracker().create_task(
                        warm_method(),
                        name="llm_router.prewarm_primary_local_runtime",
                    )
                    logger.info("✅ Scheduled background pre-warming of 72B Cortex model.")
                except RuntimeError:
                    logger.debug("No async loop running for pre-warm. Model will load on first inference.")

            logger.info("✅ Local runtime client instantiated for HealthAwareLLMRouter")
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_router_degradation(
                e,
                action="continued router build without standalone local runtime client",
                severity="degraded",
            )
            logger.error("❌ Failed to instantiate local runtime client: %s", e)
    else:
        logger.info("🛡️ HealthRouter using existing InferenceGate; skipping standalone local runtime bootstrap.")

    from core.brain.llm.model_registry import (
        get_active_model,
        get_brainstem_path,
        get_fallback_path,
    )
    active_model = get_active_model()
    brainstem_path = get_brainstem_path()
    fallback_path = get_fallback_path()

    # --- ZENITH LOCKDOWN: INFERENCE GATE REDIRECTION ---
    # We prefer the established InferenceGate from the ServiceContainer
    # instead of spawning a new standalone local worker during router setup.
    if inference_gate:
        logger.info("🛡️ HealthRouter syncing with established InferenceGate.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=inference_gate, # Direct injection of the isolated actor
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )
    else:
        # Fallback to legacy if gate not ready
        logger.warning("⚠️ InferenceGate not found in container. Falling back to legacy client.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=local_client,
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )

    if primary_proof_lane:
        logger.info(
            "🛡️ Proof-primary lane active — HealthRouter exposing only %s; "
            "Solver, Brainstem, and Reflex endpoints are not registered.",
            PRIMARY_ENDPOINT,
        )
        return router

    # Deep solver (72B) — on-demand secondary lane.
    try:
        from core.brain.llm.model_registry import get_deep_model_path
        deep_model_path = get_deep_model_path()
        router.register(
            name=DEEP_ENDPOINT,
            url="internal",
            model=deep_model_path.split("/")[-1],
            is_local=True,
            tier="local_deep",
            client=LazyLocalClient(deep_model_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 72B client.", DEEP_ENDPOINT)
    except (ImportError, AttributeError, RuntimeError) as e:
        _record_router_degradation(
            e,
            action="continued router build without deep solver lane registration",
            severity="degraded",
        )
        logger.error("❌ Failed to register %s: %s", DEEP_ENDPOINT, e)

    # Brainstem (7B) — fast local fallback.
    try:
        router.register(
            name=BRAINSTEM_ENDPOINT,
            url="internal",
            model=brainstem_path.split("/")[-1],
            is_local=True,
            tier="local_fast",
            client=LazyLocalClient(brainstem_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 7B client.", BRAINSTEM_ENDPOINT)
    except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as e:
        _record_router_degradation(
            e,
            action="continued router build without brainstem fallback lane registration",
            severity="error",
        )
        logger.error("❌ Failed to register %s: %s", BRAINSTEM_ENDPOINT, e)

    # Emergency reflex lane (1.5B / CPU-friendly).
    try:
        router.register(
            name=FALLBACK_ENDPOINT,
            url="internal",
            model=fallback_path.split("/")[-1],
            is_local=True,
            tier="emergency",
            client=LazyLocalClient(fallback_path, device="cpu"),
            failure_threshold=2,
            recovery_timeout=30.0,
        )
        logger.info("🚨 EMERGENCY Tier registered: %s lazy bypass", FALLBACK_ENDPOINT)
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        _record_router_degradation(
            e,
            action="continued router build with degraded emergency fallback coverage",
            severity="critical",
        )
        logger.error("❌ Failed to register %s: %s", FALLBACK_ENDPOINT, e)

    return router


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor
#
# Why: several call sites (e.g. core/skills/skill_evolution.py) do
# `from core.brain.llm_health_router import llm_router` at import time, expecting
# a fully-constructed router.  The real router is built later during orchestrator
# boot via build_router_from_config().  This lazy proxy bridges both styles so
# import-time references resolve to whatever router the boot registered in the
# ServiceContainer — and falls back to constructing one on first use if no
# orchestrator has booted yet (supports test harnesses and standalone scripts).
# ─────────────────────────────────────────────────────────────────────────────

_ROUTER_CONSTRUCTION_LOCK = threading.Lock()


def get_llm_router() -> HealthAwareLLMRouter:
    """Return the process-wide router, constructing it on first use if needed.

    Singleflight: concurrent first callers previously each ran the full
    check-build-register sequence, constructing multiple routers (with
    prewarm tasks) and racing the container registration.
    """
    from core.container import ServiceContainer
    existing = ServiceContainer.get("llm_router", default=None)
    if existing is not None:
        return existing
    with _ROUTER_CONSTRUCTION_LOCK:
        existing = ServiceContainer.get("llm_router", default=None)
        if existing is not None:
            return existing
        from core.config import config
        router = build_router_from_config(config)
        ServiceContainer.register_instance("llm_router", router)
        return router


class _LazyRouterProxy:
    """Attribute-access proxy that resolves to the real router on first touch."""
    __slots__ = ("_cached",)

    def __init__(self) -> None:
        self._cached = None

    def _resolve(self):
        if self._cached is None:
            self._cached = get_llm_router()
        return self._cached

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return f"<LazyRouterProxy resolved={self._cached is not None}>"


llm_router = _LazyRouterProxy()
