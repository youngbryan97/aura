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

from .llm_health_router_endpoint_call import _CallsTheEndpoint
import asyncio
import inspect  # noqa: F401  (read at call time by the lifted module)
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx  # noqa: F401  (read at call time by the lifted module)

from core.brain.generation_provenance import attributed_text
from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.deferral_record import record_deferral  # noqa: F401  (read at call time by the lifted module)
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,  # noqa: F401  (read at call time by the lifted module)
    DEEP_ENDPOINT,  # noqa: F401  (read at call time by the lifted module)
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,  # noqa: F401  (read at call time by the lifted module)
    audit_lane_assignments,
    guard_solver_request,  # noqa: F401  (read at call time by the lifted module)
    normalize_endpoint_name,  # noqa: F401  (read at call time by the lifted module)
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
from core.runtime.network_gateway import get_network_gateway  # noqa: F401  (read at call time by the lifted module)
from core.runtime.progress_bound import (
    await_while_the_task_moves,
    run_on_a_thread_while_it_works,
)
from core.runtime.proof_policy import (
    is_proof_evaluation_purpose,  # noqa: F401  (read at call time by the lifted module)
    is_strict_proof_answer_prompt,  # noqa: F401  (read at call time by the lifted module)
    mlx_strict_answer_contract_enabled,  # noqa: F401  (read at call time by the lifted module)
    proof_model_tier,  # noqa: F401  (read at call time by the lifted module)
    proof_run_active,
)
from core.runtime.turn_analysis import analyze_turn
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import get_task_tracker  # noqa: F401  (read at call time by the lifted module)

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

from core.runtime.task_ownership import create_owned_asyncio_task


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
# turns measured up to 46s live (2026-06-11), so the old 20s wait starved
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
    # Substring, deliberately: `owner` is the constructed `origin:purpose`
    # key, not a sentence — `desktop:response_generation_user`,
    # `voice_loop:reply`. The words in it run into their neighbours.
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
            except (TypeError, ValueError, OverflowError) as exc:
                logger.debug("Gate timeout is not a number, treating it as none: %s", exc)
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
    except (TypeError, ValueError) as exc:
        logger.debug("Gate wait is not a number, treating it as none: %s", exc)
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
                    # Not a failure: a gate slot this thread no longer holds cannot be released, and ValueError is how the semaphore says so.
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
                    # Not a failure: a gate slot this thread no longer holds cannot be released, and ValueError is how the semaphore says so.
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
    except (TypeError, ValueError, OverflowError) as exc:
        logger.debug("Lease bounds are not numbers, issuing no lease: %s", exc)
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
        # Not a failure: a gate slot this thread no longer holds cannot be released, and ValueError is how the semaphore says so.
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


def desktop_background_endpoint_deferral_reasons(
    endpoint_names: Iterable[str],
) -> dict[str, str]:
    """Evaluate the router's endpoint admission policy for named local lanes.

    The router remains the policy owner. This public, non-actuating adapter
    lets an upstream caller make the same decision before doing expensive
    prompt assembly; it must not grow independent thresholds or assumptions.
    Only deferred endpoints are returned, so an omitted recognized endpoint is
    currently admissible.
    """
    reasons: dict[str, str] = {}
    for name in dict.fromkeys(str(item or "").strip() for item in endpoint_names):
        if name not in {BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT}:
            continue
        endpoint = EndpointHealth(name=name, url="internal", model="admission-probe")
        reason = HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(
            endpoint
        )
        if reason:
            reasons[name] = reason
    return reasons



#: What a turn of this shape costs, for sizing the ceiling above.
#:
#: The tool loop's own allowance is ``max(4, 2 * len(tools) + 2)``, and a turn
#: reaching for one capability is offered one tool.
_GENERATIONS_A_TOOL_TURN_MAY_TAKE = 4

#: The scaffold a desktop turn carries, taken off the prompts this lane logged
#: on 2026-08-29 as the loop accumulated what it had read: 3273, 3702, 5319,
#: 7334 and 8324 characters.
_A_TURNS_PROMPT_CHARS = 8324

#: The tool-call budget a foreground turn is given.
_A_TURNS_ANSWER_TOKENS = 2048


async def _await_while_it_is_working(
    coro: Any,  # a coroutine, or a task a caller already owns
    *,
    budget_s: float,
    user_facing: bool,
    person_is_waiting: bool = False,
) -> Any:
    """Renew a foreground wait from owned progress; keep probe waits bounded.

    The endpoint owns worker liveness and cancellation. This outer wait must
    drain that cancellation before returning, so a timed-out request cannot
    leave its generation holding the lane.
    """
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
    from core.runtime.turn_outcome import current_turn
    from core.runtime.turn_progress import (
        capture_progress,
        normal_gap_between_tokens,
        seconds_since_progress,
        still_producing,
    )

    progress = capture_progress()
    owned_foreground = user_facing and person_is_waiting and current_turn() is not None
    # A caller that already owns a task hands it over rather than a coroutine,
    # and wrapping a task in another task raises "a coroutine was expected".
    # The kernel does exactly that: it creates the phase task so it can name
    # and track it, then asks this to wait on it while the person waits.
    task = coro if isinstance(coro, asyncio.Task) else create_owned_asyncio_task(coro)
    started = time.monotonic()
    try:
        done, _ = await asyncio.wait({task}, timeout=budget_s)
        if done:
            return task.result()
        if not user_facing:
            raise TimeoutError

        if owned_foreground:
            # The endpoint owns first-token, token-livelock, worker-heartbeat,
            # memory-pressure and cancellation decisions.  This outer estimate
            # cannot observe native MLX work while the event loop or response
            # queue is delayed, so silence here is not evidence that the owned
            # request stopped.  Wait for an explicit endpoint terminal state or
            # for the caller to cancel the turn.
            # Two different quantities, and the runtime has one number for
            # them: how long the WORK should take, and how long a PERSON will
            # wait. The endpoint's first-token ceiling is the first — it is
            # sized from the prompt and the token budget — and this wait is
            # the second.
            #
            # LIVE, 2026-09-08: a desktop turn with a 49,136-character prompt
            # got a 900-second first-token ceiling, the 27B took the job on a
            # host at 11.8GB free with a 9B already resident, spent fifteen
            # minutes at half a core paging weights, and produced no first
            # token. `is_inference_ready()` said False for 631 seconds while
            # it happened. Nothing was broken; the person was simply not being
            # served, and nothing said so.
            #
            # Still waiting, deliberately: the endpoint owns first-token,
            # livelock, heartbeat, memory-pressure and cancellation, and this
            # outer estimate cannot see native MLX work while the loop is
            # delayed. What changes is that the wait past a person's patience
            # is now named, with the number it passed.
            logger.info(
                "Endpoint past its %.1fs estimate; waiting for its owned terminal state.",
                budget_s,
            )
            try:
                from core.brain.llm.mlx_client import longest_a_turn_may_take

                a_person_waits = float(longest_a_turn_may_take())
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Turn-length ceiling unavailable, waiting from zero: %s", exc)
                a_person_waits = 0.0
            if a_person_waits > 0.0 and budget_s < a_person_waits:
                async def _say_when_it_passes_a_persons_patience() -> None:
                    try:
                        await asyncio.sleep(max(0.0, a_person_waits - budget_s))
                    except asyncio.CancelledError:
                        # Not a failure: cancellation is the caller's decision, not a fault in the wait.
                        return
                    if not task.done():
                        logger.warning(
                            "A person has been waiting %.0fs for a first token, "
                            "past the %.0fs a turn is meant to take; still "
                            "waiting because the endpoint owns the terminal "
                            "state and has not given one.",
                            a_person_waits,
                            a_person_waits,
                        )

                watcher = create_owned_asyncio_task(_say_when_it_passes_a_persons_patience())
                try:
                    return await task
                finally:
                    watcher.cancel()
            return await task

        try:
            from core.brain.llm.thinking_reserve import seconds_to_decode

            quiet_for = normal_gap_between_tokens(float(seconds_to_decode(64)))
        except (ImportError, AttributeError, TypeError, ValueError):
            quiet_for = normal_gap_between_tokens()
        limit = float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
        if person_is_waiting:
            from core.brain.llm.mlx_client import longest_a_turn_may_take

            limit = longest_a_turn_may_take(
                generations=_GENERATIONS_A_TOOL_TURN_MAY_TAKE,
                prompt_chars=_A_TURNS_PROMPT_CHARS,
                max_tokens=_A_TURNS_ANSWER_TOKENS,
                floor_s=limit,
            )
        ceiling = max(0.0, limit - float(budget_s))
        overrun = ceiling if person_is_waiting else min(ceiling, max(0.0, float(budget_s) * 3.0))
        ends_at = time.monotonic() + overrun
        said_it_once = False
        while time.monotonic() < ends_at:
            if task.done():
                return task.result()
            if not still_producing(within_s=quiet_for, progress=progress):
                break
            if not said_it_once:
                said_it_once = True
                logger.info(
                    "Endpoint past its %.1fs estimate; owned work is still advancing.",
                    budget_s,
                )
            interval = min(2.0, max(0.0, ends_at - time.monotonic()))
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return task.result()

        since = seconds_since_progress(progress=progress)
        logger.warning(
            "Endpoint stopped %.1fs past its estimate: %s.",
            max(0.0, time.monotonic() - started - float(budget_s)),
            "no work reported for this request"
            if since < 0.0
            else f"last owned progress {since:.1f}s ago, quiet window {quiet_for:.0f}s",
        )
        raise TimeoutError
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Not a failure: cancellation is the caller's decision, not a fault in the await.
                pass
            except Exception as exc:  # noqa: BLE001 - recorded below, not swallowed
                record_degradation(
                    "llm_health_router",
                    exc,
                    action="abandoned the cancelled endpoint task",
                )

class _DeepLaneUnavailable(RuntimeError):
    """This host cannot admit the deep solver, so no lane is built for it."""


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
        except (TypeError, ValueError, OverflowError) as exc:
            logger.debug("max_tokens is not an integer, budgeting from zero: %s", exc)
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
            # A cap shorter than the time the token budget needs makes that
            # budget impossible to consume, which is the argument written four
            # lines below for long answers. It is the same argument here: at
            # the measured decode rate 1,024 tokens takes about 171 seconds,
            # and this branch capped the call at 150.
            #
            # LIVE, 2026-08-28: a tool loop reached code_repl, the code raised
            # NameError for a missing import, the error was handed back for the
            # model to fix — and the endpoint aborted at 150 seconds before it
            # could. The turn's own clock had been extended to 345.
            #
            # An unmeasured rate raises nothing, as everywhere else.
            needed_s = _seconds_a_budget_needs(max_tokens)
            if needed_s > 0.0:
                cap_s = max(cap_s, needed_s + 2.0)
            wall_s = min(wall_s, cap_s)
            cooperative_s = min(cooperative_s, max(5.0, wall_s - 2.0))
        else:
            # A long-form answer already carries a bounded owning deadline from
            # the desktop route.  Replacing it here with the ordinary 150-second
            # cap makes the requested token budget impossible to consume and
            # turns healthy slow decoding into a false endpoint failure.
            cooperative_s = min(cooperative_s, max(5.0, wall_s - 2.0))

    return cooperative_s, wall_s


def _seconds_a_budget_needs(max_tokens: Any) -> float:
    """How long this many tokens takes at the measured rate, or 0.0 unmeasured.

    The same reading the answer clock uses, so the endpoint cap and the turn's
    deadline cannot disagree about how long the same generation takes.
    """

    try:
        from core.brain.llm.thinking_reserve import seconds_to_decode

        return float(seconds_to_decode(int(max_tokens or 0)))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Decode timing unavailable, reporting no seconds: %s", exc)
        return 0.0


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


class _WatchdogHandle:
    """Cancels whichever timer is currently armed.

    The watchdog re-arms itself while the turn is producing, so the object the
    caller holds has to cancel the LATEST timer rather than the first one.
    Handing back the first was how a rearming watchdog would have outlived the
    request that started it.
    """

    def __init__(self, holder: dict[str, Any]) -> None:
        self._holder = holder

    def cancel(self) -> None:
        self._holder["cancelled"] = True
        timer = self._holder.get("timer")
        if timer is not None:
            timer.cancel()


def _start_endpoint_wall_clock_watchdog(
    client: Any,
    *,
    reason: str,
    timeout_s: float,
    user_facing: bool = False,
    person_is_waiting: bool = False,
) -> tuple[threading.Event, dict[str, bool], _WatchdogHandle]:
    """Abort non-cooperative local inference on wall-clock time.

    ``asyncio.wait_for`` only fires when the awaited coroutine yields. The local
    MLX stack can block during native/model work, so proof and desktop routes
    need a thread-backed watchdog that can terminate the active generation even
    if the event loop is temporarily occupied.
    """

    from core.runtime.turn_outcome import current_turn
    owned_foreground = user_facing and person_is_waiting and current_turn() is not None
    fired = threading.Event()
    aborted = {"value": False}
    holder: dict[str, Any] = {}

    if owned_foreground:
        # A second clock cannot diagnose an endpoint that already owns worker
        # liveness and generation cancellation.  In particular, this thread
        # cannot receive parent-side progress while the event loop is blocked;
        # killing the worker then destroys healthy work because the observer
        # was delayed.  Caller cancellation still propagates through the
        # awaited endpoint task and the MLX client's own watchdogs remain live.
        return fired, aborted, _WatchdogHandle(holder)

    def _abort() -> None:
        if holder.get("cancelled"):
            return
        # Unowned calls are probes or bounded internal work.  They cannot
        # borrow progress from an unrelated foreground turn to renew a blocked
        # call.  Owned foreground calls returned above and are governed by the
        # endpoint's correlated worker state instead.
        fired.set()
        aborted["value"] = _force_abort_endpoint_client(client, reason=reason)

    watchdog = threading.Timer(max(0.01, float(timeout_s)), _abort)
    watchdog.daemon = True
    holder["timer"] = watchdog
    watchdog.start()
    return fired, aborted, _WatchdogHandle(holder)


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
                # A lane still warming is not failing: it is skipped until it
                # is ready, which is what this trip is for. Said as a warning
                # it was a third of the warnings a live session produced, and
                # it rotated the ones worth reading out of her panel.
                log = logger.info if _only_warming(reason) else logger.warning
                log(
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



def _only_warming(reason: str) -> bool:
    """Whether a lane was skipped only because it is still coming up."""
    said = str(reason or "")
    return said.startswith(("lane_not_ready:", "warmup_deferred", "foreground_warmup_timeout"))


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


def _local_client_failure_reason(
    client: Any, *, cold_is_standby: bool = False
) -> str:
    def _get_declared_attr(candidate: Any, attr: str) -> Any:
        try:
            inspect.getattr_static(candidate, attr)
        except AttributeError:
            # Not a failure: an attribute the candidate does not have is exactly what this is looking for.
            return None
        try:
            value = getattr(candidate, attr)
        except (RuntimeError, AttributeError, TypeError) as exc:
            logger.debug("Client attribute unreadable, reporting no failure reason: %s", exc)
            return None
        if value is candidate:
            return None
        return value

    def _extract_lane_failure(candidate: Any, *, cold_is_standby: bool = False) -> str:
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
        #
        # `cold` is in that set only for a lane the runtime keeps resident.
        # For one that loads on demand it is not a transitional state at
        # all — it is where the lane rests, and sending it work is how it
        # stops being cold. LIVE 2026-09-19: "Endpoint Brainstem failed
        # validation: lane_not_ready:cold" followed by "Circuit OPEN for
        # Brainstem", over and over, so the circuit that opened because the
        # lane was cold was then what stopped the load that would have
        # warmed it. The gate's own tier-health sweep already answers this
        # question the other way, listing spawning/handshaking/warming/
        # recovering WITHOUT cold and calling a cold brainstem standby.
        transitional = {"recovering", "spawning", "handshaking", "warming"}
        if not cold_is_standby:
            transitional.add("cold")
        if not conversation_ready and state in transitional:
            return error or f"lane_not_ready:{state}"
        return ""

    try:
        seen: set[int] = set()
        candidate = client
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            failure = _extract_lane_failure(
                candidate, cold_is_standby=cold_is_standby
            )
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


def _worker_still_healthy(endpoint: Any) -> bool:
    """Whether the local worker behind this endpoint is alive and progressing.

    Asked when OUR deadline expired, to tell "it ran out of time" apart from
    "it is wedged". Unknowable counts as unhealthy, so a caller timeout on an
    endpoint that cannot answer for itself still trips the circuit.
    """
    client = getattr(endpoint, "client", None) or getattr(endpoint, "_client", None)
    # The endpoint's client is often a wrapper — an inference gate, an adapter
    # — that knows nothing about a worker's heartbeat. Walk to whatever is
    # actually holding the worker before deciding the worker is unknowable.
    for _hop in range(3):
        if client is None or callable(getattr(client, "is_alive", None)):
            break
        client = (
            getattr(client, "_mlx_client", None)
            or getattr(client, "client", None)
            or getattr(client, "_client", None)
        )
    alive = getattr(client, "is_alive", None)
    if not callable(alive):
        return False
    try:
        if not bool(alive()):
            return False
    except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.debug("Worker liveness unreadable, reporting it unhealthy: %s", exc)
        return False
    beat = max(
        float(getattr(client, "_last_heartbeat", 0.0) or 0.0),
        float(getattr(client, "_last_progress_at", 0.0) or 0.0),
        float(getattr(client, "_last_token_progress_at", 0.0) or 0.0),
    )
    if beat <= 0.0:
        # Alive, and it keeps no clock of its own. A live process that has
        # never reported a heartbeat is still a live process, and treating it
        # as wedged is the mistake this function exists to stop.
        return True
    return (time.time() - beat) < _HEALTHY_HEARTBEAT_WINDOW_S


#: How recently a local worker must have reported in for a caller timeout to
#: read as "we ran out of time" rather than "it is wedged".
_HEALTHY_HEARTBEAT_WINDOW_S = 30.0


from .llm_background_deferral import _DefersBackgroundWork


class HealthAwareLLMRouter(_CallsTheEndpoint, _DefersBackgroundWork):
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
        self._generation_metadata_sink_context: ContextVar[
            dict[str, Any] | None
        ] = ContextVar(
            f"aura_health_router_generation_metadata_sink_{id(self)}",
            default=None,
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
        sink_slot = getattr(self, "_generation_metadata_sink_context", None)
        sink = sink_slot.get() if sink_slot is not None else None
        if isinstance(sink, dict):
            sink.clear()
            sink.update(snapshot)

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
            from core.runtime.service_access import resolve_inference_gate

            _abort_client(resolve_inference_gate())
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Inference gate unreachable, active generation not aborted: %s", exc)
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

        side_effect_free_completion = bool(
            kwargs.get("user_surface_completion_retry", False)
            or kwargs.get("user_surface_continuation_contract", False)
        )
        tool_handoff_allowed = bool(kwargs.get("allow_tools", True)) and not (
            side_effect_free_completion
        )
        # An internal generation is never a turn that must call a tool first.
        #
        # The handoff exists so she cannot answer a PERSON's question without
        # the evidence it needs. Her own authoring prompt contains no such
        # question: "Write the CONTENT of a document about ... The full
        # request was: make a file on my Desktop called aura_note.txt" was
        # read as a turn about a file, handed a tool, and refused to generate
        # at all — "grounding_required_no_tool_result". The tool it wanted is
        # the step that writes down what this call returns.
        #
        # `_non_chat_inference` already declares this and already suppresses
        # the reply contract one layer down; it had no say here.
        # The caller's own declaration, not the derived flag.
        #
        # `internal_inference` is set from several places, including for turns
        # that ARE somebody asking — keying on it stopped the chat lane
        # handing off for "run some python", which is the exact turn the
        # handoff exists for. `_non_chat_inference` is passed by the caller
        # and means only this.
        not_a_person_asking = inferred_background or non_chat_inference
        if should_force_tool_handoff(
            contract, is_background=not_a_person_asking
        ) and (tool_handoff_allowed and not _contract_tool_handoff_val):
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
        *,
        _generation_metadata_sink: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | None:
        """Bind caller-owned evidence transport for exactly one router call."""

        sink_slot = getattr(self, "_generation_metadata_sink_context", None)
        if sink_slot is None:
            sink_slot = ContextVar(
                f"aura_health_router_generation_metadata_sink_{id(self)}",
                default=None,
            )
            self._generation_metadata_sink_context = sink_slot
        sink_token = sink_slot.set(
            _generation_metadata_sink
            if isinstance(_generation_metadata_sink, dict)
            else None
        )
        try:
            return await self._think_with_generation_metadata_sink(
                prompt=prompt,
                system_prompt=system_prompt,
                prefer_tier=prefer_tier,
                schema=schema,
                **kwargs,
            )
        finally:
            sink_slot.reset(sink_token)

    async def _think_with_generation_metadata_sink(
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
                metadata = (
                    result
                    if isinstance(result, dict)
                    else self.get_last_generation_metadata()
                )
                return attributed_text(stripped, metadata)
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




    @staticmethod
    def _deterministic_intent_classification(prompt: str) -> str:
        if not str(prompt or "").strip():
            return "casual"
        return analyze_turn(prompt).semantic_mode









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
                # Not a failure: an attribute the client does not have means there is nothing nested to unwrap.
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
        Return the system to the cortex conversational brain after a 72B handoff.
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
            availability = await run_on_a_thread_while_it_works(
                checker, stall_s=5.0, name="llm_health.is_available"
            )
            if inspect.isawaitable(availability):
                availability = await await_while_the_task_moves(
                    availability, stall_s=5.0, name="llm_health.is_available"
                )
            return bool(availability)
        except TimeoutError:
            # Not a failure: an availability probe that does not answer inside its bound is unavailable, which is what the bound is for.
            return False
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            _record_router_degradation(
                exc,
                action="treated crashed client availability check as unknown; generation call will decide",
                severity="degraded",
            )
            return None

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

    from core.runtime.service_access import resolve_inference_gate

    # Prefer the established InferenceGate. If it exists, avoid spinning up a
    # second primary client and warmup path.
    inference_gate = resolve_inference_gate()

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

    # Optional local reasoning specialist — registered only when a distinct,
    # complete artifact can be admitted. Resident deep reasoning remains on
    # Cortex when this endpoint does not exist.
    #
    # LIVE, 2026-08-20. Registered unconditionally on a 64GB host, where the
    # 72B needs 48.4GB beside a resident 25.3GB cortex against a 46.1GB lane
    # budget. Admission refused every load, correctly, and the route that
    # asked came back with nothing — so a chat turn offered five tools
    # generated no text at all and ended in an apology. A lane that cannot
    # load is not a fallback; it is a hole every route falls through.
    try:
        from core.brain.inference_gate import local_deep_solver_enabled

        deep_lane_possible = local_deep_solver_enabled()
    except _ROUTER_CLIENT_ERRORS as exc:
        logger.debug("Deep solver availability unreadable, not offering the deep lane: %s", exc)
        deep_lane_possible = False
    if not deep_lane_possible:
        logger.info(
            "⏭️  %s not registered: this host cannot admit the deep solver "
            "beside the resident cortex.",
            DEEP_ENDPOINT,
        )
    try:
        from core.brain.llm.model_registry import get_deep_model_path

        if not deep_lane_possible:
            raise _DeepLaneUnavailable(DEEP_ENDPOINT)
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
        logger.info("✅ %s registered with an admitted local specialist.", DEEP_ENDPOINT)
    except _DeepLaneUnavailable:
        # Not a failure: the deep lane raised its own unavailability above; this is where that is handled.
        pass
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
