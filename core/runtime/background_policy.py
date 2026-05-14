from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import psutil

from core.health.degraded_events import get_unified_failure_state

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Invalid %s=%r; using %.1f", name, raw, default)
        return float(default)

_USER_FACING_ORIGIN_TOKENS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "external",
    "direct",
    "embodied",
    "reflex",
    "motor",
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "consolidation",
    "context",
    "dream",
    "growth",
    "impulse",
    "internal",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "pruner",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    # "system" intentionally omitted: it is too broad and would misclassify
    # user-adjacent routing paths that still use the historical default.
    "terminal",
    "volition",
    "witness",
})


@dataclass(frozen=True)
class BackgroundPolicyProfile:
    min_idle_seconds: float = 10.0
    max_memory_percent: float = 90.0
    max_failure_pressure: float = 0.60
    require_conversation_ready: bool = False


THOUGHT_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=5.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=False,
)

RESEARCH_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=15.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=False,
)

MAINTENANCE_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=60.0,
    max_memory_percent=92.0,
    max_failure_pressure=0.75,
    require_conversation_ready=False,
)


def normalize_origin(origin: Any) -> str:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    while normalized.startswith("routing_"):
        normalized = normalized[len("routing_"):]
    return normalized


def origin_tokens(origin: Any) -> set[str]:
    normalized = normalize_origin(origin)
    return {token for token in normalized.split("_") if token}


def is_user_facing_origin(origin: Any) -> bool:
    normalized = normalize_origin(origin)
    if not normalized:
        return False
    if normalized in _USER_FACING_ORIGIN_TOKENS:
        return True
    return bool(origin_tokens(normalized) & _USER_FACING_ORIGIN_TOKENS)


def is_background_origin(origin: Any, *, explicit_background: bool = False) -> bool:
    if explicit_background:
        return True
    tokens = origin_tokens(origin)
    if not tokens:
        return False
    if tokens & _USER_FACING_ORIGIN_TOKENS:
        return False
    return bool(tokens & _BACKGROUND_ORIGIN_HINTS)


def _last_user_interaction_time(orchestrator: Any = None) -> float:
    orch = orchestrator
    if orch is None:
        return 0.0

    value = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
    if value > 0.0:
        return value

    status = getattr(orch, "status", None)
    if status is not None:
        value = float(getattr(status, "last_user_interaction_time", 0.0) or 0.0)
        if value > 0.0:
            return value

    return 0.0


def _runtime_uptime_seconds(orchestrator: Any = None) -> float:
    if orchestrator is None:
        return 0.0

    candidates = [
        getattr(orchestrator, "start_time", None),
        getattr(getattr(orchestrator, "status", None), "start_time", None),
    ]
    for candidate in candidates:
        try:
            start = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if start > 0.0:
            return max(0.0, time.time() - start)
    return 0.0


def _foreground_activity_reason() -> str:
    try:
        from core.runtime.foreground_guard import foreground_activity_reason

        guard_reason = foreground_activity_reason()
        if guard_reason:
            return guard_reason
    except Exception as _exc:
        record_degradation('background_policy', _exc)
        logger.debug("Suppressed Exception: %s", _exc)

    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            lane = dict(gate.get_conversation_status() or {})
            if bool(lane.get("foreground_owned")) or int(lane.get("active_generations", 0) or 0) > 0:
                return "foreground_generation_active"
            if bool(lane.get("kernel_lock_held")):
                return "foreground_kernel_lock"
            request_age = float(lane.get("request_age_s", 0.0) or 0.0)
            if request_age > 0.0 and str(lane.get("foreground_owner") or "").strip():
                return "foreground_request_active"
    except Exception as _exc:
        record_degradation('background_policy', _exc)
        logger.debug("Suppressed Exception: %s", _exc)
    return ""


def background_activity_reason(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
) -> str:
    if profile is not None:
        if min_idle_seconds is None:
            min_idle_seconds = profile.min_idle_seconds
        if max_memory_percent is None:
            max_memory_percent = profile.max_memory_percent
        if max_failure_pressure is None:
            max_failure_pressure = profile.max_failure_pressure
        if require_conversation_ready is None:
            require_conversation_ready = profile.require_conversation_ready

    min_idle_seconds = float(min_idle_seconds if min_idle_seconds is not None else 10.0)
    max_memory_percent = float(max_memory_percent if max_memory_percent is not None else 90.0)
    max_failure_pressure = float(max_failure_pressure if max_failure_pressure is not None else 0.60)
    require_conversation_ready = bool(
        False if require_conversation_ready is None else require_conversation_ready
    )

    now = time.time()

    foreground_reason = _foreground_activity_reason()
    if foreground_reason:
        return foreground_reason

    orch = orchestrator
    if orch is not None:
        boot_grace_s = _env_float("AURA_BACKGROUND_BOOT_GRACE_S", 300.0)
        uptime_s = _runtime_uptime_seconds(orch)
        if boot_grace_s > 0.0 and 0.0 < uptime_s < boot_grace_s:
            return f"boot_grace_{int(uptime_s)}s"

        if bool(getattr(orch, "is_busy", False)):
            return "orchestrator_busy"

        if float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0) > now:
            return "suppressed"

        quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
        if quiet_until > now:
            return "foreground_quiet_window"

        last_user = _last_user_interaction_time(orch)
        if last_user <= 0.0:
            if not allow_no_user_anchor:
                return "no_user_anchor"
        elif (now - last_user) < min_idle_seconds:
            return f"recent_user_{int(now - last_user)}"

    try:
        memory_pct = float(psutil.virtual_memory().percent)
        if memory_pct >= max_memory_percent:
            return f"memory_pressure_{memory_pct:.1f}"
    except Exception as _exc:
        record_degradation('background_policy', _exc)
        logger.debug("Suppressed Exception: %s", _exc)

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= max_failure_pressure:
            return f"failure_lockdown_{pressure:.2f}"
    except Exception as _exc:
        record_degradation('background_policy', _exc)
        logger.debug("Suppressed Exception: %s", _exc)

    if require_conversation_ready:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if not bool(lane.get("conversation_ready", False)):
                    return f"conversation_lane_{str(lane.get('state', 'unready') or 'unready').lower()}"
        except Exception as _exc:
            record_degradation('background_policy', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    return ""


def background_activity_allowed(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
) -> bool:
    return not background_activity_reason(
        orchestrator,
        profile=profile,
        min_idle_seconds=min_idle_seconds,
        max_memory_percent=max_memory_percent,
        max_failure_pressure=max_failure_pressure,
        require_conversation_ready=require_conversation_ready,
        allow_no_user_anchor=allow_no_user_anchor,
    )
