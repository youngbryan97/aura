from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import psutil

from core.health.degraded_events import get_unified_failure_state

logger = logging.getLogger(__name__)

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
    min_idle_seconds: float = 600.0
    max_memory_percent: float = 80.0
    max_failure_pressure: float = 0.15
    require_conversation_ready: bool = True


THOUGHT_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=180.0,
    max_memory_percent=75.0,
    max_failure_pressure=0.10,
    require_conversation_ready=False,
)

RESEARCH_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=900.0,
    max_memory_percent=75.0,
    max_failure_pressure=0.10,
    require_conversation_ready=False,
)

MAINTENANCE_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=900.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.20,
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


def background_activity_reason(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
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

    min_idle_seconds = float(min_idle_seconds if min_idle_seconds is not None else 600.0)
    max_memory_percent = float(max_memory_percent if max_memory_percent is not None else 80.0)
    max_failure_pressure = float(max_failure_pressure if max_failure_pressure is not None else 0.15)
    require_conversation_ready = bool(
        True if require_conversation_ready is None else require_conversation_ready
    )

    now = time.time()

    try:
        memory_pct = float(psutil.virtual_memory().percent)
        if memory_pct >= max_memory_percent:
            return f"memory_pressure_{memory_pct:.1f}"
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= max_failure_pressure:
            return f"failure_lockdown_{pressure:.2f}"
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    orch = orchestrator
    if orch is not None:
        if bool(getattr(orch, "is_busy", False)):
            return "orchestrator_busy"

        if float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0) > now:
            return "suppressed"

        quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
        if quiet_until > now:
            return "foreground_quiet_window"

        last_user = _last_user_interaction_time(orch)
        if last_user <= 0.0:
            return "no_user_anchor"
        if (now - last_user) < min_idle_seconds:
            return f"recent_user_{int(now - last_user)}"

    if require_conversation_ready:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if not bool(lane.get("conversation_ready", False)):
                    return f"conversation_lane_{str(lane.get('state', 'unready') or 'unready').lower()}"
        except Exception as _exc:
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
) -> bool:
    return not background_activity_reason(
        orchestrator,
        profile=profile,
        min_idle_seconds=min_idle_seconds,
        max_memory_percent=max_memory_percent,
        max_failure_pressure=max_failure_pressure,
        require_conversation_ready=require_conversation_ready,
    )
