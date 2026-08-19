"""interface/routes/system.py
─────────────────────────────
Extracted from server.py — Health, telemetry, metrics, bootstrap,
and all collector/diagnostic helpers.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import fastapi.responses as fastapi_responses
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import config
from core.container import ServiceContainer
from core.health.boot_status import build_boot_health_snapshot
from core.health.conversation_lane import conversation_lane_is_busy
from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel
from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.health_contract import (
    REQUIRED_HEALTH_PROBE_GROUPS,
    required_probe_blockers,
    required_probe_groups_pass,
)
from core.runtime.launch_provenance import (
    capture_runtime_shell_assets as _capture_runtime_shell_assets,
    runtime_shell_assets_sha256 as _runtime_shell_assets_sha256,
)
from core.runtime.runtime_shell_snapshot import (
    clear_runtime_shell_snapshots as _clear_runtime_shell_snapshots,
    publish_runtime_shell_snapshot as _publish_runtime_shell_snapshot,
)
from core.runtime.service_access import optional_service
from core.runtime.shutdown_coordinator import (
    get_shutdown_coordinator,
    is_shutdown_requested,
)
from core.runtime.task_ownership import create_tracked_task
from core.runtime.version import VERSION, version_string
from core.scheduler import scheduler
from core.tools.runtime_tools import get_runtime_state
from interface.auth import (
    _require_internal,
    _restore_owner_session_from_request,
    paired_device_session_id,
    request_access_profile,
)
from interface.routes.devices import _owner_authenticated
from interface.websocket_manager import broadcast_bus, runtime_heartbeat_payload, ws_manager

_SYSTEM_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    json.JSONDecodeError,
    psutil.Error,
    subprocess.SubprocessError,
)

_TOOL_CATALOG_BOOTSTRAP_MAX_ITEMS = 256
_TOOL_CATALOG_BOOTSTRAP_READ_BUDGET_S = 0.35
_RUNTIME_REVISION_LOCK = threading.Lock()
_RUNTIME_REVISION_CACHE: dict[str, Any] | None = None
_RUNTIME_REVISION_CACHE_COLLECTED_AT = 0.0
_RUNTIME_REVISION_INVALIDATION_PENDING = False
_RUNTIME_REVISION_VERIFIED_TTL_S = 30.0
_RUNTIME_REVISION_UNVERIFIED_TTL_S = 2.0
# Consecutive unverified captures, for the retry backoff below. A recollection
# is two full provenance captures plus three SHA256 walks of the shell asset
# tree, so a permanently-unverified launch retrying every 2s is a permanent
# background cost with no possible payoff.
_RUNTIME_REVISION_UNVERIFIED_STREAK = 0
_RUNTIME_REVISION_FAST_RETRIES = 3


def _shutdown_health_status() -> dict[str, object]:
    try:
        return get_shutdown_coordinator().get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        return {
            "running": False,
            "request": {"requested": is_shutdown_requested()},
            "report": None,
            "error": repr(exc),
        }


def _stopping_boot_health_payload() -> tuple[dict[str, Any], int] | None:
    shutdown = _shutdown_health_status()
    request = shutdown.get("request")
    if not isinstance(request, dict) or request.get("requested") is not True:
        return None
    return (
        {
            "ready": False,
            "status": "stopping",
            "system_ready": False,
            "launcher_ready": False,
            "conversation_ready": False,
            "boot_phase": "runtime_shutdown",
            "required_probes": {"all_passed": False},
            "blockers": ["runtime_shutdown"],
            "shutdown": shutdown,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        503,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _env_positive_float(name: str, default: float) -> float:
    value = _safe_float(os.getenv(name, ""), default)
    return value if value > 0.0 else default


async def _optional_threaded_status(
    label: str,
    fn: Any,
    *,
    timeout_s: float = 0.18,
    fallback: dict[str, Any] | None = None,
    offload: bool = True,
) -> dict[str, Any]:
    """Read optional health-panel data without blocking the API loop."""

    fallback_payload = {"_stale": True, "reason": "status_unavailable"}
    if fallback:
        fallback_payload.update(fallback)
    if not offload:
        try:
            result = fn()
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                fallback_payload["reason"] = "async_status_not_supported"
                return fallback_payload
            return (
                dict(result or {})
                if isinstance(result, dict)
                else {"value": result, "_stale": False}
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system.optional_status", exc)
            logger.debug("Optional health status %s failed: %s", label, exc)
            fallback_payload["reason"] = f"{type(exc).__name__}: {exc}"
            return fallback_payload
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fn),
            timeout=max(0.05, float(timeout_s)),
        )
        return dict(result or {}) if isinstance(result, dict) else {"value": result, "_stale": False}
    except TimeoutError:
        logger.debug("Optional health status %s timed out after %.2fs", label, timeout_s)
        fallback_payload["reason"] = "status_timeout"
        return fallback_payload
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.optional_status", exc)
        logger.debug("Optional health status %s failed: %s", label, exc)
        fallback_payload["reason"] = f"{type(exc).__name__}: {exc}"
        return fallback_payload


def _runtime_component_status(
    service_name: str,
    *status_methods: str,
) -> dict[str, Any]:
    """Read a registered background component without instantiating a new one."""

    service = ServiceContainer.peek(service_name, default=None)
    if service is None:
        return {"registered": False, "running": False, "reason": "not_registered"}
    for method_name in status_methods:
        method = getattr(service, method_name, None)
        if not callable(method):
            continue
        try:
            status = method()
            if inspect.isawaitable(status):
                close = getattr(status, "close", None)
                if callable(close):
                    close()
                return {
                    "registered": True,
                    "running": False,
                    "reason": "async_status_not_supported_in_health_snapshot",
                }
            if isinstance(status, dict):
                return {"registered": True, **status}
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.background_runtime",
                exc,
                action=f"marked {service_name} status unavailable",
            )
            return {
                "registered": True,
                "running": False,
                "reason": f"status_error:{type(exc).__name__}",
            }
    task = getattr(service, "_task", None) or getattr(service, "_background_task", None)
    running = bool(
        getattr(service, "_running", False)
        or getattr(service, "_started", False)
        or (task is not None and not task.done())
    )
    return {"registered": True, "running": running}


def _collect_full_runtime_status(
    pneuma_data: dict[str, Any],
    mhaf_data: dict[str, Any],
) -> dict[str, Any]:
    """Report whether a normal desktop launch actually started Aura's organs."""

    from core.runtime.background_policy import (
        background_activity_reason,
        background_cognition_disabled_reason,
        background_loop_start_reason,
        foreground_only_runtime,
    )
    from core.runtime.desktop_boot_safety import (
        desktop_resource_guard_enabled,
        desktop_safe_boot_enabled,
    )

    conductor = _runtime_component_status("autonomy_conductor", "status")
    conductor["running"] = bool(conductor.get("active", False))
    overt = _runtime_component_status("overt_action_loop", "status")
    jobs = conductor.get("jobs", {}) if isinstance(conductor.get("jobs"), dict) else {}
    overt["scheduled"] = bool(
        conductor.get("active") and "overt_action_cycle" in jobs
    )
    overt["running"] = bool(overt.get("scheduled") and overt.get("enabled", False))
    agency = ServiceContainer.peek("agency_core", default=None)
    swarm = getattr(agency, "swarm", None)
    deliberation = (
        swarm.get_status()
        if swarm is not None and callable(getattr(swarm, "get_status", None))
        else {"available": False, "active_shards": 0}
    )
    deliberation["registered"] = bool(swarm is not None)
    deliberation["scheduled"] = bool(
        conductor.get("active") and "internal_deliberation_cycle" in jobs
    )
    deliberation["running"] = bool(
        deliberation.get("registered") and deliberation.get("scheduled")
    )

    components = {
        "pneuma": {
            "registered": ServiceContainer.peek("pneuma", default=None) is not None,
            "running": bool(pneuma_data.get("online")),
            **pneuma_data,
        },
        "mhaf": {
            "registered": ServiceContainer.peek("mhaf", default=None) is not None,
            "running": bool(mhaf_data.get("online")),
            **mhaf_data,
        },
        "curiosity": _runtime_component_status("curiosity_engine", "get_status"),
        "proactive_communication": _runtime_component_status("proactive_comm", "get_status"),
        "autonomous_initiative": _runtime_component_status(
            "autonomous_initiative_loop",
            "get_status",
        ),
        "subjective_choice": _runtime_component_status(
            "subjective_choice_engine",
            "get_status",
            "status",
        ),
        "ambient_life_director": _runtime_component_status(
            "ambient_life_director",
            "get_status",
            "status",
        ),
        "research": _runtime_component_status("research_cycle", "get_status"),
        "self_healing": _runtime_component_status("self_healing", "get_status"),
        "self_modification": _runtime_component_status(
            "self_modification_engine", "runtime_status"
        ),
        "consciousness_stream": _runtime_component_status("consciousness"),
        "autonomy_conductor": conductor,
        "overt_action": overt,
        "deliberation": deliberation,
        "wake_word": _runtime_component_status("wake_word", "get_status"),
        "screen_perception": _runtime_component_status("screen_perception", "get_status"),
        "perceptual_pump": _runtime_component_status("perceptual_pump", "get_status"),
        "cognitive_situation": _runtime_component_status(
            "cognitive_situation",
            "get_status",
            "status",
        ),
        "imagination_engine": _runtime_component_status(
            "imagination_engine",
            "get_status",
            "status",
            "snapshot",
        ),
        "timescale_bridge": _runtime_component_status(
            "timescale_bridge",
            "get_status",
            "status",
        ),
        "ambient_developer_stream": _runtime_component_status(
            "ambient_developer_stream",
            "get_status",
            "status",
        ),
        "autonomic_reflection_loop": _runtime_component_status(
            "autonomic_reflection_loop",
            "get_status",
            "status",
        ),
    }
    resource_guard = desktop_resource_guard_enabled()
    expected = (
        resource_guard
        and not foreground_only_runtime()
        and not background_cognition_disabled_reason()
    )
    required = (
        "pneuma",
        "mhaf",
        "curiosity",
        "proactive_communication",
        "autonomous_initiative",
        "subjective_choice",
        "ambient_life_director",
        "research",
        "self_healing",
        "self_modification",
        "consciousness_stream",
        "autonomy_conductor",
        "overt_action",
        "deliberation",
        "wake_word",
        "screen_perception",
        "perceptual_pump",
        "cognitive_situation",
        "imagination_engine",
        "timescale_bridge",
        "ambient_developer_stream",
        "autonomic_reflection_loop",
    )
    blockers = [name for name in required if not components[name].get("running", False)]
    running_required = [
        name for name in required if components[name].get("running", False)
    ]
    disabled_reason = background_cognition_disabled_reason(
        allow_desktop_safe_boot=True,
    )
    loop_start_reason = background_loop_start_reason(
        allow_desktop_safe_boot=True,
    )
    orchestrator = ServiceContainer.peek("orchestrator", default=None)
    activity_reason = background_activity_reason(
        orchestrator,
        min_idle_seconds=0.0,
        allow_no_user_anchor=True,
        allow_desktop_safe_boot=True,
    )
    background_enabled = bool(
        resource_guard and not foreground_only_runtime() and not disabled_reason
    )
    return {
        "profile": (
            "foreground_only"
            if foreground_only_runtime()
            else "protected_full_desktop"
            if desktop_safe_boot_enabled() and resource_guard
            else "full_desktop"
            if resource_guard
            else "recovery_safe_boot"
            if desktop_safe_boot_enabled()
            else "server_or_test"
        ),
        "full_runtime_expected": expected,
        "resource_guard_enabled": resource_guard,
        "ready": bool(expected and not blockers),
        "blockers": blockers,
        "background_cognition": {
            "enabled": background_enabled,
            "active": bool(background_enabled and not blockers),
            "loops_allowed": not bool(loop_start_reason),
            "loop_start_reason": loop_start_reason,
            "work_admission": "deferred" if activity_reason else "allowed",
            "work_defer_reason": activity_reason,
            "registered_required_count": len(required),
            "running_required_count": len(running_required),
            "offline_required": blockers,
        },
        "components": components,
    }


try:
    ORJSONResponse = fastapi_responses.ORJSONResponse
except _SYSTEM_RECOVERABLE_ERRORS:
    ORJSONResponse = JSONResponse

logger = logging.getLogger("Aura.Server.System")

router = APIRouter()

_DESKTOP_ACCESS_CACHE_TTL_S = _env_positive_float("AURA_DESKTOP_ACCESS_CACHE_TTL_S", 30.0)
_DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S",
    15.0,
)
_DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S",
    6.0,
)
_DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S",
    2.0,
)
_DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S",
    0.5,
)
_SSE_IDLE_HEARTBEAT_S = _env_positive_float("AURA_SSE_IDLE_HEARTBEAT_S", 15.0)
_SSE_QUEUE_BACKLOG_LIMIT = max(1, _safe_int(os.getenv("AURA_SSE_QUEUE_BACKLOG_LIMIT", ""), 100))
_HEALTH_PROBE_TIMEOUT_S = _env_positive_float("AURA_HEALTH_PROBE_TIMEOUT_S", 2.5)
_HEALTH_PROBE_DEGRADATION_THRESHOLD = max(
    2,
    _safe_int(os.getenv("AURA_HEALTH_PROBE_DEGRADATION_THRESHOLD", ""), 3),
)
_HEALTH_PROBE_STUCK_THRESHOLD_S = max(
    10.0,
    _env_positive_float(
        "AURA_HEALTH_PROBE_STUCK_THRESHOLD_S",
        max(30.0, _HEALTH_PROBE_TIMEOUT_S * 8.0),
    ),
)
_HEALTH_PROBE_LOCKS = {
    False: threading.Lock(),
    True: threading.Lock(),
}
# Backward-compatible canonical-runtime lock for direct contract tests.
_HEALTH_PROBE_LOCK = _HEALTH_PROBE_LOCKS[False]
_HEALTH_PROBE_STATE_LOCK = threading.Lock()
_HEALTH_PROBE_STATE: dict[str, Any] = {
    "generation": 0,
    "consecutive_failures": 0,
    "total_timeouts": 0,
    "total_contentions": 0,
    "total_terminal_failures": 0,
    "timeout_recorded_generation": 0,
    "stuck_recorded_generation": 0,
    "last_failure_reason": "",
    "last_failure_at_unix": 0.0,
    "escalated": False,
}
_HEALTH_PROBE_FUTURES: dict[bool, Future[tuple[dict[str, Any], int]]] = {}
_HEALTH_PROBE_GENERATIONS: dict[bool, int] = {}
_HEALTH_PROBE_STARTED_AT: dict[bool, float] = {}
_HEALTH_PROBE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, min(4, _safe_int(os.getenv("AURA_HEALTH_PROBE_WORKERS", ""), 2))),
    thread_name_prefix="AuraHealthProbe",
)
_HEALTH_CACHE_TTL_S = _env_positive_float("AURA_HEALTH_CACHE_TTL_S", 5.0)
_HEALTH_STALE_CACHE_TTL_S = max(
    _HEALTH_CACHE_TTL_S,
    _env_positive_float("AURA_HEALTH_STALE_CACHE_TTL_S", 30.0),
)
_HEALTH_MANIFEST_FALLBACK_TTL_S = _env_positive_float(
    "AURA_HEALTH_MANIFEST_FALLBACK_TTL_S",
    15.0,
)
_UI_SHELL_ERROR_BODY = Body(default=None)
_boot_health_cache_lock = threading.Lock()
_boot_health_cache: dict[bool, dict[str, Any]] = {
    False: {"captured_at": 0.0, "payload": None, "status_code": 503},
    True: {"captured_at": 0.0, "payload": None, "status_code": 503},
}
_desktop_access_cache: dict[str, Any] = {
    "captured_at": 0.0,
    "payload": None,
}


def reset_health_caches() -> None:
    """Drop every memoised health/access payload.

    These caches are correct in the runtime — a 5s TTL is what keeps a polling
    desktop from re-probing the whole boot contract on every tick — and wrong
    across a process that runs many independent scenarios in sequence. A test
    that installs a ready boot snapshot and then reads a payload captured by
    an unrelated test seconds earlier is reading the cache, not the code:
    observed as ``status == "booting"`` in a test that passes alone and fails
    in company.
    """
    with _boot_health_cache_lock:
        for key in (False, True):
            _boot_health_cache[key] = {
                "captured_at": 0.0,
                "payload": None,
                "status_code": 503,
            }
    _desktop_access_cache["captured_at"] = 0.0
    _desktop_access_cache["payload"] = None
_DESKTOP_ACCESS_PROBE_TASKS: dict[
    asyncio.AbstractEventLoop,
    asyncio.Task[dict[str, Any]],
] = {}
_DESKTOP_ACCESS_PROBE_STATE_LOCK = threading.Lock()
_DESKTOP_ACCESS_PROBE_STATE: dict[str, Any] = {
    "total_timeouts": 0,
    "total_failures": 0,
    "active_streaks": {},
    "last_issue": "",
    "last_issue_at_unix": 0.0,
}
_desktop_access_request_state: dict[str, Any] = {}


def _health_probe_state_snapshot() -> dict[str, Any]:
    now = time.monotonic()
    with _HEALTH_PROBE_STATE_LOCK:
        active_entries = {
            bool(surface): {
                "generation": int(_HEALTH_PROBE_GENERATIONS.get(surface) or 0),
                "started_at": float(_HEALTH_PROBE_STARTED_AT.get(surface) or 0.0),
            }
            for surface, future in _HEALTH_PROBE_FUTURES.items()
            if not future.done()
        }
        active_since = min(
            (
                float(entry["started_at"])
                for entry in active_entries.values()
                if float(entry["started_at"]) > 0.0
            ),
            default=0.0,
        )
        active_generations = sorted(
            int(entry["generation"])
            for entry in active_entries.values()
            if int(entry["generation"]) > 0
        )
        return {
            "active": bool(active_entries),
            "active_count": len(active_entries),
            "active_age_s": round(max(0.0, now - active_since), 3)
            if active_since > 0.0
            else 0.0,
            "consecutive_failures": int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ),
            "active_generation": active_generations[-1] if active_generations else 0,
            "active_generations": active_generations,
            "active_surfaces": sorted(
                "gui_proxy" if surface else "runtime"
                for surface in active_entries
            ),
            "generation": int(_HEALTH_PROBE_STATE.get("generation") or 0),
            "total_timeouts": int(_HEALTH_PROBE_STATE.get("total_timeouts") or 0),
            "total_contentions": int(
                _HEALTH_PROBE_STATE.get("total_contentions") or 0
            ),
            "total_terminal_failures": int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ),
            "last_failure_reason": str(
                _HEALTH_PROBE_STATE.get("last_failure_reason") or ""
            ),
            "last_failure_at_unix": float(
                _HEALTH_PROBE_STATE.get("last_failure_at_unix") or 0.0
            ),
            "escalated": bool(_HEALTH_PROBE_STATE.get("escalated", False)),
            "degradation_threshold": _HEALTH_PROBE_DEGRADATION_THRESHOLD,
            "stuck_threshold_s": _HEALTH_PROBE_STUCK_THRESHOLD_S,
        }


def _attach_health_probe_state(
    result: tuple[dict[str, Any], int],
) -> tuple[dict[str, Any], int]:
    payload, status_code = result
    enriched = dict(payload)
    enriched["health_probe_runtime"] = _health_probe_state_snapshot()
    return enriched, status_code


def _reset_health_probe_state_for_test() -> None:
    with _HEALTH_PROBE_STATE_LOCK:
        for surface, future in list(_HEALTH_PROBE_FUTURES.items()):
            if future.done():
                _HEALTH_PROBE_FUTURES.pop(surface, None)
                _HEALTH_PROBE_GENERATIONS.pop(surface, None)
                _HEALTH_PROBE_STARTED_AT.pop(surface, None)
        _HEALTH_PROBE_STATE.update(
            generation=0,
            consecutive_failures=0,
            total_timeouts=0,
            total_contentions=0,
            total_terminal_failures=0,
            timeout_recorded_generation=0,
            stuck_recorded_generation=0,
            last_failure_reason="",
            last_failure_at_unix=0.0,
            escalated=False,
        )


def _reset_boot_health_cache_for_test() -> None:
    with _boot_health_cache_lock:
        for entry in _boot_health_cache.values():
            entry.update(
                captured_at=0.0,
                payload=None,
                status_code=503,
            )


def _desktop_access_empty_payload() -> dict[str, Any]:
    return {
        "screen_recording": {"granted": False, "status": "unknown", "guidance": ""},
        "accessibility": {"granted": False, "status": "unknown", "guidance": ""},
        "automation": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_screen_recording": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_accessibility": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_automation": {"granted": False, "status": "unknown", "guidance": ""},
        "screen_capture_ready": False,
        "desktop_control_ready": False,
        "screen_text_ready": False,
        "direct_screen_capture_ready": False,
        "direct_desktop_control_ready": False,
        "direct_screen_text_ready": False,
        "menu_clock_ready": False,
        "menu_clock_text": "",
        "menu_clock_error": "",
        "frontmost_app": "",
        "pyautogui_ready": False,
        "pyautogui_error": "",
        "permission_confidence": "unknown",
        "permission_assumptions": [],
        "process_identity": {},
        "effective_app_identity": {},
        "desktop_access_diagnosis": [],
        "tcc_repair_plan": {},
        "tcc_request_state": dict(_desktop_access_request_state),
        "native_bridge_probe": {},
        "overall_status": "pending",
        "blocking_permissions": [],
        "reported_blocking_permissions": [],
        "direct_blocking_permissions": [],
        "unverified_permissions": [],
        "reported_probe_unavailable_permissions": [],
        "direct_probe_unavailable_permissions": [],
        "direct_probe_available": False,
        "cache_age_s": 0.0,
        "cache_stale": False,
        "probe_mode": "empty",
        "probe_runtime": _desktop_access_probe_state_snapshot(),
    }


def _desktop_access_cache_ttl(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    if (
        payload.get("overall_status") == "ready"
        and payload.get("permission_confidence") == "direct"
        and not payload.get("blocking_permissions")
    ):
        return max(1.0, _DESKTOP_ACCESS_CACHE_TTL_S)
    return max(0.25, min(_DESKTOP_ACCESS_CACHE_TTL_S, _DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S))


def _desktop_access_cached_copy(
    payload: dict[str, Any],
    *,
    captured_at: float,
    stale: bool = False,
    probe_mode: str = "cached",
) -> dict[str, Any]:
    copied = dict(payload)
    age = max(0.0, time.monotonic() - float(captured_at or 0.0))
    copied["cache_age_s"] = round(age, 3)
    copied["cache_stale"] = bool(stale)
    copied["probe_mode"] = probe_mode
    copied["cache_ttl_s"] = _desktop_access_cache_ttl(payload)
    return copied


def _desktop_access_probe_state_snapshot() -> dict[str, Any]:
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        return {
            "total_timeouts": int(
                _DESKTOP_ACCESS_PROBE_STATE.get("total_timeouts", 0) or 0
            ),
            "total_failures": int(
                _DESKTOP_ACCESS_PROBE_STATE.get("total_failures", 0) or 0
            ),
            "active_streaks": dict(
                _DESKTOP_ACCESS_PROBE_STATE.get("active_streaks", {}) or {}
            ),
            "last_issue": str(
                _DESKTOP_ACCESS_PROBE_STATE.get("last_issue", "") or ""
            ),
            "last_issue_at_unix": float(
                _DESKTOP_ACCESS_PROBE_STATE.get("last_issue_at_unix", 0.0) or 0.0
            ),
        }


def _record_desktop_access_probe_issue(
    probe: str,
    target: str,
    exc: BaseException,
) -> tuple[str, int]:
    issue = "timeout" if isinstance(exc, TimeoutError) else "probe_error"
    key = f"{probe}:{target}"
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        streaks = _DESKTOP_ACCESS_PROBE_STATE.setdefault("active_streaks", {})
        streak = int(streaks.get(key, 0) or 0) + 1
        streaks[key] = streak
        counter = "total_timeouts" if issue == "timeout" else "total_failures"
        _DESKTOP_ACCESS_PROBE_STATE[counter] = int(
            _DESKTOP_ACCESS_PROBE_STATE.get(counter, 0) or 0
        ) + 1
        detail = str(exc)[:240] or type(exc).__name__
        _DESKTOP_ACCESS_PROBE_STATE["last_issue"] = f"{key}:{detail}"
        _DESKTOP_ACCESS_PROBE_STATE["last_issue_at_unix"] = time.time()
    if streak == 1 or streak % 15 == 0:
        logger.warning(
            "Desktop access diagnostic %s for %s (%s, streak=%d)",
            issue,
            target,
            detail,
            streak,
        )
    else:
        logger.debug(
            "Desktop access diagnostic %s for %s suppressed (streak=%d)",
            issue,
            target,
            streak,
        )
    return issue, streak


def _mark_desktop_access_probe_success(probe: str, target: str) -> None:
    key = f"{probe}:{target}"
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        streaks = _DESKTOP_ACCESS_PROBE_STATE.setdefault("active_streaks", {})
        recovered = int(streaks.pop(key, 0) or 0)
    if recovered:
        logger.info(
            "Desktop access diagnostic recovered for %s after %d failed samples",
            target,
            recovered,
        )


def _desktop_access_probe_unavailable(
    guard: Any,
    ptype: Any,
    *,
    probe: str,
    exc: BaseException,
) -> dict[str, Any]:
    target = str(getattr(ptype, "name", ptype) or "unknown").lower()
    issue, streak = _record_desktop_access_probe_issue(probe, target, exc)
    guidance_getter = getattr(guard, "get_guidance", None)
    guidance = guidance_getter(ptype) if callable(guidance_getter) else ""
    return {
        "granted": False,
        "status": issue,
        "guidance": guidance,
        "detail": str(exc)[:240] or type(exc).__name__,
        "direct_probe": probe == "direct",
        "probe_unavailable": True,
        "retryable": True,
        "failure_streak": streak,
    }


# ── Collector Helpers ─────────────────────────────────────────

def _mark_runtime_service_progress(source: str) -> None:
    """Best-effort proof that the live desktop/API lane is actively serving."""
    try:
        from core.resilience.stall_watchdog import mark_runtime_service_progress

        mark_runtime_service_progress(source)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        logger.debug("Runtime service progress marker skipped for %s: %s", source, exc)

def _fallback_conversation_lane_status(reason: str) -> dict[str, Any]:
    desired_endpoint: str | None = None
    background_endpoint: str | None = None
    try:
        from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT, PRIMARY_ENDPOINT

        desired_endpoint = PRIMARY_ENDPOINT
        background_endpoint = BRAINSTEM_ENDPOINT
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane fallback endpoint lookup failed: %s", exc)

    return {
        "desired_model": "Cortex (32B)",
        "desired_endpoint": desired_endpoint,
        "foreground_endpoint": desired_endpoint,
        "background_endpoint": background_endpoint,
        "foreground_tier": "local",
        "background_tier": "local_fast",
        "state": "degraded",
        "last_failure_reason": str(reason or "conversation_lane_status_unavailable")[:240],
        "conversation_ready": False,
        "last_transition_at": time.time(),
        "warmup_attempted": False,
        "warmup_in_flight": False,
        "expected_model": "Cortex (32B)",
        "detected_models": [],
        "runtime_identity_ok": False,
        "kernel_tick_age_s": None,
    }


def _collect_recent_degraded_events(limit: int = 12) -> list[dict[str, Any]]:
    try:
        from core.health.degraded_events import get_recent_degraded_events

        return get_recent_degraded_events(limit=limit)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Recent degraded event collection failed: %s", exc)
        return []


def _collect_conversation_lane_status() -> dict[str, Any]:
    return _collect_conversation_lane_status_resilient()


def _collect_conversation_lane_status_resilient() -> dict[str, Any]:
    """Import and delegate to the canonical implementation in chat routes."""
    overridden = globals().get("_collect_conversation_lane_status")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_STATUS_WRAPPER:
        try:
            lane = overridden()
            if isinstance(lane, dict):
                return lane
            raise TypeError(f"conversation lane collector returned {type(lane).__name__}")
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane status unavailable: %s", exc)
            return _fallback_conversation_lane_status(str(exc))

    try:
        from interface.routes.chat_preflight import _collect_conversation_lane_status as _impl

        lane = _impl(observe_only=True)
        if isinstance(lane, dict):
            return lane
        raise TypeError(f"conversation lane collector returned {type(lane).__name__}")
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane status unavailable: %s", exc)
        return _fallback_conversation_lane_status(str(exc))


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    return _conversation_lane_is_standby_resilient(lane)


def _conversation_lane_is_standby_resilient(lane: dict[str, Any] | None) -> bool:
    overridden = globals().get("_conversation_lane_is_standby")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_STANDBY_WRAPPER:
        try:
            return overridden(lane)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane standby helper unavailable: %s", exc)

    try:
        from interface.routes.chat import _conversation_lane_is_standby as _impl

        return _impl(lane)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane standby helper unavailable: %s", exc)
        lane = dict(lane or {})
        state = str(lane.get("state", "") or "").strip().lower()
        return (
            not bool(lane.get("conversation_ready", False))
            and state in {"cold", "closed", ""}
            and not bool(lane.get("warmup_attempted", False))
            and not bool(lane.get("warmup_in_flight", False))
        )


def _conversation_lane_user_message(lane: dict[str, Any], **kwargs) -> str:
    return _conversation_lane_user_message_resilient(lane, **kwargs)


def _conversation_lane_user_message_resilient(lane: dict[str, Any], **kwargs) -> str:
    overridden = globals().get("_conversation_lane_user_message")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_MESSAGE_WRAPPER:
        try:
            return overridden(lane, **kwargs)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane message helper unavailable: %s", exc)

    try:
        from interface.routes.chat import _conversation_lane_user_message as _impl

        return _impl(lane, **kwargs)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane message helper unavailable: %s", exc)
        reason = str((lane or {}).get("last_failure_reason") or exc or "status unavailable")
        return f"The conversation lane is degraded right now: {reason[:180]}"


_NATIVE_CONVERSATION_LANE_STATUS_WRAPPER = _collect_conversation_lane_status
_NATIVE_CONVERSATION_LANE_STANDBY_WRAPPER = _conversation_lane_is_standby
_NATIVE_CONVERSATION_LANE_MESSAGE_WRAPPER = _conversation_lane_user_message


def _runtime_revision_unavailable(
    issue: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "aura.runtime_revision.v2",
        "required": bool(required),
        "verified": False,
        "source_verified": False,
        "revision_token": "",
        "expected_source_root_sha256": "",
        "actual_source_root_sha256": "",
        "expected_commit_sha": "",
        "actual_commit_sha": "",
        "expected_workspace_state_sha256": "",
        "actual_workspace_state_sha256": "",
        "expected_shell_assets_sha256": "",
        "actual_shell_assets_sha256": "",
        "capture_stable": False,
        "launch_mode": "unknown",
        "issues": [issue] if issue else [],
    }


def _runtime_revision_copy(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["issues"] = list(value.get("issues", []))
    return result


def _normalized_runtime_source_root(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except _SYSTEM_RECOVERABLE_ERRORS:
        return ""


def _runtime_identity_digest(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()


def _is_lower_hex_digest(value: str, length: int) -> bool:
    return bool(
        len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_revision_token(
    *,
    source_root_sha256: str,
    commit_sha: str,
    workspace_state_sha256: str,
    shell_assets_sha256: str,
) -> str:
    identity = {
        "commit_sha": commit_sha,
        "schema": "aura.runtime_revision.identity.v1",
        "shell_assets_sha256": shell_assets_sha256,
        "source_root_sha256": source_root_sha256,
        "workspace_state_sha256": workspace_state_sha256,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _runtime_revision_from_provenance(
    provenance: Any,
    *,
    shell_assets_sha256: str = "",
    capture_stable: bool = True,
) -> dict[str, Any]:
    source = dict(provenance) if isinstance(provenance, dict) else {}
    required = source.get("required") is True
    expected = source.get("expected")
    actual = source.get("actual")
    manifest = source.get("manifest")
    expected = expected if isinstance(expected, dict) else {}
    actual = actual if isinstance(actual, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}

    expected_root = _normalized_runtime_source_root(expected.get("source_root"))
    actual_root = _normalized_runtime_source_root(actual.get("source_root"))
    expected_root_sha256 = _runtime_identity_digest(expected_root)
    actual_root_sha256 = _runtime_identity_digest(actual_root)
    root_exact = bool(expected_root and actual_root == expected_root)

    expected_sha = str(expected.get("commit_sha") or "").strip().lower()
    actual_sha = str(actual.get("commit_sha") or "").strip().lower()
    commit_exact = bool(
        _is_lower_hex_digest(expected_sha, 40) and actual_sha == expected_sha
    )

    expected_workspace = str(
        expected.get("workspace_state_sha256") or ""
    ).strip().lower()
    actual_workspace = str(
        actual.get("workspace_state_sha256") or ""
    ).strip().lower()
    workspace_exact = bool(
        _is_lower_hex_digest(expected_workspace, 64)
        and actual_workspace == expected_workspace
    )
    expected_shell_digest = str(
        manifest.get("shell_assets_sha256") or ""
    ).strip().lower()
    actual_shell_digest = str(shell_assets_sha256 or "").strip().lower()
    shell_exact = bool(
        _is_lower_hex_digest(expected_shell_digest, 64)
        and actual_shell_digest == expected_shell_digest
    )

    source_verified = source.get("source_verified") is True
    # Identity is the checkout this runtime belongs to, plus a coherent capture.
    # Commit, workspace and shell digests are MEASURED live, so they describe
    # the running revision rather than agreeing with a build-time snapshot;
    # requiring them to match the manifest made every commit "unverified" and
    # left the revision token permanently empty. Their agreement with the
    # manifest is reported as currency, not demanded as identity.
    identity_exact = bool(root_exact and capture_stable)
    current = bool(commit_exact and workspace_exact and shell_exact)
    verified = bool(
        required
        and source.get("verified") is True
        and source_verified
        and identity_exact
    )
    issues = [str(item) for item in source.get("issues", []) if str(item)]
    if required:
        for exact, issue in (
            (root_exact, "source_root_identity_unverified"),
            (capture_stable, "workspace_changed_during_revision_capture"),
        ):
            if not exact and issue not in issues:
                issues.append(issue)

    revision_token = ""
    if verified:
        # Built from the MEASURED values, so the token names the revision that
        # is actually running.
        revision_token = _runtime_revision_token(
            source_root_sha256=actual_root_sha256,
            commit_sha=actual_sha,
            workspace_state_sha256=actual_workspace,
            shell_assets_sha256=actual_shell_digest,
        )
    return {
        "schema": "aura.runtime_revision.v2",
        "required": required,
        "verified": verified,
        "source_verified": source_verified,
        # Whether the bundle was built from exactly this workspace state. The
        # workspace moving on is normal; this reports it without failing it.
        "source_current": current,
        "revision_token": revision_token,
        "expected_source_root_sha256": expected_root_sha256,
        "actual_source_root_sha256": actual_root_sha256,
        "expected_commit_sha": expected_sha,
        "actual_commit_sha": actual_sha,
        "expected_workspace_state_sha256": expected_workspace,
        "actual_workspace_state_sha256": actual_workspace,
        "expected_shell_assets_sha256": expected_shell_digest,
        "actual_shell_assets_sha256": actual_shell_digest,
        "capture_stable": bool(capture_stable),
        "launch_mode": str(source.get("launch_mode") or "unknown"),
        "issues": sorted(set(issues)),
    }


def _runtime_provenance_observation(provenance: Any) -> str:
    source = dict(provenance) if isinstance(provenance, dict) else {}
    expected = source.get("expected")
    actual = source.get("actual")
    manifest = source.get("manifest")
    expected = expected if isinstance(expected, dict) else {}
    actual = actual if isinstance(actual, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    material = {
        "actual": {
            "commit_sha": actual.get("commit_sha"),
            "source_root": actual.get("source_root"),
            "workspace_state_sha256": actual.get("workspace_state_sha256"),
        },
        "expected": {
            "commit_sha": expected.get("commit_sha"),
            "source_root": expected.get("source_root"),
            "workspace_state_sha256": expected.get("workspace_state_sha256"),
        },
        "manifest_shell_assets_sha256": manifest.get("shell_assets_sha256"),
        "required": source.get("required") is True,
        "source_verified": source.get("source_verified") is True,
        "verified": source.get("verified") is True,
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)


def _invalidate_launch_provenance_source_observation_cache() -> None:
    """Force the second provenance read to observe the post-hash workspace."""

    try:
        from core.runtime import launch_provenance

        lock = launch_provenance._SOURCE_CACHE_LOCK
        cache = launch_provenance._SOURCE_CACHE
        with lock:
            cache.clear()
    except (AttributeError, ImportError, RuntimeError, TypeError):
        # Older launch-provenance implementations may not expose this cache.
        # The independent shell digest still protects the executable UI bytes.
        return


def _collect_runtime_revision_uncached() -> dict[str, Any]:
    from core.runtime.launch_provenance import collect_runtime_launch_provenance

    _invalidate_launch_provenance_source_observation_cache()
    before = collect_runtime_launch_provenance(config.paths.project_root)
    actual = before.get("actual") if isinstance(before, dict) else None
    actual = actual if isinstance(actual, dict) else {}
    source_root = actual.get("source_root") or config.paths.project_root
    shell_digest_before = _runtime_shell_assets_sha256(source_root)

    _invalidate_launch_provenance_source_observation_cache()
    after = collect_runtime_launch_provenance(config.paths.project_root)
    after_actual = after.get("actual") if isinstance(after, dict) else None
    after_actual = after_actual if isinstance(after_actual, dict) else {}
    after_source_root = after_actual.get("source_root") or config.paths.project_root
    shell_digest_after = _runtime_shell_assets_sha256(after_source_root)
    capture_stable = (
        _runtime_provenance_observation(before)
        == _runtime_provenance_observation(after)
        and shell_digest_before == shell_digest_after
    )
    result = _runtime_revision_from_provenance(
        after,
        shell_assets_sha256=shell_digest_after,
        capture_stable=capture_stable,
    )
    if shell_digest_before != shell_digest_after:
        result["issues"] = sorted(
            set(result.get("issues", []))
            | {"shell_assets_changed_during_revision_capture"}
        )
    if result.get("verified") is True:
        try:
            frozen_digest, frozen_assets = _capture_runtime_shell_assets(after_source_root)
            if frozen_digest != shell_digest_after:
                raise RuntimeError("shell assets changed before immutable publication")
            _publish_runtime_shell_snapshot(
                revision_token=str(result.get("revision_token") or ""),
                shell_assets_sha256=frozen_digest,
                assets=frozen_assets,
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            result["verified"] = False
            result["capture_stable"] = False
            result["revision_token"] = ""
            result["issues"] = sorted(
                set(result.get("issues", []))
                | {f"shell_snapshot_publication_failed:{type(exc).__name__}"}
            )
            _clear_runtime_shell_snapshots()
    elif result.get("required") is True:
        _clear_runtime_shell_snapshots()
    return result


def invalidate_runtime_revision_cache() -> None:
    """Invalidate revision evidence at app/runtime lifecycle boundaries."""

    global _RUNTIME_REVISION_CACHE, _RUNTIME_REVISION_CACHE_COLLECTED_AT
    global _RUNTIME_REVISION_INVALIDATION_PENDING
    acquired = _RUNTIME_REVISION_LOCK.acquire(blocking=False)
    if not acquired:
        _RUNTIME_REVISION_INVALIDATION_PENDING = True
        return
    try:
        _RUNTIME_REVISION_CACHE = None
        _RUNTIME_REVISION_CACHE_COLLECTED_AT = 0.0
        _RUNTIME_REVISION_INVALIDATION_PENDING = False
    finally:
        _RUNTIME_REVISION_LOCK.release()


def _runtime_revision_unverified_ttl_s() -> float:
    """Back off the unverified retry so it stops being a treadmill.

    An unverified revision retries every 2s so that identity is picked up the
    moment it becomes verifiable. But when a launch is *permanently*
    unverified — running from source rather than the signed app, the ordinary
    development case — that 2s never stops costing two provenance captures and
    three SHA256 walks of the shell asset tree, forever. That is a steady GIL
    burn underneath a runtime whose event loop is already the scarce resource.
    The first few retries keep the fast cadence; after that the interval
    doubles to the verified TTL, and any verified result resets it.
    """
    if _RUNTIME_REVISION_UNVERIFIED_STREAK <= _RUNTIME_REVISION_FAST_RETRIES:
        return _RUNTIME_REVISION_UNVERIFIED_TTL_S
    backed_off = _RUNTIME_REVISION_UNVERIFIED_TTL_S * (
        2 ** (_RUNTIME_REVISION_UNVERIFIED_STREAK - _RUNTIME_REVISION_FAST_RETRIES)
    )
    return min(backed_off, _RUNTIME_REVISION_VERIFIED_TTL_S)


def _runtime_revision_contract() -> dict[str, Any]:
    """Collect launch identity on the health worker with bounded cache TTLs."""
    global _RUNTIME_REVISION_CACHE, _RUNTIME_REVISION_CACHE_COLLECTED_AT
    global _RUNTIME_REVISION_INVALIDATION_PENDING, _RUNTIME_REVISION_UNVERIFIED_STREAK

    now = time.monotonic()
    with _RUNTIME_REVISION_LOCK:
        if _RUNTIME_REVISION_INVALIDATION_PENDING:
            _RUNTIME_REVISION_CACHE = None
            _RUNTIME_REVISION_CACHE_COLLECTED_AT = 0.0
            _RUNTIME_REVISION_INVALIDATION_PENDING = False
            _RUNTIME_REVISION_UNVERIFIED_STREAK = 0
        cache_age = max(0.0, now - _RUNTIME_REVISION_CACHE_COLLECTED_AT)
        cache_ttl = (
            _RUNTIME_REVISION_VERIFIED_TTL_S
            if _RUNTIME_REVISION_CACHE is not None
            and _RUNTIME_REVISION_CACHE.get("verified") is True
            else _runtime_revision_unverified_ttl_s()
        )
        cache_expired = bool(
            _RUNTIME_REVISION_CACHE is not None
            and cache_age >= cache_ttl
        )
        if _RUNTIME_REVISION_CACHE is None or cache_expired:
            try:
                _RUNTIME_REVISION_CACHE = _collect_runtime_revision_uncached()
            except _SYSTEM_RECOVERABLE_ERRORS as exc:
                _RUNTIME_REVISION_CACHE = _runtime_revision_unavailable(
                    f"revision_collection_failed:{type(exc).__name__}",
                    required=_launched_from_app_flag(),
                )
            _RUNTIME_REVISION_CACHE_COLLECTED_AT = now
            if _RUNTIME_REVISION_CACHE.get("verified") is True:
                _RUNTIME_REVISION_UNVERIFIED_STREAK = 0
            else:
                _RUNTIME_REVISION_UNVERIFIED_STREAK += 1
        return _runtime_revision_copy(_RUNTIME_REVISION_CACHE)


def _runtime_revision_fallback_contract() -> dict[str, Any]:
    """Read cached identity only; never run provenance probes on an HTTP fallback."""
    global _RUNTIME_REVISION_CACHE, _RUNTIME_REVISION_CACHE_COLLECTED_AT
    global _RUNTIME_REVISION_INVALIDATION_PENDING
    acquired = _RUNTIME_REVISION_LOCK.acquire(blocking=False)
    if acquired:
        try:
            if _RUNTIME_REVISION_INVALIDATION_PENDING:
                _RUNTIME_REVISION_CACHE = None
                _RUNTIME_REVISION_CACHE_COLLECTED_AT = 0.0
                _RUNTIME_REVISION_INVALIDATION_PENDING = False
            elif _RUNTIME_REVISION_CACHE is not None:
                return _runtime_revision_copy(_RUNTIME_REVISION_CACHE)
        finally:
            _RUNTIME_REVISION_LOCK.release()
    issue = (
        "runtime_revision_initializing"
        if acquired
        else "runtime_revision_collection_in_flight"
    )
    return _runtime_revision_unavailable(
        issue,
        required=_launched_from_app_flag(),
    )


def _runtime_revision_blocker(revision: Any) -> str:
    if not isinstance(revision, dict):
        return "runtime_revision_contract_missing"
    contract = revision
    if contract.get("schema") != "aura.runtime_revision.v2":
        return "runtime_revision_contract_invalid"
    required = contract.get("required")
    if not isinstance(required, bool):
        return "runtime_revision_contract_invalid"
    if _launched_from_app_flag() and required is not True:
        return "runtime_revision_required_contract_missing"
    if required is False:
        # A source/direct-launch contract is honest with verified=False and no
        # signed revision token — even when the source tree itself WAS
        # hash-verified (source_verified=True, which the collector legitimately
        # sets in direct mode). Only a claimed FULL verification (verified=True)
        # or a present signed revision token is contradictory with required=False.
        # Rejecting source_verified=True here downgraded every source/test run's
        # health to "degraded" (regression from the signed-shell hardening).
        if (
            contract.get("verified") is not False
            or str(contract.get("revision_token") or "")
        ):
            return "runtime_revision_contract_invalid"
        return ""
    if contract.get("verified") is not True:
        return "runtime_revision_unverified"
    if contract.get("source_verified") is not True or contract.get("capture_stable") is not True:
        return "runtime_revision_identity_invalid"
    if str(contract.get("launch_mode") or "") != "signed_app":
        return "runtime_revision_identity_invalid"

    token = str(contract.get("revision_token") or "")
    expected_root = str(contract.get("expected_source_root_sha256") or "")
    actual_root = str(contract.get("actual_source_root_sha256") or "")
    actual_commit = str(contract.get("actual_commit_sha") or "")
    actual_workspace = str(contract.get("actual_workspace_state_sha256") or "")
    actual_shell = str(contract.get("actual_shell_assets_sha256") or "")
    # Identity is the checkout, plus well-formed measurements of what is
    # running. The commit, workspace and shell digests are MEASURED, so they
    # must be present and well-formed — but requiring them to equal the
    # build-time manifest is the same "staleness is a fault" error corrected in
    # core.runtime.launch_provenance, and it was duplicated here. Their
    # agreement is reported as source_current; disagreement means the workspace
    # moved on, which is Aura's normal state, not a failed identity.
    if not (
        _is_lower_hex_digest(token, 64)
        and _is_lower_hex_digest(actual_root, 64)
        and expected_root == actual_root
        and _is_lower_hex_digest(actual_commit, 40)
        and _is_lower_hex_digest(actual_workspace, 64)
        and _is_lower_hex_digest(actual_shell, 64)
    ):
        return "runtime_revision_identity_invalid"
    expected_token = _runtime_revision_token(
        source_root_sha256=actual_root,
        commit_sha=actual_commit,
        workspace_state_sha256=actual_workspace,
        shell_assets_sha256=actual_shell,
    )
    if not hmac.compare_digest(token, expected_token):
        return "runtime_revision_token_invalid"
    return ""


def _apply_runtime_revision_truth(payload: dict[str, Any]) -> dict[str, Any]:
    """Make required signed-shell provenance part of every readiness verdict."""

    blocker = _runtime_revision_blocker(payload.get("runtime_revision"))
    if not blocker:
        return payload

    result = dict(payload)
    blockers = list(
        dict.fromkeys(
            [blocker]
            + [str(item) for item in result.get("blockers", []) if str(item)]
        )
    )
    result.update(
        status="degraded",
        healthy=False,
        ready=False,
        connected=False,
        system_ready=False,
        launcher_ready=False,
        proof_readiness_healthy=False,
        certification_ready=False,
        blockers=blockers,
    )

    readiness = dict(result.get("readiness_contract") or {})
    readiness.update(
        healthy=False,
        ready=False,
        connected=False,
        system_ready=False,
        proof_readiness_healthy=False,
        certification_ready=False,
        blockers=blockers,
    )
    result["readiness_contract"] = readiness

    boot = dict(result.get("boot") or {})
    boot_blockers = list(
        dict.fromkeys(
            [blocker]
            + [str(item) for item in boot.get("blockers", []) if str(item)]
        )
    )
    boot.update(
        status="launch_provenance_failed",
        ready=False,
        system_ready=False,
        launcher_ready=False,
        proof_readiness_healthy=False,
        certification_ready=False,
        blockers=boot_blockers,
    )
    result["boot"] = boot
    session = result.get("session")
    if isinstance(session, dict):
        session = dict(session)
        session["connected"] = False
        session["ready"] = False
        result["session"] = session
    return result


def _runtime_revision_response_projection(
    payload: dict[str, Any],
    *,
    include_diagnostics: bool,
) -> dict[str, Any]:
    """Keep exact source fingerprints owner-only on the public health route."""

    revision = payload.get("runtime_revision")
    if include_diagnostics:
        return payload

    result = dict(payload)
    if isinstance(revision, dict):
        required = revision.get("required") is True
        verified = revision.get("verified") is True
        result["runtime_revision"] = {
            "schema": str(revision.get("schema") or "aura.runtime_revision.v2"),
            "required": required,
            "verified": verified,
            "revision_token": (
                str(revision.get("revision_token") or "") if verified else ""
            ),
            "status": (
                "verified"
                if verified
                else "unverified"
                if required
                else "not_required"
            ),
            "blocker": (
                "runtime_revision_unverified"
                if required and not verified
                else ""
            ),
        }
    boot = result.get("boot")
    if isinstance(boot, dict) and isinstance(boot.get("launch_provenance"), dict):
        launch = boot["launch_provenance"]
        launch_required = launch.get("required") is True
        launch_verified = launch.get("verified") is True
        boot = dict(boot)
        boot["launch_provenance"] = {
            "schema": str(launch.get("schema") or "aura.launch_provenance.v1"),
            "required": launch_required,
            "verified": launch_verified,
            "status": (
                "verified"
                if launch_verified
                else "unverified"
                if launch_required
                else "not_required"
            ),
            "blocker": "launch_provenance" if launch_required and not launch_verified else "",
        }
        result["boot"] = boot
    launch = result.get("launch_provenance")
    if isinstance(launch, dict):
        launch_required = launch.get("required") is True
        launch_verified = launch.get("verified") is True
        result["launch_provenance"] = {
            "schema": str(launch.get("schema") or "aura.launch_provenance.v1"),
            "required": launch_required,
            "verified": launch_verified,
            "status": (
                "verified"
                if launch_verified
                else "unverified"
                if launch_required
                else "not_required"
            ),
            "blocker": "launch_provenance" if launch_required and not launch_verified else "",
        }
    return result


def _attach_launch_provenance_contract(
    payload: dict[str, Any],
    status_code: int,
    *,
    provenance: dict[str, Any] | None = None,
    runtime_revision: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Prevent an orphaned, stale, or incorrectly signed app runtime from looking ready."""

    collect_live_revision = provenance is None
    if provenance is None:
        try:
            from core.runtime.launch_provenance import collect_runtime_launch_provenance

            provenance = collect_runtime_launch_provenance(config.paths.project_root)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            launched_from_app = _launched_from_app_flag()
            provenance = {
                "schema": "aura.launch_provenance.v1",
                "required": launched_from_app,
                "verified": False,
                "source_verified": False,
                "issues": [f"provenance_collection_failed:{type(exc).__name__}"],
            }
            logger.warning("Launch provenance collection failed: %s", exc)
    if runtime_revision is None:
        runtime_revision = (
            _runtime_revision_contract()
            if collect_live_revision
            else _runtime_revision_fallback_contract()
        )

    result = dict(payload)
    result["launch_provenance"] = provenance
    result["runtime_revision"] = runtime_revision
    checks = dict(result.get("checks") or {})
    required = bool(provenance.get("required"))
    verified = bool(provenance.get("verified"))
    revision_blocker = _runtime_revision_blocker(runtime_revision)
    checks["launch_provenance"] = verified if required else True
    checks["runtime_revision"] = not revision_blocker
    result["checks"] = checks
    launch_blocked = bool(required and not verified)
    if not launch_blocked and not revision_blocker:
        return result, status_code

    blockers = [str(item) for item in result.get("blockers", []) if str(item)]
    if launch_blocked and "launch_provenance" not in blockers:
        blockers.append("launch_provenance")
    if revision_blocker and revision_blocker not in blockers:
        blockers.append(revision_blocker)
    if launch_blocked and revision_blocker:
        status_message = (
            "Aura's signed app/source and runtime shell provenance are not verified."
        )
        boot_phase = "launch_and_shell_provenance_failed"
    elif launch_blocked:
        status_message = (
            "Aura's runtime is alive, but its signed app/source provenance is not verified."
        )
        boot_phase = "launch_provenance_failed"
    else:
        status_message = (
            "Aura's runtime is alive, but its signed runtime shell provenance is not verified."
        )
        boot_phase = "runtime_revision_failed"
    diagnosis = _provenance_failure_diagnosis(provenance, runtime_revision)
    if diagnosis:
        status_message = f"{status_message} {diagnosis}"
    result.update(
        {
            "ready": False,
            "launcher_ready": False,
            "system_ready": False,
            "proof_readiness_healthy": False,
            "certification_ready": False,
            "status": "degraded",
            "status_message": status_message,
            "boot_phase": boot_phase,
            "blockers": blockers,
            "provenance_diagnosis": diagnosis,
        }
    )
    return result, 503


def _provenance_failure_diagnosis(
    provenance: dict[str, Any] | None,
    runtime_revision: dict[str, Any] | None,
) -> str:
    """Name the concrete identity mismatch behind a provenance block.

    "Provenance not verified" alone reads as a mysterious breakage; the most
    common cause is simply a signed Aura.app that predates the current source
    checkout. Saying WHICH identity diverged (and the remedy) turns a
    confusing stuck-at-48% boot screen into an actionable one.
    """
    revision = dict(runtime_revision) if isinstance(runtime_revision, dict) else {}
    source = dict(provenance) if isinstance(provenance, dict) else {}

    # Drift is NOT a cause of a blocked boot any more — the app is a thin
    # launcher over live source, so a moved commit, edited workspace or updated
    # UI shell all still run the current code. These used to be the three
    # "rebuild Aura.app" messages, and they fired constantly because Aura
    # commits to her own repository. If a diagnosis is being written at all,
    # something else genuinely failed, so drift is reported as context rather
    # than blamed.
    drift_note = ""
    expected_commit = str(revision.get("expected_commit_sha") or "").strip()
    actual_commit = str(revision.get("actual_commit_sha") or "").strip()
    if expected_commit and actual_commit and expected_commit != actual_commit:
        drift_note = (
            f" (For context, the bundle was built at {expected_commit[:9]} and the "
            f"checkout is at {actual_commit[:9]}; that is expected and not the cause.)"
        )

    issues = sorted(
        {
            str(item)
            for item in (
                list(revision.get("issues") or []) + list(source.get("issues") or [])
            )
            if str(item)
        }
    )
    if issues:
        return "Identity issues: " + ", ".join(issues[:5]) + "." + drift_note
    return drift_note.strip()


def _launched_from_app_flag() -> bool:
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_LAUNCHED_FROM_APP",
            kind=FlagKind.BOOL,
            default=False,
            description="Set by the desktop app launcher; gates app-managed behaviors",
            owner="interface.routes.system",
        ).value()
    )


def _fallback_launch_provenance(manifest_snapshot: Any = None) -> dict[str, Any]:
    """Return non-blocking conservative evidence for event-loop fallback paths."""

    snapshot = dict(manifest_snapshot) if isinstance(manifest_snapshot, dict) else {}
    launched_from_app = _launched_from_app_flag()
    required = bool(snapshot.get("required", launched_from_app))
    if not required:
        return {
            **snapshot,
            "schema": str(snapshot.get("schema") or "aura.launch_provenance.v1"),
            "required": False,
            "verified": False,
            "launch_mode": str(snapshot.get("launch_mode") or "direct"),
            "issues": list(snapshot.get("issues") or []),
        }
    issues = [str(item) for item in snapshot.get("issues", []) if str(item)]
    if "launch_provenance_live_refresh_unavailable" not in issues:
        issues.append("launch_provenance_live_refresh_unavailable")
    return {
        **snapshot,
        "schema": str(snapshot.get("schema") or "aura.launch_provenance.v1"),
        "required": True,
        "verified": False,
        "issues": issues,
    }


def _build_boot_health_payload_sync(*, is_gui_proxy: bool) -> tuple[dict[str, Any], int]:
    """Build boot health with a single-flight guard for HTTP readiness probes."""

    stopping = _stopping_boot_health_payload()
    if stopping is not None:
        return stopping

    probe_lock = _HEALTH_PROBE_LOCKS[bool(is_gui_proxy)]
    acquired = probe_lock.acquire(False)
    if not acquired:
        raise TimeoutError("health_probe_already_running")
    try:
        orch = ServiceContainer.get("orchestrator", default=None)
        rt = _get_runtime_state_safe()
        conversation_lane = _collect_conversation_lane_status_resilient()
        try:
            payload, status_code = build_boot_health_snapshot(
                orch,
                rt,
                is_gui_proxy=is_gui_proxy,
                conversation_lane=conversation_lane,
            )
            payload, status_code = _attach_launch_provenance_contract(payload, status_code)
            _store_boot_health_cache(
                payload,
                status_code,
                is_gui_proxy=is_gui_proxy,
            )
            return payload, status_code
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.error("Boot health snapshot failed: %s", exc, exc_info=True)
            payload = {
                "ready": False,
                "status": "degraded",
                "issues": [str(exc)],
                "conversation_lane": conversation_lane,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            payload, status_code = _attach_launch_provenance_contract(payload, 503)
            _store_boot_health_cache(
                payload,
                status_code,
                is_gui_proxy=is_gui_proxy,
            )
            return payload, status_code
    finally:
        probe_lock.release()


def _store_boot_health_cache(
    payload: dict[str, Any],
    status_code: int,
    *,
    is_gui_proxy: bool = False,
) -> None:
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        entry["captured_at"] = time.monotonic()
        entry["payload"] = dict(payload)
        entry["status_code"] = int(status_code)


def _fresh_boot_health_payload(
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], int] | None:
    now = time.monotonic()
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        captured_at = float(entry.get("captured_at") or 0.0)
        payload = entry.get("payload")
        status_code = int(entry.get("status_code") or 503)
    age_s = max(0.0, now - captured_at) if captured_at > 0.0 else float("inf")
    if (
        not isinstance(payload, dict)
        or "ready" not in payload
        or age_s > _HEALTH_CACHE_TTL_S
    ):
        return None
    cached = dict(payload)
    cached["cache_status"] = "fresh"
    cached["cache_reason"] = "health_cache_ttl"
    cached["cache_age_s"] = round(age_s, 3)
    return cached, status_code


def _complete_health_probe_future(
    future: Future[tuple[dict[str, Any], int]],
    generation: int,
    *,
    is_gui_proxy: bool,
) -> None:
    failure: Exception | None = None
    result: tuple[dict[str, Any], int] | None = None
    if future.cancelled():
        failure = CancelledError("health probe future was cancelled")
    else:
        failure = future.exception()
    if failure is None:
        candidate = future.result()
        if (
            not isinstance(candidate, tuple)
            or len(candidate) != 2
            or not isinstance(candidate[0], dict)
        ):
            failure = TypeError("health probe returned an invalid payload contract")
        else:
            result = candidate

    should_escalate = False
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        if _HEALTH_PROBE_FUTURES.get(surface) is not future:
            return
        _HEALTH_PROBE_FUTURES.pop(surface, None)
        _HEALTH_PROBE_GENERATIONS.pop(surface, None)
        _HEALTH_PROBE_STARTED_AT.pop(surface, None)
        if result is not None:
            _store_boot_health_cache(
                result[0],
                result[1],
                is_gui_proxy=surface,
            )
        if failure is None:
            _HEALTH_PROBE_STATE["consecutive_failures"] = 0
            _HEALTH_PROBE_STATE["last_failure_reason"] = ""
            _HEALTH_PROBE_STATE["escalated"] = False
        else:
            reason = f"health_probe_exception:{type(failure).__name__}"
            _HEALTH_PROBE_STATE["total_terminal_failures"] = int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["consecutive_failures"] = int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["last_failure_reason"] = reason
            _HEALTH_PROBE_STATE["last_failure_at_unix"] = time.time()
            should_escalate = bool(
                int(_HEALTH_PROBE_STATE.get("consecutive_failures") or 0)
                >= _HEALTH_PROBE_DEGRADATION_THRESHOLD
                and not bool(_HEALTH_PROBE_STATE.get("escalated", False))
            )
            if should_escalate:
                _HEALTH_PROBE_STATE["escalated"] = True

    if failure is None:
        return
    if should_escalate:
        record_degradation(
            "system",
            failure,
            severity="warning",
            action="escalated distinct terminal health-probe failures",
            extra=_health_probe_state_snapshot(),
            enforce_failure_policy=False,
        )
    else:
        logger.warning(
            "Boot-health probe generation %d failed: %s",
            generation,
            failure,
        )


def _start_or_join_health_probe(
    *,
    is_gui_proxy: bool,
) -> tuple[Future[tuple[dict[str, Any], int]], int, bool]:
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        existing = _HEALTH_PROBE_FUTURES.get(surface)
        if existing is not None:
            generation = int(
                _HEALTH_PROBE_GENERATIONS.get(surface)
                or _HEALTH_PROBE_STATE.get("generation")
                or 0
            )
            if not existing.done():
                _HEALTH_PROBE_STATE["total_contentions"] = int(
                    _HEALTH_PROBE_STATE.get("total_contentions") or 0
                ) + 1
                return existing, generation, False
            return existing, generation, True

        generation = int(_HEALTH_PROBE_STATE.get("generation") or 0) + 1
        _HEALTH_PROBE_STATE["generation"] = generation
        _HEALTH_PROBE_GENERATIONS[surface] = generation
        _HEALTH_PROBE_STARTED_AT[surface] = time.monotonic()
        future = _HEALTH_PROBE_EXECUTOR.submit(
            _build_boot_health_payload_sync,
            is_gui_proxy=surface,
        )
        _HEALTH_PROBE_FUTURES[surface] = future

    future.add_done_callback(
        lambda completed, probe_generation=generation, probe_surface=surface: _complete_health_probe_future(
            completed,
            probe_generation,
            is_gui_proxy=probe_surface,
        )
    )
    return future, generation, True


def _record_health_probe_wait_timeout(generation: int) -> tuple[dict[str, Any], bool]:
    recorded = False
    with _HEALTH_PROBE_STATE_LOCK:
        if int(_HEALTH_PROBE_STATE.get("timeout_recorded_generation") or 0) != generation:
            recorded = True
            _HEALTH_PROBE_STATE["timeout_recorded_generation"] = generation
            _HEALTH_PROBE_STATE["total_timeouts"] = int(
                _HEALTH_PROBE_STATE.get("total_timeouts") or 0
            ) + 1
    return _health_probe_state_snapshot(), recorded


def _record_stuck_health_probe_once(
    generation: int,
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], bool, bool]:
    now_monotonic = time.monotonic()
    recorded = False
    should_escalate = False
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        active_since = float(_HEALTH_PROBE_STARTED_AT.get(surface) or 0.0)
        active_generation = int(_HEALTH_PROBE_GENERATIONS.get(surface) or 0)
        active_age_s = (
            max(0.0, now_monotonic - active_since) if active_since > 0.0 else 0.0
        )
        if (
            active_generation == generation
            and active_age_s >= _HEALTH_PROBE_STUCK_THRESHOLD_S
            and int(_HEALTH_PROBE_STATE.get("stuck_recorded_generation") or 0)
            != generation
        ):
            recorded = True
            _HEALTH_PROBE_STATE["stuck_recorded_generation"] = generation
            _HEALTH_PROBE_STATE["total_terminal_failures"] = int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["consecutive_failures"] = int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["last_failure_reason"] = "health_probe_stuck"
            _HEALTH_PROBE_STATE["last_failure_at_unix"] = time.time()
            should_escalate = bool(
                int(_HEALTH_PROBE_STATE.get("consecutive_failures") or 0)
                >= _HEALTH_PROBE_DEGRADATION_THRESHOLD
                and not bool(_HEALTH_PROBE_STATE.get("escalated", False))
            )
            if should_escalate:
                _HEALTH_PROBE_STATE["escalated"] = True
    return _health_probe_state_snapshot(), recorded, should_escalate


def _cached_boot_health_payload(
    reason: str,
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], int]:
    stopping = _stopping_boot_health_payload()
    if stopping is not None:
        return stopping
    now = time.monotonic()
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        captured_at = float(entry.get("captured_at") or 0.0)
        payload = entry.get("payload")
        status_code = int(entry.get("status_code") or 503)

    cache_age_s = max(0.0, now - captured_at) if captured_at > 0.0 else float("inf")
    if (
        isinstance(payload, dict)
        and "ready" in payload
        and cache_age_s <= _HEALTH_CACHE_TTL_S
    ):
        cached = dict(payload)
        cached["cache_status"] = "fresh"
        cached["cache_reason"] = reason
        cached["cache_age_s"] = round(cache_age_s, 3)
        return cached, status_code
    if (
        reason in {
            "foreground_generation_active",
            "health_probe_in_flight",
            "health_probe_timeout",
        }
        and isinstance(payload, dict)
        and "ready" in payload
        and cache_age_s <= _HEALTH_STALE_CACHE_TTL_S
    ):
        cached = dict(payload)
        cached["cache_status"] = "stale_while_revalidate"
        cached["cache_reason"] = reason
        cached["cache_age_s"] = round(cache_age_s, 3)
        cached["cache_stale"] = True
        return cached, status_code

    manifest_payload = _runtime_manifest_boot_health_payload(reason)
    if manifest_payload is not None:
        manifest_body, manifest_status = manifest_payload
        fallback = _fallback_launch_provenance(manifest_body.get("launch_provenance"))
        return _attach_launch_provenance_contract(
            manifest_body,
            manifest_status,
            provenance=fallback,
        )

    return _attach_launch_provenance_contract(
        {
            "ready": False,
            "status": "unhealthy",
            "issues": [reason],
            "required_probes": {"all_passed": False},
            "blockers": [reason],
            "boot_phase": reason,
            "conversation_ready": False,
            "cache_status": "miss",
            "cache_reason": reason,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        503,
        provenance=_fallback_launch_provenance(),
    )


def _runtime_manifest_boot_health_payload(reason: str) -> tuple[dict[str, Any], int] | None:
    try:
        manifest_path = config.paths.project_root / "artifacts" / "current" / "runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        readiness = manifest.get("readiness_snapshot")
        if not isinstance(readiness, dict):
            return None
        generated_at = _safe_float(manifest.get("generated_at_unix"), 0.0)
        manifest_age_s = max(0.0, time.time() - generated_at) if generated_at > 0.0 else float("inf")
        if manifest_age_s > _HEALTH_MANIFEST_FALLBACK_TTL_S:
            return (
                {
                    "ready": False,
                    "status": "unhealthy",
                    "system_ready": False,
                    "launcher_ready": False,
                    "conversation_ready": False,
                    "boot_phase": "manifest_stale",
                    "required_probes": {"all_passed": False},
                    "blockers": ["health_manifest_stale", reason],
                    "cache_status": "manifest_stale",
                    "cache_reason": reason,
                    "manifest_age_s": round(manifest_age_s, 3),
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                },
                503,
            )
        ready = bool(readiness.get("ready") is True)
        blockers = [str(item) for item in readiness.get("required_probe_blockers", []) if str(item)]
        if not ready and not blockers:
            blockers = [reason]
        status_code = 200 if ready and not blockers else 503
        required_probes: dict[str, Any] = {"all_passed": ready}
        for group_name, components in REQUIRED_HEALTH_PROBE_GROUPS.items():
            required_probes[group_name] = {
                "ok": ready,
                "components": {component: ready for component in components},
            }
        return (
            {
                "ready": ready,
                "status": "ready" if status_code == 200 else "unhealthy",
                "system_ready": ready,
                "launcher_ready": ready,
                "conversation_ready": ready,
                "boot_phase": "manifest_ready" if ready else "manifest_unhealthy",
                "required_probes": required_probes,
                "blockers": blockers,
                "cache_status": "manifest",
                "cache_reason": reason,
                "manifest_generated_at_unix": manifest.get("generated_at_unix"),
                "manifest_age_s": round(manifest_age_s, 3),
                "launch_provenance": manifest.get("launch_provenance"),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
            status_code,
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime manifest health fallback failed: %s", exc)
        return None


async def _build_boot_health_payload_bounded(*, is_gui_proxy: bool) -> tuple[dict[str, Any], int]:
    """Return a boot-health snapshot without allowing probes to hang the HTTP loop."""

    fresh = _fresh_boot_health_payload(is_gui_proxy=is_gui_proxy)
    if fresh is not None:
        return _attach_health_probe_state(fresh)

    # A foreground generation already owns the expensive model/runtime lane.
    # Starting a full health sweep beside it cannot make the serving answer
    # healthier; live evidence showed the sweep repeatedly exceeded its HTTP
    # budget while the 32B was decoding. Serve the last bounded read model (or
    # the runtime manifest when no cache exists) and let the next idle poll
    # refresh it. The endpoint below independently overlays the current
    # conversation-lane state, so callers still see WORKING rather than stale
    # READY/FAILED semantics.
    conversation_lane = _collect_conversation_lane_status_resilient()
    if conversation_lane_is_busy(conversation_lane):
        foreground_fallback = _cached_boot_health_payload(
            "foreground_generation_active",
            is_gui_proxy=is_gui_proxy,
        )
        if foreground_fallback[0].get("cache_status") in {
            "manifest",
            "stale_while_revalidate",
        }:
            return _attach_health_probe_state(foreground_fallback)

    future, generation, created = _start_or_join_health_probe(
        is_gui_proxy=is_gui_proxy,
    )
    if not created:
        probe_state, newly_stuck, should_escalate = _record_stuck_health_probe_once(
            generation,
            is_gui_proxy=is_gui_proxy,
        )
        if should_escalate:
            record_degradation(
                "system",
                TimeoutError(
                    "distinct health probes exceeded the explicit stuck threshold"
                ),
                severity="warning",
                action="escalated distinct stuck health-probe generations",
                extra=probe_state,
                enforce_failure_policy=False,
            )
        elif newly_stuck:
            logger.warning(
                "Boot-health probe generation %d exceeded the %.1fs stuck threshold; "
                "recorded once while callers continue using bounded fallback evidence.",
                generation,
                _HEALTH_PROBE_STUCK_THRESHOLD_S,
            )
        return _attach_health_probe_state(
            _cached_boot_health_payload(
                "health_probe_in_flight",
                is_gui_proxy=is_gui_proxy,
            )
        )

    try:
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=_HEALTH_PROBE_TIMEOUT_S,
        )
        return _attach_health_probe_state(result)
    except TimeoutError:
        probe_state, timeout_recorded = _record_health_probe_wait_timeout(generation)
        fallback = _cached_boot_health_payload(
            "health_probe_timeout",
            is_gui_proxy=is_gui_proxy,
        )
        if timeout_recorded:
            fallback_payload = fallback[0]
            log = (
                logger.info
                if generation == 1
                and not bool(fallback_payload.get("ready"))
                and not bool(fallback_payload.get("conversation_ready"))
                else logger.warning
            )
            log(
                "Boot-health probe generation %d exceeded the %.1fs HTTP wait budget; "
                "the singleflight remains active and later polls will reuse its result.",
                generation,
                _HEALTH_PROBE_TIMEOUT_S,
            )
        return _attach_health_probe_state(fallback)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        logger.warning(
            "Boot-health probe generation %d ended before returning a payload: %s",
            generation,
            exc,
        )
        return _attach_health_probe_state(
            _cached_boot_health_payload(
                "health_probe_failed",
                is_gui_proxy=is_gui_proxy,
            )
        )


def _get_runtime_state_safe() -> dict[str, Any]:
    try:
        rt = get_runtime_state()
        if isinstance(rt, dict):
            return rt
        raise TypeError(f"runtime state returned {type(rt).__name__}")
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime state snapshot failed: %s", exc)
        return {
            "state": {},
            "status": "degraded",
            "error": str(exc)[:240],
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }


def _collect_stability_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "status": "unknown",
        "healthy": False,
        "active_issues": [],
    }
    try:
        guardian = ServiceContainer.peek("stability_guardian", default=None)
        if guardian is None:
            details["status"] = "unavailable"
            details["active_issues"].append(
                {
                    "name": "stability_guardian",
                    "message": "StabilityGuardian is not registered.",
                    "severity": "warning",
                    "action_taken": "withhold healthy status until guardian is online",
                }
            )
        elif hasattr(guardian, "get_latest_report"):
            report = guardian.get_latest_report() or {}
            checks = report.get("checks", []) if isinstance(report, dict) else []
            active_issues = []
            for check in checks:
                if not bool(check.get("healthy", False)):
                    active_issues.append(
                        {
                            "name": check.get("name", "unknown"),
                            "message": check.get("message", ""),
                            "severity": check.get("severity", "warning"),
                            "action_taken": check.get("action_taken"),
                        }
                    )
            if report:
                details["healthy"] = bool(report.get("overall_healthy", False))
                details["status"] = "healthy" if details["healthy"] else "degraded"
                details["active_issues"] = active_issues
                details["memory_pct"] = report.get("memory_pct")
                details["cpu_pct"] = report.get("cpu_pct")
            elif hasattr(guardian, "get_health_summary"):
                summary = guardian.get_health_summary()
                if isinstance(summary, dict):
                    details["healthy"] = bool(summary.get("healthy", False))
                    details["status"] = str(summary.get("status") or "unknown")
                    details["active_issues"] = list(summary.get("active_issues") or [])
                    if details["status"] == "initializing":
                        details["healthy"] = False
            else:
                details["status"] = "no_report"
                details["active_issues"] = [
                    {
                        "name": "stability_report",
                        "message": "StabilityGuardian has not produced a health report.",
                        "severity": "warning",
                        "action_taken": "withhold healthy status until probes run",
                    }
                ]
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Stability detail collection failed: %s", exc)

    try:
        lane = _collect_conversation_lane_status_resilient()
        if isinstance(lane, dict) and not bool(lane.get("conversation_ready", False)):
            details["healthy"] = False
            if details.get("status") == "unknown":
                details["status"] = "degraded"
            details.setdefault("active_issues", []).append(
                {
                    "name": "conversation_lane",
                    "message": _conversation_lane_user_message_resilient(lane),
                    "severity": "warning" if str(lane.get("state", "") or "").lower() != "failed" else "error",
                    "action_taken": None,
                }
            )
        if isinstance(lane, dict) and not bool(lane.get("runtime_identity_ok", True)):
            details["healthy"] = False
            if details.get("status") == "unknown":
                details["status"] = "degraded"
            details.setdefault("active_issues", []).append(
                {
                    "name": "conversation_lane_model_mismatch",
                    "message": (
                        f"Expected {lane.get('expected_model') or 'the configured Cortex model'}, "
                        f"but detected {', '.join(lane.get('detected_models') or []) or 'an unexpected runtime model'} "
                        "on the reserved conversation lane."
                    ),
                    "severity": "error",
                    "action_taken": None,
                }
            )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Conversation lane stability detail merge failed: %s", exc)
    if details.get("status") == "unknown":
        details["status"] = "healthy" if bool(details.get("healthy", False)) else "degraded"
    return details


def _normalize_percentish(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return max(0.0, min(100.0, number))


def _json_safe(value: Any) -> Any:
    """Recursively coerce runtime payloads into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Unable to coerce scalar-like value with item(): %s", exc)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Unable to coerce array-like value with tolist(): %s", exc)
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(coerced) or math.isinf(coerced):
        return None
    return coerced


def _collect_liquid_state_payload(
    ls_data: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    homeostasis_data: dict[str, Any],
) -> dict[str, Any]:
    runtime_affect = runtime_state.get("affect", {}) if isinstance(runtime_state.get("affect"), dict) else {}
    payload: dict[str, Any] = {}

    def _pick_metric(key: str, *, runtime_fallback: Any = None) -> float | None:
        primary = _normalize_percentish(ls_data.get(key))
        fallback = _normalize_percentish(runtime_fallback if runtime_fallback is not None else runtime_affect.get(key))
        if primary is None:
            return fallback
        if primary == 0.0 and fallback not in (None, 0.0):
            return fallback
        return primary

    derived_frustration = runtime_affect.get("frustration")
    if derived_frustration is None:
        try:
            valence = float(runtime_affect.get("valence"))
            if valence < 0.0:
                derived_frustration = min(100.0, abs(valence) * 100.0)
        except (TypeError, ValueError):
            derived_frustration = None

    for key in ("energy", "curiosity", "frustration", "focus", "confidence"):
        runtime_fallback = None
        if key == "frustration":
            runtime_fallback = derived_frustration
        elif key == "curiosity":
            runtime_fallback = runtime_affect.get("curiosity", homeostasis_data.get("curiosity"))
        elif key == "confidence":
            runtime_fallback = runtime_affect.get(
                "confidence",
                _homeostasis_vitality_value(homeostasis_data),
            )
        normalized = _pick_metric(key, runtime_fallback=runtime_fallback)
        if normalized is not None:
            payload[key] = round(normalized, 1)

    if "confidence" not in payload:
        normalized = _normalize_percentish(_homeostasis_vitality_value(homeostasis_data))
        if normalized is not None:
            payload["confidence"] = round(normalized, 1)

    if ls_data.get("mood") is not None:
        payload["mood"] = ls_data.get("mood")
    elif runtime_affect.get("mood") is not None:
        payload["mood"] = runtime_affect.get("mood")

    if isinstance(ls_data.get("vad"), dict):
        payload["vad"] = ls_data["vad"]

    return payload


def _homeostasis_vitality_value(homeostasis_data: dict[str, Any]) -> Any:
    """Return the public vitality/confidence source from homeostasis data.

    ``will_to_live`` is retained as an internal legacy key in the homeostasis
    subsystem. Public health payloads should prefer operational labels so UI and
    API consumers do not treat a homeostatic scalar as proof of subjectivity.
    """
    for key in ("operational_confidence", "vitality", "will_to_live"):
        value = homeostasis_data.get(key)
        if value is not None:
            return value
    return None


def _collect_homeostasis_public_payload(homeostasis_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(homeostasis_data or {})
    legacy_vitality = payload.pop("will_to_live", None)
    vitality_source = None
    for key in ("vitality", "operational_confidence"):
        value = payload.get(key)
        if value is not None:
            vitality_source = value
            break
    if vitality_source is None:
        vitality_source = legacy_vitality
    normalized = _normalize_percentish(vitality_source)
    if normalized is not None:
        value = round(normalized / 100.0, 4)
        payload.setdefault("vitality", value)
        payload.setdefault("operational_confidence", value)
    return payload


async def _collect_soma_payload(*, refresh: bool = True) -> dict[str, Any]:
    def _system_fallback() -> dict[str, Any]:
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None) or 0.0) / 100.0
            ram = psutil.virtual_memory()
            from core.runtime.disk_budget import state_volume_usage

            disk = state_volume_usage()
            ram_pct = float(getattr(ram, "percent", 0.0) or 0.0) / 100.0
            disk_pct = float(getattr(disk, "percent", 0.0) or 0.0) / 100.0
            vitality = max(0.0, 1.0 - (max(cpu_pct, ram_pct, disk_pct) * 0.2))
            return {
                "thermal_load": cpu_pct,
                "resource_anxiety": ram_pct,
                "vitality": vitality,
            }
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Soma fallback telemetry failed: %s", exc)
            return {}

    soma = ServiceContainer.peek("soma", default=None)
    if not soma:
        return _system_fallback()

    if refresh and hasattr(soma, "pulse"):
        try:
            await asyncio.wait_for(soma.pulse(), timeout=0.25)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Soma pulse refresh failed: %s", exc)

    try:
        if hasattr(soma, "get_status"):
            raw = soma.get_status() or {}
            if isinstance(raw.get("soma"), dict):
                payload = dict(raw["soma"])
                if payload:
                    return payload
            if isinstance(raw, dict) and {"thermal_load", "resource_anxiety", "vitality"} & set(raw.keys()):
                payload = {
                    "thermal_load": float(raw.get("thermal_load", 0.0) or 0.0),
                    "resource_anxiety": float(raw.get("resource_anxiety", 0.0) or 0.0),
                    "vitality": float(raw.get("vitality", 0.0) or 0.0),
                }
                if payload:
                    return payload
        if hasattr(soma, "get_health"):
            raw = soma.get_health() or {}
            if isinstance(raw, dict):
                payload = {
                    "thermal_load": float(raw.get("thermal_load", 0.0) or 0.0),
                    "resource_anxiety": float(raw.get("resource_anxiety", 0.0) or 0.0),
                    "vitality": float(raw.get("vitality", 0.0) or 0.0),
                }
                if any(value > 0.0 for value in payload.values()):
                    return payload
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Soma status collection failed: %s", exc)
    return _system_fallback()


def _collect_tool_catalog() -> list[dict[str, Any]]:
    engine = optional_service("capability_engine", default=None)
    if not engine:
        return []

    try:
        raw_catalog: Any = None
        if hasattr(engine, "iter_tool_catalog"):
            raw_catalog = engine.iter_tool_catalog(include_inactive=True)
        elif hasattr(engine, "get_tool_catalog"):
            get_tool_catalog = engine.get_tool_catalog
            if inspect.isgeneratorfunction(get_tool_catalog):
                raw_catalog = get_tool_catalog(include_inactive=True)
            else:
                logger.warning(
                    "Skipping materialized tool catalog during UI bootstrap; "
                    "capability_engine should expose iter_tool_catalog()."
                )
                return []

        if raw_catalog is None:
            return []

        catalog: list[dict[str, Any]] = []
        started_at = time.monotonic()
        for index, item in enumerate(raw_catalog):
            if index >= _TOOL_CATALOG_BOOTSTRAP_MAX_ITEMS:
                break
            if time.monotonic() - started_at > _TOOL_CATALOG_BOOTSTRAP_READ_BUDGET_S:
                break
            if isinstance(item, dict):
                catalog.append(item)
        catalog.sort(
            key=lambda item: (
                0 if bool(item.get("available")) else 1,
                0 if bool(item.get("active")) else 1,
                str(item.get("name") or ""),
            )
        )
        return catalog
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Tool catalog collection failed: %s", exc)
    return []


def _collect_skill_catalog_health() -> dict[str, Any]:
    """Return one bounded, deterministic readiness contract for every UI route."""

    unavailable = {
        "ready": False,
        "reason": "capability_engine_unavailable",
        "missing_live": [],
        "quarantined": [],
        "quarantined_count": 0,
        "execution_preflight": {
            "complete": False,
            "failed": [],
            "ok": False,
            "reason": "capability_engine_unavailable",
        },
    }
    engine = optional_service("capability_engine", default=None)
    if engine is None or not hasattr(engine, "get_catalog_health"):
        return unavailable

    try:
        raw = engine.get_catalog_health()
        if not isinstance(raw, dict):
            return {**unavailable, "reason": "catalog_health_invalid"}

        missing_live = sorted(
            {
                str(name).strip()
                for name in raw.get("missing_live") or ()
                if str(name).strip()
            }
        )
        quarantined = []
        for item in raw.get("quarantined") or ():
            if not isinstance(item, dict):
                continue
            normalized = {
                key: str(item.get(key) or "").strip()
                for key in ("catalog_id", "class_name", "module_path", "name", "stage")
            }
            normalized["error"] = str(item.get("error") or item.get("detail") or "").strip()
            if any(normalized.values()):
                quarantined.append(normalized)
        quarantined.sort(
            key=lambda item: (
                item.get("name") or item.get("class_name") or item.get("catalog_id") or "",
                item.get("module_path") or "",
                item.get("stage") or "",
            )
        )

        raw_preflight = raw.get("execution_preflight")
        preflight = dict(raw_preflight) if isinstance(raw_preflight, dict) else {}
        preflight["complete"] = preflight.get("complete") is True
        preflight["ok"] = preflight.get("ok") is True
        preflight["failed"] = sorted(
            {
                str(name).strip()
                for name in preflight.get("failed") or ()
                if str(name).strip()
            }
        )
        preflight["reason"] = str(preflight.get("reason") or "not_run")

        health = dict(raw)
        health.update(
            {
                "ready": raw.get("ready") is True,
                "reason": str(raw.get("reason") or "catalog_health_unverified"),
                "missing_live": missing_live,
                "quarantined": quarantined,
                "quarantined_count": max(
                    _safe_int(raw.get("quarantined_count"), 0),
                    len(quarantined),
                ),
                "execution_preflight": preflight,
            }
        )
        return health
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Skill catalog health collection failed: %s", exc)
        return {
            **unavailable,
            "reason": "catalog_health_collection_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _collect_commitment_summary() -> dict[str, Any]:
    try:
        from core.agency.commitment_engine import get_commitment_engine

        engine = get_commitment_engine()
        active = engine.get_active_commitments()
        return {
            "active_count": len(active),
            "reliability_score": round(float(engine.reliability_score), 4),
            "active": [
                {
                    "id": item.id,
                    "description": item.description,
                    "outcome": item.outcome,
                    "status": item.status.value if hasattr(item.status, "value") else str(item.status),
                    "hours_remaining": round(float(item.hours_remaining()), 2),
                }
                for item in active[:5]
            ],
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Commitment summary collection failed: %s", exc)
        return {"active_count": 0, "reliability_score": 1.0, "active": []}


def _collect_voice_summary() -> dict[str, Any]:
    try:
        from interface.routes.privacy import get_voice_engine_fn

        _voice_engine_fn = get_voice_engine_fn()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Voice engine resolver unavailable: %s", exc)
        _voice_engine_fn = None
    voice_available = bool(_voice_engine_fn)
    summary = {
        "available": voice_available,
        "microphone_enabled": voice_available,
        "speaking_enabled": voice_available,
        "listening": False,
        "auto_listen": False,
        "server_capture": False,
        "capture_available": False,
        "stt_available": False,
        "stt_initialized": False,
        "streaming_available": voice_available,
        "state": "ready" if voice_available else "unavailable",
    }
    try:
        voice = _voice_engine_fn() if _voice_engine_fn else None
        if voice is not None:
            microphone_enabled = bool(getattr(voice, "microphone_enabled", True))
            speaking_enabled = bool(getattr(voice, "speaking_enabled", True))
            listening = bool(
                getattr(voice, "_mic_listening", False)
                or getattr(voice, "is_listening", False)
            )
            summary["microphone_enabled"] = microphone_enabled
            summary["speaking_enabled"] = speaking_enabled
            summary["listening"] = listening
            if hasattr(voice, "get_status"):
                voice_status = voice.get_status() or {}
                if isinstance(voice_status, dict):
                    summary["auto_listen"] = bool(voice_status.get("auto_listen", False))
                    summary["server_capture"] = bool(voice_status.get("server_capture", False))
                    summary["capture_available"] = bool(voice_status.get("capture_available", False))
                    summary["stt_available"] = bool(voice_status.get("stt_available", False))
                    summary["stt_initialized"] = bool(voice_status.get("stt_initialized", False))
                    summary["capture_backend"] = voice_status.get("capture_backend")
                    summary["stt_backend"] = voice_status.get("stt_backend")
                    summary["stt"] = voice_status.get("stt")
                    summary["tts"] = voice_status.get("tts")
            if not microphone_enabled and not speaking_enabled:
                summary["state"] = "muted"
            elif listening:
                summary["state"] = "listening"
            else:
                voice_state = getattr(getattr(voice, "state", None), "name", "") or ""
                if voice_state:
                    summary["state"] = str(voice_state).lower()
                else:
                    summary["state"] = "ready"
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Voice summary collection failed: %s", exc)
    return summary


async def _probe_desktop_access_summary(*, allow_probe: bool = True) -> dict[str, Any]:
    cached_payload = _desktop_access_cache.get("payload")
    cached_at = float(_desktop_access_cache.get("captured_at", 0.0) or 0.0)
    if (
        isinstance(cached_payload, dict)
        and (time.monotonic() - cached_at) < _desktop_access_cache_ttl(cached_payload)
    ):
        return _desktop_access_cached_copy(cached_payload, captured_at=cached_at)
    if not allow_probe:
        if isinstance(cached_payload, dict):
            return _desktop_access_cached_copy(
                cached_payload,
                captured_at=cached_at,
                stale=True,
                probe_mode="stale_cached",
            )
        payload = _desktop_access_empty_payload()
        payload["probe_mode"] = "fast_pending"
        payload["overall_status"] = "pending"
        payload["permission_confidence"] = "pending"
        payload["desktop_access_diagnosis"] = [
            "Desktop permission probing is handled by /api/system/desktop-access so health checks stay fast."
        ]
        return payload

    payload: dict[str, Any] = _desktop_access_empty_payload()
    payload["probe_mode"] = "full"
    try:
        from core.security.permission_guard import PermissionType, get_permission_guard
        from core.skills._pyautogui_runtime import get_pyautogui

        native_ready = False
        resident_native_ready = False
        if sys.platform == "darwin":
            try:
                from core.security.native_desktop_bridge import probe_native_desktop_bridge

                native_probe = await asyncio.wait_for(
                    asyncio.to_thread(probe_native_desktop_bridge, force=False),
                    timeout=max(0.2, _DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S),
                )
                _mark_desktop_access_probe_success("native_bridge", "resident")
                payload["native_bridge_probe"] = (
                    native_probe if isinstance(native_probe, dict)
                    else {"ok": False, "error": f"invalid:{type(native_probe).__name__}"}
                )
                resident_native_ready = bool(
                    isinstance(native_probe, dict)
                    and native_probe.get("ok")
                    and native_probe.get("bridge_transport") == "resident_ipc"
                    and all(
                        bool(native_probe.get(key))
                        for key in ("screen_recording", "accessibility", "automation")
                    )
                )
                native_ready = resident_native_ready
            except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                issue, streak = _record_desktop_access_probe_issue(
                    "native_bridge",
                    "resident",
                    exc,
                )
                payload["native_bridge_probe"] = {
                    "ok": False,
                    "error": str(exc)[:240] or type(exc).__name__,
                    "status": issue,
                    "probe_unavailable": True,
                    "retryable": True,
                    "failure_streak": streak,
                }

        guard = ServiceContainer.get("permission_guard", default=None) or get_permission_guard()
        if guard:
            identity_probe = getattr(guard, "current_process_identity", None)
            if callable(identity_probe):
                try:
                    payload["process_identity"] = identity_probe()
                except _SYSTEM_RECOVERABLE_ERRORS as exc:
                    _record_desktop_access_probe_issue(
                        "identity",
                        "current_process",
                        exc,
                    )
            if not native_ready:
                async def _bounded_reported_probe(ptype: Any) -> dict[str, Any]:
                    target = ptype.name.lower()
                    try:
                        result = await asyncio.wait_for(
                            guard.check_permission(ptype, force=False),
                            timeout=max(0.2, _DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S),
                        )
                        _mark_desktop_access_probe_success("reported", target)
                        return result if isinstance(result, dict) else {
                            "granted": False,
                            "status": "invalid_probe_result",
                            "guidance": "",
                            "detail": f"got {type(result).__name__}",
                        }
                    except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                        return _desktop_access_probe_unavailable(
                            guard,
                            ptype,
                            probe="reported",
                            exc=exc,
                        )

                screen, accessibility, automation = await asyncio.gather(
                    _bounded_reported_probe(PermissionType.SCREEN),
                    _bounded_reported_probe(PermissionType.ACCESSIBILITY),
                    _bounded_reported_probe(PermissionType.AUTOMATION),
                )
                payload["screen_recording"] = screen
                payload["accessibility"] = accessibility
                payload["automation"] = automation
                payload["frontmost_app"] = str(automation.get("detail", "") or "")
                direct_probe = getattr(guard, "check_permission_direct_local", None)
                if not callable(direct_probe):
                    direct_probe = getattr(guard, "check_permission_direct", None)
                if callable(direct_probe):
                    reported_by_type = {
                        PermissionType.SCREEN: screen,
                        PermissionType.ACCESSIBILITY: accessibility,
                        PermissionType.AUTOMATION: automation,
                    }

                    async def _bounded_direct_probe(ptype: Any) -> dict[str, Any]:
                        target = ptype.name.lower()
                        reported = reported_by_type.get(ptype, {})
                        if isinstance(reported, dict) and reported.get("probe_unavailable"):
                            inherited = dict(reported)
                            inherited["direct_probe"] = True
                            inherited["probe_source"] = "reported_probe"
                            return inherited
                        try:
                            result = await asyncio.wait_for(
                                direct_probe(ptype),
                                timeout=max(0.2, _DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S),
                            )
                            _mark_desktop_access_probe_success("direct", target)
                            return result if isinstance(result, dict) else {
                                "granted": False,
                                "status": "invalid_probe_result",
                                "guidance": "",
                                "detail": f"got {type(result).__name__}",
                                "direct_probe": True,
                            }
                        except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                            return _desktop_access_probe_unavailable(
                                guard,
                                ptype,
                                probe="direct",
                                exc=exc,
                            )

                    try:
                        direct_screen, direct_accessibility, direct_automation = await asyncio.gather(
                            _bounded_direct_probe(PermissionType.SCREEN),
                            _bounded_direct_probe(PermissionType.ACCESSIBILITY),
                            _bounded_direct_probe(PermissionType.AUTOMATION),
                        )
                        payload["direct_screen_recording"] = direct_screen
                        payload["direct_accessibility"] = direct_accessibility
                        payload["direct_automation"] = direct_automation
                    except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                        _record_desktop_access_probe_issue(
                            "direct_group",
                            "permissions",
                            exc,
                        )

        native_bridge = payload.get("native_bridge_probe")
        native_bridge_is_resident = (
            isinstance(native_bridge, dict)
            and native_bridge.get("ok")
            and native_bridge.get("bridge_transport") == "resident_ipc"
        )
        if native_bridge_is_resident:
            payload["effective_app_identity"] = {
                "bundle_identifier": str(native_bridge.get("bundle_identifier", "") or ""),
                "bridge_executable": str(native_bridge.get("bridge_executable", "") or ""),
                "bridge_transport": str(native_bridge.get("bridge_transport", "") or ""),
                "code_signature": native_bridge.get("code_signature")
                if isinstance(native_bridge.get("code_signature"), dict)
                else {},
            }
            native_common = {
                "status": "active_native_bridge",
                "guidance": "",
                "native_bridge": True,
                "bridge_executable": str(native_bridge.get("bridge_executable", "") or ""),
                "bundle_identifier": str(native_bridge.get("bundle_identifier", "") or ""),
                "direct_probe": True,
            }
            if native_bridge.get("screen_recording"):
                screen_result = {"granted": True, **native_common}
                payload["screen_recording"] = screen_result
                payload["direct_screen_recording"] = screen_result
            if native_bridge.get("accessibility"):
                accessibility_result = {"granted": True, **native_common}
                payload["accessibility"] = accessibility_result
                payload["direct_accessibility"] = accessibility_result
            if native_bridge.get("automation"):
                automation_result = {
                    "granted": True,
                    **native_common,
                    "frontmost_app": str(native_bridge.get("frontmost_app", "") or ""),
                }
                payload["automation"] = automation_result
                payload["direct_automation"] = automation_result

        pyautogui, pyautogui_error = get_pyautogui()
        payload["pyautogui_ready"] = pyautogui is not None
        if pyautogui_error:
            payload["pyautogui_error"] = str(pyautogui_error)[:240]

        screen_granted = bool((payload["screen_recording"] or {}).get("granted"))
        accessibility_granted = bool((payload["accessibility"] or {}).get("granted"))
        automation_granted = bool((payload["automation"] or {}).get("granted"))
        direct_screen_granted = bool((payload["direct_screen_recording"] or {}).get("granted"))
        direct_accessibility_granted = bool((payload["direct_accessibility"] or {}).get("granted"))
        direct_automation_granted = bool((payload["direct_automation"] or {}).get("granted"))
        unavailable_statuses = {
            "",
            "unknown",
            "deferred",
            "timeout",
            "probe_error",
            "probe_failed",
            "invalid_probe_result",
            "dependency_missing",
            "resident_bridge_required",
            "unverified_assertion",
            "asserted_env",
        }

        def _probe_has_evidence(result: Any) -> bool:
            return bool(
                isinstance(result, dict)
                and not result.get("probe_unavailable")
                and str(result.get("status") or "").lower()
                not in unavailable_statuses
            )

        reported_results = {
            "screen_recording": payload["screen_recording"],
            "accessibility": payload["accessibility"],
            "automation": payload["automation"],
        }
        direct_results = {
            "screen_recording": payload["direct_screen_recording"],
            "accessibility": payload["direct_accessibility"],
            "automation": payload["direct_automation"],
        }
        reported_probe_unavailable_permissions = [
            name for name, result in reported_results.items()
            if not _probe_has_evidence(result)
        ]
        direct_probe_unavailable_permissions = [
            name for name, result in direct_results.items()
            if not _probe_has_evidence(result)
        ]
        unverified_permissions = [
            name for name in reported_results
            if not _probe_has_evidence(direct_results[name])
            and not _probe_has_evidence(reported_results[name])
        ]
        payload["reported_probe_unavailable_permissions"] = (
            reported_probe_unavailable_permissions
        )
        payload["direct_probe_unavailable_permissions"] = (
            direct_probe_unavailable_permissions
        )
        payload["unverified_permissions"] = unverified_permissions
        direct_probe_available = any(
            _probe_has_evidence(result) for result in direct_results.values()
        )
        payload["direct_probe_available"] = direct_probe_available
        payload["reported_screen_capture_ready"] = screen_granted
        payload["reported_desktop_control_ready"] = accessibility_granted and bool(payload["pyautogui_ready"])
        payload["reported_screen_text_ready"] = automation_granted and accessibility_granted
        payload["direct_screen_capture_ready"] = direct_screen_granted
        payload["direct_desktop_control_ready"] = direct_accessibility_granted and bool(payload["pyautogui_ready"])
        payload["direct_screen_text_ready"] = direct_automation_granted and direct_accessibility_granted
        effective_screen_granted = (
            direct_screen_granted
            if _probe_has_evidence(payload["direct_screen_recording"])
            else screen_granted
        )
        effective_accessibility_granted = (
            direct_accessibility_granted
            if _probe_has_evidence(payload["direct_accessibility"])
            else accessibility_granted
        )
        effective_automation_granted = (
            direct_automation_granted
            if _probe_has_evidence(payload["direct_automation"])
            else automation_granted
        )
        payload["screen_capture_ready"] = effective_screen_granted
        payload["desktop_control_ready"] = (
            effective_accessibility_granted
        ) and bool(payload["pyautogui_ready"])
        payload["screen_text_ready"] = (
            effective_automation_granted and effective_accessibility_granted
        )
        payload["menu_clock_ready"] = (
            effective_automation_granted and effective_accessibility_granted
        )
        if payload["menu_clock_ready"]:
            from core.skills.computer_use import ComputerUseSkill

            def _probe_menu_clock() -> dict[str, Any]:
                from core.governance_context import local_internal_governed_scope
                skill = ComputerUseSkill()
                try:
                    with local_internal_governed_scope("system.probe_menu_clock", domain="tool_execution"):
                        text = skill._read_menu_clock_macos()
                    return {"ready": True, "text": text[:240]}
                except _SYSTEM_RECOVERABLE_ERRORS as exc:
                    return {"ready": False, "error": str(exc)[:240]}

            try:
                menu_clock_probe = await asyncio.wait_for(
                    asyncio.to_thread(_probe_menu_clock),
                    timeout=max(0.25, _DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S),
                )
            except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                _record_desktop_access_probe_issue(
                    "menu_clock",
                    "system_events",
                    exc,
                )
                menu_clock_probe = {
                    "ready": False,
                    "error": str(exc)[:240] or type(exc).__name__,
                }
            payload["menu_clock_ready"] = bool(menu_clock_probe.get("ready"))
            payload["menu_clock_text"] = str(menu_clock_probe.get("text", "") or "")
            payload["menu_clock_error"] = str(menu_clock_probe.get("error", "") or "")
        primary_ready = [
            payload["screen_capture_ready"],
            payload["desktop_control_ready"],
            payload["screen_text_ready"],
        ]
        reported_primary_ready = [
            payload["reported_screen_capture_ready"],
            payload["reported_desktop_control_ready"],
            payload["reported_screen_text_ready"],
        ]
        direct_primary_ready = [
            payload["direct_screen_capture_ready"],
            payload["direct_desktop_control_ready"],
            payload["direct_screen_text_ready"],
        ]
        payload["permission_assumptions"] = [
            name for name, result in (
                ("screen_recording", payload["screen_recording"]),
                ("accessibility", payload["accessibility"]),
                ("automation", payload["automation"]),
            )
            if str((result or {}).get("status") or "") == "asserted_env"
        ]
        reported_blocking_permissions = [
            name for name, granted in (
                ("screen_recording", screen_granted),
                ("accessibility", accessibility_granted),
                ("automation", automation_granted),
            ) if not granted
        ]
        direct_blocking_permissions = [
            name for name, granted in (
                ("screen_recording", direct_screen_granted),
                ("accessibility", direct_accessibility_granted),
                ("automation", direct_automation_granted),
            ) if not granted
        ]
        payload["reported_blocking_permissions"] = reported_blocking_permissions
        payload["direct_blocking_permissions"] = direct_blocking_permissions
        payload["blocking_permissions"] = [
            name for name, granted in (
                ("screen_recording", effective_screen_granted),
                ("accessibility", effective_accessibility_granted),
                ("automation", effective_automation_granted),
            ) if not granted
        ]
        payload["permission_confidence"] = (
            "direct"
            if all(direct_primary_ready) else
            "partial_direct"
            if any(direct_primary_ready) else
            "claims_only"
            if direct_probe_available and all(reported_primary_ready) and payload["permission_assumptions"] else
            "asserted_env"
            if all(reported_primary_ready) and payload["permission_assumptions"] else
            "unavailable"
            if unverified_permissions and not any(primary_ready) else
            "unverified"
            if payload["permission_assumptions"] else
            "blocked"
        )
        payload["overall_status"] = (
            "ready"
            if all(direct_primary_ready) else
            "claims_only"
            if direct_probe_available and all(reported_primary_ready) and payload["permission_assumptions"] else
            "assumed_ready"
            if all(reported_primary_ready) and payload["permission_assumptions"] else
            "partial"
            if any(direct_primary_ready) or (not direct_probe_available and any(primary_ready)) else
            "probe_unavailable"
            if unverified_permissions and not any(primary_ready) else
            "partial"
            if any(
                bool((payload[key] or {}).get("granted"))
                for key in ("screen_recording", "accessibility", "automation")
            ) else
            "blocked"
        )
        diagnosis: list[str] = []
        if payload.get("unverified_permissions"):
            diagnosis.append(
                "One or more passive permission probes were unavailable; Aura is preserving the distinction between unknown and macOS-denied access."
            )
        signature = {}
        if isinstance(payload.get("effective_app_identity"), dict):
            signature = payload["effective_app_identity"].get("code_signature") or {}
        if isinstance(signature, dict) and signature.get("stable_tcc_identity") is False:
            if signature.get("adhoc") or str(signature.get("signature") or "").strip().lower() == "adhoc":
                diagnosis.append(
                    "Aura.app is ad-hoc signed, so macOS permissions can attach to a stale rebuild instead of the currently running app."
                )
            else:
                diagnosis.append(
                    "Aura.app does not expose a stable signing authority, so macOS may not retain permissions reliably across rebuilds."
                )
            hint = str(signature.get("tcc_repair_hint") or "").strip()
            if hint:
                diagnosis.append(hint)
        if native_bridge_is_resident and payload["blocking_permissions"]:
            diagnosis.append(
                "The resident Aura.app bridge is reachable, but macOS denies the requested TCC grants for this exact app identity."
            )
            bundle_identifier = str(native_bridge.get("bundle_identifier") or "com.aura.desktop")
            bridge_executable = str(native_bridge.get("bridge_executable") or "/Applications/Aura.app/Contents/MacOS/aura-launcher")
            payload["tcc_repair_plan"] = {
                "reason": "resident_bridge_denied_current_tcc_grants",
                "bundle_identifier": bundle_identifier,
                "bridge_executable": bridge_executable,
                "blocking_permissions": list(payload["blocking_permissions"]),
                "commands": [
                    f"tccutil reset ScreenCapture {bundle_identifier}",
                    f"tccutil reset Accessibility {bundle_identifier}",
                ],
                "manual_steps": [
                    "Quit Aura completely.",
                    "Run the reset commands for the current Aura.app bundle identifier.",
                    "Open /Applications/Aura.app.",
                    "Approve Screen Recording and Accessibility when macOS prompts.",
                    "If System Settings still shows Aura as enabled but the bridge is denied, remove the Aura row with the minus button and add /Applications/Aura.app again.",
                ],
                "request_state": dict(_desktop_access_request_state),
                "verification_endpoint": "/api/system/desktop-access",
            }
        if isinstance(native_bridge, dict) and native_bridge.get("bridge_transport") == "one_shot_subprocess":
            diagnosis.append(
                "A diagnostic one-shot Aura.app bridge responded, but the resident Aura.app bridge is not alive; durable desktop control is blocked until the signed app stays resident."
            )
        if (
            payload.get("process_identity", {}).get("bundle_identifier") == "org.python.python"
            and payload.get("overall_status") != "ready"
        ):
            diagnosis.append(
                "The cognitive runtime is a Python child; durable desktop control should route through the resident Aura.app bridge, not Python's own TCC row."
            )
        payload["desktop_access_diagnosis"] = diagnosis
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Desktop access summary collection failed: %s", exc)
    payload["captured_at_unix"] = time.time()
    payload["probe_runtime"] = _desktop_access_probe_state_snapshot()
    payload["cache_ttl_s"] = _desktop_access_cache_ttl(payload)
    _desktop_access_cache["captured_at"] = time.monotonic()
    _desktop_access_cache["payload"] = payload
    return payload


async def _collect_desktop_access_summary(*, allow_probe: bool = True) -> dict[str, Any]:
    """Share one full desktop probe per event loop and preserve it on caller cancel."""
    cached_payload = _desktop_access_cache.get("payload")
    cached_at = float(_desktop_access_cache.get("captured_at", 0.0) or 0.0)
    if (
        isinstance(cached_payload, dict)
        and (time.monotonic() - cached_at) < _desktop_access_cache_ttl(cached_payload)
    ):
        return _desktop_access_cached_copy(cached_payload, captured_at=cached_at)
    if not allow_probe:
        return await _probe_desktop_access_summary(allow_probe=False)

    loop = asyncio.get_running_loop()
    task = _DESKTOP_ACCESS_PROBE_TASKS.get(loop)
    shared = task is not None and not task.done()
    if task is None or task.done():
        task = create_tracked_task(
            _probe_desktop_access_summary(allow_probe=True),
            name="system.desktop_access.shared_probe",
            owner="system.desktop_access",
        )
        _DESKTOP_ACCESS_PROBE_TASKS[loop] = task

        def _clear(completed: asyncio.Task[dict[str, Any]]) -> None:
            if _DESKTOP_ACCESS_PROBE_TASKS.get(loop) is completed:
                _DESKTOP_ACCESS_PROBE_TASKS.pop(loop, None)
            if completed.cancelled():
                return
            try:
                completed.exception()
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                return

        task.add_done_callback(_clear)

    result = await asyncio.shield(task)
    if not shared:
        return result
    copied = dict(result)
    copied["probe_mode"] = "shared_probe"
    copied["singleflight_shared"] = True
    return copied


@router.get("/system/desktop-access")
async def desktop_access_summary() -> dict[str, Any]:
    return await _collect_desktop_access_summary()


@router.post("/system/desktop-access/request-screen")
async def request_screen_access() -> dict[str, Any]:
    try:
        native_result: dict[str, Any] = {}
        try:
            from core.security.native_desktop_bridge import invoke_native_desktop_bridge

            native_result = invoke_native_desktop_bridge(
                "request_screen",
                read_only=True,
                timeout=45.0,
                prefer_one_shot=False,
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.desktop_access.native_request_screen",
                exc,
                action="falling back to Python Screen Recording request",
                severity="warning",
            )
        if native_result:
            granted = bool(native_result.get("screen_recording"))
            status = "granted" if granted else "approval_required"
            _desktop_access_request_state["screen_recording"] = {
                "requested": True,
                "granted": granted,
                "status": status,
                "target": "Aura.app",
                "bundle_identifier": str(native_result.get("bundle_identifier") or ""),
                "requested_at": time.time(),
                "detail": (
                    "macOS still requires user approval in Screen Recording for /Applications/Aura.app"
                    if not granted else
                    "Screen Recording is granted for the signed Aura.app bridge"
                ),
            }
            _desktop_access_cache["captured_at"] = 0.0
            return {
                "requested": True,
                "granted": granted,
                "status": status,
                "approval_required": not granted,
                "native_bridge": native_result,
                "target": "Aura.app",
            }

        from core.security.permission_guard import get_permission_guard

        guard = get_permission_guard()
        request = getattr(guard, "request_screen_capture_access", None)
        granted = bool(request()) if callable(request) else False
        _desktop_access_request_state["screen_recording"] = {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "target": "Python runtime",
            "requested_at": time.time(),
        }
        _desktop_access_cache["captured_at"] = 0.0
        return {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "approval_required": not granted,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.request_screen",
            exc,
            action="reported Screen Recording request failure",
            severity="warning",
        )
        return {"requested": False, "granted": False, "error": str(exc)[:240]}


@router.post("/system/desktop-access/request-accessibility")
async def request_accessibility_access() -> dict[str, Any]:
    try:
        native_result: dict[str, Any] = {}
        try:
            from core.security.native_desktop_bridge import invoke_native_desktop_bridge

            native_result = invoke_native_desktop_bridge(
                "request_accessibility",
                read_only=True,
                timeout=45.0,
                prefer_one_shot=False,
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.desktop_access.native_request_accessibility",
                exc,
                action="falling back to Python Accessibility request",
                severity="warning",
            )
        if native_result:
            granted = bool(native_result.get("accessibility"))
            status = "granted" if granted else "approval_required"
            _desktop_access_request_state["accessibility"] = {
                "requested": True,
                "granted": granted,
                "status": status,
                "target": "Aura.app",
                "bundle_identifier": str(native_result.get("bundle_identifier") or ""),
                "requested_at": time.time(),
                "detail": (
                    "macOS still requires user approval in Accessibility for /Applications/Aura.app"
                    if not granted else
                    "Accessibility is granted for the signed Aura.app bridge"
                ),
            }
            _desktop_access_cache["captured_at"] = 0.0
            return {
                "requested": True,
                "granted": granted,
                "status": status,
                "approval_required": not granted,
                "native_bridge": native_result,
                "target": "Aura.app",
            }

        from core.security.permission_guard import get_permission_guard

        guard = get_permission_guard()
        request = getattr(guard, "request_accessibility_trust", None)
        granted = bool(request()) if callable(request) else False
        _desktop_access_request_state["accessibility"] = {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "target": "Python runtime",
            "requested_at": time.time(),
        }
        _desktop_access_cache["captured_at"] = 0.0
        return {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "approval_required": not granted,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.request_accessibility",
            exc,
            action="reported Accessibility request failure",
            severity="warning",
        )
        return {"requested": False, "granted": False, "error": str(exc)[:240]}


@router.post("/system/desktop-access/open-settings/{permission}")
async def open_desktop_access_settings(permission: str) -> dict[str, Any]:
    aliases = {
        "screen": "SCREEN",
        "screen_recording": "SCREEN",
        "screencapture": "SCREEN",
        "accessibility": "ACCESSIBILITY",
        "automation": "AUTOMATION",
    }
    normalized = aliases.get(str(permission or "").strip().lower())
    if not normalized:
        return {
            "opened": False,
            "permission": permission,
            "error": "unknown_permission",
        }
    try:
        from core.security.permission_setup import open_settings_pane

        opened = bool(open_settings_pane(normalized))
        return {
            "opened": opened,
            "permission": normalized,
            "target": "System Settings",
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.open_settings",
            exc,
            action="reported desktop permission settings launch failure",
            severity="warning",
        )
        return {
            "opened": False,
            "permission": normalized,
            "error": str(exc)[:240],
        }


def _collect_neurodynamic_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "idle",
        "action": "",
        "uncertainty": 0.0,
        "confidence": 0.0,
        "advisory_only": True,
        "authority_gateway_required_for_effects": True,
    }
    try:
        advisor = ServiceContainer.get("spiking_active_inference", default=None)
        if advisor is None or not hasattr(advisor, "snapshot"):
            return payload
        snapshot = advisor.snapshot() or {}
        if not isinstance(snapshot, dict):
            return payload
        governance = snapshot.get("governance") or {}
        if not isinstance(governance, dict):
            governance = {}
        payload.update(
            {
                "status": str(snapshot.get("status") or "active"),
                "action": str(snapshot.get("action") or ""),
                "uncertainty": _safe_float(snapshot.get("uncertainty"), 0.0),
                "confidence": _safe_float(snapshot.get("confidence"), 0.0),
                "advisory_only": bool(governance.get("advisory_only", True)),
                "authority_gateway_required_for_effects": bool(
                    governance.get("authority_gateway_required_for_effects", True)
                ),
            }
        )
        features = snapshot.get("features")
        if isinstance(features, dict):
            payload["features"] = {
                "tool_pressure": _safe_float(features.get("tool_pressure"), 0.0),
                "error_pressure": _safe_float(features.get("error_pressure"), 0.0),
                "memory_pressure": _safe_float(features.get("memory_pressure"), 0.0),
            }
        stability = snapshot.get("stability")
        if isinstance(stability, dict):
            payload["stability"] = {
                "spectral_radius": _safe_float(stability.get("spectral_radius"), 0.0),
                "entropy": _safe_float(stability.get("entropy"), 0.0),
                "winner_margin": _safe_float(stability.get("winner_margin"), 0.0),
                "decision_instability": _safe_float(
                    stability.get("decision_instability"), 0.0
                ),
                "ode_spectral_abscissa": _safe_float(
                    stability.get("ode_spectral_abscissa"), 0.0
                ),
                "fixed_point_residual": _safe_float(
                    stability.get("fixed_point_residual"), 0.0
                ),
                "bifurcation_pressure": _safe_float(
                    stability.get("bifurcation_pressure"), 0.0
                ),
            }
        working_memory = snapshot.get("working_memory")
        if isinstance(working_memory, dict):
            payload["working_memory"] = {
                "admission": str(working_memory.get("admission") or "unknown"),
                "admitted": bool(working_memory.get("admitted", True)),
                "queue_load": _safe_float(working_memory.get("queue_load"), 0.0),
                "overload_pressure": _safe_float(working_memory.get("overload_pressure"), 0.0),
                "utilization": _safe_float(working_memory.get("utilization"), 0.0),
                "expected_wait_s": _safe_float(working_memory.get("expected_wait_s"), 0.0),
            }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Neurodynamic status collection failed: %s", exc)
    return payload


def _collect_imagination_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "idle",
        "frames": 0,
        "latest": None,
        "working_memory": {},
        "attractor_bias": {},
        "eligibility_trace": {},
        "recent_outcomes": [],
        "advisory_only": True,
        "no_external_effects": True,
        "authority_gateway_required_for_effects": True,
    }
    try:
        engine = ServiceContainer.peek("imagination_engine", default=None)
        if engine is None or not hasattr(engine, "snapshot"):
            return payload
        snapshot = engine.snapshot() or {}
        if not isinstance(snapshot, dict):
            return payload
        governance = snapshot.get("governance") or {}
        if not isinstance(governance, dict):
            governance = {}
        payload.update(
            {
                "status": str(snapshot.get("status") or "active"),
                "frames": int(_safe_float(snapshot.get("frames"), 0.0)),
                "latest": snapshot.get("latest") if isinstance(snapshot.get("latest"), dict) else None,
                "working_memory": snapshot.get("working_memory") if isinstance(snapshot.get("working_memory"), dict) else {},
                "attractor_bias": snapshot.get("attractor_bias") if isinstance(snapshot.get("attractor_bias"), dict) else {},
                "eligibility_trace": snapshot.get("eligibility_trace") if isinstance(snapshot.get("eligibility_trace"), dict) else {},
                "recent_outcomes": snapshot.get("recent_outcomes") if isinstance(snapshot.get("recent_outcomes"), list) else [],
                "advisory_only": bool(governance.get("advisory_only", True)),
                "no_external_effects": bool(governance.get("no_external_effects", True)),
                "authority_gateway_required_for_effects": bool(
                    governance.get("authority_gateway_required_for_effects", True)
                ),
            }
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Imagination status collection failed: %s", exc)
    return payload


# Renders of imagination frames, newest last. Keyed by frame_id so a frame is
# never rendered twice and the panel can pair an image with the frame that
# produced it. Bounded: this is a view cache, not a gallery.
_IMAGINATION_RENDERS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_IMAGINATION_RENDER_LIMIT = 12
_IMAGINATION_RENDER_LOCK = asyncio.Lock()


@router.post("/imagination/visualize")
async def api_imagination_visualize(request: Request) -> JSONResponse:
    """Render the current imagination frame's own mental canvas into an image.

    Aura's imagination engine describes what it would picture — `mental_canvas.
    image_prompt` is prose, not pixels — and its own externalization_path says to
    request governed image execution to actually see it. This is that request: it
    takes the prompt she already wrote for herself and puts it through the
    ordinary governed image_gen lane. It invents no prompt of its own.

    Owner-only, because it spends real compute (loads a diffusion pipeline
    beside the resident model) rather than just reading state.
    """
    if not _owner_authenticated(request):
        raise HTTPException(status_code=403, detail="Rendering imagination is owner-only")

    snapshot = await asyncio.to_thread(_collect_imagination_status)
    frame = snapshot.get("latest") if isinstance(snapshot, dict) else None
    if not isinstance(frame, dict):
        return JSONResponse(
            {"ok": False, "error": "Aura has not imagined anything to render."},
            status_code=409,
        )

    canvas = frame.get("mental_canvas") if isinstance(frame.get("mental_canvas"), dict) else {}
    prompt = str(canvas.get("image_prompt") or "").strip()
    if not prompt:
        return JSONResponse(
            {"ok": False, "error": "This frame carries no mental canvas to render."},
            status_code=409,
        )

    frame_id = str(frame.get("frame_id") or "")
    async with _IMAGINATION_RENDER_LOCK:
        cached = _IMAGINATION_RENDERS.get(frame_id)
        if cached:
            return JSONResponse({"ok": True, "render": cached, "cached": True})

        try:
            from core.capability_engine import execute_tool

            result = await execute_tool(
                "image_gen",
                {
                    "prompt": prompt,
                    # Her canvas names its own sensory style; use hers.
                    "style": canvas.get("sensory_style") or None,
                    # No quality boosters: "masterpiece, 8k, HDR, cinematic
                    # lighting" overrides an abstract prompt and renders a
                    # photoreal scene instead of the canvas she described. This
                    # must draw her words, not a stock-photo reading of them.
                    "enhance": False,
                    "width": 512,
                    "height": 512,
                    "steps": 4,
                    "guidance_scale": 0.0,
                },
                # The owner clicked RENDER THIS on the authenticated desktop
                # panel. Without this contract the governed spine correctly
                # treats the render as an ungrounded autonomous act: standing
                # authority finds no user-facing origin (live veto:
                # "signed_standing_authority_lease_missing") and the AuraNow
                # present-state policy defers it. The desktop execution
                # contract is what lets an explicit owner action traverse the
                # full tool spine — same envelope as chat's desktop lane.
                context={
                    "origin": "desktop_ui",
                    "source": "desktop_ui",
                    "route": "system.imagination_visualize",
                    "objective": (
                        "Render the imagination frame's own mental canvas — "
                        "owner clicked RENDER THIS"
                    ),
                    "desktop_execution_contract": True,
                    "foreground_request": True,
                    "user_requested_action": True,
                    "user_explicit_action_request": True,
                    "user_explicitly_authorized": True,
                    "user_visible_desktop_action": True,
                    "local_desktop_action": True,
                    "verification_required": True,
                    "predicted_outcome": (
                        "A rendered image of the frame's mental canvas is "
                        "returned and shown in the Imagine panel."
                    ),
                },
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc, action="imagination render failed")
            logger.warning("Imagination render failed: %s", exc)
            return JSONResponse(
                {"ok": False, "error": f"Render failed: {exc}"}, status_code=503
            )

        if not isinstance(result, dict) or not result.get("ok") or not result.get("url"):
            detail = ""
            if isinstance(result, dict):
                detail = str(result.get("error") or result.get("message") or "")
            return JSONResponse(
                {"ok": False, "error": detail or "Image generation did not return an image."},
                status_code=503,
            )

        render = {
            "frame_id": frame_id,
            "url": result.get("url"),
            "prompt": prompt,
            "objective": frame.get("objective"),
            "modality": canvas.get("modality"),
            "created_at": time.time(),
        }
        _IMAGINATION_RENDERS[frame_id] = render
        while len(_IMAGINATION_RENDERS) > _IMAGINATION_RENDER_LIMIT:
            _IMAGINATION_RENDERS.popitem(last=False)
        return JSONResponse({"ok": True, "render": render, "cached": False})


@router.get("/imagination")
async def api_imagination() -> JSONResponse:
    """Aura's live imagination workspace, for the Imagine panel.

    The same frame the engine is actually reasoning with — it already ships
    inside /api/health, but that payload is large and slow to assemble, and the
    panel wants to poll the workspace on its own cadence. Read-only: this is a
    view onto ``ImaginationEngine.snapshot()``, it never constructs a frame.

    ``status`` is "idle" until she has imagined something. The panel renders
    that as an honest empty state rather than inventing a canvas.
    """
    payload = await asyncio.to_thread(_collect_imagination_status)
    worlds: list[dict[str, Any]] = []
    try:
        from core.worlds import get_world_host

        worlds = await asyncio.to_thread(get_world_host().list_worlds)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="imagination panel served without the worlds list",
        )
        logger.debug("World list unavailable for imagination panel: %s", exc)
    payload["worlds"] = worlds if isinstance(worlds, list) else []
    # Newest last; the panel pairs render.frame_id with latest.frame_id so an
    # image is only ever shown against the frame that actually produced it.
    payload["renders"] = list(_IMAGINATION_RENDERS.values())
    return JSONResponse(_json_safe(payload))


def _collect_runtime_capabilities(conversation_lane: dict[str, Any] | None = None) -> dict[str, Any]:
    lane = conversation_lane if isinstance(conversation_lane, dict) else _collect_conversation_lane_status_resilient()
    payload: dict[str, Any] = {
        "local_backend": "unknown",
        "local_runtime": "offline",
        "conversation_model": str(lane.get("desired_model", "") or ""),
        "conversation_endpoint": str(lane.get("desired_endpoint", "") or ""),
        "conversation_state": str(lane.get("state", "") or ""),
        "conversation_ready": bool(lane.get("conversation_ready", False)),
        "neurodynamic_advisor": _collect_neurodynamic_status(),
        "imagination_engine": _collect_imagination_status(),
    }
    try:
        from core.brain.llm.model_registry import (
            ACTIVE_MODEL,
            BRAINSTEM_MODEL,
            DEEP_MODEL,
            FALLBACK_MODEL,
            get_local_backend,
        )

        payload.update(
            {
                "local_backend": get_local_backend(),
                "cortex_model": ACTIVE_MODEL,
                "solver_model": DEEP_MODEL,
                "brainstem_model": BRAINSTEM_MODEL,
                "fallback_model": FALLBACK_MODEL,
            }
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Runtime capability backend lookup failed: %s", exc)

    state = str(payload.get("conversation_state", "") or "").lower()
    if bool(payload.get("conversation_ready")):
        payload["local_runtime"] = "online"
    elif _conversation_lane_is_standby_resilient(lane):
        payload["local_runtime"] = "standby"
    elif state in {"cold", "warming", "spawning", "handshaking", "recovering", "ready"}:
        payload["local_runtime"] = "warming"
    elif state == "failed":
        payload["local_runtime"] = "degraded"
    return payload


def _derive_ui_status_flags(
    *,
    state_summary: dict[str, Any],
    executive_status: dict[str, Any],
    boot_snapshot: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    skill_catalog_health: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if not bool(boot_snapshot.get("ready", False)):
        flags.append("booting")
    if bool(state_summary.get("thermal_guard")):
        flags.append("thermal_guard")
    if _safe_float(state_summary.get("coherence_score"), 1.0) < 0.72:
        flags.append("coherence_low")
    if _safe_float(state_summary.get("fragmentation_score"), 0.0) > 0.4:
        flags.append("fragmentation_high")
    if _safe_int(state_summary.get("contradiction_count"), 0) > 3:
        flags.append("contradictions_present")
    epistemics = state_summary.get("epistemics", {}) or {}
    if _safe_int(epistemics.get("contested"), 0) > 0:
        flags.append("beliefs_contested")
    unavailable_count = sum(1 for tool in tool_catalog if not bool(tool.get("available")))
    if unavailable_count >= 3:
        flags.append("tool_unavailable")
    if skill_catalog_health.get("ready") is False:
        flags.append("skill_catalog_blocked")
    if skill_catalog_health.get("missing_live"):
        flags.append("skill_missing_live")
    if _safe_int(skill_catalog_health.get("quarantined_count"), 0) > 0:
        flags.append("skill_quarantined")
    if str(executive_status.get("last_target") or "").strip().lower() == "secondary":
        flags.append("executive_hold")
    return flags


# ── Routes ────────────────────────────────────────────────────

@router.get("/telemetry/stream")
async def telemetry_stream(request: Request):
    """Server-Sent Events stream for HUD telemetry."""
    _require_internal(request)

    async def event_generator():
        try:
            init_payload = {
                "type": "telemetry",
                "cpu_usage": psutil.cpu_percent(interval=None),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": time.time(),
            }
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation("system", e)
            logger.debug("SSE initial telemetry snapshot failed: %s", e)
            init_payload = {"type": "telemetry", "cpu_usage": 0.0, "memory_usage": 0.0, "timestamp": time.time()}
        init_data = json.dumps(init_payload)
        yield f"event: telemetry\ndata: {init_data}\n\n"

        q = None
        try:
            q = await broadcast_bus.subscribe()
            while not await request.is_disconnected():
                while q.qsize() > _SSE_QUEUE_BACKLOG_LIMIT:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        break

                try:
                    item = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_HEARTBEAT_S)
                except TimeoutError:
                    heartbeat = json.dumps(runtime_heartbeat_payload("heartbeat"))
                    yield f"event: heartbeat\ndata: {heartbeat}\n\n"
                    continue

                try:
                    _priority, _ts, msg = item
                    safe_msg = _json_safe(msg) if isinstance(msg, dict) else {"type": "message", "payload": _json_safe(msg)}
                    msg_type = str(safe_msg.get("type", "message") or "message")
                    data = json.dumps(safe_msg)
                    yield f"event: {msg_type}\ndata: {data}\n\n"
                except asyncio.CancelledError:
                    break
                except _SYSTEM_RECOVERABLE_ERRORS as e:
                    record_degradation('system', e)
                    logger.debug("SSE generate error: %s", e)
                    await asyncio.sleep(0.1)
                    continue
                finally:
                    q.task_done()
        finally:
            if q is not None:
                await broadcast_bus.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/metrics", tags=["metrics"])
async def metrics(request: Request):
    """System metrics for monitoring (JSON format, backwards compatible)."""
    _require_internal(request)
    try:
        from core.runtime.health_contract import runtime_health_report

        orch = ServiceContainer.get("orchestrator", default=None)
        orch_status = orch.get_status() if orch else {}
        contract = runtime_health_report()

        return {
            "status": contract.get("status", "unknown"),
            "healthy": bool(contract.get("healthy", False)),
            "operational": bool(contract.get("operational", False)),
            "required_probes": contract.get("required_probes", {}),
            "uptime": time.time() - (orch_status.get("start_time", time.time()) if orch_status else time.time()),
            "active_connections": ws_manager.count(),
            "cycle_count": orch_status.get("cycle_count", 0),
            "cpu_usage": float(int(psutil.cpu_percent() * 10)) / 10.0 if 'psutil' in sys.modules else 0,
            "memory_usage": float(int(psutil.virtual_memory().percent * 10)) / 10.0 if 'psutil' in sys.modules else 0,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Metrics collection failed: %s", e, exc_info=True)
        return ORJSONResponse({"status": "error", "message": "Metrics collection failed"}, status_code=500)


@router.get("/metrics/prometheus", tags=["metrics"])
async def metrics_prometheus(request: Request):
    """Prometheus-compatible metrics in text exposition format.

    Scrape this endpoint with Prometheus or any compatible collector.
    """
    _require_internal(request)
    try:
        from fastapi.responses import Response

        from core.observability.metrics import get_metrics

        text = get_metrics().render_prometheus()
        return Response(
            content=text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Prometheus metrics render failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"status": "error", "message": "Prometheus metrics unavailable"},
            status_code=500,
        )


@router.get("/system/incidents", tags=["health"])
async def api_system_incidents(request: Request, minutes: float = 60.0):
    """Receipt-backed incident narrative over Aura's own forensics.

    Deterministic synthesis of stall dumps, degraded events, the memory
    sentinel, and boot timings into causal episodes — 'what happened and
    why', with a receipt for every claim. This is the operator's answer to
    'why was she slow?' without an hour of grep.
    """
    try:
        from core.observability.incident_narrator import get_incident_narrator

        minutes = max(1.0, min(float(minutes), 24 * 60.0))
        report = await asyncio.to_thread(get_incident_narrator().narrate, minutes)
        return JSONResponse(_json_safe(report))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned empty incident narrative after narrator failure",
        )
        logger.warning("Incident narrative unavailable: %s", exc)
        return JSONResponse(
            {
                "schema": "aura.incident_narrative.v1",
                "episodes": [],
                "error": "incident narrative unavailable",
            },
            status_code=200,
        )


@router.get("/system/control-plane", tags=["health"])
async def api_system_control_plane(request: Request):
    """Return owner-only desired-state convergence and scheduler evidence."""

    _require_internal(request)
    control_plane = ServiceContainer.peek("runtime_control_plane", default=None)
    if control_plane is None or not callable(getattr(control_plane, "get_status", None)):
        return JSONResponse(
            {
                "schema": "aura.runtime_control_plane.diagnostics.v1",
                "available": False,
                "reason": "runtime_control_plane_not_registered",
            },
            status_code=503,
        )
    status = await _optional_threaded_status(
        "runtime_control_plane",
        control_plane.get_status,
        timeout_s=1.0,
        fallback={"alive": False, "ready": False},
    )
    scheduler_health = scheduler.get_health()
    reconcile_task = dict(
        (scheduler_health.get("task_details") or {}).get(
            "runtime_control_plane_reconcile",
            {},
        )
    )
    return JSONResponse(
        _json_safe(
            {
                "schema": "aura.runtime_control_plane.diagnostics.v1",
                "available": not bool(status.get("_stale", False)),
                "control_plane": status,
                "reconcile_task": reconcile_task,
            }
        )
    )


@router.get("/system/memory/growth", tags=["health"])
async def api_system_memory_growth(request: Request, top: int = 25):
    """Allocation-growth attribution for the idle-leak investigation.

    Requires a launch with AURA_RUNTIME_HYGIENE_TRACEMALLOC=1 (opt-in;
    ~2x allocation overhead). First call arms the baseline snapshot;
    later calls return the top-N call sites by size growth since the
    baseline — the direct answer to 'WHAT is growing', not just how much.
    """
    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        if not hasattr(hygiene, "allocation_growth"):
            return JSONResponse(
                {"available": False, "reason": "runtime_hygiene_unavailable"},
                status_code=200,
            )
        report = await asyncio.to_thread(hygiene.allocation_growth, top)
        return JSONResponse(_json_safe(report))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned unavailable memory-growth report after hygiene failure",
        )
        return JSONResponse(
            {"available": False, "reason": "memory_growth_failed"},
            status_code=200,
        )


@router.get("/system/learning", tags=["health"])
async def api_system_learning(request: Request):
    """The weight-learning stack's live state, receipts included.

    One view over the whole loop: the compounding scheduler (when it last
    trained, what happened), the self-play flywheel (practice bursts,
    correct-rate trace, pairs produced), the lineage ledger's verdict (the
    only place a compounding claim may come from), and the expert-adapter
    library. This is the operator's answer to 'what has she learned lately?'
    """
    payload: dict = {"schema": "aura.learning_status.v1"}
    try:
        from core.container import ServiceContainer

        def _collect() -> dict:
            out: dict = {}
            scheduler = ServiceContainer.get("weight_compounding", default=None)
            if scheduler is not None and hasattr(scheduler, "get_status"):
                out["compounding"] = scheduler.get_status()
            flywheel = ServiceContainer.get("selfplay_flywheel", default=None)
            if flywheel is not None and hasattr(flywheel, "get_status"):
                out["selfplay"] = flywheel.get_status()
            try:
                from core.learning.verifiable_preference_harness import (
                    get_verifiable_preference_harness,
                )

                out["preference_store"] = get_verifiable_preference_harness().stats()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["preference_store"] = {"error": "unavailable"}
            try:
                from core.brain.expert_lora_library import get_expert_lora_library

                out["expert_library"] = get_expert_lora_library().stats()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["expert_library"] = {"error": "unavailable"}
            try:
                from core.runtime.service_access import resolve_practice_director

                director = resolve_practice_director(default=None)
                if director is not None and hasattr(director, "get_status"):
                    out["practice_director"] = director.get_status()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["practice_director"] = {"error": "unavailable"}
            return out

        payload.update(await asyncio.to_thread(_collect))
        return JSONResponse(_json_safe(payload))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned degraded learning status after collection failure",
        )
        logger.warning("Learning status unavailable: %s", exc)
        payload["error"] = "learning status unavailable"
        return JSONResponse(_json_safe(payload), status_code=200)


@router.get("/healthz", tags=["health"])
async def healthz(request: Request):
    """Liveness probe: is the process alive and responsive?

    Returns 200 if the server can respond to HTTP at all.
    Used by orchestrators (systemd, launchd, docker) to detect crashes.
    """
    try:
        from core.observability.metrics import check_liveness

        result = check_liveness()
        if is_shutdown_requested():
            result = dict(result)
            result["status"] = "stopping"
            result["shutdown"] = _shutdown_health_status()
        return JSONResponse(result, status_code=200)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.warning("Liveness check degraded; returning process-level alive response: %s", exc)
        payload: dict[str, Any] = {"status": "alive", "pid": os.getpid()}
        if is_shutdown_requested():
            payload["status"] = "stopping"
            payload["shutdown"] = _shutdown_health_status()
        return JSONResponse(payload, status_code=200)


@router.get("/readyz", tags=["health"])
async def readyz(request: Request):
    """Serve a compact readiness verdict from the versioned health read model."""
    if is_shutdown_requested():
        return JSONResponse(
            {
                "status": "stopping",
                "ready": False,
                "issues": ["runtime_shutdown"],
                "shutdown": _shutdown_health_status(),
            },
            status_code=503,
        )
    try:
        from core.runtime.health_contract import required_probe_groups_pass

        snapshot = _apply_health_read_model_truth(_HEALTH_READ_MODEL.read())
        snapshot = _apply_runtime_revision_truth(snapshot)
        snapshot = _apply_current_shutdown_truth(snapshot)
        readiness = dict(snapshot.get("readiness_contract") or {})
        required_probes = dict(
            readiness.get("required_probes")
            or snapshot.get("required_probes")
            or {}
        )
        ready = bool(
            readiness.get("healthy") is True
            and readiness.get("system_ready") is True
            and readiness.get("conversation_ready") is True
            and readiness.get("runtime_probe_healthy") is True
            and required_probe_groups_pass(required_probes)
        )
        issues = list(
            dict.fromkeys(
                str(item)
                for item in (
                    list(snapshot.get("blockers") or [])
                    + list(readiness.get("blockers") or [])
                )
                if str(item)
            )
        )
        if not ready and not issues:
            if readiness.get("system_ready") is not True:
                issues.append("system_not_ready")
            if readiness.get("conversation_ready") is not True:
                issues.append("conversation_lane_not_ready")
            if readiness.get("runtime_probe_healthy") is not True:
                issues.append("runtime_probe_unhealthy")
            if not required_probe_groups_pass(required_probes):
                issues.append("runtime_required_probes")
        metadata = dict(snapshot.get("health_read_model") or {})
        result = {
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "issues": issues,
            "uptime_s": round(float(snapshot.get("uptime", 0.0) or 0.0), 1),
            "conversation_ready": readiness.get("conversation_ready") is True,
            "runtime_probe_healthy": readiness.get("runtime_probe_healthy") is True,
            "required_probes_passed": required_probe_groups_pass(required_probes),
            "snapshot_generation": int(metadata.get("snapshot_generation", 0) or 0),
            "snapshot_age_s": round(float(metadata.get("age_s", 0.0) or 0.0), 3),
            "serving": str(metadata.get("serving", "unknown") or "unknown"),
        }
        status_code = 200 if ready else 503
        return JSONResponse(result, status_code=status_code)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse(
            {"status": "not_ready", "ready": False, "issues": [str(e)]},
            status_code=503,
        )


@router.get("/incidents", tags=["observability"])
async def incidents(request: Request):
    """Active incidents and incident manager summary."""
    _require_internal(request)
    try:
        from core.resilience.incident_manager import get_incident_manager

        manager = get_incident_manager()
        return JSONResponse({
            "summary": manager.get_summary(),
            "active": manager.get_active(),
        })
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse(
            {"summary": {}, "active": [], "error": str(e)},
            status_code=200,
        )


@router.get("/db-maintenance", tags=["observability"])
async def db_maintenance_status(request: Request):
    """Database maintenance status and last run results."""
    _require_internal(request)
    try:
        from core.persistence.db_maintenance import get_db_maintenance
        return JSONResponse(get_db_maintenance().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/resources", tags=["observability"])
async def resource_status(request: Request):
    """Resource governor status: thermal, memory, inference."""
    _require_internal(request)
    try:
        from core.resource.resource_governor import get_resource_governor
        return JSONResponse(get_resource_governor().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/initiative-overflow", tags=["observability"])
async def initiative_overflow_status(request: Request):
    """Initiative overflow and skill gap status."""
    _require_internal(request)
    try:
        from core.autonomy.initiative_overflow import get_initiative_overflow
        return JSONResponse(get_initiative_overflow().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/user-engagement", tags=["observability"])
async def user_engagement_status(request: Request):
    """User response tracking and engagement metrics."""
    _require_internal(request)
    try:
        from core.autonomy.user_response_tracker import get_user_response_tracker
        return JSONResponse(get_user_response_tracker().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


async def _collect_api_health_payload(
    *,
    allow_owner_loop_reads: bool = True,
) -> dict[str, Any]:
    """Build the rich UI health payload outside the public request path."""

    orch       = ServiceContainer.peek("orchestrator", default=None)
    rt         = _get_runtime_state_safe()
    runtime_payload = rt.get("state", {}) if isinstance(rt.get("state"), dict) else {}
    status_obj = getattr(orch, "status", None)

    initialized = getattr(status_obj, "initialized", False)
    connected   = orch is not None and getattr(status_obj, "running", False)

    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        p_core = per_cpu[0] if len(per_cpu) > 1 else cpu
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Hardware stats collection failed: %s", e)
        cpu, ram, p_core = 0, 0, 0

    orch_status = {}
    if orch and hasattr(orch, "get_status"):
        try:
            orch_status = orch.get_status()
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation('system', e)
            logger.debug("get_status failed: %s", e)
    conversation_lane = _collect_conversation_lane_status_resilient()
    boot_snapshot, _ = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    connected = bool(
        boot_snapshot.get("system_ready", False)
        or (
            boot_snapshot.get("ready", False)
            and boot_snapshot.get("conversation_ready", False)
        )
    )

    ls_data = {}
    try:
        ls = ServiceContainer.peek("liquid_substrate", default=None) or ServiceContainer.peek("liquid_state", default=None)
        if ls and hasattr(ls, "get_status"):
            ls_data = ls.get_status()

        vad_data = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "_stale": True}
        engine = ServiceContainer.peek("cognitive_engine", default=None)
        if (
            allow_owner_loop_reads
            and engine
            and hasattr(engine, "consciousness")
        ):
            v_state = await asyncio.wait_for(
                engine.consciousness.substrate.get_state_summary(),
                timeout=0.25,
            )
            vad_data = {
                "valence": v_state.get("valence", 0.0),
                "arousal": v_state.get("arousal", 0.0),
                "dominance": v_state.get("dominance", 0.0),
                "volatility": v_state.get("volatility", 0.0),
                "_stale": False,
            }
            ls_dict = cast(dict, ls_data)
            ls_dict["vad"] = vad_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Liquid state/VAD lookup failed: %s", e)
    curiosity_status = orch_status.get("curiosity_status", {})

    transcendence_data = {"meta_evolution": {"active": False, "acceleration_factor": 1.0}}
    try:
        meta = ServiceContainer.peek("meta_cognition", default=None)
        if meta:
            transcendence_data["meta_evolution"] = meta.get_health()
            transcendence_data["meta_evolution"]["active"] = True
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Transcendence status collection failed: %s", e)

    # Agency: derive from energy + curiosity + active autonomous thought.
    _energy_raw = _normalize_percentish(ls_data.get("energy")) or 0.0
    _curiosity_raw = _normalize_percentish(ls_data.get("curiosity")) or 0.0
    thought_task = getattr(orch, "_current_thought_task", None) if orch else None
    try:
        _thinking = bool(thought_task and hasattr(thought_task, "done") and not thought_task.done())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Current thought task status failed: %s", e)
        _thinking = False
    _agency_score = (_energy_raw * 0.4 + _curiosity_raw * 0.4 + (30.0 if _thinking else 0.0))
    _agency_score = min(100.0, max(0.0, _agency_score))

    scratchpad_engine = ServiceContainer.peek("scratchpad_engine", default=None)
    subconscious_loop = ServiceContainer.peek("subconscious_loop", default=None)
    subconscious_active = bool(
        subconscious_loop is not None
        and getattr(subconscious_loop, "_running", False)
    )

    cortex = {
        "agency":    float(int(_agency_score * 10)) / 10.0,
        "curiosity": float(int(_curiosity_raw * 10)) / 10.0,
        "fixes":     orch_status.get("stats", {}).get("modifications_made", 0),
        "beliefs":   0,
        "episodes":  0,
        "active_topic": curiosity_status.get("active_topic", "None"),
        "goals":     orch_status.get("stats", {}).get("goals_processed", 0),
        "autonomy":  config.security.aura_full_autonomy,
        "stealth":   config.security.enable_stealth_mode,
        "scratchpad": scratchpad_engine is not None,
        "forge":      ServiceContainer.peek("hephaestus_engine", default=None) is not None,
        "subconscious": "dreaming" if subconscious_active and _safe_float(getattr(orch, "boredom", 0), 0.0) > 45 else ("awake" if subconscious_active else "idle"),
        "unity":      ServiceContainer.peek("soma", default=None) is not None,
        "p_core_usage": float(int(_safe_float(p_core) * 10)) / 10.0,
        "singularity_factor": float(int(_safe_float(transcendence_data.get("meta_evolution", {}).get("acceleration_factor"), 1.0) * 100)) / 100.0,
        "meta_loop_active": transcendence_data.get("meta_evolution", {}).get("active", False)
    }

    if config.security.force_unity_on:
        cortex["unity"] = True
    try:
        if orch and hasattr(orch, "self_model") and orch.self_model:
            cortex["beliefs"] = len(getattr(orch.self_model, "beliefs", []))

        ep_mem = ServiceContainer.peek("episodic_memory", default=None)
        if ep_mem and hasattr(ep_mem, "get_summary_cached"):
            # Off-loop + TTL-cached: the fresh get_summary() runs eight
            # aggregate queries and stalled the event loop for 5.1s live.
            ep_summary = (
                await asyncio.to_thread(ep_mem.get_summary_cached)
                if allow_owner_loop_reads
                else ep_mem.get_summary_cached()
            )
            cortex["episodes"] = ep_summary.get("total_episodes", 0)
        elif ep_mem and hasattr(ep_mem, "get_summary"):
            ep_summary = (
                await asyncio.to_thread(ep_mem.get_summary)
                if allow_owner_loop_reads
                else ep_mem.get_summary()
            )
            cortex["episodes"] = ep_summary.get("total_episodes", 0)
        else:
            mem_mgr = ServiceContainer.peek("memory_manager", default=None)
            if mem_mgr and hasattr(mem_mgr, "get_stats"):
                mem_stats = mem_mgr.get_stats()
                cortex["episodes"] = mem_stats.get("episodic_count", 0)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Cortex supplementary metrics failed: %s", e)

    moral_data = {}
    try:
        moral = ServiceContainer.peek("moral", default=None)
        moral_data = moral.get_health() if moral and hasattr(moral, "get_health") else {}
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Moral health collection failed: %s", e)

    homeo_data = {}
    try:
        homeostasis = ServiceContainer.peek("homeostasis", default=None)
        homeo_data = homeostasis.get_health() if homeostasis and hasattr(homeostasis, "get_health") else {}
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Homeostasis health collection failed: %s", e)
    homeostasis_payload = _collect_homeostasis_public_payload(
        homeo_data if isinstance(homeo_data, dict) else {}
    )
    liquid_state_payload = _collect_liquid_state_payload(
        cast(dict[str, Any], ls_data if isinstance(ls_data, dict) else {}),
        runtime_state=runtime_payload if isinstance(runtime_payload, dict) else {},
        homeostasis_data=homeostasis_payload,
    )
    soma_data = await _collect_soma_payload(refresh=allow_owner_loop_reads)

    social_data = {"depth": 0.0}
    try:
        social = ServiceContainer.peek("social", default=None)
        social_data = social.get_health() if social and hasattr(social, "get_health") else social_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Social health collection failed: %s", e)

    swarm_data = {"active_count": 0}
    try:
        swarm_data = orch.swarm_status if orch and hasattr(orch, 'swarm_status') else swarm_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Swarm status collection failed: %s", e)

    executive_closure_data = {}
    try:
        executive_closure_data = orch_status.get("executive_closure", {}) or {}
        if not executive_closure_data:
            executive_closure = ServiceContainer.peek("executive_closure", default=None)
            if executive_closure and hasattr(executive_closure, "get_status"):
                executive_closure_data = executive_closure.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Executive closure status collection failed: %s", e)

    consciousness_evidence = {}
    try:
        consciousness_evidence = orch_status.get("consciousness_evidence", {}) or {}
        if not consciousness_evidence:
            evidence = ServiceContainer.peek("consciousness_evidence", default=None)
            if evidence and hasattr(evidence, "snapshot"):
                consciousness_evidence = evidence.snapshot()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Consciousness evidence collection failed: %s", e)

    executive_authority_data = {}
    try:
        executive_authority = ServiceContainer.peek("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_authority_data = executive_authority.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Executive authority status collection failed: %s", e)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.peek("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Interaction signal status collection failed: %s", e)

    # ── Resilience Status ──
    resilience_data: dict[str, Any] = {"circuit_breakers": {}, "snapshot": "unknown", "llm_tier": "unknown"}
    try:
        voice = ServiceContainer.peek("voice_engine", default=None)
        if voice:
            for attr_name in ("_stt_breaker", "_tts_breaker"):
                breaker = getattr(voice, attr_name, None)
                if breaker and hasattr(breaker, "state"):
                    cast(dict[str, Any], resilience_data["circuit_breakers"])[breaker.name] = breaker.state.value

        cog = ServiceContainer.peek("cognitive_engine", default=None)
        if cog:
            for attr_name in dir(cog):
                obj = getattr(cog, attr_name, None)
                if obj and hasattr(obj, "state") and hasattr(obj, "name") and hasattr(obj.state, "value"):
                    if "breaker" in attr_name.lower():
                        cast(dict[str, Any], resilience_data["circuit_breakers"])[obj.name] = obj.state.value

        snap_mgr = ServiceContainer.peek("snapshot_manager", default=None)
        if snap_mgr and hasattr(snap_mgr, "snapshot_file"):
            resilience_data["snapshot"] = "saved" if snap_mgr.snapshot_file.exists() else "none"

        llm_router = ServiceContainer.peek("llm_router", default=None)
        tier_value = conversation_lane.get("foreground_tier")
        if llm_router and hasattr(llm_router, "get_health_report"):
            report = llm_router.get_health_report()
            tier_value = report.get("foreground_tier") or tier_value
        if not tier_value and cog:
            tier_value = (getattr(cog, "_current_tier", None)
                          or getattr(cog, "last_tier", None))
        if tier_value:
            resilience_data["llm_tier"] = str(tier_value)
        else:
            if llm_router and hasattr(llm_router, "_active_model"):
                model = str(getattr(llm_router, "_active_model", "") or "")
                resilience_data["llm_tier"] = "local" if "mlx" in model.lower() or "local" in model.lower() else "cloud"

        resilience_data["active_endpoint"] = conversation_lane.get("foreground_endpoint")
        resilience_data["background_endpoint"] = conversation_lane.get("background_endpoint")
        resilience_data["conversation_lane"] = conversation_lane
        if llm_router:
            if hasattr(llm_router, "endpoints"):
                ep_status = {}
                for name, ep in llm_router.endpoints.items():
                    ep_status[name] = {
                        "tier": getattr(ep, "tier", "unknown"),
                        "available": ep.is_available() if hasattr(ep, "is_available") else True,
                        "state": ep.state.value if hasattr(ep, "state") and hasattr(ep.state, "value") else "unknown",
                    }
                resilience_data["llm_endpoints"] = ep_status

        resilience_data["hardening_active"] = ServiceContainer.peek("stability_guardian", default=None) is not None
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Resilience status collection failed: %s", e)

    # ── Qualia Status ──
    qualia_data: dict[str, Any] = {"pri": 0.0, "q_norm": 0.0, "dominant_dim": "none", "in_attractor": False, "_stale": True}
    try:
        qualia = ServiceContainer.peek("qualia_synthesizer", default=None)
        if not qualia and orch:
            qualia = getattr(orch, "qualia", None)
        if qualia:
            qualia_data["_stale"] = False
            qualia_data["pri"] = round(float(getattr(qualia, "pri", 0.0)), 4)
            qualia_data["q_norm"] = round(float(getattr(qualia, "q_norm", 0.0)), 4)
            qualia_data["dominant_dim"] = getattr(qualia, "_history", None) and len(qualia._history) > 0 and qualia._history[-1].dominant_dimension or "none"
            qualia_data["in_attractor"] = getattr(qualia, "_in_attractor", False)
            qualia_data["identity_coherence"] = round(float(getattr(qualia, "identity_drift_score", 1.0)) * 100, 1)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Qualia status collection failed: %s", e)

    # ── Mycelial Network Status ──
    mycelial_data: dict[str, Any] = {"nodes": 0, "edges": 0, "health": "offline"}
    try:
        mycelium = ServiceContainer.peek("mycelial_network", default=None)
        if mycelium:
            counter = getattr(mycelium, "get_topology_counts", None)
            if callable(counter):
                counts = counter()
                mycelial_data["nodes"] = int(counts.get("pathways", 0))
                mycelial_data["edges"] = int(counts.get("hyphae", 0))
            else:
                topology_reader = getattr(mycelium, "get_network_topology", None)
                topology = topology_reader() if callable(topology_reader) else {}
                mycelial_data["nodes"] = int(topology.get("pathway_count", 0) or 0)
                mycelial_data["edges"] = len(topology.get("hyphae") or {})
            mycelial_data["health"] = "online"
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Mycelial network status collection failed: %s", e)

    # ── PNEUMA Engine Status ──
    pneuma_data: dict[str, Any] = {"temperature": 0.7, "arousal": 0.0, "stability": 0.0,
                   "attractor_count": 0, "efe_score": 0.0, "online": False, "_stale": True}
    try:
        pn = ServiceContainer.peek("pneuma", default=None)
        if pn:
            runtime_state = pn.get_state_dict()
            pneuma_data["online"] = bool(runtime_state.get("online", False))
            pneuma_data["_stale"] = not bool(runtime_state.get("online", False))
            pneuma_data["temperature"] = round(pn.get_llm_temperature(), 3)
            pe = getattr(pn, "precision", None)
            if pe and hasattr(pe, "fhn"):
                s = pe.fhn.state
                pneuma_data["arousal"] = round(float(s.v), 3)
                pneuma_data["stability"] = round(float(s.w), 3)
            tm = getattr(pn, "topo_memory", None)
            if tm:
                pneuma_data["attractor_count"] = int(tm.attractor_count)
            pneuma_data["tick_count"] = runtime_state.get("tick_count", 0)
            pneuma_data["last_tick"] = runtime_state.get("last_tick", 0.0)
            pneuma_data["loop_errors"] = runtime_state.get("loop_errors", 0)
            pneuma_data["compute_budget"] = runtime_state.get("compute_budget", {})
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("PNEUMA status collection failed: %s", e)

    # ── MHAF Field Status ──
    mhaf_data: dict[str, Any] = {"phi": 0.0, "nodes": 0, "edges": 0, "free_energy": 0.0,
                 "lexicon_size": 0, "online": False, "_stale": True}
    try:
        mhaf = ServiceContainer.peek("mhaf", default=None)
        if mhaf:
            runtime_state = mhaf.get_state_dict()
            mhaf_data["online"] = bool(runtime_state.get("online", False))
            mhaf_data["_stale"] = not bool(runtime_state.get("online", False))
            mhaf_data["nodes"] = len(mhaf._nodes)
            mhaf_data["edges"] = len(mhaf._edges)
            mhaf_data["free_energy"] = round(float(mhaf._free_energy), 4)
            mhaf_data["tick_count"] = runtime_state.get("tick_count", 0)
            mhaf_data["last_tick"] = runtime_state.get("last_tick", 0.0)
            mhaf_data["loop_errors"] = runtime_state.get("loop_errors", 0)
            mhaf_data["compute_budget"] = runtime_state.get("compute_budget", {})
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("MHAF status collection failed: %s", e)
    # Wire real PhiCore IIT 4.0 phi into the MHAF data (replaces the surrogate)
    try:
        phi_core = ServiceContainer.peek("phi_core", default=None)
        if phi_core is not None:
            result = phi_core._last_result
            live_phi = 0.0
            if hasattr(phi_core, "get_live_phi"):
                live_phi = float(phi_core.get_live_phi(include_surrogate=True))
            if live_phi > 0.0:
                mhaf_data["phi"] = round(live_phi, 4)
                mhaf_data["phi_source"] = "phi_s" if result is not None else "surrogate"
            if result is not None:
                mhaf_data["phi"] = round(float(result.phi_s), 4)
                mhaf_data["phi_complex"] = result.is_complex
                mhaf_data["phi_mip"] = result.mip_description
                mhaf_data["phi_samples"] = result.tpm_n_samples
                # A bare φ is the number that could not stand on its own: the
                # old estimator scored a MEMORYLESS system at 0.60 and could
                # rank it above a genuinely coupled ring. What makes a value
                # readable is what it was measured on, by which estimator, and
                # how much of it survives its own sampling null — so the
                # provenance travels with it rather than being available
                # somewhere else.
                if hasattr(result, "provenance"):
                    mhaf_data["phi_provenance"] = result.provenance()
                selection = getattr(phi_core, "_last_selection", None)
                if selection is not None and hasattr(selection, "as_metrics"):
                    mhaf_data["phi_selection"] = selection.as_metrics()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("PhiCore status collection failed: %s", e)
    if mhaf_data.get("phi", 0.0) <= 0.0:
        try:
            closed_loop = ServiceContainer.peek("closed_causal_loop", default=None)
            if closed_loop is not None and hasattr(closed_loop, "get_status"):
                closed_loop_phi = float(
                    ((closed_loop.get_status() or {}).get("phi") or {}).get("estimate") or 0.0
                )
                if closed_loop_phi > 0.0:
                    mhaf_data["phi"] = round(closed_loop_phi, 4)
                    mhaf_data["phi_source"] = "closed_loop"
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation('system', e)
            logger.debug("Closed-loop phi fallback failed: %s", e)
    try:
        neo = ServiceContainer.peek("neologism_engine", default=None)
        if neo:
            mhaf_data["lexicon_size"] = len(neo._lexicon)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Neologism lexicon count failed: %s", e)

    # ── Security Status ──
    security_data: dict[str, Any] = {
        "trust_level": "unknown", "threat_score": 0.0,
        "integrity_ok": True, "passphrase_set": False, "_stale": True,
    }
    try:
        te = ServiceContainer.peek("trust_engine", default=None)
        if te is not None:
            ts = te.get_status()
            security_data["trust_level"] = ts.get("level", "guest")
            security_data["_stale"] = False
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Security status collection failed: %s", e)
    try:
        ep = ServiceContainer.peek("emergency_protocol", default=None)
        if ep is not None:
            eps = ep.get_status()
            security_data["threat_score"] = eps.get("threat_score", 0.0)
            security_data["threat_level"] = eps.get("threat_level", "none")
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("Emergency protocol status collection failed: %s", _exc)
    try:
        integrity_guardian = ServiceContainer.peek(
            "integrity_guardian", default=None
        )
        if integrity_guardian is not None:
            igs = integrity_guardian.get_status()
            security_data["integrity_ok"] = bool(
                igs.get("integrity_ok", igs.get("alert_count", 0) == 0)
            )
            security_data["integrity_files"] = igs.get("manifest_files", 0)
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("Integrity guardian status collection failed: %s", _exc)
    try:
        user_recognizer = ServiceContainer.peek("user_recognizer", default=None)
        if user_recognizer is not None:
            security_data["passphrase_set"] = user_recognizer.has_passphrase()
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("User recognizer status collection failed: %s", _exc)

    # ── Circadian State ──
    circadian_data: dict[str, Any] = {}
    try:
        ce = ServiceContainer.peek("circadian", default=None)
        if ce is not None:
            s = ce.state
            circadian_data = {
                "phase": s.phase.value,
                "arousal_baseline": round(s.arousal_baseline, 3),
                "energy_modifier": round(s.energy_modifier, 3),
                "cognitive_mode": s.cognitive_mode,
            }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Circadian status collection failed: %s", e)

    # ── Substrate Learning ──
    substrate_data: dict[str, Any] = {}
    try:
        lora_bridge = ServiceContainer.peek("crsm_lora_bridge", default=None)
        if lora_bridge is not None:
            substrate_data["lora_bridge"] = await _optional_threaded_status(
                "crsm_lora_bridge",
                lora_bridge.get_status,
                timeout_s=0.18,
                fallback={"loop": None},
                offload=allow_owner_loop_reads,
            )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("LoRA bridge status failed: %s", e)
    try:
        consolidator = ServiceContainer.peek(
            "experience_consolidator", default=None
        )
        if consolidator is not None:
            substrate_data["consolidator"] = await _optional_threaded_status(
                "experience_consolidator",
                consolidator.get_status,
                timeout_s=0.18,
                offload=allow_owner_loop_reads,
            )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Consolidator status failed: %s", e)

    # ── Morphogenesis Status ──
    morphogenesis_data: dict[str, Any] = {"online": False, "cells": 0, "organs": 0, "_stale": True}
    try:
        morpho_rt = ServiceContainer.peek("morphogenetic_runtime", default=None)
        if morpho_rt is not None and hasattr(morpho_rt, "status"):
            ms = morpho_rt.status()
            morphogenesis_data = {
                "online": ms.get("running", False),
                "enabled": ms.get("enabled", False),
                "tick": ms.get("tick", 0),
                "cells": ms.get("registry", {}).get("cells", 0),
                "organs": ms.get("registry", {}).get("organs", 0),
                "queued_signals": ms.get("queued_signals", 0),
                "last_tick_error": ms.get("last_tick_error", ""),
                "_stale": False,
            }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Morphogenesis status collection failed: %s", e)

    # ── Terminal Fallback Status ──
    terminal_data: dict[str, Any] = {"active": False, "pending": 0, "watchdog": False}
    try:
        tf = ServiceContainer.peek("terminal_fallback", default=None)
        if tf is not None:
            terminal_data["active"] = tf.is_active
            terminal_data["pending"] = len(tf._pending)
        tw = ServiceContainer.peek("terminal_watchdog", default=None)
        terminal_data["watchdog"] = tw._running if tw else False
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Terminal fallback status collection failed: %s", e)

    desktop_access_data = await _collect_desktop_access_summary(allow_probe=False)
    imagination_data = _collect_imagination_status()

    # ── Final Response Assembly ──
    try:
        voice_mod = ServiceContainer.peek("voice_engine", default=None)
        smc_mod = ServiceContainer.peek("sensory_motor_cortex", default=None)
        from interface.routes.privacy import get_browser_camera_privacy

        browser_camera_privacy = get_browser_camera_privacy()

        privacy_data = {
            "camera_enabled": bool(browser_camera_privacy.get("enabled", False)),
            "camera_mode": browser_camera_privacy.get("mode", "off"),
            "camera_reason": browser_camera_privacy.get("reason"),
            "continuous_camera_enabled": getattr(smc_mod, "camera_enabled", False),
            "microphone_enabled": getattr(voice_mod, "microphone_enabled", True),
            "microphone_listening": bool(
                getattr(voice_mod, "_mic_listening", False)
                or getattr(voice_mod, "is_listening", False)
            ),
            "speaking_enabled": getattr(voice_mod, "speaking_enabled", True),
        }

        full_runtime = _collect_full_runtime_status(pneuma_data, mhaf_data)

        conversation_ready = bool(conversation_lane.get("conversation_ready", False))
        conversation_busy = conversation_lane_is_busy(conversation_lane)
        lane_is_standby = _conversation_lane_is_standby_resilient(conversation_lane)
        service_ok = bool(boot_snapshot.get("system_ready", False))
        required_probes = boot_snapshot.get("required_probes", {})
        probe_blockers = required_probe_blockers(required_probes)
        required_probes_ok = required_probe_groups_pass(required_probes)
        health_blockers = list(dict.fromkeys(
            [str(item) for item in (boot_snapshot.get("blockers", []) or []) if str(item)]
            + probe_blockers
        ))
        health_blockers = _normalize_conversation_health_blockers(
            health_blockers,
            conversation_ready=conversation_ready,
            conversation_busy=conversation_busy,
        )
        if full_runtime.get("full_runtime_expected") and not full_runtime.get("ready"):
            health_blockers.extend(
                f"full_runtime:{name}"
                for name in full_runtime.get("blockers", [])
            )
            health_blockers = list(dict.fromkeys(health_blockers))
        healthy_ready = bool(
            service_ok
            and required_probes_ok
            and conversation_ready
            and not health_blockers
        )
        integrity_report = _collect_runtime_integrity_report()
        integrity_payload = _runtime_integrity_public_payload(integrity_report)
        proof_readiness_healthy = bool(integrity_payload.get("proof_readiness", False))
        certification_ready = bool(healthy_ready and proof_readiness_healthy)
        diagnostics_data = {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        }

        health_status = _derive_api_health_status(
            healthy_ready=healthy_ready,
            service_ok=service_ok,
            lane_is_standby=lane_is_standby,
            lane_state=str(conversation_lane.get("state", "") or ""),
            conversation_ready=conversation_ready,
            conversation_busy=conversation_busy,
            boot_snapshot=boot_snapshot,
        )

        payload = {
            "status":      health_status,
            "healthy":     healthy_ready,
            "version":     version_string("full"),
            "connected":   connected,
            "initialized": initialized,
            "cycle_count": orch_status.get("cycle_count", getattr(status_obj, "cycle_count", 0)),
            "uptime":      round(float(time.time() - (getattr(status_obj, "start_time", None) or getattr(orch, "start_time", None) or time.time())), 1),
            "cpu_usage":   cpu,
            "ram_usage":   ram,
            "cortex":      cortex,
            "liquid_state": liquid_state_payload,
            "soma":        soma_data,
            "moral":       moral_data,
            "homeostasis": homeostasis_payload,
            "social":      social_data,
            "swarm":       swarm_data,
            "resilience":  resilience_data,
            "qualia":         qualia_data,
            "mycelial":       mycelial_data,
            "pneuma":         pneuma_data,
            "mhaf":           mhaf_data,
            "security":       security_data,
            "circadian":      circadian_data,
            "substrate":      substrate_data,
            "morphogenesis":  morphogenesis_data,
            "terminal":       terminal_data,
            "desktop_access": desktop_access_data,
            "imagination":    imagination_data,
            "transcendence": transcendence_data,
            "privacy":        privacy_data,
            "executive_closure": executive_closure_data,
            "consciousness_evidence": consciousness_evidence,
            "executive_authority": executive_authority_data,
            "interaction_signals": interaction_signals_data,
            "integrity": integrity_payload,
            "runtime_revision": _runtime_revision_contract(),
            "full_runtime": full_runtime,
            "full_runtime_ready": bool(full_runtime.get("ready")),
            "proof_readiness_healthy": proof_readiness_healthy,
            "certification_ready": certification_ready,
            "integrity_blockers": integrity_payload.get("proof_blockers", []),
            "conversation_lane": conversation_lane,
            "diagnostics": diagnostics_data,
            "readiness_contract": {
                "healthy": healthy_ready,
                "system_ready": service_ok,
                "conversation_ready": conversation_ready,
                "conversation_busy": conversation_busy,
                "runtime_probe_healthy": required_probes_ok,
                "full_runtime_ready": bool(full_runtime.get("ready")),
                "full_runtime": full_runtime,
                "proof_readiness_healthy": proof_readiness_healthy,
                "certification_ready": certification_ready,
                "integrity": integrity_payload,
                "integrity_blockers": integrity_payload.get("proof_blockers", []),
                "required_probes": required_probes,
                "blockers": health_blockers,
            },
            "runtime_probe_healthy": required_probes_ok,
            "conversation_ready": conversation_ready,
            "conversation_busy": conversation_busy,
            "required_probes": required_probes,
            "blockers": health_blockers,
            "runtime":        rt,
            "scheduler":      scheduler.get_health(),
            "boot":           boot_snapshot,
            "timestamp":      datetime.now(tz=UTC).isoformat(),
        }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Final health payload assembly failed: %s", e)
        payload = {
            "status": "degraded",
            "error": str(e),
            "version": version_string("full"),
            "uptime": 0.0,
            "cycle_count": 0,
            "cpu_usage": 0,
            "ram_usage": 0,
            "runtime_revision": _runtime_revision_fallback_contract(),
            "timestamp": datetime.now(tz=UTC).isoformat()
        }

    payload = _apply_runtime_revision_truth(payload)
    shutdown = _shutdown_health_status()
    shutdown_request = shutdown.get("request")
    if isinstance(shutdown_request, dict) and shutdown_request.get("requested") is True:
        payload["status"] = "stopping"
        payload["healthy"] = False
        payload["connected"] = False
        payload["conversation_ready"] = False
        payload["runtime_probe_healthy"] = False
        payload["certification_ready"] = False
        payload["shutdown"] = shutdown
        required_probe_payload = payload.get("required_probes")
        if isinstance(required_probe_payload, dict):
            required_probe_payload["all_passed"] = False
        blockers = [str(item) for item in payload.get("blockers", [])]
        if "runtime_shutdown" not in blockers:
            blockers.insert(0, "runtime_shutdown")
        payload["blockers"] = blockers
        readiness = payload.get("readiness_contract")
        if isinstance(readiness, dict):
            readiness["healthy"] = False
            readiness["system_ready"] = False
            readiness["conversation_ready"] = False
            readiness["runtime_probe_healthy"] = False
            readiness["certification_ready"] = False
            readiness["blockers"] = blockers
    else:
        payload["shutdown"] = shutdown

    safe_payload = _json_safe(payload)
    return safe_payload if isinstance(safe_payload, dict) else {"status": "degraded"}


def _health_snapshot_fallback() -> dict[str, Any]:
    timestamp = datetime.now(tz=UTC).isoformat()
    blockers = ["health_snapshot_initializing"]
    required_probes = {"all_passed": False}
    boot = {
        "status": "booting",
        "ready": False,
        "system_ready": False,
        "conversation_ready": False,
        "boot_phase": "health_snapshot_initializing",
        "blockers": blockers,
        "required_probes": required_probes,
        "timestamp": timestamp,
    }
    return {
        "status": "booting",
        "healthy": False,
        "connected": False,
        "initialized": False,
        "version": version_string("full"),
        "cycle_count": 0,
        "uptime": 0.0,
        "cpu_usage": 0.0,
        "ram_usage": 0.0,
        "conversation_lane": {
            "state": "initializing",
            "conversation_ready": False,
        },
        "conversation_ready": False,
        "conversation_busy": False,
        "runtime_probe_healthy": False,
        "proof_readiness_healthy": False,
        "certification_ready": False,
        "required_probes": required_probes,
        "blockers": blockers,
        "runtime_revision": _runtime_revision_fallback_contract(),
        "readiness_contract": {
            "healthy": False,
            "system_ready": False,
            "conversation_ready": False,
            "conversation_busy": False,
            "runtime_probe_healthy": False,
            "proof_readiness_healthy": False,
            "certification_ready": False,
            "required_probes": required_probes,
            "blockers": blockers,
        },
        "boot": boot,
        "timestamp": timestamp,
    }


def _collect_api_health_snapshot_sync() -> dict[str, Any]:
    """Run the pull collector on its sole worker, never on the HTTP loop."""

    return asyncio.run(
        _collect_api_health_payload(allow_owner_loop_reads=False)
    )


def _new_health_read_model() -> HealthSnapshotReadModel:
    refresh_s = _env_positive_float("AURA_UI_HEALTH_REFRESH_S", 5.0)
    return HealthSnapshotReadModel(
        _collect_api_health_snapshot_sync,
        _health_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=refresh_s,
            max_stale_s=max(
                refresh_s,
                _env_positive_float("AURA_UI_HEALTH_MAX_STALE_S", 30.0),
            ),
            collection_timeout_s=_env_positive_float(
                "AURA_UI_HEALTH_COLLECTION_TIMEOUT_S", 8.0
            ),
            retry_base_s=_env_positive_float(
                "AURA_UI_HEALTH_RETRY_BASE_S", 2.0
            ),
            retry_max_s=_env_positive_float(
                "AURA_UI_HEALTH_RETRY_MAX_S", 30.0
            ),
        ),
    )


_HEALTH_READ_MODEL = _new_health_read_model()


def start_health_read_model() -> bool:
    """Prewarm the public health snapshot after server services register."""

    from core.runtime.integrity_audit import start_integrity_read_model

    invalidate_runtime_revision_cache()
    start_integrity_read_model()
    return _HEALTH_READ_MODEL.start()


def stop_health_read_model() -> None:
    from core.runtime.integrity_audit import stop_integrity_read_model

    _HEALTH_READ_MODEL.close()
    stop_integrity_read_model()
    invalidate_runtime_revision_cache()


def _reset_health_read_model_for_test() -> None:
    from core.runtime.integrity_audit import reset_integrity_read_model_for_test

    _HEALTH_READ_MODEL.reset_for_test()
    reset_integrity_read_model_for_test()
    invalidate_runtime_revision_cache()


def _force_unhealthy_snapshot(
    payload: dict[str, Any],
    *,
    blocker: str,
    status: str,
) -> dict[str, Any]:
    """Withhold current-health claims when the read model has no current proof."""

    result = dict(payload)
    blockers = list(
        dict.fromkeys(
            [blocker]
            + [str(item) for item in result.get("blockers", []) if str(item)]
        )
    )
    result.update(
        status=status,
        healthy=False,
        connected=False,
        conversation_ready=False,
        runtime_probe_healthy=False,
        proof_readiness_healthy=False,
        certification_ready=False,
        blockers=blockers,
    )

    required_probes = dict(result.get("required_probes") or {})
    required_probes["all_passed"] = False
    result["required_probes"] = required_probes

    readiness = dict(result.get("readiness_contract") or {})
    readiness.update(
        healthy=False,
        system_ready=False,
        conversation_ready=False,
        runtime_probe_healthy=False,
        proof_readiness_healthy=False,
        certification_ready=False,
        required_probes=required_probes,
        blockers=blockers,
    )
    result["readiness_contract"] = readiness

    boot = dict(result.get("boot") or {})
    boot_blockers = list(
        dict.fromkeys(
            [blocker]
            + [str(item) for item in boot.get("blockers", []) if str(item)]
        )
    )
    boot.update(
        status=status,
        ready=False,
        system_ready=False,
        conversation_ready=False,
        blockers=boot_blockers,
        required_probes=required_probes,
    )
    result["boot"] = boot
    return result


def _apply_health_read_model_truth(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("health_read_model")
    if not isinstance(metadata, dict) or not bool(metadata.get("expired")):
        return payload
    initializing = metadata.get("captured_at_unix") is None
    result = _force_unhealthy_snapshot(
        payload,
        blocker=(
            "health_snapshot_initializing"
            if initializing
            else "health_snapshot_expired"
        ),
        status="booting" if initializing else "stale",
    )
    revision = payload.get("runtime_revision")
    if isinstance(revision, dict):
        result["runtime_revision"] = _runtime_revision_unavailable(
            "health_snapshot_expired",
            required=revision.get("required") is True,
        )
    return result


def _apply_current_shutdown_truth(payload: dict[str, Any]) -> dict[str, Any]:
    shutdown = _shutdown_health_status()
    request = shutdown.get("request")
    if not isinstance(request, dict) or request.get("requested") is not True:
        result = dict(payload)
        result["shutdown"] = shutdown
        return result
    result = _force_unhealthy_snapshot(
        payload,
        blocker="runtime_shutdown",
        status="stopping",
    )
    result["shutdown"] = shutdown
    return result


@router.get("/health")
async def api_health(request: Request):
    """Serve the latest versioned snapshot without running live probes inline."""

    _mark_runtime_service_progress("api.health")
    _restore_owner_session_from_request(request)
    payload = _apply_health_read_model_truth(_HEALTH_READ_MODEL.read())
    payload = _apply_runtime_revision_truth(payload)
    payload = _apply_current_shutdown_truth(payload)
    access_profile = request_access_profile(request)
    payload = _runtime_revision_response_projection(
        payload,
        include_diagnostics=access_profile.get("surface") == "owner",
    )
    metadata = payload.get("health_read_model") or {}
    return JSONResponse(
        _json_safe(payload),
        headers={
            "Cache-Control": "no-store",
            "X-Aura-Health-Generation": str(metadata.get("snapshot_generation", 0)),
            "X-Aura-Health-Serving": str(metadata.get("serving", "unknown")),
        },
    )


@router.get("/tools/catalog")
async def api_tools_catalog():
    catalog = _collect_tool_catalog()
    health = _collect_skill_catalog_health()
    return JSONResponse({"tools": catalog, "count": len(catalog), "health": health})


@router.get("/ui/bootstrap")
async def api_ui_bootstrap(request: Request = None):
    _mark_runtime_service_progress("api.ui.bootstrap")
    _restore_owner_session_from_request(request)
    access_profile = request_access_profile(request)
    conversation_only = bool(access_profile.get("conversation_only", True))
    orch = ServiceContainer.get("orchestrator", default=None)
    rt = _get_runtime_state_safe()
    constitutional_status = {}
    executive_status = {}
    state_summary = {
        "current_objective": "",
        "pending_initiatives": 0,
        "active_goals": 0,
        "policy_mode": "unknown",
        "health": {},
        "rolling_summary": "",
        "coherence_score": 1.0,
        "fragmentation_score": 0.0,
        "contradiction_count": 0,
        "phenomenal_state": "",
        "thermal_guard": False,
        "health_flags": [],
        "epistemics": {},
    }

    try:
        from core.constitution import get_constitutional_core

        constitutional_core = get_constitutional_core(orch)
        constitutional_status = constitutional_core.get_status()
        state_summary = constitutional_core.snapshot()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap constitutional snapshot failed: %s", exc)

    try:
        executive_authority = ServiceContainer.get("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_status = executive_authority.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap executive snapshot failed: %s", exc)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap interaction signal snapshot failed: %s", exc)

    tool_catalog = _collect_tool_catalog()
    skill_catalog_health = _collect_skill_catalog_health()
    conversation_lane = _collect_conversation_lane_status_resilient()
    boot_snapshot, _status_code = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    status_obj = getattr(orch, "status", None)
    recent_conversation: list[dict[str, Any]] = []
    try:
        from interface.routes.chat import _conversation_log, _conversation_log_lock

        async with _conversation_log_lock:
            recent_conversation = list(_conversation_log)[-40:]
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Bootstrap conversation log snapshot failed: %s", exc)
    if conversation_only:
        session_id = paired_device_session_id(request)
        recent_conversation = [
            entry
            for entry in recent_conversation
            if session_id
            and str(entry.get("session_id") or "") == session_id
        ]

    static_dir = config.paths.project_root / "interface" / "static"
    shell_dist_dir = static_dir / "shell" / "dist"
    legacy_ui_index = static_dir / "index.html"

    legacy_ui_status = {
        "shell": "legacy_shell" if legacy_ui_index.exists() else "react_shell",
        "legacy_fallback_available": legacy_ui_index.exists(),
        "experimental_shell_available": (shell_dist_dir / "index.html").exists(),
        "experimental_shell_enabled": os.environ.get("AURA_ENABLE_REACT_SHELL", "").strip().lower()
        in {"1", "true", "yes", "on"},
    }
    legacy_ui_status["canonical_shell"] = (
        "legacy_shell"
        if legacy_ui_index.exists() and not legacy_ui_status["experimental_shell_enabled"]
        else "react_shell"
    )
    shell_status_helper = globals().get("_collect_legacy_shell_status")
    if callable(shell_status_helper):
        try:
            helper_payload = shell_status_helper() or {}
            if isinstance(helper_payload, dict):
                legacy_ui_status.update(helper_payload)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Bootstrap legacy shell status sync failed: %s", exc)

    try:
        bootstrap_cpu = psutil.cpu_percent(interval=None)
        bootstrap_ram = psutil.virtual_memory().percent
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Bootstrap telemetry resource sample failed: %s", exc)
        bootstrap_cpu = 0.0
        bootstrap_ram = 0.0

    payload = {
        "identity": {
            "name": "Aura Luna",
            "version": version_string("full"),
            "build": VERSION,
        },
        "session": {
            "connected": bool(
                boot_snapshot.get("system_ready", False)
                or (
                    boot_snapshot.get("ready", False)
                    and boot_snapshot.get("conversation_ready", False)
                )
            ),
            "initialized": bool(getattr(status_obj, "initialized", False)),
            "websocket_clients": ws_manager.count(),
            "is_gui_proxy": os.environ.get("AURA_GUI_PROXY") == "1",
        },
        "access": access_profile,
        "runtime_revision": _runtime_revision_fallback_contract(),
        "constitutional": constitutional_status,
        "executive": executive_status,
        "state": state_summary,
        "commitments": _collect_commitment_summary(),
        "tools": tool_catalog,
        "skill_catalog": skill_catalog_health,
        "capabilities": _collect_runtime_capabilities(conversation_lane),
        "desktop_access": await _collect_desktop_access_summary(allow_probe=False),
        "conversation": {
            "recent": recent_conversation,
            "count": len(recent_conversation),
            "lane": conversation_lane,
        },
        "voice": _collect_voice_summary(),
        "interaction_signals": interaction_signals_data,
        "telemetry": {
            "cpu_usage": bootstrap_cpu,
            "ram_usage": bootstrap_ram,
            "runtime": rt,
            "boot": boot_snapshot,
        },
        "diagnostics": {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        },
        "ui": {
            "shell": legacy_ui_status.get("shell", "legacy_shell" if legacy_ui_index.exists() else "react_shell"),
            "legacy_fallback_available": bool(legacy_ui_status.get("legacy_fallback_available", legacy_ui_index.exists())),
            "experimental_shell_available": bool(legacy_ui_status.get("experimental_shell_available", (shell_dist_dir / "index.html").exists())),
            "experimental_shell_enabled": bool(legacy_ui_status.get("experimental_shell_enabled", False)),
            "canonical_shell": legacy_ui_status.get("canonical_shell", legacy_ui_status.get("shell", "legacy_shell")),
            "status_flags": _derive_ui_status_flags(
                state_summary=state_summary,
                executive_status=executive_status,
                boot_snapshot=boot_snapshot,
                tool_catalog=tool_catalog,
                skill_catalog_health=skill_catalog_health,
            ),
        },
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    if conversation_only:
        lane = payload["conversation"].get("lane") or {}
        public_lane = {
            key: lane.get(key)
            for key in (
                "state",
                "conversation_ready",
                "active_generation",
                "active_generations",
            )
            if key in lane
        }
        boot = payload["telemetry"].get("boot") or {}
        public_boot = {
            key: boot.get(key)
            for key in (
                "ready",
                "status",
                "system_ready",
                "conversation_ready",
                "progress",
            )
            if key in boot
        }
        public_flags = [
            flag
            for flag in payload["ui"].get("status_flags", [])
            if flag == "booting"
        ]
        payload.update(
            {
                "session": {
                    "connected": bool(payload["session"].get("connected", False)),
                    "surface": "paired_device",
                },
                "constitutional": {},
                "executive": {},
                "state": {},
                "commitments": {},
                "tools": [],
                "capabilities": {"conversation": True, "world_read": True},
                "desktop_access": {
                    "available": False,
                    "overall_status": "surface_not_authorized",
                },
                "voice": {"available": False, "state": "surface_not_authorized"},
                "interaction_signals": {},
                "telemetry": {"runtime": {}, "boot": public_boot},
                "diagnostics": {},
                "ui": {"status_flags": public_flags},
            }
        )
        payload["conversation"] = {
            "recent": recent_conversation,
            "count": len(recent_conversation),
            "lane": public_lane,
        }
    payload = _runtime_revision_response_projection(
        payload,
        include_diagnostics=access_profile.get("surface") == "owner",
    )
    return JSONResponse(_json_safe(payload))


@router.post("/ui/shell-error")
async def api_ui_shell_error(payload: dict[str, Any] | None = _UI_SHELL_ERROR_BODY):
    """Record desktop shell render faults without blocking UI recovery."""
    safe_payload = _json_safe(payload if isinstance(payload, dict) else {})
    message = str(safe_payload.get("error") or "unknown shell render fault")[:500]
    logger.error("Aura desktop shell render fault: %s", message)
    try:
        await broadcast_bus.publish(
            {
                "kind": "log",
                "level": "error",
                "source": "Aura.Desktop.Shell",
                "message": f"Desktop shell render fault recovered: {message}",
                "payload": safe_payload,
                "event_ts": datetime.now(tz=UTC).isoformat(),
            },
            priority=0,
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Shell error broadcast failed: %s", exc)
    return JSONResponse({"ok": True})


@router.get("/health/boot")
async def api_boot_health(request: Request = None):
    _mark_runtime_service_progress("api.health.boot")
    payload, status_code = await _build_boot_health_payload_bounded(
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
    )
    access_profile = request_access_profile(request)
    payload = _runtime_revision_response_projection(
        payload,
        include_diagnostics=access_profile.get("surface") == "owner",
    )
    return JSONResponse(payload, status_code=status_code)


def _heartbeat_probe_blockers(required_probes: Any) -> list[str]:
    """Return blockers that make a readiness heartbeat unhealthy.

    A healthy heartbeat is a launch contract, not a process ping. It must have
    every required probe group and every group must report ok.
    """
    return required_probe_blockers(required_probes)


def _derive_api_health_status(
    *,
    healthy_ready: bool,
    service_ok: bool,
    lane_is_standby: bool,
    lane_state: str,
    conversation_ready: bool,
    conversation_busy: bool,
    boot_snapshot: dict[str, Any] | None = None,
) -> str:
    """Public ``/api/health`` status word.

    Every state below "ok" used to require ``service_ok`` — which is
    ``boot_snapshot["system_ready"]``, and that is False whenever ANY
    important-tier service is degraded. A runtime with a degraded important
    service therefore fell through every branch to "booting", no matter how
    long it had been up or how well it was answering. Measured live: 52 minutes
    of uptime, chat turns answering normally, top-level ``status: "booting"``,
    while the boot snapshot one layer down had already correctly concluded
    ``status="degraded" boot_phase="conversation_operational"``.

    The boot layer learned this exact lesson once already (its own note records
    "55 minutes of booting, 48%" on a fully conversational instance). This
    ladder never got the same fix, so it now defers to the snapshot that knows.
    """

    snapshot = boot_snapshot if isinstance(boot_snapshot, dict) else {}
    state = lane_state.strip().lower()

    if healthy_ready:
        return "ok"
    if service_ok:
        if lane_is_standby:
            return "standby"
        if state == "failed":
            return "unavailable"
        if state == "recovering":
            return "recovering"
        if conversation_busy:
            return "working"
        if not conversation_ready:
            return "warming"

    # Not fully ready. "booting" is only honest while this process has never
    # served — otherwise it is a degradation, and saying "booting" tells the
    # user to wait for something that already happened.
    #
    # `conversation_busy` is deliberately NOT evidence of having served: a COLD
    # boot is busy while it warms the lane, and counting that as degradation
    # made a genuine first boot report "degraded" instead of "booting" (caught
    # on a fresh start whose blockers still included critical:inference_gate).
    # A busy lane on a ready system is already answered by the "working" rung
    # above. The boot snapshot's own status/phase carries the has-ever-served
    # latch, so defer to it.
    if (
        conversation_ready
        or str(snapshot.get("boot_phase") or "").strip().lower()
        in {"conversation_operational", "runtime_degraded"}
        or str(snapshot.get("status") or "").strip().lower() == "degraded"
    ):
        return "degraded"
    return "booting"


def _normalize_conversation_health_blockers(
    blockers: list[Any],
    *,
    conversation_ready: bool,
    conversation_busy: bool = False,
) -> list[str]:
    """Merge health blockers without preserving stale conversation failures."""
    normalized = [
        str(item)
        for item in (blockers or [])
        if str(item or "").strip()
    ]
    if conversation_ready or conversation_busy:
        normalized = [
            item
            for item in normalized
            if item != "conversation_ready"
            and not item.startswith("conversation_lane:")
            and not item.startswith("conversation_reason:")
        ]
    elif "conversation_ready" not in normalized:
        normalized.append("conversation_ready")
    return list(dict.fromkeys(normalized))


def _collect_runtime_integrity_report() -> dict[str, Any]:
    """Return the non-blocking proof/learning integrity read model.

    This is intentionally separated from launch readiness. CRSM/CAA learning
    debt should not masquerade as a clean proof state, but it should also not
    make the desktop shell refuse to open when kernel/inference/memory/tool
    probes are otherwise safe.
    """
    try:
        from core.runtime.integrity_audit import read_integrity_audit

        report = read_integrity_audit()
        if isinstance(report, dict):
            return report
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime integrity audit unavailable: %s", exc)
    return {
        "healthy": False,
        "concerns": ["integrity_audit_unavailable"],
        "strict_mode": False,
        "degradations": {},
        "crsm_loop": {},
        "caa_readiness": {},
        "at": time.time(),
    }


def _runtime_integrity_blockers(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return ["integrity_audit_unavailable"]
    concerns = [
        str(item).strip()
        for item in (report.get("concerns") or [])
        if str(item or "").strip()
    ]
    if bool(report.get("healthy", False)) and not concerns:
        return []
    return [f"integrity:{concern}" for concern in (concerns or ["integrity_unknown"])]


def _runtime_integrity_proof_blockers(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return ["integrity:integrity_audit_unavailable"]
    blockers = list(_runtime_integrity_blockers(report))
    advisory = [
        str(item).strip()
        for item in (report.get("advisory") or [])
        if str(item or "").strip()
    ]
    blockers.extend(f"integrity:{item}" for item in advisory)
    read_model = report.get("integrity_read_model")
    if isinstance(read_model, dict) and bool(read_model.get("expired", False)):
        blockers.append(
            "integrity:integrity_snapshot_initializing"
            if read_model.get("captured_at_unix") is None
            else "integrity:integrity_snapshot_expired"
        )
    return list(dict.fromkeys(blockers))


def _runtime_integrity_public_payload(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report if isinstance(report, dict) else {}
    blockers = _runtime_integrity_blockers(report)
    proof_blockers = _runtime_integrity_proof_blockers(report)
    return {
        "healthy": not blockers,
        "status": "healthy" if not blockers else "degraded",
        "concerns": [
            str(item)
            for item in (report.get("concerns") or [])
            if str(item or "").strip()
        ],
        "advisory": [
            str(item)
            for item in (report.get("advisory") or [])
            if str(item or "").strip()
        ],
        "blockers": blockers,
        "proof_blockers": proof_blockers,
        "proof_readiness": not proof_blockers,
        "operational_blocking": bool(report.get("strict_mode", False)) and bool(blockers),
        "crsm_loop": report.get("crsm_loop") or {},
        "caa_readiness": report.get("caa_readiness") or {},
        "at": report.get("at"),
        "read_model": report.get("integrity_read_model") or {},
    }


@router.get("/health/mind_tick")
async def api_mind_tick_diagnostics():
    """Diagnostic: MindTick's internal liveness state — is the supervised loop
    running, tick_count, last successful/progress timestamps + their ages, the
    active tick stage, consecutive failures, liveness-repair count. Read-only.

    Built 2026-07-07 to pin the false-death → launcher-respawn loop: is_alive()
    flipping False at exactly 180s means the boot-grace branch fired, i.e. BOTH
    progress timestamps are still 0 — the loop body never marked progress. This
    surfaces whether the loop is running at all and where it is stuck.
    """
    import time as _t

    mt = ServiceContainer.get("mind_tick", default=None)
    if mt is None or not hasattr(mt, "get_health_status"):
        return JSONResponse({"error": "mind_tick unavailable"}, status_code=503)
    try:
        status = dict(mt.get_health_status())
    except _SYSTEM_RECOVERABLE_ERRORS as exc:  # diagnostic must never itself 500 the health lane
        return JSONResponse({"error": f"get_health_status failed: {exc}"}, status_code=200)
    now = _t.time()
    for key in ("last_successful_tick_at", "last_loop_progress_at", "active_tick_started_at"):
        value = float(status.get(key) or 0.0)
        status[key + "_age_s"] = round(now - value, 1) if value > 0 else None
    return JSONResponse(_json_safe(status) if "_json_safe" in globals() else status)


@router.get("/health/heartbeat")
async def api_heartbeat():
    """Readiness heartbeat for GUI/runtime watchdogs.

    This is intentionally not a process-only ping. It may report healthy only
    when the kernel, inference, memory, scheduler, and tool-governance probes
    pass through the canonical boot health contract.
    """
    _mark_runtime_service_progress("api.health.heartbeat")
    payload, status_code = await _build_boot_health_payload_bounded(
        is_gui_proxy=False,
    )
    conversation_lane = _collect_conversation_lane_status_resilient()
    conversation_ready = bool(conversation_lane.get("conversation_ready", False))
    conversation_busy = conversation_lane_is_busy(conversation_lane)
    required_probes = payload.get("required_probes", {})
    probe_blockers = _heartbeat_probe_blockers(required_probes)
    runtime_revision = payload.get("runtime_revision")
    if not isinstance(runtime_revision, dict):
        runtime_revision = _runtime_revision_fallback_contract()
    revision_blocker = _runtime_revision_blocker(runtime_revision)
    integrity_report = _collect_runtime_integrity_report()
    integrity_payload = _runtime_integrity_public_payload(integrity_report)
    proof_readiness_healthy = bool(
        integrity_payload.get("proof_readiness", False)
        and not revision_blocker
    )
    blockers = _normalize_conversation_health_blockers(
        list(payload.get("blockers", []) or [])
        + probe_blockers
        + ([revision_blocker] if revision_blocker else []),
        conversation_ready=conversation_ready,
        conversation_busy=conversation_busy,
    )
    runtime_probe_healthy = not probe_blockers
    healthy = (
        status_code in {200, 202}
        and bool(payload.get("system_ready", payload.get("ready", False)))
        and runtime_probe_healthy
        and conversation_ready
        and not blockers
    )
    if not healthy and not (runtime_probe_healthy and conversation_busy and not blockers):
        status_code = 503
    status = "healthy" if healthy else "working" if runtime_probe_healthy and conversation_busy else "unhealthy"
    heartbeat_payload = {
        "status": status,
        "healthy": healthy,
        "runtime_probe_healthy": runtime_probe_healthy,
        "time": time.time(),
        "required_probes": required_probes,
        "blockers": blockers,
        "boot_phase": payload.get("boot_phase"),
        "conversation_ready": conversation_ready,
        "conversation_busy": conversation_busy,
        "conversation_lane": conversation_lane,
        "integrity": integrity_payload,
        "proof_readiness_healthy": proof_readiness_healthy,
        "certification_ready": bool(healthy and proof_readiness_healthy),
        "integrity_blockers": integrity_payload.get("proof_blockers", []),
        "runtime_revision": _runtime_revision_response_projection(
            {"runtime_revision": runtime_revision},
            include_diagnostics=False,
        ).get("runtime_revision"),
    }
    return JSONResponse(heartbeat_payload, status_code=status_code)


# ── Hot Reload ────────────────────────────────────────────────

@router.post("/system/hot-reload", tags=["system"])
async def api_hot_reload(request: Request):
    """Reload Aura's cognitive modules without restarting the process.

    Query params:
        scope  – reload scope (phases, skills, consciousness, llm, affect,
                 memory, identity, resilience, orchestrator_mixins, learning,
                 agency, all). Defaults to "all", which is a curated live-safe
                 union rather than every loaded core module.
        file   – reload a single file by path (relative to project root).

    The kernel, ServiceContainer, event loop, loaded models, and
    conversation history are preserved.
    """
    _require_internal(request)

    try:
        from core.ops.hot_reload import get_hot_reloader

        reloader = get_hot_reloader()
        if ServiceContainer.get("hot_reloader", default=None) is None:
            ServiceContainer.register_instance("hot_reloader", reloader)

        filepath = request.query_params.get("file")
        scope = request.query_params.get("scope", "all")
        if filepath:
            result = await asyncio.to_thread(reloader.reload_file, filepath)
        else:
            result = await asyncio.to_thread(reloader.reload_scope, scope)

        status_code = 200 if result.ok else 207  # 207 Multi-Status for partial failure
        return JSONResponse(result.to_dict(), status_code=status_code)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.error("Hot reload failed: %s", exc, exc_info=True)
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=500,
        )


@router.post("/system/browser-pursuit", tags=["system"])
async def api_browser_pursuit(request: Request):
    """Run one browser pursuit directly and return every round of it.

    Diagnosing this through chat cost six minutes a cycle — restart, type,
    wait, read a one-line failure — and the failures were reported in the
    vocabulary of whichever layer noticed them, never in terms of what the
    loop actually did. Four separate causes were found that way, each taking
    several cycles, and each one hid the next.

    This runs the same governed pursuit the desktop lane delegates to, with no
    chat turn around it, and returns the trace: what she understood the page
    to be, what she chose each round, why, what she expected, and whether the
    page moved. Owner-only, and it changes nothing about how the capability
    behaves in a real turn.
    """
    _require_internal(request)

    goal = str(request.query_params.get("goal") or "").strip()
    url = str(request.query_params.get("url") or "").strip()
    try:
        max_steps = int(request.query_params.get("max_steps") or 6)
    except (TypeError, ValueError):
        max_steps = 6
    if not goal or not url:
        return JSONResponse({"ok": False, "error": "goal and url are required"}, status_code=400)

    try:
        # Through `execute`, not `_handle_pursue`.
        #
        # Calling the loop directly skips the transaction that mints the
        # browser lease, so every click came back
        # `browser_interaction_authority_unavailable` — the governance working,
        # and a probe that exercises a path the real capability never takes is
        # worth very little.
        # Through the capability engine, which is the only governed way in.
        #
        # Calling the skill object directly is refused outright — "Ungoverned
        # skill execution blocked: skill:sovereign_browser called outside
        # governed context" — and rightly: the engine is where authority,
        # receipts and the lease come from. The probe takes the same route the
        # desktop lane takes, so what it exercises is what really runs.
        capability_engine = ServiceContainer.get("capability_engine", default=None)
        if capability_engine is None or not hasattr(capability_engine, "execute"):
            return JSONResponse(
                {"ok": False, "error": "capability_engine_unavailable"}, status_code=503
            )
        report = await capability_engine.execute(
            "sovereign_browser",
            {"mode": "pursue", "url": url, "goal": goal, "max_steps": max_steps},
            context={
                # A real foreground origin, because the probe is exercising
                # the foreground capability. "system.browser_pursuit_probe"
                # coerces to nothing known, carries no user authority, and the
                # will refuses the action — which tests the probe's label
                # rather than the capability.
                "source": "desktop_ui",
                "origin": "desktop_ui",
                "user_explicitly_authorized": True,
                "user_requested_action": True,
            },
        )
        return JSONResponse(report if isinstance(report, dict) else {"ok": False})
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.browser_pursuit", exc)
        logger.error("Browser pursuit probe failed: %s", exc, exc_info=True)
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)


@router.get("/self/measured", tags=["system"])
async def api_self_measured(request: Request):
    """Exactly what she is carrying about herself, right now.

    Read-only, owner-only, and it exists because of a specific blindness. On
    2026-08-10 three separate defects were about a self-report disagreeing with
    an instrument — "Your RAM pressure is currently 37%" at 0.717, "I feel
    energized" at energy 0.058, a thirteen-line vitals panel that was almost
    entirely invented. Each took a long forensic detour, because the only way
    to see what she had actually been handed was to ask her and infer backwards
    from the answer. The runtime knew; nothing exposed it.

    Two surfaces, together, because the whole class of defect lives in the gap
    between them: the one-line self-knowledge string that rides every turn, and
    the typed self-condition projection with its supported and missing
    dimensions. A dimension that is missing HERE is a dimension she will answer
    from a language model's beliefs about AIs instead of from her own body.
    """
    _require_internal(request)

    payload: dict[str, Any] = {}
    try:
        from core.self.capability_ledger import (
            get_capability_ledger,
            measured_self_metrics,
            self_knowledge_line,
        )

        payload["self_knowledge_line"] = self_knowledge_line()
        payload["measured_metrics"] = measured_self_metrics()
        payload["capabilities"] = {
            name: availability.as_dict()
            for name, availability in get_capability_ledger().measure_all().items()
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.self_measured", exc)
        payload["capability_error"] = str(exc)

    try:
        from core.self.self_condition import build_self_condition_projection

        projection = build_self_condition_projection()
        payload["self_condition"] = projection.to_dict()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.self_measured", exc)
        payload["self_condition_error"] = str(exc)

    return JSONResponse(payload)


@router.get("/system/source-drift", tags=["system"])
async def api_source_drift(request: Request):
    """Which loaded modules no longer match their file on disk.

    `SourceBodyAwareness` watches the GIT DIRTY STATE, so committing an edit
    leaves the tree clean while this process stays exactly as stale. Asked "are
    you running my latest fix?", nothing could answer. This reads the bytecode
    cache header — the interpreter's own record of what it compiled — and
    confirms every timestamp suspicion against the compiled code objects, so a
    touch, a `git checkout` that restores identical bytes, and a whitespace-only
    edit all correctly read clean.
    """
    _require_internal(request)

    try:
        from core.runtime.loaded_source_drift import scan_drift

        report = scan_drift()
        payload = report.to_dict()
        payload["narrative"] = report.narrative()
        return JSONResponse(payload)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.source_drift", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.get("/system/hot-reload/status", tags=["system"])
async def api_hot_reload_status(request: Request):
    """Return the current state of the hot-reload engine."""
    _require_internal(request)

    try:
        from core.ops.hot_reload import get_hot_reloader

        reloader = get_hot_reloader()
        return JSONResponse(reloader.get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.error("Hot reload status failed: %s", exc, exc_info=True)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.get("/system/hot-reload/scopes", tags=["system"])
async def api_hot_reload_scopes(request: Request):
    """List all available reload scopes and their module prefixes."""
    _require_internal(request)

    try:
        from core.ops.hot_reload import PROTECTED_MODULES, PROTECTED_PREFIXES, RELOAD_SCOPES

        return JSONResponse({
            "scopes": {
                name: {"prefixes": prefixes}
                for name, prefixes in RELOAD_SCOPES.items()
            },
            "special_scopes": ["all"],
            "special_scope_details": {
                "all": "Curated live-safe union of reload scopes; excludes runtime-owned infrastructure that requires reboot."
            },
            "protected_modules": sorted(PROTECTED_MODULES),
            "protected_prefixes": sorted(PROTECTED_PREFIXES),
        })
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.error("Hot reload scope listing failed: %s", exc, exc_info=True)
        return JSONResponse({"ok": False, "error": str(exc), "scopes": {}}, status_code=500)
