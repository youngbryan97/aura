"""interface/routes/system.py
─────────────────────────────
Extracted from server.py — Health, telemetry, metrics, bootstrap,
and all collector/diagnostic helpers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import psutil
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from fastapi.responses import ORJSONResponse
except Exception:
    ORJSONResponse = JSONResponse

from core.config import config
from core.container import ServiceContainer
from core.health.boot_status import build_boot_health_snapshot
from core.runtime_tools import get_runtime_state
from core.scheduler import scheduler
from core.version import VERSION, version_string

from interface.auth import _require_internal, _check_rate_limit, _restore_owner_session_from_request
from interface.websocket_manager import broadcast_bus, ws_manager

logger = logging.getLogger("Aura.Server.System")

router = APIRouter()


# ── Collector Helpers ─────────────────────────────────────────

def _collect_recent_degraded_events(limit: int = 12) -> List[Dict[str, Any]]:
    try:
        from core.health.degraded_events import get_recent_degraded_events

        return get_recent_degraded_events(limit=limit)
    except Exception as exc:
        logger.debug("Recent degraded event collection failed: %s", exc)
        return []


def _collect_conversation_lane_status() -> Dict[str, Any]:
    """Import and delegate to the canonical implementation in chat routes."""
    from interface.routes.chat import _collect_conversation_lane_status as _impl
    return _impl()


def _conversation_lane_is_standby(lane: Optional[Dict[str, Any]]) -> bool:
    from interface.routes.chat import _conversation_lane_is_standby as _impl
    return _impl(lane)


def _conversation_lane_user_message(lane: Dict[str, Any], **kwargs) -> str:
    from interface.routes.chat import _conversation_lane_user_message as _impl
    return _impl(lane, **kwargs)


def _collect_stability_details() -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "status": "unknown",
        "healthy": True,
        "active_issues": [],
    }
    try:
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian and hasattr(guardian, "get_latest_report"):
            report = guardian.get_latest_report() or {}
            checks = report.get("checks", []) if isinstance(report, dict) else []
            active_issues = []
            for check in checks:
                if not bool(check.get("healthy", True)):
                    active_issues.append(
                        {
                            "name": check.get("name", "unknown"),
                            "message": check.get("message", ""),
                            "severity": check.get("severity", "warning"),
                            "action_taken": check.get("action_taken"),
                        }
                    )
            details["healthy"] = bool(report.get("overall_healthy", True))
            details["status"] = "healthy" if details["healthy"] else "degraded"
            details["active_issues"] = active_issues
            details["memory_pct"] = report.get("memory_pct")
            details["cpu_pct"] = report.get("cpu_pct")
    except Exception as exc:
        logger.debug("Stability detail collection failed: %s", exc)

    try:
        lane = _collect_conversation_lane_status()
        lane_is_standby = _conversation_lane_is_standby(lane) if isinstance(lane, dict) else False
        if isinstance(lane, dict) and not bool(lane.get("conversation_ready", False)) and not lane_is_standby:
            details["healthy"] = False
            if details.get("status") == "unknown":
                details["status"] = "degraded"
            details.setdefault("active_issues", []).append(
                {
                    "name": "conversation_lane",
                    "message": _conversation_lane_user_message(lane),
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
    except Exception as exc:
        logger.debug("Conversation lane stability detail merge failed: %s", exc)
    if details.get("status") == "unknown":
        details["status"] = "healthy" if bool(details.get("healthy", True)) else "degraded"
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
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(coerced) or math.isinf(coerced):
        return None
    return coerced


def _collect_liquid_state_payload(
    ls_data: Dict[str, Any],
    *,
    runtime_state: Dict[str, Any],
    homeostasis_data: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_affect = runtime_state.get("affect", {}) if isinstance(runtime_state.get("affect"), dict) else {}
    payload: Dict[str, Any] = {}

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
            runtime_fallback = runtime_affect.get("confidence", homeostasis_data.get("will_to_live"))
        normalized = _pick_metric(key, runtime_fallback=runtime_fallback)
        if normalized is not None:
            payload[key] = round(normalized, 1)

    if "confidence" not in payload:
        normalized = _normalize_percentish(homeostasis_data.get("will_to_live"))
        if normalized is not None:
            payload["confidence"] = round(normalized, 1)

    if ls_data.get("mood") is not None:
        payload["mood"] = ls_data.get("mood")
    elif runtime_affect.get("mood") is not None:
        payload["mood"] = runtime_affect.get("mood")

    if isinstance(ls_data.get("vad"), dict):
        payload["vad"] = ls_data["vad"]

    return payload


async def _collect_soma_payload() -> Dict[str, Any]:
    def _system_fallback() -> Dict[str, Any]:
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None) or 0.0) / 100.0
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            ram_pct = float(getattr(ram, "percent", 0.0) or 0.0) / 100.0
            disk_pct = float(getattr(disk, "percent", 0.0) or 0.0) / 100.0
            vitality = max(0.0, 1.0 - (max(cpu_pct, ram_pct, disk_pct) * 0.2))
            return {
                "thermal_load": cpu_pct,
                "resource_anxiety": ram_pct,
                "vitality": vitality,
            }
        except Exception as exc:
            logger.debug("Soma fallback telemetry failed: %s", exc)
            return {}

    soma = ServiceContainer.get("soma", default=None)
    if not soma:
        return _system_fallback()

    if hasattr(soma, "pulse"):
        try:
            await asyncio.wait_for(soma.pulse(), timeout=0.25)
        except Exception as exc:
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
    except Exception as exc:
        logger.debug("Soma status collection failed: %s", exc)
    return _system_fallback()


def _collect_tool_catalog() -> List[Dict[str, Any]]:
    engine = ServiceContainer.get("capability_engine", default=None)
    if engine and hasattr(engine, "get_tool_catalog"):
        try:
            return list(engine.get_tool_catalog(include_inactive=True))
        except Exception as exc:
            logger.debug("Tool catalog collection failed: %s", exc)
    return []


def _collect_commitment_summary() -> Dict[str, Any]:
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
    except Exception as exc:
        logger.debug("Commitment summary collection failed: %s", exc)
        return {"active_count": 0, "reliability_score": 1.0, "active": []}


def _collect_voice_summary() -> Dict[str, Any]:
    from interface.routes.privacy import get_voice_engine_fn
    _voice_engine_fn = get_voice_engine_fn()
    voice_available = bool(_voice_engine_fn)
    summary = {
        "available": voice_available,
        "microphone_enabled": True,
        "speaking_enabled": True,
        "streaming_available": True,
        "state": "ready" if voice_available else "unavailable",
    }
    try:
        voice = _voice_engine_fn() if _voice_engine_fn else None
        if voice is not None:
            microphone_enabled = bool(getattr(voice, "microphone_enabled", True))
            speaking_enabled = bool(getattr(voice, "speaking_enabled", True))
            summary["microphone_enabled"] = microphone_enabled
            summary["speaking_enabled"] = speaking_enabled
            if not microphone_enabled and not speaking_enabled:
                summary["state"] = "muted"
            else:
                voice_state = getattr(getattr(voice, "state", None), "name", "") or ""
                if voice_state:
                    summary["state"] = str(voice_state).lower()
                else:
                    summary["state"] = "listening" if getattr(voice, "is_listening", False) else "ready"
    except Exception as exc:
        logger.debug("Voice summary collection failed: %s", exc)
    return summary


async def _collect_desktop_access_summary() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "screen_recording": {"granted": False, "status": "unknown", "guidance": ""},
        "accessibility": {"granted": False, "status": "unknown", "guidance": ""},
        "automation": {"granted": False, "status": "unknown", "guidance": ""},
        "screen_capture_ready": False,
        "desktop_control_ready": False,
        "screen_text_ready": False,
        "menu_clock_ready": False,
        "menu_clock_text": "",
        "menu_clock_error": "",
        "frontmost_app": "",
        "pyautogui_ready": False,
        "pyautogui_error": "",
    }
    try:
        from core.security.permission_guard import PermissionGuard, PermissionType
        from core.skills._pyautogui_runtime import get_pyautogui

        guard = ServiceContainer.get("permission_guard", default=None) or PermissionGuard()
        if guard:
            screen = await guard.check_permission(PermissionType.SCREEN, force=True)
            accessibility = await guard.check_permission(PermissionType.ACCESSIBILITY, force=True)
            automation = await guard.check_permission(PermissionType.AUTOMATION, force=True)
            payload["screen_recording"] = screen
            payload["accessibility"] = accessibility
            payload["automation"] = automation
            payload["frontmost_app"] = str(automation.get("detail", "") or "")

        pyautogui, pyautogui_error = get_pyautogui()
        payload["pyautogui_ready"] = pyautogui is not None
        if pyautogui_error:
            payload["pyautogui_error"] = str(pyautogui_error)[:240]

        screen_granted = bool((payload["screen_recording"] or {}).get("granted"))
        accessibility_granted = bool((payload["accessibility"] or {}).get("granted"))
        automation_granted = bool((payload["automation"] or {}).get("granted"))
        payload["screen_capture_ready"] = screen_granted
        payload["desktop_control_ready"] = accessibility_granted and bool(payload["pyautogui_ready"])
        payload["screen_text_ready"] = automation_granted and accessibility_granted
        payload["menu_clock_ready"] = automation_granted and accessibility_granted
        if payload["menu_clock_ready"]:
            from core.skills.computer_use import ComputerUseSkill

            def _probe_menu_clock() -> Dict[str, Any]:
                skill = ComputerUseSkill()
                try:
                    text = skill._read_menu_clock_macos()
                    return {"ready": True, "text": text[:240]}
                except Exception as exc:
                    return {"ready": False, "error": str(exc)[:240]}

            menu_clock_probe = await asyncio.to_thread(_probe_menu_clock)
            payload["menu_clock_ready"] = bool(menu_clock_probe.get("ready"))
            payload["menu_clock_text"] = str(menu_clock_probe.get("text", "") or "")
            payload["menu_clock_error"] = str(menu_clock_probe.get("error", "") or "")
        primary_ready = [
            payload["screen_capture_ready"],
            payload["desktop_control_ready"],
            payload["screen_text_ready"],
        ]
        payload["blocking_permissions"] = [
            name for name, granted in (
                ("screen_recording", screen_granted),
                ("accessibility", accessibility_granted),
                ("automation", automation_granted),
            ) if not granted
        ]
        payload["overall_status"] = (
            "ready"
            if all(primary_ready) else
            "partial"
            if any(primary_ready) or any(
                bool((payload[key] or {}).get("granted"))
                for key in ("screen_recording", "accessibility", "automation")
            ) else
            "blocked"
        )
    except Exception as exc:
        logger.debug("Desktop access summary collection failed: %s", exc)
    return payload


def _collect_runtime_capabilities(conversation_lane: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    lane = conversation_lane if isinstance(conversation_lane, dict) else _collect_conversation_lane_status()
    payload: Dict[str, Any] = {
        "local_backend": "unknown",
        "local_runtime": "offline",
        "conversation_model": str(lane.get("desired_model", "") or ""),
        "conversation_endpoint": str(lane.get("desired_endpoint", "") or ""),
        "conversation_state": str(lane.get("state", "") or ""),
        "conversation_ready": bool(lane.get("conversation_ready", False)),
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
    except Exception as exc:
        logger.debug("Runtime capability backend lookup failed: %s", exc)

    state = str(payload.get("conversation_state", "") or "").lower()
    if bool(payload.get("conversation_ready")):
        payload["local_runtime"] = "online"
    elif _conversation_lane_is_standby(lane):
        payload["local_runtime"] = "standby"
    elif state in {"cold", "warming", "spawning", "handshaking", "recovering", "ready"}:
        payload["local_runtime"] = "warming"
    elif state == "failed":
        payload["local_runtime"] = "degraded"
    return payload


def _derive_ui_status_flags(
    *,
    state_summary: Dict[str, Any],
    executive_status: Dict[str, Any],
    boot_snapshot: Dict[str, Any],
    tool_catalog: List[Dict[str, Any]],
) -> List[str]:
    flags: List[str] = []
    if not bool(boot_snapshot.get("ready", False)):
        flags.append("booting")
    if bool(state_summary.get("thermal_guard")):
        flags.append("thermal_guard")
    if float(state_summary.get("coherence_score", 1.0) or 1.0) < 0.72:
        flags.append("coherence_low")
    if float(state_summary.get("fragmentation_score", 0.0) or 0.0) > 0.4:
        flags.append("fragmentation_high")
    if int(state_summary.get("contradiction_count", 0) or 0) > 3:
        flags.append("contradictions_present")
    epistemics = state_summary.get("epistemics", {}) or {}
    if int(epistemics.get("contested", 0) or 0) > 0:
        flags.append("beliefs_contested")
    unavailable_count = sum(1 for tool in tool_catalog if not bool(tool.get("available")))
    if unavailable_count >= 3:
        flags.append("tool_unavailable")
    if str(executive_status.get("last_target") or "").strip().lower() == "secondary":
        flags.append("executive_hold")
    return flags


# ── Routes ────────────────────────────────────────────────────

@router.get("/telemetry/stream")
async def telemetry_stream(request: Request):
    """Server-Sent Events stream for HUD telemetry."""
    from interface.auth import _require_internal
    _require_internal(request)

    async def event_generator():
        init_data = json.dumps({"type": "telemetry", "cpu_usage": psutil.cpu_percent(), "memory_usage": psutil.virtual_memory().percent})
        yield f"event: telemetry\ndata: {init_data}\n\n"

        try:
            q = await broadcast_bus.subscribe()
            while True:
                if await request.is_disconnected():
                    break

                while q.qsize() > 100:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        break

                try:
                    _priority, _ts, msg = await q.get()
                    msg_type = msg.get("type", "message")
                    data = json.dumps(msg)
                    yield f"event: {msg_type}\ndata: {data}\n\n"
                    q.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("SSE generate error: %s", e)
                    await asyncio.sleep(0.1)
                    continue
        finally:
            if 'q' in locals() and q:
                await broadcast_bus.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/metrics", tags=["metrics"])
async def metrics(request: Request):
    """System metrics for monitoring."""
    from interface.auth import _require_internal
    _require_internal(request)
    try:
        orch = ServiceContainer.get("orchestrator", default=None)
        orch_status = orch.get_status() if orch else {}

        return {
            "status": "healthy",
            "uptime": time.time() - (orch_status.get("start_time", time.time()) if orch_status else time.time()),
            "active_connections": ws_manager.count(),
            "cycle_count": orch_status.get("cycle_count", 0),
            "cpu_usage": float(int(psutil.cpu_percent() * 10)) / 10.0 if 'psutil' in sys.modules else 0,
            "memory_usage": float(int(psutil.virtual_memory().percent * 10)) / 10.0 if 'psutil' in sys.modules else 0,
        }
    except Exception as e:
        logger.error("Metrics collection failed: %s", e, exc_info=True)
        return ORJSONResponse({"status": "error", "message": "Metrics collection failed"}, status_code=500)


@router.get("/gemini-usage")
async def gemini_usage(request: Request):
    """Return daily Gemini API usage stats."""
    from interface.auth import _require_internal
    _require_internal(request)
    try:
        from core.brain.llm.gemini_adapter import DailyRateLimiter
        orch = ServiceContainer.get("orchestrator", default=None)
        if orch and hasattr(orch, 'cognitive_engine'):
            brain = getattr(orch.cognitive_engine, 'brain', None) or getattr(orch.cognitive_engine, '_brain', None)
            if brain and hasattr(brain, 'llm_router'):
                for name, adapter in brain.llm_router.adapters.items():
                    if hasattr(adapter, 'rate_limiter'):
                        return JSONResponse(adapter.rate_limiter.get_usage())
        from core.config import config
        state_path = str(config.paths.data_dir / "gemini_rate_state.json")
        limiter = DailyRateLimiter(state_path=state_path)
        return JSONResponse(limiter.get_usage())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/health")
async def api_health(request: Request):
    from interface.routes.privacy import get_voice_engine_fn
    _voice_engine_fn = get_voice_engine_fn()
    
    _restore_owner_session_from_request(request)
    orch       = ServiceContainer.get("orchestrator", default=None)
    rt         = get_runtime_state()
    runtime_payload = rt.get("state", {}) if isinstance(rt.get("state"), dict) else {}
    status_obj = getattr(orch, "status", None)

    initialized = getattr(status_obj, "initialized", False)
    connected   = orch is not None and getattr(status_obj, "running", False)

    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        p_core = per_cpu[0] if len(per_cpu) > 1 else cpu
    except Exception as e:
        logger.debug("Hardware stats collection failed: %s", e)
        cpu, ram, p_core = 0, 0, 0

    orch_status = {}
    if orch and hasattr(orch, "get_status"):
        try:
            orch_status = orch.get_status()
        except Exception as e:
            logger.debug("get_status failed: %s", e)
    conversation_lane = _collect_conversation_lane_status()
    boot_snapshot, _ = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    connected = bool(boot_snapshot.get("system_ready", False))

    ls_data = {}
    try:
        ls = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get("liquid_state", default=None)
        if ls and hasattr(ls, "get_status"):
            ls_data = ls.get_status()

        vad_data = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "_stale": True}
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if engine and hasattr(engine, "consciousness"):
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
    except Exception as e:
        logger.debug("Liquid state/VAD lookup failed: %s", e)
    curiosity_status = orch_status.get("curiosity_status", {})

    transcendence_data = {"meta_evolution": {"active": False, "acceleration_factor": 1.0}}
    try:
        meta = ServiceContainer.get("meta_cognition", default=None)
        if meta:
            transcendence_data["meta_evolution"] = meta.get_health()
            transcendence_data["meta_evolution"]["active"] = True
    except Exception as e:
        logger.debug("Transcendence status collection failed: %s", e)

    # Agency: derive from energy + curiosity + active autonomous thought
    _energy_raw = float(ls_data.get("energy", 0))
    _curiosity_raw = float(ls_data.get("curiosity", 0))
    _thinking = bool(orch and getattr(orch, "_current_thought_task", None)
                     and not orch._current_thought_task.done())
    _agency_score = (_energy_raw * 0.4 + _curiosity_raw * 0.4 + (30.0 if _thinking else 0.0))
    _agency_score = min(100.0, max(0.0, _agency_score))

    cortex = {
        "agency":    float(int(_agency_score * 10)) / 10.0,
        "curiosity": float(int(float(ls_data.get("curiosity", 0)) * 10)) / 10.0,
        "fixes":     orch_status.get("stats", {}).get("modifications_made", 0),
        "beliefs":   0,
        "episodes":  0,
        "active_topic": curiosity_status.get("active_topic", "None"),
        "goals":     orch_status.get("stats", {}).get("goals_processed", 0),
        "autonomy":  config.security.aura_full_autonomy,
        "stealth":   config.security.enable_stealth_mode,
        "scratchpad": ServiceContainer.get("scratchpad_engine", default=None) is not None,
        "forge":      ServiceContainer.get("hephaestus_engine", default=None) is not None,
        "subconscious": "dreaming" if getattr(orch, "boredom", 0) > 45 else "idle",
        "unity":      ServiceContainer.get("soma", default=None) is not None,
        "p_core_usage": float(int(p_core * 10)) / 10.0,
        "singularity_factor": float(int(transcendence_data.get("meta_evolution", {}).get("acceleration_factor", 1.0) * 100)) / 100.0,
        "meta_loop_active": transcendence_data.get("meta_evolution", {}).get("active", False)
    }

    if config.security.force_unity_on:
        cortex["unity"] = True
    try:
        if orch and hasattr(orch, "self_model") and orch.self_model:
            cortex["beliefs"] = len(getattr(orch.self_model, "beliefs", []))

        ep_mem = ServiceContainer.get("episodic_memory", default=None)
        if ep_mem and hasattr(ep_mem, "get_summary"):
            ep_summary = ep_mem.get_summary()
            cortex["episodes"] = ep_summary.get("total_episodes", 0)
        else:
            mem_mgr = ServiceContainer.get("memory_manager", default=None)
            if mem_mgr and hasattr(mem_mgr, "get_stats"):
                mem_stats = mem_mgr.get_stats()
                cortex["episodes"] = mem_stats.get("episodic_count", 0)
    except Exception as e:
        logger.debug("Cortex supplementary metrics failed: %s", e)

    moral = ServiceContainer.get("moral", default=None)
    moral_data = moral.get_health() if moral and hasattr(moral, "get_health") else {}

    homeostasis = ServiceContainer.get("homeostasis", default=None)
    homeo_data = homeostasis.get_health() if homeostasis else {}
    liquid_state_payload = _collect_liquid_state_payload(
        cast(Dict[str, Any], ls_data if isinstance(ls_data, dict) else {}),
        runtime_state=runtime_payload if isinstance(runtime_payload, dict) else {},
        homeostasis_data=homeo_data if isinstance(homeo_data, dict) else {},
    )
    soma_data = await _collect_soma_payload()

    social = ServiceContainer.get("social", default=None)
    social_data = social.get_health() if social else {"depth": 0.0}

    swarm_data = orch.swarm_status if orch and hasattr(orch, 'swarm_status') else {"active_count": 0}

    executive_closure_data = {}
    try:
        executive_closure_data = orch_status.get("executive_closure", {}) or {}
        if not executive_closure_data:
            executive_closure = ServiceContainer.get("executive_closure", default=None)
            if executive_closure and hasattr(executive_closure, "get_status"):
                executive_closure_data = executive_closure.get_status()
    except Exception as e:
        logger.debug("Executive closure status collection failed: %s", e)

    consciousness_evidence = {}
    try:
        consciousness_evidence = orch_status.get("consciousness_evidence", {}) or {}
        if not consciousness_evidence:
            evidence = ServiceContainer.get("consciousness_evidence", default=None)
            if evidence and hasattr(evidence, "snapshot"):
                consciousness_evidence = evidence.snapshot()
    except Exception as e:
        logger.debug("Consciousness evidence collection failed: %s", e)

    executive_authority_data = {}
    try:
        executive_authority = ServiceContainer.get("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_authority_data = executive_authority.get_status()
    except Exception as e:
        logger.debug("Executive authority status collection failed: %s", e)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except Exception as e:
        logger.debug("Interaction signal status collection failed: %s", e)

    # ── Resilience Status ──
    resilience_data: Dict[str, Any] = {"circuit_breakers": {}, "snapshot": "unknown", "llm_tier": "unknown"}
    try:
        voice = ServiceContainer.get("voice_pipeline", default=None)
        if voice:
            for attr_name in ("_stt_breaker", "_tts_breaker"):
                breaker = getattr(voice, attr_name, None)
                if breaker and hasattr(breaker, "state"):
                    cast(Dict[str, Any], resilience_data["circuit_breakers"])[breaker.name] = breaker.state.value

        cog = ServiceContainer.get("cognitive_engine", default=None)
        if cog:
            for attr_name in dir(cog):
                obj = getattr(cog, attr_name, None)
                if obj and hasattr(obj, "state") and hasattr(obj, "name") and hasattr(obj.state, "value"):
                    if "breaker" in attr_name.lower():
                        cast(Dict[str, Any], resilience_data["circuit_breakers"])[obj.name] = obj.state.value

        snap_mgr = ServiceContainer.get("snapshot_manager", default=None)
        if snap_mgr and hasattr(snap_mgr, "snapshot_file"):
            resilience_data["snapshot"] = "saved" if snap_mgr.snapshot_file.exists() else "none"

        llm_router = ServiceContainer.get("llm_router", default=None)
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

        resilience_data["hardening_active"] = ServiceContainer.get("stability_guardian", default=None) is not None
    except Exception as e:
        logger.debug("Resilience status collection failed: %s", e)

    # ── Qualia Status ──
    qualia_data: Dict[str, Any] = {"pri": 0.0, "q_norm": 0.0, "dominant_dim": "none", "in_attractor": False, "_stale": True}
    try:
        qualia = ServiceContainer.get("qualia_synthesizer", default=None)
        if not qualia and orch:
            qualia = getattr(orch, "qualia", None)
        if qualia:
            qualia_data["_stale"] = False
            qualia_data["pri"] = round(float(getattr(qualia, "pri", 0.0)), 4)
            qualia_data["q_norm"] = round(float(getattr(qualia, "q_norm", 0.0)), 4)
            qualia_data["dominant_dim"] = getattr(qualia, "_history", None) and len(qualia._history) > 0 and qualia._history[-1].dominant_dimension or "none"
            qualia_data["in_attractor"] = getattr(qualia, "_in_attractor", False)
            qualia_data["identity_coherence"] = round(float(getattr(qualia, "identity_drift_score", 1.0)) * 100, 1)
    except Exception as e:
        logger.debug("Qualia status collection failed: %s", e)

    # ── Mycelial Network Status ──
    mycelial_data: Dict[str, Any] = {"nodes": 0, "edges": 0, "health": "offline"}
    try:
        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium:
            if hasattr(mycelium, "pathways") and hasattr(mycelium, "hyphae"):
                mycelial_data["nodes"] = len(mycelium.pathways)
                mycelial_data["edges"] = len(mycelium.hyphae)
            mycelial_data["health"] = "online"
    except Exception as e:
        logger.debug("Mycelial network status collection failed: %s", e)

    # ── PNEUMA Engine Status ──
    pneuma_data: Dict[str, Any] = {"temperature": 0.7, "arousal": 0.0, "stability": 0.0,
                   "attractor_count": 0, "efe_score": 0.0, "online": False, "_stale": True}
    try:
        from core.pneuma.pneuma import get_pneuma
        pn = get_pneuma()
        if pn and pn._running:
            pneuma_data["online"] = True
            pneuma_data["_stale"] = False
            pneuma_data["temperature"] = round(pn.get_llm_temperature(), 3)
            pe = getattr(pn, "precision", None)
            if pe and hasattr(pe, "fhn"):
                s = pe.fhn.state
                pneuma_data["arousal"] = round(float(s.v), 3)
                pneuma_data["stability"] = round(float(s.w), 3)
            tm = getattr(pn, "topo_memory", None)
            if tm:
                pneuma_data["attractor_count"] = int(tm.attractor_count)
    except Exception as e:
        logger.debug("PNEUMA status collection failed: %s", e)

    # ── MHAF Field Status ──
    mhaf_data: Dict[str, Any] = {"phi": 0.0, "nodes": 0, "edges": 0, "free_energy": 0.0,
                 "lexicon_size": 0, "online": False, "_stale": True}
    try:
        from core.consciousness.mhaf_field import get_mhaf
        mhaf = get_mhaf()
        if mhaf and mhaf._running:
            mhaf_data["online"] = True
            mhaf_data["_stale"] = False
            mhaf_data["nodes"] = len(mhaf._nodes)
            mhaf_data["edges"] = len(mhaf._edges)
            mhaf_data["free_energy"] = round(float(mhaf._free_energy), 4)
    except Exception as e:
        logger.debug("MHAF status collection failed: %s", e)
    # Wire real PhiCore IIT 4.0 phi into the MHAF data (replaces the surrogate)
    try:
        phi_core = ServiceContainer.get("phi_core", default=None)
        if phi_core is not None:
            result = phi_core._last_result
            if result is not None:
                mhaf_data["phi"] = round(float(result.phi_s), 4)
                mhaf_data["phi_complex"] = result.is_complex
                mhaf_data["phi_mip"] = result.mip_description
                mhaf_data["phi_samples"] = result.tpm_n_samples
    except Exception as e:
        logger.debug("PhiCore status collection failed: %s", e)
    try:
        from core.consciousness.neologism_engine import get_neologism_engine
        neo = get_neologism_engine()
        if neo:
            mhaf_data["lexicon_size"] = len(neo._lexicon)
    except Exception as e:
        logger.debug("Neologism lexicon count failed: %s", e)

    # ── Security Status ──
    security_data: Dict[str, Any] = {
        "trust_level": "unknown", "threat_score": 0.0,
        "integrity_ok": True, "passphrase_set": False, "_stale": True,
    }
    try:
        from core.security.trust_engine import get_trust_engine
        te = get_trust_engine()
        ts = te.get_status()
        security_data["trust_level"] = ts.get("level", "guest")
        security_data["_stale"] = False
    except Exception as e:
        logger.debug("Security status collection failed: %s", e)
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        ep = get_emergency_protocol()
        eps = ep.get_status()
        security_data["threat_score"] = eps.get("threat_score", 0.0)
        security_data["threat_level"] = eps.get("threat_level", "none")
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)
    try:
        from core.security.integrity_guardian import get_integrity_guardian
        igs = get_integrity_guardian().get_status()
        security_data["integrity_ok"] = bool(
            igs.get("integrity_ok", igs.get("alert_count", 0) == 0)
        )
        security_data["integrity_files"] = igs.get("manifest_files", 0)
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)
    try:
        from core.security.user_recognizer import get_user_recognizer
        security_data["passphrase_set"] = get_user_recognizer().has_passphrase()
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    # ── Circadian State ──
    circadian_data: Dict[str, Any] = {}
    try:
        from core.senses.circadian import get_circadian
        ce = get_circadian()
        ce.update()
        s = ce.state
        circadian_data = {
            "phase": s.phase.value,
            "arousal_baseline": round(s.arousal_baseline, 3),
            "energy_modifier": round(s.energy_modifier, 3),
            "cognitive_mode": s.cognitive_mode,
        }
    except Exception as e:
        logger.debug("Circadian status collection failed: %s", e)

    # ── Substrate Learning ──
    substrate_data: Dict[str, Any] = {}
    try:
        from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
        substrate_data["lora_bridge"] = get_crsm_lora_bridge().get_status()
    except Exception as e:
        logger.debug("LoRA bridge status failed: %s", e)
    try:
        from core.consciousness.experience_consolidator import get_experience_consolidator
        substrate_data["consolidator"] = get_experience_consolidator().get_status()
    except Exception as e:
        logger.debug("Consolidator status failed: %s", e)

    # ── Terminal Fallback Status ──
    terminal_data: Dict[str, Any] = {"active": False, "pending": 0, "watchdog": False}
    try:
        from core.terminal_chat import get_terminal_fallback, get_terminal_watchdog
        tf = get_terminal_fallback()
        terminal_data["active"] = tf.is_active
        terminal_data["pending"] = len(tf._pending)
        tw = get_terminal_watchdog()
        terminal_data["watchdog"] = tw._running if tw else False
    except Exception as e:
        logger.debug("Terminal fallback status collection failed: %s", e)

    desktop_access_data = await _collect_desktop_access_summary()

    # ── Final Response Assembly ──
    try:
        voice_mod = _voice_engine_fn() if _voice_engine_fn else None
        smc_mod = ServiceContainer.get("sensory_motor_cortex", default=None)
        from interface.routes.privacy import get_browser_camera_privacy

        browser_camera_privacy = get_browser_camera_privacy()

        privacy_data = {
            "camera_enabled": bool(browser_camera_privacy.get("enabled", False)),
            "camera_mode": browser_camera_privacy.get("mode", "off"),
            "camera_reason": browser_camera_privacy.get("reason"),
            "continuous_camera_enabled": getattr(smc_mod, "camera_enabled", False),
            "microphone_enabled": getattr(voice_mod, "microphone_enabled", True),
            "speaking_enabled": getattr(voice_mod, "speaking_enabled", True),
        }

        conversation_ready = bool(conversation_lane.get("conversation_ready", False))
        lane_is_standby = _conversation_lane_is_standby(conversation_lane)
        service_ok = bool(boot_snapshot.get("system_ready", False))
        diagnostics_data = {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        }

        health_status = (
            "ok"
            if service_ok and (conversation_ready or lane_is_standby) else
            "unavailable"
            if service_ok and str(conversation_lane.get("state", "") or "").lower() == "failed" else
            "recovering"
            if service_ok and str(conversation_lane.get("state", "") or "").lower() == "recovering" else
            "warming"
            if service_ok and not conversation_ready else
            "booting"
        )

        payload = {
            "status":      health_status,
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
            "homeostasis": homeo_data,
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
            "terminal":       terminal_data,
            "desktop_access": desktop_access_data,
            "transcendence": transcendence_data,
            "privacy":        privacy_data,
            "executive_closure": executive_closure_data,
            "consciousness_evidence": consciousness_evidence,
            "executive_authority": executive_authority_data,
            "interaction_signals": interaction_signals_data,
            "conversation_lane": conversation_lane,
            "diagnostics": diagnostics_data,
            "runtime":        rt,
            "scheduler":      scheduler.get_health(),
            "boot":           boot_snapshot,
            "timestamp":      datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error("Final health payload assembly failed: %s", e)
        payload = {
            "status": "degraded",
            "error": str(e),
            "version": version_string("full"),
            "uptime": 0.0,
            "cycle_count": 0,
            "cpu_usage": 0,
            "ram_usage": 0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }

    return JSONResponse(_json_safe(payload))


@router.get("/tools/catalog")
async def api_tools_catalog():
    catalog = _collect_tool_catalog()
    return JSONResponse({"tools": catalog, "count": len(catalog)})


@router.get("/ui/bootstrap")
async def api_ui_bootstrap(request: Request = None):
    from interface.routes.chat import _conversation_log, _conversation_log_lock
    _restore_owner_session_from_request(request)
    orch = ServiceContainer.get("orchestrator", default=None)
    rt = get_runtime_state()
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
    except Exception as exc:
        logger.debug("Bootstrap constitutional snapshot failed: %s", exc)

    try:
        executive_authority = ServiceContainer.get("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_status = executive_authority.get_status()
    except Exception as exc:
        logger.debug("Bootstrap executive snapshot failed: %s", exc)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except Exception as exc:
        logger.debug("Bootstrap interaction signal snapshot failed: %s", exc)

    tool_catalog = _collect_tool_catalog()
    conversation_lane = _collect_conversation_lane_status()
    boot_snapshot, _status_code = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    status_obj = getattr(orch, "status", None)
    async with _conversation_log_lock:
        recent_conversation = list(_conversation_log)[-40:]

    from pathlib import Path
    STATIC_DIR = config.paths.project_root / "interface" / "static"
    SHELL_DIST_DIR = STATIC_DIR / "shell" / "dist"
    LEGACY_UI_INDEX = STATIC_DIR / "index.html"

    legacy_ui_status = {
        "shell": "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell",
        "legacy_fallback_available": LEGACY_UI_INDEX.exists(),
        "experimental_shell_available": (SHELL_DIST_DIR / "index.html").exists(),
    }
    shell_status_helper = globals().get("_collect_legacy_shell_status")
    if callable(shell_status_helper):
        try:
            helper_payload = shell_status_helper() or {}
            if isinstance(helper_payload, dict):
                legacy_ui_status.update(helper_payload)
        except Exception as exc:
            logger.debug("Bootstrap legacy shell status sync failed: %s", exc)

    payload = {
        "identity": {
            "name": "Aura Luna",
            "version": version_string("full"),
            "build": VERSION,
        },
        "session": {
            "connected": bool(boot_snapshot.get("system_ready", False)),
            "initialized": bool(getattr(status_obj, "initialized", False)),
            "websocket_clients": ws_manager.count(),
            "is_gui_proxy": os.environ.get("AURA_GUI_PROXY") == "1",
        },
        "constitutional": constitutional_status,
        "executive": executive_status,
        "state": state_summary,
        "commitments": _collect_commitment_summary(),
        "tools": tool_catalog,
        "capabilities": _collect_runtime_capabilities(conversation_lane),
        "desktop_access": await _collect_desktop_access_summary(),
        "conversation": {
            "recent": recent_conversation,
            "count": len(recent_conversation),
            "lane": conversation_lane,
        },
        "voice": _collect_voice_summary(),
        "interaction_signals": interaction_signals_data,
        "telemetry": {
            "cpu_usage": psutil.cpu_percent(interval=None),
            "ram_usage": psutil.virtual_memory().percent,
            "runtime": rt,
            "boot": boot_snapshot,
        },
        "diagnostics": {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        },
        "ui": {
            "shell": legacy_ui_status.get("shell", "legacy_shell" if LEGACY_UI_INDEX.exists() else "react_shell"),
            "legacy_fallback_available": bool(legacy_ui_status.get("legacy_fallback_available", LEGACY_UI_INDEX.exists())),
            "experimental_shell_available": bool(legacy_ui_status.get("experimental_shell_available", (SHELL_DIST_DIR / "index.html").exists())),
            "status_flags": _derive_ui_status_flags(
                state_summary=state_summary,
                executive_status=executive_status,
                boot_snapshot=boot_snapshot,
                tool_catalog=tool_catalog,
            ),
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    return JSONResponse(payload)


@router.get("/health/boot")
async def api_boot_health():
    orch = ServiceContainer.get("orchestrator", default=None)
    rt = get_runtime_state()
    conversation_lane = _collect_conversation_lane_status()
    payload, status_code = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    return JSONResponse(payload, status_code=status_code)


@router.get("/health/heartbeat")
async def api_heartbeat():
    """Minimal heartbeat for GUI Actor watchdog."""
    return {"status": "ok", "time": time.time()}
