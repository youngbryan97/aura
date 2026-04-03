from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from core.brain.llm.model_registry import PRIMARY_ENDPOINT
from core.version import VERSION, version_string


def _boot_progress_for_phase(boot_phase: str) -> int:
    normalized = str(boot_phase or "").strip().lower()
    mapping = {
        "kernel_bootstrap": 14,
        "kernel_warming": 48,
        "conversation_warming": 78,
        "conversation_recovering": 86,
        "conversation_failed": 92,
        "kernel_ready": 100,
        "proxy_ready": 100,
    }
    return mapping.get(normalized, 8)


def _boot_status_message(
    boot_phase: str,
    *,
    blockers: list[str],
    conversation_lane: dict[str, Any] | None,
) -> str:
    normalized = str(boot_phase or "").strip().lower()
    lane = conversation_lane if isinstance(conversation_lane, dict) else {}
    endpoint = str(lane.get("foreground_endpoint", "") or PRIMARY_ENDPOINT)
    lane_state = str(lane.get("state", "") or "").strip().lower()
    warmup_attempted = bool(lane.get("warmup_attempted", False))
    warmup_in_flight = bool(lane.get("warmup_in_flight", False))
    conversation_ready = bool(lane.get("conversation_ready", False))
    failure_reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "")

    if normalized == "proxy_ready":
        return "Aura proxy is ready."
    if normalized == "kernel_ready":
        if (
            not conversation_ready
            and lane_state in {"cold", "closed", ""}
            and not warmup_attempted
            and not warmup_in_flight
        ):
            return "Aura is awake. Cortex will warm on first turn."
        return "Aura is awake."
    if normalized == "conversation_recovering":
        if "cortex" in endpoint.lower():
            return "Recovering local Cortex (32B)…"
        return "Recovering Aura's conversation lane…"
    if normalized == "conversation_failed":
        if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
            return "Local Cortex (32B) is unavailable: Aura's managed backend failed during startup."
        if "cortex" in endpoint.lower():
            return "Local Cortex (32B) is unavailable."
        return "Aura's conversation lane is unavailable."
    if normalized == "conversation_warming":
        if "cortex" in endpoint.lower():
            return "Warming local Cortex (32B)…"
        return "Warming Aura's conversation lane…"
    if normalized == "kernel_warming":
        if "runtime_integrity" in blockers:
            return "Validating Aura runtime integrity…"
        return "Booting Aura core systems…"
    return "Starting Aura kernel…"


def build_boot_health_snapshot(
    orchestrator: Any,
    runtime_state: dict[str, Any] | None,
    *,
    is_gui_proxy: bool,
    conversation_lane: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
    runtime_payload = runtime_state.get("state", {}) if isinstance(runtime_state.get("state"), dict) else {}
    runtime_hash = str(runtime_state.get("sha256", "") or "")
    runtime_signature_present = bool(runtime_state.get("signature"))
    runtime_integrity_ok = bool(runtime_hash and runtime_signature_present)
    now = time.time()

    status = getattr(orchestrator, "status", None)
    initialized = bool(getattr(status, "initialized", False))
    running = bool(getattr(status, "running", False))
    healthy = bool(getattr(status, "healthy", initialized or running))
    last_error = str(getattr(status, "last_error", "") or "")
    cycle_count = int(getattr(status, "cycle_count", 0) or 0)
    start_time = getattr(status, "start_time", None) or getattr(orchestrator, "start_time", None)

    health_check_error = ""
    if orchestrator is not None and hasattr(orchestrator, "health_check"):
        try:
            healthy = bool(orchestrator.health_check())
        except Exception as exc:
            healthy = False
            health_check_error = str(exc)

    uptime = 0.0
    try:
        if start_time:
            uptime = round(max(0.0, now - float(start_time)), 1)
    except (TypeError, ValueError):
        uptime = 0.0

    runtime_fresh = False
    runtime_timestamp = runtime_payload.get("timestamp_utc")
    if runtime_timestamp:
        try:
            runtime_dt = datetime.fromisoformat(str(runtime_timestamp).replace("Z", "+00:00"))
            runtime_fresh = (now - runtime_dt.timestamp()) <= 120.0
        except ValueError:
            runtime_fresh = False
    if not runtime_fresh:
        heartbeat_tick = runtime_payload.get("heartbeat_tick")
        if isinstance(heartbeat_tick, (int, float)):
            runtime_fresh = (now - float(heartbeat_tick)) <= 120.0

    if is_gui_proxy:
        system_ready = True
        conversation_ready = True
        boot_phase = "proxy_ready"
        status_text = "ready"
        http_status = 200
        blockers: list[str] = []
    else:
        blockers = []
        if orchestrator is None:
            blockers.append("orchestrator")
        if not initialized:
            blockers.append("initialized")
        if not healthy:
            blockers.append("healthy")
        if last_error:
            blockers.append("last_error")
        if not runtime_integrity_ok:
            blockers.append("runtime_integrity")
        if not (running or runtime_fresh or cycle_count > 0):
            blockers.append("running")

        conversation_ready = True
        conversation_state = "ready"
        warmup_attempted = False
        warmup_in_flight = False
        if isinstance(conversation_lane, dict) and conversation_lane:
            conversation_ready = bool(conversation_lane.get("conversation_ready", False))
            conversation_state = str(conversation_lane.get("state", "warming") or "warming")
            warmup_attempted = bool(conversation_lane.get("warmup_attempted", False))
            warmup_in_flight = bool(conversation_lane.get("warmup_in_flight", False))

        system_ready = (
            orchestrator is not None
            and initialized
            and healthy
            and not last_error
            and runtime_integrity_ok
            and (running or runtime_fresh or cycle_count > 0)
        )
        status_text = "ready" if system_ready else "booting"
        http_status = 200 if system_ready else 503

        if system_ready and conversation_ready:
            boot_phase = "kernel_ready"
            status_text = "ready"
        elif system_ready and not conversation_ready:
            lane_is_standby = (
                conversation_state in {"cold", "closed", ""}
                and not warmup_attempted
                and not warmup_in_flight
            )
            if lane_is_standby:
                boot_phase = "kernel_ready"
                status_text = "ready"
            else:
                blockers.append("conversation_ready")
                if conversation_state == "failed":
                    blockers.append("conversation_failed")
                    boot_phase = "conversation_failed"
                    status_text = "degraded"
                else:
                    boot_phase = "conversation_recovering" if conversation_state == "recovering" else "conversation_warming"
                    status_text = "recovering" if conversation_state == "recovering" else "warming"
        elif initialized or running or runtime_fresh or cycle_count > 0:
            boot_phase = "kernel_warming"
            status_text = "booting"
        else:
            boot_phase = "kernel_bootstrap"
            status_text = "booting"

    progress = _boot_progress_for_phase(boot_phase)
    status_message = _boot_status_message(
        boot_phase,
        blockers=blockers,
        conversation_lane=conversation_lane,
    )

    payload: dict[str, Any] = {
        "version": version_string("full"),
        "semver": VERSION,
        "status": status_text,
        "status_message": status_message,
        "ready": system_ready,
        "launcher_ready": system_ready,
        "system_ready": system_ready,
        "conversation_ready": conversation_ready,
        "boot_phase": boot_phase,
        "progress": progress,
        "mode": "gui_proxy" if is_gui_proxy else "kernel",
        "checks": {
            "orchestrator_present": orchestrator is not None,
            "initialized": initialized,
            "running": running,
            "runtime_fresh": runtime_fresh,
            "healthy": healthy,
            "runtime_integrity": runtime_integrity_ok,
        },
        "orchestrator": {
            "cycle_count": cycle_count,
            "last_error": last_error,
            "uptime": uptime,
        },
        "runtime_age_s": uptime,
        "runtime": runtime_payload,
        "integrity": {
            "sha256": runtime_hash,
            "signature_present": runtime_signature_present,
        },
        "blockers": blockers,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    if isinstance(conversation_lane, dict) and conversation_lane:
        payload["conversation_lane"] = conversation_lane

    if health_check_error:
        payload["health_check_error"] = health_check_error

    return payload, http_status
