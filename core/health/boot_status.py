from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from core.health.conversation_lane import (
    conversation_lane_is_busy,
    conversation_lane_is_serving,
)
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.health_contract import (
    evaluate_health,
    probes_from_report,
    required_probe_groups_pass,
    required_probe_status,
    startup_complete_at,
)
from core.runtime.version import VERSION, version_string

_BOOT_STATUS_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)

#: The health surface must be able to report on a mind that failed to start,
#: which it cannot do if importing it drags in core.brain. This is the display
#: NAME of the primary endpoint, not a handle to it —
#: core.brain.llm.model_registry.PRIMARY_ENDPOINT is the same literal, and
#: tests/test_health_does_not_import_the_brain.py holds the two together.
_PRIMARY_ENDPOINT_NAME = "Cortex"


def _record_boot_degradation(
    error: BaseException,
    *,
    severity: Severity,
    action: str,
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "boot_status",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _runtime_contract_snapshot() -> dict[str, Any]:
    try:
        return evaluate_health().to_report()
    except _BOOT_STATUS_RECOVERABLE_ERRORS as exc:
        _record_boot_degradation(
            exc,
            severity="critical",
            action="failed closed: runtime contract unavailable during boot health snapshot",
            classification=FallbackClassification.AUDIT_GAP,
        )
        return {
            "status": "unknown",
            "healthy": False,
            "operational": False,
            "status_code": 503,
            "failures": {
                "critical": [
                    {
                        "name": "Runtime Health Contract",
                        "container_key": "runtime_health_contract",
                        "tier": "critical",
                        "present": False,
                        "liveness": "failed",
                        "error": str(exc),
                    }
                ],
                "important": [],
                "optional": [],
            },
            "tier_summary": {},
        }


def _contract_failure_keys(contract: dict[str, Any], tier: str) -> list[str]:
    failures = contract.get("failures", {})
    tier_failures = failures.get(tier, []) if isinstance(failures, dict) else []
    if not isinstance(tier_failures, list):
        return []
    keys: list[str] = []
    for failure in tier_failures:
        if isinstance(failure, dict):
            key = str(failure.get("container_key") or failure.get("name") or "").strip()
            if key:
                keys.append(key)
    return keys


def _boot_progress_for_phase(boot_phase: str) -> int:
    normalized = str(boot_phase or "").strip().lower()
    mapping = {
        "kernel_bootstrap": 14,
        "kernel_warming": 48,
        "conversation_warming": 78,
        "conversation_operational": 100,
        "conversation_working": 100,
        "conversation_recovering": 86,
        "conversation_failed": 92,
        "kernel_ready": 100,
        "proxy_ready": 100,
        "proxy_transport_only": 24,
        # Startup already completed once in this process: boot progress is
        # over (100), the runtime is just degraded right now.
        "runtime_degraded": 100,
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
    endpoint = str(lane.get("foreground_endpoint", "") or _PRIMARY_ENDPOINT_NAME)
    model_label = str(
        lane.get("desired_model")
        or lane.get("expected_model")
        or endpoint
        or _PRIMARY_ENDPOINT_NAME
    )
    failure_reason = str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "")

    if normalized == "proxy_ready":
        return "Aura proxy is ready."
    if normalized == "proxy_transport_only":
        return "Aura proxy is alive; canonical runtime is not ready."
    if normalized == "kernel_ready":
        return "Aura is awake."
    if normalized == "conversation_operational":
        return "Aura's conversation lane is ready; non-critical runtime checks remain degraded."
    if normalized == "conversation_working":
        return "Aura is answering through the live conversation lane."
    if normalized == "conversation_recovering":
        if "cortex" in endpoint.lower():
            return f"Recovering local {model_label}…"
        return "Recovering Aura's conversation lane…"
    if normalized == "conversation_failed":
        if failure_reason.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
            return (
                f"Local {model_label} is unavailable: Aura's managed backend "
                "failed during startup."
            )
        if "cortex" in endpoint.lower():
            return f"Local {model_label} is unavailable."
        return "Aura's conversation lane is unavailable."
    if normalized == "conversation_warming":
        if "cortex" in endpoint.lower():
            return f"Warming local {model_label}…"
        return "Warming Aura's conversation lane…"
    if normalized == "kernel_warming":
        if "runtime_integrity" in blockers:
            return "Validating Aura runtime integrity…"
        return "Booting Aura core systems…"
    if normalized == "runtime_degraded":
        return "Aura is running but degraded; recovery is in progress."
    return "Starting Aura kernel…"


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    return (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
        and not str(lane.get("last_failure_reason", "") or lane.get("last_error", "") or "").strip()
    )


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
        except _BOOT_STATUS_RECOVERABLE_ERRORS as exc:
            _record_boot_degradation(
                exc,
                severity="degraded",
                action="failed closed: marked orchestrator unhealthy and exposed health_check_error",
                classification=FallbackClassification.SAFE_FALLBACK,
                extra={"orchestrator_present": orchestrator is not None},
            )
            healthy = False
            health_check_error = str(exc)

    runtime_contract = _runtime_contract_snapshot()
    runtime_contract_operational = bool(runtime_contract.get("operational", False))
    runtime_contract_healthy = bool(runtime_contract.get("healthy", False))
    runtime_required_probes = required_probe_status(runtime_contract)
    runtime_required_probes_ok = required_probe_groups_pass(runtime_required_probes)
    # The K2 probe split (startup/liveness/readiness) derives from the same
    # report — no second health evaluation. Its startup verdict latches the
    # first time readiness passes for the life of this process.
    probe_split = probes_from_report(runtime_contract)
    startup_latched = startup_complete_at() is not None
    critical_contract_failures = _contract_failure_keys(runtime_contract, "critical")
    important_contract_failures = _contract_failure_keys(runtime_contract, "important")

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
        blockers = []
        if not runtime_integrity_ok:
            blockers.append("runtime_integrity")
        if not runtime_contract_operational:
            blockers.append("runtime_contract")
            blockers.extend(f"critical:{key}" for key in critical_contract_failures)
        if not runtime_contract_healthy:
            blockers.append("runtime_contract_healthy")
            blockers.extend(f"important:{key}" for key in important_contract_failures)
        if not runtime_required_probes_ok:
            blockers.append("runtime_required_probes")
            blockers.extend(
                f"probe:{name}"
                for name, probe in runtime_required_probes.items()
                if isinstance(probe, dict) and not bool(probe.get("ok", False))
            )
        system_ready = (
            runtime_integrity_ok
            and runtime_contract_healthy
            and runtime_required_probes_ok
        )
        conversation_ready = system_ready
        boot_phase = "proxy_ready" if system_ready else "proxy_transport_only"
        status_text = "ready" if system_ready else "not_ready"
        http_status = 200 if system_ready else 503
        user_ready = bool(system_ready)
        launcher_ready = bool(system_ready)
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
        if not runtime_contract_operational:
            blockers.append("runtime_contract")
            blockers.extend(f"critical:{key}" for key in critical_contract_failures)
        if not runtime_contract_healthy:
            blockers.append("runtime_contract_healthy")
            blockers.extend(f"important:{key}" for key in important_contract_failures)
        if not runtime_required_probes_ok:
            blockers.append("runtime_required_probes")
            blockers.extend(
                f"probe:{name}"
                for name, probe in runtime_required_probes.items()
                if isinstance(probe, dict) and not bool(probe.get("ok", False))
            )
        if not (running or runtime_fresh or cycle_count > 0):
            blockers.append("running")

        conversation_ready = True
        conversation_busy = False
        conversation_state = "ready"
        if isinstance(conversation_lane, dict) and conversation_lane:
            conversation_ready = bool(conversation_lane.get("conversation_ready", False))
            conversation_busy = conversation_lane_is_busy(conversation_lane)
            conversation_state = str(conversation_lane.get("state", "warming") or "warming")
        conversation_standby = _conversation_lane_is_standby(conversation_lane)

        system_ready = (
            orchestrator is not None
            and initialized
            and healthy
            and not last_error
            and runtime_integrity_ok
            and runtime_contract_healthy
            and runtime_required_probes_ok
            and (running or runtime_fresh or cycle_count > 0)
        )
        launcher_openable = bool(
            orchestrator is not None
            and initialized
            and (running or runtime_fresh or cycle_count > 0)
        )
        user_ready = bool(system_ready)
        launcher_ready = launcher_openable
        status_text = "ready" if system_ready else "booting"
        http_status = 200 if system_ready else 503
        # NOTE: deliberately does NOT require the orchestrator-level `healthy`
        # flag. An important-tier degradation (e.g. mind_tick under load) turns
        # `healthy` false while the conversation lane works perfectly — that
        # state must present as "degraded but conversational", not trap the
        # user at "booting" forever (observed live: 55 minutes of "booting,
        # 48%" on a fully conversational instance). Critical failures still
        # gate via runtime_contract_operational + required probes; `healthy`
        # stays visible in checks/blockers.
        conversation_operational = bool(
            orchestrator is not None
            and initialized
            and not last_error
            and runtime_integrity_ok
            and runtime_contract_operational
            and runtime_required_probes_ok
            and conversation_ready
            and (running or runtime_fresh or cycle_count > 0)
        )

        if system_ready and conversation_ready:
            boot_phase = "kernel_ready"
            status_text = "ready"
            user_ready = True
            launcher_ready = True
            http_status = 200
        elif conversation_operational:
            # Do not claim full health while important services are degraded.
            # But if all critical probes pass and the live conversation lane is
            # ready, the desktop chat surface must connect instead of trapping
            # the user behind the launcher/loading shell.
            boot_phase = "conversation_operational"
            status_text = "degraded"
            user_ready = True
            launcher_ready = True
            http_status = 200
        elif system_ready and conversation_busy:
            boot_phase = "conversation_working"
            status_text = "working"
            # A functional lane actively answering a turn is READY — the desktop
            # must connect and show the streaming reply, not fall back to
            # "Connecting to runtime" for the length of a long turn or a run of
            # back-to-back turns (observed live: a busy lane reported ready=false
            # and the shell sat at "Initializing"). A lane that is busy *warming
            # up* (handshaking/spawning) is genuinely not-ready and stays so.
            user_ready = conversation_lane_is_serving(conversation_lane)
            launcher_ready = True
            http_status = 200
        elif system_ready and not conversation_ready:
            blockers.append("conversation_ready")
            user_ready = False
            # The launcher and diagnostics shell may open once the governed
            # runtime contract is alive. User chat remains not-ready until the
            # conversation lane passes, so heartbeat/readiness cannot report a
            # false healthy state while the desktop is still inspectable.
            launcher_ready = True
            http_status = 200 if conversation_standby else 503
            if conversation_state == "failed":
                blockers.append("conversation_failed")
                boot_phase = "conversation_failed"
                status_text = "degraded"
                http_status = 503
            else:
                boot_phase = "conversation_recovering" if conversation_state == "recovering" else "conversation_warming"
                status_text = "recovering" if conversation_state == "recovering" else "warming"
        elif initialized or running or runtime_fresh or cycle_count > 0:
            # K2 startup latch: once this process has EVER been ready, a
            # later fall into this branch is a DEGRADATION, not a boot. The
            # shell must never regress to "booting N%" over a mind that
            # already served (observed live: 55 minutes of "booting, 48%").
            # Traffic gating (503) is unchanged — only the presentation.
            boot_phase = "runtime_degraded" if startup_latched else "kernel_warming"
            status_text = "degraded" if startup_latched else "booting"
            user_ready = False
            launcher_ready = launcher_openable
        else:
            boot_phase = "runtime_degraded" if startup_latched else "kernel_bootstrap"
            status_text = "degraded" if startup_latched else "booting"
            user_ready = False
            launcher_ready = False

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
        "ready": user_ready,
        "launcher_ready": launcher_ready,
        "system_ready": system_ready,
        "conversation_ready": conversation_ready,
        "conversation_busy": bool(
            conversation_lane_is_busy(conversation_lane)
            if isinstance(conversation_lane, dict)
            else False
        ),
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
            "runtime_contract_operational": runtime_contract_operational,
            "runtime_contract_healthy": runtime_contract_healthy,
            "runtime_required_probes": runtime_required_probes_ok,
            "startup_latched": startup_latched,
        },
        "probes": {name: probe.to_dict() for name, probe in probe_split.items()},
        "orchestrator": {
            "cycle_count": cycle_count,
            "last_error": last_error,
            "uptime": uptime,
        },
        "runtime_age_s": uptime,
        "runtime": runtime_payload,
        "runtime_contract": runtime_contract,
        "required_probes": runtime_required_probes,
        "runtime_degradations": {
            "important": important_contract_failures,
            "critical": critical_contract_failures,
        },
        "integrity": {
            "sha256": runtime_hash,
            "signature_present": runtime_signature_present,
        },
        "blockers": blockers,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    if isinstance(conversation_lane, dict) and conversation_lane:
        payload["conversation_lane"] = conversation_lane

    if health_check_error:
        payload["health_check_error"] = health_check_error

    return payload, http_status
