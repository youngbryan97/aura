import logging
import os
import time
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_inference_gate
from core.runtime.shutdown_coordinator import is_shutdown_requested

logger = logging.getLogger("Aura.SubsystemAudit")


def _health_pulse_boot_grace_s() -> float:
    raw = os.getenv("AURA_HEALTH_PULSE_BOOT_GRACE_S") or os.getenv(
        "AURA_WATCHDOG_BOOT_GRACE_S", "120"
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError, OverflowError):
        return 120.0


_CONVERSATION_BOOT_TRANSIENT_BLOCKERS = {
    "chat_dependencies_warming",
    "foreground_owner",
    "foreground_warming",
    "prewarm_not_attempted",
    "startup_grace",
    "visible_conversation_probe_missing",
    "warmup_foreground_owner",
    "warmup_in_flight",
}

_CONVERSATION_BUSY_BLOCKERS = {
    "active_generation_in_flight",
    "foreground_generation_active",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed == parsed else default


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    state = str(lane.get("state", "") or "").strip().lower()
    cold_standby = (
        not bool(lane.get("conversation_ready", False))
        and state in {"cold", "closed", ""}
        and not bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
    )
    if cold_standby:
        return True

    blockers_raw = lane.get("readiness_blockers", [])
    blockers_seq = blockers_raw if isinstance(blockers_raw, (list, tuple, set)) else [blockers_raw]
    blockers = {
        str(item or "").strip().lower()
        for item in blockers_seq
        if str(item or "").strip()
    }
    reason = str(
        lane.get("last_failure_reason", "")
        or lane.get("last_error", "")
        or ""
    ).strip()
    # The model-load probe can prove an internally ready Cortex before any
    # user-visible answer has completed. That state is intentionally not yet
    # `conversation_ready`, but it is also not a failed conversation lane.
    # Keep this classification narrow: a request, blocker, prior visible
    # success, or failure reason all move the lane out of standby.
    return bool(
        not lane.get("conversation_ready", False)
        and state == "ready"
        and bool(lane.get("warmup_attempted", False))
        and not bool(lane.get("warmup_in_flight", False))
        and not bool(lane.get("has_generated_successfully", False))
        and _safe_float(lane.get("active_generations", 0.0), 0.0) <= 0.0
        and _safe_float(lane.get("current_request_started_at", 0.0), 0.0) <= 0.0
        and not blockers
        and not reason
    )


def _conversation_lane_is_boot_warming(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    if bool(lane.get("conversation_ready", False)):
        return False

    state = str(lane.get("state", "") or "").strip().lower()
    if state in {"critical", "dead", "failed"}:
        return False

    blockers_raw = lane.get("readiness_blockers", [])
    blockers_seq = blockers_raw if isinstance(blockers_raw, (list, tuple, set)) else [blockers_raw]
    blockers = {
        str(item or "").strip().lower()
        for item in blockers_seq
        if str(item or "").strip()
    }
    reason = str(
        lane.get("last_failure_reason", "")
        or lane.get("last_error", "")
        or ""
    ).strip().lower()
    warmup_active = bool(lane.get("warmup_in_flight", False)) or bool(
        lane.get("warmup_attempted", False)
    )

    if blockers and blockers <= _CONVERSATION_BOOT_TRANSIENT_BLOCKERS:
        return True
    if reason in _CONVERSATION_BOOT_TRANSIENT_BLOCKERS:
        return True
    if warmup_active and state in {
        "",
        "cold",
        "handshaking",
        "initializing",
        "recovering",
        "spawning",
        "starting",
        "unknown",
        "warming",
    }:
        return True
    return state in {
        "handshaking",
        "initializing",
        "spawning",
        "starting",
        "warming",
    } and not reason


def _conversation_lane_is_busy(lane: dict[str, Any] | None) -> bool:
    lane = dict(lane or {})
    if bool(lane.get("conversation_ready", False)):
        return False

    state = str(lane.get("state", "") or "").strip().lower()
    if state in {"critical", "dead", "failed"}:
        return False

    blockers_raw = lane.get("readiness_blockers", [])
    blockers_seq = blockers_raw if isinstance(blockers_raw, (list, tuple, set)) else [blockers_raw]
    blockers = {
        str(item or "").strip().lower()
        for item in blockers_seq
        if str(item or "").strip()
    }
    reason = str(
        lane.get("last_failure_reason", "")
        or lane.get("last_error", "")
        or ""
    ).strip().lower()
    active_generations = _safe_float(lane.get("active_generations", 0.0), 0.0) > 0.0
    request_started = _safe_float(lane.get("current_request_started_at", 0.0), 0.0) > 0.0
    foreground_owned = bool(lane.get("foreground_owned", False))
    busy_blocker = bool(blockers & _CONVERSATION_BUSY_BLOCKERS) or reason in _CONVERSATION_BUSY_BLOCKERS
    return (
        state in {"ready", "recovering", "warming", "starting"}
        and (active_generations or request_started or foreground_owned or busy_blocker)
    )


def _collect_conversation_lane_status() -> dict[str, Any]:
    """Collect the foreground conversation lane without depending on routes.

    SubsystemAudit runs in core runtime code, so importing interface routes here
    would create fragile boot-time coupling. The inference gate owns the live
    lane status; if it cannot provide that status, the health pulse fails
    closed rather than calling the process healthy from background heartbeats.
    """
    try:

        gate = resolve_inference_gate()
        if gate is not None and hasattr(gate, "get_conversation_status"):
            lane = gate.get_conversation_status()
            if isinstance(lane, dict):
                return lane
            raise TypeError(f"inference_gate.get_conversation_status returned {type(lane).__name__}")
        return {
            "conversation_ready": False,
            "state": "unknown",
            "last_failure_reason": "conversation_lane_probe_missing",
        }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "subsystem_audit",
            exc,
            severity="critical",
            action="health pulse failed closed: conversation lane status unavailable",
            enforce_failure_policy=False,
        )
        return {
            "conversation_ready": False,
            "state": "unknown",
            "last_failure_reason": str(exc)[:240],
        }


class SubsystemAudit:
    """Tracks and verifies that all cognitive subsystems are actively running."""
    
    # All known subsystems and their expected pulse interval (seconds)
    SUBSYSTEMS = {
        "personality_engine":   60,
        "liquid_state":         5,
        "liquid_substrate":     5,
        "drive_controller":     60,
        "consciousness":        120,
        "affect_engine":        30,
        "agency_core":          10,
        "capability_engine":    60,
        "identity":             300,
        "cognitive_engine":     120,
        "sovereign_scanner":    10,
        # v47 FIX: Removed 'database_hygiene', 'memory', 'belief_graph', 'memory_manager', 'pulse_manager', 'soma'
        # These either heartbeat infrequently from the metabolic loop, or are obsolete aliases
        # that create false alarm noise in the health dashboard.
    }
    
    def __init__(self):
        self._heartbeats: dict[str, float] = {}
        self._failures: dict[str, list[dict[str, Any]]] = {}
        self._last_health_pulse = time.time()
        self._health_pulse_interval = 15  # TEMPORARY TEST: Emit full health report every 15s
        self._cycle_counts = 0
        self._start_time = time.time()
        logger.info("🫀 SubsystemAudit initialized. Tracking %d subsystems.", len(self.SUBSYSTEMS))
    
    def heartbeat(self, subsystem_name: str):
        """Register a heartbeat from a subsystem."""
        self._heartbeats[subsystem_name] = time.time()

    def report_failure(self, subsystem_name: str, error: str):
        """Record a failure event for a subsystem."""
        if subsystem_name not in self._failures:
            self._failures[subsystem_name] = []
        
        self._failures[subsystem_name].append({
            "timestamp": time.time(),
            "error": error
        })
        # Keep only last 5 failures
        history = self._failures[subsystem_name]
        if len(history) > 5:
            self._failures[subsystem_name] = history[-5:]
        logger.error("🚨 Subsystem [%s] reported failure: %s", subsystem_name, error)

    def get_status(self, subsystem_name: str | None = None) -> dict[str, Any]:
        """Get subsystem health.

        With a name, returns that subsystem's health. Without a name, returns
        the aggregate health report expected by generic ServiceContainer status
        readers.
        """
        if subsystem_name is None:
            return self.check_health()

        now = time.time()
        last_beat = self._heartbeats.get(subsystem_name)
        failures = self._failures.get(subsystem_name, [])
        
        degraded = len(failures) >= 3  # Simple threshold for degradation
        
        stale = (now - last_beat) if last_beat else None
        max_interval = self.SUBSYSTEMS.get(subsystem_name, 300)
        
        is_stale = stale > max_interval * 2 if stale else False
        
        # Derive human-readable status
        if last_beat is None:
            status = "NEVER_SEEN"
        elif is_stale:
            status = "STALE"
        elif degraded:
            status = "DEGRADED"
        else:
            status = "ACTIVE"
        
        return {
            "name": subsystem_name,
            "status": status,
            "active": last_beat is not None and not is_stale,
            "degraded": degraded,
            "stale_seconds": int(stale) if stale else None,
            "failure_count": len(failures),
            "last_error": failures[-1]["error"] if failures else None
        }
    
    def check_health(self) -> dict[str, Any]:
        """Check all subsystems and return their status."""
        now = time.time()
        report = {}
        all_ok = True
        
        # 1. Standard Subsystem Checks
        for name in self.SUBSYSTEMS:
            status_info = self.get_status(name)
            report[name] = status_info
            if not status_info["active"] or status_info["degraded"]:
                all_ok = False
        
        # 2. AFFECTIVE ESCALATION (Phase 23)
        try:
            from core.container import ServiceContainer
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis:
                status = homeostasis.get_status()
                # If Will to Live/Vitality or Integrity is critically low, escalate
                vitality = status.get("will_to_live", 1.0)
                integrity = status.get("integrity", 1.0)
                
                if integrity < 0.3 or vitality < 0.3:
                    all_ok = False
                    report["homeostasis_escalation"] = {
                        "status": "CRITICAL",
                        "vitality": vitality,
                        "integrity": integrity,
                        "reason": "Affective/Homeostatic collapse imminent"
                    }
                    logger.warning("🚨 [ESC] Homeostatic collapse detected in SubsystemAudit (Vitality: %.2f)", vitality)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('subsystem_audit', e)
            logger.debug("Affective escalation check failed: %s", e)

        return {"all_ok": all_ok, "subsystems": report, "checked_at": now}
    
    def should_emit_pulse(self) -> bool:
        """Check if it's time to emit a full health pulse."""
        return time.time() - self._last_health_pulse > self._health_pulse_interval
    
    def emit_pulse(self) -> str:
        """Generate a human-readable health pulse for the Neural Feed.

        Heartbeats alone do not prove health. The pulse is allowed to claim a
        healthy runtime only when the canonical runtime contract and required
        probe groups pass.
        """
        self._last_health_pulse = time.time()
        health = self.check_health()
        uptime = time.time() - self._start_time
        try:
            from core.runtime.health_contract import runtime_health_report

            contract = runtime_health_report()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "subsystem_audit",
                exc,
                severity="critical",
                action="health pulse failed closed: runtime contract unavailable",
                enforce_failure_policy=False,
            )
            contract = {
                "healthy": False,
                "status": "contract_unavailable",
                "required_probes": {"all_passed": False},
                "failures": {"critical": [], "important": [], "optional": []},
            }
        
        # System Metrics
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        lines = ["═══ UNIFIED HEALTH PULSE ═══"]
        lines.append(f"System: CPU {cpu}% | RAM {mem}% | Uptime: {int(uptime)}s")
        
        active_count = 0
        stale_count = 0
        missing_count = 0
        degraded_count = 0
        
        for name, info in health["subsystems"].items():
            status = info.get("status", "UNKNOWN")
            if status == "ACTIVE":
                active_count += 1
            elif status == "STALE":
                stale_count += 1
                lines.append(f"  ⚠️ {name}: STALE ({info['stale_seconds']}s)")
            elif status == "DEGRADED":
                degraded_count += 1
                lines.append(f"  ⚠️ {name}: DEGRADED ({info.get('last_error') or 'recent failures'})")
            else:
                missing_count += 1
                lines.append(f"  ❌ {name}: NEVER SEEN")
        
        required = contract.get("required_probes", {}) if isinstance(contract, dict) else {}
        try:
            from core.runtime.health_contract import required_probe_groups_pass

            required_ok = required_probe_groups_pass(required)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "subsystem_audit",
                exc,
                severity="critical",
                action="health pulse failed closed: required probe validator unavailable",
                enforce_failure_policy=False,
            )
            required_ok = False
        contract_healthy = bool(contract.get("healthy", False)) if isinstance(contract, dict) else False
        subsystem_ok = bool(health.get("all_ok", False))
        conversation_lane = _collect_conversation_lane_status()
        conversation_ready = bool(conversation_lane.get("conversation_ready", False))
        conversation_busy = _conversation_lane_is_busy(conversation_lane)
        conversation_working = bool(conversation_busy)
        conversation_standby = _conversation_lane_is_standby(conversation_lane)
        conversation_state = str(conversation_lane.get("state", "unknown") or "unknown").lower()
        contract_status = str(contract.get("status", "unknown") if isinstance(contract, dict) else "unknown")
        shutdown_active = is_shutdown_requested()
        boot_grace_active = uptime <= _health_pulse_boot_grace_s()
        core_boot_warming = (
            boot_grace_active
            and not required_ok
            and not shutdown_active
            and conversation_state in {"cold", "warming", "recovering", "standby", "unknown", ""}
        )
        conversation_boot_warming = (
            boot_grace_active
            and not shutdown_active
            and not conversation_ready
            and _conversation_lane_is_boot_warming(conversation_lane)
        )
        conversation_grace_standby = (
            boot_grace_active
            and not shutdown_active
            and required_ok
            and subsystem_ok
            and not conversation_ready
            and conversation_standby
        )
        boot_warming = core_boot_warming or conversation_boot_warming or conversation_grace_standby
        if shutdown_active:
            probe_status = "SHUTDOWN"
            conversation_status = "STOPPING"
        elif core_boot_warming:
            probe_status = "WARMING"
            conversation_status = "STANDBY" if conversation_standby else "WARMING"
        elif conversation_boot_warming:
            probe_status = "PASS" if required_ok else "WARMING"
            conversation_status = "WARMING"
        elif conversation_grace_standby:
            probe_status = "PASS"
            conversation_status = "STANDBY"
        else:
            probe_status = "PASS" if required_ok else "FAIL"
            conversation_status = (
                "PASS"
                if conversation_ready
                else "WORKING"
                if conversation_busy
                else "STANDBY"
                if conversation_standby
                else "FAIL"
            )
        if contract_healthy and required_ok and subsystem_ok and conversation_ready:
            runtime_status = contract_status.upper()
            subsystem_status = "PASS"
        elif contract_healthy and required_ok and subsystem_ok and conversation_working:
            runtime_status = "WORKING"
            subsystem_status = "PASS"
        elif shutdown_active:
            runtime_status = "SHUTTING_DOWN"
            subsystem_status = "PASS" if subsystem_ok else "STOPPING"
        elif boot_warming:
            runtime_status = "BOOTING"
            subsystem_status = "PASS" if subsystem_ok else "WARMING"
        elif not required_ok:
            runtime_status = "CRITICAL"
            subsystem_status = "FAIL" if not subsystem_ok else "PASS"
        elif contract_status.lower() in {"dead", "critical"}:
            runtime_status = contract_status.upper()
            subsystem_status = "FAIL" if not subsystem_ok else "PASS"
        else:
            runtime_status = "DEGRADED"
            subsystem_status = "FAIL" if not subsystem_ok else "PASS"
        summary = (
            f"Runtime: {runtime_status} | Required probes: {probe_status} | "
            f"Subsystem audit: {subsystem_status} | Conversation: {conversation_status} | "
            f"Heartbeats: {active_count}/{len(self.SUBSYSTEMS)} ACTIVE"
        )
        if core_boot_warming:
            lines.append("  ⏳ boot: required runtime probes are still warming.")
        elif conversation_boot_warming:
            lines.append("  ⏳ boot: conversation lane is still warming.")
        elif conversation_grace_standby:
            lines.append("  ⏳ boot: conversation lane is standing by for the first visible turn.")
        if not shutdown_active and not boot_warming and (not contract_healthy or not required_ok):
            failures = contract.get("failures", {}) if isinstance(contract, dict) else {}
            for tier in ("critical", "important"):
                for failure in failures.get(tier, [])[:3]:
                    if not isinstance(failure, dict):
                        continue
                    name = failure.get("container_key") or failure.get("name") or "unknown"
                    reason = failure.get("error") or failure.get("liveness") or "failed"
                    lines.append(f"  ❌ contract/{tier}: {name} ({reason})")
        if required_ok and contract_healthy and not subsystem_ok and not shutdown_active:
            lines.append(
                "  ❌ subsystem_audit: required subsystem heartbeat contract not satisfied"
            )
        if not (conversation_ready or conversation_working) and not shutdown_active and not boot_warming:
            reason = str(conversation_lane.get("last_failure_reason", "") or "conversation lane unavailable")
            lines.append(
                f"  ❌ conversation_lane: {conversation_state} ({reason})"
            )
        if stale_count:
            summary += f" | ⚠️ {stale_count} STALE"
        if degraded_count:
            summary += f" | ⚠️ {degraded_count} DEGRADED"
        if missing_count:
            summary += f" | ❌ {missing_count} MISSING"
        
        lines.insert(2, summary)
        lines.append("═══════════════════════════")
        
        return "\n".join(lines)
