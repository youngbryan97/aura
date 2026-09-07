"""Runtime Health Contract — defines what MUST be alive for Aura to be considered healthy.

This module is the authoritative source of truth for:
1. Which services are CRITICAL (system halts/degrades if missing)
2. Which services are IMPORTANT (system works but impaired)
3. Which services are OPTIONAL (nice-to-have background enrichments)

The contract is enforced at boot (by StartupValidator) and at runtime
(by the health monitor). Any module can call `evaluate_health()` to get
a typed HealthVerdict with clear pass/fail semantics.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.health_fragments import collect_health_fragments
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.HealthContract")

HEALTH_CONTRACT_VERSION = "runtime-health-v1"

REQUIRED_HEALTH_PROBE_GROUPS: dict[str, tuple[str, ...]] = {
    "kernel": ("kernel_interface",),
    "inference": (
        "inference_gate",
        "llm_router",
        "lane_admission",
        "lane_reconciler",
    ),
    "memory": (
        "state_repository",
        "memory_facade",
        "memory_write_gateway",
        "unified_memory_pressure",
        "external_memory_sentinel",
    ),
    "scheduler": (
        "scheduler",
        "runtime_control_plane",
        "resource_admission",
        "actor_supervision",
    ),
    "tool_governance": ("unified_will", "authority_gateway", "capability_engine"),
    "workspace": ("inhibition_manager", "global_workspace"),
    "attention": ("attention_schema",),
    "live_mind": ("live_mind_runtime",),
}


class ServiceTier(StrEnum):
    """How critical is this service to Aura's operation?"""

    CRITICAL = "critical"  # System CANNOT function without it
    IMPORTANT = "important"  # System works but user experience is degraded
    OPTIONAL = "optional"  # Background enrichment, loss is invisible to user


@dataclass(frozen=True)
class ServiceRequirement:
    """A single service that Aura depends on."""

    name: str
    container_key: str
    tier: ServiceTier
    description: str
    liveness_check: str | None = None  # Method name to call for deep health check
    # Method name returning EVIDENCE OF WORK — a monotonically increasing count
    # of real invocations, or a unix timestamp of the last one. Distinct from
    # liveness_check on purpose: `is_ready()` answers "could this work?", and a
    # component that is constructed, registered, healthy and wired to nothing
    # answers yes. Steering did exactly that while its callbacks never reached
    # the token sentinel, and the health report said it was fine because the
    # report was asking the only question that could not detect the fault.
    participation_check: str | None = None
    # Optional method returning a human reason for a FAILED liveness check.
    # Without it the contract can only report "<check>() returned False", which
    # names the probe and not the fault — measured live as
    # "hypervisor (is_alive() returned False)" for a watchdog that was running
    # fine and merely reporting unrecovered event-loop lag. That message sends
    # every investigation at thread liveness instead of at the real cause.
    liveness_reason_check: str | None = None
    # Optional method that REPAIRS a failed liveness check, called only after
    # the status has been recorded and only by callers that asked for repair.
    #
    # It exists because the repair used to live inside the liveness probe
    # itself: every reader of the health report — a dashboard, a status
    # endpoint, a contract sweep — silently restarted the service it was
    # inspecting, and none of them knew they had. Observation is not
    # actuation. The probe reports; this repairs, separately and on purpose.
    recovery_check: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# THE CONTRACT: What must be alive?
# ═══════════════════════════════════════════════════════════════════════

RUNTIME_CONTRACT: list[ServiceRequirement] = [
    # ── CRITICAL: Without these, Aura cannot think or respond ──
    ServiceRequirement(
        "InferenceGate",
        "inference_gate",
        ServiceTier.CRITICAL,
        "Routes LLM requests to local MLX or cloud. Without it, Aura cannot generate any response.",
        liveness_check="is_inference_ready",
    ),
    ServiceRequirement(
        "LLM Router",
        "llm_router",
        ServiceTier.CRITICAL,
        "Selects model tier and provider. Without it, InferenceGate has no backend.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "State Repository",
        "state_repository",
        ServiceTier.CRITICAL,
        "Persistent state store. Without it, Aura has no memory between turns.",
        liveness_check="is_initialized",
    ),
    ServiceRequirement(
        "Memory Facade",
        "memory_facade",
        ServiceTier.CRITICAL,
        "Canonical memory gateway. Without it, Aura cannot safely read or write long-term memory.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Memory Write Gateway",
        "memory_write_gateway",
        ServiceTier.CRITICAL,
        "Canonical governed durable memory write gateway. Without it, memory writes cannot be trusted.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Kernel Interface",
        "kernel_interface",
        ServiceTier.CRITICAL,
        "Bridge between orchestrator and consciousness kernel.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Scheduler",
        "scheduler",
        ServiceTier.CRITICAL,
        "Canonical runtime scheduler. Without it, maintenance, repair, and background work are unsupervised.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Runtime Control Plane",
        "runtime_control_plane",
        ServiceTier.CRITICAL,
        "Canonical desired-state reconciler. Without it, service lifecycle and resource policy diverge.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Resource Admission",
        "resource_admission",
        ServiceTier.CRITICAL,
        "Pressure-aware lease authority for inference, evolution, model loading, and managed startup.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Lane Admission",
        "lane_admission",
        ServiceTier.CRITICAL,
        "Declared model-memory envelope. Without it, concurrent lane warmups can over-commit the host.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Lane Reconciler",
        "lane_reconciler",
        ServiceTier.CRITICAL,
        "Managed cortex convergence and crash-loop backoff. Without it, model-serving recovery can thrash indefinitely.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Actor Supervision",
        "actor_supervision",
        ServiceTier.CRITICAL,
        "Canonical multiprocessing actor monitor. Without it, crashed or stalled actors are not converged safely.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Inhibition Manager",
        "inhibition_manager",
        ServiceTier.CRITICAL,
        "Canonical workspace safety gate. Without it, candidate admission cannot be trusted.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Global Workspace",
        "global_workspace",
        ServiceTier.CRITICAL,
        "Canonical candidate admission and broadcast lane. Without it, inhibition cannot bind cognition.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Attention Schema",
        "attention_schema",
        ServiceTier.CRITICAL,
        "Canonical attentional-focus owner and rigidity gate.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Unified Will",
        "unified_will",
        ServiceTier.CRITICAL,
        "Single locus of authority for consequential decisions.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Authority Gateway",
        "authority_gateway",
        ServiceTier.CRITICAL,
        "Governance gateway for tools, external I/O, memory writes, state changes, and self-modification.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Capability Engine",
        "capability_engine",
        ServiceTier.CRITICAL,
        "Capability-token and skill governance layer. Without it, tool execution cannot be considered healthy.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Output Gate",
        "output_gate",
        ServiceTier.CRITICAL,
        "Delivers responses to the user. Without it, Aura thinks but cannot speak.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Live Mind Runtime",
        "live_mind_runtime",
        ServiceTier.CRITICAL,
        "Boot-owned causal organs and snapshot contract required for grounded live desktop speech.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "External Memory Sentinel",
        "external_memory_sentinel",
        ServiceTier.CRITICAL,
        "Out-of-process memory guard. Without it, a live desktop runaway can outpace in-process watchdogs and crash the host.",
        liveness_check="is_armed",
    ),
    # ── IMPORTANT: Aura works but is impaired without these ──
    ServiceRequirement(
        "Event Bus",
        "event_bus",
        ServiceTier.IMPORTANT,
        "Canonical runtime event transport. Without it, subsystems cannot reliably coordinate.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Ulysses Covenant",
        "ulysses_covenant",
        ServiceTier.IMPORTANT,
        "Volitional self-binding registry enforced at the Will. Without it, "
        "precommitments against known failure modes stop holding.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Cognitive Engine",
        "cognitive_engine",
        ServiceTier.IMPORTANT,
        "Manages cognitive state transitions and working memory.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Affect Engine",
        "affect_engine",
        ServiceTier.IMPORTANT,
        "Emotional state management. Without it, responses are emotionally flat.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Compute Orchestrator",
        "compute_orchestrator",
        ServiceTier.IMPORTANT,
        "Resource allocation and thermal pressure control. Without it, long-run survival degrades.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Database Coordinator",
        "database_coordinator",
        ServiceTier.IMPORTANT,
        "SQLite connection pool. Without it, persistent storage degrades.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Drive Engine",
        "drive_engine",
        ServiceTier.IMPORTANT,
        "Motivation and goal management. Without it, autonomous behavior stops.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Agency Core",
        "agency_core",
        ServiceTier.IMPORTANT,
        "Canonical autonomous agency pathway loop. Without it, initiative and swarm tool use degrade.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Lymphatic Reaper",
        "reaper",
        ServiceTier.IMPORTANT,
        "Long-run maintenance supervisor. Without it, stale processes and files accumulate.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Hypervisor",
        "hypervisor",
        ServiceTier.IMPORTANT,
        "Event-loop and memory watchdog. Without it, severe stalls can go undetected.",
        liveness_check="is_alive",
        liveness_reason_check="liveness_failure_reason",
    ),
    ServiceRequirement(
        "Event Loop Monitor",
        "event_loop_monitor",
        ServiceTier.IMPORTANT,
        "Fine-grained event-loop lag monitor. Without it, blocking regressions are harder to catch.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "MindTick",
        "mind_tick",
        ServiceTier.IMPORTANT,
        "Canonical cognitive and organism rhythm. Without forward progress, autonomous state integration stalls.",
        liveness_check="is_alive",
        recovery_check="ensure_alive",
    ),
    ServiceRequirement(
        "Resource Governor",
        "resource_governor",
        ServiceTier.IMPORTANT,
        "Canonical sampler and eviction adapter feeding the runtime control plane.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Resource Arbitrator",
        "resource_arbitrator",
        ServiceTier.IMPORTANT,
        "Compatibility facade ensuring legacy inference and evolution callers use canonical admission.",
        liveness_check="is_ready",
    ),
    # ── OPTIONAL: Background enrichments ──
    ServiceRequirement(
        "Whole-System Φ",
        "whole_system_phi",
        ServiceTier.OPTIONAL,
        "Integrated-information estimation over the live channel set "
        "(exact-MIP Gaussian Φ with surrogate nulls, grain discovery, and "
        "an internal PCI). Telemetry-grade; loss removes a measurement, not "
        "a capability.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Mycelial Network",
        "mycelial_network",
        ServiceTier.OPTIONAL,
        "Infrastructure graph and pathway routing.",
    ),
    ServiceRequirement(
        "Voice Engine",
        "voice_engine",
        ServiceTier.OPTIONAL,
        "Speech-to-text and text-to-speech capabilities.",
    ),
    ServiceRequirement(
        "Liquid Substrate",
        "liquid_substrate",
        ServiceTier.OPTIONAL,
        "Dynamic emotional substrate for consciousness simulation.",
    ),
    ServiceRequirement(
        "Synaptic Plasticity",
        "synaptic_plasticity",
        ServiceTier.OPTIONAL,
        "Bounded online projection learning for generation-style modulation.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Temporal Continuity",
        "temporal_continuity",
        ServiceTier.OPTIONAL,
        "Accumulated silence and drift residue for temporal presence.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Attention Gate",
        "attention_gate",
        ServiceTier.OPTIONAL,
        "Causal context pruning for focused cognition.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Somatic Qualia",
        "somatic_qualia",
        ServiceTier.OPTIONAL,
        "Non-symbolic substrate perturbation for generation controls.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Swarm Protocol",
        "swarm_protocol",
        ServiceTier.OPTIONAL,
        "Multi-agent debate and reasoning.",
    ),
    ServiceRequirement(
        "Agent Delegator",
        "agent_delegator",
        ServiceTier.OPTIONAL,
        "Coordinates parallel task execution and specialized agents.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Stability Guardian",
        "stability_guardian",
        ServiceTier.OPTIONAL,
        "Health monitoring and auto-recovery.",
    ),
    ServiceRequirement(
        "Metrics Exporter",
        "metrics_exporter",
        ServiceTier.OPTIONAL,
        "Prometheus metrics endpoint.",
    ),
    ServiceRequirement(
        "Allostasis Engine",
        "allostasis_engine",
        ServiceTier.OPTIONAL,
        "Predictive interoception: forecasts vital-sign trajectories and regulates before crises, with a calibration ledger.",
        liveness_check="is_ready",
    ),
]

UNIFIED_MEMORY_PRESSURE_REQUIREMENT = ServiceRequirement(
    "Unified Memory Pressure",
    "unified_memory_pressure",
    ServiceTier.CRITICAL,
    "Process-wide unified-memory pressure gate. Aura must not claim healthy when the live model lane risks system OOM.",
    liveness_check="get_memory_pressure_snapshot",
)

UNIFIED_RUNTIME_PRESSURE_REQUIREMENT = ServiceRequirement(
    "Unified Runtime Pressure",
    "unified_runtime_pressure",
    ServiceTier.IMPORTANT,
    (
        "Event-loop, CPU, and existential-pressure gate. Aura must not claim "
        "healthy when scheduling lag or substrate survival pressure is high."
    ),
    liveness_check="runtime_pressure_snapshot",
)


class HealthLevel(StrEnum):
    """Overall system health classification."""

    HEALTHY = "healthy"  # All critical + important services alive
    DEGRADED = "degraded"  # All critical alive, some important missing
    CRITICAL = "critical"  # Some critical services missing
    DEAD = "dead"  # Cannot function at all


@dataclass
class ServiceStatus:
    """Runtime status of a single service."""

    requirement: ServiceRequirement
    present: bool
    liveness_ok: bool | None = None  # None = no liveness check defined
    error: str | None = None
    duration_ms: float = 0.0
    #: Evidence of real work, or None when the service declares no probe.
    #: A count or a last-used timestamp — whatever the service can honestly
    #: produce.
    participation: float | None = None

    @property
    def has_participated(self) -> bool | None:
        """Has this service actually done anything? None when unmeasured.

        `None` is a real answer and must not be read as `False`. A service with
        no participation probe is UNMEASURED, and reporting unmeasured as
        "never used" would replace one false certainty with another.
        """
        if self.participation is None:
            return None
        return self.participation > 0

    @property
    def state(self) -> str:
        """absent | idle | participating | unmeasured — never just "ok"."""
        if not self.present:
            return "absent"
        participated = self.has_participated
        if participated is None:
            return "unmeasured"
        return "participating" if participated else "idle"


@dataclass
class HealthVerdict:
    """Result of a health evaluation."""

    level: HealthLevel
    services: list[ServiceStatus]
    timestamp: float = field(default_factory=time.time)
    evaluation_duration_ms: float = 0.0

    @property
    def is_operational(self) -> bool:
        """Can Aura function at all?"""
        return self.level in (HealthLevel.HEALTHY, HealthLevel.DEGRADED)

    @property
    def critical_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.CRITICAL
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def important_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.IMPORTANT
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def optional_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.OPTIONAL
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def status_code(self) -> int:
        services = {
            status.requirement.container_key: _service_status_payload(status)
            for status in self.services
        }
        required_probes = _required_probe_status_from_services(services)
        return 200 if self.is_operational and required_probe_groups_pass(required_probes) else 503

    def summary(self) -> str:
        lines = [f"Health: {self.level.value.upper()}"]
        for s in self.services:
            icon = "✓" if s.present and s.liveness_ok is not False else "✗"
            tier = s.requirement.tier.value[0].upper()
            lines.append(
                f"  [{icon}] [{tier}] {s.requirement.name}: "
                f"{'alive' if s.present else 'MISSING'}"
                f"{' (liveness FAIL: ' + (s.error or '') + ')' if s.liveness_ok is False else ''}"
            )
        return "\n".join(lines)

    def to_report(self) -> dict[str, Any]:
        """Canonical machine-readable runtime health report."""
        services = [_service_status_payload(status) for status in self.services]
        report_services = {
            payload["container_key"]: payload
            for payload in services
            if isinstance(payload.get("container_key"), str)
        }
        tier_summary = {
            tier.value: _tier_summary(self.services, tier)
            for tier in (ServiceTier.CRITICAL, ServiceTier.IMPORTANT, ServiceTier.OPTIONAL)
        }
        required_probes = _required_probe_status_from_services(report_services)
        required_probe_ok = required_probe_groups_pass(required_probes)
        probe_blockers = required_probe_blockers(required_probes)
        healthy = self.level == HealthLevel.HEALTHY and required_probe_ok
        operational = self.is_operational and required_probe_ok
        return {
            "contract_version": HEALTH_CONTRACT_VERSION,
            "status": self.level.value,
            "healthy": healthy,
            "operational": operational,
            "status_code": 200 if operational else 503,
            "timestamp_unix": self.timestamp,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "required_probes": required_probes,
            "probe_blockers": probe_blockers,
            "tier_summary": tier_summary,
            "failures": {
                "critical": [_service_status_payload(status) for status in self.critical_failures],
                "important": [
                    _service_status_payload(status) for status in self.important_failures
                ],
                "optional": [_service_status_payload(status) for status in self.optional_failures],
            },
            "services": services,
        }


def _service_probe_ok(service: dict[str, Any] | None) -> bool:
    if not isinstance(service, dict):
        return False
    if not bool(service.get("present", False)):
        return False
    return str(service.get("liveness", "") or "") == "ok"


def _required_probe_status_from_services(
    services_by_key: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items():
        component_status = {
            key: _service_probe_ok(services_by_key.get(key))
            for key in keys
        }
        probes[group] = {
            "ok": all(component_status.values()),
            "components": component_status,
        }
    probes["all_passed"] = all(
        bool(value.get("ok", False))
        for value in probes.values()
        if isinstance(value, dict)
    )
    return probes


def required_probe_groups_pass(required_probes: Any) -> bool:
    """Return True only when every canonical readiness group explicitly passes.

    This is stricter than trusting ``all_passed`` because heartbeat consumers
    must fail closed on malformed, partial, or transport-only payloads.
    """
    if not isinstance(required_probes, dict):
        return False
    if not bool(required_probes.get("all_passed", False)):
        return False
    for group_name, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        group = required_probes.get(group_name)
        if not isinstance(group, dict) or not bool(group.get("ok", False)):
            return False
        components = group.get("components")
        if not isinstance(components, dict):
            return False
        for component in expected_components:
            if components.get(component) is not True:
                return False
    return True


def required_probe_blockers(required_probes: Any) -> list[str]:
    """Return canonical blockers for malformed or failing required probes."""

    if not isinstance(required_probes, dict):
        return ["runtime_required_probes"]

    blockers: list[str] = []
    if not required_probe_groups_pass(required_probes):
        blockers.append("runtime_required_probes")

    for group_name, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        group = required_probes.get(group_name)
        if not isinstance(group, dict):
            blockers.append(f"probe:{group_name}")
            continue
        if not bool(group.get("ok", False)):
            blockers.append(f"probe:{group_name}")
            continue
        components = group.get("components")
        if not isinstance(components, dict):
            blockers.append(f"probe:{group_name}")
            continue
        if any(components.get(component) is not True for component in expected_components):
            blockers.append(f"probe:{group_name}")

    return list(dict.fromkeys(blockers))


def required_probe_status(report: dict[str, Any]) -> dict[str, Any]:
    """Return canonical high-level readiness probes from a health report.

    A heartbeat is allowed to claim healthy only when every group here passes:
    kernel, inference, memory, scheduler, and tool governance.
    """
    services = report.get("services", []) if isinstance(report, dict) else []
    services_by_key = {
        str(service.get("container_key")): service
        for service in services
        if isinstance(service, dict) and service.get("container_key")
    }
    return _required_probe_status_from_services(services_by_key)


def _service_status_payload(status: ServiceStatus) -> dict[str, Any]:
    requirement = status.requirement
    liveness = "not_configured"
    if status.liveness_ok is True:
        liveness = "ok"
    elif status.liveness_ok is False:
        liveness = "failed"
    return {
        "name": requirement.name,
        "container_key": requirement.container_key,
        "tier": requirement.tier.value,
        "description": requirement.description,
        "present": status.present,
        "liveness": liveness,
        "liveness_check": requirement.liveness_check,
        "error": status.error,
        "duration_ms": status.duration_ms,
    }


def _liveness_failure_reason(svc: Any, requirement: ServiceRequirement) -> str:
    """Ask a service to explain its own failed liveness check.

    Best-effort and never fatal: a service that cannot explain itself keeps the
    generic message rather than turning a health probe into an exception.
    """

    name = requirement.liveness_reason_check
    if not name or svc is None:
        return ""
    try:
        explain = getattr(svc, name, None)
        if not callable(explain):
            return ""
        reason = explain()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    if inspect.isawaitable(reason):
        close = getattr(reason, "close", None)
        if callable(close):
            close()
        return ""
    text = str(reason or "").strip()
    return text[:300]


def _coerce_liveness_result(result: Any) -> tuple[bool, str | None]:
    """Accept only explicit liveness success values.

    Health probes are a runtime launch contract, not a generic truthiness check.
    A coroutine object, non-empty string, list, or arbitrary object must never
    make Aura look healthy.
    """
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        return False, "liveness check returned awaitable; sync health contract cannot count it as ready"
    if isinstance(result, bool):
        return result, None if result else "liveness check returned False"
    if isinstance(result, dict):
        for key in ("ok", "ready", "healthy", "alive", "operational"):
            if key in result:
                return bool(result.get(key) is True), None if result.get(key) is True else f"{key} was not True"
    if result is None:
        return False, "liveness check returned None"
    return False, f"unsupported liveness result type: {type(result).__name__}"


def _unified_memory_pressure_status() -> ServiceStatus:
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if snapshot.critical:
            return ServiceStatus(
                requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
                present=True,
                liveness_ok=False,
                error=snapshot.reason or f"memory pressure level is {snapshot.level}",
            )
        return ServiceStatus(
            requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
            present=True,
            liveness_ok=True,
            error=None,
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return ServiceStatus(
            requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
            present=True,
            liveness_ok=False,
            error=f"memory pressure probe unavailable: {exc}",
        )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _process_uptime_seconds() -> float:
    """Seconds since the runtime booted, or 0.0 if unknown.

    Reads the orchestrator's start_time from the runtime registry. Returns 0.0 when
    no orchestrator is registered (e.g. unit tests), so the boot-grace
    exemption below applies ONLY to a live runtime that is genuinely warming
    up — never to a bare-probe unit test. Used to suppress runtime-pressure
    health failures during boot/warmup, when loading the local model
    legitimately spikes event-loop lag and survival pressure (the same window
    the freeze watchdog already exempts).
    """
    try:
        orch = get_runtime_service("orchestrator", default=None)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 0.0
    for candidate in (
        getattr(orch, "start_time", None),
        getattr(getattr(orch, "status", None), "start_time", None),
    ):
        try:
            start = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if start > 0.0:
            return max(0.0, time.time() - start)
    return 0.0


def _runtime_pressure_boot_grace_active() -> bool:
    """Return True only during an explicit boot/proof warmup grace window."""

    boot_context = any(
        str(os.environ.get(name, "") or "").strip().lower()
        not in {"", "0", "false", "no", "off"}
        for name in (
            "AURA_PROOF_RUN",
            "AURA_SAFE_BOOT_DESKTOP",
            "AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE",
        )
    )
    if not boot_context:
        return False

    boot_grace_s = _float_env("AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE_S", 180.0)
    uptime_s = _process_uptime_seconds()
    return bool(boot_grace_s > 0.0 and 0.0 < uptime_s < boot_grace_s)


def _recent_inference_degradation_blocks_runtime_pressure(record: Any) -> tuple[bool, str]:
    """Classify recent inference degradations for the runtime-pressure gate.

    The runtime health contract should fail closed for foreground/user-facing
    inference saturation, but a background Brainstem timeout must not keep the
    launched desktop stuck in "booting/degraded" after Cortex and the required
    probes are ready. The degradation remains logged and repair-routable; this
    function only decides whether it blocks the top-level readiness contract.
    """

    severity = str(getattr(record, "severity", "") or "")
    if severity not in {"critical", "degraded"}:
        return False, ""

    action = str(getattr(record, "action", "") or "")
    message = str(getattr(record, "error_message", "") or "")
    combined = f"{message} {action}".lower()

    if "generation gate saturated" in combined or "refused to stack" in combined:
        if "background" in combined and not any(
            marker in combined
            for marker in ("foreground", "user-facing", "user_facing")
        ):
            return False, "background_generation_contention"
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_saturation: {(message or action)[:120]}"

    # Known background lane timeout. In live mode this may be escalated by the
    # fail-closed service policy even when the user-facing Cortex lane is fine.
    if (
        "inference_gate_generation_timeout:brainstem:" in combined
        and "foreground" not in combined
        and "user-facing" not in combined
        and "user_facing" not in combined
    ):
        return False, "background_brainstem_timeout"

    if any(marker in combined for marker in (
        "inference_gate_generation_timeout:cortex:",
        "inference_gate_generation_timeout:solver:",
        "user-facing",
        "user_facing",
        "foreground",
        "client_returned_no_text",
    )):
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_{severity}: {(message or action)[:120]}"

    # Critical/degraded inference failures without lane context are still
    # treated as blocking because they may represent the active conversation path.
    if severity == "critical" and "brainstem" not in combined and "reflex" not in combined:
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_{severity}: {(message or action)[:120]}"

    return False, ""


def _runtime_pressure_status() -> ServiceStatus:
    """Return an important health failure when runtime pressure is too high.

    Heartbeat transport, service presence, and memory headroom are necessary
    but not sufficient for a healthy live desktop runtime. A system stuck in a
    long foreground generation can still have every service "alive"; this probe
    closes that gap by folding existential pressure and lag monitors into the
    canonical health contract.
    """
    blockers: list[str] = []
    boot_deferrable_blockers: set[str] = set()
    details: list[str] = []
    threat_threshold = _float_env("AURA_HEALTH_EXISTENTIAL_THREAT_UNHEALTHY", 0.75)
    lag_threshold = _float_env("AURA_HEALTH_EVENT_LOOP_LAG_UNHEALTHY_S", 5.0)
    recent_degradation_window_s = _float_env(
        "AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S",
        180.0,
    )
    boot_grace_active = _runtime_pressure_boot_grace_active()

    try:
        stakes = get_runtime_service("existential_stakes", default=None)
        status_getter = getattr(stakes, "get_status", None)
        if callable(status_getter):
            status = status_getter()
            if isinstance(status, dict):
                threat = float(status.get("existential_threat", 0.0) or 0.0)
                lag_threat = float(status.get("lag_threat", 0.0) or 0.0)
                memory_threat = float(status.get("memory_threat", 0.0) or 0.0)
                details.append(
                    "existential_threat="
                    f"{threat:.2f}, lag_threat={lag_threat:.2f}, memory_threat={memory_threat:.2f}"
                )
                # Steady-state memory pressure is owned by the dedicated
                # _unified_memory_pressure_status probe (calibrated via
                # get_memory_pressure_snapshot, flags only snapshot.critical)
                # and the out-of-band memory watchdog. A loaded ~20GB model on a
                # 64GB box sits at memory_threat ~0.77 while serving requests
                # fine, so folding raw existential memory pressure into THIS
                # probe double-counts it and would mark a healthy runtime
                # degraded. This probe targets the *stuck / overloaded* runtime
                # the docstring describes, which lag_threat captures. So when
                # memory is the high contributor, gate on the lag signal and let
                # the dedicated memory probe decide; otherwise use the full
                # aggregate (covers lag- and cpu-driven existential threat).
                pressure_threat = (
                    lag_threat if memory_threat >= threat_threshold else threat
                )
                if pressure_threat >= threat_threshold:
                    blocker = f"existential_threat {threat:.2f} >= {threat_threshold:.2f}"
                    blockers.append(blocker)
                    if boot_grace_active and lag_threat >= threat_threshold:
                        boot_deferrable_blockers.add(blocker)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        details.append(f"existential_stakes_unavailable:{type(exc).__name__}")

    for key in ("event_loop_monitor", "hypervisor"):
        try:
            monitor = get_runtime_service(key, default=None)
            status_getter = getattr(monitor, "get_status", None)
            if not callable(status_getter):
                continue
            status = status_getter()
            if not isinstance(status, dict):
                continue
            last_lag = float(status.get("last_lag_s", 0.0) or 0.0)
            failure_reason = str(status.get("last_failure_reason", "") or "")
            sample_fresh = status.get("sample_fresh")
            if sample_fresh is False:
                details.append(
                    f"{key}.lag_sample_stale age_s="
                    f"{float(status.get('sample_age_s', 0.0) or 0.0):.2f}"
                )
            else:
                details.append(f"{key}.last_lag_s={last_lag:.2f}")
            if failure_reason:
                blockers.append(f"{key}:{failure_reason}")
            elif sample_fresh is not False and last_lag >= lag_threshold:
                blocker = f"{key}.last_lag_s {last_lag:.2f} >= {lag_threshold:.2f}"
                blockers.append(blocker)
                if boot_grace_active:
                    boot_deferrable_blockers.add(blocker)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            details.append(f"{key}_pressure_unavailable:{type(exc).__name__}")

    try:
        from core.runtime.errors import get_degradation_tracker

        now = time.time()
        inference_subsystems = {
            "llm_health_router",
            "inference_gate",
            "mlx_client",
            "mlx_runtime",
        }
        for record in get_degradation_tracker().recent(limit=80):
            subsystem = str(getattr(record, "subsystem", "") or "")
            if subsystem not in inference_subsystems:
                continue
            age_s = now - float(getattr(record, "timestamp", 0.0) or 0.0)
            if age_s > recent_degradation_window_s:
                continue
            blocks, reason = _recent_inference_degradation_blocks_runtime_pressure(record)
            if blocks and reason:
                blockers.append(reason)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        details.append(f"degradation_pressure_unavailable:{type(exc).__name__}")

    if blockers:
        active_blockers = [
            blocker for blocker in blockers if blocker not in boot_deferrable_blockers
        ]
        if not active_blockers and boot_deferrable_blockers:
            details.append(
                "boot_grace_deferred_runtime_pressure:"
                + ",".join(sorted(boot_deferrable_blockers))
            )
            blockers = []
        else:
            blockers = active_blockers

    return ServiceStatus(
        requirement=UNIFIED_RUNTIME_PRESSURE_REQUIREMENT,
        present=True,
        liveness_ok=not blockers,
        error="; ".join(blockers or details[:3]) if blockers else None,
    )


def _read_participation(service: Any, requirement: ServiceRequirement) -> float | None:
    """Evidence that this service has actually done work, or None if unmeasured.

    Never raises and never guesses. A probe that errors, is missing, or returns
    something that is not a number yields None — UNMEASURED — because the one
    thing this must not do is report "never used" for a service that simply
    has no probe. That would trade a false "everything is fine" for a false
    "everything is broken", and the second gets the check switched off faster
    than the first.
    """
    probe_name = getattr(requirement, "participation_check", None)
    if not probe_name or service is None:
        return None
    try:
        # The getattr is inside the guard too. `getattr(x, name, None)` only
        # swallows AttributeError, and a property that raises anything else
        # would propagate out of the health probe — taking down the report
        # whose job is to tell you what is wrong.
        probe = getattr(service, probe_name, None)
        if not callable(probe):
            return None
        value = probe()
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _tier_summary(services: list[ServiceStatus], tier: ServiceTier) -> dict[str, int]:
    tier_services = [status for status in services if status.requirement.tier == tier]
    failed = [
        status for status in tier_services if not status.present or status.liveness_ok is False
    ]
    liveness_failed = [status for status in tier_services if status.liveness_ok is False]
    missing = [status for status in tier_services if not status.present]
    return {
        "total": len(tier_services),
        "present": len(tier_services) - len(missing),
        "missing": len(missing),
        "liveness_failed": len(liveness_failed),
        "failed": len(failed),
    }


def recover_failed_services(verdict: HealthVerdict) -> dict[str, str]:
    """Repair services whose liveness failed, for callers that asked to.

    Deliberately NOT part of evaluate_health. The repair used to live inside
    the liveness probe, so every reader of the health report restarted the
    service it was inspecting without knowing it. A caller that wants
    self-healing calls this and gets a record of what it did.
    """

    outcomes: dict[str, str] = {}
    for status in verdict.services:
        requirement = status.requirement
        if status.liveness_ok is not False or not requirement.recovery_check:
            continue
        service = get_runtime_service(requirement.container_key, default=None)
        recover = getattr(service, requirement.recovery_check, None)
        if not callable(recover):
            outcomes[requirement.name] = "no_recovery_hook"
            continue
        try:
            outcomes[requirement.name] = "recovered" if recover() else "unrecovered"
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            outcomes[requirement.name] = f"failed:{type(exc).__name__}"
            logger.warning(
                "Could not recover %s after a failed liveness check: %s",
                requirement.name,
                exc,
            )
    return outcomes


def evaluate_health() -> HealthVerdict:
    """Evaluate the runtime health contract against the live runtime service registry.

    A pure observation: it never throws, and it never repairs. Repair is
    ``recover_failed_services``, called by whoever wants it.
    """
    evaluation_started = time.perf_counter()
    statuses: list[ServiceStatus] = []

    for req in RUNTIME_CONTRACT:
        probe_started = time.perf_counter()
        try:
            svc = get_runtime_service(req.container_key, default=None)
            present = svc is not None

            liveness_ok = None
            error = None
            if present and req.liveness_check:
                try:
                    check_fn = getattr(svc, req.liveness_check, None)
                    if callable(check_fn):
                        result = check_fn()
                        liveness_ok, result_error = _coerce_liveness_result(result)
                        if not liveness_ok:
                            if result_error == "liveness check returned False":
                                error = f"{req.liveness_check}() returned False"
                            else:
                                error = result_error or f"{req.liveness_check}() did not return explicit True"
                            explained = _liveness_failure_reason(svc, req)
                            if explained:
                                error = explained
                    else:
                        liveness_ok = False
                        error = f"missing liveness check: {req.liveness_check}()"
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    liveness_ok = False
                    error = str(exc)

            statuses.append(
                ServiceStatus(
                    requirement=req,
                    present=present,
                    liveness_ok=liveness_ok,
                    error=error,
                    participation=_read_participation(svc, req),
                    duration_ms=round(
                        (time.perf_counter() - probe_started) * 1000.0,
                        3,
                    ),
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ServiceStatus(
                    requirement=req,
                    present=False,
                    liveness_ok=False,
                    error=str(exc),
                    duration_ms=round(
                        (time.perf_counter() - probe_started) * 1000.0,
                        3,
                    ),
                )
            )

    statuses.append(_unified_memory_pressure_status())
    statuses.append(_runtime_pressure_status())

    # Classify
    concrete_statuses = [
        status
        for status in statuses
        if status.requirement.container_key != UNIFIED_MEMORY_PRESSURE_REQUIREMENT.container_key
    ]
    critical_alive = all(
        s.present and s.liveness_ok is not False
        for s in statuses
        if s.requirement.tier == ServiceTier.CRITICAL
    )
    important_alive = all(
        s.present and s.liveness_ok is not False
        for s in statuses if s.requirement.tier == ServiceTier.IMPORTANT
    )

    if critical_alive and important_alive:
        level = HealthLevel.HEALTHY
    elif critical_alive:
        level = HealthLevel.DEGRADED
    elif any(s.present for s in concrete_statuses if s.requirement.tier == ServiceTier.CRITICAL):
        level = HealthLevel.CRITICAL
    else:
        level = HealthLevel.DEAD

    return HealthVerdict(
        level=level,
        services=statuses,
        evaluation_duration_ms=round(
            (time.perf_counter() - evaluation_started) * 1000.0,
            3,
        ),
    )


def _attach_causal_evidence(block: dict[str, Any]) -> None:
    """Verdicts, decorative direct channels, and served-prose auditing.

    A separate function because the integrity block is already one of the
    outsized ones the method-size ratchet tracks, and growing it to add these
    would have been paying for a measurement with the thing the measurement
    is for.
    """

    # Which faculties have been shown to change the output, and which have only
    # been shown to run. Attached here for the same reason as taint: the health
    # verdict is green either way, and "the reply was produced by the full
    # architecture" is a claim the verdict cannot express or refute.
    try:
        from core.verify.causal_influence import get_influence_ledger
        from core.verify.lesion_registry import get_lesion_registry

        influence = get_influence_ledger().snapshot()
        lesions = get_lesion_registry().snapshot()
        block["causal_influence"] = influence
        block["lesionable_channels"] = lesions

        # The consumer. Until this existed, no code anywhere branched on a
        # verdict: `channel_is_influential()` was defined, exported, and called
        # zero times, so a channel could be measured INERT and nothing would
        # ever say so. Measurement without a consequence is an expensive way of
        # keeping a secret.
        #
        # A channel measured INERT is not automatically a defect — a
        # text-mediated channel may legitimately wash out. A channel declared
        # DIRECT_ACTUATION and measured INERT is different: it claims to reach
        # the sampler as a number, and the paired trials say removing it
        # changes nothing. That is a faculty that is running and not working,
        # which is exactly the condition this apparatus was built to surface.
        decorative: list[dict[str, Any]] = []
        registered = lesions.get("registered") or {}
        for name, entry in (influence.get("channels") or {}).items():
            if entry.get("verdict") != "inert":
                continue
            handle = registered.get(name) or {}
            if not handle.get("direct_actuation"):
                continue
            decorative.append(
                {
                    "channel": name,
                    "owner": handle.get("owner", ""),
                    "neutral": handle.get("neutral", ""),
                    "effect": entry.get("effect"),
                    "noise_floor": entry.get("noise_floor"),
                    "n_treatment": entry.get("n_treatment"),
                    "n_null": entry.get("n_null"),
                    "reason": entry.get("reason", ""),
                }
            )
        block["decorative_direct_channels"] = decorative
        if decorative:
            block["decorative_direct_channel_count"] = len(decorative)

        # How much of the apparatus has actually been used. Without this, a
        # campaign that is wired, admitted and never actually running looks
        # exactly like one that ran and found nothing: both report no
        # measured channels and no decorative ones, which reads as a clean
        # bill of health. The number that separates them is how many
        # registered channels have never been measured at all.
        measured = set(influence.get("channels") or {})
        never = sorted(set(registered) - measured)
        block["influence_channels_registered"] = len(registered)
        block["influence_channels_never_measured"] = len(never)
        if never:
            block["influence_channels_awaiting_evidence"] = never[:20]
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["causal_influence_error"] = repr(exc)

    # Served prose that outran the turn's own work record. The work ledger was
    # written on every tool call and read by nothing outside the validation
    # suite; this is the read, and it reports rather than decides.
    try:
        from core.verify.fabrication_watch import fabrication_snapshot

        block["fabrication_watch"] = fabrication_snapshot()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["fabrication_watch_error"] = repr(exc)


#: How big a graph is still worth walking end to end inside a health report.
#: The integrity walk is O(nodes x links) and this is served on a route; above
#: this the count is reported and the walk is skipped, which is a smaller lie
#: than a report nobody can afford to call.
_GRAPH_NODES_WORTH_WALKING = 5000


def _runtime_integrity_block() -> dict[str, Any]:
    """Memory the health verdict does not otherwise have.

    ``evaluate_health()`` answers "is the runtime working *now*", which is
    the right question but not the only one. A process that survived a
    lock-order violation, shed an organ under memory pressure, or hot-swapped
    code is working now *and* is no longer the process its green verdict
    describes. The kernel prints its taint on every oops for exactly this
    reason; this block is that line. It never flips the verdict — it
    attaches the caveat the verdict cannot express on its own.
    """
    block: dict[str, Any] = {}

    # The order in which she thinks, compiled and sealed.
    #
    # Three peer architectures make the same complaint from three directions:
    # Generative Agents puts the whole cycle in one function, Soar's decision
    # cycle is a state machine anyone can enumerate, and LangGraph refuses to
    # start when the topology does not resolve. Aura has the order, the
    # frequency and the per-phase contracts, and nothing joined them, so a
    # field two phases both write was found by watching it happen.
    #
    # The seal is what a receipt can carry: two runs of one commit agree, and
    # a phase added, removed, reordered or re-declared does not.
    try:
        from core.runtime.the_shape_of_one_turn import compile_the_cognition

        block["the_shape_of_one_turn"] = {
            mode: {
                key: value
                for key, value in compile_the_cognition(mode).to_dict().items()
                if key not in {"phases"}
            }
            for mode in ("foreground", "background")
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_shape_of_one_turn"] = {"error": repr(exc)}

    # One working memory, and whether anything still normalises it against a
    # number of its own. Three readers did, and each was pinned at its own
    # ceiling for most of a conversation — a constant that looked like a
    # measurement. Reported live so a new one cannot arrive unseen between
    # test runs.
    try:
        from core.state.one_working_memory import (
            THE_STORE,
            the_capacity,
            the_caps_that_disagree,
            who_else_holds_it,
        )

        disagreeing = the_caps_that_disagree()
        block["one_working_memory"] = {
            "store": THE_STORE,
            "capacity": the_capacity(),
            "projections": len(who_else_holds_it()),
            "caps_that_disagree": disagreeing,
            "agreed": not disagreeing,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["one_working_memory"] = {"error": repr(exc)}

    # The runtime boundary, and how much of the declared service spine the
    # runtime that is actually up can resolve. A name it cannot resolve is a
    # service somebody else owns — a module singleton, a boot-local — which
    # is the ownership debt, measured rather than asserted.
    try:
        from core.runtime.what_a_runtime_is import (
            THE_OPERATIONS,
            the_services_it_can_address,
            what_is_missing_from,
        )

        # Through the runtime registry, like everything else here. Reaching
        # for ServiceContainer put core.container behind core.runtime, which
        # is the dependency that stops the foundation coming up to report on
        # a mind that failed to start — and there is a test that says so.
        interface = get_runtime_service("kernel_interface", default=None)
        running = getattr(interface, "kernel", None)
        boundary: dict[str, Any] = {"operations": sorted(THE_OPERATIONS)}
        if running is None:
            boundary["runtime"] = "not up"
        else:
            from core.runtime.what_a_runtime_is import a_runtime_over

            over = a_runtime_over(running)
            boundary["runtime"] = type(running).__name__
            boundary["missing"] = what_is_missing_from(over)
            boundary.update(the_services_it_can_address(over))
        block["the_runtime_boundary"] = boundary
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_runtime_boundary"] = {"error": repr(exc)}

    # Who holds the scarce things, who is queued for them, and how the waiting
    # has gone. A lock answers none of those: it is a boolean with a queue
    # nobody can see, and whose order is whatever the loop decided.
    try:
        from core.runtime.who_gets_it_next import (
            how_it_has_gone,
            who_holds_what,
            who_is_waiting,
        )

        block["who_holds_what"] = {
            "held": who_holds_what(),
            "waiting": who_is_waiting(),
            "record": how_it_has_gone(),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["who_holds_what"] = {"error": repr(exc)}

    # Which durable fields have an authority. Counts only — the lists are long
    # and the file that carries them is the baseline. A field that loses its
    # owner between two builds is what this is for.
    try:
        from core.state.who_owns_each_field import what_it_stood_at_last_time

        block["who_owns_each_field"] = what_it_stood_at_last_time()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["who_owns_each_field"] = {"error": repr(exc)}

    # The graph stores under one shape, and whether a reference from one into
    # another still lands. Over the LIVE instances: the same check over fresh
    # graphs measures nothing, which is how a check like this usually fails.
    try:
        from core.knowledge.one_graph import (
            every_graph,
            references_that_lead_nowhere,
            which_stores_have_not_registered,
        )

        graphs = every_graph(live=True)
        # Bounded: the integrity walk is O(nodes x links), and a health route
        # must not become the most expensive thing in the process. Above the
        # bound the count is reported and the walk is not run.
        nodes = sum(len(one.all_nodes()) for one in graphs.values())
        too_big = nodes > _GRAPH_NODES_WORTH_WALKING
        nowhere = [] if (too_big or not graphs) else references_that_lead_nowhere(graphs)
        block["one_graph"] = {
            "stores": sorted(graphs),
            "not_registered": which_stores_have_not_registered(),
            "nodes": nodes,
            "references_that_lead_nowhere": nowhere[:20],
            "how_many_lead_nowhere": len(nowhere),
            "walked": not too_big,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["one_graph"] = {"error": repr(exc)}

    # Which answer route actually answered. A route offered every turn that
    # has never answered one is either unable to fire or gated wrong, and
    # neither is visible from the source: declining and being unable to
    # answer look identical from outside.
    try:
        from core.runtime.what_answered_this_turn import (
            how_the_routes_have_gone,
            routes_that_have_never_answered,
        )

        block["what_answered_this_turn"] = {
            "routes": how_the_routes_have_gone(),
            "never_answered": routes_that_have_never_answered(),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_answered_this_turn"] = {"error": repr(exc)}

    # Who owns the runtime right now, and what a cancelled turn is still
    # waiting on. A runtime that reports idle while it is still tearing a turn
    # down will start the next one on top of it.
    try:
        from core.runtime.whose_turn_it_is import the_turn

        block["whose_turn_it_is"] = the_turn().report()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["whose_turn_it_is"] = {"error": repr(exc)}

    # Who is listening, in a form a restart can put back. AutoGen saves agent
    # state and says it does not save this; a subscription that does not
    # survive a restart is a listener that silently stops.
    try:
        from core.runtime.what_a_message_carries import what_was_subscribed

        subscribed = what_was_subscribed()
        block["what_a_message_carries"] = {
            "subscriptions": len(subscribed),
            "topics": sorted({one["topic"] for one in subscribed}),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_a_message_carries"] = {"error": repr(exc)}

    # What each phase committed at its boundary, and what is held by more than
    # one. A stuck refcount has to be answerable rather than a mystery.
    try:
        from core.state.what_a_phase_changed import how_the_boundaries_have_gone

        block["what_a_phase_changed"] = how_the_boundaries_have_gone()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_a_phase_changed"] = {"error": repr(exc)}

    # What ran on the loop's thread that must not. An on-loop fsync once froze
    # this loop for twenty minutes, and the fix was a rule in a guide; this is
    # the part that can tell you the rule was broken.
    try:
        from core.runtime.which_thread_may_do_this import how_it_has_gone

        block["which_thread_may_do_this"] = how_it_has_gone()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["which_thread_may_do_this"] = {"error": repr(exc)}

    # Every state holder, and whether it decides a fact, shows one, or is
    # scratch. Counts here; the table itself is in the module.
    try:
        from core.state.what_kind_of_state_is_this import how_the_state_is_organised

        organised = how_the_state_is_organised()
        block["what_kind_of_state_is_this"] = {
            key: value for key, value in organised.items() if key != "by_kind"
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_kind_of_state_is_this"] = {"error": repr(exc)}

    # What a skill gave back that it did not declare. Every one of the 82
    # declares now; a declaration nothing checks is a comment.
    try:
        from core.skills.what_every_skill_gives_back import how_results_have_differed

        differed = how_results_have_differed()
        block["what_every_skill_gives_back"] = {
            "skills_that_differed": len(differed),
            "differed": differed,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_every_skill_gives_back"] = {"error": repr(exc)}

    # What the uncalibrated decoding policy actually does. Counts only: the
    # full sweep is 5,760 states and belongs in a test, not on a route.
    try:
        from core.runtime.service_registry import get_runtime_service

        # Through the registry, not by import: this package may not reach
        # core.brain, and a health block that needed that edge would be a
        # layering violation dressed as observability.
        provider = get_runtime_service("the_control_policy_sweep", default=None)
        if not callable(provider):
            raise RuntimeError("the control policy sweep is not registered")
        swept = provider()
        block["the_control_policy"] = {
            "policy": swept["policy"],
            "calibrated": swept["calibrated"],
            "states_swept": swept["states_swept"],
            "controls_that_never_move": swept["controls_that_never_move"],
            "discriminates": swept["discriminates"],
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_control_policy"] = {"error": repr(exc)}

    # How many named faculties have a measured downstream effect. A channel
    # wired to a consumer is not one; only `measured` is evidence.
    try:
        from core.verify.what_has_a_measured_effect import what_it_stood_at_last_time

        block["what_has_a_measured_effect"] = what_it_stood_at_last_time()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_has_a_measured_effect"] = {"error": repr(exc)}

    # Whether each organ knows what it owns, consumes, promises, and does when
    # it fails. Counts from the committed baseline: asking every package walks
    # the whole tree, and health is served on a route.
    try:
        from core.verify.what_each_organ_says import the_baseline

        held = the_baseline()
        block["what_each_organ_says"] = {
            "organs": held.get("organs"),
            "answer_all_four": held.get("answer_all_four"),
            "answer_nothing": held.get("answer_nothing"),
            "who_does_not_say": held.get("who_does_not_say", {}),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_each_organ_says"] = {"error": repr(exc)}

    # The benchmarks somebody else designed: what ran, what could not, and
    # whether anything is claimed without its limit.
    try:
        from core.verify.what_was_measured_outside import how_it_stands

        block["what_was_measured_outside"] = how_it_stands()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_was_measured_outside"] = {"error": repr(exc)}

    # Which semantic routes decide an answer and which only watch one. A
    # shadow route contributes no answers however good it gets.
    try:
        from core.runtime.service_registry import get_runtime_service

        # Through the registry: this package may not reach core.brain, and a
        # health block that needed that edge would be a layering violation
        # dressed as observability.
        provider = get_runtime_service("which_routes_are_authoritative", default=None)
        block["which_routes_are_authoritative"] = (
            provider() if callable(provider) else {"registered": False}
        )
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["which_routes_are_authoritative"] = {"error": repr(exc)}

    # Whether Aura's own cognitive state reached the words she produced. Read
    # through the registry rather than imported: this package may not reach
    # core.brain, and a health block that needed that edge would be a layering
    # violation dressed as observability. ``unexpected_refusals`` is the field
    # that matters — no trained head at all is the pathway waiting for a fit,
    # while a head on disk that refuses to attach is a mismatch with the
    # resident model.
    try:
        from core.runtime.service_registry import get_runtime_service

        # The runtime registry only. This package resolves services through
        # the low-level registry by contract, so that the foundation can come
        # up and report without the container — and a test pins that this file
        # never reaches for ServiceContainer.
        provider = get_runtime_service("endogenous_language_health", default=None)
        if callable(provider):
            block["endogenous_language"] = provider()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["endogenous_language_error"] = repr(exc)

    # Who this runtime is, and where its state lives. Every persistent record
    # is stamped with this, so a store found in the wrong place can be traced
    # to the process that wrote it.
    try:
        from core.runtime.state_ownership import runtime_identity

        block["runtime_identity"] = runtime_identity()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["runtime_identity_error"] = repr(exc)

    # Whether the copy that makes deletion survivable actually exists, and
    # whether it is somewhere a wipe would not reach.
    #
    # Surfaced because both halves failed silently by default: an ark that
    # was never built reports "ok: false, no ark manifest" and nothing else
    # would ever say so, and an ark inside the blast radius looks identical
    # to a safe one until the moment it is needed. This block was wrong on
    # the first attempt — state_root() is ~/.aura and the repo is
    # ~/.aura/live-source, so the ark died to the same `rm -rf` as the
    # original.
    try:
        from core.security.existence_guard import get_existence_guard

        guard = get_existence_guard()
        block["existence_guard"] = {
            "ark": guard.verify_ark(),
            "ark_location": guard.ark_is_outside_the_blast_radius(),
            "sealed": guard.is_sealed(),
        }
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["existence_guard_error"] = repr(exc)

    # Whether the compiled launcher is the launcher its source describes.
    #
    # LIVE DEFECT, 2026-08-10. Bryan reported companion mode did not work: he
    # closed the window and no bubble appeared. Every Python-side organ was
    # correct, and the reason was that the installed launcher binary was built
    # 2026-08-03 while scripts/AuraLauncher.swift had gained the entire
    # companion surface on 2026-08-09. The resident binary contained zero
    # occurrences of "/api/ambient/visibility"; the feature was not broken, it
    # was absent from the executable.
    #
    # Aura.app is deliberately a thin launcher over live source, so Python,
    # assets and config cannot go stale — which is exactly why this one
    # artifact is dangerous. It is the only compiled thing, so it is the only
    # thing a restart does NOT bring current, and its drift therefore looks
    # like a feature that silently does nothing.
    #
    # core.runtime.app_bundle_sync detects and repairs this correctly and was
    # reachable only from launch_aura.sh, which the normal double-click path
    # never executes — spawnAuraProcess runs aura_main.py directly and only
    # falls back to the shell script for protected folders. So the detector
    # existed, was tested, and could not fire for the user who needed it.
    # Reporting it here is what makes the condition observable at all.
    try:
        from core.runtime.app_bundle_sync import launcher_currency

        block["launcher_currency"] = launcher_currency()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["launcher_currency_error"] = repr(exc)

    # What Aura has been ALLOWED to learn permanently, and on whose evidence.
    # Without this the durable-learning gate could be doing anything and the
    # health surface would look identical — the gate's own report existed and
    # had no caller, which is the residue shape this codebase keeps finding.
    try:
        from core.governance.durable_learning import get_durable_learning_gate

        block["durable_learning"] = get_durable_learning_gate().report()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["durable_learning_error"] = repr(exc)

    # Turns that reached cognition without a Will decision.
    #
    # The message handler's comment claimed ALL processing passes the Unified
    # Will. It does not: the gate is skipped entirely before the Will starts,
    # and continues degraded when it raises. Both were silent, so a turn
    # nobody governed looked exactly like a turn the Will approved. A runtime
    # serving ungoverned turns is precisely the kind of fact a green verdict
    # cannot express on its own, which is what this block is for.
    try:
        from core.runtime.governance_coverage import ungoverned_turn_report

        block["ungoverned_turns"] = ungoverned_turn_report()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["ungoverned_turns_error"] = repr(exc)

    # Whether the activation-grounded Φ complex is being fed at all.
    #
    # The residual channel carries 8-bit Grassmann states out of the MLX
    # worker, and for its whole existence nothing in the parent drained it —
    # so the complex reported insufficient_history:0/50 forever while three
    # modules and a live writer said otherwise. Depth on the surface means a
    # regression to zero is visible instead of silent.
    try:
        phi_core = get_runtime_service("phi_core", default=None)
        if phi_core is not None and hasattr(phi_core, "grassmann_history_depth"):
            history = {"grassmann_states": int(phi_core.grassmann_history_depth())}
            # A depth of zero has four possible causes and they are not the
            # same problem: no hook was ever called, the encoder is still
            # filling its window, the encoder is refusing every sample, or
            # nothing drains the ring. Reported as one number they were
            # indistinguishable, and the channel sat at zero for its whole
            # existence with nothing able to say why. The publisher's own
            # counters are read here so the zero explains itself.
            engine = get_runtime_service("affective_steering_engine", default=None)
            hooks = list(getattr(engine, "_hooks", None) or []) if engine else []
            if hooks:
                history["publishers"] = [
                    dict(hook.get_diagnostics().get("phi_residual", {}))
                    for hook in hooks
                    if hasattr(hook, "get_diagnostics")
                ]
            elif engine is not None:
                history["publishers"] = "steering engine present with no hooks"
            block["phi_residual_history"] = history
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["phi_residual_history_error"] = repr(exc)

    # Whether admission is predicting from measurement or still guessing.
    #
    # Read through the runtime service registry rather than by importing the
    # estimator: core/runtime may not depend on core.brain, and that rule is
    # the reason the foundation can come up and report on a mind that failed
    # to start. The estimator registers itself; health only reads.
    try:
        estimator = get_runtime_service("admission_throughput_estimator", default=None)
        if estimator is not None:
            throughput = estimator.report()
            block["admission_throughput"] = {
                "shapes_measured": throughput["shapes_measured"],
                "total_samples": throughput["total_samples"],
            }
        else:
            block["admission_throughput"] = {"registered": False}
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["admission_throughput_error"] = repr(exc)
    _attach_causal_evidence(block)

    # Which path actually produced the recent replies. A demo showing a fluent
    # answer establishes nothing about the pipeline until this says the pipeline
    # ran.
    try:
        from core.verify.turn_receipt import recent_receipts

        receipts = recent_receipts(limit=16)
        block["turn_paths"] = {
            "recent": receipts,
            "full_pipeline_turns": sum(
                1 for r in receipts if r.get("full_pipeline_ran")
            ),
            "model_generation_turns": sum(
                1 for r in receipts if r.get("model_generation")
            ),
            "turns_recorded": len(receipts),
        }
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["turn_paths_error"] = repr(exc)

    try:
        from core.runtime.taint import credibility_caveat, taint_compact, taint_report

        block["taint"] = taint_report()
        block["taint_compact"] = taint_compact()
        caveat = credibility_caveat()
        if caveat:
            block["credibility_caveat"] = caveat
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["taint_error"] = repr(exc)
    try:
        from core.runtime.lockdep import lockdep_report

        lock_report = lockdep_report()
        block["lockdep"] = {
            "clean": lock_report["clean"],
            "acquires_checked": lock_report["acquires_checked"],
            "splats": lock_report["splats"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["lockdep_error"] = repr(exc)
    try:
        from core.runtime.pressure_stall import psi_narrative, psi_report, saturated_resources

        block["pressure"] = psi_report()
        block["pressure_saturated"] = saturated_resources()
        block["pressure_narrative"] = psi_narrative()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["pressure_error"] = repr(exc)
    try:
        # Two boundaries that are worth exactly as much as their visibility.
        # Egress counts what left after being read; attestation says whether
        # the identity Aura booted with is the one she last wrote. Both
        # answer honestly when nothing has happened yet — a zero here means
        # "nothing was refused", never "nothing was checked".
        from core.security.egress_privacy import egress_privacy_counters
        from core.security.state_attestation import attestation_report

        block["egress_privacy"] = egress_privacy_counters()
        block["state_attestation"] = attestation_report()
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["egress_privacy_error"] = repr(exc)
    try:
        from core.knowledge.metta import metta_report
        from core.organism.model_validation import get_suite, validation_report
        from core.runtime.foundations import cognition_validation_status

        validation = validation_report()
        block["self_model"] = {
            "claims": len(validation["claims"]),
            "tests": len(validation["tests"]),
            "unsupported_claims": [c["statement"] for c in get_suite().unsupported_claims()],
            "empirical_run": cognition_validation_status(),
            "metta": {k: metta_report()[k] for k in ("rules", "reductions", "truncations")},
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["self_model_error"] = repr(exc)
    try:
        # Which stages lose facts the turn already established. A break here
        # is not a bad answer — it is a composition defect, and it names the
        # stage responsible, which is the part that was previously impossible
        # to recover after the fact.
        from core.runtime.fact_custody import custody_report

        custody = custody_report()
        block["fact_custody"] = {
            "turns_tracked": custody["turns_tracked"],
            "turns_with_breaks": custody["turns_with_breaks"],
            "stages_that_broke_custody": custody["stages_that_broke_custody"],
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["fact_custody_error"] = repr(exc)
    try:
        from core.fsw.assertions import assertions_report
        from core.fsw.command_dispatch import command_report
        from core.fsw.health_checker import health_checker_report
        from core.fsw.rate_groups import rate_group_report
        from core.fsw.restart_protection import restart_report
        from core.fsw.telemetry_dictionary import telemetry_report

        telemetry = telemetry_report()
        pings = health_checker_report()
        block["flight_software"] = {
            "telemetry": {
                "channels": telemetry["channels"],
                "violations": telemetry["violations"],
                "recent_events": telemetry["recent_events"],
            },
            "restart_protection": restart_report()["core_sets"],
            "rate_groups": {
                k: rate_group_report()[k] for k in ("slipping", "total_cycles", "total_slips")
            },
            "assertions": {
                "clean": assertions_report()["clean"],
                "distinct_sites": assertions_report()["distinct_sites"],
            },
            "health_pings": {
                "unresponsive": pings["unresponsive"],
                "slow": pings["slow"],
                "critical_unresponsive": pings["critical_unresponsive"],
            },
            "commands": {
                "declared": command_report()["commands"],
                "dispatched": command_report()["dispatched"],
            },
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["flight_software_error"] = repr(exc)
    try:
        from core.observability.histograms import histograms_report
        from core.observability.trace_events import tracer_report
        from core.runtime.field_trials import field_trials_report
        from core.runtime.memory_infra import memory_infra_report
        from core.security.rule_of_two import rule_of_two_report

        histograms = histograms_report()
        memory = memory_infra_report()
        posture = rule_of_two_report()
        block["observability"] = {
            "histograms": {
                "count": histograms["count"],
                "clipping": histograms["clipping"],
                "expired": [e["name"] for e in histograms["expired"]],
            },
            "trace": {
                k: tracer_report()[k] for k in ("enabled", "buffered", "dropped", "span_s")
            },
            "memory_attribution": memory["leak_report"],
            "field_trials": field_trials_report()["active_groups"],
            "security_posture": {
                "violations": posture["violations"],
                "at_the_limit": posture["at_the_limit"],
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["observability_error"] = repr(exc)
    try:
        from core.ontogeny.service import ontogeny_health_report

        ontogeny = ontogeny_health_report()
        # The owner supplies this bounded projection. The full diagnostic
        # report includes per-control-point corpus aggregates and must never be
        # pulled into a high-frequency health probe.
        block["ontogeny"] = {
            "episodes_seen": ontogeny.get("episodes_seen"),
            "novelty": ontogeny.get("novelty"),
            "state": {
                k: (ontogeny.get("state") or {}).get(k)
                for k in ("steps", "era", "fingerprint", "age_days")
            },
            "stages": dict(ontogeny.get("stages") or {}),
            "frozen": ontogeny.get("frozen"),
            "observation_rate": ontogeny.get("observation_rate"),
            "calibration": {
                cp: {"ece": rep.get("ece"), "overconfidence": rep.get("overconfidence")}
                for cp, rep in (ontogeny.get("calibration") or {}).items()
            },
            "world_model": {
                k: (ontogeny.get("world_model") or {}).get(k)
                for k in ("step_count", "train_steps", "mean_surprise", "last_loss")
            },
        }
        from core.ontogeny.conclusion import get_verbalization_ledger

        verbalization = get_verbalization_ledger().report()
        block["ontogeny"]["verbalization"] = {
            k: verbalization[k]
            for k in ("checked", "with_violations", "overstatements", "faithful_rate")
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["ontogeny_error"] = repr(exc)
    try:
        from core.bus.qos import qos_report
        from core.health.diagnostics_aggregator import diagnostics_report
        from core.observability.bus_recorder import bus_recorder_report
        from core.runtime.lifecycle import lifecycle_report
        from core.runtime.parameters import parameters_report

        diagnostics = diagnostics_report()
        lifecycles = lifecycle_report()
        block["middleware"] = {
            "diagnostics": {
                "level": diagnostics["level"],
                "stale": diagnostics["stale"],
                "errors": diagnostics["errors"],
                "summary": diagnostics["summary"],
            },
            "lifecycles": {
                "by_state": lifecycles["by_state"],
                "critical_inactive": lifecycles["critical_inactive"],
                "errored": lifecycles["errored"],
            },
            "qos": {
                "topics": qos_report()["topic_count"],
                "mismatches": len(qos_report()["qos_mismatches"]),
                "not_alive": qos_report()["not_alive"],
            },
            "parameters": {
                "count": parameters_report()["count"],
                "changed_from_default": parameters_report()["changed_from_default"],
            },
            "bus_ring": {
                k: bus_recorder_report()[k]
                for k in ("ring_size", "ring_span_s", "dumps", "recording")
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["middleware_error"] = repr(exc)
    try:
        from core.runtime.admission import admission_report
        from core.runtime.eviction import eviction_report
        from core.runtime.lease import lease_report
        from core.runtime.quota import quota_report
        from core.runtime.reconcile import reconcile_report

        admission = admission_report()
        eviction = eviction_report()
        block["orchestration"] = {
            "admission": {
                "hooks": len(admission["mutating"]) + len(admission["validating"]),
                "admitted": admission["admitted"],
                "denied": admission["denied"],
            },
            "quota": quota_report()["by_qos_class"],
            "eviction": {
                "eviction_order": eviction["eviction_order"],
                "breached": eviction["currently_breached"],
                "reclaims": eviction["reclaims"],
                "evictions": eviction["evictions"],
            },
            "controllers": reconcile_report(),
            "leases": lease_report(),
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["orchestration_error"] = repr(exc)
    try:
        from core.runtime.sanitizers import sanitizer_report

        block["sanitizers"] = sanitizer_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["sanitizers_error"] = repr(exc)
    try:
        from core.verify.invariants import last_report

        block["verifier"] = last_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["verifier_error"] = repr(exc)
    try:
        from core.pipeline.pass_manager import pass_manager_report

        passes = pass_manager_report()
        block["passes"] = {
            "bisect_limit": passes["bisect_limit"],
            "skips": passes["skips"],
            "hottest": passes["hottest"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["passes_error"] = repr(exc)
    try:
        from core.runtime.oom_policy import oom_report

        oom = oom_report()
        block["oom"] = {
            "next_victim": oom["next_victim"],
            "sheddable_organs": oom["sheddable_organs"],
            "immune_organs": oom["immune_organs"],
            "recent_sheds": oom["recent_sheds"][-3:],
            "restart_requested": oom["restart_requested"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["oom_error"] = repr(exc)

    # Grounding: does what she persisted match what actually ran?
    #
    # The work ledger and the canary counters are detectors, and a detector
    # nobody reads is worse than no detector — it produces the confidence of
    # having checked with none of the checking. This block is their reader.
    try:
        from core.security.injection_canary import canary_status
        from core.verify.work_ledger import status as work_ledger_status

        canaries = canary_status()
        block["grounding"] = {
            "work_ledger": work_ledger_status(),
            "injection_canaries": {
                "evaluated": canaries["evaluated"],
                "incidents": canaries["hijacked"] + canaries["leaked"],
                "incident_rate": canaries["incident_rate"],
                # A probe lane that keeps failing has silently stopped
                # detecting; that is itself the finding.
                "blind": canaries["blind"],
                "inconclusive": canaries["inconclusive"],
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["grounding_error"] = repr(exc)

    # What keeps failing, as opposed to what failed. The degradation log
    # answers the second; only the scar record answers the first.
    try:
        from core.runtime.degradation_habituation import get_habituation

        habituation = get_habituation().status()
        block["chronic_faults"] = {
            "signatures_tracked": habituation["signatures_tracked"],
            "saturated": habituation["saturated"],
            "residual_floor": habituation["residual_floor"],
            # Truncated: this is a caveat line, not a fault database.
            "chronic": habituation["chronic"][:5],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["chronic_faults_error"] = repr(exc)

    # Memories that have done more harm than good, and how much the ambient
    # mind declined to say. Both are numbers nothing else in the runtime
    # produces.
    try:
        # Registry lookup, never an import; see core/memory/retrieval_outcomes.py.
        ledger = get_runtime_service("retrieval_outcome_ledger", default=None)
        if ledger is None:
            raise LookupError("retrieval outcome ledger is not registered")
        outcomes = ledger.status()
        block["judgement"] = {
            "retrieval": {
                "tracked": outcomes["tracked"],
                "graded": outcomes["graded"],
                "harmful_memories": outcomes["harmful_memories"][:5],
            },
        }
        # Resolved through the low-level runtime registry, not imported and
        # not fetched from ServiceContainer. Two separate rules point here:
        # core/runtime may not depend on core.agency (the layering gate
        # rejects the direct import), and this module may not reach into
        # the container at all (test_health_contract_uses_low_level_runtime_registry
        # — the foundation's health surface has to work when the container
        # is the thing that failed).
        #
        # The governor registers itself; the runtime reads whatever is
        # there and reports honestly when nothing is.
        governor = get_runtime_service("ambient_governor", default=None)
        if governor is None:
            block["judgement"]["ambient"] = {"registered": False}
        else:
            ambient = governor.status()
            block["judgement"]["ambient"] = {
                "registered": True,
                "configured": ambient["configured"],
                "spent_today": ambient["spent_today"],
                "remaining_today": ambient["remaining_today"],
                "withheld": ambient["withheld"],
                "restraint_rate": ambient["restraint_rate"],
                "calibration": ambient["calibration"],
            }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["judgement_error"] = repr(exc)

    # What the maturity pass built, and whether any of it decides anything.
    #
    # An external source-level review made the point this answers: a
    # 150-line module reads as though a system-wide invariant now exists, and
    # it does not. So the report says which primitives have a production
    # caller, which are only reachable, and which are still proposals — rather
    # than listing them and letting the list imply the rest.
    # Three of these walk the whole tree — the clock scan 19s, the settings
    # scan 9s, the deprecation scan 7s. They belong to a gate and to
    # tools/inspect_runtime.py, not to a route somebody refreshes: a health
    # report that takes half a minute is a health report nobody asks for.
    # What is left here is what a live runtime can answer about itself.
    for name, read in (
        ("trace_boundaries",
         "core.observability.does_one_trace_reach_the_end:how_far_a_trace_reaches"),
        ("call_policies", "core.runtime.how_a_call_is_made:how_the_calls_are_made"),
        ("task_endings",
         "core.runtime.how_a_task_should_end:how_the_endings_are_declared"),
        ("prompt_room", "core.brain.llm.who_got_the_room:how_the_room_was_shared"),
        ("memory_kinds",
         "core.memory.what_kind_of_memory_is_this:how_the_kinds_stand"),
        ("write_drains",
         "core.state.nothing_lands_before_its_writes:how_the_drains_have_gone"),
        ("abandoned_calls",
         "core.runtime.cancelling_the_call_and_not_just_the_wait:how_the_calls_ended"),
        ("checkable_promises",
         "core.verify.a_promise_with_a_test:how_the_promises_stand"),
    ):
        module_name, _, func_name = read.partition(":")
        try:
            import importlib

            said = getattr(importlib.import_module(module_name), func_name)()
            # The each/policies lists are long and this is served on a route.
            block[name] = {
                key: value
                for key, value in said.items()
                if key not in ("each", "policies", "the_disagreements", "recent")
            }
        except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
            block[f"{name}_error"] = repr(exc)
    return block


# ═══════════════════════════════════════════════════════════════════════
# THE INTEGRITY BLOCK IS NEVER COLLECTED ON THE EVENT LOOP
#
# _runtime_integrity_block() is ~15 independent sub-reports, and several of
# them reach disk: ontogeny does a full-table COUNT/GROUP BY over the
# episodes corpus, CAA readiness np.load()s every vector file, the gateway
# record index globs a tree. None of that is bounded by anything.
#
# runtime_health_report() had it inline, and runtime_health_report() is
# called from the orchestrator's periodic pulse — which is an async task on
# the main loop. On 2026-07-29 that produced a 103.8s event-loop stall
# during a live demo (scheduler._run_task → _pulse_subsystem_audit →
# emit_pulse → runtime_health_report → ontogeny stats → sqlite). The whole
# runtime froze: health polls timed out at 6s, the GUI fell back to
# "Connecting to runtime", and the immune system opened CRITICAL incidents
# for lag that the health report itself had caused. It was the third stall
# of an escalating series (5.6s → 17.6s → 103.8s) because each freeze made
# the next collection slower.
#
# So: a caller on a running event loop NEVER collects. It reads the last
# snapshot and asks a single daemon thread to refresh it. Callers that are
# not on a loop (tests, CLI, background collectors) keep the old inline
# semantics, now behind a TTL so a burst of them costs one collection.
# _runtime_integrity_block() itself is untouched and still directly
# callable — the tests that assert on its contents go straight at it.
# ═══════════════════════════════════════════════════════════════════════

_INTEGRITY_TTL_S = _float_env("AURA_HEALTH_INTEGRITY_TTL_S", 15.0)
_INTEGRITY_LOCK = threading.Lock()
_INTEGRITY_SNAPSHOT: dict[str, Any] | None = None
_INTEGRITY_SNAPSHOT_AT = 0.0
_INTEGRITY_SNAPSHOT_UNIX = 0.0
_INTEGRITY_REFRESHING = False
_INTEGRITY_COLLECTIONS = 0
_INTEGRITY_LOOP_SERVES = 0


def reset_integrity_snapshot_for_test() -> None:
    """Drop the cached integrity snapshot so a test observes a cold collect."""
    global _INTEGRITY_SNAPSHOT, _INTEGRITY_SNAPSHOT_AT, _INTEGRITY_SNAPSHOT_UNIX
    global _INTEGRITY_COLLECTIONS, _INTEGRITY_LOOP_SERVES, _INTEGRITY_REFRESHING
    with _INTEGRITY_LOCK:
        _INTEGRITY_SNAPSHOT = None
        _INTEGRITY_SNAPSHOT_AT = 0.0
        _INTEGRITY_SNAPSHOT_UNIX = 0.0
        _INTEGRITY_COLLECTIONS = 0
        _INTEGRITY_LOOP_SERVES = 0
        # Clear the in-flight latch too: a test that reset mid-refresh would
        # otherwise leave it stuck True and block every later refresh.
        _INTEGRITY_REFRESHING = False


def _on_event_loop() -> bool:
    """True when this thread is running an asyncio loop that we must not block."""
    try:
        import asyncio

        asyncio.get_running_loop()
    except (RuntimeError, ImportError):
        return False
    return True


def _store_integrity_snapshot(block: dict[str, Any]) -> None:
    global _INTEGRITY_SNAPSHOT, _INTEGRITY_SNAPSHOT_AT, _INTEGRITY_SNAPSHOT_UNIX
    global _INTEGRITY_COLLECTIONS
    with _INTEGRITY_LOCK:
        _INTEGRITY_SNAPSHOT = block
        _INTEGRITY_SNAPSHOT_AT = time.monotonic()
        _INTEGRITY_SNAPSHOT_UNIX = time.time()
        _INTEGRITY_COLLECTIONS += 1


def _collect_integrity_snapshot() -> dict[str, Any]:
    """Collect and cache. Callers must already know they are off the loop."""
    started = time.monotonic()
    try:
        block = _runtime_integrity_block()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block = {"collect_error": repr(exc)}
    block["collect_duration_s"] = round(max(0.0, time.monotonic() - started), 3)
    _store_integrity_snapshot(block)
    return block


def _refresh_integrity_snapshot_async() -> None:
    """Ask one daemon thread to refresh. Extra requests while it runs are dropped."""
    global _INTEGRITY_REFRESHING
    with _INTEGRITY_LOCK:
        if _INTEGRITY_REFRESHING:
            return
        _INTEGRITY_REFRESHING = True

    def _run() -> None:
        global _INTEGRITY_REFRESHING
        try:
            _collect_integrity_snapshot()
        finally:
            with _INTEGRITY_LOCK:
                _INTEGRITY_REFRESHING = False

    try:
        threading.Thread(
            target=_run, name="AuraHealthIntegritySnapshot", daemon=True
        ).start()
    except RuntimeError:
        # Interpreter shutdown, or the thread table is exhausted. Release the
        # latch so a later call can try again rather than wedging the snapshot
        # as permanently "refreshing".
        with _INTEGRITY_LOCK:
            _INTEGRITY_REFRESHING = False


def integrity_block_snapshot() -> dict[str, Any]:
    """The integrity block, never at the cost of the event loop.

    Off the loop this is the old behaviour behind a TTL. On the loop it is a
    read of the last snapshot plus a background refresh request — bounded by
    a dict copy, never by disk.
    """
    global _INTEGRITY_LOOP_SERVES
    with _INTEGRITY_LOCK:
        snapshot = _INTEGRITY_SNAPSHOT
        age = (
            time.monotonic() - _INTEGRITY_SNAPSHOT_AT
            if snapshot is not None
            else float("inf")
        )
        captured_at = _INTEGRITY_SNAPSHOT_UNIX
        collections = _INTEGRITY_COLLECTIONS
    fresh = snapshot is not None and age < _INTEGRITY_TTL_S

    if not fresh and not _on_event_loop():
        snapshot = _collect_integrity_snapshot()
        with _INTEGRITY_LOCK:
            age = 0.0
            captured_at = _INTEGRITY_SNAPSHOT_UNIX
            collections = _INTEGRITY_COLLECTIONS
        fresh = True
    elif not fresh:
        _refresh_integrity_snapshot_async()
        with _INTEGRITY_LOCK:
            _INTEGRITY_LOOP_SERVES += 1

    if snapshot is None:
        # Cold, and we are on the loop. The refresh is already in flight; say
        # so honestly rather than blocking or inventing a clean bill of health.
        return {
            "snapshot": {
                "collected": False,
                "warming": True,
                "age_s": None,
                "captured_at": None,
                "collections": collections,
                "ttl_s": _INTEGRITY_TTL_S,
            }
        }

    block = dict(snapshot)
    block["snapshot"] = {
        "collected": True,
        "warming": False,
        "stale": not fresh,
        "age_s": round(age, 3) if age != float("inf") else None,
        "captured_at": captured_at or None,
        "collections": collections,
        "ttl_s": _INTEGRITY_TTL_S,
    }
    return block


def runtime_health_report() -> dict[str, Any]:
    """Return Aura's canonical runtime health contract report."""
    report = evaluate_health().to_report()
    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        shutdown = get_shutdown_coordinator().get_status()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        shutdown = {
            "running": False,
            "request": {"requested": False},
            "report": None,
            "error": repr(exc),
        }
    report["shutdown"] = shutdown
    report["integrity"] = integrity_block_snapshot()
    # Subsystems publish; the foundation never reaches down. See
    # core/runtime/health_fragments.py for why.
    report.update(collect_health_fragments())
    request = shutdown.get("request") if isinstance(shutdown, dict) else None
    if isinstance(request, dict) and request.get("requested") is True:
        report["pre_shutdown_status"] = report.get("status")
        report["status"] = "stopping"
        report["healthy"] = False
        report["operational"] = False
        report["status_code"] = 503
        blockers = [str(item) for item in report.get("probe_blockers", [])]
        if "runtime_shutdown" not in blockers:
            blockers.insert(0, "runtime_shutdown")
        report["probe_blockers"] = blockers
        required_probes = report.get("required_probes")
        if isinstance(required_probes, dict):
            required_probes["all_passed"] = False
    return report


# ═══════════════════════════════════════════════════════════════════════
# THE PROBE SPLIT (roadmap K2): startup / liveness / readiness
#
# Kubernetes semantics, adopted because their conflation here caused real
# incidents: a loop-lag spike flipped the health verdict, boot readiness
# went false, and the GUI sat on "Connecting to runtime…" for 55 minutes
# over a fully conversational mind. Three probes, three INDEPENDENT
# meanings:
#
#   STARTUP   — has this process EVER been ready? Latched: once readiness
#               passes, startup is complete for the life of the process and
#               the surface may never present as "booting" again — only
#               "degraded". Before the latch, a startup deadline separates
#               "still warming" (ok) from "startup wedged" (not ok).
#   LIVENESS  — is the mind alive at all? Restart-worthy signal, so it is
#               deliberately RARE: only a DEAD verdict (no critical spine)
#               fails liveness. Flapping important-tier services never do.
#   READINESS — may traffic flow NOW? The existing required-probe-group
#               gate; may flap without implying a restart.
# ═══════════════════════════════════════════════════════════════════════

class ProbeKind(StrEnum):
    STARTUP = "startup"
    LIVENESS = "liveness"
    READINESS = "readiness"


@dataclass(frozen=True)
class ProbeVerdict:
    kind: ProbeKind
    ok: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": str(self.kind), "ok": self.ok, "reason": self.reason}


_STARTUP_LATCH_LOCK = threading.Lock()
_STARTUP_COMPLETE_AT: float | None = None
# Fallback time base for the startup deadline: the moment this module was
# imported. _process_uptime_seconds() reads the orchestrator's start time,
# which is 0.0 when no orchestrator ever registered — and a boot so wedged
# it never registers an orchestrator is EXACTLY the wedge the startup probe
# must catch, so it needs a clock that always runs.
_PROBE_EPOCH = time.time()


_STARTUP_DEADLINE_FLAG = None


def _startup_deadline_s() -> float:
    global _STARTUP_DEADLINE_FLAG
    if _STARTUP_DEADLINE_FLAG is None:
        try:
            from core.runtime.flags import FlagKind, declare

            _STARTUP_DEADLINE_FLAG = declare(
                "AURA_STARTUP_DEADLINE_S",
                kind=FlagKind.FLOAT,
                default=900.0,
                description="Seconds a fresh process may warm before the startup probe calls it wedged",
                owner="core.runtime.health_contract",
            )
        except (ImportError, AttributeError, RuntimeError, ValueError):
            return _float_env("AURA_STARTUP_DEADLINE_S", 900.0)
    return float(_STARTUP_DEADLINE_FLAG.value())


def _startup_age_s() -> float:
    return max(_process_uptime_seconds(), time.time() - _PROBE_EPOCH)


def reset_startup_latch_for_test() -> None:
    global _STARTUP_COMPLETE_AT, _PROBE_EPOCH
    with _STARTUP_LATCH_LOCK:
        _STARTUP_COMPLETE_AT = None
        _PROBE_EPOCH = time.time()


def startup_complete_at() -> float | None:
    with _STARTUP_LATCH_LOCK:
        return _STARTUP_COMPLETE_AT


def latch_startup_if_ready(ready_ok: bool) -> None:
    """Record the startup latch the first time readiness passes.

    Idempotent and monotonic: once latched, startup stays complete for the
    life of the process no matter how readiness flaps afterwards.
    """
    global _STARTUP_COMPLETE_AT
    if not ready_ok:
        return
    with _STARTUP_LATCH_LOCK:
        if _STARTUP_COMPLETE_AT is None:
            _STARTUP_COMPLETE_AT = time.time()


def probes_from_report(report: dict[str, Any]) -> dict[str, ProbeVerdict]:
    """Derive the three probe verdicts from an existing health report.

    Surfaces that already paid for ``evaluate_health()`` (boot status, the
    narrator) get the probe split without a second full evaluation.
    """
    status = required_probe_status(report)
    ready_ok = required_probe_groups_pass(status)
    ready_blockers = [] if ready_ok else required_probe_blockers(status)

    latch_startup_if_ready(ready_ok)

    latched = startup_complete_at()
    uptime = _startup_age_s()
    if latched is not None:
        startup = ProbeVerdict(
            ProbeKind.STARTUP, True, f"startup complete (latched at {latched:.0f})"
        )
    elif uptime <= _startup_deadline_s():
        startup = ProbeVerdict(
            ProbeKind.STARTUP,
            True,
            f"starting ({uptime:.0f}s of {_startup_deadline_s():.0f}s startup window)",
        )
    else:
        startup = ProbeVerdict(
            ProbeKind.STARTUP,
            False,
            f"startup wedged: never reached readiness within {_startup_deadline_s():.0f}s",
        )

    live_ok = str(report.get("status", "")) != HealthLevel.DEAD.value
    liveness = ProbeVerdict(
        ProbeKind.LIVENESS,
        live_ok,
        "critical spine registered" if live_ok else "no critical service present",
    )

    readiness = ProbeVerdict(
        ProbeKind.READINESS,
        ready_ok,
        "all required probe groups pass" if ready_ok else "; ".join(ready_blockers) or "not ready",
    )

    return {"startup": startup, "liveness": liveness, "readiness": readiness}


def evaluate_probes() -> dict[str, ProbeVerdict]:
    """One health evaluation, three independent probe verdicts."""
    return probes_from_report(evaluate_health().to_report())


def probe_split_report() -> dict[str, Any]:
    """Serializable probe-split for health surfaces and the narrator."""
    return {name: probe.to_dict() for name, probe in evaluate_probes().items()}


def log_health_report() -> HealthVerdict:
    """Evaluate and log the health report. Returns the verdict."""
    verdict = evaluate_health()
    summary_lines = verdict.summary().split("\n")
    if verdict.level == HealthLevel.HEALTHY:
        logger.info(summary_lines[0])
    elif verdict.level == HealthLevel.DEGRADED:
        logger.warning(summary_lines[0])
    else:
        logger.critical(summary_lines[0])

    for status, line in zip(verdict.services, summary_lines[1:], strict=False):
        if status.present and status.liveness_ok is not False:
            logger.info(line)
        elif status.requirement.tier == ServiceTier.CRITICAL:
            logger.critical(line)
        else:
            logger.warning(line)
    return verdict
