"""Activation audit for Aura's runtime architecture.

Many systems can exist in a repo without actually running.  This module makes
that visible and actionable: expected loops are checked against the service
container and task tracker, safe missing loops can be reconciled, and every
disabled subsystem records a concrete reason.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

Starter = Callable[[Any], Awaitable[Any] | Any]


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    service_keys: tuple[str, ...] = ()
    task_name_contains: tuple[str, ...] = ()
    required: bool = True
    auto_start: bool = False
    reason: str = ""
    starter: Starter | None = None


@dataclass(frozen=True)
class ActivationStatus:
    name: str
    active: bool
    required: bool
    auto_start: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    reconciled: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active": self.active,
            "required": self.required,
            "auto_start": self.auto_start,
            "reason": self.reason,
            "evidence": self.evidence,
            "reconciled": self.reconciled,
            "error": self.error,
        }


@dataclass(frozen=True)
class ActivationReport:
    generated_at: float
    statuses: tuple[ActivationStatus, ...]

    @property
    def required_active_ratio(self) -> float:
        required = [s for s in self.statuses if s.required]
        return sum(1 for s in required if s.active) / max(1, len(required))

    @property
    def missing_required(self) -> tuple[ActivationStatus, ...]:
        return tuple(s for s in self.statuses if s.required and not s.active)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "required_active_ratio": round(self.required_active_ratio, 4),
            "missing_required": [s.name for s in self.missing_required],
            "statuses": [s.to_dict() for s in self.statuses],
        }


async def _start_autonomy_conductor(orchestrator: Any) -> Any:
    from core.runtime.autonomy_conductor import start_default_conductor

    conductor = await start_default_conductor()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("autonomy_conductor", conductor, required=False)
    except Exception as exc:
        record_degradation("activation_audit", exc)
    return conductor


async def _start_scar_formation(orchestrator: Any) -> Any:
    from core.memory.scar_formation import get_scar_formation

    scars = get_scar_formation()
    await scars.start()
    return scars


async def _start_keep_awake_if_enabled(orchestrator: Any) -> Any:
    from core.runtime.keep_awake import start_from_environment

    return start_from_environment()


async def _register_octopus(orchestrator: Any) -> Any:
    from core.consciousness.octopus_arms import get_octopus_federation
    from core.container import ServiceContainer

    federation = get_octopus_federation()
    ServiceContainer.register_instance("octopus_federation", federation, required=False)
    return federation.get_status() if hasattr(federation, "get_status") else {"registered": True}


async def _register_criticality(orchestrator: Any) -> Any:
    from core.consciousness.criticality_regulator import get_criticality_regulator
    from core.container import ServiceContainer

    regulator = get_criticality_regulator()
    ServiceContainer.register_instance("criticality_regulator", regulator, required=False)
    return regulator.get_status() if hasattr(regulator, "get_status") else {"registered": True}


async def _register_substrate_policy(orchestrator: Any) -> Any:
    from core.consciousness.substrate_policy_head import get_substrate_policy_head
    from core.container import ServiceContainer

    head = get_substrate_policy_head()
    ServiceContainer.register_instance("substrate_policy_head", head, required=False)
    return {"registered": True}


async def _start_proof_kernel_bridge(orchestrator: Any) -> Any:
    from core.runtime.proof_kernel_bridge import start_proof_kernel_bridge

    bridge = await start_proof_kernel_bridge()
    return bridge.status()


async def _start_lock_watchdog(orchestrator: Any) -> Any:
    from core.container import ServiceContainer
    from core.resilience.lock_watchdog import get_lock_watchdog

    watchdog = get_lock_watchdog()
    watchdog.start()
    ServiceContainer.register_instance("lock_watchdog", watchdog, required=False)
    return watchdog.get_snapshot()


async def _start_concurrency_health(orchestrator: Any) -> Any:
    from core.runtime.concurrency_health import start_concurrency_health_monitor

    monitor = await start_concurrency_health_monitor()
    return monitor.status()


async def _register_aura_workspace(orchestrator: Any) -> Any:
    from core.container import ServiceContainer
    from core.workspace.aura_workspace import AuraWorkspace
    from core.workspace.markdown_workspace import MarkdownWorkspace

    store = ServiceContainer.get("markdown_workspace", default=None)
    if store is None:
        store = MarkdownWorkspace()
        ServiceContainer.register_instance("markdown_workspace", store, required=False)
    workspace = ServiceContainer.get("aura_workspace", default=None)
    if workspace is None:
        workspace = AuraWorkspace(store=store)
        ServiceContainer.register_instance("aura_workspace", workspace, required=False)
    if not ServiceContainer.has("agent_workspace"):
        ServiceContainer.register_instance("agent_workspace", workspace, required=False)
    return {"registered": True, "storage_path": str(store.storage_path)}


async def _register_architecture_governor(orchestrator: Any) -> Any:
    from core.architect.config import ASAConfig
    from core.architect.governor import AutonomousArchitectureGovernor
    from core.container import ServiceContainer

    governor = ServiceContainer.get("architecture_governor", default=None)
    if governor is None:
        governor = AutonomousArchitectureGovernor(ASAConfig.from_env())
        ServiceContainer.register_instance("architecture_governor", governor, required=False)
    if not ServiceContainer.has("autonomous_architecture_governor"):
        ServiceContainer.register_instance("autonomous_architecture_governor", governor, required=False)
    boot_audit_scheduled = False
    if getattr(governor.config, "enabled", False):
        try:
            from core.utils.task_tracker import get_task_tracker

            get_task_tracker().create_task(
                governor.boot_background_audit(),
                name="ArchitectureGovernorBootAudit",
            )
            boot_audit_scheduled = True
        except (ImportError, AttributeError, RuntimeError):
            boot_audit_scheduled = False
    return {
        "registered": True,
        "mode": "autopromote" if governor.config.autopromote else "audit_only",
        "max_tier": governor.config.max_tier.name,
        "boot_audit_scheduled": boot_audit_scheduled,
    }


DEFAULT_SPECS: tuple[ActivationSpec, ...] = (
    ActivationSpec(
        name="autonomy_conductor",
        service_keys=("autonomy_conductor",),
        task_name_contains=("Aura.AutonomyConductor",),
        required=True,
        auto_start=True,
        starter=_start_autonomy_conductor,
        reason="runs proof, validation, metabolic, and self-maintenance loops on schedule",
    ),
    ActivationSpec(
        name="mind_tick",
        task_name_contains=("mind_tick",),
        required=True,
        reason="continuous cognitive heartbeat",
    ),
    ActivationSpec(
        name="scheduler",
        task_name_contains=("scheduler.start",),
        required=True,
        reason="runtime scheduled actions",
    ),
    ActivationSpec(
        name="output_gate",
        service_keys=("output_gate",),
        required=True,
        reason="foreground/background output governance",
    ),
    ActivationSpec(
        name="scar_formation",
        service_keys=("scar_formation",),
        required=True,
        auto_start=True,
        starter=_start_scar_formation,
        reason="immune tolerance and learned caution",
    ),
    ActivationSpec(
        name="self_healing",
        service_keys=("self_healing",),
        required=True,
        reason="crash and stale-heartbeat recovery",
    ),
    ActivationSpec(
        name="performance_guard",
        service_keys=("performance_guard",),
        required=True,
        reason="resource and event-loop safety",
    ),
    ActivationSpec(
        name="criticality_regulator",
        service_keys=("criticality_regulator",),
        required=True,
        auto_start=True,
        starter=_register_criticality,
        reason="edge-of-chaos regulation for substrate activity",
    ),
    ActivationSpec(
        name="octopus_federation",
        service_keys=("octopus_federation",),
        required=True,
        auto_start=True,
        starter=_register_octopus,
        reason="semi-autonomous arm federation feeding action candidates",
    ),
    ActivationSpec(
        name="substrate_policy_head",
        service_keys=("substrate_policy_head",),
        required=True,
        auto_start=True,
        starter=_register_substrate_policy,
        reason="turns substrate state into goal, memory, tool, risk, and repair policy weights",
    ),
    ActivationSpec(
        name="proof_kernel_bridge",
        service_keys=("proof_kernel_bridge",),
        required=True,
        auto_start=True,
        starter=_start_proof_kernel_bridge,
        reason="connects standalone proof-kernel proxies to live runtime evidence and honest claim scope",
    ),
    ActivationSpec(
        name="lock_watchdog",
        service_keys=("lock_watchdog",),
        task_name_contains=("aura.lock_watchdog",),
        required=True,
        auto_start=True,
        starter=_start_lock_watchdog,
        reason="detects and recovers stalled lock acquisition/hold paths",
    ),
    ActivationSpec(
        name="concurrency_health",
        service_keys=("concurrency_health",),
        required=True,
        auto_start=True,
        starter=_start_concurrency_health,
        reason="causally samples task liveness, lock stalls, DLQ backlog, and degradation pressure",
    ),
    ActivationSpec(
        name="agent_workspace",
        service_keys=("aura_workspace", "agent_workspace"),
        required=True,
        auto_start=True,
        starter=_register_aura_workspace,
        reason="governed durable workspace for evidence, memory, decisions, repairs, and versioned artifacts",
    ),
    ActivationSpec(
        name="curiosity_engine",
        service_keys=("curiosity_engine", "curiosity"),
        required=False,
        reason="autonomous learning and anomaly investigation",
    ),
    ActivationSpec(
        name="autonomous_self_modification",
        service_keys=("autonomous_self_modification", "self_modification_engine", "self_modifier"),
        task_name_contains=("safe_self_modification_loop", "AutonomousSelfModification"),
        required=False,
        reason="Will-gated improvement proposals and repair loops",
    ),
    ActivationSpec(
        name="architecture_governor",
        service_keys=("architecture_governor", "autonomous_architecture_governor"),
        required=False,
        auto_start=True,
        starter=_register_architecture_governor,
        reason="audit-first autonomous software architect with shadow proof, rollback, and monitor gates",
    ),
    ActivationSpec(
        name="research_cycle",
        service_keys=("research_cycle",),
        task_name_contains=("research_cycle",),
        required=False,
        reason="idle research daemon",
    ),
    ActivationSpec(
        name="keep_awake",
        required=False,
        auto_start=True,
        starter=_start_keep_awake_if_enabled,
        reason="runs only when AURA_KEEP_AWAKE is enabled",
    ),
)


class ActivationAuditor:
    def __init__(self, specs: tuple[ActivationSpec, ...] = DEFAULT_SPECS) -> None:
        self.specs = specs

    async def audit(self, orchestrator: Any = None, *, reconcile: bool = False) -> ActivationReport:
        statuses: list[ActivationStatus] = []
        for spec in self.specs:
            status = self._check(spec)
            if reconcile and not status.active and spec.auto_start and spec.starter is not None:
                status = await self._reconcile(spec, orchestrator, status)
            statuses.append(status)
        return ActivationReport(generated_at=time.time(), statuses=tuple(statuses))

    def _check(self, spec: ActivationSpec) -> ActivationStatus:
        evidence: dict[str, Any] = {}
        service_hits: list[str] = []
        task_hits: list[str] = []
        try:
            from core.container import ServiceContainer

            service_status: dict[str, Any] = {}
            for key in spec.service_keys:
                if ServiceContainer.has(key):
                    value = ServiceContainer.get(key, default=None)
                    if value is not None:
                        service_hits.append(key)
                        status_fn = getattr(value, "status", None)
                        if callable(status_fn):
                            try:
                                service_status[key] = self._safe_json(status_fn())
                            except Exception as exc:
                                service_status[key] = {"status_error": repr(exc)}
            if service_status:
                evidence["service_status"] = service_status
        except Exception as exc:
            evidence["service_error"] = repr(exc)
        try:
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            for task in list(tracker.tasks):
                name = task.get_name()
                if any(part in name for part in spec.task_name_contains):
                    if not task.done():
                        task_hits.append(name)
        except Exception as exc:
            evidence["task_error"] = repr(exc)
        evidence["service_hits"] = service_hits
        evidence["task_hits"] = task_hits
        active = bool(service_hits or task_hits)
        if spec.name == "keep_awake":
            try:
                from core.runtime.keep_awake import get_keep_awake_controller

                status = get_keep_awake_controller().status()
                active = status.active or not self._keep_awake_enabled()
                evidence["keep_awake_status"] = status.to_dict()
            except Exception as exc:
                evidence["keep_awake_error"] = repr(exc)
        return ActivationStatus(
            name=spec.name,
            active=active,
            required=spec.required,
            auto_start=spec.auto_start,
            reason=spec.reason,
            evidence=evidence,
        )

    async def _reconcile(self, spec: ActivationSpec, orchestrator: Any, prior: ActivationStatus) -> ActivationStatus:
        try:
            result = spec.starter(orchestrator) if spec.starter else None
            if asyncio.iscoroutine(result):
                result = await result
            status = self._check(spec)
            return ActivationStatus(
                name=status.name,
                active=status.active,
                required=status.required,
                auto_start=status.auto_start,
                reason=status.reason,
                evidence={**status.evidence, "starter_result": self._safe_json(result)},
                reconciled=True,
            )
        except Exception as exc:
            record_degradation("activation_audit", exc)
            return ActivationStatus(
                name=prior.name,
                active=False,
                required=prior.required,
                auto_start=prior.auto_start,
                reason=prior.reason,
                evidence=prior.evidence,
                reconciled=True,
                error=repr(exc),
            )

    def write_report(self, report: ActivationReport, path: str | Path) -> Path:
        target = Path(path)
        atomic_write_text(target, json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @staticmethod
    def _safe_json(value: Any) -> Any:
        try:
            json.dumps(value, default=str)
            return value
        except Exception:
            return repr(value)

    @staticmethod
    def _keep_awake_enabled() -> bool:
        import os

        return os.environ.get("AURA_KEEP_AWAKE", "").strip().lower() in {"1", "true", "yes", "on"}


_instance: ActivationAuditor | None = None


def get_activation_auditor() -> ActivationAuditor:
    global _instance
    if _instance is None:
        _instance = ActivationAuditor()
    return _instance


__all__ = [
    "ActivationSpec",
    "ActivationStatus",
    "ActivationReport",
    "ActivationAuditor",
    "get_activation_auditor",
    "DEFAULT_SPECS",
]
