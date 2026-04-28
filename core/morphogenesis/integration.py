from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
from typing import Any, Dict, Iterable, List, Optional

from .runtime import MorphogeneticRuntime, get_morphogenetic_runtime
from .types import CellManifest, CellRole, MorphogenSignal, SignalKind

logger = logging.getLogger("Aura.Morphogenesis.Integration")


def _safe_get_service(name: str) -> Any:
    try:
        from core.container import ServiceContainer
        return ServiceContainer.get(name, default=None)
    except Exception:
        return None


async def _service_health_handler(cell, signals, field_state):
    """Generic handler: query a service's health/status if available."""
    service_name = cell.manifest.metadata.get("service_name") or cell.manifest.name
    service = _safe_get_service(service_name)
    actions = []
    out_signals = []

    if service is None:
        actions.append({"kind": "service_missing", "service": service_name})
        out_signals.append(
            MorphogenSignal(
                kind=SignalKind.REPAIR,
                source=cell.cell_id,
                subsystem=cell.manifest.subsystem,
                intensity=0.55,
                payload={"service_missing": service_name},
                ttl_ticks=5,
            )
        )
        return {"actions": actions, "signals": out_signals}

    status = None
    for method in ("get_health", "get_status", "status_dict", "status"):
        fn = getattr(service, method, None)
        if callable(fn):
            try:
                value = fn()
                if hasattr(value, "__await__"):
                    value = await value
                status = value
                break
            except Exception as exc:
                record_degradation('integration', exc)
                actions.append({"kind": "health_probe_error", "service": service_name, "method": method, "error": f"{type(exc).__name__}: {exc}"})
                out_signals.append(
                    MorphogenSignal(
                        kind=SignalKind.ERROR,
                        source=cell.cell_id,
                        subsystem=cell.manifest.subsystem,
                        intensity=0.65,
                        payload={"service": service_name, "method": method, "error": str(exc)[:300]},
                        ttl_ticks=6,
                    )
                )
                return {"actions": actions, "signals": out_signals}

    actions.append({"kind": "service_health_probe", "service": service_name, "status": status})
    return {"actions": actions, "signals": out_signals}


def build_default_cells() -> List[CellManifest]:
    """Default cell ecology mapped to Aura's existing architecture.

    These are conservative: they observe and request repair, but do not mutate
    source code.  Each service-specific cell can be formalized into organs
    after repeated co-activation.
    """

    base_consumes = [
        SignalKind.TASK.value,
        SignalKind.ERROR.value,
        SignalKind.EXCEPTION.value,
        SignalKind.DANGER.value,
        SignalKind.REPAIR.value,
        SignalKind.RESOURCE_PRESSURE.value,
        SignalKind.HEARTBEAT.value,
    ]

    specs = [
        ("adaptive_immunity", "resilience", CellRole.REPAIR, True, 0.95),
        ("autonomous_resilience_mesh", "resilience", CellRole.REPAIR, True, 0.90),
        ("state_repository", "state", CellRole.MEMORY, True, 0.95),
        ("episodic_memory", "memory", CellRole.MEMORY, True, 0.85),
        ("llm_router", "llm_router", CellRole.ROUTER, True, 0.85),
        ("liquid_state", "cognition", CellRole.SENSOR, False, 0.65),
        ("homeostasis", "homeostasis", CellRole.GOVERNOR, True, 0.90),
        ("soma", "homeostasis", CellRole.SENSOR, False, 0.70),
        ("qualia_synthesizer", "consciousness", CellRole.SENSOR, False, 0.70),
        ("affect_engine", "affect", CellRole.SENSOR, False, 0.70),
        ("sovereign_browser", "tools", CellRole.EFFECTOR, False, 0.55),
        ("proactive_communication", "social", CellRole.EFFECTOR, False, 0.55),
    ]

    cells: List[CellManifest] = []
    for service_name, subsystem, role, protected, criticality in specs:
        cells.append(
            CellManifest(
                name=service_name,
                role=role,
                subsystem=subsystem,
                capabilities=[service_name, subsystem, "health_probe"],
                consumes=list(base_consumes),
                emits=[SignalKind.REPAIR.value, SignalKind.GROWTH.value, SignalKind.ERROR.value],
                protected=protected,
                criticality=criticality,
                baseline_energy=0.35 + (criticality * 0.25),
                activation_threshold=0.16 if protected else 0.22,
                max_parallel_tasks=1,
                timeout_s=3.5,
                metadata={"service_name": service_name},
            )
        )
    return cells


def register_morphogenesis_services(runtime: Optional[MorphogeneticRuntime] = None) -> MorphogeneticRuntime:
    """Register runtime + default cells with ServiceContainer.

    Call during boot after the ServiceContainer exists, before the main
    orchestrator enters long-running loops.
    """
    rt = runtime or get_morphogenetic_runtime()

    for manifest in build_default_cells():
        rt.registry.register_cell(manifest, handler=_service_health_handler)

    # Add tissue adjacency: these are conservative "nearby tissues."
    rt.field.register_edge("state", "memory", 0.9)
    rt.field.register_edge("memory", "cognition", 0.7)
    rt.field.register_edge("cognition", "llm_router", 0.8)
    rt.field.register_edge("resilience", "state", 0.8)
    rt.field.register_edge("resilience", "llm_router", 0.8)
    rt.field.register_edge("homeostasis", "cognition", 0.75)
    rt.field.register_edge("affect", "consciousness", 0.7)
    rt.field.register_edge("social", "cognition", 0.55)
    rt.field.register_edge("tools", "cognition", 0.5)

    try:
        from core.container import ServiceContainer
        try:
            ServiceContainer.register_instance("morphogenetic_runtime", rt, required=False)
        except TypeError:
            ServiceContainer.register_instance("morphogenetic_runtime", rt)
        logger.info("MorphogeneticRuntime registered in ServiceContainer.")
    except Exception as exc:
        record_degradation('integration', exc)
        logger.debug("ServiceContainer registration skipped: %s", exc)

    return rt


async def start_morphogenesis_runtime(runtime: Optional[MorphogeneticRuntime] = None) -> MorphogeneticRuntime:
    rt = register_morphogenesis_services(runtime)
    await rt.start()

    # Wire bidirectional hooks into existing subsystems so that morphogenesis
    # actually influences routing, repair, resource allocation, module health,
    # autonomous initiative, and long-term development — not just logging.
    try:
        from core.morphogenesis.hooks import wire_all_hooks
        hook_results = await wire_all_hooks()
        logger.info("Morphogenesis hooks: %s", hook_results)
    except Exception as hook_exc:
        record_degradation('integration', hook_exc)
        logger.warning("Morphogenesis hook wiring degraded: %s", hook_exc)

    return rt

