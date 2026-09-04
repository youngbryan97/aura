from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

from .graph import EdgeType, MorphEdge
from .runtime import MorphogeneticRuntime, get_morphogenetic_runtime
from .types import CellManifest, CellRole, MorphogenSignal, SignalKind

logger = logging.getLogger("Aura.Morphogenesis.Integration")


def _record_morphogenesis_integration_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "morphogenesis.integration",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation(
                "morphogenesis.integration",
                error,
                severity=severity,
                action=action,
            )
        except TypeError:
            logger.debug(
                "Morphogenesis integration degradation could not be recorded: %s",
                signature_exc,
            )


def _safe_get_service(name: str) -> Any:
    try:
        from core.container import ServiceContainer
        return ServiceContainer.get(name, default=None)
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_morphogenesis_integration_degradation(
            exc,
            action="returned missing service so morphogenesis cell can emit repair signal",
            severity="warning",
            extra={"service": name},
        )
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
            except (RuntimeError, AttributeError, TypeError) as exc:
                _record_morphogenesis_integration_degradation(
                    exc,
                    action="emitted morphogenesis error signal after service health probe failed",
                    severity="degraded",
                    extra={"service": service_name, "method": method},
                )
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


def build_default_cells() -> list[CellManifest]:
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

    cells: list[CellManifest] = []
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


def register_morphogenesis_services(runtime: MorphogeneticRuntime | None = None) -> MorphogeneticRuntime:
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

    _seed_topology(rt)

    try:
        from core.container import ServiceContainer
        try:
            ServiceContainer.register_instance("morphogenetic_runtime", rt, required=False)
        except TypeError:
            ServiceContainer.register_instance("morphogenetic_runtime", rt)
        logger.info("MorphogeneticRuntime registered in ServiceContainer.")
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_morphogenesis_integration_degradation(
            exc,
            action="continued with local morphogenesis runtime after ServiceContainer registration failed",
            severity="degraded",
        )
        logger.debug("ServiceContainer registration skipped: %s", exc)

    return rt


#: Which subsystems watch which. The field already carried this as diffusion
#: adjacency; the graph carries it as reachability, so "who could notice this
#: cell's distress" has an answer that is not "everyone, through one queue".
_TISSUE_ADJACENCY: tuple[tuple[str, str, float], ...] = (
    ("state", "memory", 0.9),
    ("memory", "cognition", 0.7),
    ("cognition", "llm_router", 0.8),
    ("resilience", "state", 0.8),
    ("resilience", "llm_router", 0.8),
    ("homeostasis", "cognition", 0.75),
    ("affect", "consciousness", 0.7),
    ("social", "cognition", 0.55),
    ("tools", "cognition", 0.5),
    ("resilience", "homeostasis", 0.8),
    ("resilience", "memory", 0.7),
    ("consciousness", "cognition", 0.7),
)


def _seed_topology(rt: MorphogeneticRuntime) -> None:
    """Give the live population its founding bindings.

    Without these every cell is its own connected component, the partition
    channel goes red on the first tick, and an alarm that fires on a healthy
    boot is an alarm nobody reads.

    The edges are OBSERVE, which is the honest type for this ecology: these
    cells watch services and raise repair signals. They do not hand work to
    each other, and typing them as data flow would claim a pipeline that does
    not exist. OBSERVE is also the one type exempt from the port contract,
    because watching something needs no agreement from the thing watched.
    """
    if not rt.config.topology_enabled:
        return
    try:
        by_subsystem: dict[str, list[str]] = {}
        for cell in rt.registry.active_cells():
            by_subsystem.setdefault(cell.manifest.subsystem, []).append(cell.cell_id)
            rt.substrate.place(cell.cell_id)
            rt.lineage.seed(cell.cell_id, cause="boot")
            rt.governor.set_capabilities(cell.cell_id, cell.manifest.capabilities)

        def seed(scratch: Any) -> None:
            for cell_ids in by_subsystem.values():
                for cell_id in cell_ids:
                    scratch.add_node(cell_id)
            for left, right, weight in _TISSUE_ADJACENCY:
                for source in by_subsystem.get(left, ()):
                    for target in by_subsystem.get(right, ()):
                        if source == target:
                            continue
                        scratch.add_edge(MorphEdge(
                            source=source, target=target,
                            edge_type=EdgeType.OBSERVE, weight=weight,
                        ))
                        scratch.add_edge(MorphEdge(
                            source=target, target=source,
                            edge_type=EdgeType.OBSERVE, weight=weight,
                        ))

        rt.graph.transaction(seed, cause="boot_topology")
        for edge in rt.graph.edges():
            rt.substrate.bind(edge)
        logger.info(
            "🧬 Morphogenesis topology seeded: %d cell(s), %d binding(s), %d component(s).",
            rt.graph.node_count, rt.graph.edge_count, len(rt.graph.components()),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _record_morphogenesis_integration_degradation(
            exc,
            action="left the morphogenesis topology unseeded after boot wiring failed",
            severity="degraded",
        )


async def start_morphogenesis_runtime(runtime: MorphogeneticRuntime | None = None) -> MorphogeneticRuntime:
    rt = register_morphogenesis_services(runtime)
    await rt.start()

    # Wire bidirectional hooks into existing subsystems so that morphogenesis
    # actually influences routing, repair, resource allocation, module health,
    # autonomous initiative, and long-term development — not just logging.
    try:
        # Importing registers the layer's invariants with core.verify.
        from core.morphogenesis import invariants as _morph_invariants  # noqa: F401
        from core.morphogenesis import telemetry as _morph_telemetry

        _morph_telemetry.declare()
    except (ImportError, AttributeError, RuntimeError) as verify_exc:
        _record_morphogenesis_integration_degradation(
            verify_exc,
            action="started morphogenesis without its invariants or telemetry declared",
            severity="warning",
        )

    try:
        from core.morphogenesis.hooks import wire_all_hooks
        hook_results = await wire_all_hooks()
        if hasattr(rt, "mark_hooks_wired"):
            rt.mark_hooks_wired(hook_results, ok=True)
        logger.info("Morphogenesis hooks: %s", hook_results)
    except (ImportError, AttributeError, RuntimeError) as hook_exc:
        if hasattr(rt, "mark_hooks_wired"):
            rt.mark_hooks_wired({"error": f"{type(hook_exc).__name__}: {hook_exc}"}, ok=False)
        _record_morphogenesis_integration_degradation(
            hook_exc,
            action="kept morphogenesis runtime alive and emitted hook-wiring error signal",
            severity="degraded",
        )
        try:
            rt.emit_signal(
                MorphogenSignal(
                    kind=SignalKind.ERROR,
                    source="morphogenesis.integration",
                    subsystem="morphogenesis",
                    intensity=0.72,
                    payload={"hook_error": f"{type(hook_exc).__name__}: {hook_exc}"},
                    ttl_ticks=6,
                )
            )
        except (AttributeError, RuntimeError, TypeError) as signal_exc:
            _record_morphogenesis_integration_degradation(
                signal_exc,
                action="kept morphogenesis runtime alive after hook error signal emission failed",
                severity="warning",
            )
        logger.warning("Morphogenesis hook wiring degraded: %s", hook_exc)

    return rt
