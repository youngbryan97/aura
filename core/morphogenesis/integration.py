from __future__ import annotations

import asyncio
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


async def service_health_handler(cell, signals, field_state):
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
        rt.registry.register_cell(manifest, handler=service_health_handler)

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


#: Kept under the old private name for anything that imported it.
_service_health_handler = service_health_handler


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

        # The envelope the governor holds the whole topology to, applied to the
        # founding bindings as well.
        #
        # This seed added every adjacency pair in both directions and checked
        # nothing but the graph's own degree caps, so the boot it is supposed to
        # make healthy ended with 258 bindings against a ceiling of 256, and
        # `morphogenesis.edges` reported red_high from the first tick of every
        # process (LIVE 21, 23 and 28 September, the same 258 each time, 53
        # cells). The population sync already clips to this envelope; the seed
        # that runs before it did not.
        #
        # What the seed exists for is connectivity: without it every cell is its
        # own component and the partition channel goes red on a healthy boot.
        # So the bindings that join two pieces into one go in first, whatever
        # their place in the adjacency table; the rest of the forward edges are
        # density, and the reverse ones only make a binding mutual. Clipping in
        # table order instead left the last subsystems in the table unbound —
        # nine components, measured, which is the very alarm this seed prevents.
        envelope = max(0, int(rt.governor.bounds.max_edges))
        clipped = 0

        def seed(scratch: Any) -> None:
            nonlocal clipped
            for cell_ids in by_subsystem.values():
                for cell_id in cell_ids:
                    scratch.add_node(cell_id)

            # And the per-cell degree budget the graph validates against. The
            # seed did not know it either: a transaction that passes it raises
            # and `rt.graph.transaction` changes nothing, so the whole boot
            # topology was abandoned and recorded as a MARGINAL fault — every
            # cell its own component, which is the state this seed exists to
            # avoid. Measured while reordering the clip: in-degree 22 at one
            # cell against a cap of 16.
            out_degree: dict[str, int] = {}
            in_degree: dict[str, int] = {}
            for edge in scratch.edges.values():
                out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
                in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
            max_out = rt.graph.max_out_degree
            max_in = rt.graph.max_in_degree

            def admit(source: str, target: str, weight: float) -> bool:
                # Counted from the scratch itself, which carries whatever the
                # graph already holds and collapses a binding proposed twice.
                # A counter of calls would have done neither.
                nonlocal clipped
                if (
                    len(scratch.edges) >= envelope
                    or out_degree.get(source, 0) >= max_out
                    or in_degree.get(target, 0) >= max_in
                ):
                    clipped += 1
                    return False
                scratch.add_edge(MorphEdge(
                    source=source, target=target,
                    edge_type=EdgeType.OBSERVE, weight=weight,
                ))
                out_degree[source] = out_degree.get(source, 0) + 1
                in_degree[target] = in_degree.get(target, 0) + 1
                return True

            wanted: list[tuple[str, str, float]] = [
                (source, target, weight)
                for left, right, weight in _TISSUE_ADJACENCY
                for source in by_subsystem.get(left, ())
                for target in by_subsystem.get(right, ())
                if source != target
            ]

            piece: dict[str, str] = {}

            def root(node: str) -> str:
                while piece.setdefault(node, node) != node:
                    piece[node] = piece[piece[node]]
                    node = piece[node]
                return node

            joining: list[tuple[str, str, float]] = []
            density: list[tuple[str, str, float]] = []
            for source, target, weight in wanted:
                left_root, right_root = root(source), root(target)
                if left_root == right_root:
                    density.append((source, target, weight))
                    continue
                piece[left_root] = right_root
                joining.append((source, target, weight))

            made: list[tuple[str, str, float]] = []
            for source, target, weight in joining + density:
                if admit(source, target, weight):
                    made.append((source, target, weight))
            for source, target, weight in made:
                admit(target, source, weight)

        rt.graph.transaction(seed, cause="boot_topology")
        if clipped:
            logger.info(
                "🧬 Morphogenesis seed left %d reverse binding(s) unmade at the "
                "envelope of %d.", clipped, envelope,
            )
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


#: Modules the tick reaches for lazily. Importing one costs a few hundred
#: milliseconds of compile and file I/O the first time, and the tick that
#: happens to be first pays all of it on the event loop.
#:
#: Measured 2026-09-03: first tick after signals arrived 827ms, every tick
#: after it under 23ms. Boot is where a few hundred milliseconds is already
#: expected and where nothing is waiting on a turn.
_WARM_IMPORTS: tuple[str, ...] = (
    "core.runtime.receipts",
    "core.memory.episodic_memory",
    "core.adaptation.adaptive_immunity",
    "core.resilience.stability_guardian",
    "core.runtime.self_healing",
    "core.morphogenesis.telemetry",
    "core.morphogenesis.invariants",
    "core.morphogenesis.live_policy",
    "core.runtime.resource_observation",
    "core.fsw.telemetry_dictionary",
    "core.verify.invariants",
)


def _warm_lazy_imports() -> list[str]:
    """Import what the tick will need, so no tick is the one that pays.

    Also takes the first resource reading. psutil's first sample opens and
    parses the host's counters, and the tick that takes it wears the cost.
    """
    import importlib

    warmed: list[str] = []
    for name in _WARM_IMPORTS:
        try:
            importlib.import_module(name)
            warmed.append(name)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("morphogenesis warm import skipped for %s: %s", name, exc)
    try:
        from core.runtime.resource_observation import get_resource_observer

        observer = get_resource_observer()
        observer.memory()
        observer.compute()
        warmed.append("resource_observation:first_sample")
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("morphogenesis resource warm-up skipped: %s", exc)
    return warmed


async def start_morphogenesis_runtime(runtime: MorphogeneticRuntime | None = None) -> MorphogeneticRuntime:
    rt = register_morphogenesis_services(runtime)

    # Off the loop: these are file reads and bytecode compilation, and doing
    # them here is the difference between one 800ms stall on whichever tick
    # happens to need them first and none at all.
    try:
        warmed = await asyncio.to_thread(_warm_lazy_imports)
        logger.info("🧬 Morphogenesis warmed %d lazy import(s).", len(warmed))
    except (ImportError, RuntimeError, TypeError) as exc:
        _record_morphogenesis_integration_degradation(
            exc,
            action="started morphogenesis without warming its lazy imports",
            severity="warning",
        )

    await rt.start()

    # Wire bidirectional hooks into existing subsystems so that morphogenesis
    # actually influences routing, repair, resource allocation, module health,
    # autonomous initiative, and long-term development — not just logging.
    try:
        # Importing registers the layer's invariants with core.verify.
        from core.morphogenesis import invariants as _morph_invariants  # noqa: F401
        from core.morphogenesis import telemetry as _morph_telemetry

        _morph_telemetry.declare(rt.governor.bounds.to_dict())
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
