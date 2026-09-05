"""core/morphogenesis/bridge.py — what the rest of Aura tells this layer, and
what it can ask back.

A morphogenetic layer with no demand input develops nothing: pressure has to
come from somewhere real or the population sits at whatever boot registered.
This module is the seam, and it is deliberately small. Everything here reads
existing systems and writes signals; nothing here decides a topology, because
that is the governor's job and there should be exactly one way in.

Three connections are live:

**Demand.** ``core.conversation.capability_condition.needed_capabilities``
already answers "what does this turn reach for", in cue order, and is the
router's own definition rather than a second one invented here. That is the
goal demand a cell may read.

**Distress.** A degradation or an exception in a subsystem is a danger signal
at that tissue. The runtime already accepted these; what was missing was
anything that could act on them structurally.

**Reachability out.** ``observers_of`` answers "who could notice this cell's
trouble", which before the graph had no answer but "everyone, through one
global queue".

One connection is deliberately *not* made. ``core/language`` is the semantics
of language — concepts, propositions, what a reply commits to. It is a source
of demand, which is why ``demand_from_message`` goes through the capability
router. It is not a substrate the population could run on, and wiring it as
one would be a category error dressed as integration.

Program DNA is likewise left alone. ``ProgramDNAGenome`` reconstructs source
from evidence; a ``MorphMotif`` is a prior over which capabilities to have and
how to wire them. Both are called a genotype in the literature and they sit at
different levels, so :func:`describe_genotype_relationship` states the
relationship rather than forcing a link that would make neither mean anything.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

from .graph import EdgeType
from .types import MorphogenSignal, SignalKind

logger = logging.getLogger("Aura.Morphogenesis.Bridge")

#: Which tissue a named capability belongs to. Used to turn "this turn needs
#: web_search" into pressure at the subsystem that would serve it.
_CAPABILITY_TISSUE: dict[str, str] = {
    "web_search": "tools",
    "browser": "tools",
    "desktop": "tools",
    "email_adapter": "social",
    "vision": "consciousness",
    "memory": "memory",
    "recall": "memory",
    "initiative": "cognition",
    "code_execution": "tools",
    "file_access": "state",
}


def _runtime() -> Any:
    try:
        from core.container import ServiceContainer
    except ImportError:
        return None
    try:
        return ServiceContainer.peek("morphogenetic_runtime", default=None)
    except (AttributeError, RuntimeError, TypeError):
        return None


def demand_from_message(user_message: Any) -> dict[str, float]:
    """What this turn reaches for, as a demand over capabilities.

    Delegates to the capability router rather than matching cues again here.
    Two definitions of "this turn needs the desktop" is one definition too
    many, and the second one is always the one that goes stale.
    """
    try:
        from core.conversation.capability_condition import needed_capabilities
    except ImportError:
        return {}
    try:
        found = needed_capabilities(user_message)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "morphogenesis.bridge", exc, severity="warning",
            action="returned no capability demand after the capability router failed",
        )
        return {}
    # Cue order is a ranking, so the first-named capability weighs most.
    total = len(found)
    return {
        str(name): round(1.0 - (index / (total + 1)), 4)
        for index, name in enumerate(found)
    }


def announce_demand(user_message: Any, *, intensity: float = 0.5) -> list[str]:
    """Turn a turn's capability demand into task pressure at the right tissues.

    Returns the tissues signalled, so a caller can tell whether anything
    listened rather than assuming it did.
    """
    runtime = _runtime()
    if runtime is None:
        return []
    demand = demand_from_message(user_message)
    if not demand:
        return []
    signalled: list[str] = []
    try:
        for capability, weight in demand.items():
            tissue = _CAPABILITY_TISSUE.get(capability, "cognition")
            runtime.emit_signal(MorphogenSignal(
                kind=SignalKind.TASK,
                source="bridge.capability_demand",
                subsystem=tissue,
                intensity=min(1.0, max(0.05, float(intensity) * float(weight))),
                payload={"capability": capability, "weight": weight},
                ttl_ticks=6,
            ))
            signalled.append(tissue)
        setter = getattr(runtime, "set_goal_demand", None)
        if callable(setter):
            setter(demand)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "morphogenesis.bridge", exc, severity="warning",
            action="signalled part of a turn's capability demand before failing",
        )
    return signalled


def announce_degradation(subsystem: str, *, detail: str = "", danger: float = 0.6) -> bool:
    """Report a subsystem in trouble as danger at that tissue."""
    runtime = _runtime()
    if runtime is None:
        return False
    try:
        runtime.emit_signal(MorphogenSignal(
            kind=SignalKind.DANGER,
            source="bridge.degradation",
            subsystem=str(subsystem or "global"),
            intensity=min(1.0, max(0.05, float(danger))),
            payload={"detail": str(detail)[:400]},
            ttl_ticks=8,
        ))
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def observers_of(cell_id: str) -> list[str]:
    """Which cells can reach this one, and could notice its trouble.

    Before the graph this question had no answer that meant anything: every
    cell saw every signal through one queue, so "who is watching" was
    "everyone" and therefore nobody in particular.
    """
    runtime = _runtime()
    graph = getattr(runtime, "graph", None) if runtime is not None else None
    if graph is None:
        return []
    try:
        return sorted({
            edge.source for edge in graph.in_edges(cell_id)
            if edge.edge_type in {EdgeType.OBSERVE, EdgeType.REPAIR, EdgeType.CONTROL}
        })
    except (AttributeError, RuntimeError, TypeError):
        return []


def reaches(source_cell: str, target_cell: str) -> bool:
    """Whether one cell can reach another along declared bindings."""
    runtime = _runtime()
    graph = getattr(runtime, "graph", None) if runtime is not None else None
    if graph is None:
        return False
    try:
        return bool(graph.path_exists(source_cell, target_cell))
    except (AttributeError, RuntimeError, TypeError):
        return False


def topology_summary() -> dict[str, Any]:
    """A compact shape report for health surfaces and the system route."""
    runtime = _runtime()
    graph = getattr(runtime, "graph", None) if runtime is not None else None
    if graph is None:
        return {"online": False}
    try:
        components = graph.components()
        return {
            "online": True,
            "version": graph.version,
            "digest": graph.snapshot().digest(),
            "cells": graph.node_count,
            "bindings": graph.edge_count,
            "components": len(components),
            "largest_component": max((len(c) for c in components), default=0),
            "partitioned": len(components) > 1,
        }
    except (AttributeError, RuntimeError, TypeError) as exc:
        record_degradation(
            "morphogenesis.bridge", exc, severity="warning",
            action="reported morphogenesis topology as unavailable",
        )
        return {"online": False, "error": f"{type(exc).__name__}: {exc}"}


def isolated_cells() -> list[str]:
    """Cells nothing binds to in either direction.

    An isolated cell can neither receive work nor be noticed when it fails.
    In a population that is supposed to have organised itself, one of these is
    either a cell development has not reached yet or one it has abandoned, and
    the difference is worth a look either way.
    """
    runtime = _runtime()
    graph = getattr(runtime, "graph", None) if runtime is not None else None
    if graph is None:
        return []
    try:
        return sorted(
            node for node in graph.nodes()
            if not graph.out_edges(node) and not graph.in_edges(node)
        )
    except (AttributeError, RuntimeError, TypeError):
        return []


def describe_genotype_relationship() -> dict[str, Any]:
    """Where a motif sits relative to program DNA. Stated, not wired.

    The spec this layer was built against asks whether the existing program-DNA
    machinery can serve as the genotype with the runtime graph as the
    phenotype. It cannot, and saying so is more useful than a bridge that
    typechecks and means nothing.

    ``ProgramDNAGenome`` reconstructs *source* from evidence: it answers "what
    code would produce this behaviour". A ``MorphMotif`` is a prior over
    *arrangement*: which capabilities to have, how many, and how to wire them.
    One is about what a part is; the other is about how parts are put together.
    A system could hold both, and the honest composition is that program DNA
    supplies the cells a motif then arranges — which is Phase 3 work and is not
    implemented.
    """
    return {
        "motif": {
            "level": "arrangement",
            "answers": "which capabilities to have, how many, and how to bind them",
            "module": "core/morphogenesis/motifs.py",
            "validated_by": "beating its own absence on the same workload and seed",
        },
        "program_dna": {
            "level": "implementation",
            "answers": "what source would produce this behaviour",
            "module": "core/self_improvement/program_dna.py",
        },
        "composition": (
            "program DNA supplies a part; a motif arranges parts. Neither is the "
            "other's genotype, and treating them as one would leave both meaning "
            "less than they do apart."
        ),
        "implemented": False,
    }


def substrate_roadmap() -> dict[str, Any]:
    """What a future substrate would have to satisfy, and what exists today.

    Kept as data rather than prose so a later phase can check itself against
    it. Every entry names a real module in this tree; nothing here is a plan
    for something that would have to be invented from nothing.
    """
    return {
        "contract": "core/morphogenesis/substrate.py::SubstrateAdapter",
        "implemented": [
            {
                "name": "SimulationSubstrate",
                "physical": False,
                "models": ["latency", "energy", "partial failure", "unreachability"],
            },
            {
                "name": "LocalRuntimeSubstrate",
                "physical": False,
                "locus": "aura_main",
                "refuses": ["migrate"],
            },
        ],
        "candidates": [
            {
                "phase": 2,
                "name": "process or host placement",
                "existing": ["core/swarm/worker_pool.py", "core/swarm/ray_backend.py"],
                "maps": {
                    "bind": "open a channel between workers",
                    "migrate": "move a worker to another host",
                    "spawn": "start a sandboxed worker",
                },
                "blocked_on": (
                    "a shadow evaluator for the live system; without one every "
                    "non-routine change is refused, which is the correct behaviour "
                    "and also means nothing would move"
                ),
            },
            {
                "phase": 3,
                "name": "reconfigurable logic",
                "existing": [],
                "maps": {"bind": "load a partial bitstream into a region"},
                "blocked_on": "hardware this tree has no access to",
            },
            {
                "phase": 4,
                "name": "docking modules",
                "existing": [],
                "maps": {"bind": "navigate, align, latch, handshake"},
                "blocked_on": "hardware this tree has no access to",
                "note": (
                    "the simulation already models what these have in common — a "
                    "transition that takes time, costs energy, can fail halfway and "
                    "leaves a cell unreachable meanwhile"
                ),
            },
        ],
    }


def audit() -> dict[str, Any]:
    """One report of every seam, live or not. Used by the sandbox CLI."""
    return {
        "topology": topology_summary(),
        "isolated_cells": isolated_cells(),
        "connections": {
            "demand_in": "core/conversation/capability_condition.py::needed_capabilities",
            "distress_in": "core/runtime/errors.py::record_degradation via announce_degradation",
            "routing_out": "core/brain/inference_gate.py via hooks.get_morphogenesis_routing_advice",
            "health_out": "core/resilience/stability_guardian.py via hooks",
            "watchdog": "core/runtime/self_healing.py via hooks",
            "memory_out": "core/memory/episodic_memory.py via hooks",
            "telemetry_out": "core/fsw/telemetry_dictionary.py channels 0x0801-0x080C",
            "invariants": "core/verify/invariants.py scope morphogenesis",
        },
        "not_connected": {
            "core/language": (
                "the semantics of language, not a substrate; reached only as a "
                "demand source through the capability router"
            ),
            "core/self_improvement/program_dna.py": (
                "a different level of genotype; see describe_genotype_relationship"
            ),
            "core/swarm": "the Phase 2 substrate candidate, not yet implemented",
        },
        "genotype": describe_genotype_relationship(),
        "substrate_roadmap": substrate_roadmap(),
    }


def emit_signal_for_capability(capability: str, *, intensity: float = 0.5) -> str:
    """Signal demand for one named capability. Returns the tissue signalled."""
    runtime = _runtime()
    if runtime is None:
        return ""
    tissue = _CAPABILITY_TISSUE.get(str(capability), "cognition")
    try:
        runtime.emit_signal(MorphogenSignal(
            kind=SignalKind.TASK,
            source="bridge.capability",
            subsystem=tissue,
            intensity=min(1.0, max(0.05, float(intensity))),
            payload={"capability": str(capability)},
            ttl_ticks=6,
        ))
        return tissue
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ""


__all__ = [
    "announce_degradation",
    "announce_demand",
    "audit",
    "demand_from_message",
    "describe_genotype_relationship",
    "emit_signal_for_capability",
    "isolated_cells",
    "observers_of",
    "reaches",
    "substrate_roadmap",
    "topology_summary",
]
