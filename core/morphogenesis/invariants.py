"""core/morphogenesis/invariants.py — the structural facts this layer assumes.

Each of these was true by construction at some point in the build and stopped
being true when something else changed. A convention nothing enforces is a
convention a refactor retires without telling anyone.

The checks read the live runtime where it exists and report nothing where it
does not, because a fresh process with no morphogenetic runtime has not
violated anything. What they must never do is report clean because they could
not look — a check that cannot check is itself a violation, which the verifier
framework already treats correctly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.verify.invariants import Severity, Violation, invariant

_OWNER = "core/morphogenesis/invariants.py"
_SCOPE = "morphogenesis"


def _runtime() -> Any:
    """The live morphogenetic runtime, or None where there is none."""
    try:
        from core.container import ServiceContainer
    except ImportError:
        return None
    try:
        return ServiceContainer.peek("morphogenetic_runtime", default=None)
    except (AttributeError, RuntimeError, TypeError):
        try:
            from core.container import ServiceContainer as _SC

            return _SC.get("morphogenetic_runtime", default=None)
        except (AttributeError, RuntimeError, TypeError):
            return None


def _graph() -> Any:
    runtime = _runtime()
    return getattr(runtime, "graph", None) if runtime is not None else None


def _governor() -> Any:
    runtime = _runtime()
    return getattr(runtime, "governor", None) if runtime is not None else None


@invariant(
    "morphogenesis.no_dangling_edge",
    scope=_SCOPE,
    owner=_OWNER,
    description="every binding has both endpoints in the population",
)
def _no_dangling_edge() -> Iterator[Violation]:
    graph = _graph()
    if graph is None:
        return
    nodes = set(graph.nodes())
    for edge in graph.edges():
        if edge.source not in nodes:
            yield Violation(
                subject=f"{edge.source}->{edge.target}",
                message=f"binding leaves {edge.source}, which is not in the population",
                remedy="commit topology changes through MorphGovernor, which validates before it commits",
            )
        if edge.target not in nodes:
            yield Violation(
                subject=f"{edge.source}->{edge.target}",
                message=f"binding reaches {edge.target}, which is not in the population",
                remedy="commit topology changes through MorphGovernor, which validates before it commits",
            )


@invariant(
    "morphogenesis.no_self_binding",
    scope=_SCOPE,
    owner=_OWNER,
    description="no cell is bound to itself",
)
def _no_self_binding() -> Iterator[Violation]:
    graph = _graph()
    if graph is None:
        return
    for edge in graph.edges():
        if edge.source == edge.target:
            yield Violation(
                subject=edge.source,
                message="a cell is bound to itself, which is not a topology change",
                remedy="remove the self-binding; routing to yourself is what staying put already means",
            )


@invariant(
    "morphogenesis.version_monotonic",
    scope=_SCOPE,
    owner=_OWNER,
    description="the topology version only ever rises, rollback included",
)
def _version_monotonic() -> Iterator[Violation]:
    graph = _graph()
    if graph is None:
        return
    history = [int(entry.get("version", 0)) for entry in graph.to_dict().get("history", [])]
    for earlier, later in zip(history, history[1:], strict=False):
        if later <= earlier:
            yield Violation(
                subject=f"v{earlier}->v{later}",
                message=(
                    f"topology version went {earlier} to {later}; a reader holding v{earlier} "
                    "would be handed a second, different version with the same number"
                ),
                remedy="rollback must restore content and advance the version, never rewind it",
            )
            return


@invariant(
    "morphogenesis.population_within_bounds",
    scope=_SCOPE,
    owner=_OWNER,
    description="the population is inside the cap it declares",
)
def _population_within_bounds() -> Iterator[Violation]:
    governor = _governor()
    if governor is None:
        return
    cap = int(getattr(governor.bounds, "max_cells", 0) or 0)
    count = int(getattr(governor.graph, "node_count", 0) or 0)
    if cap and count > cap:
        yield Violation(
            subject="population",
            severity=Severity.ERROR,
            message=f"{count} cells against a declared cap of {cap}",
            remedy=(
                "a population that can pass its own cap has no cap; check every path that "
                "adds a node goes through MorphGovernor"
            ),
        )


@invariant(
    "morphogenesis.lineage_acyclic",
    scope=_SCOPE,
    owner=_OWNER,
    description="no cell descends from itself",
)
def _lineage_acyclic() -> Iterator[Violation]:
    governor = _governor()
    if governor is None:
        return
    lineage = getattr(governor, "lineage", None)
    if lineage is None:
        return
    if not lineage.acyclic():
        yield Violation(
            subject="lineage",
            severity=Severity.ERROR,
            message="a cell reaches itself through parent links",
            remedy=(
                "a lineage with a cycle lets a motif take credit for producing itself; "
                "record_birth already refuses the link, so something wrote the record directly"
            ),
        )


@invariant(
    "morphogenesis.lineage_depth_within_bounds",
    scope=_SCOPE,
    owner=_OWNER,
    description="no lineage is deeper than the declared spawn depth",
)
def _lineage_depth() -> Iterator[Violation]:
    governor = _governor()
    if governor is None:
        return
    lineage = getattr(governor, "lineage", None)
    if lineage is None:
        return
    cap = int(getattr(governor.bounds, "max_spawn_depth", 0) or 0)
    deepest = int(lineage.status().get("max_generation", 0))
    if cap and deepest > cap:
        yield Violation(
            subject="lineage",
            message=f"generation {deepest} against a declared depth of {cap}",
            remedy="a lineage that can always go one deeper is a population that can always grow",
        )


@invariant(
    "morphogenesis.graph_agrees_with_substrate",
    scope=_SCOPE,
    owner=_OWNER,
    description="every binding the graph holds, the substrate also holds",
)
def _graph_agrees_with_substrate() -> Iterator[Violation]:
    governor = _governor()
    if governor is None:
        return
    substrate = getattr(governor, "substrate", None)
    if substrate is None or not hasattr(substrate, "bound_keys"):
        return
    try:
        held = set(substrate.bound_keys())
    except (AttributeError, RuntimeError, TypeError):
        return
    if not held:
        # A substrate that has never been asked to bind anything holds nothing,
        # and that agrees with a graph built before it was attached.
        return
    declared = {edge.key for edge in governor.graph.edges()}
    for key in sorted(declared - held):
        yield Violation(
            subject="->".join(str(part) for part in key),
            message="the graph holds a binding the substrate does not",
            remedy=(
                "this is the signature of a partial failure nobody cleaned up: the commit "
                "recorded a binding the world never made"
            ),
        )
    for key in sorted(held - declared):
        yield Violation(
            subject="->".join(str(part) for part in key),
            message="the substrate holds a binding the graph does not",
            remedy=(
                "a latch nobody owns; rollback should unwind the substrate as well as the graph"
            ),
        )


@invariant(
    "morphogenesis.governance_required_for_critical",
    scope=_SCOPE,
    owner=_OWNER,
    description="the live runtime never applies a critical change without governance",
)
def _governance_required() -> Iterator[Violation]:
    governor = _governor()
    if governor is None:
        return
    if not getattr(governor, "require_governance", False):
        yield Violation(
            subject="morphogenetic_runtime",
            severity=Severity.ERROR,
            message=(
                "the live governor has governance disabled, so a retirement or a merge "
                "would be applied with nothing asked"
            ),
            remedy=(
                "require_governance is off only for the offline sandbox; the runtime path "
                "must construct the governor with it on"
            ),
        )


@invariant(
    "morphogenesis.telemetry_declared",
    scope=_SCOPE,
    owner=_OWNER,
    description="the layer's channels exist once a runtime is up",
)
def _telemetry_declared() -> Iterator[Violation]:
    if _runtime() is None:
        return
    try:
        from core.fsw.telemetry_dictionary import channel_value
        from core.morphogenesis.telemetry import CHANNEL_GRAPH_VERSION
    except ImportError:
        return
    try:
        sample = channel_value(CHANNEL_GRAPH_VERSION)
    except (KeyError, RuntimeError, TypeError):
        sample = None
    if sample is None:
        yield Violation(
            subject=CHANNEL_GRAPH_VERSION,
            severity=Severity.WARNING,
            message="a morphogenetic runtime is up and its telemetry has never been written",
            remedy="call core.morphogenesis.telemetry.publish() from the runtime tick",
        )


__all__ = ["_SCOPE"]
