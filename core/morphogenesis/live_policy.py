"""core/morphogenesis/live_policy.py — what the running instance may develop.

The sandbox measures throughput because its cells carry work. The live cells
do not: they watch services and raise repair signals. Scoring them on
throughput would be scoring a number nothing produces.

What the live ecology is actually for is **observability** — that a subsystem
in trouble is reachable from something that can act on it. So the live shadow
evaluator measures coverage of the current danger field, and the live policy
proposes only what improves it:

* a subsystem under danger that no repair-capable cell can reach → bind one
* a subsystem under danger that no cell anywhere covers → grow an observer
* an observer nothing has needed for a long time → retire it

Three deliberate limits, because this runs inside the live instance:

The population it may grow is one kind of cell — a service observer with the
same health-probe handler every boot cell has. It cannot invent a cell type.

Every non-routine change is measured by :class:`LiveCoverageEvaluator` against
the same field the rest of the layer reads, and CRITICAL changes still need a
governed scope on top.

The score deliberately charges for edges. A shape that binds everything to
everything covers perfectly and is the global singleton this layer was built
to get away from, so coverage that costs unbounded wiring does not read as an
improvement.
"""

from __future__ import annotations

import logging
from typing import Any

from .graph import EdgeType, MorphGraph
from .proposal import MorphProposal, bind, grow, retire
from .types import CellRole

logger = logging.getLogger("Aura.Morphogenesis.LivePolicy")

#: Roles that can act on a subsystem in trouble rather than only notice it.
_ACTING_ROLES = frozenset({CellRole.REPAIR.value, CellRole.GOVERNOR.value, CellRole.ROUTER.value})

#: Field reading above which a subsystem counts as needing cover.
DANGER_FLOOR = 0.25


class LiveCoverageEvaluator:
    """Scores a candidate topology by how much of the trouble it can reach.

    Deliberately not a traffic simulation. There is no offline replica of the
    running instance to replay work against, and inventing one would produce a
    number that looked like a measurement and was not.

    What it does measure is real and checkable: for every subsystem the field
    currently reports as troubled, whether some healthy acting cell has a
    directed path to a cell in that subsystem, weighted by how troubled it is,
    minus a charge per binding.
    """

    #: What one binding costs against the coverage it buys. Set so that a
    #: binding has to cover something genuinely uncovered to pay for itself.
    edge_cost = 0.01

    def __init__(self, runtime: Any):
        self._runtime = runtime

    def __call__(self, graph: MorphGraph, proposal: Any = None) -> float | None:
        runtime = self._runtime
        registry = getattr(runtime, "registry", None)
        field = getattr(runtime, "field", None)
        if registry is None or field is None:
            return None

        cells = {c.cell_id: c for c in registry.active_cells()}
        # A proposal may create a cell that does not exist yet; score the shape
        # it would produce, not the one before it.
        promised: dict[str, tuple[str, str]] = {}
        if proposal is not None:
            for transition in getattr(proposal, "transitions", ()):
                data = getattr(transition, "manifest_data", None) or {}
                if data and transition.subject:
                    promised[transition.subject] = (
                        str(data.get("subsystem", "generic")),
                        str(data.get("role", CellRole.SENSOR.value)),
                    )

        troubled: dict[str, float] = {}
        for cell in cells.values():
            subsystem = cell.manifest.subsystem
            if subsystem in troubled:
                continue
            try:
                need = float(field.need(subsystem))
            except (AttributeError, TypeError, ValueError):
                continue
            if need >= DANGER_FLOOR:
                troubled[subsystem] = need
        if not troubled:
            # Nothing is in trouble, so no shape covers trouble better than
            # another. A constant would let any change through on a tie, so
            # this refuses instead: a quiet system is not the time to
            # reorganise, and there is nothing here to justify it.
            return None

        def actor_role(node: str) -> str:
            if node in promised:
                return promised[node][1]
            cell = cells.get(node)
            if cell is None:
                return ""
            role = cell.manifest.role
            return role.value if hasattr(role, "value") else str(role)

        def subsystem_of(node: str) -> str:
            if node in promised:
                return promised[node][0]
            cell = cells.get(node)
            return cell.manifest.subsystem if cell is not None else ""

        def healthy(node: str) -> bool:
            if node in promised:
                return True
            cell = cells.get(node)
            return cell is not None and float(cell.state.health) >= 0.35

        actors = [
            node for node in graph.nodes()
            if actor_role(node) in _ACTING_ROLES and healthy(node)
        ]
        if not actors:
            # Nothing can act, so nothing is covered. That is a coverage of
            # zero — a measurement — and not an inability to measure. Returning
            # None here refused every proposal in the one state where growing
            # something that can act is obviously the right move: the evaluator
            # declined to score the case it exists for.
            return 0.0 - self.edge_cost * graph.edge_count

        covered = 0.0
        total = 0.0
        for subsystem, need in troubled.items():
            total += need
            targets = [n for n in graph.nodes() if subsystem_of(n) == subsystem]
            if not targets:
                continue
            reachable = any(
                actor == target or graph.path_exists(actor, target)
                for actor in actors
                for target in targets
            )
            if reachable:
                covered += need

        coverage = covered / total if total else 0.0
        return coverage - self.edge_cost * graph.edge_count


class LiveObserverPolicy:
    """Proposals a running instance may make about its own anatomy."""

    name = "live_observer"

    def __init__(self, *, idle_ticks_before_retire: int = 600):
        self.idle_ticks_before_retire = int(idle_ticks_before_retire)

    #: Ports a repair binding may use, best first.
    _REPAIR_PORTS = ("repair", "error", "exception", "danger", "growth")

    @staticmethod
    def _out_ports(cell: Any) -> set[str]:
        return {str(v) for v in cell.manifest.emits} | {
            str(v) for v in cell.manifest.capabilities
        }

    @staticmethod
    def _in_ports(cell: Any) -> set[str]:
        return {str(v) for v in cell.manifest.consumes} | {
            str(v) for v in cell.manifest.capabilities
        }

    @staticmethod
    def _observer_manifest(subsystem: str, port: str) -> Any:
        """The one kind of cell this policy may grow.

        Built in one place so the id the governor commits and the id the
        registry assigns are the same string by construction rather than by
        two pieces of code agreeing.
        """
        from .types import CellManifest

        consumes = ["error", "exception", "danger", "repair"]
        if port and port not in consumes:
            consumes.append(str(port))
        return CellManifest(
            name=f"observer_{subsystem}",
            role=CellRole.REPAIR,
            subsystem=str(subsystem),
            capabilities=[str(subsystem), "health_probe"],
            consumes=consumes,
            emits=["repair"],
            criticality=0.6,
            metadata={"grown_by": "live_observer_policy"},
        )

    @staticmethod
    def _port_between(source: Any, target: Any) -> str:
        """A port both ends can carry.

        Both halves have to hold. The source declares what it can send and the
        target what it will accept, and a binding satisfying only one of them
        is refused at commit — after the proposal has already cost a
        measurement. Picking from the intersection means a proposal that gets
        as far as being scored is one that could actually be built.
        """
        shared = (
            LiveObserverPolicy._out_ports(source) & LiveObserverPolicy._in_ports(target)
        )
        for candidate in LiveObserverPolicy._REPAIR_PORTS:
            if candidate in shared:
                return candidate
        return sorted(shared)[0] if shared else ""

    def propose(self, runtime: Any) -> list[MorphProposal]:
        registry = getattr(runtime, "registry", None)
        graph = getattr(runtime, "graph", None)
        field = getattr(runtime, "field", None)
        if registry is None or graph is None or field is None:
            return []

        cells = {c.cell_id: c for c in registry.active_cells()}
        if not cells:
            return []

        by_subsystem: dict[str, list[str]] = {}
        actors: list[str] = []
        for cell_id, cell in cells.items():
            by_subsystem.setdefault(cell.manifest.subsystem, []).append(cell_id)
            role = cell.manifest.role
            value = role.value if hasattr(role, "value") else str(role)
            if value in _ACTING_ROLES and float(cell.state.health) >= 0.35:
                actors.append(cell_id)

        out: list[MorphProposal] = []
        for subsystem, members in sorted(by_subsystem.items()):
            try:
                need = float(field.need(subsystem))
            except (AttributeError, TypeError, ValueError):
                continue
            if need < DANGER_FLOOR:
                continue

            reachable = any(
                actor in members or graph.path_exists(actor, member)
                for actor in actors
                for member in members
            )
            if reachable:
                continue

            if actors:
                # Something can act; it just cannot get here. Wiring is the
                # cheap fix and it is reversible.
                actor = sorted(actors)[0]
                target = sorted(members)[0]
                port = self._port_between(cells[actor], cells[target])
                if not port:
                    logger.debug(
                        "morphogenesis: %s and %s share no port, leaving %s uncovered",
                        actor, target, subsystem,
                    )
                    continue
                out.append(bind(
                    actor, target, port,
                    proposer=actor,
                    edge_type=EdgeType.REPAIR,
                    subsystem=subsystem,
                    benefit=min(1.0, need),
                    cost=0.05,
                    rationale=f"{subsystem} reads {need:.2f} on the need field and nothing that can act reaches it",
                    evidence={"need": round(need, 4), "subsystem": subsystem, "port": port},
                ))
                continue

            # Nothing anywhere can act on this. Grow one observer, wired in.
            anchor = sorted(members)[0]
            # One identity rule. The registry names a cell by
            # CellManifest.canonical_id(), so the governor has to use the same
            # id or it commits a graph node, a lineage birth and a substrate
            # placement under a name the registry never uses — and the next
            # population sync deletes the node as unknown and adds the real one
            # as new, forever.
            manifest = self._observer_manifest(subsystem, "")
            new_id = manifest.canonical_id()
            if new_id in cells:
                continue
            # The grown observer is written to accept whatever the anchor can
            # send, so the pair always shares a port.
            anchor_out = self._out_ports(cells[anchor])
            port = next((p for p in self._REPAIR_PORTS if p in anchor_out), "")
            if not port:
                port = sorted(anchor_out)[0] if anchor_out else ""
            if not port:
                continue
            out.append(grow(
                self._observer_manifest(subsystem, port).to_dict(),
                cell_id=new_id,
                attach_from=anchor,
                port=port,
                return_port="repair",
                proposer=anchor,
                parent=anchor,
                edge_type=EdgeType.REPAIR,
                subsystem=subsystem,
                benefit=min(1.0, need),
                cost=0.8,
                rationale=f"{subsystem} reads {need:.2f} on the need field and nothing in the population can act on it",
                evidence={"need": round(need, 4), "subsystem": subsystem},
            ))

        out.extend(self._retire_idle(runtime, cells))
        return out

    def _retire_idle(self, runtime: Any, cells: dict[str, Any]) -> list[MorphProposal]:
        """Grown observers nothing has needed for a long time.

        Only cells this policy grew. A boot cell is registered by
        integration.py every start, so retiring one buys nothing and it
        reappears on the next sync — churn that looks like development.
        """
        out: list[MorphProposal] = []
        lineage = getattr(runtime, "lineage", None)
        for cell_id, cell in sorted(cells.items()):
            if not cell_id.startswith("observer_") or cell.protected:
                continue
            record = lineage.get(cell_id) if lineage is not None else None
            if record is None or record.generation == 0:
                continue
            idle = cell.state.age_ticks - cell.state.activation_count
            if idle < self.idle_ticks_before_retire:
                continue
            out.append(retire(
                cell_id,
                proposer=cell_id,
                manifest_data=cell.manifest.to_dict(),
                subsystem=cell.manifest.subsystem,
                benefit=0.3,
                rationale=f"nothing has needed this observer for {idle} ticks",
                evidence={"idle_ticks": idle},
            ))
        return out


__all__ = ["DANGER_FLOOR", "LiveCoverageEvaluator", "LiveObserverPolicy"]
