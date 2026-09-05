"""core/morphogenesis/policy.py — who decides what to propose, and on what.

Three policies, run against the same scenarios so the comparison means
something.

:class:`LocalMorphPolicy` sees one cell's own queue, its bounded-radius
neighbours, and the goal signal every cell gets. It cannot enumerate the
population or read a global pressure table. That restriction is the whole
claim: a system that reorganises usefully from local information is doing
something a scheduler is not.

:class:`CentralPolicy` sees everything and picks the largest imbalance. It is
the honest competitor. If it wins on every scenario, the local one bought
nothing and the report has to say so.

:class:`RandomPolicy` proposes valid transitions at random within the same
bounds. It exists because a topology that changes and improves is not evidence
until a topology that changes arbitrarily fails to.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .graph import EdgeType, MorphEdge, MorphGraph
from .proposal import (
    MorphProposal,
    bind,
    despecialize,
    grow,
    retire,
    route,
    specialize,
    unbind,
)
from .workload import CAPABILITIES, RoutedWorkload, WorkerProfile


@dataclass
class PolicyContext:
    """What a policy is handed each round.

    ``goal_demand`` is the one global thing a local cell may read, because a
    cell in a body does get told what the body is trying to do. It says which
    capabilities the current goal needs, not where the load is or who is
    struggling.
    """

    graph: MorphGraph
    workload: RoutedWorkload
    goal_demand: dict[str, float] = field(default_factory=dict)
    round_index: int = 0
    radius: int = 1

    def local_view(self, cell_id: str) -> dict[str, Any]:
        """Everything one cell is allowed to know."""
        neighbours = sorted(self.graph.neighbours(cell_id, radius=self.radius))
        return {
            "cell_id": cell_id,
            "self": self.workload.local_signals(cell_id),
            "demand": self.workload.local_demand(cell_id),
            "neighbours": {
                n: self.workload.local_signals(n) for n in neighbours
            },
            "neighbour_capabilities": {
                n: list(self.workload.workers[n].capabilities)
                for n in neighbours
                if n in self.workload.workers
            },
            "out_ports": sorted({e.port for e in self.graph.out_edges(cell_id)}),
            "goal_demand": dict(self.goal_demand),
        }


class MorphPolicy(Protocol):
    name: str

    def propose(self, context: PolicyContext) -> list[MorphProposal]: ...


#: The founding population. A seed cell never retires itself: the scenarios
#: need a stable door, and a population that can dissolve its own founders can
#: dissolve itself.
_SEED_IDS = frozenset({"g1", "g2", "g3", "g4", "g5", "g6"})


def _worker_manifest(capability: str, index: int) -> dict[str, Any]:
    return {
        "name": f"w_{capability}_{index}",
        "capabilities": [capability],
        "subsystem": "sandbox",
        "service_rate": 2,
    }


class LocalMorphPolicy:
    """Proposals from a bounded neighbourhood and nothing else.

    Four local rules, in the order a cell would reach for them:

    * an out-edge nothing has used for a while, and it is not the last one
      → unbind it
    * nothing has needed it for long enough, and it was grown rather than
      seeded → ask to be retired
    * work waiting for something it specialized away from, that nothing in
      reach covers → give the capability back
    * work it cannot serve and cannot forward → bind to a neighbour that can
    * a neighbour it never routes to that serves what it needs → bind
    * work needing something no neighbour provides at all → grow that cell
    * its queue deep in one capability it holds, and a visible neighbour
      covers everything it would be giving up → specialize into it
    * still deep after that → spawn a helper

    Binding before spawning is deliberate. Wiring is cheap and reversible;
    population is neither, and a policy that reaches for population first grows
    whenever it is busy.
    """

    name = "local"

    def __init__(
        self,
        *,
        seed: int = 0,
        queue_pressure_bind: float = 0.15,
        queue_pressure_spawn: float = 0.55,
        idle_rounds_before_retire: int = 8,
    ):
        self._rng = random.Random(seed)
        self.queue_pressure_bind = float(queue_pressure_bind)
        self.queue_pressure_spawn = float(queue_pressure_spawn)
        self.idle_rounds_before_retire = int(idle_rounds_before_retire)
        self.idle_rounds_before_unbind = max(3, int(idle_rounds_before_retire) // 2)
        #: How long each cell has had nothing to do. Local state at the cell,
        #: not a register something central keeps.
        self._idle: dict[str, int] = {}
        #: How long each of a cell's out-edges has carried nothing. Without
        #: this the graph only ever gains edges: every rule adds and none
        #: removes, so the shape accumulates every demand it has ever seen.
        self._port_idle: dict[tuple[str, str], int] = {}

    def propose(self, context: PolicyContext) -> list[MorphProposal]:
        out: list[MorphProposal] = []
        for cell_id in sorted(context.workload.workers):
            out.extend(self._propose_for(cell_id, context))
        return out

    def _propose_for(self, cell_id: str, context: PolicyContext) -> list[MorphProposal]:
        view = context.local_view(cell_id)
        signals = view["self"]
        demand: dict[str, int] = view["demand"]

        if demand:
            self._idle[cell_id] = 0
        else:
            self._idle[cell_id] = self._idle.get(cell_id, 0) + 1
            # A cell grown for a demand that has passed is dead weight: it
            # costs a hop to route through and holds capacity the current
            # demand cannot use. Growth without release is a ratchet, so a
            # grown cell that has had nothing to do for long enough asks to go.
            grown = context.workload.workers[cell_id].cell_id not in _SEED_IDS
            if grown and self._idle[cell_id] >= self.idle_rounds_before_retire:
                self._idle[cell_id] = 0
                worker = context.workload.workers[cell_id]
                return [retire(
                    cell_id,
                    proposer=cell_id,
                    manifest_data={
                        "name": cell_id,
                        "capabilities": list(worker.capabilities),
                        "service_rate": worker.service_rate,
                    },
                    subsystem="sandbox",
                    benefit=0.4,
                    rationale=(
                        f"nothing has needed this cell for "
                        f"{self.idle_rounds_before_retire} rounds"
                    ),
                    evidence={"idle_rounds": self.idle_rounds_before_retire},
                )]
            return []
        out: list[MorphProposal] = []
        worker = context.workload.workers.get(cell_id)
        if worker is None:
            return []

        # What this cell can serve *now*, which is not the same as what it
        # holds: a specialized cell keeps its other capabilities on paper and
        # serves none of them. Reading the manifest instead of the rate is how
        # a cell sits on work it cannot do and never notices.
        served_here = {c for c in worker.capabilities if worker.rate_for(c) > 0}
        existing_ports = set(view["out_ports"])
        neighbour_caps: dict[str, list[str]] = view["neighbour_capabilities"]

        # 0: a specialization is only safe while something else covers what it
        # gave up. Work is waiting here for a capability this cell owns, cannot
        # currently serve, and nothing in reach provides. Give the capability
        # back — cheaper than growing a replacement, and it is the only move
        # available when the damage took the cover away.
        if worker.specialization:
            stranded = [
                (count, cap) for cap, count in demand.items()
                if cap in set(worker.capabilities) and cap not in served_here
                and not any(cap in caps for caps in neighbour_caps.values())
            ]
            if stranded:
                count, capability = max(stranded)
                return [despecialize(
                    cell_id,
                    proposer=cell_id,
                    previous=worker.specialization,
                    subsystem="sandbox",
                    benefit=min(1.0, 0.5 + count / 10.0),
                    rationale=(
                        f"{count} task(s) here need {capability}, which this cell gave up "
                        f"to specialize in {worker.specialization} and nothing in reach covers"
                    ),
                    evidence={"stranded": count, "capability": capability},
                )]

        # 1: an out-edge for work that has stopped coming. Wiring costs a hop
        # to consider and an in-degree slot at the far end, and a shape that
        # keeps every route it has ever needed is not a shape, it is a
        # sediment.
        for port in existing_ports:
            key = (cell_id, port)
            if port in demand:
                self._port_idle[key] = 0
                continue
            self._port_idle[key] = self._port_idle.get(key, 0) + 1
            if self._port_idle[key] < self.idle_rounds_before_unbind:
                continue
            stale = [e for e in context.graph.out_edges(cell_id) if e.port == port]
            others = {e.port for e in context.graph.out_edges(cell_id)} - {port}
            if not stale or not others:
                # Never cut the last route out. A cell with no out-edge can
                # only ever hand work back the way it came.
                continue
            self._port_idle[key] = 0
            out.append(unbind(
                stale[0],
                proposer=cell_id,
                subsystem="sandbox",
                benefit=0.3,
                rationale=f"nothing has needed {port} from here for {self.idle_rounds_before_unbind} rounds",
                evidence={"port": port, "idle_rounds": self.idle_rounds_before_unbind},
            ))

        # 2 and 3: bind toward a neighbour that serves what is stuck here.
        blocked = sorted(
            ((count, cap) for cap, count in demand.items() if cap not in served_here),
            reverse=True,
        )
        for count, capability in blocked[:2]:
            if capability in existing_ports:
                continue
            candidates = [n for n, caps in neighbour_caps.items() if capability in caps and n != cell_id]
            if not candidates:
                continue
            # Prefer the least loaded neighbour this cell can see.
            target = min(candidates, key=lambda n: (view["neighbours"][n]["queue_depth"], n))
            out.append(bind(
                cell_id,
                target,
                capability,
                proposer=cell_id,
                subsystem="sandbox",
                benefit=min(1.0, count / 8.0),
                cost=0.05,
                rationale=f"{count} task(s) here need {capability} and nothing leads to it",
                evidence={"local_demand": count, "queue_depth": signals["queue_depth"]},
            ))

        # 4: nothing visible can do this at all — grow it back.
        # This is what a lesion needs. Replicating only capabilities that still
        # exist cannot restore one the damage took entirely, and a population
        # that loses a stage keeps every task forever without it.
        for count, capability in blocked[:1]:
            visible = any(capability in caps for caps in neighbour_caps.values())
            if visible or capability in set(worker.capabilities):
                continue
            new_id = f"r_{capability}_{context.round_index}_{cell_id}"
            out.append(grow(
                _worker_manifest(capability, context.round_index),
                cell_id=new_id,
                attach_from=cell_id,
                port=capability,
                return_port=sorted(served_here)[0] if served_here else "",
                proposer=cell_id,
                parent=cell_id,
                placement="local",
                subsystem="sandbox",
                benefit=min(1.0, 0.4 + count / 8.0),
                cost=0.6,
                rationale=f"{count} task(s) need {capability} and nothing in reach provides it",
                evidence={"local_demand": count, "capability": capability, "regenerating": True},
            ))

        # 5: specialize into the capability this cell is asked for most.
        mine = sorted(
            ((count, cap) for cap, count in demand.items() if cap in served_here),
            reverse=True,
        )
        if mine and signals["queue_pressure"] > self.queue_pressure_bind and not worker.specialization:
            count, capability = mine[0]
            share = count / max(1, sum(demand.values()))
            # Specializing costs this cell its other capabilities. Give one up
            # only where a visible neighbour also holds it — a cell that
            # specializes away from something nothing nearby covers takes the
            # population's only path to that capability with it.
            covered = all(
                any(other in caps for caps in neighbour_caps.values())
                for other in served_here
                if other != capability
            )
            if share >= 0.5 and covered:
                out.append(specialize(
                    cell_id,
                    capability,
                    proposer=cell_id,
                    subsystem="sandbox",
                    benefit=min(1.0, share),
                    rationale=f"{share:.0%} of the work waiting here is {capability}",
                    evidence={"share": round(share, 3), "queue_depth": signals["queue_depth"]},
                ))

        # 6: still overloaded with a capability this cell holds — ask for help.
        if mine and signals["queue_pressure"] >= self.queue_pressure_spawn:
            count, capability = mine[0]
            new_id = f"h_{capability}_{context.round_index}_{cell_id}"
            out.append(grow(
                _worker_manifest(capability, context.round_index),
                cell_id=new_id,
                attach_from=cell_id,
                port=capability,
                return_port=sorted(served_here - {capability})[0] if (served_here - {capability}) else "",
                proposer=cell_id,
                parent=cell_id,
                placement="local",
                subsystem="sandbox",
                benefit=min(1.0, signals["queue_pressure"]),
                cost=0.6,
                rationale=f"queue pressure {signals['queue_pressure']:.2f} on {capability}",
                evidence={"queue_depth": signals["queue_depth"], "capability": capability},
            ))
        return out


class CentralPolicy:
    """The scheduler baseline. Reads the whole system and fixes the worst gap.

    Given the same bounds and the same substrate, this is what the local policy
    has to justify itself against.
    """

    name = "central"

    def __init__(self, *, seed: int = 0, pressure_threshold: float = 0.8):
        self._rng = random.Random(seed)
        self.pressure_threshold = float(pressure_threshold)

    def propose(self, context: PolicyContext) -> list[MorphProposal]:
        pressure = context.workload.pressure_by_capability()
        if not pressure:
            return []
        worst = max(pressure.items(), key=lambda kv: (kv[1], kv[0]))
        capability, value = worst
        if value < self.pressure_threshold:
            return []
        providers = context.workload.providers(capability)
        out: list[MorphProposal] = []

        # Wire every cell holding work for this capability to a provider.
        for cell_id in sorted(context.workload.workers):
            if cell_id in providers:
                continue
            if capability not in context.workload.local_demand(cell_id):
                continue
            if any(e.port == capability for e in context.graph.out_edges(cell_id)):
                continue
            if not providers:
                break
            target = min(providers, key=lambda n: (len(context.workload.queues.get(n, ())), n))
            out.append(bind(
                cell_id, target, capability,
                proposer="central",
                subsystem="sandbox",
                benefit=min(1.0, value / 4.0),
                cost=0.05,
                rationale=f"global pressure on {capability} is {value:.2f}",
                evidence={"global_pressure": round(value, 4)},
            ))

        if providers:
            busiest = max(providers, key=lambda n: (len(context.workload.queues.get(n, ())), n))
            out.append(grow(
                _worker_manifest(capability, context.round_index),
                cell_id=f"c_{capability}_{context.round_index}",
                attach_from=busiest,
                port=capability,
                return_port=sorted(context.workload.workers[busiest].capabilities)[0],
                proposer="central",
                parent=busiest,
                placement="local",
                subsystem="sandbox",
                benefit=min(1.0, value / 3.0),
                cost=0.6,
                rationale=f"global pressure on {capability} is {value:.2f}",
                evidence={"global_pressure": round(value, 4)},
            ))
        return out


class RandomPolicy:
    """Valid transitions chosen at random, under the same bounds.

    The control. A change that helps has to beat this, or "the topology
    changed" was the only thing being measured.
    """

    name = "random"

    def __init__(self, *, seed: int = 0, rate: float = 0.5):
        self._rng = random.Random(seed)
        self.rate = float(rate)

    def propose(self, context: PolicyContext) -> list[MorphProposal]:
        if self._rng.random() > self.rate:
            return []
        cells = sorted(context.workload.workers)
        if len(cells) < 2:
            return []
        source = self._rng.choice(cells)
        target = self._rng.choice([c for c in cells if c != source])
        capability = self._rng.choice(list(context.workload.workers[target].capabilities) or list(CAPABILITIES))
        if self._rng.random() < 0.25:
            return [grow(
                _worker_manifest(capability, context.round_index),
                cell_id=f"x_{capability}_{context.round_index}",
                attach_from=source, port=capability,
                proposer=source, parent=source, placement="local", subsystem="sandbox",
                benefit=0.5, cost=0.6, rationale="random",
            )]
        return [bind(
            source, target, capability,
            proposer=source, subsystem="sandbox",
            benefit=0.5, cost=0.05, rationale="random",
        )]


class FrozenPolicy:
    """Proposes nothing. The fixed-topology ablation."""

    name = "frozen"

    def propose(self, context: PolicyContext) -> list[MorphProposal]:
        return []


class BlindLocalPolicy(LocalMorphPolicy):
    """The local policy with its local signals removed.

    It still holds the goal demand and still proposes, so this separates two
    things a "morphology off" ablation confuses: acting on nothing, and acting
    on the wrong thing.
    """

    name = "blind_local"

    def _propose_for(self, cell_id: str, context: PolicyContext) -> list[MorphProposal]:
        view = context.local_view(cell_id)
        wanted = sorted(context.goal_demand.items(), key=lambda kv: (-kv[1], kv[0]))
        if not wanted:
            return []
        capability = wanted[0][0]
        neighbour_caps: dict[str, list[str]] = view["neighbour_capabilities"]
        candidates = [n for n, caps in neighbour_caps.items() if capability in caps and n != cell_id]
        if not candidates or capability in set(view["out_ports"]):
            return []
        return [bind(
            cell_id, sorted(candidates)[0], capability,
            proposer=cell_id, subsystem="sandbox",
            benefit=0.5, cost=0.05,
            rationale=f"goal names {capability}; local state not consulted",
        )]


def build_policy(name: str, *, seed: int = 0) -> MorphPolicy:
    policies = {
        "local": lambda: LocalMorphPolicy(seed=seed),
        "central": lambda: CentralPolicy(seed=seed),
        "random": lambda: RandomPolicy(seed=seed),
        "frozen": lambda: FrozenPolicy(),
        "blind_local": lambda: BlindLocalPolicy(seed=seed),
    }
    if name not in policies:
        raise ValueError(f"unknown policy {name!r}; have {sorted(policies)}")
    return policies[name]()


__all__ = [
    "BlindLocalPolicy",
    "CentralPolicy",
    "FrozenPolicy",
    "LocalMorphPolicy",
    "MorphPolicy",
    "PolicyContext",
    "RandomPolicy",
    "build_policy",
]
