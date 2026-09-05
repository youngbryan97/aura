"""core/morphogenesis/workload.py — a computation that can tell what shape it is.

The spec this layer is built against names one way the whole exercise fails:
if two graph states produce identical execution because everything still talks
through a global singleton, the topology was never load-bearing and the result
means nothing.

So the sandbox workload delivers work along declared edges and nowhere else. A
task carries an ordered list of capabilities it needs applied. It enters at an
ingress cell and can only reach a cell that provides its next capability if a
directed, port-matching edge leads there. Cut that edge and the task cannot be
finished. Add a second cell with the same capability and the queue drains twice
as fast, but only if something binds it in.

Four measurable things follow from the shape, and each scenario is scored on
one of them:

* **completion** — whether a path to the needed capabilities exists at all
* **latency** — the summed hop cost of the path actually taken
* **backlog** — what accumulates when a stage's capacity is below its arrival rate
* **loss** — tasks that expire waiting

Nothing here calls a model, touches the network, or reads live state.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import EdgeType, MorphGraph
from .types import stable_digest

#: The capability alphabet the scenarios draw from. Each is a port name, so
#: a cell that provides ``retrieve`` emits ``retrieve`` and a cell that wants
#: retrieved material consumes it.
CAPABILITIES = ("ingest", "retrieve", "recall", "plan", "solve", "synthesize", "verify", "emit")


@dataclass
class WorkloadTask:
    """One unit of work moving through the population."""

    task_id: str
    stages: tuple[str, ...]
    created_step: int
    deadline_steps: int = 60
    stage_index: int = 0
    at_cell: str = ""
    latency_cost: float = 0.0
    hops: int = 0
    waited: int = 0
    done: bool = False
    failed: str = ""

    @property
    def next_stage(self) -> str:
        if self.stage_index >= len(self.stages):
            return ""
        return self.stages[self.stage_index]

    @property
    def expired_at(self) -> int:
        return self.created_step + self.deadline_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stages": list(self.stages),
            "stage_index": self.stage_index,
            "at_cell": self.at_cell,
            "latency_cost": round(self.latency_cost, 4),
            "hops": self.hops,
            "waited": self.waited,
            "done": self.done,
            "failed": self.failed,
        }


@dataclass
class WorkloadMetrics:
    """What a run produced. Every field is counted, none is estimated."""

    admitted: int = 0
    completed: int = 0
    expired: int = 0
    unroutable: int = 0
    total_latency: float = 0.0
    total_sojourn: float = 0.0
    total_hops: int = 0
    total_wait: int = 0
    backlog_samples: list[int] = field(default_factory=list)
    per_stage_backlog: dict[str, int] = field(default_factory=dict)
    service_events: int = 0

    @property
    def completion_rate(self) -> float:
        return self.completed / self.admitted if self.admitted else 0.0

    @property
    def mean_latency(self) -> float:
        """Hop cost only — how far the work travelled."""
        return self.total_latency / self.completed if self.completed else 0.0

    @property
    def mean_sojourn(self) -> float:
        """End to end: travel plus every step spent waiting in a queue.

        Hop cost alone cannot see a backlog. A shape that makes work wait ten
        times longer moves it exactly as far, so scoring on hops scores
        congestion and free flow identically.
        """
        return self.total_sojourn / self.completed if self.completed else 0.0

    @property
    def mean_hops(self) -> float:
        return self.total_hops / self.completed if self.completed else 0.0

    @property
    def mean_backlog(self) -> float:
        return sum(self.backlog_samples) / len(self.backlog_samples) if self.backlog_samples else 0.0

    @property
    def peak_backlog(self) -> int:
        return max(self.backlog_samples, default=0)

    @property
    def loss_rate(self) -> float:
        lost = self.expired + self.unroutable
        return lost / self.admitted if self.admitted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "completed": self.completed,
            "expired": self.expired,
            "unroutable": self.unroutable,
            "completion_rate": round(self.completion_rate, 5),
            "mean_latency": round(self.mean_latency, 4),
            "mean_sojourn": round(self.mean_sojourn, 4),
            "mean_hops": round(self.mean_hops, 4),
            "mean_backlog": round(self.mean_backlog, 4),
            "peak_backlog": self.peak_backlog,
            "loss_rate": round(self.loss_rate, 5),
            "service_events": self.service_events,
            "per_stage_backlog": dict(sorted(self.per_stage_backlog.items())),
        }

    def merge(self, other: WorkloadMetrics) -> None:
        self.admitted += other.admitted
        self.completed += other.completed
        self.total_sojourn += other.total_sojourn
        self.expired += other.expired
        self.unroutable += other.unroutable
        self.total_latency += other.total_latency
        self.total_hops += other.total_hops
        self.total_wait += other.total_wait
        self.backlog_samples.extend(other.backlog_samples)
        self.service_events += other.service_events
        for key, value in other.per_stage_backlog.items():
            self.per_stage_backlog[key] = self.per_stage_backlog.get(key, 0) + value


@dataclass
class WorkerProfile:
    """What a cell can do for the workload.

    ``service_rate`` is tasks per step. ``specialization`` multiplies it for
    one capability and divides it for the rest, so specializing is a real
    trade rather than a label: a cell that specializes in ``solve`` gets worse
    at ``retrieve``, and a population that specializes wrongly performs worse
    than one that did not.
    """

    cell_id: str
    capabilities: tuple[str, ...]
    service_rate: int = 2
    specialization: str = ""
    specialization_gain: float = 2.5
    healthy: bool = True

    def rate_for(self, capability: str) -> int:
        if capability not in self.capabilities or not self.healthy:
            return 0
        if not self.specialization:
            return self.service_rate
        if capability == self.specialization:
            return max(1, int(round(self.service_rate * self.specialization_gain)))
        return max(0, int(self.service_rate // self.specialization_gain))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "capabilities": list(self.capabilities),
            "service_rate": self.service_rate,
            "specialization": self.specialization,
            "healthy": self.healthy,
        }


class RoutedWorkload:
    """Runs tasks over a :class:`MorphGraph`, and only over it.

    The queue lives at a cell, not in a shared pool. Moving a task from one
    cell to another needs an edge whose port is the capability the task needs
    next. That single rule is what makes the topology causal: there is no
    fallback path, no global dispatcher, and no way for a task to arrive
    somewhere nothing bound it to.
    """

    def __init__(
        self,
        graph: MorphGraph,
        *,
        seed: int = 0,
        deadline_steps: int = 60,
    ):
        self.graph = graph
        self.deadline_steps = int(deadline_steps)
        self._rng = random.Random(seed)
        self.workers: dict[str, WorkerProfile] = {}
        self.queues: dict[str, deque[WorkloadTask]] = {}
        self.ingress: list[str] = []
        self.egress: list[str] = []
        self.metrics = WorkloadMetrics()
        self._step = 0
        self._issued = 0
        self.completed_tasks: list[WorkloadTask] = []

    # ── population ──────────────────────────────────────────────────────

    def add_worker(self, profile: WorkerProfile) -> None:
        self.workers[profile.cell_id] = profile
        self.queues.setdefault(profile.cell_id, deque())

    def remove_worker(self, cell_id: str) -> list[WorkloadTask]:
        """Take a worker out and hand back whatever it was holding.

        A lesion drops these on the floor; a planned retirement can requeue
        them. The difference is the caller's, and the metric shows it.
        """
        self.workers.pop(cell_id, None)
        return list(self.queues.pop(cell_id, deque()))

    def set_health(self, cell_id: str, healthy: bool) -> None:
        worker = self.workers.get(cell_id)
        if worker is not None:
            worker.healthy = bool(healthy)

    def providers(self, capability: str) -> list[str]:
        return sorted(
            cell_id for cell_id, w in self.workers.items()
            if capability in w.capabilities and w.healthy
        )

    def port_contract(self) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
        """``(out_ports, in_ports)`` per worker, for the graph's port check.

        A worker may route work of any kind onward, so its out-face accepts the
        whole alphabet. Its in-face carries only what it can actually serve, so
        a binding to a cell that cannot do the job is refused at commit rather
        than discovered by a task that arrives and stalls.
        """
        every = frozenset(CAPABILITIES)
        return {
            cell_id: (every, frozenset(w.capabilities))
            for cell_id, w in self.workers.items()
        }

    # ── task flow ───────────────────────────────────────────────────────

    def admit(self, stages: Sequence[str], *, at: str = "") -> WorkloadTask | None:
        """Put one task into the system at a cell that can start it.

        Arrivals spread across every healthy provider of the first stage, and
        fall back to the declared ingress. Pinning every arrival to one cell
        would make a second door useless however the population reorganises,
        so a spawned intake worker could never show a gain.
        """
        entry = at
        if not entry:
            doors = [d for d in self.providers(stages[0]) if d in self.queues] if stages else []
            doors = doors or [d for d in self.ingress if d in self.queues]
            if doors:
                entry = min(doors, key=lambda d: (len(self.queues[d]), d))
        if not entry or entry not in self.queues:
            self.metrics.admitted += 1
            self.metrics.unroutable += 1
            return None
        self._issued += 1
        task = WorkloadTask(
            task_id=f"t{self._issued:06d}",
            stages=tuple(stages),
            created_step=self._step,
            deadline_steps=self.deadline_steps,
            at_cell=entry,
        )
        self.queues[entry].append(task)
        self.metrics.admitted += 1
        return task

    def _serves_here(self, cell_id: str, capability: str) -> bool:
        worker = self.workers.get(cell_id)
        return worker is not None and worker.rate_for(capability) > 0

    def _forward(self, task: WorkloadTask) -> bool:
        """Move a task one hop toward a cell that can serve its next stage.

        Prefers an out-edge whose port is the needed capability and whose
        target can serve it, choosing among several by edge weight — that
        weighting is what a ROUTE transition changes. Failing that, takes one
        hop along a DATA edge that leads somewhere the capability is reachable
        from, which is how a longer path carries the work when the short one
        is cut.

        Returns False when nothing leads anywhere useful, and the caller keeps
        the task where it is rather than dropping it.
        """
        need = task.next_stage
        if not need:
            return False
        here = task.at_cell
        direct = [
            e for e in self.graph.out_edges(here)
            if e.port == need and self._serves_here(e.target, need)
        ]
        chosen = None
        if direct:
            total = sum(max(0.001, e.weight) for e in direct)
            pick = self._rng.random() * total
            running = 0.0
            for edge in direct:
                running += max(0.001, edge.weight)
                if pick <= running:
                    chosen = edge
                    break
            chosen = chosen or direct[-1]
        else:
            providers = set(self.providers(need))
            relays = [
                e for e in self.graph.out_edges(here, edge_type=EdgeType.DATA)
                if e.target in self.workers and e.target != here
                and (e.target in providers or any(self.graph.path_exists(e.target, p) for p in providers))
            ]
            if relays:
                chosen = relays[self._rng.randrange(len(relays))]
        if chosen is None:
            return False
        task.at_cell = chosen.target
        task.hops += 1
        task.latency_cost += 1.0 + chosen.latency_ms / 1000.0
        self.queues.setdefault(chosen.target, deque()).append(task)
        return True

    def step(self) -> dict[str, Any]:
        """One step: serve what can be served, forward what cannot, expire the rest."""
        self._step += 1
        served = 0
        expired = 0
        moved = 0
        stalled = 0

        for cell_id in sorted(self.queues):
            queue = self.queues[cell_id]
            worker = self.workers.get(cell_id)
            carried: deque[WorkloadTask] = deque()
            served_here: dict[str, int] = {}

            while queue:
                task = queue.popleft()
                if self._step > task.expired_at:
                    task.failed = "deadline"
                    self.metrics.expired += 1
                    self.metrics.total_wait += task.waited
                    expired += 1
                    continue
                need = task.next_stage
                rate = worker.rate_for(need) if worker is not None else 0
                if rate > 0 and served_here.get(need, 0) < rate:
                    served_here[need] = served_here.get(need, 0) + 1
                    task.stage_index += 1
                    self.metrics.service_events += 1
                    served += 1
                    if task.stage_index >= len(task.stages):
                        task.done = True
                        self.metrics.completed += 1
                        self.metrics.total_latency += task.latency_cost
                        self.metrics.total_sojourn += task.latency_cost + task.waited
                        self.metrics.total_hops += task.hops
                        self.metrics.total_wait += task.waited
                        self.completed_tasks.append(task)
                        continue
                    # Served one stage. Stay if the next stage is servable
                    # here too; a hop that leaves a capable cell only adds
                    # latency.
                    if self._serves_here(cell_id, task.next_stage):
                        carried.append(task)
                        continue
                    if self._forward(task):
                        moved += 1
                        continue
                    carried.append(task)
                    continue
                if rate > 0:
                    # Capable but out of budget this step. Waiting here beats
                    # touring the graph, which is what makes a backlog show up
                    # as a backlog instead of as latency.
                    task.waited += 1
                    stalled += 1
                    carried.append(task)
                    continue
                if self._forward(task):
                    moved += 1
                    continue
                task.waited += 1
                stalled += 1
                carried.append(task)
            self.queues[cell_id] = carried

        backlog = sum(len(q) for q in self.queues.values())
        self.metrics.backlog_samples.append(backlog)
        for cell_id, queue in self.queues.items():
            for task in queue:
                stage = task.next_stage or "done"
                self.metrics.per_stage_backlog[stage] = self.metrics.per_stage_backlog.get(stage, 0) + 1
        return {
            "step": self._step,
            "served": served,
            "moved": moved,
            "stalled": stalled,
            "expired": expired,
            "backlog": backlog,
        }

    # ── observation the local policy is allowed ─────────────────────────

    def local_signals(self, cell_id: str) -> dict[str, float]:
        """What one cell can see about itself. No global view.

        A policy given this and its bounded neighbourhood is genuinely local;
        one given :meth:`pressure_by_capability` is not, and the ablation
        harness runs both on purpose.
        """
        queue = self.queues.get(cell_id, deque())
        worker = self.workers.get(cell_id)
        depth = len(queue)
        waiting = sum(1 for t in queue if t.waited > 0)
        blocked = sum(
            1 for t in queue
            if worker is None or worker.rate_for(t.next_stage) <= 0
        )
        return {
            "queue_depth": float(depth),
            "queue_pressure": min(1.0, depth / 12.0),
            "waiting_share": (waiting / depth) if depth else 0.0,
            "blocked_share": (blocked / depth) if depth else 0.0,
            "service_rate": float(worker.service_rate if worker else 0),
            "healthy": 1.0 if (worker and worker.healthy) else 0.0,
        }

    def local_demand(self, cell_id: str) -> dict[str, int]:
        """Which capabilities the tasks sitting at this cell need next."""
        demand: dict[str, int] = {}
        for task in self.queues.get(cell_id, deque()):
            stage = task.next_stage
            if stage:
                demand[stage] = demand.get(stage, 0) + 1
        return demand

    def pressure_by_capability(self) -> dict[str, float]:
        """The global view. Available to the centralized baseline only."""
        demand: dict[str, int] = {}
        for queue in self.queues.values():
            for task in queue:
                stage = task.next_stage
                if stage:
                    demand[stage] = demand.get(stage, 0) + 1
        capacity: dict[str, int] = {}
        for worker in self.workers.values():
            for capability in worker.capabilities:
                capacity[capability] = capacity.get(capability, 0) + worker.rate_for(capability)
        return {
            capability: demand.get(capability, 0) / max(1, capacity.get(capability, 0))
            for capability in set(demand) | set(capacity)
        }

    def status(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "workers": len(self.workers),
            "backlog": sum(len(q) for q in self.queues.values()),
            "metrics": self.metrics.to_dict(),
            "graph_version": self.graph.version,
            "graph_digest": self.graph.snapshot().digest(),
        }

    def signature(self) -> str:
        """A digest of the outcome. Two runs with the same signature computed
        the same thing, which is how a fixed-topology ablation proves it did."""
        return stable_digest(
            self.metrics.completed,
            self.metrics.expired,
            self.metrics.unroutable,
            round(self.metrics.total_latency, 3),
            self.metrics.total_hops,
            length=16,
        )


def make_workers(
    specs: Iterable[tuple[str, Sequence[str]]],
    *,
    service_rate: int = 2,
) -> list[WorkerProfile]:
    return [
        WorkerProfile(cell_id=cell_id, capabilities=tuple(caps), service_rate=service_rate)
        for cell_id, caps in specs
    ]


def task_families() -> dict[str, tuple[str, ...]]:
    """The stage sequences the scenarios draw from.

    ``memory_heavy`` and ``reason_heavy`` differ in which capability they
    demand most, which is what a task shift has to notice. ``unknown`` is not
    named anywhere in the policies and exists so a scenario can present a
    shape nothing was written for.
    """
    return {
        "memory_heavy": ("ingest", "retrieve", "recall", "retrieve", "synthesize", "emit"),
        "reason_heavy": ("ingest", "plan", "solve", "verify", "solve", "emit"),
        "balanced": ("ingest", "retrieve", "plan", "solve", "emit"),
        "verify_heavy": ("ingest", "solve", "verify", "verify", "emit"),
        "unknown": ("ingest", "recall", "verify", "plan", "recall", "emit"),
    }


__all__ = [
    "CAPABILITIES",
    "RoutedWorkload",
    "WorkerProfile",
    "WorkloadMetrics",
    "WorkloadTask",
    "make_workers",
    "task_families",
]
