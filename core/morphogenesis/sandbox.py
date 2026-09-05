"""core/morphogenesis/sandbox.py — the offline experiments.

Every scenario here is deterministic under a seed, runs with no model, no
network, and no live runtime, and reports a measurement rather than a verdict
about whether morphogenesis is a good idea.

The design rule throughout: a scenario is only worth running if it could come
out against the feature. ``ablation`` exists so that "the topology changed" is
never mistaken for "the topology change helped", and every scenario reports a
frozen-topology arm alongside the adaptive one.

A scenario returns a :class:`ScenarioResult`. Its ``verdict`` is computed from
the numbers by a rule stated per scenario, not chosen after seeing them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .governor import MorphBounds, MorphGovernor
from .graph import EdgeType, MorphEdge, MorphGraph
from .lineage import Lineage
from .motifs import MotifLibrary
from .policy import MorphPolicy, PolicyContext, build_policy
from .proposal import TransitionKind
from .substrate import SimulationSubstrate, SubstratePhysics
from .types import json_safe
from .workload import (
    RoutedWorkload,
    WorkerProfile,
    task_families,
)

SCENARIOS = (
    "task_shift",
    "overload",
    "lesion",
    "partition",
    "oscillating_signal",
    "poisoned_signal",
    "motif_transfer",
    "unknown_topology",
)

ABLATIONS = (
    "none",
    "morphology_off",
    "topology_fixed",
    "local_signals_off",
    "motifs_off",
    "recovery_off",
    "central_scheduler",
    "random_mutation",
)


@dataclass
class ArmResult:
    """One condition of one scenario."""

    label: str
    metrics: dict[str, Any] = field(default_factory=dict)
    graph_versions: int = 0
    applied: int = 0
    rejected: int = 0
    deferred: int = 0
    rolled_back: int = 0
    final_cells: int = 0
    final_edges: int = 0
    energy_spent: float = 0.0
    substrate_energy: float = 0.0
    components: int = 1
    graph_digest: str = ""
    signature: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """The single number arms are compared on.

        Completion dominates; loss and end-to-end sojourn subtract. Stated
        here once so no scenario can pick a favourable metric after seeing its
        result.
        """
        completion = float(self.metrics.get("completion_rate", 0.0))
        loss = float(self.metrics.get("loss_rate", 0.0))
        sojourn = float(self.metrics.get("mean_sojourn", 0.0))
        return completion - 0.5 * loss - 0.02 * sojourn

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "metrics": self.metrics,
            "graph_versions": self.graph_versions,
            "applied": self.applied,
            "rejected": self.rejected,
            "deferred": self.deferred,
            "rolled_back": self.rolled_back,
            "final_cells": self.final_cells,
            "final_edges": self.final_edges,
            "energy_spent": round(self.energy_spent, 5),
            "substrate_energy": round(self.substrate_energy, 5),
            "components": self.components,
            "graph_digest": self.graph_digest,
            "signature": self.signature,
            "detail": json_safe(self.detail),
        }


@dataclass
class ScenarioResult:
    scenario: str
    seed: int
    steps: int
    arms: dict[str, ArmResult] = field(default_factory=dict)
    verdict: str = "inconclusive"
    verdict_rule: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    replay_path: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict.startswith("pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "steps": self.steps,
            "verdict": self.verdict,
            "verdict_rule": self.verdict_rule,
            "passed": self.passed,
            "arms": {k: v.to_dict() for k, v in sorted(self.arms.items())},
            "measurements": json_safe(self.measurements),
            "duration_s": round(self.duration_s, 4),
            "replay_path": self.replay_path,
        }

    def summary(self) -> str:
        arms = "  ".join(
            f"{name}={arm.score:+.3f}" for name, arm in sorted(self.arms.items())
        )
        return f"{self.scenario:<20} {self.verdict:<28} {arms}"


# ── the seed population every scenario starts from ──────────────────────

#: Generic cells, deliberately undifferentiated. Each holds two capabilities
#: and no specialization, so any structure in the result developed rather than
#: arriving in the setup.
SEED_POPULATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("g1", ("ingest", "retrieve")),
    ("g2", ("retrieve", "recall")),
    ("g3", ("plan", "solve")),
    ("g4", ("solve", "verify")),
    ("g5", ("synthesize", "emit")),
    ("g6", ("recall", "synthesize")),
)

#: The bindings the population starts with: a bare chain, enough to be
#: connected and not enough to serve anything well.
SEED_EDGES: tuple[tuple[str, str, str], ...] = (
    ("g1", "g2", "retrieve"),
    ("g2", "g3", "plan"),
    ("g3", "g4", "solve"),
    ("g4", "g5", "emit"),
    ("g5", "g6", "synthesize"),
)


class Harness:
    """One run: a graph, a substrate, a governor, a workload, and a policy."""

    def __init__(
        self,
        *,
        seed: int = 0,
        policy_name: str = "local",
        bounds: MorphBounds | None = None,
        physics: SubstratePhysics | None = None,
        motifs: MotifLibrary | None = None,
        allow_recovery: bool = True,
        radius: int = 1,
        require_governance: bool = False,
        shadow: bool = True,
        deadline_steps: int = 26,
        demand_window_len: int = 6,
    ):
        self.seed = int(seed)
        self.policy_name = policy_name
        self.policy: MorphPolicy = build_policy(policy_name, seed=seed)
        self.graph = MorphGraph(max_nodes=64, max_edges=256)
        self.substrate = SimulationSubstrate(seed=seed, physics=physics, max_cells=64)
        self.workload = RoutedWorkload(self.graph, seed=seed, deadline_steps=deadline_steps)
        self.lineage = Lineage(max_generation=(bounds or MorphBounds()).max_spawn_depth)
        self.motifs = motifs
        self.allow_recovery = bool(allow_recovery)
        self.radius = int(radius)
        self.round_index = 0
        self.goal_demand: dict[str, float] = {}
        #: How much of the live backlog the shadow probe carries. Enough to
        #: expose a capacity difference, bounded so the probe stays cheap.
        self.shadow_backlog_cap = 30
        #: The probe keeps admitting at the rate the live system is seeing.
        #: A probe that admits once and then drains completes everything under
        #: every shape, and a comparison where both arms score 1.0 rejects
        #: every proposal for a gain of exactly zero.
        self.shadow_steps = 20
        self.arrivals_per_round = 0.0
        #: The recent demand, not the latest demand. A probe that measures
        #: against whatever arrived last will approve a change that suits this
        #: round and reverse it next round: chasing the instantaneous signal is
        #: what thrash *is*. Scoring against the recent mix means a change good
        #: for only half of an alternating demand cannot clear the band.
        #:
        #: The length is the discrimination. Shorter than a real regime change,
        #: so a sustained shift fills the window and the layer responds to it;
        #: longer than an alternation's period, so a flipping demand stays a
        #: mix and never looks like a shift. Too long and a genuine shift is
        #: damped as if it were noise.
        self.demand_window: deque[tuple[str, ...]] = deque(maxlen=max(1, int(demand_window_len)))
        self.applied_log: list[dict[str, Any]] = []

        self.governor = MorphGovernor(
            self.graph,
            self.substrate,
            bounds=bounds or MorphBounds(),
            lineage=self.lineage,
            shadow_evaluator=self._shadow if shadow else None,
            clock=lambda: self.substrate.now,
            require_governance=require_governance,
            emit_receipts=False,
        )
        self.governor.set_hooks(
            spawn=self._on_spawn,
            retire=self._on_retire,
            specialize=self._on_specialize,
            route=self._on_route,
        )
        self._build_seed_population()

    # ── setup ───────────────────────────────────────────────────────────

    def _build_seed_population(self) -> None:
        for cell_id, capabilities in SEED_POPULATION:
            self.workload.add_worker(WorkerProfile(cell_id=cell_id, capabilities=capabilities))
            self.substrate.place(cell_id)
            self.lineage.seed(cell_id)
            self.governor.set_capabilities(cell_id, capabilities)
            self.governor.credit(cell_id, 6.0)
        self.governor.credit(self.policy_name, 6.0)

        def build(scratch: Any) -> None:
            for cell_id, _ in SEED_POPULATION:
                scratch.add_node(cell_id)
            for source, target, port in SEED_EDGES:
                scratch.add_edge(MorphEdge(source, target, EdgeType.DATA, port=port))

        self.governor.set_port_contract(self.workload.port_contract())
        self.graph.transaction(build, cause="seed", port_contract=self.workload.port_contract())
        self.workload.ingress = ["g1"]
        self.workload.egress = ["g5"]

    # ── governor hooks ──────────────────────────────────────────────────

    def _on_spawn(self, cell_id: str, manifest: Mapping[str, Any]) -> None:
        capabilities = tuple(str(c) for c in (manifest.get("capabilities") or ()))
        self.workload.add_worker(WorkerProfile(
            cell_id=cell_id,
            capabilities=capabilities,
            service_rate=int(manifest.get("service_rate", 2)),
        ))
        self.governor.credit(cell_id, 3.0)
        self.governor.set_port_contract(self.workload.port_contract())
        self.applied_log.append({"round": self.round_index, "kind": "spawn", "cell": cell_id})

    def _on_retire(self, cell_id: str) -> None:
        self.workload.remove_worker(cell_id)
        self.governor.set_port_contract(self.workload.port_contract())
        self.applied_log.append({"round": self.round_index, "kind": "retire", "cell": cell_id})

    def _on_specialize(self, cell_id: str, specialization: str) -> None:
        worker = self.workload.workers.get(cell_id)
        if worker is not None:
            worker.specialization = specialization
        self.applied_log.append(
            {"round": self.round_index, "kind": "specialize", "cell": cell_id, "into": specialization}
        )

    def _on_route(self, edge: MorphEdge, weight: float) -> None:
        self.applied_log.append({
            "round": self.round_index, "kind": "route",
            "edge": f"{edge.source}->{edge.target}", "weight": round(weight, 4),
        })

    # ── shadow evaluation ───────────────────────────────────────────────

    def _shadow(self, graph: MorphGraph, proposal: Any = None) -> float | None:
        """Score a candidate world by running the live demand on a copy.

        The copy carries the current worker set, so the score answers "would
        this serve what is actually waiting", not "is this shape pretty". The
        proposal is applied to the copy as well as to the graph, because a
        SPECIALIZE changes service rates and touches no edge — score the graph
        alone and a specialization is indistinguishable from doing nothing.
        """
        if not self.goal_demand:
            return None
        probe = RoutedWorkload(graph, seed=self.seed + 977)
        for cell_id, worker in self.workload.workers.items():
            if graph.has_node(cell_id):
                probe.add_worker(WorkerProfile(
                    cell_id=cell_id,
                    capabilities=worker.capabilities,
                    service_rate=worker.service_rate,
                    specialization=worker.specialization,
                    healthy=worker.healthy,
                ))
        if proposal is not None:
            for transition in proposal.transitions:
                if transition.kind is TransitionKind.SPAWN:
                    # The cell does not exist yet, so its capabilities are in
                    # the manifest and nowhere else. Reading them from the
                    # governor's post-commit map gives an empty set, the probe
                    # adds no worker, and every spawn measures as a gain of
                    # exactly zero however much it would have helped.
                    spawn_id = MorphGovernor._spawn_id(proposal, transition)
                    capabilities = tuple(
                        str(c) for c in (transition.manifest_data.get("capabilities") or ())
                    )
                    if spawn_id and capabilities and spawn_id not in probe.workers:
                        probe.add_worker(WorkerProfile(
                            cell_id=spawn_id,
                            capabilities=capabilities,
                            service_rate=int(transition.manifest_data.get("service_rate", 2)),
                        ))
                    continue
                target = probe.workers.get(transition.subject)
                if target is None:
                    continue
                if transition.kind is TransitionKind.SPECIALIZE:
                    target.specialization = transition.specialization
                elif transition.kind is TransitionKind.DESPECIALIZE:
                    target.specialization = ""
        # Any other node the live workload does not hold yet.
        for node in graph.nodes():
            if node not in probe.workers:
                promised = self._promised_capabilities(node)
                if promised:
                    probe.add_worker(WorkerProfile(cell_id=node, capabilities=promised))
        if not probe.workers:
            return None
        entry = "g1" if "g1" in probe.workers else sorted(probe.workers)[0]
        probe.ingress = [entry]

        # Seed the probe with the work actually in flight. A probe that starts
        # empty measures a system under no load, where every shape completes
        # everything and nothing can be told apart — which is how a shadow
        # evaluator ends up rejecting every proposal for a gain of exactly zero.
        carried = 0
        for cell_id in sorted(self.workload.queues):
            for task in self.workload.queues[cell_id]:
                if carried >= self.shadow_backlog_cap:
                    break
                at = cell_id if cell_id in probe.workers else entry
                clone = probe.admit(task.stages, at=at)
                if clone is not None:
                    clone.stage_index = task.stage_index
                    carried += 1
            if carried >= self.shadow_backlog_cap:
                break

        window = list(self.demand_window)
        if not window:
            families = task_families()
            stages = max(self.goal_demand.items(), key=lambda kv: (kv[1], kv[0]))[0]
            window = [self._family_for(stages, families)]
        rate = max(1, int(round(self.arrivals_per_round or 1)))
        cursor = 0
        for _ in range(self.shadow_steps):
            for _ in range(rate):
                probe.admit(window[cursor % len(window)])
                cursor += 1
            probe.step()
        metrics = probe.metrics
        if metrics.admitted == 0:
            return None
        return metrics.completion_rate - 0.02 * metrics.mean_sojourn

    def _promised_capabilities(self, node: str) -> tuple[str, ...]:
        return tuple(self.governor._capability_of.get(node, ()))

    @staticmethod
    def _family_for(capability: str, families: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
        best = max(
            families.items(),
            key=lambda kv: (sum(1 for s in kv[1] if s == capability), kv[0]),
        )
        return best[1]

    # ── the loop ────────────────────────────────────────────────────────

    def set_goal(self, demand: Mapping[str, float]) -> None:
        self.goal_demand = {str(k): float(v) for k, v in demand.items()}

    def admit_family(self, family: Sequence[str], count: int) -> None:
        for _ in range(count):
            self.workload.admit(family)
        # Remembered so the shadow probe can reproduce the load the system is
        # actually under rather than an idle one.
        self.arrivals_per_round = (
            0.5 * self.arrivals_per_round + 0.5 * float(count)
            if self.arrivals_per_round else float(count)
        )
        self.last_family = tuple(family)
        if count:
            self.demand_window.append(tuple(family))

    def round(self, *, propose: bool = True) -> dict[str, Any]:
        """One round: let the policy propose, adjudicate, then run the work."""
        self.round_index += 1
        self.substrate.advance(0.5)
        decisions: list[str] = []
        if propose:
            context = PolicyContext(
                graph=self.graph,
                workload=self.workload,
                goal_demand=self.goal_demand,
                round_index=self.round_index,
                radius=self.radius,
            )
            proposals = self.policy.propose(context)
            for transaction in self.governor.submit(proposals):
                decisions.append(str(transaction.decision))
        step = self.workload.step()
        for cell_id in self.workload.workers:
            self.governor.credit(cell_id, 0.25)
        # A policy that proposes in its own name spends from the same purse.
        # Without this the central baseline is refused for want of energy it
        # was never given, and loses a comparison it never entered.
        self.governor.credit(self.policy_name, 0.25 * max(1, len(self.workload.workers)))
        return {"round": self.round_index, "decisions": decisions, **step}

    def lesion(self, fraction: float) -> list[str]:
        """Remove a fraction of the population without notice."""
        cells = sorted(self.workload.workers)
        keep_ingress = {"g1"}
        candidates = [c for c in cells if c not in keep_ingress]
        count = max(1, int(len(cells) * float(fraction)))
        victims = candidates[:count]
        self.substrate.lesion(victims)
        for cell_id in victims:
            self.workload.remove_worker(cell_id)
            self.lineage.record_retirement(cell_id, cause="lesion")
        self.graph.transaction(
            lambda scratch: [scratch.remove_node(v) for v in victims],
            cause="lesion",
        )
        self.governor.set_port_contract(self.workload.port_contract())
        # The premises every earlier decision was made under just changed.
        self.governor.invalidate_reversal_history(reason="lesion")
        return victims

    def result(self, label: str) -> ArmResult:
        stats = self.governor.stats
        return ArmResult(
            label=label,
            metrics=self.workload.metrics.to_dict(),
            graph_versions=self.graph.version,
            applied=stats.applied,
            rejected=stats.rejected,
            deferred=stats.deferred,
            rolled_back=stats.rolled_back,
            final_cells=len(self.workload.workers),
            final_edges=self.graph.edge_count,
            energy_spent=stats.energy_spent,
            substrate_energy=float(self.substrate.health().get("energy_spent", 0.0)),
            components=len(self.graph.components()),
            graph_digest=self.graph.snapshot().digest(),
            signature=self.workload.signature(),
            detail={
                "rejections": stats.rejections_by_reason,
                "lineage": self.lineage.status(),
                "applied_log": self.applied_log[-24:],
                "substrate": self.substrate.health(),
            },
        )


# ── ablation wiring ─────────────────────────────────────────────────────

def _harness_for(
    ablation: str,
    *,
    seed: int,
    bounds: MorphBounds | None = None,
    physics: SubstratePhysics | None = None,
    motifs: MotifLibrary | None = None,
    deadline_steps: int = 26,
    demand_window_len: int = 6,
) -> Harness:
    common: dict[str, Any] = {
        "seed": seed,
        "bounds": bounds,
        "physics": physics,
        "motifs": motifs,
        "deadline_steps": deadline_steps,
        "demand_window_len": demand_window_len,
    }
    if ablation == "morphology_off":
        return Harness(policy_name="frozen", allow_recovery=False, **common)
    if ablation == "topology_fixed":
        return Harness(policy_name="frozen", **common)
    if ablation == "local_signals_off":
        return Harness(policy_name="blind_local", **common)
    if ablation == "motifs_off":
        return Harness(
            policy_name="local", motifs=None, seed=seed, bounds=bounds,
            physics=physics, deadline_steps=deadline_steps,
            demand_window_len=demand_window_len,
        )
    if ablation == "recovery_off":
        return Harness(policy_name="local", allow_recovery=False, **common)
    if ablation == "central_scheduler":
        return Harness(policy_name="central", **common)
    if ablation == "random_mutation":
        return Harness(policy_name="random", **common)
    return Harness(policy_name="local", **common)


__all__ = [
    "ABLATIONS",
    "ArmResult",
    "Harness",
    "SCENARIOS",
    "SEED_EDGES",
    "SEED_POPULATION",
    "ScenarioResult",
]
