"""core/morphogenesis/governor.py — the only thing that may change the shape.

A cell proposes. The governor decides. Everything that keeps self-organisation
from becoming a cancer lives here, in one place, so a new local rule cannot
route around it by being written somewhere else.

The order matters and each step can only reject:

1. **Well-formedness.** A transition that cannot describe its own inverse never
   reaches a budget.
2. **Bounds.** Population, per-capability replicas, spawn depth, transitions per
   window, cooldown, out-degree. Refused here costs nothing.
3. **Resource budget.** The proposer pays. A cell with no energy cannot spawn,
   which is what makes replication expensive rather than free.
4. **Shadow evaluation.** Apply to a copy, run the workload on it, compare
   against the same workload on the current shape. A proposal whose own claimed
   benefit does not survive measurement is refused with the measurement
   attached, so a policy that always claims 1.0 is caught by arithmetic.
5. **Governance.** Anything CRITICAL goes through Aura's existing internal
   governed scope. No governed scope, no critical change.
6. **Commit.** Substrate first, then graph. A substrate failure rolls the graph
   back to the snapshot taken before the attempt, and a *partial* substrate
   failure additionally unwinds what the substrate did.

The last point is the one that only shows up on hardware. A dock that latched
and then failed its handshake has changed the world; committing the graph
anyway would leave the two disagreeing, and treating it as a clean failure
would leave a latch nobody owns.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import EdgeType, GraphIntegrityError, GraphSnapshot, MorphEdge, MorphGraph
from .lineage import Lineage, LineageCycleError
from .proposal import (
    Decision,
    MorphProposal,
    MorphTransaction,
    MorphTransition,
    RiskClass,
    TransitionKind,
    rank_proposals,
)
from .substrate import SubstrateAdapter, TransitionOutcome
from .types import json_safe, stable_digest

logger = logging.getLogger("Aura.Morphogenesis.Governor")


@dataclass
class MorphBounds:
    """The homeostatic envelope.

    Every field is a refusal the governor can make without consulting anything
    else. A population that can grow without one of these is a population that
    will, given a signal that says grow — which the poisoned-signal scenario
    supplies on purpose.
    """

    max_cells: int = 64
    max_edges: int = 256
    max_replicas_per_capability: int = 6
    max_spawn_depth: int = 4
    max_transitions_per_window: int = 12
    window_s: float = 10.0
    #: A cell that just changed cannot change again until this passes. Without
    #: it, an oscillating demand signal makes the topology chase it.
    cooldown_s: float = 3.0
    #: Each change at a cell doubles its own cooldown, up to this many
    #: doublings, and the count decays once the cell has been quiet for a full
    #: window. A flat cooldown only sets a floor on how fast a cell can chase a
    #: signal; a cell reorganising over and over is thrashing whatever the
    #: period, and it should get slower each time until it stops.
    #:
    #: Kept low, because backoff is blunt: it slows the fifth change at a cell
    #: whether that change is thrash or the next step of a repair. The targeted
    #: guard is `reversal_window_s` below.
    churn_backoff_doublings: int = 2
    #: A change may not be undone within this many seconds of being made.
    #:
    #: This is what thrash actually is — not changing often, but changing back.
    #: Slowing every repeated change punishes regeneration, which is a long
    #: run of different changes at the same cell; refusing only the reversals
    #: leaves that untouched and still breaks the loop an alternating demand
    #: would otherwise drive.
    reversal_window_s: float = 12.0
    #: A proposal must beat the current shape by this much in shadow before it
    #: is worth its cost. The band is the hysteresis: a change that only just
    #: helps is not worth the disruption of making it.
    min_shadow_gain: float = 0.02
    #: Below this baseline score, a change need only avoid making things worse.
    #:
    #: A band that always demands immediate improvement cannot cross a valley,
    #: and recovery from damage is always a valley: with two capabilities
    #: missing, restoring either one alone completes nothing and measures as a
    #: gain of exactly zero, so both are refused and the population stays dead.
    #: The bound is a floor rather than a switch, so exploration is available
    #: only where there is nothing left to lose; the population and replica
    #: caps still hold, so this widens what may be tried and not how much.
    exploration_floor: float = 0.05
    #: Energy the proposer must hold to spend on a transition.
    min_proposer_energy: float = 0.12
    #: A run may not spend more than this in total. The ceiling stops a slow
    #: leak of small approvals from adding up to an unbounded reorganisation.
    max_total_energy: float = 250.0
    #: Refuse a change that would break the existing structure into more
    #: pieces than this. Counted over cells that already existed, so a freshly
    #: spawned cell waiting to be bound is not mistaken for fragmentation.
    max_components: int = 1
    #: Cells that may never be retired, migrated, or unbound from.
    protected: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cells": self.max_cells,
            "max_edges": self.max_edges,
            "max_replicas_per_capability": self.max_replicas_per_capability,
            "max_spawn_depth": self.max_spawn_depth,
            "max_transitions_per_window": self.max_transitions_per_window,
            "window_s": self.window_s,
            "cooldown_s": self.cooldown_s,
            "churn_backoff_doublings": self.churn_backoff_doublings,
            "reversal_window_s": self.reversal_window_s,
            "min_shadow_gain": self.min_shadow_gain,
            "exploration_floor": self.exploration_floor,
            "min_proposer_energy": self.min_proposer_energy,
            "max_total_energy": self.max_total_energy,
            "max_components": self.max_components,
            "protected": sorted(self.protected),
        }


#: A shadow evaluator scores a candidate world. Higher is better. Returning
#: None means "could not measure", which the governor treats as a refusal
#: rather than as approval — an unmeasured change is not a safe change.
#:
#: It receives the proposal as well as the graph, because not every transition
#: shows up in the topology. SPECIALIZE changes what a cell can serve and
#: leaves the edges alone; an evaluator handed only a graph would score the
#: change and its absence identically and pass anything.
ShadowEvaluator = Callable[[MorphGraph, "MorphProposal | None"], float | None]


@dataclass
class GovernorStats:
    proposals_seen: int = 0
    applied: int = 0
    rejected: int = 0
    deferred: int = 0
    rolled_back: int = 0
    energy_spent: float = 0.0
    #: Counted whether or not the guard is enforcing, so a run with the guard
    #: switched off still says how many reversals it made. A guard that is only
    #: observable by its own refusals cannot be shown to be load-bearing.
    reversals_detected: int = 0
    reversals_refused: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    def note_rejection(self, reason: str) -> None:
        key = reason.split(":")[0].strip()[:64]
        self.rejections_by_reason[key] = self.rejections_by_reason.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals_seen": self.proposals_seen,
            "applied": self.applied,
            "rejected": self.rejected,
            "deferred": self.deferred,
            "rolled_back": self.rolled_back,
            "energy_spent": round(self.energy_spent, 5),
            "reversals_detected": self.reversals_detected,
            "reversals_refused": self.reversals_refused,
            "rejections_by_reason": dict(sorted(self.rejections_by_reason.items())),
        }


class MorphGovernor:
    """Adjudicates proposals against bounds, measurement, and governance."""

    def __init__(
        self,
        graph: MorphGraph,
        substrate: SubstrateAdapter,
        *,
        bounds: MorphBounds | None = None,
        lineage: Lineage | None = None,
        shadow_evaluator: ShadowEvaluator | None = None,
        clock: Callable[[], float] = time.time,
        require_governance: bool = True,
        emit_receipts: bool = True,
    ):
        self.graph = graph
        self.substrate = substrate
        self.bounds = bounds or MorphBounds()
        self.lineage = lineage or Lineage(max_generation=self.bounds.max_spawn_depth)
        self.shadow_evaluator = shadow_evaluator
        self.clock = clock
        self.require_governance = bool(require_governance)
        self.emit_receipts = bool(emit_receipts)
        self.stats = GovernorStats()
        self.transactions: list[MorphTransaction] = []
        self._recent: deque[float] = deque(maxlen=512)
        self._last_change: dict[str, float] = {}
        self._churn: dict[str, int] = {}
        self._applied_signatures: deque[tuple[float, str]] = deque(maxlen=256)
        self._energy: dict[str, float] = {}
        self._capability_of: dict[str, tuple[str, ...]] = {}
        self._port_contract: Mapping[str, tuple[frozenset[str], frozenset[str]]] | None = None
        self._spawn_hook: Callable[[str, Mapping[str, Any]], None] | None = None
        self._retire_hook: Callable[[str], None] | None = None
        self._specialize_hook: Callable[[str, str], None] | None = None
        self._route_hook: Callable[[MorphEdge, float], None] | None = None

    # ── wiring the governor to a population ─────────────────────────────

    def set_port_contract(self, contract: Mapping[str, tuple[frozenset[str], frozenset[str]]]) -> None:
        self._port_contract = dict(contract)

    def set_capabilities(self, cell_id: str, capabilities: Sequence[str]) -> None:
        self._capability_of[cell_id] = tuple(capabilities)

    def set_hooks(
        self,
        *,
        spawn: Callable[[str, Mapping[str, Any]], None] | None = None,
        retire: Callable[[str], None] | None = None,
        specialize: Callable[[str, str], None] | None = None,
        route: Callable[[MorphEdge, float], None] | None = None,
    ) -> None:
        """Attach the side effects that make a committed transition real.

        Kept as hooks so the governor never imports the population it governs,
        and so a scenario can watch what was applied without the graph and the
        workers drifting apart.
        """
        self._spawn_hook = spawn or self._spawn_hook
        self._retire_hook = retire or self._retire_hook
        self._specialize_hook = specialize or self._specialize_hook
        self._route_hook = route or self._route_hook

    #: A cell may bank at most this much change budget. Without a ceiling,
    #: energy accrues forever and stops being a constraint: a cell that has sat
    #: quiet for a day could then reorganise continuously until it ran out,
    #: which is the burst the rate window and cooldown exist to prevent.
    max_banked_energy = 3.0

    def credit(self, cell_id: str, amount: float) -> float:
        banked = self._energy.get(cell_id, 0.0) + float(amount)
        self._energy[cell_id] = max(0.0, min(self.max_banked_energy, banked))
        return self._energy[cell_id]

    def energy(self, cell_id: str) -> float:
        return self._energy.get(cell_id, 0.0)

    # ── adjudication ────────────────────────────────────────────────────

    def submit(self, proposals: Sequence[MorphProposal]) -> list[MorphTransaction]:
        """Adjudicate a batch. Returns one transaction per proposal seen."""
        out: list[MorphTransaction] = []
        for proposal in rank_proposals(list(proposals)):
            out.append(self.adjudicate(proposal))
        return out

    def adjudicate(self, proposal: MorphProposal) -> MorphTransaction:
        started = time.monotonic()
        self.stats.proposals_seen += 1
        now = self.clock()
        before_version = self.graph.version

        def finish(decision: Decision, reason: str, **kwargs: Any) -> MorphTransaction:
            transaction = MorphTransaction(
                proposal=proposal,
                decision=decision,
                reason=reason,
                graph_version_before=before_version,
                graph_version_after=self.graph.version,
                duration_ms=(time.monotonic() - started) * 1000.0,
                **kwargs,
            )
            self.transactions.append(transaction)
            if len(self.transactions) > 4096:
                del self.transactions[:-4096]
            if decision is Decision.APPLIED:
                self.stats.applied += 1
            elif decision is Decision.REJECTED:
                self.stats.rejected += 1
                self.stats.note_rejection(reason)
            elif decision is Decision.DEFERRED:
                self.stats.deferred += 1
            else:
                self.stats.rolled_back += 1
                self.stats.note_rejection(reason)
            return transaction

        problem = proposal.validate()
        if problem:
            return finish(Decision.REJECTED, f"malformed: {problem}")

        reversal = self._reversal_of_recent(proposal, now)
        if reversal:
            return finish(Decision.REJECTED, reversal)

        bound_problem = self._check_bounds(proposal, now)
        if bound_problem:
            decision = Decision.DEFERRED if bound_problem.startswith("rate") or bound_problem.startswith("cooldown") else Decision.REJECTED
            return finish(decision, bound_problem)

        cost = proposal.estimated_cost
        if cost > 0:
            available = self.energy(proposal.proposer)
            if available < max(cost, self.bounds.min_proposer_energy):
                return finish(
                    Decision.DEFERRED,
                    f"budget: {proposal.proposer} holds {available:.3f} against a cost of {cost:.3f}",
                )
            if self.stats.energy_spent + cost > self.bounds.max_total_energy:
                return finish(Decision.REJECTED, "budget: run energy ceiling reached")

        candidate, why_not = self._candidate_graph(proposal)
        if candidate is None:
            return finish(Decision.REJECTED, f"shape: {why_not}")

        shadow_score: float | None = None
        baseline_score: float | None = None
        if proposal.risk is not RiskClass.ROUTINE and self.shadow_evaluator is None:
            # No evaluator is the same situation as an evaluator that returns
            # None, and it must have the same answer. Skipping the block
            # instead applied every spawn, migration and specialization
            # unmeasured — the precise fail-open this ladder exists to prevent,
            # and it read as "no shadow configured" rather than as a hole.
            return finish(
                Decision.REJECTED,
                (
                    f"shadow: {proposal.risk} changes need a shadow evaluator and this "
                    "governor has none; an unmeasured change is not a safe change"
                ),
            )
        if self.shadow_evaluator is not None and proposal.risk is not RiskClass.ROUTINE:
            baseline_score = self.shadow_evaluator(self.graph, None)
            shadow_score = self.shadow_evaluator(candidate, proposal)
            if shadow_score is None or baseline_score is None:
                return finish(
                    Decision.REJECTED,
                    "shadow: could not measure the change, and an unmeasured change is not a safe one",
                    shadow_score=shadow_score,
                    baseline_score=baseline_score,
                )
            gain = shadow_score - baseline_score
            exploring = baseline_score <= self.bounds.exploration_floor
            required = 0.0 if exploring else self.bounds.min_shadow_gain
            if gain < required:
                band = (
                    f"must not be worse (baseline {baseline_score:.4f} is at the "
                    f"exploration floor)"
                    if exploring
                    else f"is under the {required:.4f} band"
                )
                return finish(
                    Decision.REJECTED,
                    (
                        f"shadow: measured gain {gain:+.4f} {band} "
                        f"(claimed {proposal.expected_benefit:.3f})"
                    ),
                    shadow_score=shadow_score,
                    baseline_score=baseline_score,
                )

        if proposal.risk is RiskClass.CRITICAL and self.require_governance:
            token = self._governance_token(proposal)
            if token is None:
                return finish(
                    Decision.REJECTED,
                    "governance: a critical topology change needs a governed scope and none was open",
                    shadow_score=shadow_score,
                    baseline_score=baseline_score,
                )

        return self._commit(
            proposal,
            finish=finish,
            now=now,
            shadow_score=shadow_score,
            baseline_score=baseline_score,
        )

    # ── bounds ──────────────────────────────────────────────────────────

    def _check_bounds(self, proposal: MorphProposal, now: float) -> str:
        window_start = now - self.bounds.window_s
        recent = [t for t in self._recent if t >= window_start]
        if len(recent) >= self.bounds.max_transitions_per_window:
            return (
                f"rate: {len(recent)} transitions in the last {self.bounds.window_s:.0f}s "
                f"is at the cap of {self.bounds.max_transitions_per_window}"
            )

        for cell_id in proposal.affected_cells():
            last = self._last_change.get(cell_id, 0.0)
            if not last:
                continue
            wait = self._cooldown_for(cell_id, now)
            if now - last < wait:
                churn = self._churn.get(cell_id, 0)
                return (
                    f"cooldown: {cell_id} changed {now - last:.2f}s ago and owes "
                    f"{wait:.2f}s after {churn} change(s)"
                )

        projected = self.graph.node_count + proposal.population_delta
        if projected > self.bounds.max_cells:
            return f"population: {projected} cells would pass the cap of {self.bounds.max_cells}"

        for transition in proposal.transitions:
            if transition.kind is TransitionKind.SPAWN:
                parent = str(transition.metadata.get("parent", proposal.proposer))
                if self.lineage.would_exceed_depth(parent):
                    return (
                        f"depth: spawning from {parent} would reach generation "
                        f"{self.lineage.generation_of(parent) + 1}, past {self.bounds.max_spawn_depth}"
                    )
                capability = self._primary_capability(transition.manifest_data)
                if capability:
                    replicas = sum(
                        1 for caps in self._capability_of.values() if capability in caps
                    )
                    if replicas >= self.bounds.max_replicas_per_capability:
                        return (
                            f"replicas: {capability} already has {replicas}, at the cap of "
                            f"{self.bounds.max_replicas_per_capability}"
                        )
            if transition.kind in {TransitionKind.RETIRE, TransitionKind.MIGRATE}:
                if transition.subject in self.bounds.protected:
                    return f"protected: {transition.subject} may not be {transition.kind}d"
            if transition.kind is TransitionKind.UNBIND and transition.edge is not None:
                if transition.edge.source in self.bounds.protected and transition.edge.target in self.bounds.protected:
                    return f"protected: the binding {transition.edge.source}->{transition.edge.target} is load-bearing"
        return ""

    @staticmethod
    def _signature(transition: MorphTransition) -> str:
        edge = transition.edge
        return "|".join((
            str(transition.kind),
            transition.subject,
            "->".join(str(v) for v in edge.key) if edge is not None else "",
            transition.specialization,
            transition.placement,
        ))

    def invalidate_reversal_history(self, *, cells: Sequence[str] = (), reason: str = "") -> int:
        """Forget that these changes were made, because the world moved.

        The reversal guard is a bet that nothing has changed underneath the
        earlier decision. When something demonstrably has — a cell lost, a link
        cut — the bet is void: undoing a specialization whose cover just died
        is not thrash, it is the only correct move left. Without this the guard
        blocks the repair it was never aimed at.
        """
        if not cells:
            dropped = len(self._applied_signatures)
            self._applied_signatures.clear()
            logger.debug("morphogenesis reversal history cleared (%s)", reason or "world change")
            return dropped
        touched = {str(c) for c in cells}
        keep = [
            (at, sig) for at, sig in self._applied_signatures
            if not any(cell in sig for cell in touched)
        ]
        dropped = len(self._applied_signatures) - len(keep)
        self._applied_signatures.clear()
        self._applied_signatures.extend(keep)
        return dropped

    def _reversal_of_recent(self, proposal: MorphProposal, now: float) -> str:
        """Note, and where the guard is armed refuse, a proposal that undoes
        something just done.

        Thrash is not changing often, it is changing back. A demand that
        alternates every other round drives exactly this: bind, unbind, bind
        again, each justified by the round it was measured in. Growing more
        structure to serve two demands instead of one is not thrash, and a
        rule that counts transitions cannot tell the two apart.

        Detection runs whatever the window, so a run with the guard disarmed
        still reports the reversals it made.
        """
        window = max(0.0, float(self.bounds.reversal_window_s))
        horizon = window if window > 0.0 else float(self.bounds.window_s)
        recent = {sig for at, sig in self._applied_signatures if now - at <= horizon}
        if not recent:
            return ""
        for transition in proposal.transitions:
            inverse = transition.inverse()
            if inverse is None:
                continue
            if self._signature(inverse) in recent:
                self.stats.reversals_detected += 1
                if window <= 0.0:
                    return ""
                self.stats.reversals_refused += 1
                return (
                    f"reversal: {transition.kind} on {transition.subject or 'an edge'} would undo "
                    f"a change made inside the last {window:.0f}s"
                )
        return ""

    def _cooldown_for(self, cell_id: str, now: float) -> float:
        """This cell's own cooldown, doubled once per recent change.

        The count decays after a quiet window, so a cell that reorganises once
        and settles pays the base rate again rather than carrying a penalty
        forever.
        """
        last = self._last_change.get(cell_id, 0.0)
        churn = self._churn.get(cell_id, 0)
        if last and now - last > self.bounds.window_s:
            churn = 0
            self._churn[cell_id] = 0
        doublings = min(max(0, churn), max(0, self.bounds.churn_backoff_doublings))
        return self.bounds.cooldown_s * float(2 ** doublings)

    @staticmethod
    def _primary_capability(manifest_data: Mapping[str, Any]) -> str:
        capabilities = list(manifest_data.get("capabilities") or ())
        return str(capabilities[0]) if capabilities else ""

    # ── shadow ──────────────────────────────────────────────────────────

    def _fragmentation(self, graph: MorphGraph, scope: frozenset[str]) -> int:
        """How many pieces the cells in ``scope`` fall into, in ``graph``.

        Scoping to the cells that already existed is what separates two things
        that look alike in a raw component count: a change that severed the
        population, and a spawn that has not been bound in yet.
        """
        pieces = [c & scope for c in graph.components()]
        return sum(1 for piece in pieces if piece)

    def _effective_ports(
        self, proposal: MorphProposal
    ) -> Mapping[str, tuple[frozenset[str], frozenset[str]]] | None:
        """The port contract, extended with the cells this proposal creates.

        A grow() binds to a cell that does not exist yet. Validated against a
        contract built only from cells that already exist, that binding is
        refused for a target that "does not accept" the port — and growing a
        part and wiring it in could never happen in one step.
        """
        if self._port_contract is None:
            return None
        extended = dict(self._port_contract)
        for transition in proposal.transitions:
            if transition.kind is not TransitionKind.SPAWN:
                continue
            cell_id = self._spawn_id(proposal, transition)
            data = transition.manifest_data or {}
            capabilities = frozenset(str(c) for c in (data.get("capabilities") or ()))
            emits = frozenset(str(c) for c in (data.get("emits") or ()))
            consumes = frozenset(str(c) for c in (data.get("consumes") or ()))
            # Derive the ports the same way the population does. Reading only
            # `capabilities` here gave the governor a narrower contract than
            # the runtime's, so a manifest that plainly declared it consumes
            # `repair` was refused for not accepting `repair` — a disagreement
            # between two descriptions of one cell, and the sort of thing that
            # reads as an unexplained rejection forever.
            # Declared beats derived. A manifest that says what it emits is
            # taken at its word; one that says nothing falls back to what the
            # population as a whole can send, which is what a manifest with no
            # port declarations meant before there were any.
            #
            # The distinction is between a manifest that declared nothing and
            # one whose declaration came out empty. Collapsing them narrowed
            # every undeclared cell to its own capability, and a grown cell
            # could no longer send work back the way it came.
            if emits:
                out_ports = emits | capabilities
            elif self._port_contract:
                out_ports = frozenset().union(
                    *(value[0] for value in self._port_contract.values())
                )
            else:
                out_ports = capabilities
            in_ports = (consumes | capabilities) if consumes else capabilities
            extended[cell_id] = (out_ports or capabilities, in_ports or capabilities)
        return extended

    def _candidate_graph(self, proposal: MorphProposal) -> tuple[MorphGraph | None, str]:
        """A copy of the graph with the proposal applied, and why not if not.

        The copy is what gets measured. The live graph is untouched, so a
        proposal that would break the shape is refused having never made it.
        """
        scope = frozenset(self.graph.nodes())
        before = self._fragmentation(self.graph, scope)
        candidate = MorphGraph.from_dict(self.graph.to_dict())
        try:
            candidate.transaction(
                lambda scratch: self._apply_to_scratch(scratch, proposal, dry_run=True),
                cause=f"shadow:{proposal.proposal_id}",
                port_contract=self._effective_ports(proposal),
            )
        except GraphIntegrityError as exc:
            return None, str(exc)
        after = self._fragmentation(candidate, scope)
        ceiling = max(1, self.bounds.max_components, before)
        if after > ceiling:
            return None, (
                f"it would break the population into {after} pieces, up from {before}, "
                f"past the ceiling of {ceiling}"
            )
        return candidate, ""

    # ── commit ──────────────────────────────────────────────────────────

    def _apply_to_scratch(self, scratch: Any, proposal: MorphProposal, *, dry_run: bool) -> None:
        for transition in proposal.transitions:
            kind = transition.kind
            if kind is TransitionKind.BIND and transition.edge is not None:
                scratch.add_edge(transition.edge)
            elif kind is TransitionKind.UNBIND and transition.edge is not None:
                scratch.remove_edge(transition.edge.key)
            elif kind is TransitionKind.ROUTE and transition.edge is not None:
                edge = transition.edge
                scratch.remove_edge(edge.key)
                scratch.add_edge(MorphEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    port=edge.port,
                    weight=float(transition.weight),
                    latency_ms=edge.latency_ms,
                    capacity=edge.capacity,
                    created_at_version=edge.created_at_version,
                    metadata=dict(edge.metadata),
                ))
            elif kind is TransitionKind.SPAWN:
                scratch.add_node(self._spawn_id(proposal, transition))
            elif kind is TransitionKind.RETIRE:
                scratch.remove_node(transition.subject)
            elif kind is TransitionKind.MERGE:
                for absorbed in transition.metadata.get("absorb", ()):
                    scratch.remove_node(str(absorbed))

    @staticmethod
    def _spawn_id(proposal: MorphProposal, transition: MorphTransition) -> str:
        """A deterministic id for a cell about to exist.

        Derived from the proposal, so a replay produces the same population
        with the same names and two runs stay comparable.
        """
        if transition.subject:
            return transition.subject
        name = str(transition.manifest_data.get("name", "cell"))
        return f"{name}_{stable_digest(proposal.proposal_id, name, length=8)}"

    def _commit(
        self,
        proposal: MorphProposal,
        *,
        finish: Callable[..., MorphTransaction],
        now: float,
        shadow_score: float | None,
        baseline_score: float | None,
    ) -> MorphTransaction:
        snapshot: GraphSnapshot = self.graph.snapshot()
        substrate_events: list[dict[str, Any]] = []
        done: list[tuple[MorphTransition, str]] = []

        for transition in proposal.transitions:
            result, subject = self._drive_substrate(proposal, transition)
            substrate_events.append({"transition": str(transition.kind), "subject": subject, **result.to_dict()})
            if result.ok:
                done.append((transition, subject))
                continue
            # Undo whatever the substrate already did, including the step that
            # only half happened.
            if result.partial:
                self._undo_substrate(transition, subject)
            for prior, prior_subject in reversed(done):
                self._undo_substrate(prior, prior_subject)
            self.graph.restore(snapshot, cause=f"rollback:{proposal.proposal_id}")
            return finish(
                Decision.ROLLED_BACK,
                f"substrate: {result.detail or result.outcome}"
                + (" (partial, unwound)" if result.partial else ""),
                shadow_score=shadow_score,
                baseline_score=baseline_score,
                substrate_events=tuple(substrate_events),
            )

        try:
            self.graph.transaction(
                lambda scratch: self._apply_to_scratch(scratch, proposal, dry_run=False),
                cause=f"{proposal.proposer}:{proposal.proposal_id}",
                port_contract=self._effective_ports(proposal),
            )
        except GraphIntegrityError as exc:
            for prior, prior_subject in reversed(done):
                self._undo_substrate(prior, prior_subject)
            self.graph.restore(snapshot, cause=f"rollback:{proposal.proposal_id}")
            return finish(
                Decision.ROLLED_BACK,
                f"graph: {exc}",
                shadow_score=shadow_score,
                baseline_score=baseline_score,
                substrate_events=tuple(substrate_events),
            )

        applied: list[MorphTransition] = []
        for transition, subject in done:
            settled = self._settle(proposal, transition, subject, now)
            applied.append(settled)
            self._applied_signatures.append((now, self._signature(settled)))

        self._recent.append(now)
        for cell_id in proposal.affected_cells():
            if self._last_change.get(cell_id, 0.0) and now - self._last_change[cell_id] <= self.bounds.window_s:
                self._churn[cell_id] = self._churn.get(cell_id, 0) + 1
            self._last_change[cell_id] = now
        for _, subject in done:
            if subject:
                self._last_change[subject] = now
        if proposal.estimated_cost > 0:
            self._energy[proposal.proposer] = max(
                0.0, self.energy(proposal.proposer) - proposal.estimated_cost
            )
            self.stats.energy_spent += proposal.estimated_cost

        receipt_id = self._emit_receipt(proposal) if self.emit_receipts else ""
        return finish(
            Decision.APPLIED,
            "applied",
            applied_transitions=tuple(applied),
            shadow_score=shadow_score,
            baseline_score=baseline_score,
            receipt_id=receipt_id,
            substrate_events=tuple(substrate_events),
        )

    def _drive_substrate(self, proposal: MorphProposal, transition: MorphTransition):
        kind = transition.kind
        if kind is TransitionKind.BIND and transition.edge is not None:
            return self.substrate.bind(transition.edge), transition.edge.source
        if kind is TransitionKind.UNBIND and transition.edge is not None:
            return self.substrate.unbind(transition.edge), transition.edge.source
        if kind is TransitionKind.SPAWN:
            cell_id = self._spawn_id(proposal, transition)
            return self.substrate.spawn(cell_id, transition.manifest_data, locus=transition.placement), cell_id
        if kind is TransitionKind.RETIRE:
            return self.substrate.retire(transition.subject), transition.subject
        if kind is TransitionKind.MIGRATE:
            return self.substrate.migrate(transition.subject, transition.placement), transition.subject
        # ROUTE, SPECIALIZE, DESPECIALIZE and MERGE are logical: nothing in the
        # substrate moves, so there is nothing for it to fail at.
        from .substrate import SubstrateResult

        return SubstrateResult(outcome=TransitionOutcome.OK, detail=f"{kind} is logical"), transition.subject

    def _undo_substrate(self, transition: MorphTransition, subject: str) -> None:
        kind = transition.kind
        try:
            if kind is TransitionKind.BIND and transition.edge is not None:
                self.substrate.unbind(transition.edge)
            elif kind is TransitionKind.UNBIND and transition.edge is not None:
                self.substrate.bind(transition.edge)
            elif kind is TransitionKind.SPAWN and subject:
                self.substrate.retire(subject)
            elif kind is TransitionKind.MIGRATE and subject:
                previous = str(transition.metadata.get("previous_placement", ""))
                if previous:
                    self.substrate.migrate(subject, previous)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("morphogenesis rollback step failed for %s: %s", kind, exc)

    def _settle(
        self,
        proposal: MorphProposal,
        transition: MorphTransition,
        subject: str,
        now: float,
    ) -> MorphTransition:
        """Record a committed transition and fire its population-side effect."""
        kind = transition.kind
        if kind is TransitionKind.SPAWN:
            parent = str(transition.metadata.get("parent", proposal.proposer))
            try:
                self.lineage.record_birth(
                    subject,
                    parent_id=parent,
                    version=self.graph.version,
                    cause=proposal.rationale or proposal.proposal_id,
                    motif_id=str(proposal.evidence.get("motif_id", "")),
                )
            except LineageCycleError as exc:
                logger.warning("lineage refused a birth link: %s", exc)
            self.set_capabilities(subject, transition.manifest_data.get("capabilities") or ())
            if self._spawn_hook is not None:
                self._spawn_hook(subject, transition.manifest_data)
            return MorphTransition(
                kind=kind,
                subject=subject,
                manifest_data=transition.manifest_data,
                placement=transition.placement,
                metadata=transition.metadata,
            )
        if kind is TransitionKind.RETIRE:
            self.lineage.record_retirement(subject, cause=proposal.rationale or "retired")
            self._capability_of.pop(subject, None)
            if self._retire_hook is not None:
                self._retire_hook(subject)
        elif kind is TransitionKind.SPECIALIZE and self._specialize_hook is not None:
            self._specialize_hook(transition.subject, transition.specialization)
        elif kind is TransitionKind.DESPECIALIZE and self._specialize_hook is not None:
            self._specialize_hook(transition.subject, "")
        elif kind is TransitionKind.ROUTE and self._route_hook is not None and transition.edge is not None:
            self._route_hook(transition.edge, transition.weight)
        return transition

    # ── governance and receipts ─────────────────────────────────────────

    def _governance_token(self, proposal: MorphProposal) -> Any:
        """Aura's internal governed scope, where the runtime provides one.

        Absent the runtime — a sandbox run, a unit test — there is no governor
        to ask, and a critical change is refused rather than waved through. The
        absence of a check is not a passed check.
        """
        try:
            from core.governance_context import get_active_governance
        except ImportError:
            return None
        try:
            return get_active_governance()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("governance lookup failed for %s: %s", proposal.proposal_id, exc)
            return None

    def _emit_receipt(self, proposal: MorphProposal) -> str:
        try:
            from core.runtime.receipts import StateMutationReceipt, get_receipt_store
        except ImportError:
            return ""
        receipt_id = f"morph-{stable_digest(proposal.proposal_id, self.graph.version, length=16)}"
        try:
            get_receipt_store().emit(
                StateMutationReceipt(
                    receipt_id=receipt_id,
                    cause=f"morphogenesis.topology.{proposal.risk}",
                    domain="morphogenesis",
                    key=f"graph.v{self.graph.version}",
                    schema_version=1,
                    metadata={
                        "proposal_id": proposal.proposal_id,
                        "proposer": proposal.proposer,
                        "transitions": [str(t.kind) for t in proposal.transitions],
                        "graph_digest": self.graph.snapshot().digest(),
                    },
                )
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("morphogenesis receipt skipped: %s", exc)
            return ""
        return receipt_id

    # ── reporting ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "stats": self.stats.to_dict(),
            "graph_version": self.graph.version,
            "graph_digest": self.graph.snapshot().digest(),
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
            "components": len(self.graph.components()),
            "lineage": self.lineage.status(),
            "substrate": self.substrate.health(),
            "require_governance": self.require_governance,
        }

    def history(self, *, limit: int = 64) -> list[dict[str, Any]]:
        return [json_safe(t.to_dict()) for t in self.transactions[-int(limit):]]


__all__ = ["GovernorStats", "MorphBounds", "MorphGovernor", "ShadowEvaluator"]
