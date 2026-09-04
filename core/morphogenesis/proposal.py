"""core/morphogenesis/proposal.py — what a cell may ask for, and what it costs.

A cell never changes the topology. It describes a change it wants and hands
that description upward. Everything between the wanting and the change —
budget, invariants, a shadow evaluation, governance, the commit — happens to
the description, where it can be refused, logged, and undone.

The previous cell layer emitted ``{"kind": "replication_candidate"}`` into an
action list that nothing read, so the population could never change size. A
proposal is the same intent with somewhere to go.

Every proposal carries its own undo. A transition that cannot say how to
reverse itself is refused before it is evaluated, because the lesion and
partition scenarios both depend on being able to put the graph back.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .graph import EdgeType, MorphEdge
from .types import clamp01, json_safe, stable_digest


class TransitionKind(StrEnum):
    """The morphological actions available to a cell.

    ``ROUTE`` changes an existing edge's weight rather than its existence,
    which is how traffic shifts without the population changing. ``MERGE``
    folds two cells into one composite and is the only transition that both
    removes and adds a node in the same step.
    """

    BIND = "bind"
    UNBIND = "unbind"
    SPAWN = "spawn"
    RETIRE = "retire"
    MIGRATE = "migrate"
    SPECIALIZE = "specialize"
    DESPECIALIZE = "despecialize"
    ROUTE = "route"
    MERGE = "merge"


#: Transitions that change how many cells exist. These are the ones a
#: replication bound has to count, and the ones a poisoned signal would try
#: to drive without limit.
POPULATION_TRANSITIONS = frozenset({TransitionKind.SPAWN, TransitionKind.RETIRE, TransitionKind.MERGE})

#: Transitions that change only the wiring. Cheap, reversible, and the ones a
#: task shift should mostly be made of.
WIRING_TRANSITIONS = frozenset({TransitionKind.BIND, TransitionKind.UNBIND, TransitionKind.ROUTE})


class RiskClass(StrEnum):
    """How much scrutiny a proposal earns.

    ``ROUTINE`` is reversible wiring inside one subsystem. ``ELEVATED`` changes
    the population or crosses a subsystem boundary. ``CRITICAL`` touches a
    protected cell or would disconnect the graph, and needs governance before
    anything is applied.
    """

    ROUTINE = "routine"
    ELEVATED = "elevated"
    CRITICAL = "critical"


_RISK_ORDER = {RiskClass.ROUTINE: 0, RiskClass.ELEVATED: 1, RiskClass.CRITICAL: 2}


def max_risk(*classes: RiskClass) -> RiskClass:
    return max(classes, key=lambda c: _RISK_ORDER[c], default=RiskClass.ROUTINE)


@dataclass(frozen=True)
class MorphTransition:
    """One step of a proposed change.

    A transition names its subject and carries only the fields its kind needs.
    ``BIND`` needs an edge; ``SPAWN`` needs a manifest to instantiate;
    ``MIGRATE`` needs a destination placement. The governor reads ``kind`` and
    validates the rest against it, so a malformed transition is caught before
    a substrate ever sees it.
    """

    kind: TransitionKind
    subject: str = ""
    edge: MorphEdge | None = None
    manifest_data: dict[str, Any] = field(default_factory=dict)
    placement: str = ""
    specialization: str = ""
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> str:
        """Return an empty string when well-formed, else why it is not."""
        if self.kind in {TransitionKind.BIND, TransitionKind.UNBIND, TransitionKind.ROUTE}:
            if self.edge is None:
                return f"{self.kind} needs an edge"
            if not self.edge.source or not self.edge.target:
                return f"{self.kind} edge is missing an endpoint"
            if self.edge.source == self.edge.target:
                return f"{self.kind} edge binds a cell to itself"
        if self.kind is TransitionKind.SPAWN and not self.manifest_data:
            return "spawn needs a manifest"
        if self.kind is TransitionKind.RETIRE and not self.subject:
            return "retire needs a subject"
        if self.kind is TransitionKind.MIGRATE:
            if not self.subject:
                return "migrate needs a subject"
            if not self.placement:
                return "migrate needs a destination placement"
        if self.kind is TransitionKind.SPECIALIZE:
            if not self.subject:
                return "specialize needs a subject"
            if not self.specialization:
                return "specialize needs a specialization"
        if self.kind is TransitionKind.DESPECIALIZE and not self.subject:
            return "despecialize needs a subject"
        if self.kind is TransitionKind.MERGE:
            if not self.subject:
                return "merge needs a subject"
            if len(self.metadata.get("absorb", ())) < 1:
                return "merge needs at least one cell to absorb"
        return ""

    def inverse(self) -> MorphTransition | None:
        """The transition that undoes this one, where one exists.

        SPAWN inverts to RETIRE of whatever it created, which is only knowable
        after the commit; the governor fills the subject in. MERGE has no
        single-step inverse and is undone by restoring a snapshot instead.
        """
        if self.kind is TransitionKind.BIND and self.edge is not None:
            return MorphTransition(kind=TransitionKind.UNBIND, subject=self.subject, edge=self.edge)
        if self.kind is TransitionKind.UNBIND and self.edge is not None:
            return MorphTransition(kind=TransitionKind.BIND, subject=self.subject, edge=self.edge)
        if self.kind is TransitionKind.ROUTE and self.edge is not None:
            return MorphTransition(
                kind=TransitionKind.ROUTE,
                subject=self.subject,
                edge=self.edge,
                weight=float(self.metadata.get("previous_weight", self.edge.weight)),
            )
        if self.kind is TransitionKind.SPAWN:
            return MorphTransition(kind=TransitionKind.RETIRE, subject=self.subject)
        if self.kind is TransitionKind.MIGRATE:
            return MorphTransition(
                kind=TransitionKind.MIGRATE,
                subject=self.subject,
                placement=str(self.metadata.get("previous_placement", "")),
            )
        if self.kind is TransitionKind.SPECIALIZE:
            return MorphTransition(kind=TransitionKind.DESPECIALIZE, subject=self.subject)
        if self.kind is TransitionKind.DESPECIALIZE:
            return MorphTransition(
                kind=TransitionKind.SPECIALIZE,
                subject=self.subject,
                specialization=str(self.metadata.get("previous_specialization", "")),
            )
        return None

    @property
    def risk(self) -> RiskClass:
        if self.kind in {TransitionKind.MERGE, TransitionKind.RETIRE}:
            return RiskClass.CRITICAL
        if self.kind in POPULATION_TRANSITIONS or self.kind is TransitionKind.MIGRATE:
            return RiskClass.ELEVATED
        return RiskClass.ROUTINE

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "subject": self.subject,
            "edge": self.edge.to_dict() if self.edge is not None else None,
            "manifest_data": json_safe(self.manifest_data),
            "placement": self.placement,
            "specialization": self.specialization,
            "weight": round(float(self.weight), 6),
            "metadata": json_safe(self.metadata),
            "risk": str(self.risk),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphTransition:
        payload = dict(data or {})
        raw_edge = payload.get("edge")
        return cls(
            kind=TransitionKind(str(payload.get("kind", TransitionKind.BIND))),
            subject=str(payload.get("subject", "")),
            edge=MorphEdge.from_dict(raw_edge) if raw_edge else None,
            manifest_data=dict(payload.get("manifest_data") or {}),
            placement=str(payload.get("placement", "")),
            specialization=str(payload.get("specialization", "")),
            weight=float(payload.get("weight", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MorphProposal:
    """A described change, with its evidence and its price.

    ``expected_benefit`` and ``estimated_cost`` are what the governor ranks by
    when more proposals arrive in a tick than the rate limit allows. They are
    the proposer's own numbers, so the shadow evaluation exists to check them:
    a policy that always claims a benefit of 1.0 gets caught by measurement,
    not by trusting the field.
    """

    proposer: str
    transitions: tuple[MorphTransition, ...]
    evidence: dict[str, Any] = field(default_factory=dict)
    expected_benefit: float = 0.0
    estimated_cost: float = 0.0
    subsystem: str = "generic"
    rationale: str = ""
    created_at: float = field(default_factory=time.time)
    proposal_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_benefit", clamp01(self.expected_benefit))
        object.__setattr__(self, "estimated_cost", max(0.0, float(self.estimated_cost)))
        if not self.proposal_id:
            # Deterministic in content, so the same proposal from the same
            # cell in the same graph state is the same id across a replay.
            parts = [self.proposer, self.subsystem]
            for transition in self.transitions:
                parts.append(str(transition.kind))
                parts.append(transition.subject)
                if transition.edge is not None:
                    parts.append("|".join(str(v) for v in transition.edge.key))
                parts.append(transition.placement)
                parts.append(transition.specialization)
            object.__setattr__(self, "proposal_id", "morph_" + stable_digest(*parts, length=18))

    @property
    def risk(self) -> RiskClass:
        return max_risk(*(t.risk for t in self.transitions))

    @property
    def population_delta(self) -> int:
        """Net change in cell count this proposal would cause."""
        delta = 0
        for transition in self.transitions:
            if transition.kind is TransitionKind.SPAWN:
                delta += 1
            elif transition.kind is TransitionKind.RETIRE:
                delta -= 1
            elif transition.kind is TransitionKind.MERGE:
                delta -= len(transition.metadata.get("absorb", ()))
        return delta

    def validate(self) -> str:
        if not self.transitions:
            return "a proposal with no transitions changes nothing"
        if not self.proposer:
            return "a proposal needs a proposer"
        for transition in self.transitions:
            problem = transition.validate()
            if problem:
                return problem
            if transition.kind is not TransitionKind.MERGE and transition.inverse() is None:
                return f"{transition.kind} cannot say how to reverse itself"
        return ""

    def rollback_plan(self) -> tuple[MorphTransition, ...]:
        """The inverses, in reverse order. Empty when any step is not
        individually reversible, which tells the governor to snapshot instead."""
        inverses: list[MorphTransition] = []
        for transition in reversed(self.transitions):
            inverse = transition.inverse()
            if inverse is None:
                return ()
            inverses.append(inverse)
        return tuple(inverses)

    def affected_cells(self) -> tuple[str, ...]:
        touched: set[str] = set()
        for transition in self.transitions:
            if transition.subject:
                touched.add(transition.subject)
            if transition.edge is not None:
                touched.add(transition.edge.source)
                touched.add(transition.edge.target)
            for absorbed in transition.metadata.get("absorb", ()):
                touched.add(str(absorbed))
        return tuple(sorted(touched))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "subsystem": self.subsystem,
            "transitions": [t.to_dict() for t in self.transitions],
            "evidence": json_safe(self.evidence),
            "expected_benefit": round(float(self.expected_benefit), 6),
            "estimated_cost": round(float(self.estimated_cost), 6),
            "rationale": self.rationale,
            "risk": str(self.risk),
            "population_delta": self.population_delta,
            "affected_cells": list(self.affected_cells()),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphProposal:
        payload = dict(data or {})
        return cls(
            proposer=str(payload.get("proposer", "")),
            transitions=tuple(MorphTransition.from_dict(t) for t in payload.get("transitions", [])),
            evidence=dict(payload.get("evidence") or {}),
            expected_benefit=float(payload.get("expected_benefit", 0.0)),
            estimated_cost=float(payload.get("estimated_cost", 0.0)),
            subsystem=str(payload.get("subsystem", "generic")),
            rationale=str(payload.get("rationale", "")),
            created_at=float(payload.get("created_at", time.time())),
            proposal_id=str(payload.get("proposal_id", "")),
        )


class Decision(StrEnum):
    """What happened to a proposal. ``DEFERRED`` is not ``REJECTED``: it means
    the proposal was well-formed and lost to a rate limit or a cooldown, and a
    caller that treats the two the same throws away work that was fine."""

    APPLIED = "applied"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class MorphTransaction:
    """The record of one adjudicated proposal.

    Kept whatever the outcome, because a rejection tells you as much about the
    policy as an application does, and the poisoned-signal scenario is scored
    entirely on what was refused.
    """

    proposal: MorphProposal
    decision: Decision
    reason: str = ""
    graph_version_before: int = 0
    graph_version_after: int = 0
    applied_transitions: tuple[MorphTransition, ...] = ()
    shadow_score: float | None = None
    baseline_score: float | None = None
    receipt_id: str = ""
    duration_ms: float = 0.0
    substrate_events: tuple[dict[str, Any], ...] = ()
    at: float = field(default_factory=time.time)

    @property
    def committed(self) -> bool:
        return self.decision is Decision.APPLIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "decision": str(self.decision),
            "reason": self.reason,
            "graph_version_before": self.graph_version_before,
            "graph_version_after": self.graph_version_after,
            "applied_transitions": [t.to_dict() for t in self.applied_transitions],
            "shadow_score": self.shadow_score,
            "baseline_score": self.baseline_score,
            "receipt_id": self.receipt_id,
            "duration_ms": round(float(self.duration_ms), 3),
            "substrate_events": [json_safe(e) for e in self.substrate_events],
            "at": self.at,
        }


def bind(
    source: str,
    target: str,
    port: str,
    *,
    proposer: str,
    edge_type: EdgeType = EdgeType.DATA,
    weight: float = 1.0,
    latency_ms: float = 0.0,
    subsystem: str = "generic",
    benefit: float = 0.0,
    cost: float = 0.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    """Shorthand for the commonest proposal: one new binding."""
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(
            MorphTransition(
                kind=TransitionKind.BIND,
                subject=source,
                edge=MorphEdge(
                    source=source,
                    target=target,
                    edge_type=edge_type,
                    port=port,
                    weight=weight,
                    latency_ms=latency_ms,
                ),
            ),
        ),
        expected_benefit=benefit,
        estimated_cost=cost,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def unbind(
    edge: MorphEdge,
    *,
    proposer: str,
    subsystem: str = "generic",
    benefit: float = 0.0,
    cost: float = 0.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(MorphTransition(kind=TransitionKind.UNBIND, subject=edge.source, edge=edge),),
        expected_benefit=benefit,
        estimated_cost=cost,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def spawn(
    manifest_data: Mapping[str, Any],
    *,
    proposer: str,
    parent: str = "",
    placement: str = "",
    subsystem: str = "generic",
    benefit: float = 0.0,
    cost: float = 1.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    """A new cell. ``cost`` defaults above zero because replication that is
    free is replication without a bound, and the population would only grow."""
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(
            MorphTransition(
                kind=TransitionKind.SPAWN,
                subject="",
                manifest_data=dict(manifest_data),
                placement=placement,
                metadata={"parent": parent or proposer},
            ),
        ),
        expected_benefit=benefit,
        estimated_cost=max(0.05, float(cost)),
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def retire(
    subject: str,
    *,
    proposer: str,
    subsystem: str = "generic",
    benefit: float = 0.0,
    cost: float = 0.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(MorphTransition(kind=TransitionKind.RETIRE, subject=subject),),
        expected_benefit=benefit,
        estimated_cost=cost,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def route(
    edge: MorphEdge,
    new_weight: float,
    *,
    proposer: str,
    subsystem: str = "generic",
    benefit: float = 0.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(
            MorphTransition(
                kind=TransitionKind.ROUTE,
                subject=edge.source,
                edge=edge,
                weight=float(new_weight),
                metadata={"previous_weight": float(edge.weight)},
            ),
        ),
        expected_benefit=benefit,
        estimated_cost=0.02,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def specialize(
    subject: str,
    specialization: str,
    *,
    proposer: str,
    previous: str = "",
    subsystem: str = "generic",
    benefit: float = 0.0,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(
            MorphTransition(
                kind=TransitionKind.SPECIALIZE,
                subject=subject,
                specialization=specialization,
                metadata={"previous_specialization": previous},
            ),
        ),
        expected_benefit=benefit,
        estimated_cost=0.15,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def migrate(
    subject: str,
    placement: str,
    *,
    proposer: str,
    previous_placement: str = "",
    subsystem: str = "generic",
    benefit: float = 0.0,
    cost: float = 0.3,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> MorphProposal:
    return MorphProposal(
        proposer=proposer,
        subsystem=subsystem,
        transitions=(
            MorphTransition(
                kind=TransitionKind.MIGRATE,
                subject=subject,
                placement=placement,
                metadata={"previous_placement": previous_placement},
            ),
        ),
        expected_benefit=benefit,
        estimated_cost=cost,
        rationale=rationale,
        evidence=dict(evidence or {}),
    )


def rank_proposals(proposals: Sequence[MorphProposal]) -> list[MorphProposal]:
    """Order by benefit per unit cost, then by id.

    The id tiebreak is what stops a hash-ordered set from producing a different
    winner each process for proposals that score identically.
    """
    def score(proposal: MorphProposal) -> tuple[float, str]:
        cost = max(0.01, proposal.estimated_cost)
        return (-(proposal.expected_benefit / cost), proposal.proposal_id)

    return sorted(proposals, key=score)


__all__ = [
    "Decision",
    "MorphProposal",
    "MorphTransaction",
    "MorphTransition",
    "POPULATION_TRANSITIONS",
    "RiskClass",
    "TransitionKind",
    "WIRING_TRANSITIONS",
    "bind",
    "max_risk",
    "migrate",
    "rank_proposals",
    "retire",
    "route",
    "spawn",
    "specialize",
    "unbind",
]
