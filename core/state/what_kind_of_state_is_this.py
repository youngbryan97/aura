"""Every state holder, and whether it is an authority, a view, or scratch.

The conclusion the blind comparison cared about most was not "Aura needs
prettier code". It was that Aura needs lower causal ambiguity per unit of
functionality — and it made the point by listing what Aura has where the peers
have one thing:

    canonical self, AuraState, BeingRuntime state, conscious substrate state,
    interiority state, workspace state, somatic state, affect engine state,
    identity engine state, orchestrator state, kernel state...

and then said the honest thing: many of those are legitimately different. The
remaining problem is deciding which are authorities, which are derived
projections, and which are temporary computational state.

So that is what this holds. Three kinds:

* **An authority** decides a fact. Nothing else may. If two authorities claim
  the same fact, one of them is wrong and this file is where that shows.
* **A projection** is derived from an authority and says which one, and how
  fresh it is. A projection nobody can trace back to a source is a fork.
* **Scratch** lives for one turn or one pass and is never read afterwards.
  Naming it as scratch is what stops somebody persisting it.

The gate is that every holder is one of the three and every projection names
its source. A holder that is not in this table is not classified, and the
count of those only goes down.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhatKindOfStateIsThis")

__all__ = [
    "AHolder",
    "AKindOfState",
    "THE_HOLDERS",
    "how_the_state_is_organised",
    "what_is_not_classified",
]


class AKindOfState(StrEnum):
    AUTHORITY = "authority"
    PROJECTION = "projection"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class AHolder:
    """One thing that holds state, and what kind of holding it is."""

    where: str
    kind: AKindOfState
    #: What it decides, if it is an authority. What it shows, otherwise.
    holds: str
    #: For a projection: which authority it comes from, and how fresh it is.
    derived_from: str = ""
    fresh: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "where": self.where,
            "kind": str(self.kind),
            "holds": self.holds,
            "derived_from": self.derived_from,
            "fresh": self.fresh,
        }


#: The eleven the review listed, and the ones that turned up beside them.
#: Adding a holder is an edit here and a sentence about what it decides.
THE_HOLDERS: tuple[AHolder, ...] = (
    AHolder(
        where="core/state/aura_state.py:AuraState",
        kind=AKindOfState.AUTHORITY,
        holds="the mind's durable fields — identity, affect, motivation, "
        "cognition, world, soma, cold — 98 leaves, owned per field by "
        "core/state/who_owns_each_field.py",
    ),
    AHolder(
        where="core/kernel/aura_kernel.py:AuraKernel.state",
        kind=AKindOfState.PROJECTION,
        holds="the AuraState the current tick is acting on",
        derived_from="core/state/aura_state.py:AuraState",
        fresh="the state object itself, held for the length of a tick",
    ),
    AHolder(
        where="core/self/canonical_self.py",
        kind=AKindOfState.AUTHORITY,
        holds="who she is across restarts — name, narrative, stability, "
        "dominant affect; the identity fields AuraState carries a copy of",
    ),
    AHolder(
        where="core/being/runtime.py:BeingRuntime",
        kind=AKindOfState.PROJECTION,
        holds="the felt present, assembled for one moment",
        derived_from="AuraState plus the interiority service",
        fresh="per tick; nothing reads it from a later tick",
    ),
    AHolder(
        where="core/interiority/service.py",
        kind=AKindOfState.AUTHORITY,
        holds="what the 43 appraisal faculties found this turn, and the "
        "census across turns",
    ),
    AHolder(
        where="core/workspace/global_workspace.py",
        kind=AKindOfState.AUTHORITY,
        holds="what won attention and was broadcast, and the ignition record",
    ),
    AHolder(
        where="core/consciousness/global_workspace.py",
        kind=AKindOfState.PROJECTION,
        holds="the workspace as the consciousness layer reads it",
        derived_from="core/workspace/global_workspace.py",
        fresh="per broadcast",
    ),
    AHolder(
        where="core/state/aura_state.py:SomaState",
        kind=AKindOfState.PROJECTION,
        holds="hardware and interoceptive readings",
        derived_from="core/runtime/resource_observation.py and the host",
        fresh="per observation; a reading older than its interval is stale "
        "rather than wrong",
    ),
    AHolder(
        where="core/affect/affective_resonance.py",
        kind=AKindOfState.AUTHORITY,
        holds="valence, arousal and the dominant affect, which the response "
        "path reads and the feedback percepts feed back into",
    ),
    AHolder(
        where="core/orchestrator/main.py:RobustOrchestrator",
        kind=AKindOfState.SCRATCH,
        holds="queues, locks, task handles and the process's own liveness; "
        "no durable fact, which is why runtime_state() goes to the kernel",
    ),
    AHolder(
        where="core/runtime/whose_turn_it_is.py:TheTurn",
        kind=AKindOfState.AUTHORITY,
        holds="which turn owns the runtime, and what a cancelled one is "
        "still waiting on",
    ),
    AHolder(
        where="core/state/one_working_memory.py",
        kind=AKindOfState.PROJECTION,
        holds="the accessor and capacity for cognition.working_memory",
        derived_from="core/state/aura_state.py:CognitiveContext",
        fresh="always; it reads the field rather than copying it",
    ),
    AHolder(
        where="core/runtime/event_spine.py:EventLog",
        kind=AKindOfState.AUTHORITY,
        holds="what happened, in order, appended and never edited",
    ),
    AHolder(
        where="core/knowledge/atomspace.py:AtomSpace",
        kind=AKindOfState.AUTHORITY,
        holds="the metagraph, its truth values and its attention economy",
    ),
    AHolder(
        where="core/state/what_a_phase_changed.py",
        kind=AKindOfState.SCRATCH,
        holds="one phase's buffered additions and removals, until the "
        "boundary commits them",
    ),
)


def how_the_state_is_organised() -> dict[str, Any]:
    """The whole table, by kind, with what each projection derives from."""
    by_kind: dict[str, list[dict[str, str]]] = {}
    for holder in THE_HOLDERS:
        by_kind.setdefault(str(holder.kind), []).append(holder.to_dict())
    return {
        "holders": len(THE_HOLDERS),
        "authorities": len(by_kind.get("authority", ())),
        "projections": len(by_kind.get("projection", ())),
        "scratch": len(by_kind.get("scratch", ())),
        "unclassified": what_is_not_classified(),
        "by_kind": by_kind,
    }


def what_is_not_classified() -> list[str]:
    """Holders that say nothing useful about what they are.

    An authority that does not say what it decides, a projection with no
    source or no freshness, or scratch with no description. Empty is the
    baseline and it only stays empty.
    """
    wrong: list[str] = []
    for holder in THE_HOLDERS:
        if not holder.holds.strip():
            wrong.append(f"{holder.where}: does not say what it holds")
            continue
        if holder.kind is AKindOfState.PROJECTION:
            if not holder.derived_from.strip():
                wrong.append(f"{holder.where}: a projection of nothing named")
            elif not holder.fresh.strip():
                wrong.append(f"{holder.where}: does not say how fresh it is")
    return wrong
