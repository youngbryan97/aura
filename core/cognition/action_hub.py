"""core/cognition/action_hub.py — every proposal through one door, before authority.

An action reaches the world from six places. A habit fires. The planner
expands. The cortex writes one. A promoted rule proposes. The world model
predicts one is good. A skill offers itself. Each has its own scoring, its own
notion of confidence, and its own path to the gateway, so nobody can say what
fraction of Aura's actions came from learning versus from the model, and an
ablation of any one source measures whatever that source's path happened to
include.

The hub is one door, and it is deliberately in front of governance rather than
instead of it. :class:`ActionHub` collects proposals, prices them in the common
currency, resolves them through the preference algebra, and hands ONE candidate
to whatever authority normally decides. Nothing here can approve an action; the
hub cannot widen what Aura may do, only make visible where a proposal came
from.

The attribution is the point
----------------------------
Every proposal carries its source, so :meth:`ActionHub.attribution` answers the
question no ablation could: of the actions actually taken, how many came from
each source, and what would have been chosen without it. That second number is
a counterfactual over the same proposal set rather than a re-run, which makes
it cheap enough to compute on every decision instead of in a campaign.

Typed operators
---------------
A proposal carries a :class:`~core.cognition.procedure.Signature`, so the hub
can reject one whose preconditions the situation does not meet before anything
scores it. An untyped proposal is accepted and marked, because most of the six
sources do not have types yet and refusing them would turn the hub off.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.cognition.procedure import Signature

__all__ = [
    "Source",
    "Proposal",
    "Decision",
    "ActionHub",
    "get_action_hub",
    "reset_action_hub_for_test",
]


class Source(StrEnum):
    """Where a proposed action came from. One value per real proposer."""

    HABIT = "habit"
    PLANNER = "planner"
    CORTEX = "cortex"
    RULE = "rule"
    WORLD_MODEL = "world_model"
    SKILL = "skill"
    PROCEDURE = "procedure"

    @property
    def is_learned(self) -> bool:
        """Whether this source's competence was acquired rather than pretrained."""
        return self in (Source.HABIT, Source.RULE, Source.PROCEDURE, Source.WORLD_MODEL)


@dataclass(frozen=True, slots=True)
class Proposal:
    """One suggested action, with where it came from and what it expects."""

    action: str
    source: Source
    value: float = 0.0
    confidence: float = 0.0
    signature: Signature | None = None
    cost: float = 0.0
    risk: float = 0.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def typed(self) -> bool:
        return self.signature is not None

    @property
    def score(self) -> float:
        """Expected value net of cost and risk. The common currency."""
        return self.confidence * self.value - self.cost - self.risk

    def applicable(self, situation: Mapping[str, Any]) -> bool:
        """Whether the situation meets what this proposal needs.

        An untyped proposal is applicable: most sources have no signature yet,
        and refusing them would turn the hub off rather than type it.
        """
        return True if self.signature is None else self.signature.matches(situation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source.value,
            "value": self.value,
            "confidence": self.confidence,
            "cost": self.cost,
            "risk": self.risk,
            "score": self.score,
            "typed": self.typed,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """What the hub hands to authority, and what it would have handed without each source."""

    chosen: Proposal | None
    considered: tuple[Proposal, ...]
    rejected_untyped_mismatch: tuple[Proposal, ...]
    counterfactual: Mapping[str, str]
    impasse: str = ""

    @property
    def decided(self) -> bool:
        return self.chosen is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "considered": [p.to_dict() for p in self.considered],
            "rejected_for_preconditions": [p.action for p in self.rejected_untyped_mismatch],
            "without_each_source": dict(self.counterfactual),
            "impasse": self.impasse,
        }


class ActionHub:
    """One door for proposals. It decides nothing about authority."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.action_hub.ActionHub", reentrant=True)
        self._taken: dict[str, int] = {}
        self._proposed: dict[str, int] = {}
        self._counterfactual_changes: dict[str, int] = {}
        self._decisions = 0

    def decide(
        self, proposals: Sequence[Proposal], situation: Mapping[str, Any] | None = None
    ) -> Decision:
        """Choose one proposal to send onward, and record what each source did.

        A tie is reported as an impasse rather than settled here: the hub's job
        is to present a candidate, and a deadlock is information the impasse
        bus is for.
        """
        situation = dict(situation or {})
        applicable = [p for p in proposals if p.applicable(situation)]
        rejected = [p for p in proposals if not p.applicable(situation)]

        with self._lock:
            self._decisions += 1
            for proposal in proposals:
                self._proposed[proposal.source.value] = (
                    self._proposed.get(proposal.source.value, 0) + 1
                )

        if not applicable:
            return Decision(None, tuple(proposals), tuple(rejected), {},
                            impasse="rejection: nothing proposed was applicable")

        ranked = sorted(applicable, key=lambda p: -p.score)
        chosen = ranked[0]
        impasse = ""
        if len(ranked) > 1 and abs(ranked[0].score - ranked[1].score) < 1e-9:
            impasse = f"tie: {ranked[0].source.value} and {ranked[1].source.value}"

        counterfactual: dict[str, str] = {}
        for source in {p.source for p in applicable}:
            without = [p for p in applicable if p.source is not source]
            counterfactual[source.value] = (
                max(without, key=lambda p: p.score).action if without else "<nothing>"
            )

        with self._lock:
            self._taken[chosen.source.value] = self._taken.get(chosen.source.value, 0) + 1
            for source, alternative in counterfactual.items():
                if alternative != chosen.action:
                    self._counterfactual_changes[source] = (
                        self._counterfactual_changes.get(source, 0) + 1
                    )

        return Decision(chosen, tuple(applicable), tuple(rejected), counterfactual, impasse)

    def attribution(self) -> dict[str, Any]:
        """Of the actions taken, where they came from, and who mattered.

        ``changed_the_outcome`` is the counterfactual count: how often removing
        a source would have produced a different action. A source that proposes
        constantly and never changes the outcome is producing noise, and the
        two numbers together say so.
        """
        with self._lock:
            taken = dict(self._taken)
            proposed = dict(self._proposed)
            changed = dict(self._counterfactual_changes)
            decisions = self._decisions
        total = sum(taken.values())
        learned = sum(v for k, v in taken.items() if Source(k).is_learned)
        return {
            "decisions": decisions,
            "taken_by_source": dict(sorted(taken.items())),
            "proposed_by_source": dict(sorted(proposed.items())),
            "changed_the_outcome": dict(sorted(changed.items())),
            "learned_fraction": (learned / total) if total else None,
            "proposes_but_never_matters": sorted(
                source for source in proposed
                if proposed[source] >= 10 and changed.get(source, 0) == 0
            ),
        }


_lock = checked_lock("core.cognition.action_hub.singleton")
_hub: ActionHub | None = None


def get_action_hub() -> ActionHub:
    global _hub
    with _lock:
        if _hub is None:
            _hub = ActionHub()
        return _hub


def reset_action_hub_for_test() -> ActionHub:
    global _hub
    with _lock:
        _hub = ActionHub()
        return _hub
