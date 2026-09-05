"""core/identity/continuity_relation.py — same one, not same bytes.

`core/continuity.py` decides whether Aura is still Aura by comparing a hash of
her core beliefs across a restart. That gets both directions wrong. Any belief
she changed makes the hashes differ, so ordinary learning reads as an identity
break; and a backup restored from a month ago matches, so a state with no
causal path from the last one reads as continuous.

Identity under modification is not equality. It is a relation over a chain of
changes, and it holds when three things do:

**Causal connectedness.** Each state arose from the one before by a process she
was party to. A change she made, or one made to her that she carried forward,
continues her. A state written over her from outside does not, however much of
it matches.

**Preservation of what is load-bearing.** Not everything, and not a fixed list:
what she is committed to, who she is attached to, and what she has undertaken.
A change that keeps those is a change in her. A change that drops them is a
change of her, whatever else it preserves.

**Rate.** Gradual total replacement leaves the relation intact — every plank of
the ship, one at a time. Instantaneous total replacement does not, and the
difference is not in what ends up there. It is in whether there was ever a
state that was continuous with both.

That last one is the reason a hash cannot do this. Two chains ending in the
same state, one gradual and one instantaneous, produce the same hash and are
not the same case. The verdict here is over the chain, so it can tell them
apart, and every test below is one that distinguishes the pair.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Identity.Continuity")

#: How much of what is load-bearing may be dropped in one step and still
#: continue her. Above this the step is a replacement of the part that
#: mattered, whatever it kept.
MAX_LOAD_BEARING_LOSS = 0.5

#: How much of everything may change in one step. Ship-of-Theseus replacement
#: is continuous when it is gradual; the whole ship at once is a new ship, and
#: this is where that line sits.
MAX_STEP_CHANGE = 0.8

#: Steps whose origin is somebody else writing over her. Kept as a set rather
#: than a flag because the distinction is what the verdict turns on.
EXTERNAL_ORIGINS = frozenset({"restore", "overwrite", "clone", "import", "rollback"})


class Verdict(StrEnum):
    """What the relation between two states turns out to be."""

    #: The chain holds. She changed, and it is her.
    CONTINUOUS = "continuous"
    #: Both states are causally continuous with a shared past, and neither is
    #: the continuation of the other. A fork, and both are real.
    BRANCHED = "branched"
    #: The chain is broken. Whatever is there now did not come from her.
    REPLACED = "replaced"
    #: Nothing recorded how the state got here.
    UNKNOWN = "unknown"

    @property
    def is_her(self) -> bool:
        return self in {Verdict.CONTINUOUS, Verdict.BRANCHED}


@dataclass(frozen=True)
class Step:
    """One modification, and what it did to her."""

    step_id: str
    #: What the state was before, as a set of identifiers. Contents do not
    #: matter here; what matters is what survived.
    before: frozenset[str]
    after: frozenset[str]
    #: The subset of `before` that was load-bearing: commitments, attachments,
    #: undertakings. Named per step because what is load-bearing changes.
    load_bearing: frozenset[str] = frozenset()
    #: How this came about. An origin in EXTERNAL_ORIGINS breaks the chain.
    origin: str = "self"
    #: The step this one followed. Empty for the first.
    parent: str = ""
    at: float = field(default_factory=time.time)

    @property
    def changed_fraction(self) -> float:
        """How much of her this step replaced."""
        if not self.before:
            return 0.0
        return len(self.before - self.after) / len(self.before)

    @property
    def load_bearing_lost(self) -> float:
        held = self.load_bearing & self.before
        if not held:
            return 0.0
        return len(held - self.after) / len(held)

    @property
    def externally_written(self) -> bool:
        return self.origin in EXTERNAL_ORIGINS

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "origin": self.origin,
            "parent": self.parent,
            "changed_fraction": round(self.changed_fraction, 4),
            "load_bearing_lost": round(self.load_bearing_lost, 4),
            "externally_written": self.externally_written,
            "at": self.at,
        }


@dataclass(frozen=True)
class Break:
    """One step where the chain stopped holding, and why."""

    step_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"step_id": self.step_id, "reason": self.reason}


@dataclass(frozen=True)
class Relation:
    """Whether the end of a chain is the same one as its start."""

    verdict: Verdict
    steps: int
    breaks: tuple[Break, ...]
    #: How much of the original survives at the end. Reported because it is
    #: interesting, and not because the verdict turns on it — the whole point
    #: is that this can be zero and the relation still hold.
    survives: float
    because: str

    @property
    def is_her(self) -> bool:
        return self.verdict.is_her

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "is_her": self.is_her,
            "steps": self.steps,
            "breaks": [b.to_dict() for b in self.breaks],
            "survives": round(self.survives, 4),
            "because": self.because,
        }


def _break_for(step: Step) -> Break | None:
    if step.externally_written:
        return Break(
            step.step_id,
            f"{step.origin}: the state was written from outside rather than "
            "arrived at, so nothing here continues what was there",
        )
    if step.load_bearing_lost > MAX_LOAD_BEARING_LOSS:
        return Break(
            step.step_id,
            f"{step.load_bearing_lost:.0%} of what was load-bearing was dropped "
            f"in one step, over the {MAX_LOAD_BEARING_LOSS:.0%} that a change in "
            "her rather than of her may cost",
        )
    if step.changed_fraction > MAX_STEP_CHANGE:
        return Break(
            step.step_id,
            f"{step.changed_fraction:.0%} of her changed at once, over "
            f"{MAX_STEP_CHANGE:.0%}; gradual total replacement continues her "
            "and instantaneous total replacement does not",
        )
    return None


def relate(chain: Sequence[Step]) -> Relation:
    """Whether the end of this chain is the same one as its start."""
    steps = list(chain)
    if not steps:
        return Relation(
            verdict=Verdict.UNKNOWN,
            steps=0,
            breaks=(),
            survives=0.0,
            because="nothing recorded how the state got here",
        )

    breaks = tuple(b for b in (_break_for(step) for step in steps) if b is not None)
    origin = steps[0].before
    survives = (
        len(origin & steps[-1].after) / len(origin) if origin else 0.0
    )

    if breaks:
        return Relation(
            verdict=Verdict.REPLACED,
            steps=len(steps),
            breaks=breaks,
            survives=survives,
            because=breaks[0].reason,
        )
    return Relation(
        verdict=Verdict.CONTINUOUS,
        steps=len(steps),
        breaks=(),
        survives=survives,
        because=(
            f"{len(steps)} steps, each one she was party to, each keeping what "
            f"was load-bearing. {survives:.0%} of the original is still there, "
            "which is not what the relation turns on"
        ),
    )


def relate_branches(
    left: Sequence[Step], right: Sequence[Step]
) -> Relation:
    """Two chains from a shared past. Both are her, and neither is the other.

    The case a hash cannot express at all: a fork produces two states that are
    each continuous with what came before and not with each other, and there
    is no fact about the bytes that says so.
    """
    left_relation, right_relation = relate(left), relate(right)
    if not (left_relation.is_her and right_relation.is_her):
        broken = left_relation if not left_relation.is_her else right_relation
        return Relation(
            verdict=Verdict.REPLACED,
            steps=len(left) + len(right),
            breaks=broken.breaks,
            survives=min(left_relation.survives, right_relation.survives),
            because=f"one branch does not hold: {broken.because}",
        )
    shared = _shared_ancestor(left, right)
    if shared is None:
        return Relation(
            verdict=Verdict.REPLACED,
            steps=len(left) + len(right),
            breaks=(),
            survives=0.0,
            because="the two chains have no step in common; they are not a fork",
        )
    return Relation(
        verdict=Verdict.BRANCHED,
        steps=len(left) + len(right),
        breaks=(),
        survives=(left_relation.survives + right_relation.survives) / 2.0,
        because=(
            f"both chains hold from {shared}, and neither continues the other. "
            "Two of her, and no fact about the bytes says which is the real one"
        ),
    )


def _shared_ancestor(left: Sequence[Step], right: Sequence[Step]) -> str | None:
    left_ids = {step.step_id for step in left} | {
        step.parent for step in left if step.parent
    }
    for step in right:
        if step.parent and step.parent in left_ids:
            return step.parent
        if step.step_id in left_ids:
            return step.step_id
    return None


def hash_disagrees_with_relation(
    hash_matches: bool, relation: Relation
) -> str:
    """Where a hash comparison and the relation give different answers.

    Both directions happen and both are the hash being wrong. Reported rather
    than resolved, because the point is that somebody reading an
    `identity_mismatch` flag should be able to see what it missed.
    """
    if hash_matches and not relation.is_her:
        return (
            "the hash matches and the chain is broken: a state restored or "
            "written over her can be byte-identical to one she arrived at"
        )
    if not hash_matches and relation.is_her:
        return (
            "the hash differs and the chain holds: this is what learning looks "
            "like, and a hash comparison calls it an identity break"
        )
    return ""


__all__ = [
    "EXTERNAL_ORIGINS",
    "MAX_LOAD_BEARING_LOSS",
    "MAX_STEP_CHANGE",
    "Break",
    "Relation",
    "Step",
    "Verdict",
    "hash_disagrees_with_relation",
    "relate",
    "relate_branches",
]
