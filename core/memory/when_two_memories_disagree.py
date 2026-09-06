"""Three-way merge for semantic memory, with conflicts as objects.

Letta's memory is a git-backed filesystem, so it gets an explicit merge and
conflict vocabulary for free. The review said Aura's memory conflicts vary by
backend, and asked for three-way merge with conflict objects and
evidence-aware resolution — and specifically that a divergent identity or
self-model edit is never silently overwritten.

Three-way is the part that matters. Last-write-wins needs only two versions
and is wrong whenever both sides changed something: it cannot tell "you
changed this and I did not" from "we both changed it differently", so it
silently discards one of them. With the common ancestor, those are different
cases and only the second is a conflict.

What is resolved automatically:

* One side changed it — that side wins, and nothing was lost.
* Both changed it to the same value — agreement, not a conflict.
* Both added the same key with the same value — the same thing.

What is never resolved automatically:

* Both changed it differently. That is a conflict object naming the ancestor,
  both sides, and the evidence each carries.
* Either side touched an identity field. Even a one-sided change is offered
  rather than applied, because "who she is" is the one place where a merge
  being usually right is not good enough.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhenTwoMemoriesDisagree")

__all__ = [
    "AConflict",
    "AMerge",
    "THE_IDENTITY_FIELDS",
    "merge_three_ways",
]

#: Fields where even a one-sided change is offered rather than applied.
#: A merge that is usually right is not good enough for who she is.
THE_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "core_values",
        "current_narrative",
        "self_model",
        "identity",
        "commitments",
        "who_i_am",
    }
)


@dataclass(frozen=True)
class AConflict:
    """One field two sides changed differently, with what each side had."""

    field: str
    ancestor: Any
    mine: Any
    theirs: Any
    why: str
    #: What each side offers as its reason. Evidence-aware resolution needs a
    #: reason to weigh, and a conflict with no reasons can only be resolved by
    #: whoever is asked last.
    my_evidence: Any = None
    their_evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "ancestor": self.ancestor,
            "mine": self.mine,
            "theirs": self.theirs,
            "why": self.why,
            "my_evidence": self.my_evidence,
            "their_evidence": self.their_evidence,
        }


@dataclass
class AMerge:
    """What the merge settled and what it refused to."""

    merged: dict[str, Any] = field(default_factory=dict)
    conflicts: list[AConflict] = field(default_factory=list)
    #: Fields taken from one side because only that side changed them.
    took_mine: list[str] = field(default_factory=list)
    took_theirs: list[str] = field(default_factory=list)
    agreed: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "merged": dict(self.merged),
            "conflicts": [one.to_dict() for one in self.conflicts],
            "took_mine": list(self.took_mine),
            "took_theirs": list(self.took_theirs),
            "agreed": list(self.agreed),
        }


_MISSING = object()


def merge_three_ways(
    ancestor: dict[str, Any],
    mine: dict[str, Any],
    theirs: dict[str, Any],
    *,
    my_evidence: dict[str, Any] | None = None,
    their_evidence: dict[str, Any] | None = None,
    identity_fields: frozenset[str] | None = None,
) -> AMerge:
    """Merge two divergent memories against the version they both came from.

    The merged dict holds only what was settled. A field in conflict is absent
    from it on purpose: writing a guess there and listing the conflict beside
    it is how a caller ends up applying the guess.
    """
    guarded = identity_fields if identity_fields is not None else THE_IDENTITY_FIELDS
    my_reasons = my_evidence or {}
    their_reasons = their_evidence or {}
    out = AMerge()

    for name in sorted(set(ancestor) | set(mine) | set(theirs)):
        was = ancestor.get(name, _MISSING)
        one = mine.get(name, _MISSING)
        other = theirs.get(name, _MISSING)

        if one == other:
            # Both sides agree, whether or not they changed it.
            if one is not _MISSING:
                out.merged[name] = one
                if one != was:
                    out.agreed.append(name)
            continue

        i_changed = one != was
        they_changed = other != was
        guard = name in guarded

        if i_changed and not they_changed:
            if guard:
                out.conflicts.append(
                    AConflict(
                        field=name, ancestor=_shown(was), mine=_shown(one),
                        theirs=_shown(other),
                        why="an identity field: even a one-sided change is offered",
                        my_evidence=my_reasons.get(name),
                        their_evidence=their_reasons.get(name),
                    )
                )
                continue
            if one is not _MISSING:
                out.merged[name] = one
            out.took_mine.append(name)
            continue

        if they_changed and not i_changed:
            if guard:
                out.conflicts.append(
                    AConflict(
                        field=name, ancestor=_shown(was), mine=_shown(one),
                        theirs=_shown(other),
                        why="an identity field: even a one-sided change is offered",
                        my_evidence=my_reasons.get(name),
                        their_evidence=their_reasons.get(name),
                    )
                )
                continue
            if other is not _MISSING:
                out.merged[name] = other
            out.took_theirs.append(name)
            continue

        out.conflicts.append(
            AConflict(
                field=name,
                ancestor=_shown(was),
                mine=_shown(one),
                theirs=_shown(other),
                why="both sides changed it differently",
                my_evidence=my_reasons.get(name),
                their_evidence=their_reasons.get(name),
            )
        )

    if out.conflicts:
        logger.info(
            "merge left %d conflict(s): %s",
            len(out.conflicts), ", ".join(one.field for one in out.conflicts),
        )
    return out


def _shown(value: Any) -> Any:
    """What to put in a conflict for a field one side does not have."""
    return None if value is _MISSING else value
