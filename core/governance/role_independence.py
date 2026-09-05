"""core/governance/role_independence.py — who filled which role, and were they the same?

Operationally: this measures how many DISTINCT sources filled the four roles in
one governed decision — proposer, authorizer, criterion-setter,
evidence-interpreter — and names the roles that collapsed onto one source.

The gap it closes is stated most clearly by an outside reviewer, and it is
correct:

    Even if every action passes through UnifiedWill and AuthorityGateway, the
    same architecture and model-generated context may participate in proposing
    the action, evaluating it, authorizing it, defining success, and
    interpreting the evidence. A model can be consistently wrong at every stage
    and still generate a complete receipt chain.

`WillDecision` records `source` — "who requested this action" — and that is the
only role it names. So a receipt showing PROCEED, a signature, and observed
evidence is fully compatible with one model having proposed the action, set the
bar it had to clear, and then declared it cleared. The receipt is not lying.
It was never asked the question.

This does not make decisions better. It makes the structure of a decision
legible, so "the Will approved it" can be distinguished from "the thing that
wanted it also graded it". A high-blast-radius action whose proposer and
interpreter are the same source is not thereby wrong — it is unwitnessed, and
that is a different and reportable state.

Deliberately NOT here: any attempt to score whether the decision was good. That
is the failure this module exists to name, and reproducing it one level up
would be worse than not trying.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """The four jobs in a governed decision that should not all be one job."""

    #: Who wanted the action. A model, a user, a scheduler, a skill.
    PROPOSER = "proposer"
    #: Which authority approved it.
    AUTHORIZER = "authorizer"
    #: Who defined what counts as success BEFORE the action ran. Preregistration
    #: is the whole value here: a criterion written afterwards is a description.
    CRITERION = "criterion"
    #: Who judged the evidence against that criterion.
    INTERPRETER = "interpreter"


#: The pair that matters most, and why. If the thing that proposed an action is
#: also the thing that decides whether it worked, no part of the chain can
#: report failure — which is exactly how a receipt chain stays complete while
#: being consistently wrong.
CRITICAL_PAIRS: tuple[tuple[Role, Role], ...] = (
    (Role.PROPOSER, Role.INTERPRETER),
    (Role.PROPOSER, Role.CRITERION),
    (Role.PROPOSER, Role.AUTHORIZER),
)

#: Effect scopes where a collapsed critical pair must be reported rather than
#: tolerated. Reading a file with one source in every role is unremarkable;
#: mutating the world that way is the thing worth seeing.
CONSEQUENTIAL_SCOPES = frozenset(
    {"privileged_mutation", "external_effect", "irreversible", "financial", "unknown"}
)


@dataclass(frozen=True)
class RoleAttribution:
    """Who filled each role in one decision.

    An unfilled role is `None` — UNATTRIBUTED, which is not the same as "filled
    by the proposer". Guessing would manufacture exactly the false comfort this
    module exists to remove, in the opposite direction.
    """

    proposer: str | None = None
    authorizer: str | None = None
    criterion: str | None = None
    interpreter: str | None = None
    effect_scope: str = "unknown"

    def get(self, role: Role) -> str | None:
        value = getattr(self, str(role))
        text = str(value).strip() if value is not None else ""
        return text or None

    def filled(self) -> dict[Role, str]:
        return {role: value for role in Role if (value := self.get(role)) is not None}

    def to_dict(self) -> dict[str, Any]:
        return {
            **{str(role): self.get(role) for role in Role},
            "effect_scope": self.effect_scope,
        }


@dataclass(frozen=True)
class IndependenceReport:
    attribution: RoleAttribution
    distinct_sources: int
    unattributed: tuple[Role, ...]
    collapsed: tuple[tuple[Role, Role], ...]
    consequential: bool

    @property
    def fully_circular(self) -> bool:
        """One source filled every role that was filled at all."""
        filled = self.attribution.filled()
        return len(filled) >= 2 and self.distinct_sources == 1

    @property
    def self_graded(self) -> bool:
        """Did the thing that wanted the action also decide whether it worked?

        The single pair that matters most. Every other collapse degrades the
        decision; this one removes the possibility of the decision being
        reported as a failure, which is precisely how a receipt chain stays
        complete while being consistently wrong.
        """
        proposer = self.attribution.get(Role.PROPOSER)
        interpreter = self.attribution.get(Role.INTERPRETER)
        return proposer is not None and proposer == interpreter

    @property
    def witnessed(self) -> bool:
        """Did anything other than the proposer touch this decision?

        The honest bar. Not "was this correct" — nothing here can answer that —
        but "was anyone else involved at all".
        """
        proposer = self.attribution.get(Role.PROPOSER)
        if proposer is None:
            return False
        others = {
            value
            for role, value in self.attribution.filled().items()
            if role is not Role.PROPOSER
        }
        return bool(others - {proposer})

    @property
    def verdict(self) -> str:
        """unattributed | circular | self_graded | unwitnessed | witnessed.

        Ordered most-collapsed first, and `self_graded` sits above `witnessed`
        on purpose: a decision where three parties were involved but the
        proposer still graded its own work is not witnessed in the sense that
        matters, and reporting it as such was the first version's mistake.
        """
        if not self.attribution.filled():
            return "unattributed"
        if self.fully_circular:
            return "circular"
        if self.self_graded:
            return "self_graded"
        return "witnessed" if self.witnessed else "unwitnessed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.role_independence.v1",
            "attribution": self.attribution.to_dict(),
            "distinct_sources": self.distinct_sources,
            "unattributed": [str(role) for role in self.unattributed],
            "collapsed": [[str(a), str(b)] for a, b in self.collapsed],
            "consequential": self.consequential,
            "verdict": self.verdict,
            "witnessed": self.witnessed,
        }

    def describe(self) -> str:
        if self.verdict == "unattributed":
            return "no role attribution recorded for this decision"
        if self.verdict == "circular":
            source = next(iter(self.attribution.filled().values()))
            return (
                f"one source ({source}) proposed, judged and approved this "
                "decision; the receipt chain is complete and nothing in it "
                "could have reported failure"
            )
        if self.verdict == "self_graded":
            proposer = self.attribution.get(Role.PROPOSER)
            return (
                f"{proposer} both proposed this action and judged whether it "
                "worked; no part of this chain could have reported failure"
            )
        if self.verdict == "unwitnessed":
            pairs = ", ".join(f"{a}={b}" for a, b in self.collapsed)
            return f"the proposer also filled: {pairs or 'every attributed role'}"
        return f"{self.distinct_sources} distinct sources across the filled roles"


def analyse(attribution: RoleAttribution) -> IndependenceReport:
    """Report the structure of a decision. Never judges whether it was right."""
    filled = attribution.filled()
    distinct = len(set(filled.values()))
    unattributed = tuple(role for role in Role if attribution.get(role) is None)
    collapsed = tuple(
        (first, second)
        for first, second in CRITICAL_PAIRS
        if (a := attribution.get(first)) is not None
        and (b := attribution.get(second)) is not None
        and a == b
    )
    return IndependenceReport(
        attribution=attribution,
        distinct_sources=distinct,
        unattributed=unattributed,
        collapsed=collapsed,
        consequential=str(attribution.effect_scope) in CONSEQUENTIAL_SCOPES,
    )


def from_mapping(payload: Mapping[str, Any]) -> RoleAttribution:
    """Build an attribution from a receipt-shaped mapping. Missing is None."""
    return RoleAttribution(
        proposer=payload.get("proposer") or payload.get("source"),
        authorizer=payload.get("authorizer") or payload.get("authority"),
        criterion=payload.get("criterion") or payload.get("criterion_setter"),
        interpreter=payload.get("interpreter") or payload.get("evidence_interpreter"),
        effect_scope=str(payload.get("effect_scope") or "unknown"),
    )


__all__ = [
    "CONSEQUENTIAL_SCOPES",
    "CRITICAL_PAIRS",
    "IndependenceReport",
    "Role",
    "RoleAttribution",
    "analyse",
    "from_mapping",
]
