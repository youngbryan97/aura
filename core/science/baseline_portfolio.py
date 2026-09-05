"""core/science/baseline_portfolio.py — the comparisons a result has to beat.

``core/evaluation/baselines.py`` holds one comparator: a compact stateful
controller that learns the same endpoint mapping, which is the right *kind* of
baseline and one of them. A result that beats Aura-minus-one-organ has shown
that the organ does something. It has not shown that the architecture is worth
having, because the comparison it never ran is against the resident model with
the same tools and no architecture at all.

Six baselines, and each answers a question the others cannot:

* ``cortex_only`` — the resident model, same weights, same prompt. **If Aura
  does not beat this, nothing else matters**, and this is the arm the second
  audit calls the fulcrum of the whole roadmap.
* ``cortex_with_tools`` — the same model with retrieval and tools but no
  persistent structure. Separates "tools help" from "the architecture helps".
* ``search_only`` — enumeration with no learning. Says whether the learned part
  is doing anything a wider search would not.
* ``symbolic_only`` and ``learned_only`` — the two halves of the hybrid, so a
  hybrid claim has to beat both of its own parts.
* ``simple_scaffold`` — the strongest thing a competent engineer would write in
  an afternoon. This is the one that hurts, and the one an external reader
  will ask about first.

Cortex parity
-------------
:func:`check_parity` is the other direction and the more dangerous one. An
architecture can make its own model *worse* — by truncating its context, by
prefilling it into a corner, by spending its budget before it answers — and
every internal comparison will still look fine because every arm has the same
handicap. ``PARITY_TOLERANCE`` is the margin below the cortex-only arm that
counts as a regression, and it is deliberately small: a system that degrades
its own model by five percent has to say what it bought with that.

Contamination
-------------
:func:`contamination_risk` classifies a task by how likely the resident model
is to have seen it. A developmental gain measured on a task the cortex
memorised is not a developmental gain, and "procedurally generated" is the only
class that settles it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "BaselineKind",
    "Contamination",
    "BaselineResult",
    "PortfolioVerdict",
    "PARITY_TOLERANCE",
    "compare",
    "check_parity",
    "contamination_risk",
]

#: How far below the cortex-only arm counts as degrading the model. Small on
#: purpose: an architecture that costs its own model more than this owes an
#: account of what it bought.
PARITY_TOLERANCE = 0.02


class BaselineKind(StrEnum):
    CORTEX_ONLY = "cortex_only"
    CORTEX_WITH_TOOLS = "cortex_with_tools"
    SEARCH_ONLY = "search_only"
    SYMBOLIC_ONLY = "symbolic_only"
    LEARNED_ONLY = "learned_only"
    SIMPLE_SCAFFOLD = "simple_scaffold"

    @property
    def question(self) -> str:
        return {
            BaselineKind.CORTEX_ONLY: "does the architecture beat the model it is built on",
            BaselineKind.CORTEX_WITH_TOOLS: "is it the architecture or is it the tools",
            BaselineKind.SEARCH_ONLY: "would a wider search have found this anyway",
            BaselineKind.SYMBOLIC_ONLY: "does the hybrid beat its symbolic half",
            BaselineKind.LEARNED_ONLY: "does the hybrid beat its learned half",
            BaselineKind.SIMPLE_SCAFFOLD: "does it beat what an engineer writes in an afternoon",
        }[self]


class Contamination(StrEnum):
    """How likely the resident model already knows the answer."""

    #: Generated from a seed at evaluation time. Cannot have been memorised.
    PROCEDURAL = "procedural"
    #: Written for this repository and never published.
    PRIVATE = "private"
    #: Published, but after the model's training cutoff.
    POST_CUTOFF = "post_cutoff"
    #: A public benchmark. Assume the model has seen it.
    PUBLIC = "public"
    #: Nobody classified it, which is the same as PUBLIC for claim purposes.
    UNKNOWN = "unknown"

    @property
    def supports_a_novelty_claim(self) -> bool:
        return self in (Contamination.PROCEDURAL, Contamination.PRIVATE)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One arm's score on the same tasks, under the same budget."""

    kind: BaselineKind
    score: float
    n: int
    ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "question": self.kind.question,
            "score": self.score,
            "n": self.n,
            "ci": list(self.ci) if self.ci else None,
        }


@dataclass(frozen=True, slots=True)
class PortfolioVerdict:
    """What a result is entitled to say after meeting the portfolio."""

    treatment: float
    baselines: tuple[BaselineResult, ...]
    contamination: Contamination
    missing: tuple[BaselineKind, ...]
    beaten: tuple[BaselineKind, ...]
    lost_to: tuple[BaselineKind, ...]

    @property
    def cortex_parity(self) -> bool | None:
        cortex = next((b for b in self.baselines if b.kind is BaselineKind.CORTEX_ONLY), None)
        return None if cortex is None else self.treatment >= cortex.score - PARITY_TOLERANCE

    @property
    def entitled_to_claim(self) -> str:
        """The strongest honest sentence this comparison supports."""
        if self.cortex_parity is False:
            return "the architecture degrades the model it is built on"
        if self.missing:
            return (
                "nothing yet: "
                + ", ".join(k.value for k in self.missing)
                + " was never run"
            )
        if self.lost_to:
            return "loses to " + ", ".join(k.value for k in self.lost_to)
        if not self.contamination.supports_a_novelty_claim:
            return (
                "beats every baseline on tasks the model may have memorised "
                f"({self.contamination.value}); no novelty claim"
            )
        return "beats every baseline on tasks that cannot have been memorised"

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment,
            "baselines": [b.to_dict() for b in self.baselines],
            "contamination": self.contamination.value,
            "supports_novelty_claim": self.contamination.supports_a_novelty_claim,
            "cortex_parity": self.cortex_parity,
            "missing": [k.value for k in self.missing],
            "beaten": [k.value for k in self.beaten],
            "lost_to": [k.value for k in self.lost_to],
            "entitled_to_claim": self.entitled_to_claim,
        }


#: Arms a capability claim must run. The rest are strengthening, not required.
REQUIRED = (
    BaselineKind.CORTEX_ONLY,
    BaselineKind.CORTEX_WITH_TOOLS,
    BaselineKind.SIMPLE_SCAFFOLD,
)


def compare(
    treatment: float,
    baselines: Sequence[BaselineResult],
    *,
    contamination: Contamination = Contamination.UNKNOWN,
    required: Sequence[BaselineKind] = REQUIRED,
) -> PortfolioVerdict:
    """Score a treatment against its portfolio and say what it may claim."""
    present = {b.kind for b in baselines}
    return PortfolioVerdict(
        treatment=treatment,
        baselines=tuple(baselines),
        contamination=contamination,
        missing=tuple(k for k in required if k not in present),
        beaten=tuple(sorted((b.kind for b in baselines if treatment > b.score), key=str)),
        lost_to=tuple(sorted((b.kind for b in baselines if treatment <= b.score), key=str)),
    )


def check_parity(
    full_system: Mapping[str, float], cortex_only: Mapping[str, float]
) -> dict[str, Any]:
    """Per-capability check that the architecture has not made its model worse.

    Reports every capability where the full system falls more than
    ``PARITY_TOLERANCE`` below the bare cortex. This is the regression an
    internal A/B cannot see, because both of its arms carry it.
    """
    shared = sorted(set(full_system) & set(cortex_only))
    regressions = [
        {
            "capability": name,
            "full_system": full_system[name],
            "cortex_only": cortex_only[name],
            "delta": full_system[name] - cortex_only[name],
        }
        for name in shared
        if full_system[name] < cortex_only[name] - PARITY_TOLERANCE
    ]
    improvements = [
        name for name in shared if full_system[name] > cortex_only[name] + PARITY_TOLERANCE
    ]
    return {
        "capabilities_compared": len(shared),
        "unmeasured": sorted(set(cortex_only) - set(full_system)),
        "regressions": regressions,
        "improvements": improvements,
        "parity_held": not regressions,
        "tolerance": PARITY_TOLERANCE,
    }


def contamination_risk(
    *, procedurally_generated: bool = False, published: bool = True, before_cutoff: bool = True
) -> Contamination:
    """Classify a task's exposure risk from three facts about it."""
    if procedurally_generated:
        return Contamination.PROCEDURAL
    if not published:
        return Contamination.PRIVATE
    return Contamination.PUBLIC if before_cutoff else Contamination.POST_CUTOFF
