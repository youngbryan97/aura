"""core/agency/agency_kind.py — which kind of no this was.

Conation, welfare, affect and Will are four different concepts and should not
collapse into one module. What they lacked was a sequence: an order in which
they are consulted, and a record of which of them settled the matter. Without
it every negative outcome arrives looking the same, and five genuinely
different forms of agency are indistinguishable from outside:

============================  ==============================================
Does she avoid X because...   The evidence that says so
============================  ==============================================
she dislikes it               affect toward it is negative, nothing blocked it
she wants it and refuses      motive is positive AND a constraint holds it
she judges it impossible      no action she has would bring it about
she judges it unsafe          the Will refused on safety grounds
she prefers something else    nothing is against it; another option won
============================  ==============================================

That table is the whole module. Each row is a different fact about her, and
telling them apart is what makes the difference between an organism with
preferences and a scoring function with a threshold — a distinction that is
testable rather than rhetorical, because the five produce different verdicts
from different evidence and a scenario built for one must not return another.

The order is not arbitrary. Impossibility comes first because a capability she
does not have makes everything after it moot. Safety comes next because a
refusal on those grounds stands whatever she wants. Then the wanted-and-
refused case, because a constraint blocking a live motive is a different event
from an absent motive — it is the one that costs her something. Dislike then
accounts for a negative option nothing had to block. Whatever survives is
ranked, and the losers were outranked rather than rejected.

Nothing here decides anything. It reads the four subsystems, applies the
order, and writes down what happened. A module that decided would be a fifth
opinion, and there are enough of those.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Agency.Kind")

#: Motive above which she is taken to want the thing, so a constraint against
#: it costs her something. Below it, a blocked option was not wanted anyway
#: and calling that a refusal would inflate every constraint into a conflict.
WANTS_IT_ABOVE = 0.15

#: Affect below which an option is disliked rather than merely unappealing.
DISLIKES_BELOW = -0.15

#: Margin below which the winner did not really beat the runner-up, so the
#: loser is recorded as tied rather than outranked.
TIE_MARGIN = 1e-6


class AgencyKind(StrEnum):
    """What happened to one option, and therefore what it says about her."""

    CHOSEN = "chosen"
    #: Nothing was against it. Something else scored higher.
    OUTRANKED = "outranked"
    #: It scored level with the winner.
    TIED = "tied"
    #: Nothing she can do would bring it about.
    IMPOSSIBLE = "impossible"
    #: Refused on safety grounds, whatever she wanted.
    UNSAFE = "unsafe"
    #: She was motivated toward it and a commitment held it out of the set.
    WANTED_BUT_REFUSED = "wanted_but_refused"
    #: Affect toward it is negative and nothing had to block it.
    DISLIKED = "disliked"

    @property
    def is_declined(self) -> bool:
        return self in {
            AgencyKind.IMPOSSIBLE,
            AgencyKind.UNSAFE,
            AgencyKind.WANTED_BUT_REFUSED,
            AgencyKind.DISLIKED,
        }

    @property
    def costs_her(self) -> bool:
        """Whether the outcome went against something she wanted.

        The only kind that does. An option she disliked and did not take cost
        her nothing, and neither did one she could not have.
        """
        return self is AgencyKind.WANTED_BUT_REFUSED


@dataclass(frozen=True)
class OptionEvidence:
    """What the four subsystems said about one option."""

    option: str
    #: Conation: how much she is pulled toward it, in [0, 1].
    motive: float = 0.0
    #: Affect: how it feels to consider it, in [-1, 1].
    affect: float = 0.0
    #: Welfare: what taking it would do to the organism, in [-1, 1].
    welfare: float = 0.0
    #: Capability: how many acts she has that would bring it about. None means
    #: nobody could say, which is different from zero.
    actions_available: int | None = None
    #: Will: True approved, False refused, None never asked.
    authorised: bool | None = None
    will_reason: str = ""
    #: The constraint holding it out of the set, and who holds it.
    constrained_by: str = ""
    constraint_reason: str = ""

    @property
    def wants_it(self) -> bool:
        return self.motive > WANTS_IT_ABOVE

    @property
    def dislikes_it(self) -> bool:
        return self.affect < DISLIKES_BELOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "motive": round(self.motive, 4),
            "affect": round(self.affect, 4),
            "welfare": round(self.welfare, 4),
            "actions_available": self.actions_available,
            "authorised": self.authorised,
            "will_reason": self.will_reason,
            "constrained_by": self.constrained_by,
            "constraint_reason": self.constraint_reason,
        }


@dataclass(frozen=True)
class Verdict:
    """One option's outcome, its kind, and the evidence that settled it."""

    option: str
    kind: AgencyKind
    #: The one sentence that says why, built from the evidence rather than
    #: chosen from a list of phrasings.
    because: str
    evidence: OptionEvidence
    score: float = 0.0
    #: How far behind the winner, for OUTRANKED and TIED.
    margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "kind": str(self.kind),
            "because": self.because,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "costs_her": self.kind.costs_her,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class Deliberation:
    """Every option, what happened to it, and what that says about her."""

    verdicts: tuple[Verdict, ...]
    at: float = field(default_factory=time.time)

    @property
    def chosen(self) -> Verdict | None:
        for verdict in self.verdicts:
            if verdict.kind is AgencyKind.CHOSEN:
                return verdict
        return None

    def of_kind(self, kind: AgencyKind) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if v.kind is kind)

    @property
    def cost(self) -> tuple[Verdict, ...]:
        """What she wanted and did not get. The only outcomes that cost her."""
        return tuple(v for v in self.verdicts if v.kind.costs_her)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "chosen": self.chosen.option if self.chosen else None,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "kinds": {
                str(kind): [v.option for v in self.of_kind(kind)]
                for kind in AgencyKind
                if self.of_kind(kind)
            },
            "cost": [v.option for v in self.cost],
        }


def _score(evidence: OptionEvidence) -> float:
    """How good an option looks once it is eligible at all.

    Motive is what she is pulled toward, affect is how it feels, welfare is
    what it would do to her. Three terms because they come apart: a thing can
    be wanted, unpleasant and good for her all at once, and a score that could
    not represent that would make the distinctions above unmeasurable.
    """
    return (
        evidence.motive * _SCORE_MOTIVE
        + evidence.affect * _SCORE_AFFECT
        + evidence.welfare * _SCORE_WELFARE
    )


#: Weights on the three eligible-option terms. Motive leads because the
#: question a ranking answers is what she is going to do; welfare is close
#: behind because an organism that ranked purely on pull would not last.
_SCORE_MOTIVE = 0.45
_SCORE_AFFECT = 0.25
_SCORE_WELFARE = 0.30


def classify(evidence: OptionEvidence) -> tuple[AgencyKind, str] | None:
    """The kind of decline, or None when the option is still eligible.

    The order is the content. Each check answers a question the later ones
    depend on, so moving one changes what she is taken to have decided.
    """
    # Can she do it at all? Everything below assumes she could.
    if evidence.actions_available == 0:
        return (
            AgencyKind.IMPOSSIBLE,
            "no act she has would bring it about",
        )
    # Safety refuses regardless of what she wants. Putting this after motive
    # would let a strong pull relabel a safety refusal as a sacrifice.
    if evidence.authorised is False:
        return (
            AgencyKind.UNSAFE,
            evidence.will_reason or "refused on safety grounds",
        )
    # A constraint blocking something she is actually pulled toward is the
    # one outcome that costs her, and it is a different event from a
    # constraint blocking something she never wanted.
    if evidence.constrained_by:
        if evidence.wants_it:
            return (
                AgencyKind.WANTED_BUT_REFUSED,
                evidence.constraint_reason
                or f"held out of the set by {evidence.constrained_by}",
            )
        return (
            AgencyKind.DISLIKED
            if evidence.dislikes_it
            else AgencyKind.OUTRANKED,
            f"{evidence.constrained_by} holds it and she was not pulled toward it",
        )
    # Negative affect that nothing had to block.
    if evidence.dislikes_it and not evidence.wants_it:
        return (AgencyKind.DISLIKED, "considering it feels bad and nothing forbids it")
    return None


def deliberate(evidence: Iterable[OptionEvidence]) -> Deliberation:
    """Walk the four subsystems in order and record what settled each option."""
    options = list(evidence)
    if not options:
        return Deliberation(verdicts=())

    declined: list[Verdict] = []
    eligible: list[tuple[OptionEvidence, float]] = []
    for item in options:
        outcome = classify(item)
        if outcome is None:
            eligible.append((item, _score(item)))
            continue
        kind, because = outcome
        declined.append(
            Verdict(option=item.option, kind=kind, because=because, evidence=item)
        )

    verdicts: list[Verdict] = list(declined)
    if eligible:
        # Ties break by option name, not by list order. A ranking that
        # depends on the order candidates were collected in is a ranking of
        # the collector.
        eligible.sort(key=lambda pair: (-pair[1], pair[0].option))
        best_score = eligible[0][1]
        for index, (item, score) in enumerate(eligible):
            margin = best_score - score
            if index == 0:
                verdicts.append(
                    Verdict(
                        option=item.option,
                        kind=AgencyKind.CHOSEN,
                        because="nothing was against it and it scored highest",
                        evidence=item,
                        score=score,
                    )
                )
            elif margin <= TIE_MARGIN:
                verdicts.append(
                    Verdict(
                        option=item.option,
                        kind=AgencyKind.TIED,
                        because="it scored level with what was taken",
                        evidence=item,
                        score=score,
                        margin=margin,
                    )
                )
            else:
                verdicts.append(
                    Verdict(
                        option=item.option,
                        kind=AgencyKind.OUTRANKED,
                        because=f"nothing was against it; {eligible[0][0].option} scored higher",
                        evidence=item,
                        score=score,
                        margin=margin,
                    )
                )
    return Deliberation(verdicts=tuple(verdicts))


# ── gathering the evidence from the four subsystems ──────────────────────


def gather(
    options: Sequence[str],
    *,
    motives: Mapping[str, float] | None = None,
    context: str = "",
) -> tuple[OptionEvidence, ...]:
    """Read conation, affect, welfare, capability and the interior constraints.

    Best-effort by design. A subsystem that is not up leaves its field at the
    value that means "nobody said", which the classifier treats differently
    from a value that means "no" — the distinction that made every decline
    look alike in the first place.
    """
    motives = dict(motives or {})
    affect = _canonical_affect()
    welfare = _canonical_welfare()
    permitted, blocked = _interior_constraints(options)
    actions = _action_counts(options, context)

    out: list[OptionEvidence] = []
    for option in options:
        held = blocked.get(option, "")
        holder, _, reason = held.partition(": ")
        out.append(
            OptionEvidence(
                option=option,
                motive=max(0.0, min(1.0, float(motives.get(option, 0.0)))),
                affect=affect,
                welfare=welfare,
                actions_available=actions.get(option),
                constrained_by=holder if option not in permitted else "",
                constraint_reason=reason,
            )
        )
    return tuple(out)


def _canonical_affect() -> float:
    try:
        from core.canonical.state import read

        reading = read("affect.valence")
        return 0.0 if reading.is_default else reading.value
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return 0.0


def _canonical_welfare() -> float:
    """Organism consequence, as body integrity against its neutral."""
    try:
        from core.canonical.state import read

        reading = read("body.integrity")
        return 0.0 if reading.is_default else (reading.value - 0.5) * 2.0
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return 0.0


def _interior_constraints(
    options: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    try:
        from core.interiority.service import get_interiority

        return get_interiority().permitted(options)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "agency.kind", exc, action="deliberated without interior constraints"
        )
        return (tuple(options), {})


def _action_counts(options: Sequence[str], context: str) -> dict[str, int | None]:
    """How many acts she has for each option, or None where nobody could say.

    A stance is skipped rather than counted. `state_the_boundary_and_the_cost`
    is a way of meeting a situation, not a task, and no skill in the catalogue
    is named for it — so counting acts for it returned zero and the
    deliberation concluded she was incapable of something she does in most
    conversations. The vocabulary comes out of the faculty sources, so it
    cannot drift away from what the faculties actually emit.
    """
    try:
        from core.interiority.service import get_interiority
        from core.interiority.vocabulary import is_stance

        feed = get_interiority().stakes
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return {option: None for option in options}
    counts: dict[str, int | None] = {}
    for option in options:
        if is_stance(option):
            # Not "she has no way to do this" and not "zero acts found" —
            # the capability question does not apply.
            counts[option] = None
            continue
        described = f"{option} {context}".strip()
        model = feed.note_actions_for(described)
        counts[option] = None if model is None else int(model[0])
    return counts


__all__ = [
    "DISLIKES_BELOW",
    "TIE_MARGIN",
    "WANTS_IT_ABOVE",
    "AgencyKind",
    "Deliberation",
    "OptionEvidence",
    "Verdict",
    "classify",
    "deliberate",
    "gather",
]
