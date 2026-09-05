"""core/interiority/appraisal.py — what an event means to this agent.

Appraisal theory's claim is that emotions are not responses to events.
They are responses to *relational meaning*: the same news is relief to
one person and catastrophe to another because they were holding
different things. Scherer's component process model calls the checks
that establish this the stimulus evaluation checks, run in four groups —
relevance, implication, coping potential, normative significance —
and Lazarus's core relational themes are the same idea from the other
end (Scherer 2001; Lazarus 1991; Ortony, Clore & Collins 1988).

So the frame below is not a feature vector a caller fills in. It is
*computed* from an :class:`~core.interiority.event.InteriorEvent` and
the agent's own :class:`~core.interiority.ledger.RelationalLedger` —
what she is attached to, what she promised, what she has custody of,
what she is trying to do. Change nothing about the event and change the
ledger, and the appraisal changes. That is the property that makes it an
appraisal rather than a classifier, and it is the property every
prototype in the review set lacks: all six take goal relevance,
attachment and harm as arguments.

Three checks here have no counterpart in any of the reviewed work, and
each fixes a specific misfire:

``other_capability``
    Could the other agent have done otherwise? Anger's function is to
    correct a welfare tradeoff (Sell, Tooby & Cosmides 2009), and
    correcting one that was never made is the commonest unjust anger
    there is. Without this variable, a system snaps at a person who was
    unable to comply exactly as hard as at one who would not.

``norm_endorsed``
    Is the violated standard one she holds, or one imposed on her?
    Guilt needs the first; the second produces resentment. Collapsing
    them is how a system learns to feel bad about breaking rules it
    never agreed to.

``publicity``
    Is anyone watching? Not to change what she does, but so that
    ``tests/interiority/test_audience_invariance.py`` can check that it
    doesn't. A motive with a reputational term is a different motive,
    and the only way to know which one is running is to vary the
    audience and measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from core.interiority.evidence import (
    Provenance,
    Reading,
    absent,
    assumed,
    ceiling_for,
    inferred,
    joint_confidence,
    measured,
    weakest,
)
from core.interiority.event import EventKind, InteriorEvent
from core.interiority.params import ParamKind, declare

#: Checks in the frame, in the order Scherer runs them. The order is not
#: decorative: relevance gates implication, implication gates coping, and
#: a faculty that reads coping potential without relevance is asking how
#: to handle something that does not concern this agent.
RELEVANCE_CHECKS = ("relevance", "novelty", "certainty", "urgency")
IMPLICATION_CHECKS = (
    "congruence",
    "expectation_deviation",
    "agency_self",
    "agency_other",
    "agency_circumstance",
    "other_capability",
    "other_coping",
    "irreversibility",
    "attachment_impact",
)
COPING_CHECKS = ("control", "power", "adjustment", "repair_available")
NORMATIVE_CHECKS = ("norm_fit", "norm_endorsed", "vulnerability", "publicity")

ALL_CHECKS = RELEVANCE_CHECKS + IMPLICATION_CHECKS + COPING_CHECKS + NORMATIVE_CHECKS

_AGENCY_TOLERANCE = declare(
    "interiority.appraisal.agency_sum_tolerance",
    0.05,
    unit="probability",
    basis=(
        "Causal attribution across self, other and circumstance is a partition, "
        "so the three shares sum to one. The tolerance is float slack, not a "
        "modelling choice."
    ),
    kind=ParamKind.DERIVED,
    sensitivity="Widen it and an attribution that double-counts a cause passes.",
    owner="core/interiority/appraisal.py",
)

_DEFAULT_CONTROL = declare(
    "interiority.appraisal.unknown_control",
    0.5,
    unit="probability",
    basis=(
        "When nothing is known about controllability the honest prior is "
        "indifference, and the reading is marked ASSUMED so it carries the "
        "0.25 intensity ceiling with it. Grok's valence_from_appraisal instead "
        "adds 0.25 * controllability with a 0.5 default, which reports a "
        "neutral event as mildly pleasant; the verification run measured "
        "valence 0.250 for goal_congruence 0."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Only reachable when a faculty needs coping potential and no measurement "
        "exists. Moving it moves nothing that is not already capped."
    ),
    owner="core/interiority/appraisal.py",
)


@dataclass(frozen=True)
class AppraisalFrame:
    """The relational meaning of one event for this agent, check by check."""

    event: InteriorEvent
    checks: Mapping[str, Reading]
    #: The ledger revision this was computed against, so a frame can be
    #: re-derived and compared after the ledger moves.
    ledger_revision: int = 0

    def __post_init__(self) -> None:
        missing = set(ALL_CHECKS) - set(self.checks)
        if missing:
            raise ValueError(
                f"appraisal frame is missing checks {sorted(missing)}; every check "
                "must be present, as an absent Reading if nothing is known. A "
                "missing key and a known-nothing are different situations and "
                "collapsing them is how a default becomes a measurement"
            )
        extra = set(self.checks) - set(ALL_CHECKS)
        if extra:
            raise ValueError(f"unknown appraisal checks {sorted(extra)}")
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    def __getitem__(self, name: str) -> Reading:
        return self.checks[name]

    def get(self, name: str) -> Reading:
        return self.checks.get(name, absent(source=f"frame:{name}"))

    def value(self, name: str) -> float:
        """The bare number. Use only where provenance is handled separately."""
        return self.checks[name].value

    def present(self, *names: str) -> bool:
        return all(self.checks[n].present for n in names)

    def ceiling(self, *names: str) -> float:
        """Intensity ceiling implied by the named checks."""
        return ceiling_for(self.checks[n] for n in names)

    def confidence(self, *names: str) -> float:
        return joint_confidence(self.checks[n] for n in names)

    def provenance(self, *names: str) -> Provenance:
        return weakest(self.checks[n] for n in names)

    def coping_potential(self) -> float:
        """Scherer's coping check as one number, for faculties that need it.

        Control (can the outcome be changed by anyone), power (can *I*
        change it) and adjustment (can I change myself to live with it)
        are separate variables and are kept separate; this is their
        maximum, because an agent copes if any one route is open.
        """
        return max(
            self.checks["control"].value,
            self.checks["power"].value,
            self.checks["adjustment"].value,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event": self.event.to_dict(),
            "ledger_revision": self.ledger_revision,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return lo
    return lo if x < lo else hi if x > hi else x


class AppraisalEngine:
    """Computes a frame from an event and what the agent is holding.

    The engine reads three things and only three: the event, the
    relational ledger, and an optional estimate of the other agent's
    state. It never reads free text, and it never takes an appraisal
    variable as an argument.
    """

    def __init__(self, ledger: "RelationalLedger") -> None:  # noqa: F821 — cycle
        self._ledger = ledger

    # ── relevance group ───────────────────────────────────────────────
    def _relevance(self, event: InteriorEvent) -> Reading:
        """Does this touch anything the agent is holding?

        Relevance is the maximum over the stakes the ledger says are
        exposed: an attachment to the subject, a commitment naming it, a
        custody obligation, an active goal, a work she authored. Nothing
        held, nothing at stake — and the reading is a measurement,
        because the ledger is a fact about her, not a guess about the
        world.
        """
        stakes = self._ledger.stakes_for(subject=event.subject, object_=event.object)
        if not stakes:
            return measured(0.0, source="ledger:no-stake")
        return measured(
            max(s.weight for s in stakes),
            source="ledger:" + ",".join(sorted({s.kind for s in stakes})[:4]),
        )

    def _novelty(self, event: InteriorEvent) -> Reading:
        seen = self._ledger.notes.times_seen(event.kind, event.subject)
        if seen < 0:
            return absent(source="ledger:unseen-counter-missing")
        return measured(1.0 / (1.0 + seen), source=f"ledger:seen={seen}")

    def _certainty(self, event: InteriorEvent) -> Reading:
        return measured(event.confidence, source=f"event:{event.source or 'unsourced'}")

    def _urgency(self, event: InteriorEvent) -> Reading:
        deadline = self._ledger.nearest_deadline(event.object)
        if deadline is None:
            return absent(source="ledger:no-deadline")
        remaining = max(0.0, deadline - event.at)
        # Urgency rises hyperbolically as the deadline approaches; one hour
        # out is the half-way point, which is the horizon the runtime's own
        # commitment records use.
        return measured(3600.0 / (3600.0 + remaining), source="ledger:deadline")

    # ── implication group ─────────────────────────────────────────────
    def _congruence(self, event: InteriorEvent) -> Reading:
        """Does this advance or block what the agent is holding? [-1, 1]."""
        delta = self._ledger.notes.goal_delta(event.object)
        if delta is None:
            if event.kind is EventKind.LOSS:
                return measured(-1.0, source="event:loss-is-incongruent-by-kind")
            return absent(source="ledger:no-goal-delta")
        return measured(_clamp(delta, -1.0, 1.0), source="ledger:goal-delta")

    def _expectation_deviation(self, event: InteriorEvent) -> Reading:
        expected = self._ledger.notes.expectation(event.kind, event.subject)
        if expected is None:
            return absent(source="ledger:no-expectation")
        return measured(abs(1.0 - expected), source="ledger:expectation")

    def _agency(self, event: InteriorEvent) -> tuple[Reading, Reading, Reading]:
        """Attribute the cause across self, other and circumstance."""
        attribution = self._ledger.notes.attribution(event.event_id)
        if attribution is None and event.subject:
            # A promise she made and did not keep is a self-attribution
            # whatever else the event was. Without this, guilt declined on
            # "required appraisal checks are absent: agency_self" while the
            # ledger held the broken promise that answers exactly that
            # question.
            if self._ledger.broken_promises(event.subject):
                return (
                    measured(1.0, source="ledger:promise-she-broke"),
                    measured(0.0, source="ledger:promise-she-broke"),
                    measured(0.0, source="ledger:promise-she-broke"),
                )
        if attribution is None:
            if event.kind is EventKind.OWN_ACTION:
                return (
                    measured(1.0, source="event:own-action"),
                    measured(0.0, source="event:own-action"),
                    measured(0.0, source="event:own-action"),
                )
            return (
                absent(source="ledger:no-attribution"),
                absent(source="ledger:no-attribution"),
                absent(source="ledger:no-attribution"),
            )
        own, other, circumstance = attribution
        total = own + other + circumstance
        if total <= 0.0:
            return (
                absent(source="ledger:empty-attribution"),
                absent(source="ledger:empty-attribution"),
                absent(source="ledger:empty-attribution"),
            )
        return (
            measured(own / total, source="ledger:attribution"),
            measured(other / total, source="ledger:attribution"),
            measured(circumstance / total, source="ledger:attribution"),
        )

    def _other_coping(
        self, event: InteriorEvent, other: "OtherEstimate | None"  # noqa: F821
    ) -> Reading:
        """Can the other agent change their own outcome?

        Scherer's coping check applied to the other rather than the self,
        and the variable that separates despair from anguish. Absent when
        no estimate exists, because assuming someone can cope is how a
        system answers grief with advice.
        """
        if other is None:
            return absent(source="other-minds:no-estimate")
        return other.coping

    def _other_capability(
        self, event: InteriorEvent, other: "OtherEstimate | None"  # noqa: F821
    ) -> Reading:
        """Could the other agent have done otherwise?

        The variable anger needs and no reviewed prototype has. It comes
        from the other-minds layer as an inference with a confidence, or
        it is absent — never assumed, because assuming capability is
        assuming blame.
        """
        if other is None:
            return absent(source="other-minds:no-estimate")
        return other.capability

    def _irreversibility(self, event: InteriorEvent) -> Reading:
        if event.kind is EventKind.LOSS:
            return event.channel("context").at_least(
                measured(1.0, source="event:loss-kind")
            )
        undo = self._ledger.notes.undo_cost(event.object)
        if undo is None:
            return absent(source="ledger:no-undo-cost")
        return measured(_clamp(undo), source="ledger:undo-cost")

    def _attachment_impact(self, event: InteriorEvent) -> Reading:
        if event.subject is None:
            return measured(0.0, source="event:no-subject")
        bond = self._ledger.attachment(event.subject)
        if bond is None:
            return measured(0.0, source="ledger:no-bond")
        return measured(bond, source="ledger:attachment")

    # ── coping group ──────────────────────────────────────────────────
    def _control(self, event: InteriorEvent) -> Reading:
        actions = self._ledger.notes.actions_that_change(event.object)
        if actions is None:
            return assumed(_DEFAULT_CONTROL.value, source="no-action-model")
        return measured(1.0 - 1.0 / (1.0 + float(actions)), source="ledger:actions")

    def _power(self, event: InteriorEvent) -> Reading:
        mine = self._ledger.notes.own_actions_that_change(event.object)
        if mine is None:
            return assumed(_DEFAULT_CONTROL.value, source="no-own-action-model")
        return measured(1.0 - 1.0 / (1.0 + float(mine)), source="ledger:own-actions")

    def _adjustment(self, event: InteriorEvent) -> Reading:
        """Can the agent change herself to live with the outcome?

        Substitutability of what was lost. A goal with alternatives can
        be adjusted to; a bond with a named person cannot, which is why
        grief and disappointment are different states rather than the
        same state at different sizes.
        """
        substitutes = self._ledger.substitutes_for(event.object, event.subject)
        if substitutes is None:
            return absent(source="ledger:no-substitute-model")
        return measured(1.0 - 1.0 / (1.0 + float(substitutes)), source="ledger:substitutes")

    def _repair_available(self, event: InteriorEvent) -> Reading:
        """Is there an act that would put it right?

        Guilt without this is shame, and shame's action tendency is
        concealment (Tangney & Dearing 2002). It is a separate check from
        control because an outcome can be irreversible while an apology
        is still available, and that is exactly the case guilt is for.
        """
        repairs = self._ledger.notes.repairs_for(event.event_id, event.subject)
        if repairs is None and event.subject:
            # A broken promise has an obvious repair — do the thing, or say
            # so — unless the ledger knows otherwise. Absent here turns guilt
            # into shame, whose action tendency is concealment, so the
            # default matters: the honest reading of a late review is that it
            # can still be done.
            if self._ledger.broken_promises(event.subject):
                return measured(0.8, source="ledger:promise-can-be-answered")
        if repairs is None:
            return absent(source="ledger:no-repair-model")
        return measured(1.0 if repairs else 0.0, source="ledger:repairs")

    # ── normative group ───────────────────────────────────────────────
    def _norm_fit(self, event: InteriorEvent) -> Reading:
        fit = self._ledger.notes.norm_fit(event.event_id)
        if fit is not None:
            return measured(_clamp(fit, -1.0, 1.0), source="ledger:norm-fit")

        # A promise she broke to this person is a standing normative fact
        # about dealing with them, and it was unreadable: settle_promise
        # recorded it and no faculty read `kept`, so an event involving
        # someone she had let down appraised exactly like one involving a
        # stranger. Weighted by what the promise was worth, and only for the
        # person it was made to — a breach to one person is not a general
        # failing.
        if event.subject:
            broken = self._ledger.broken_promises(event.subject)
            if broken:
                weight = max(p.importance for p in broken)
                return measured(
                    _clamp(-weight, -1.0, 1.0), source="ledger:promise-broken"
                )
        return absent(source="ledger:no-norm-judgement")

    def _norm_endorsed(self, event: InteriorEvent) -> Reading:
        endorsed = self._ledger.notes.norm_endorsement(event.event_id)
        if endorsed is not None:
            return measured(_clamp(endorsed), source="ledger:endorsement")
        # Making a promise IS endorsing the standard it creates. Guilt
        # separates from shame on whether she holds the standard she broke,
        # and this read absent for every promise she ever made.
        if event.subject and self._ledger.broken_promises(event.subject):
            return measured(1.0, source="ledger:promise-is-its-own-standard")
        return absent(source="ledger:no-endorsement")

    def _vulnerability(
        self, event: InteriorEvent, other: "OtherEstimate | None"  # noqa: F821
    ) -> Reading:
        if other is not None and other.vulnerability.present:
            return other.vulnerability
        if event.subject is None:
            return measured(0.0, source="event:no-subject")
        return absent(source="other-minds:no-vulnerability-estimate")

    def _publicity(self, event: InteriorEvent) -> Reading:
        observers = self._ledger.notes.observer_count(event.event_id)
        if observers < 0:
            return absent(source="ledger:observers-unknown")
        return measured(1.0 - 1.0 / (1.0 + float(observers)), source="ledger:observers")

    # ── the frame ─────────────────────────────────────────────────────
    def appraise(
        self,
        event: InteriorEvent,
        other: "OtherEstimate | None" = None,  # noqa: F821
    ) -> AppraisalFrame:
        agency_self, agency_other, agency_circumstance = self._agency(event)
        checks: dict[str, Reading] = {
            "relevance": self._relevance(event),
            "novelty": self._novelty(event),
            "certainty": self._certainty(event),
            "urgency": self._urgency(event),
            "congruence": self._congruence(event),
            "expectation_deviation": self._expectation_deviation(event),
            "agency_self": agency_self,
            "agency_other": agency_other,
            "agency_circumstance": agency_circumstance,
            "other_capability": self._other_capability(event, other),
            "other_coping": self._other_coping(event, other),
            "irreversibility": self._irreversibility(event),
            "attachment_impact": self._attachment_impact(event),
            "control": self._control(event),
            "power": self._power(event),
            "adjustment": self._adjustment(event),
            "repair_available": self._repair_available(event),
            "norm_fit": self._norm_fit(event),
            "norm_endorsed": self._norm_endorsed(event),
            "vulnerability": self._vulnerability(event, other),
            "publicity": self._publicity(event),
        }
        _check_agency_partition(checks)
        return AppraisalFrame(
            event=event, checks=checks, ledger_revision=self._ledger.revision
        )


def _check_agency_partition(checks: Mapping[str, Reading]) -> None:
    parts = [checks["agency_self"], checks["agency_other"], checks["agency_circumstance"]]
    if not all(p.present for p in parts):
        return
    total = sum(p.value for p in parts)
    if abs(total - 1.0) > _AGENCY_TOLERANCE.value:
        raise ValueError(
            f"causal attribution sums to {total:.3f}, not 1. Self, other and "
            "circumstance partition one cause; an attribution that does not sum "
            "to one is double-counting, and every faculty that reads blame reads "
            "it wrong"
        )


__all__ = [
    "ALL_CHECKS",
    "COPING_CHECKS",
    "IMPLICATION_CHECKS",
    "NORMATIVE_CHECKS",
    "RELEVANCE_CHECKS",
    "AppraisalEngine",
    "AppraisalFrame",
]
