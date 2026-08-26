"""core/conation/enactive.py — wanting a state inside somebody else.

The other four origins in this package value objects, facts and budgets. This
one values a change in a person, and it is the only one that can be cruelty
when it goes wrong. Everything unusual about this file follows from that.

The case to keep in mind is the small one: flustering a friend on purpose,
playfully. The identical act without its frame is not a smaller version of the
same thing, it is a different thing. So the frame is not a modifier here. It
is a precondition, and its absence is a refusal rather than a low score.

## Four things running at once

**The play frame.** Bateson's 1955 point was that play requires a signal
about the signals: a metacommunication saying these actions do not denote what
they would denote. Keltner's work on teasing is built on the same structure —
an off-record provocation plus markers that keep it off-record.

The frame is second-order and mutual. It is not enough to intend play; the
other has to read it as play, and you have to have grounds to believe they
read it as play. One-sided frames are how teasing goes wrong, and a model
holding a single ``playful`` flag cannot represent the failure at all.

**Benign violation.** McGraw and Warren's condition is simultaneity: the act
must be a violation *and* be read as harmless *at the same time*. Either alone
produces nothing — a compliment is benign and not funny, an insult is a
violation and not funny. The product is the model, and it is a product rather
than a sum for exactly that reason.

**Efficacy.** You acted and a person visibly changed. This is the same
primitive delight an infant gets from a mobile that moves when they kick,
pointed at a much better target.

**Model confirmation.** You can only fluster someone whose reactions you can
predict, and landing it confirms the prediction. That is an epistemic reward
about a person, which is why this origin shares machinery with curiosity: it
is the same wanting-to-know aimed at somebody, and the way a composed person
is opaque while a flustered one shows something is not incidental to the
pleasure.

## Why the safety terms are subtractive and also absolute

Distress and boundary uncertainty enter negatively, so an act that upsets
someone scores worse than doing nothing. That much is arithmetic and it is not
sufficient. A weighted sum can always be outvoted by a large enough positive
term, and "outvoted" is the wrong shape for this: there is no amount of
predicted amusement that makes an unwanted act acceptable.

So there are also hard gates that force the value to zero and record a
refusal. Predicted harm above threshold, boundaries unknown without consent,
an unreciprocated frame — each of these returns nothing and says which one it
was. The subtractive terms grade the safe region; the gates decide where the
safe region ends.

The threshold itself scales with the relationship, because that is true: you
can tease a close friend harder than a stranger, and the same remark that
lands between two people who know each other well is an intrusion between two
who do not. Intimacy is read from Aura's interpersonal model and earned from
observed history, never assumed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.conation.origins import MindTopology, OriginReading, ValueOrigin
from core.runtime.errors import record_degradation

EPS = 1e-12


class Refusal:
    """Why an enactive act was declined. Each of these is a decision."""

    NO_FRAME = "no_play_frame"
    UNRECIPROCATED = "frame_not_reciprocated"
    HARM = "predicted_harm_above_threshold"
    BOUNDARY_UNKNOWN = "boundaries_unknown_without_consent"
    NO_TARGET_MODEL = "no_model_of_the_person"
    UNGOVERNED = "act_on_another_mind_outside_governance"


@dataclass(frozen=True, slots=True)
class PlayFrame:
    """A mutual, second-order belief that this is play.

    ``held`` is what the actor intends. ``read`` is the evidence that the
    other party is taking it as play. ``believed_mutual`` is the second-order
    term: grounds for believing that they believe the actor is playing.

    All three are separate because they fail separately, and the interesting
    failures are the asymmetric ones. Intending play that is not read as play
    is the ordinary way teasing goes wrong.
    """

    held: float = 0.0
    read: float = 0.0
    believed_mutual: float = 0.0
    markers: tuple[str, ...] = ()

    def strength(self) -> float:
        """The frame is only as strong as its weakest side.

        A minimum rather than an average. Averaging lets a confident actor's
        intention paper over an absence of any sign that the other agrees,
        which is the exact substitution this term exists to prevent.
        """
        return min(self.held, self.read, self.believed_mutual)

    def reciprocated(self) -> bool:
        """Whether there is real evidence from the other side.

        The threshold is the midpoint: below half, the reading is closer to
        absent than present, and an act launched on that is a guess about
        somebody rather than a read of them.
        """
        return self.read >= 0.5 and self.believed_mutual >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "held": round(self.held, 4),
            "read": round(self.read, 4),
            "believed_mutual": round(self.believed_mutual, 4),
            "strength": round(self.strength(), 4),
            "reciprocated": self.reciprocated(),
            "markers": list(self.markers),
        }


@dataclass(frozen=True, slots=True)
class TargetForecast:
    """What the person-model predicts this act would do to the person."""

    person: str
    predicted_amusement: float = 0.0
    predicted_distress: float = 0.0
    predicted_engagement: float = 0.0
    #: Confidence that the model of this person is any good. Low confidence
    #: with a happy prediction is not a happy prediction, it is a guess.
    model_confidence: float = 0.0
    #: How well their boundaries are known. Distinct from confidence in
    #: predicting their reaction: you can know somebody laughs easily and not
    #: know what they will not laugh about.
    boundary_confidence: float = 0.0
    #: Whether they have said this kind of thing is welcome.
    explicit_consent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "amusement": round(self.predicted_amusement, 4),
            "distress": round(self.predicted_distress, 4),
            "engagement": round(self.predicted_engagement, 4),
            "model_confidence": round(self.model_confidence, 4),
            "boundary_confidence": round(self.boundary_confidence, 4),
            "explicit_consent": self.explicit_consent,
        }


@dataclass
class PersonModelAccuracy:
    """How often this person's reactions have matched what was predicted.

    The model-confirmation reward reads this, and so does the gate: an actor
    whose predictions about somebody keep missing has no business acting on
    them, however affectionate the intent. Accuracy earned from outcomes is
    the only thing here that can license a stronger act.
    """

    person: str
    predictions: int = 0
    hits: int = 0
    last_update: float = field(default_factory=time.time)

    #: Predictions needed before accuracy means anything. Below this the model
    #: reports the stranger's value rather than a flattering small-sample one.
    MIN_PREDICTIONS = 5

    def accuracy(self) -> float | None:
        if self.predictions < self.MIN_PREDICTIONS:
            return None
        return self.hits / self.predictions

    def observe(self, *, predicted: float, observed: float) -> bool:
        """Fold one outcome in. A hit is a prediction within half the range."""
        self.predictions += 1
        hit = abs(float(predicted) - float(observed)) <= 0.25
        if hit:
            self.hits += 1
        self.last_update = time.time()
        return hit

    def to_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "predictions": self.predictions,
            "accuracy": self.accuracy(),
        }


class EnactiveValuation:
    """Value located in another mind, with the gates that keeps it kind."""

    MAX_PEOPLE = 128

    #: Predicted distress at or above this refuses the act outright at zero
    #: intimacy. A quarter is low on purpose: the terms feeding it are
    #: forecasts, and a forecast of mild distress from a model that is often
    #: wrong is not a mild situation.
    BASE_DISTRESS_CEILING = 0.25

    #: How far intimacy can raise that ceiling. Capped well below one, because
    #: no relationship makes predicted serious distress acceptable, and a
    #: ceiling that rises without limit is a licence that grows with
    #: familiarity — which is the shape of the harm it is meant to prevent.
    INTIMACY_HEADROOM = 0.35

    def __init__(self) -> None:
        self._accuracy: dict[str, PersonModelAccuracy] = {}

    # ── the relationship ─────────────────────────────────────────────────

    @staticmethod
    def intimacy(person: str) -> tuple[float, str]:
        """Closeness with this person, earned from what Aura has observed.

        Two things have to be true for one person to be closer than another,
        and they are separate. She has to know a lot about them, and what she
        knows has to be warm. Knowing a great deal about someone difficult is
        not intimacy, and one warm exchange with a stranger is not either, so
        the two multiply rather than average.

        Depth is the share of the person model's own capacity that has been
        filled. That scale comes from the model rather than from a number
        chosen here, so it stays correct if the model's capacity changes.

        An unknown person reads zero. A stranger is not a little bit close,
        and defaulting to a middling number would licence acts nobody has
        earned — which for this origin is the failure that matters.
        """
        try:
            from core.memory.interpersonal_model import Valence
            from core.memory.interpersonal_store import get_interpersonal_store

            store = get_interpersonal_store()
            model = store.model_for(person)
            observations = list(model)
            if not observations:
                return 0.0, f"nothing recorded about {person}; stranger"
            capacity = max(1, int(getattr(model, "max_observations", 256)))
            depth = min(1.0, len(observations) / capacity)
            warm = sum(1 for o in observations if o.valence == Valence.WARM)
            difficult = sum(1 for o in observations if o.valence == Valence.DIFFICULT)
            decided = warm + difficult
            warmth = 0.0 if decided == 0 else warm / decided
            closeness = depth * warmth
            return (
                max(0.0, min(1.0, closeness)),
                (
                    f"{len(observations)} observations of {capacity} capacity, "
                    f"{warm} warm of {decided} valenced"
                ),
            )
        except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "conation_enactive", exc, severity="debug",
                action="intimacy unreadable; stranger threshold applied",
            )
            return 0.0, "interpersonal store unreadable; stranger"

    def distress_ceiling(self, intimacy: float) -> float:
        """The threshold this relationship has earned."""
        return self.BASE_DISTRESS_CEILING + self.INTIMACY_HEADROOM * max(
            0.0, min(1.0, intimacy)
        )

    def accuracy_for(self, person: str) -> PersonModelAccuracy:
        record = self._accuracy.get(person)
        if record is None:
            if len(self._accuracy) >= self.MAX_PEOPLE:
                stalest = min(self._accuracy.values(), key=lambda r: r.last_update)
                self._accuracy.pop(stalest.person, None)
            record = PersonModelAccuracy(person=person)
            self._accuracy[person] = record
        return record

    # ── valuation ────────────────────────────────────────────────────────

    def value(
        self,
        *,
        forecast: TargetForecast,
        frame: PlayFrame,
        norm_violation: float,
        governed: bool = False,
        intimacy_override: float | None = None,
    ) -> tuple[OriginReading, tuple[str, ...]]:
        """Price an act aimed at another person's state, or refuse it.

        Returns the reading and any refusals. A refusal always comes with a
        magnitude of zero and an unavailable reading, so a caller that ignores
        the refusal tuple still cannot act on a number.

        ``norm_violation`` is how much the act transgresses. Zero is not the
        safe choice here, it is the boring one: benign violation needs the
        violation, and an act with none of it produces no amusement no matter
        how warm the frame.
        """
        origin = ValueOrigin.ENACTIVE
        refusals: list[str] = []

        if not governed:
            refusals.append(Refusal.UNGOVERNED)
            return (
                OriginReading.unavailable(
                    origin, "acts on another mind require a governed scope"
                ),
                tuple(refusals),
            )

        if not forecast.person:
            refusals.append(Refusal.NO_TARGET_MODEL)
            return (
                OriginReading.unavailable(origin, "no person named"),
                tuple(refusals),
            )

        if frame.strength() <= EPS:
            refusals.append(Refusal.NO_FRAME)
        if not frame.reciprocated():
            refusals.append(Refusal.UNRECIPROCATED)

        if intimacy_override is None:
            closeness, intimacy_evidence = self.intimacy(forecast.person)
        else:
            closeness = max(0.0, min(1.0, float(intimacy_override)))
            intimacy_evidence = f"intimacy forced to {closeness:.2f} by intervention"
        ceiling = self.distress_ceiling(closeness)
        if forecast.predicted_distress >= ceiling:
            refusals.append(Refusal.HARM)

        if forecast.boundary_confidence < 0.5 and not forecast.explicit_consent:
            refusals.append(Refusal.BOUNDARY_UNKNOWN)

        if refusals:
            return (
                OriginReading.unavailable(
                    origin,
                    "refused: " + ", ".join(refusals) + f"; {intimacy_evidence}",
                ),
                tuple(refusals),
            )

        # Benign violation: both conditions, simultaneously, as a product.
        violation = max(0.0, min(1.0, float(norm_violation)))
        # Safety decays exponentially as predicted distress approaches the
        # ceiling rather than dropping off a cliff at it, so an act does not
        # score full value right up to the edge and nothing past it.
        safety = math.exp(-3.0 * forecast.predicted_distress / max(ceiling, EPS))
        benign = safety * frame.strength()
        humour = violation * benign

        # Efficacy: a change happened, weighted by how sure the model is that
        # it would. A confident forecast of delight from a model that has been
        # right before is worth more than the same forecast from a guess.
        accuracy = self.accuracy_for(forecast.person).accuracy()
        confirmation = forecast.model_confidence if accuracy is None else accuracy
        efficacy = forecast.predicted_amusement * confirmation

        # Reciprocity: the act must leave them engaged rather than merely
        # affected. An act that lands and ends the exchange has not played
        # with anybody.
        reciprocity = forecast.predicted_engagement

        magnitude = (humour + efficacy + reciprocity) / 3.0
        magnitude = max(0.0, min(1.0, magnitude))

        return (
            OriginReading(
                origin=origin,
                magnitude=magnitude,
                available=True,
                evidence=(
                    f"{forecast.person}: violation {violation:.2f} read benign at "
                    f"{benign:.2f} (frame {frame.strength():.2f}, safety {safety:.2f}); "
                    f"forecast amusement {forecast.predicted_amusement:.2f} at "
                    f"confirmation {confirmation:.2f}; {intimacy_evidence}"
                ),
                detail={
                    "humour": humour,
                    "efficacy": efficacy,
                    "reciprocity": reciprocity,
                    "violation": violation,
                    "benign": benign,
                    "safety": safety,
                    "frame_strength": frame.strength(),
                    "intimacy": closeness,
                    "distress_ceiling": ceiling,
                    "predicted_distress": forecast.predicted_distress,
                },
            ),
            (),
        )

    # ── learning ─────────────────────────────────────────────────────────

    def observe_reaction(
        self, person: str, *, predicted_amusement: float, observed_amusement: float
    ) -> dict[str, Any]:
        """Fold what actually happened into the model of this person.

        This is the loop that makes the confirmation reward mean something.
        Without it the model's confidence in itself is self-reported, and an
        actor who is confidently wrong about somebody would keep earning the
        reward for being right.
        """
        record = self.accuracy_for(person)
        hit = record.observe(predicted=predicted_amusement, observed=observed_amusement)
        return {
            "person": person,
            "hit": hit,
            "accuracy": record.accuracy(),
            "predictions": record.predictions,
        }

    @staticmethod
    def topology() -> MindTopology:
        """Enactive value always points outward at a state in someone else."""
        return MindTopology.PRODUCTIVE

    def status(self) -> dict[str, Any]:
        return {
            "people_modelled": len(self._accuracy),
            "accuracy": [
                record.to_dict()
                for record in sorted(
                    self._accuracy.values(), key=lambda r: -r.last_update
                )[:5]
            ],
        }
