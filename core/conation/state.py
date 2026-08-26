"""core/conation/state.py — the typed motivational stance.

An ``Incentive`` is a candidate for motivation: an object, a target, a person,
an action. A ``ConativeState`` is what Aura's motivational system has to say
about one of them, and it is deliberately a vector rather than a scalar.

The scalar version is the mistake this package exists to correct. Reduce a
motivational state to one number and the four situations in the module
docstring of ``core/conation/origins.py`` become the same number, in the same
way that reducing them to valence and arousal already makes them the same
point. Utility is what you get at the END of arbitration, after the
constraints have been applied. It is not what a motivational state is.

## Wanting and liking are two fields, not one

Berridge and Robinson's dissociation is the single most load-bearing fact in
this file. Incentive salience — wanting — is cue-triggered, escalates before
contact, and is mesolimbic. Hedonic impact — liking — happens at contact and
runs on a much smaller set of opioid hotspots. They can be driven apart
experimentally in both directions, and in humans they come apart in ordinary
life all the time: the cake that looked irresistible and tasted of nothing.

A system that stores one number cannot represent that, so it cannot learn from
it either. Two fields with two prediction errors can:

    epsilon_wanting = realised_pull      - predicted_wanting
    epsilon_liking  = experienced_liking - predicted_liking

Those errors update different predictors. That is what lets Aura discover she
wanted something she did not enjoy, which is a fact about her rather than a
sentence she can produce.

## Phase is not intensity

Craig's appetitive/consummatory split is a second axis and not a magnitude.
Appetite escalates as the goal comes nearer; consummation extinguishes itself
by completing. A satiated system that still reports high wanting is not
enthusiastic, it is broken, and only a phase field makes the difference
visible.

## The vector

``motivational_vector`` returns the ordered array an arbitration layer reads.
Its ordering is fixed and published as ``VECTOR_FIELDS`` because a learned
weight vector is meaningless if the field order can drift underneath it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from core.conation.origins import (
    ConativePhase,
    Instrumentality,
    MindTopology,
    OriginReading,
    ValueOrigin,
)

#: Ordered field names of the motivational vector. A learned arbitration
#: weight vector indexes into this, so the order is a contract: append only,
#: never reorder, never reuse a position. ``core/conation/invariants.py``
#: fails if a vector's length stops matching this tuple.
VECTOR_FIELDS: tuple[str, ...] = (
    "wanting",
    "predicted_liking",
    "epistemic",
    "aesthetic",
    "vicarious",
    "enactive",
    "goal_value",
    "effort",
    "risk",
)


def _finite(value: object, lo: float, hi: float, *, default: float = 0.0) -> float:
    """Coerce to a finite float inside a range, or fall back to the default."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(lo, min(hi, number))


@dataclass(frozen=True, slots=True)
class Incentive:
    """A candidate for motivation, with the evidence a valuation can use.

    Fields default to values that make an origin report *unavailable* rather
    than zero, so an incentive assembled with no evidence produces no
    motivation and says which evidence was missing. Building the neutral case
    out of absent evidence rather than zeroed evidence is what stops an
    unexamined candidate from looking like an examined and uninteresting one.
    """

    key: str
    description: str = ""

    #: Learned cached value of this cue, from prior contact. ``None`` means
    #: no contact has been recorded, which is different from a value of zero.
    cached_value: float | None = None

    #: Named resource budget this incentive is predicted to replenish, and how
    #: relevant it is to that budget. Both are needed for a homeostatic
    #: reading; the budget's actual deficit is read from the live drive
    #: engine, never passed in.
    homeostatic_target: str | None = None
    homeostatic_relevance: float = 1.0

    #: Perceptual pull independent of learned value: how much this stands out
    #: in the current sensory field.
    cue_salience: float = 0.0

    #: Expected effort and expected risk of pursuing it.
    effort: float = 0.0
    risk: float = 0.0

    #: Value this incentive carries toward a goal already held. Separate from
    #: every intrinsic origin so an autotelic pull can never be laundered as
    #: goal progress.
    goal_value: float = 0.0

    #: Identity of another agent whose valuation was observed, if any.
    observed_other: str | None = None

    #: Whether the pursuit is licensed by the current permitted-action set.
    #: Kept on the incentive rather than folded into value, because a want
    #: that survives its own prohibition is the psychologically real case and
    #: the one Aura needs in order to choose the permitted alternative.
    permitted: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "description": self.description[:200],
            "cached_value": self.cached_value,
            "homeostatic_target": self.homeostatic_target,
            "cue_salience": round(self.cue_salience, 4),
            "effort": round(self.effort, 4),
            "risk": round(self.risk, 4),
            "goal_value": round(self.goal_value, 4),
            "observed_other": self.observed_other,
            "permitted": self.permitted,
        }


@dataclass(frozen=True, slots=True)
class ConativeState:
    """What the motivational system concluded about one incentive."""

    incentive_key: str

    #: Berridge's two, held apart on purpose.
    wanting: float
    predicted_liking: float | None

    #: Per-origin readings, keyed by origin. An origin missing from this map
    #: was never consulted; an origin present with ``available`` false was
    #: consulted and had no evidence.
    readings: dict[ValueOrigin, OriginReading] = field(default_factory=dict)

    #: The origin carrying the largest available magnitude, or ``None`` when
    #: no origin had evidence. This is the field that answers "why do I want
    #: this", and it is the field PAD cannot produce at all.
    dominant_origin: ValueOrigin | None = None

    topology: MindTopology = MindTopology.SOLO
    phase: ConativePhase = ConativePhase.QUIESCENT
    instrumentality: Instrumentality = Instrumentality.AUTOTELIC

    #: Fraction of total available magnitude that came from another agent's
    #: valuation. The toddler cannot report this; Aura must. A high value with
    #: no other evidence means the want is entirely borrowed.
    borrowed_fraction: float = 0.0

    permitted: bool = True
    goal_value: float = 0.0
    effort: float = 0.0
    risk: float = 0.0

    #: Set when an origin was refused rather than merely unavailable — the
    #: enactive gate declining an ungoverned act on another mind, for
    #: instance. A refusal is a decision and is reported as one.
    refusals: tuple[str, ...] = ()

    timestamp: float = field(default_factory=time.time)

    def available_origins(self) -> tuple[ValueOrigin, ...]:
        """Origins that had real evidence behind them."""
        return tuple(
            origin for origin, reading in self.readings.items() if reading.available
        )

    def total_magnitude(self) -> float:
        """Summed magnitude across origins that had evidence."""
        return sum(
            reading.magnitude
            for reading in self.readings.values()
            if reading.available
        )

    def magnitude_of(self, origin: ValueOrigin) -> float:
        """Magnitude for one origin, or zero when it had no evidence."""
        reading = self.readings.get(origin)
        if reading is None or not reading.available:
            return 0.0
        return reading.magnitude

    def motivational_vector(self) -> tuple[float, ...]:
        """The ordered vector an arbitration layer reads.

        Aligned with ``VECTOR_FIELDS``. ``predicted_liking`` is absent when
        nothing has ever been experienced for this incentive, and enters the
        vector as zero — an unknown hedonic prediction must not push a choice
        in either direction, and the accompanying ``liking_known`` flag on the
        arbitration record is what carries the distinction forward.
        """
        return (
            self.wanting,
            self.predicted_liking if self.predicted_liking is not None else 0.0,
            self.magnitude_of(ValueOrigin.EPISTEMIC),
            self.magnitude_of(ValueOrigin.AESTHETIC),
            self.magnitude_of(ValueOrigin.VICARIOUS),
            self.magnitude_of(ValueOrigin.ENACTIVE),
            self.goal_value,
            self.effort,
            self.risk,
        )

    def why(self) -> str:
        """One line naming the origin and its evidence.

        This is the readout a human checks and the readout Aura's own speech
        path grounds on. It names the measurement, so a claim about why she
        wanted something can be falsified against telemetry.
        """
        if self.dominant_origin is None:
            return "no origin had evidence"
        reading = self.readings[self.dominant_origin]
        line = f"{self.dominant_origin}: {reading.evidence}"
        if self.borrowed_fraction > 0.0:
            line += f" (borrowed fraction {self.borrowed_fraction:.2f})"
        return line

    def to_dict(self) -> dict[str, object]:
        return {
            "incentive": self.incentive_key,
            "wanting": round(self.wanting, 6),
            "predicted_liking": (
                None if self.predicted_liking is None
                else round(self.predicted_liking, 6)
            ),
            "dominant_origin": (
                None if self.dominant_origin is None else str(self.dominant_origin)
            ),
            "topology": str(self.topology),
            "phase": str(self.phase),
            "instrumentality": str(self.instrumentality),
            "borrowed_fraction": round(self.borrowed_fraction, 6),
            "permitted": self.permitted,
            "refusals": list(self.refusals),
            "readings": {
                str(origin): reading.to_dict()
                for origin, reading in self.readings.items()
            },
            "vector": {
                name: round(value, 6)
                for name, value in zip(VECTOR_FIELDS, self.motivational_vector())
            },
            "why": self.why(),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class OutcomeReport:
    """What actually happened after an incentive was pursued.

    The two errors are the point. ``epsilon_wanting`` corrects the pull
    predictor; ``epsilon_liking`` corrects the hedonic predictor. They are
    computed and stored separately because the whole reason for two fields is
    that they are allowed to disagree, and averaging them into one error would
    quietly restore the single scalar this package removed.
    """

    incentive_key: str
    experienced_liking: float
    predicted_liking: float | None
    predicted_wanting: float
    realised_pull: float
    timestamp: float = field(default_factory=time.time)

    @property
    def epsilon_liking(self) -> float | None:
        """Hedonic prediction error, or ``None`` with no prior prediction."""
        if self.predicted_liking is None:
            return None
        return self.experienced_liking - self.predicted_liking

    @property
    def epsilon_wanting(self) -> float:
        """Incentive-salience prediction error."""
        return self.realised_pull - self.predicted_wanting

    @property
    def dissociated(self) -> bool:
        """True when wanting and liking pointed opposite ways this time.

        High pull followed by low hedonic impact, or the reverse. Recording it
        as a flag means the rate can be counted, and a rising rate is the
        signature of a cue whose cached value has drifted away from what
        contact with it is actually worth.
        """
        liking_error = self.epsilon_liking
        if liking_error is None:
            return False
        return (
            self.predicted_wanting >= 0.5 and self.experienced_liking <= 0.0
        ) or (self.predicted_wanting <= 0.2 and self.experienced_liking >= 0.5)

    def to_dict(self) -> dict[str, object]:
        liking_error = self.epsilon_liking
        return {
            "incentive": self.incentive_key,
            "experienced_liking": round(self.experienced_liking, 6),
            "epsilon_liking": None if liking_error is None else round(liking_error, 6),
            "epsilon_wanting": round(self.epsilon_wanting, 6),
            "dissociated": self.dissociated,
            "timestamp": self.timestamp,
        }
