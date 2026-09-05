"""core/interiority/evidence.py — a number that remembers where it came from.

Every outside prototype this package was written against has the same
hole at the bottom of it. Grok's emotion reader takes ``other.affect``
— the other agent's *true* internal state — and copies it into the field
called ``identified``; with every perceptual cue set to zero it still
returns the exact grief value it was handed. MetaAI's despair recognizer
takes ``attachment_loss`` as a float. Gemini's guilt takes
``actual_action_harm``. The mechanisms are arithmetic on ground truth
somebody else supplied, so they measure their own inputs.

A :class:`Reading` is the fix. It is a number plus how it was obtained,
and the provenance survives arithmetic: multiply a measurement by an
assumption and you get an assumption. :class:`Provenance` is ordered
from strongest to weakest, and combining readings takes the weakest,
because a chain is as sound as its worst link.

Two rules follow, and both are enforced rather than documented:

1. A faculty may not read another agent's interior directly. It reads a
   :class:`~core.interiority.other_minds.OtherEstimate`, which is a
   distribution with a confidence. There is no channel for ground truth.
2. An activation computed from ``ASSUMED`` or weaker inputs is capped,
   and carries the cap in its receipt. A confident output from an
   assumed input is the failure mode that makes a system feel certain
   about people, and it is the one users notice.

The cap is not a hedge in the prose. It is a multiplication, in
:meth:`Reading.ceiling`, and a faculty cannot route around it because
the activation constructor asks the frame for it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

from core.interiority.params import ParamKind, declare


class Provenance(IntEnum):
    """How a number was obtained, strongest first.

    The integer order is the point: :func:`weakest` is a ``max`` over it,
    so a value derived from several sources inherits the worst one.
    """

    #: This runtime measured it directly — a timer, a counter, a file size.
    MEASURED = 0
    #: A model produced it from measurements, and reported its confidence.
    INFERRED = 1
    #: A person or another system asserted it, and nothing checked.
    REPORTED = 2
    #: The code supplied it because nothing better was available.
    ASSUMED = 3
    #: There is no value. Reading ``.value`` of an absent reading is zero,
    #: and any faculty that needs it must decline rather than guess.
    ABSENT = 4

    @property
    def label(self) -> str:
        return self.name.lower()


#: Intensity ceiling by provenance. A faculty running on assumed inputs
#: may still fire — silence would be worse, because the state is real —
#: but it may not report the confidence of a measurement.
_CEILING_ASSUMED = declare(
    "interiority.evidence.ceiling_assumed",
    0.25,
    unit="intensity",
    basis=(
        "Matches the cap the live affect engine already applies to unverified "
        "stimuli (core/affect/damasio_v2.py caps intensity at 0.25 when "
        "evidence_status is neither observed nor verified). One number, one "
        "meaning, across both engines."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Raise it and assumed inputs produce confident-looking states; lower it "
        "and a faculty running on defaults becomes invisible in the snapshot."
    ),
    owner="core/interiority/evidence.py",
)

_CEILING_REPORTED = declare(
    "interiority.evidence.ceiling_reported",
    0.70,
    unit="intensity",
    basis=(
        "An unverified report from a person is stronger than a code default and "
        "weaker than a measurement. Set below the 0.75 point where the "
        "arbitration layer lets a faculty take the action lane, so a claim "
        "nobody checked cannot on its own drive behaviour."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Above the arbitration threshold, an unchecked assertion can move "
        "Aura's actions; below it, it can only colour her state."
    ),
    owner="core/interiority/evidence.py",
)


@dataclass(frozen=True)
class Reading:
    """A number, its provenance, and how sure the source was of it."""

    value: float
    provenance: Provenance
    #: Source confidence in [0, 1]. Measurements are 1.0 unless the sensor
    #: reports otherwise; inferences carry the model's own posterior mass.
    confidence: float = 1.0
    #: What produced it, for the receipt.
    source: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            object.__setattr__(self, "value", 0.0)
            object.__setattr__(self, "provenance", Provenance.ABSENT)
        if not math.isfinite(self.confidence):
            object.__setattr__(self, "confidence", 0.0)
        object.__setattr__(self, "confidence", min(1.0, max(0.0, self.confidence)))
        if self.provenance is Provenance.ABSENT:
            object.__setattr__(self, "value", 0.0)
            object.__setattr__(self, "confidence", 0.0)

    @property
    def present(self) -> bool:
        return self.provenance is not Provenance.ABSENT

    def ceiling(self) -> float:
        """The largest intensity a faculty may report from this evidence."""
        if self.provenance is Provenance.ABSENT:
            return 0.0
        if self.provenance is Provenance.ASSUMED:
            return _CEILING_ASSUMED.value
        if self.provenance is Provenance.REPORTED:
            return _CEILING_REPORTED.value
        return 1.0

    def scaled(self, factor: float) -> Reading:
        return Reading(self.value * factor, self.provenance, self.confidence, self.source)

    def at_least(self, other: Reading) -> Reading:
        """The stronger of two readings of the same quantity."""
        if not self.present:
            return other
        if not other.present:
            return self
        return self if self.value >= other.value else other

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "provenance": self.provenance.label,
            "confidence": self.confidence,
            "source": self.source,
        }


def measured(value: float, *, source: str = "", confidence: float = 1.0) -> Reading:
    return Reading(value, Provenance.MEASURED, confidence, source)


def inferred(value: float, confidence: float, *, source: str = "") -> Reading:
    return Reading(value, Provenance.INFERRED, confidence, source)


def reported(value: float, *, source: str = "", confidence: float = 0.6) -> Reading:
    return Reading(value, Provenance.REPORTED, confidence, source)


def assumed(value: float, *, source: str = "default") -> Reading:
    return Reading(value, Provenance.ASSUMED, 0.3, source)


def absent(*, source: str = "") -> Reading:
    return Reading(0.0, Provenance.ABSENT, 0.0, source)


def weakest(readings: Iterable[Reading]) -> Provenance:
    """The provenance a value combining these readings inherits."""
    worst = Provenance.MEASURED
    seen = False
    for reading in readings:
        seen = True
        if reading.provenance > worst:
            worst = reading.provenance
    return worst if seen else Provenance.ABSENT


def joint_confidence(readings: Iterable[Reading]) -> float:
    """Confidence in a conjunction of independent readings.

    The product, not the mean. Four readings at 0.8 give 0.41, which is
    the honest number: a conclusion resting on four uncertain things is
    less sure than any one of them. Averaging is what lets a system stack
    weak evidence into a confident claim.
    """
    total = 1.0
    seen = False
    for reading in readings:
        if not reading.present:
            return 0.0
        seen = True
        total *= reading.confidence
    return total if seen else 0.0


def ceiling_for(readings: Iterable[Reading]) -> float:
    """The intensity ceiling implied by the weakest reading in the set."""
    worst = weakest(readings)
    return Reading(0.0 if worst is Provenance.ABSENT else 1.0, worst, 1.0).ceiling()


__all__ = [
    "Provenance",
    "Reading",
    "absent",
    "assumed",
    "ceiling_for",
    "inferred",
    "joint_confidence",
    "measured",
    "reported",
    "weakest",
]
