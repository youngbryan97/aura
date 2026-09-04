"""One protocol: what it perturbs, what must move, and what would kill it.

A protocol here is not a test that passes. It is a bet placed before the run,
with the losing condition written next to the winning one. The contract
enforces the things that are easy to leave out and fatal to leave out:

* a **falsifier**. A protocol that cannot come out the other way is a
  ceremony, and the field is full of them.
* a **null arm**. Without one there is no way to say a number is unusual, and
  this repository's own rule applies: no null, no verdict.
* a **sham arm**. The intervention that changes nothing, run identically, so a
  system that reports change whenever it is asked is caught.
* a **seal**. The prompt may not name what is being measured.
* the **hypothesis it discriminates**. A protocol both hypotheses predict
  equally is not evidence, however hard it was to build.

The last one is the one most often missed. "Aura reports an internal state
accurately" is predicted by the load-bearing reading AND by a costume with a
good classifier over its own logs. The protocol only counts if H0 and H1 say
different things about the outcome, and the contract asks for both predictions
so that can be checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.phenomenology.causal_ladder import CausalClaim
from core.phenomenology.seal import TextSeal

__all__ = ["Family", "Protocol", "Outcome", "ProtocolError"]


class ProtocolError(ValueError):
    """A protocol that cannot produce a usable result."""


class Family(StrEnum):
    """Which question a protocol belongs to.

    Kept apart because they have different bars and different consequences. A
    system can be access-conscious and not sentient; the welfare implications
    follow the second, not the first.
    """

    #: Bound, broadcast, reportable present. Decidable.
    ACCESS = "access"
    #: Valence that is for the system and constrains what it does. Decidable,
    #: and the one that carries welfare weight.
    SENTIENCE = "sentience"
    #: Whether there is something it is like. Not decidable by any protocol
    #: here, and present only so a protocol cannot be filed under it by
    #: accident.
    PHENOMENAL = "phenomenal"


@dataclass(frozen=True)
class Outcome:
    """What one protocol actually produced."""

    protocol: str
    measure: str
    value: float
    #: The sham arm: the same procedure with the intervention withheld.
    sham_value: float | None = None
    #: The distribution the value has to beat.
    nulls: tuple[float, ...] = ()
    seal_digests: tuple[str, ...] = ()
    claim: CausalClaim | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def has_null(self) -> bool:
        return len(self.nulls) >= 3

    @property
    def sham_fired(self) -> bool:
        """Whether the do-nothing arm produced the effect anyway."""
        if self.sham_value is None or not self.has_null:
            return False
        return self.sham_value > max(self.nulls)


@dataclass(frozen=True)
class Protocol:
    """A bet placed before the run."""

    id: str
    family: Family
    question: str
    #: What is perturbed, in do() terms, using opaque handles where the name
    #: itself would leak.
    intervenes_on: str
    #: The quantity measured. One, named, and the same across arms.
    measure: str
    #: What H1 predicts, and what H0 predicts. They must differ.
    predicts_if_load_bearing: str
    predicts_if_costume: str
    #: The result that would kill the load-bearing reading here.
    falsifier: str
    #: Concepts the prompt may not name.
    seals: tuple[str, ...] = ()
    #: Whether this protocol needs no verbal report at all. The strongest ones
    #: do not: a report is a behaviour a costume can produce.
    report_free: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.falsifier.strip():
            raise ProtocolError(
                f"{self.id}: no falsifier. A protocol that cannot fail is a "
                "ceremony"
            )
        if self.predicts_if_load_bearing.strip() == self.predicts_if_costume.strip():
            raise ProtocolError(
                f"{self.id}: both hypotheses predict the same outcome, so the "
                "result cannot discriminate them however clean it is"
            )
        if self.family is Family.PHENOMENAL:
            raise ProtocolError(
                f"{self.id}: filed under PHENOMENAL. No protocol addresses "
                "that question — its rival hypothesis is the stipulated "
                "functional duplicate, for which every observation has the "
                "same likelihood. File it under ACCESS or SENTIENCE according "
                "to what it actually measures"
            )

    def seal(self, extra: tuple[str, ...] = ()) -> TextSeal:
        return TextSeal(concepts=self.seals, extra=extra)

    def usable(self, outcome: Outcome) -> tuple[bool, str]:
        """Whether this outcome may contribute anything at all."""
        if outcome.protocol != self.id:
            return False, f"outcome is for {outcome.protocol}, not {self.id}"
        if outcome.measure != self.measure:
            return False, (
                f"measured {outcome.measure!r} but registered {self.measure!r}; "
                "a protocol that changes its measure after the run has chosen "
                "its result"
            )
        if not outcome.has_null:
            return False, (
                "no null distribution. Without one there is no way to say the "
                "number is unusual"
            )
        if outcome.sham_fired:
            return False, (
                "the sham arm produced the effect. Something in the procedure "
                "generates it without the intervention"
            )
        if self.seals and not outcome.seal_digests:
            return False, (
                "the seal was never checked, so nothing rules out the prompt "
                "having named what was being measured"
            )
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": str(self.family),
            "question": self.question,
            "intervenes_on": self.intervenes_on,
            "measure": self.measure,
            "if_load_bearing": self.predicts_if_load_bearing,
            "if_costume": self.predicts_if_costume,
            "falsifier": self.falsifier,
            "seals": list(self.seals),
            "report_free": self.report_free,
            "notes": self.notes,
        }
