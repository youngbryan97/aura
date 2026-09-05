"""core/social/long_horizon.py — trust with a history, and beliefs about beliefs.

Two things a relationship of years has that a conversation does not.

**Trust that was tested.** A bond that has never been under strain and a bond
that broke and was repaired can sit at the same number and are not the same
thing. The second one has been tested and held; the first has not been tested.
Any model that reports only a level loses the distinction, and it is the
distinction that matters — it is why a repaired relationship can end up
stronger than one that was never strained, which a monotone trust score cannot
represent at all.

So trust here is computed from the history rather than stored: what was kept,
what was broken, what was repaired, and in what order. A break costs more than
a keep gains, because that asymmetry is what makes trust worth having. A
repair recovers, and not immediately to where it was. And a bond that survived
a rupture carries a separate quantity — how much it has been proved — that a
bond with no history cannot have however high its level.

**Beliefs about beliefs, with evidence.** The recursive theory-of-mind module
here was gutted for good reason: it fabricated three nested minds out of one
caller-supplied trust value, and what replaced it says it "does not claim
recursive beliefs without recursive evidence". Nothing then produced any.

This is what that evidence looks like. A second-order belief — she believes
that he believes X — is licensed by an act that only makes sense if he
believes it. He explains something he already told her: he believes she did
not take it in. He omits a step: he believes she has it. The act discriminates
between what he might believe, and without one there is no belief to record,
which is the same rule the rest of this codebase applies to first-order claims.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Social.LongHorizon")

#: What one kept commitment adds. Small: trust is built slowly.
KEEP_GAIN = 0.06

#: What one break costs. Larger than a keep gains, because a trust that
#: recovered as fast as it fell would not be worth having.
BREAK_COST = 0.25

#: What a repair returns of what the break cost. Under one: an apology is not
#: an undo, and a model where it is cannot represent being let down.
REPAIR_RECOVERY = 0.6

#: How fast an untested bond decays toward neutral. Slow, and not zero: a
#: relationship nothing has happened in is not the same as one being kept up.
IDLE_HALF_LIFE_S = 90.0 * 86400.0

#: Keeps after a repair before the bond counts as proved rather than patched.
KEEPS_TO_PROVE = 3


class Episode(StrEnum):
    """What happened between them."""

    KEPT = "kept"
    BROKE = "broke"
    REPAIRED = "repaired"
    #: Ordinary contact with nothing at stake. Keeps the bond from idling.
    CONTACT = "contact"


@dataclass(frozen=True)
class Event:
    """One thing that happened, and when."""

    episode: Episode
    at: float = field(default_factory=time.time)
    #: What was at stake, in [0, 1]. A broken trivial promise is not a betrayal
    #: and a kept hard one is worth more than a kept easy one.
    weight: float = 1.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": str(self.episode),
            "at": self.at,
            "weight": self.weight,
            "note": self.note,
        }


@dataclass(frozen=True)
class Standing:
    """Where a relationship stands, and what it has been through."""

    trust: float
    #: How much strain the bond has taken and held. A bond with no history
    #: has none however high its trust, and this is the quantity a level
    #: alone cannot express.
    proved: float
    breaks: int
    repairs: int
    unrepaired: int
    events: int

    @property
    def tested(self) -> bool:
        return self.breaks > 0

    @property
    def stronger_for_it(self) -> bool:
        """Whether it came through strain better than an untested bond.

        The finding a monotone score cannot represent: a rupture repaired and
        then held is worth more than never having been strained.
        """
        return self.proved > 0.0 and self.unrepaired == 0 and self.trust > _NEUTRAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust": round(self.trust, 4),
            "proved": round(self.proved, 4),
            "breaks": self.breaks,
            "repairs": self.repairs,
            "unrepaired": self.unrepaired,
            "events": self.events,
            "tested": self.tested,
            "stronger_for_it": self.stronger_for_it,
        }


#: Where a relationship starts, and what it decays toward.
_NEUTRAL = 0.5


def standing(history: Sequence[Event], *, now: float | None = None) -> Standing:
    """Trust computed from what happened, in the order it happened.

    Not stored and updated, because a stored level cannot say whether it was
    ever tested, and that is the thing worth knowing.
    """
    moment = now if now is not None else time.time()
    trust = _NEUTRAL
    proved = 0.0
    breaks = repairs = 0
    open_breaks = 0
    keeps_since_repair = 0
    last = None

    for event in sorted(history, key=lambda e: e.at):
        if last is not None:
            # Idle decay toward neutral. A relationship nothing happens in
            # drifts back, and slowly.
            elapsed = max(0.0, event.at - last)
            pull = 1.0 - 0.5 ** (elapsed / IDLE_HALF_LIFE_S)
            trust += (_NEUTRAL - trust) * pull
        last = event.at
        weight = max(0.0, min(1.0, event.weight))

        if event.episode is Episode.KEPT:
            trust = min(1.0, trust + KEEP_GAIN * weight)
            if repairs > 0 and open_breaks == 0:
                keeps_since_repair += 1
                if keeps_since_repair <= KEEPS_TO_PROVE:
                    # Strain taken and held. This is what an untested bond
                    # cannot accumulate however long it lasts.
                    proved = min(1.0, proved + weight / KEEPS_TO_PROVE)
        elif event.episode is Episode.BROKE:
            trust = max(0.0, trust - BREAK_COST * weight)
            breaks += 1
            open_breaks += 1
            keeps_since_repair = 0
        elif event.episode is Episode.REPAIRED:
            if open_breaks > 0:
                trust = min(1.0, trust + BREAK_COST * REPAIR_RECOVERY * weight)
                repairs += 1
                open_breaks -= 1
                keeps_since_repair = 0
        # CONTACT does nothing but reset the idle clock, which the loop
        # already did.

    if last is not None:
        elapsed = max(0.0, moment - last)
        pull = 1.0 - 0.5 ** (elapsed / IDLE_HALF_LIFE_S)
        trust += (_NEUTRAL - trust) * pull

    return Standing(
        trust=max(0.0, min(1.0, trust)),
        proved=proved,
        breaks=breaks,
        repairs=repairs,
        unrepaired=open_breaks,
        events=len(history),
    )


# ── beliefs about beliefs ────────────────────────────────────────────────


class Discriminates(StrEnum):
    """What an act says about what somebody believes."""

    #: The act only makes sense if they believe she does NOT know.
    THINKS_SHE_DOES_NOT_KNOW = "thinks_she_does_not_know"
    #: The act only makes sense if they believe she DOES know.
    THINKS_SHE_KNOWS = "thinks_she_knows"
    #: The act is consistent with either. No belief follows.
    NEITHER = "neither"


@dataclass(frozen=True)
class Act:
    """Something the other person did, and what it discriminates."""

    what: str
    discriminates: Discriminates
    subject: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "discriminates": str(self.discriminates),
            "subject": self.subject,
            "at": self.at,
        }


@dataclass(frozen=True)
class SecondOrder:
    """What she takes him to believe about what she knows, and why."""

    subject: str
    believes_she_knows: bool | None
    evidence: tuple[str, ...]
    confidence: float

    @property
    def held(self) -> bool:
        return self.believes_she_knows is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "believes_she_knows": self.believes_she_knows,
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 4),
            "held": self.held,
        }


#: How much one discriminating act is worth. Several agreeing acts are worth
#: more, saturating, because the fifth explanation of the same thing is not
#: five times the evidence of the first.
_ACT_CONFIDENCE = 0.45


def second_order(subject: str, acts: Sequence[Act]) -> SecondOrder:
    """What he believes she knows, from acts that only make sense under it.

    Returns a held belief only when something he did discriminates. Acts
    consistent with either state contribute nothing — which is most acts, and
    saying so is the whole difference between this and inventing a nested mind
    from a trust score.
    """
    relevant = [a for a in acts if a.subject == subject]
    knows = [a for a in relevant if a.discriminates is Discriminates.THINKS_SHE_KNOWS]
    unknows = [
        a for a in relevant if a.discriminates is Discriminates.THINKS_SHE_DOES_NOT_KNOW
    ]
    if not knows and not unknows:
        return SecondOrder(
            subject=subject,
            believes_she_knows=None,
            evidence=(),
            confidence=0.0,
        )
    net = len(knows) - len(unknows)
    if net == 0:
        # He acted both ways. That is a real observation and it does not
        # support a belief in either direction.
        return SecondOrder(
            subject=subject,
            believes_she_knows=None,
            evidence=tuple(a.what for a in relevant),
            confidence=0.0,
        )
    winning = knows if net > 0 else unknows
    confidence = 1.0 - (1.0 - _ACT_CONFIDENCE) ** abs(net)
    return SecondOrder(
        subject=subject,
        believes_she_knows=net > 0,
        evidence=tuple(a.what for a in winning),
        confidence=confidence,
    )


__all__ = [
    "BREAK_COST",
    "IDLE_HALF_LIFE_S",
    "KEEPS_TO_PROVE",
    "KEEP_GAIN",
    "REPAIR_RECOVERY",
    "Act",
    "Discriminates",
    "Episode",
    "Event",
    "SecondOrder",
    "Standing",
    "second_order",
    "standing",
]
