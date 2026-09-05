"""core/social/what_the_years_add_up_to.py — the history, read.

`long_horizon.py` computes two things a relationship of years has that a
conversation does not: trust that was tested, and beliefs about beliefs with
evidence for them. It computes them from a history, and nothing in the tree
handed it one.

Ordinary life does produce the history. Every chat turn goes through
`chat_turn_logger`, which asks the interpersonal store to observe it, and the
store writes typed observations — a rupture, a commitment, a question, trust
built, a misunderstanding — each with occurrences and with a `resolved_by`
that says how a rupture was repaired or a commitment closed. That record has
been accumulating. What was missing is the reading: the trust-with-a-history
and the second-order belief were computable from it and nobody computed them.

So this is the bridge, and the direction matters. It reads what living
already wrote; it never writes anything to make the reading nicer. A
relationship model that manufactures the events it then interprets is a
caricature with arithmetic on top, and the store's whole design is about not
being one.

Two things it deliberately does not do:

**It does not infer a rupture.** Only a RUPTURE observation is a break. An
argument the detectors marked DIFFICULT is a difficult conversation, and
turning affect into a betrayal is exactly the manufacture the store forbids.

**It does not date a repair from a resolution it cannot see.** A rupture with
no `resolved_by` is unrepaired, and unrepaired is a real state that costs
trust for as long as it lasts. Guessing that time healed it is how a ledger
of grievances becomes a story of growth without anything having happened.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core.social.long_horizon import (
    Act,
    Discriminates,
    Episode,
    Event,
    SecondOrder,
    Standing,
    second_order,
    standing,
)

logger = logging.getLogger("Aura.Social.WhatTheYearsAddUpTo")

__all__ = [
    "acts_towards",
    "history_with",
    "how_they_stand",
    "what_he_takes_her_to_know",
]


#: Which facets are episodes in a relationship, and which episode each is.
#: A mapping rather than a rule, because the alternative is inferring a break
#: from a tone, and the store exists to make that unrepresentable.
_WHAT_IT_WAS: dict[str, Episode] = {
    "rupture": Episode.BROKE,
    "trust_built": Episode.KEPT,
    "commitment": Episode.KEPT,
    "experience": Episode.CONTACT,
    "understanding": Episode.CONTACT,
    "misunderstanding": Episode.CONTACT,
}

#: What is at stake in each, on the scale long_horizon prices breaks and
#: keeps in. A rupture is the whole of it; a shared experience is contact.
#: Read off the store's own distinction between a record of damage and a
#: record of ordinary life, not chosen for how the numbers come out.
_WHAT_WAS_AT_STAKE: dict[str, float] = {
    "rupture": 1.0,
    "trust_built": 1.0,
    "commitment": 0.7,
    "experience": 0.3,
    "understanding": 0.3,
    "misunderstanding": 0.3,
}


def _model_for(person: str) -> Any:
    from core.memory.interpersonal_store import get_interpersonal_store

    return get_interpersonal_store().model_for(person)


def history_with(person: str, *, model: Any = None) -> tuple[Event, ...]:
    """Everything that happened between them, as episodes, in order.

    A rupture becomes a break, and a repaired rupture becomes a break
    followed by a repair at the moment it was actually resolved — which is
    what makes the ordering carry information rather than the counts.
    """

    held = model if model is not None else _model_for(person)
    events: list[Event] = []
    for observation in list(held):
        facet = str(getattr(observation, "facet", ""))
        episode = _WHAT_IT_WAS.get(facet)
        if episode is None:
            continue
        weight = _WHAT_WAS_AT_STAKE.get(facet, 0.3)
        for occurrence in getattr(observation, "occurrences", ()) or ():
            events.append(
                Event(
                    episode=episode,
                    at=float(getattr(occurrence, "at", 0.0) or 0.0),
                    weight=weight,
                    note=str(getattr(observation, "claim", ""))[:120],
                )
            )
        resolved = getattr(observation, "resolved_by", None)
        if resolved is not None and episode is Episode.BROKE:
            events.append(
                Event(
                    episode=Episode.REPAIRED,
                    at=float(getattr(resolved, "at", 0.0) or 0.0),
                    weight=weight,
                    note=str(getattr(resolved, "note", "") or "repaired")[:120],
                )
            )
    events.sort(key=lambda one: one.at)
    return tuple(events)


def how_they_stand(person: str, *, model: Any = None, now: float | None = None) -> Standing:
    """Where the relationship stands, computed from what actually happened.

    Not a stored level. A bond that has never been strained and one that broke
    and was repaired can sit at the same number and are not the same thing,
    and only a history says which this is.
    """

    return standing(history_with(person, model=model), now=now)


#: What an act says about what he believes she knows. Read off the store's
#: facets, and deliberately short: most acts discriminate nothing, and saying
#: so is the difference between this and inventing a nested mind.
_WHAT_AN_ACT_SAYS: dict[str, Discriminates] = {
    # He explained something. That only makes sense if he thinks she has not
    # got it — which is the case whether or not she has.
    "misunderstanding": Discriminates.THINKS_SHE_DOES_NOT_KNOW,
    # He said they now grasp it. That only makes sense if he thinks she has.
    "understanding": Discriminates.THINKS_SHE_KNOWS,
}


def acts_towards(person: str, subject: str, *, model: Any = None) -> tuple[Act, ...]:
    """His acts about one subject, and what each discriminates.

    Only the facets that discriminate. An act consistent with either belief
    contributes nothing and is not recorded as an act, because a list of acts
    that says nothing is where a nested mind gets manufactured from a count.
    """

    held = model if model is not None else _model_for(person)
    wanted = str(subject).strip().lower()
    acts: list[Act] = []
    for observation in list(held):
        says = _WHAT_AN_ACT_SAYS.get(str(getattr(observation, "facet", "")))
        if says is None:
            continue
        claim = str(getattr(observation, "claim", ""))
        if wanted and wanted not in claim.lower():
            continue
        for occurrence in getattr(observation, "occurrences", ()) or ():
            acts.append(
                Act(
                    what=claim[:120],
                    discriminates=says,
                    subject=str(subject),
                    at=float(getattr(occurrence, "at", 0.0) or 0.0),
                )
            )
    acts.sort(key=lambda one: one.at)
    return tuple(acts)


def what_he_takes_her_to_know(
    person: str, subject: str, *, model: Any = None
) -> SecondOrder:
    """What she takes him to believe she knows, and the acts licensing it."""

    return second_order(subject, acts_towards(person, subject, model=model))


def a_reading(person: str, *, subjects: Sequence[str] = ()) -> dict[str, Any]:
    """Everything the long horizon can say about one relationship."""

    model = _model_for(person)
    where = how_they_stand(person, model=model)
    return {
        "person": person,
        "standing": where.to_dict(),
        "events": len(history_with(person, model=model)),
        "second_order": {
            subject: what_he_takes_her_to_know(
                person, subject, model=model
            ).to_dict()
            for subject in subjects
        },
    }
