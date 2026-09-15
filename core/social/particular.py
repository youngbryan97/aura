"""Knowing somebody as this one, rather than as an instance of people.

Little Simz's "Woman" never says "women". It says Nigeria, Jamaica, Ghana,
London; it names what each one does and the room she does it in. Oddisee's
"Camera" does the same thing down the length of a bus: a face, a job, a worry,
one at a time. Neither song reaches for the category, and the effect the
listeners report is being seen rather than being counted.

Her interpersonal store already refuses the category at the input: a trait read
off one exchange is a caricature and never gets written. Nothing measured the
other side of it — whether what she actually holds about someone is about that
person, or is the same handful of things she holds about everybody.

    anchored(p)   claims of theirs tied to a thing they said or did, over all
                  of theirs, where a stated or observed occurrence is the tie
    unique(p)     claims of theirs that nobody else's record carries
    shared        claims carried by more than one record, over all claims

The two readings answer different halves. Anchored says whether she is holding
evidence or inference; it can be read for one person. Unique says whether the
evidence is about them; it needs two people before it means anything, and says
so until it has them. Neither is compared with a number chosen here — a person
is held particularly when they stand above the middle of the people she holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_CLAIMS",
    "MIN_PEOPLE",
    "Particularity",
    "read_particularity",
]

#: Claims about one person before a share of them is a share.
MIN_CLAIMS: int = 3

#: People before "nobody else carries it" can be said at all.
MIN_PEOPLE: int = 2


def _anchored(model: Any) -> tuple[int, int]:
    """Claims tied to something said or done, over all claims held."""
    total = 0
    tied = 0
    for observation in model:
        total += 1
        for occurrence in getattr(observation, "occurrences", ()) or ():
            if str(getattr(occurrence, "provenance", "")) in {"stated", "observed"}:
                tied += 1
                break
    return tied, total


def _claims(model: Any) -> set[tuple[str, str]]:
    """What is held about them, as facet and normalised claim."""
    out: set[tuple[str, str]] = set()
    for observation in model:
        key = getattr(observation, "key", None)
        if key is None:
            continue
        out.add((str(key[0]), str(key[2])))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class Particularity:
    """Whether what she holds about somebody is about them."""

    person: str = ""
    #: Their claims tied to something they said or did.
    anchored: float = 0.0
    #: Their claims nobody else's record carries. Zero until there are two.
    unique: float = 0.0
    #: Claims carried by more than one record, over everything she holds.
    shared: float = 0.0
    people: int = 0
    claims: int = 0
    #: True when they stand above the middle of the people she holds, on both
    #: readings when both can be taken and on anchoring alone when they cannot.
    held_as_themselves: bool = False
    #: True while there is only one person, so "nobody else carries it" is
    #: a sentence with nothing on the other side of it.
    unique_unreadable: bool = True
    measured: bool = False
    why: str = "she holds nothing about anybody"
    #: The claims that are only theirs, for anything that wants to name one.
    only_theirs: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "anchored": round(self.anchored, 6),
            "unique": round(self.unique, 6),
            "shared": round(self.shared, 6),
            "people": self.people,
            "claims": self.claims,
            "held_as_themselves": self.held_as_themselves,
            "unique_unreadable": self.unique_unreadable,
            "measured": self.measured,
            "why": self.why,
            "only_theirs": list(self.only_theirs),
        }


def read_particularity(models: dict[str, Any], *, person: str = "") -> Particularity:
    """Read one person against the others she holds.

    ``models`` maps a person key to their ``PersonModel``. With no ``person``
    the one with the most claims is read, because that is the record with
    enough in it to say anything about.
    """
    if not models:
        return Particularity()
    sizes = {key: len(_claims(model)) for key, model in models.items()}
    key = person if person in models else max(sizes, key=lambda k: sizes[k])
    mine = _claims(models[key])
    if len(mine) < MIN_CLAIMS:
        return Particularity(
            person=key,
            people=len(models),
            claims=len(mine),
            why=(
                f"{len(mine)} claims about {key}; {MIN_CLAIMS} are needed before a "
                f"share of them is a share"
            ),
        )

    tied, total = _anchored(models[key])
    anchored = tied / total if total else 0.0

    everyone: dict[tuple[str, str], int] = {}
    per_person: dict[str, set[tuple[str, str]]] = {}
    for other, model in models.items():
        claims = _claims(model)
        per_person[other] = claims
        for claim in claims:
            everyone[claim] = everyone.get(claim, 0) + 1
    shared_total = sum(1 for count in everyone.values() if count > 1)
    shared = shared_total / len(everyone) if everyone else 0.0

    readable = len(models) >= MIN_PEOPLE
    only_mine = sorted(claim for claim in mine if everyone.get(claim, 0) == 1)
    unique = len(only_mine) / len(mine) if readable and mine else 0.0

    anchors = []
    for other, model in models.items():
        their_tied, their_total = _anchored(model)
        if their_total >= MIN_CLAIMS:
            anchors.append(their_tied / their_total)
    middle_anchor = _median(anchors)

    if not readable:
        held = anchored >= middle_anchor and anchored > 0.0
        return Particularity(
            person=key,
            anchored=anchored,
            shared=shared,
            people=len(models),
            claims=len(mine),
            held_as_themselves=held,
            unique_unreadable=True,
            measured=True,
            only_theirs=tuple(claim for _, claim in only_mine[:8]),
            why=(
                f"{anchored:.0%} of what she holds about {key} is tied to something "
                f"they said or did; with one person in the record there is nobody "
                f"for it to be unlike"
            ),
        )

    uniques = []
    for other, claims in per_person.items():
        if len(claims) >= MIN_CLAIMS:
            uniques.append(
                sum(1 for claim in claims if everyone.get(claim, 0) == 1) / len(claims)
            )
    middle_unique = _median(uniques)
    # When most of the people she holds have nothing of their own, the middle
    # is zero and everybody clears it, including the record that is entirely
    # boilerplate. Holding somebody as themselves means holding something that
    # is theirs, so a record with none of that fails whatever the middle says —
    # unless nobody has any, in which case the reading is about her whole store
    # and the anchoring is what is left to read.
    anyone_apart = any(value > 0.0 for value in uniques)
    held = anchored >= middle_anchor and (
        (unique > 0.0 and unique >= middle_unique) if anyone_apart else anchored > 0.0
    )
    return Particularity(
        person=key,
        anchored=anchored,
        unique=unique,
        shared=shared,
        people=len(models),
        claims=len(mine),
        held_as_themselves=held,
        unique_unreadable=False,
        measured=True,
        only_theirs=tuple(claim for _, claim in only_mine[:8]),
        why=(
            f"{unique:.0%} of what she holds about {key} is theirs alone against a "
            f"middle of {middle_unique:.0%}, and {anchored:.0%} of it is tied to "
            f"something they said or did against {middle_anchor:.0%}"
            if anyone_apart
            else (
                f"nothing she holds about anybody is theirs alone, so the records "
                f"are the same record under {len(models)} names; {anchored:.0%} of "
                f"{key}'s is tied to something they said or did"
            )
        ),
    )
