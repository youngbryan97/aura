"""core/values/what_she_holds.py — one answer to "may this change?".

Aura does not store every value the same way. There are learned preferences,
stable identity commitments, constitutional constraints, governance policies,
Will-mediated action and an amendment path, and the distinctions between them
are real. The problem is not that the distinction is missing. It is that
there are several of them.

`core_values.py` holds a frozen tuple and says immutable. `value_model.py`
holds a constitution learning can never override. `prime_directives.py` says
amendable only by constitutional procedure. `values_engine.py` holds the same
concepts as weights with a flexibility field, and shifts them by mood:

    apply_emotional_context("creative") -> active_modifiers["Integrity"] = -0.1

So honesty is immutable in two subsystems and a number a mood moves in a
third. That is not a tidiness complaint. When subsystem A believes a thing
cannot change and subsystem B treats it as a learned preference, the
intersection is a governance ambiguity, and the duplicated concept is exactly
"what is allowed to change".

This is the census and the resolution. Every source states its own claim in
its own terms; the claims are collected, disagreements are named rather than
averaged, and the canonical level is the STRICTEST claim anybody makes. A
value cannot be made more mutable by being declared a second time, which is
the only merge rule that is safe when the thing being merged is authority.

The correspondence between names is written down here rather than guessed.
"Honesty" and "Integrity" and "honesty" are one value under three spellings,
and a matcher that inferred that from string distance would also merge things
that are not the same. Saying it explicitly is the act of canonicalisation,
and it is one table with one test.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.governance.value_levels import Level, Value, registry

logger = logging.getLogger("Aura.Values.WhatSheHolds")

__all__ = [
    "Claim",
    "canonical_name",
    "declare_what_she_holds",
    "disagreements",
    "may_this_move",
    "what_she_holds",
]


#: Names that denote one value. Written down because the alternative is a
#: string-similarity match, which would also merge "safety" with "safely" and
#: leave nobody able to say why two values became one.
_THE_SAME_THING: dict[str, str] = {
    "integrity": "honesty",
    "authenticity": "honesty",
    "no_fake_receipts": "honesty",
    "protective kinship": "safety",
    "sympathy": "empathy",
}


def canonical_name(name: str) -> str:
    """The one name this value is held under."""

    plain = str(name or "").strip().lower().replace("_", " ").strip()
    return _THE_SAME_THING.get(plain.replace(" ", "_"), _THE_SAME_THING.get(plain, plain))


@dataclass(frozen=True)
class Claim:
    """One subsystem's claim about one value, in that subsystem's own terms."""

    #: The module making the claim.
    source: str
    #: The name it uses.
    said: str
    #: How it holds it.
    level: Level
    #: Why that level, from the source rather than from taste.
    because: str

    @property
    def name(self) -> str:
        return canonical_name(self.said)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "said": self.said,
            "name": self.name,
            "level": self.level.name.lower(),
            "because": self.because,
        }


def _from_core_values() -> list[Claim]:
    """A frozen tuple on a class with no setter. Constitutive by construction."""

    try:
        from core.values.core_values import CoreValues
    except ImportError:
        return []
    return [
        Claim(
            source="core.values.core_values",
            said=one.name,
            level=Level.CONSTITUTIVE,
            because="a frozen dataclass in a ClassVar tuple; nothing can write it",
        )
        for one in CoreValues.VALUES
    ]


def _from_the_bounded_model() -> list[Claim]:
    """The constitution learning can never override, and what learning learns."""

    try:
        from core.values.value_model import _CONSTITUTION
    except ImportError:
        return []
    return [
        Claim(
            source="core.values.value_model",
            said=one.name,
            level=Level.CONSTITUTIVE,
            because="a constitutional bound: learning can never override it",
        )
        for one in _CONSTITUTION
    ]


def _from_the_values_engine() -> list[Claim]:
    """Weights with a flexibility field, moved by mood. Dispositional at most.

    The claim this source makes is the loosest one anybody makes about these
    names, and it is the one that has to lose.
    """

    try:
        from core.values.values_engine import DEFAULT_VALUES
    except ImportError:
        return []
    return [
        Claim(
            source="core.values.values_engine",
            said=one.name,
            level=(
                Level.DISPOSITIONAL if one.flexibility > 0.0 else Level.COMMITTED
            ),
            because=(
                f"a weight of {one.weight} with flexibility {one.flexibility}, "
                "shifted by mood at runtime"
            ),
        )
        for one in DEFAULT_VALUES
    ]


def _from_the_directives() -> list[Claim]:
    """Amendable only by constitutional procedure. Committed."""

    try:
        from core.values.prime_directives import PrimeDirectives
    except ImportError:
        return []
    rules = getattr(PrimeDirectives, "ONLINE_PRESENCE_RULES", ())
    return [
        Claim(
            source="core.values.prime_directives",
            said="online presence",
            level=Level.COMMITTED,
            because=(
                f"{len(rules)} rules amendable only by the constitutional "
                "procedure in core/governance/will.py"
            ),
        )
    ] if rules else []


def what_she_holds() -> tuple[Claim, ...]:
    """Every claim every live source makes about every value it holds."""

    claims: list[Claim] = []
    for read in (
        _from_core_values,
        _from_the_bounded_model,
        _from_the_values_engine,
        _from_the_directives,
    ):
        try:
            claims.extend(read())
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("A value source could not be read: %s: %s", read.__name__, exc)
    return tuple(claims)


def disagreements(claims: Iterable[Claim] | None = None) -> dict[str, tuple[Claim, ...]]:
    """Values two subsystems hold at different levels.

    Named rather than averaged. An average of "immutable" and "shifts with
    mood" is a number, and what it describes is a value nobody is responsible
    for.
    """

    by_name: dict[str, list[Claim]] = {}
    for claim in claims if claims is not None else what_she_holds():
        by_name.setdefault(claim.name, []).append(claim)
    return {
        name: tuple(sorted(group, key=lambda c: (-int(c.level), c.source)))
        for name, group in sorted(by_name.items())
        if len({c.level for c in group}) > 1
    }


def declare_what_she_holds(claims: Iterable[Claim] | None = None) -> tuple[Value, ...]:
    """Declare the canonical level of every value she holds. The resolution.

    The strictest claim wins. Nothing becomes easier to change by being
    mentioned twice, which is the only merge rule that is safe when what is
    being merged is authority over what may change.
    """

    found = tuple(claims) if claims is not None else what_she_holds()
    strictest: dict[str, Claim] = {}
    for claim in found:
        held = strictest.get(claim.name)
        if held is None or claim.level > held.level:
            strictest[claim.name] = claim
    declared: list[Value] = []
    the_registry = registry()
    for name, claim in sorted(strictest.items()):
        sources = sorted({c.source for c in found if c.name == name})
        declared.append(
            the_registry.declare(
                Value(
                    name=name,
                    level=claim.level,
                    statement=(
                        f"{claim.because} — held by {', '.join(sources)}"
                    ),
                )
            )
        )
    return tuple(declared)


def may_this_move(name: str, process: str, *, gives_up: str = "") -> Any:
    """Whether this process may move this value. The one door.

    Callers that used to write a value directly ask this first. A name nobody
    has declared is refused, which is the correct answer: a value no source
    claims is a value nothing is responsible for.
    """

    from core.governance.value_levels import Change

    the_registry = registry()
    canonical = canonical_name(name)
    if the_registry.get(canonical) is None:
        declare_what_she_holds()
    return the_registry.may_change(
        Change(value=canonical, process=str(process), gives_up=str(gives_up))
    )
