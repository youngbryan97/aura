"""Every part of her that could be different, and what each one is worth.

`what_rests_on_what` says which parts hold up which. This says what the parts
ARE, as one list with one kind of address, so a question about herself does not
have to know which registry an answer lives in.

Why that matters more than it sounds
------------------------------------
"Which part of me is the bottleneck" has an answer that does not need a
taxonomy. Do not ask a classifier whether this is a representation problem or a
search problem. Ask, for each part, what would change if that part were
different — and the part with the largest answer is the bottleneck. The label
comes afterwards and only for the reader: if the winning change was to a word,
we call it a representation problem; if to the order, a search problem. She
never needed the words to choose.

`what_a_part_is_worth` computes that by lesion, which is the one estimate here
that is causal rather than correlational. Take the part out, measure the
families she has met, put it back. What the number says is what the part is
doing, not what it is near.

The probe is held out on purpose. A part judged on the family it was invented
for looks indispensable every time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "APart",
    "what_a_part_is_worth",
    "what_she_is_made_of",
    "where_the_time_goes",
    "the_most_they_have_in_common",
]

logger = logging.getLogger("Aura.WhatSheIsMadeOf")


@dataclass(frozen=True, slots=True)
class APart:
    """One thing about her that could be different."""

    #: `kind/name`, and the kind says which registry holds it.
    at: str
    kind: str
    name: str
    #: The term, where the part is one. Nothing for parts that are not terms.
    term: Any = None
    #: How many times the record says it has been used.
    used: int = 0
    #: Episodes since it was last used, or nothing where it never was.
    idle: int | None = None
    #: What rests on it, by address.
    holds_up: tuple[str, ...] = ()

    def describes(self) -> str:
        rested = f", {len(self.holds_up)} rest on it" if self.holds_up else ""
        return f"{self.at} (used {self.used}{rested})"


def what_she_is_made_of() -> list[APart]:
    """Every mutable part, with one kind of address.

    The kinds are not a taxonomy of development. They are the registries this
    codebase keeps, and a part is here exactly when something can install a
    different one in its place.
    """
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import (
        WAYS_TO_BUILD,
        WHAT_OF_IT,
        WHERE_FROM,
    )
    from core.cognition.one_algebra import DERIVED_HEADS
    from core.cognition.the_order_she_tries_them_in import the_order_she_uses
    from core.cognition.the_proposer_she_can_replace import the_proposer_in_use
    from core.cognition.the_record_of_her_own_work import (
        how_long_since,
        the_record,
    )
    from core.cognition.what_it_is_worth_doing import the_worth_she_uses
    from core.cognition.what_rests_on_what import what_rests_on_it

    uses = the_record().uses
    made: list[APart] = []

    def add(kind: str, name: str, term: Any = None) -> None:
        made.append(
            APart(
                at=f"{kind}/{name}",
                kind=kind,
                name=name,
                term=term,
                used=int(uses.get(name, 0)),
                idle=how_long_since(name),
                holds_up=tuple(what_rests_on_it(name)),
            )
        )

    for name, word in WHERE_FROM.items():
        add("word", name, getattr(word, "term", None))
    for name in WHAT_OF_IT:
        add("what is done", name)
    for name in WAYS_TO_BUILD:
        add("way of building", name)
    for name, head in DERIVED_HEADS.items():
        add("way of computing", name, head.body)
    for name, rule in RULES_WITH_NO_SHAPE.items():
        add("rule", name, getattr(rule, "body", None))
    add("the search", "the order she tries them in", the_order_she_uses())
    add("the search", "the proposer", the_proposer_in_use())
    add("the deciding", "what a change is worth", the_worth_she_uses())
    return made


def what_a_part_is_worth(
    part: APart,
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Callable[[Sequence[Any]], int],
) -> int | None:
    """What the families she has met cost without this part, less with it.

    Positive means the part pays: taking it out makes everything dearer. Zero
    means it does nothing measurable and is a candidate for letting go.
    Nothing means the part cannot be taken out, which is an answer too.

    A lesion, so the number is causal. Correlational readings — this part was
    active when things went well — are what make a system confident about the
    wrong component.
    """
    from core.cognition.sequence_induction import (  # noqa: PLC2701
        _everything_she_can_say,
        _put_back,
    )

    if not probe:
        return None
    with_it = sum(costs(cases) for _name, cases in probe)
    held = _everything_she_can_say()
    if not _take_it_out(part):
        return None
    try:
        without_it = sum(costs(cases) for _name, cases in probe)
    finally:
        _put_back(held)
    return without_it - with_it


def _take_it_out(part: APart) -> bool:
    """Remove one part, whatever registry it is in. False where it cannot be."""
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import (
        WAYS_TO_BUILD,
        WHAT_OF_IT,
        WHERE_FROM,
    )
    from core.cognition.one_algebra import DERIVED_HEADS

    where = {
        "word": WHERE_FROM,
        "what is done": WHAT_OF_IT,
        "way of building": WAYS_TO_BUILD,
        "way of computing": DERIVED_HEADS,
        "rule": RULES_WITH_NO_SHAPE,
    }.get(part.kind)
    if where is None or part.name not in where:
        return False
    del where[part.name]
    return True


def where_the_time_goes(
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Callable[[Sequence[Any]], int],
) -> list[dict[str, Any]]:
    """Every part, ranked by what changing it would be worth.

    The bottleneck without a taxonomy. What comes back is a number per part,
    and whichever part the number is largest for is where a change would pay.
    Calling that a representation problem or a search problem is something a
    reader does afterwards.
    """
    found: list[dict[str, Any]] = []
    for part in what_she_is_made_of():
        worth = what_a_part_is_worth(part, probe, costs=costs)
        found.append(
            {
                "at": part.at,
                "pays": worth,
                "used": part.used,
                "idle": part.idle,
                "holds up": len(part.holds_up),
            }
        )
    found.sort(key=lambda row: -(row["pays"] or 0))
    return found


def the_most_they_have_in_common(first: Any, second: Any) -> Any | None:
    """The least general term both of these are instances of.

    Anti-unification, and it is what makes consolidating two parts into one
    possible: where two terms agree, keep the agreement; where they differ,
    leave a hole. A result that is all hole says they have nothing in common,
    and nothing is what comes back.
    """
    from core.cognition.the_floor_she_stands_on import Code

    holes: list[int] = [0]

    def both(a: Any, b: Any) -> Any:
        if not isinstance(a, Code) or not isinstance(b, Code):
            return None
        if a == b:
            return a
        if (
            a.head != b.head
            or len(a.parts) != len(b.parts)
            # A head that carries a value is only the same term when the value
            # is the same. Without this the generalisation of one plus two and
            # one plus five is one plus two, which is not a generalisation of
            # anything — it is the first of them with the difference hidden.
            or (not a.parts and a.value != b.value)
        ):
            holes[0] += 1
            return Code("the one it was given", parts=(), value=0)
        parts = tuple(both(one, other) for one, other in zip(a.parts, b.parts))
        if any(one is None for one in parts):
            return None
        return Code(a.head, parts=parts, value=a.value)

    made = both(first, second)
    if made is None:
        return None
    from core.cognition.the_floor_she_stands_on import how_long

    # All hole is not a generalisation; it is the statement that they share
    # nothing, and saying so is more useful than handing back a variable.
    return made if how_long(made) > holes[0] + 1 else None
