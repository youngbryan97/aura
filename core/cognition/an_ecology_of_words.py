"""What happens to a word after it is admitted, which until now was nothing.

Two gates already ask whether something is worth adding: a word must remove
more search than it adds, and a maker must earn its place on families it was
never shown. Both look forward, both are asked once, and neither is ever asked
again. So the language could only grow, and a word that was a good idea in
August stayed a branch at every step of every search in December.

Growth on fixed hardware ends whatever she does — there are 2**B languages and
no more. What decides whether it ends well is whether anything ever leaves.

A word and a maker do not earn in the same currency, and treating them alike is
how forgetting goes wrong.

A word earns by being used. Nothing needs to be tallied to know that: the
meanings she has settled name the words they are made of, so use is read off
what she learned rather than counted alongside it. A count that drifts from
what it counts cannot happen here.

A maker earns by what it can still reach. It may have made nothing yet and be
the most valuable thing in the language, which is exactly why its admission
test was held-out reach and not use. What retires a maker is another maker that
makes everything it makes for less.

Which drawer a word goes in is not a matter of taste either, and building this
settled it: a word a maker makes is rebuilt from the maker every time the
language is asked for, so dropping it does nothing at all — it is back on the
next call. Only its maker can retire it. In the language as it stands after one
invention, twenty of the twenty-five words are made ones, so a forgetting that
only knew how to drop derived words would have reported success while touching
almost none of the cost.

One rule sits over both. She may forget a word only if every meaning she has
settled is still sayable without it. Retention is about what she learned, not
about every expression the language could form — and a word that is the only
way to say something she has never needed is precisely the thing to forget.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.cognition.keeping_the_language_small import what_a_word_is_worth
from core.runtime.errors import record_degradation

__all__ = [
    "AfterForgetting",
    "WhatItEarned",
    "a_maker_that_stopped_paying",
    "a_season",
    "forget",
    "what_can_be_forgotten",
    "what_each_word_earned",
]

logger = logging.getLogger("Aura.EcologyOfWords")


@dataclass(frozen=True)
class WhatItEarned:
    """One word or maker, weighed on what it did rather than what it promised."""

    name: str
    #: "derived" if she worked the word out and it is hers to drop; "made" if a
    #: maker rebuilds it on every call, so only retiring the maker removes it;
    #: "given" for the words the source handed her, which never go.
    sort: str
    #: The settled meanings that name it. Empty means nothing she learned uses it.
    used_by: tuple[str, ...]
    #: Expressions its shortening takes out of the search.
    removes: int
    #: Expressions it adds by being one more thing to try at every position.
    adds: int

    @property
    def pays(self) -> bool:
        return self.removes > self.adds

    def __str__(self) -> str:
        used = f"used by {len(self.used_by)}" if self.used_by else "used by nothing"
        return (
            f"{self.name!r} ({self.sort}, {used}): removes {self.removes:,}, "
            f"adds {self.adds:,}"
        )


@dataclass(frozen=True)
class AfterForgetting:
    """What left the language, and what its leaving bought back."""

    forgotten: tuple[str, ...]
    kept: int
    #: Expressions no longer to be walked, at the depth she actually thinks at.
    smaller_by: int
    #: Meanings still sayable. Forgetting that changes this is a bug, not a season.
    still_sayable: int

    def __str__(self) -> str:
        if not self.forgotten:
            return f"nothing forgotten; {self.kept} words, {self.still_sayable} meanings"
        return (
            f"forgot {', '.join(sorted(self.forgotten))} — {self.kept} words left, "
            f"{self.smaller_by:,} fewer expressions to walk, "
            f"{self.still_sayable} meanings still sayable"
        )


def _the_language() -> dict[str, Any]:
    from core.cognition.an_invented_kind import WHAT_OF_IT, addressings

    return {**addressings(), **WHAT_OF_IT}


def _where_it_came_from(name: str) -> str:
    """Which drawer a word belongs to, which decides what can be done about it."""
    from core.cognition.an_invented_kind import WHAT_OF_IT, WHERE_FROM
    from core.cognition.widening_the_language import DerivedAddressing, DerivedOperation

    if name not in WHERE_FROM and name not in WHAT_OF_IT:
        return "made"
    word = WHERE_FROM.get(name, WHAT_OF_IT.get(name))
    if isinstance(word, (DerivedAddressing, DerivedOperation)):
        return "derived"
    return "given"


def _what_the_settled_meanings_name() -> dict[str, list[str]]:
    """Every word named by a meaning she settled, and which meanings name it."""
    from core.cognition.an_invented_kind import KINDS

    names: dict[str, list[str]] = {}
    for kind, meaning in KINDS.items():
        for part in (meaning.where_from, meaning.and_from, meaning.what_of_it):
            names.setdefault(part, []).append(kind)
    return names


def _how_deep_she_thinks() -> int:
    """The length of the longest thought she has actually had.

    Read off rather than chosen, so the arithmetic below is about her language
    and not about a number somebody picked for it.
    """
    from core.cognition.an_invented_kind import KINDS
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    deepest = 1
    words = _the_language()
    for meaning in KINDS.values():
        cost = 0
        for part in (meaning.where_from, meaning.and_from, meaning.what_of_it):
            word = words.get(part)
            cost += what_it_costs_to_be(word, part) if word is not None else 1
        deepest = max(deepest, cost)
    return deepest


def what_each_word_earned(
    *, deepest: int | None = None
) -> tuple[WhatItEarned, ...]:
    """Every word she derived, re-weighed on the use it actually had.

    The same arithmetic that admitted it, with the promised use replaced by the
    observed one. A word that was going to save six symbols a hundred times and
    saved them twice is not the word that was admitted.
    """
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    words = _the_language()
    used = _what_the_settled_meanings_name()
    at_depth = int(deepest) if deepest is not None else _how_deep_she_thinks()
    earned: list[WhatItEarned] = []
    for name, word in sorted(words.items()):
        sort = _where_it_came_from(name)
        if sort == "given":
            continue
        by = tuple(sorted(set(used.get(name, ()))))
        # What saying it in one symbol saves over saying it the long way.
        shorter_by = max(0, what_it_costs_to_be(word, name) - 1)
        worth = what_a_word_is_worth(
            name,
            vocabulary=len(words),
            longest=at_depth,
            shorter_by=shorter_by,
            used=len(by),
        )
        earned.append(
            WhatItEarned(
                name=name,
                sort=sort,
                used_by=by,
                removes=worth.removes,
                adds=worth.adds,
            )
        )
    return tuple(earned)


def what_can_be_forgotten(
    *, deepest: int | None = None
) -> tuple[WhatItEarned, ...]:
    """Words that stopped paying and that nothing she settled needs.

    Both conditions, never one. A word may cost more than it saves and still be
    the only way to say something she worked out, and forgetting that is losing
    knowledge to save search — the wrong trade at any price.
    """
    return tuple(
        one
        for one in what_each_word_earned(deepest=deepest)
        if one.sort == "derived" and not one.pays and not one.used_by
    )


def forget(names: Iterable[str]) -> AfterForgetting:
    """Drop these words, and say what dropping them bought.

    Refuses to drop anything a settled meaning names, whatever the caller
    believes, because the caller may be working from a list made before the
    last thing she learned.
    """
    from core.cognition.an_invented_kind import KINDS, WHAT_OF_IT, WHERE_FROM
    from core.cognition.keeping_the_language_small import how_many_expressions

    used = _what_the_settled_meanings_name()
    at_depth = _how_deep_she_thinks()
    before = len(_the_language())
    went: list[str] = []
    for name in names:
        if used.get(name):
            logger.debug("%r is named by a meaning she settled; keeping it", name)
            continue
        if _where_it_came_from(name) != "derived":
            # A made word is rebuilt from its maker on the next call, so
            # dropping it here would report a saving that does not happen.
            logger.debug("%r is not hers to drop directly; keeping it", name)
            continue
        WHERE_FROM.pop(name, None)
        WHAT_OF_IT.pop(name, None)
        went.append(name)
    after = len(_the_language())
    return AfterForgetting(
        forgotten=tuple(went),
        kept=after,
        smaller_by=how_many_expressions(before, at_depth)
        - how_many_expressions(max(1, after), at_depth),
        still_sayable=len(KINDS),
    )


def _makers_that_make_the_same_things(
    makers: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Makers grouped by what they produce, shortest spelling first.

    A maker is retired by another maker, never by disuse: it may have made
    nothing and still be the only route to a family she has not met yet.
    """
    from core.cognition.one_thing_many_spellings import how_it_behaves
    from core.cognition.the_ruler_she_cannot_move import what_it_costs_to_be

    by_product: dict[tuple[Any, ...], list[tuple[int, str]]] = {}
    # A maker that raises makes nothing, and a maker that makes nothing is
    # kept. Counted rather than swallowed: if every maker raises, the grouping
    # is empty for the same reason as a language with no makers at all.
    broken: list[str] = []
    for name, maker in sorted(makers.items()):
        try:
            made = maker(dict(_the_language()))
        except Exception:  # noqa: BLE001 - makers are hers, counted below
            broken.append(name)
            continue
        marks = []
        costs = 0
        for word_name in sorted(made):
            does = how_it_behaves(made[word_name])
            if does is None:
                continue
            marks.append((word_name, does))
            costs += what_it_costs_to_be(made[word_name], word_name)
        if not marks:
            continue
        by_product.setdefault(tuple(marks), []).append((costs, name))
    if broken:
        record_degradation(
            "an_ecology_of_words",
            RuntimeError(f"makers raised instead of making: {', '.join(broken)}"),
            action="grouped the rest; a maker that raises is not retired for it",
        )
    kept: dict[str, tuple[str, ...]] = {}
    for group in by_product.values():
        if len(group) < 2:
            continue
        # The one whose products cost least to BE survives, on the same ruler
        # that decides every other question of this shape. Alphabetical order
        # is not a reason to keep one maker over another.
        order = sorted(group)
        kept[order[0][1]] = tuple(name for _cost, name in order[1:])
    return kept


def a_maker_that_stopped_paying(
    name: str, maker: Any, *, held_out: Iterable[Iterable[Any]]
) -> Any:
    """Re-run a maker's admission test on families it was never shown.

    The same measurement that let it in, asked again. A maker is not retired
    for having made nothing — it is retired when what it reaches no longer
    covers what its products cost as branches.
    """
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, addressings
    from core.cognition.is_it_worth_keeping import what_it_is_worth

    families = [list(one) for one in held_out]
    with_it = set(addressings())
    standing = WAYS_TO_BUILD.pop(name, None)
    try:
        without_it = set(addressings())
        was = [_is_sayable(one) for one in families]
    finally:
        if standing is not None:
            WAYS_TO_BUILD[name] = standing
    # What its products are used for: the settled meanings that name a word
    # only this maker puts in the language.
    named = _what_the_settled_meanings_name()
    its_own = with_it - without_it
    used = sum(len(named.get(word, ())) for word in its_own)
    return what_it_is_worth(
        now_sayable=_is_sayable,
        held_out=families,
        was_sayable=was,
        vocabulary_before=max(1, len(without_it)),
        vocabulary_after=max(1, len(with_it)),
        longest=_how_deep_she_thinks(),
        shorter_by=len(its_own),
        used=used,
    )


def _is_sayable(family: Iterable[Any]) -> bool:
    from core.cognition.an_invented_kind import induce_from

    pairs = [(tuple(before), tuple(after)) for before, after in family]
    try:
        return induce_from(pairs) is not None
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
        return False


def a_season(*, held_out: Iterable[Iterable[Any]] = ()) -> AfterForgetting:
    """One turn of the ecology: merge what is the same, drop what stopped paying.

    Named for what it is. A language pruned once has been tidied; one pruned
    every so often can go on growing at the edges without the middle becoming
    impassable.

    The report counts what a retired maker takes with it. It did not, at first,
    and said "nothing forgotten" in the same breath as dropping six words —
    because dropping a maker and dropping a word are two code paths and only
    one of them was writing the record.
    """
    from core.cognition.an_invented_kind import KINDS, WAYS_TO_BUILD
    from core.cognition.keeping_the_language_small import how_many_expressions

    at_depth = _how_deep_she_thinks()
    before = len(_the_language())
    went: list[str] = []

    doubled = _makers_that_make_the_same_things(WAYS_TO_BUILD)
    for _keeping, others in doubled.items():
        for other in others:
            if WAYS_TO_BUILD.pop(other, None) is not None:
                went.append(f"the maker {other!r}, which made what another made")

    families = [list(one) for one in held_out]
    if families:
        for name, maker in list(WAYS_TO_BUILD.items()):
            worth = a_maker_that_stopped_paying(name, maker, held_out=families)
            if worth.keep_it:
                continue
            logger.info("retiring %r: %s", name, worth.describes())
            WAYS_TO_BUILD.pop(name, None)
            went.append(f"the maker {name!r}, which {worth.describes()}")

    dropped = forget(one.name for one in what_can_be_forgotten())
    went.extend(dropped.forgotten)

    after = len(_the_language())
    return AfterForgetting(
        forgotten=tuple(went),
        kept=after,
        smaller_by=how_many_expressions(max(1, before), at_depth)
        - how_many_expressions(max(1, after), at_depth),
        still_sayable=len(KINDS),
    )
