"""What a word she invented is worth, measured in symbols rather than claimed.

There are two things "she needed the word she made" can mean, and only one of
them can be true.

The strong one says the behaviour is not expressible without it. That is false
for anything she can invent here, and false by a one-line argument rather than
by experiment: a word she made is a term over words she already had, so any
term mentioning it can have it substituted away, and the result denotes exactly
the same thing. Naming is a ``let``. Unfolding a ``let`` never changes what an
expression means, so the set of behaviours cannot grow.

The weak one says the behaviour was out of reach and is now in reach. That is
true, measurable, and the thing worth having. She searches to a bounded depth,
which admits terms up to some length and no longer. A behaviour whose shortest
saying is twenty-three symbols is not findable at a horizon of nine, however
expressible it is. Give her a word that says the awkward part in one symbol and
the same behaviour is three symbols long, and now it is findable.

So what an invention buys is the difference between those two lengths, and
whether that difference carries the behaviour across the horizon she actually
searches to. Both are counted here, and the second is checked by taking the
word away again — an invention that cannot be ablated has not been shown to
have done anything.

This was written after claiming the strong version and being wrong. The
experiment walked 18,658,023 terms, found nothing, and concluded necessity; it
had skipped every term with no hole in it, and the answer was a
twenty-three-symbol term with no holes at all, which says the behaviour exactly
at every size from two to fifty-nine.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "WhatItBought",
    "WhatItPaidFor",
    "does_it_pay_for_itself_across",
    "the_horizon_of",
    "the_shortest_way_to_say",
    "what_an_invention_buys",
]

logger = logging.getLogger("Aura.WhatAnInventionBuys")


def the_horizon_of(deepest: int) -> int:
    """The longest term a search to this depth can reach.

    Read off the enumerator rather than chosen: it walks sizes up to twice the
    depth plus one, so that is the longest thing it can offer.
    """
    return 2 * max(1, int(deepest)) + 1


@dataclass(frozen=True)
class WhatItBought:
    """What having the word did to the length of saying one behaviour."""

    #: Symbols the shortest saying takes without it, or None if none was found.
    without: int | None
    #: Symbols the shortest saying takes with it.
    with_it: int | None
    #: The longest term the search can reach.
    horizon: int
    #: How far the search was allowed to look for the version without it.
    looked_to: int

    @property
    def shorter_by(self) -> int | None:
        if self.without is None or self.with_it is None:
            return None
        return self.without - self.with_it

    @property
    def crossed_the_horizon(self) -> bool:
        """Whether it went from out of reach to in reach.

        Out of reach means longer than the horizon or not found inside a
        search that looked further than the horizon — which is the honest
        reading of not finding something.
        """
        if self.with_it is None or self.with_it > self.horizon:
            return False
        return self.without is None or self.without > self.horizon

    def describes(self) -> str:
        without = "not found" if self.without is None else f"{self.without} symbols"
        with_it = "not found" if self.with_it is None else f"{self.with_it} symbols"
        said = f"{without} without it, {with_it} with it, horizon {self.horizon}"
        if self.crossed_the_horizon:
            return f"{said} — it carried the behaviour into reach"
        return f"{said} — it did not carry the behaviour into reach"


def the_shortest_way_to_say(
    says_it: Callable[[Any], bool],
    words: Mapping[str, Any],
    *,
    up_to: int,
    holes: int = 2,
    constants: Sequence[int] = (0, 1, 2, 3, 4, 5),
    within: float = 60.0,
) -> tuple[int | None, str, bool]:
    """The length of the shortest term saying it, how it said it, and whether
    the search finished.

    Terms with no hole are walked too. Skipping them is what turned a
    twenty-three-symbol closed answer into a proof of impossibility.

    The third value says whether every term up to ``up_to`` was seen. Not
    finding something inside a budget is not the same as its not being there,
    and only one of those is a fact about the language.
    """
    from core.cognition.one_algebra import Made, every_term, holes_in
    from core.cognition.one_algebra import _choose  # noqa: PLC2701 - one algebra

    names = sorted(words)
    began = time.monotonic()
    deepest = max(1, (int(up_to) - 1) // 2)
    for term in every_term(tuple(constants), holes=max(1, holes), deepest=deepest):
        if term.how_long() > up_to:
            continue
        if time.monotonic() - began > within:
            return None, "", False
        needs = holes_in(term)
        if needs == 0:
            try:
                if says_it(Made(term=term, words=())):
                    return term.how_long(), term.name, True
            except Exception:
                pass
            continue
        if needs > holes:
            continue
        for chosen in _choose(names, needs):
            try:
                if says_it(Made(term=term, words=tuple(words[one] for one in chosen))):
                    return (
                        term.how_long(),
                        f"{term.name} [{', '.join(chosen)}]",
                        True,
                    )
            except Exception:
                continue
    return None, "", True


def what_an_invention_buys(
    says_it: Callable[[Any], bool],
    *,
    given: Mapping[str, Any],
    invented: Mapping[str, Any],
    deepest: int = 4,
    holes: int = 2,
    look_past_the_horizon_to: int | None = None,
    within: float = 60.0,
) -> WhatItBought:
    """Measure it, and check by taking the word away again.

    ``given`` is what she had; ``invented`` is what she made. The version
    without is searched FURTHER than the horizon, because the interesting
    answer is that the short saying needs a word and the long one does not —
    and a search stopped at the horizon could never tell that from absence.
    """
    horizon = the_horizon_of(deepest)
    looked_to = int(look_past_the_horizon_to or horizon)
    without, how_without, finished = the_shortest_way_to_say(
        says_it, given, up_to=looked_to, holes=holes, within=within
    )
    with_it, how_with, _done = the_shortest_way_to_say(
        says_it, {**given, **invented}, up_to=horizon, holes=holes, within=within
    )
    if not finished:
        logger.info("the search without it did not finish; 'not found' is a budget")
    logger.info("without it: %s | with it: %s", how_without or "—", how_with or "—")
    return WhatItBought(
        without=without, with_it=with_it, horizon=horizon, looked_to=looked_to
    )


@dataclass(frozen=True)
class WhatItPaidFor:
    """Whether a word pays for itself over several tasks rather than one."""

    #: Symbols to say all of them without it, counting only those both can say.
    without: int
    #: Symbols to say all of them with it.
    with_it: int
    #: What the word itself costs to carry, written out in the old language.
    the_word_costs: int
    #: How many tasks were weighed, and how many it helped.
    over: int
    helped: int

    @property
    def pays(self) -> bool:
        """One task can always be shortened by a word made for it.

        The question is whether the saving survives paying for the word once,
        which is why the word's own length is on the other side of this.
        """
        return self.without > self.the_word_costs + self.with_it

    def describes(self) -> str:
        verdict = "pays for itself" if self.pays else "does not pay for itself"
        return (
            f"{verdict}: {self.without} symbols for {self.over} task(s) without it, "
            f"{self.the_word_costs} to carry it plus {self.with_it} with it; "
            f"it shortened {self.helped}"
        )


def does_it_pay_for_itself_across(
    tasks: Sequence[Callable[[Any], bool]],
    *,
    given: Mapping[str, Any],
    invented: Mapping[str, Any],
    up_to: int = 9,
    holes: int = 1,
    within: float = 20.0,
) -> WhatItPaidFor:
    """Weigh a word over several tasks, paying for it once.

    A word made for one task shortens that task; that is what making it did,
    and reading it as evidence is reading the fitting rather than the word. A
    word worth carrying shortens tasks it was not made for, by more than it
    costs to carry — which is the ordinary description-length trade and needs
    no weights, both sides being symbols.

    Only tasks BOTH languages can say are counted. A task the old language
    cannot say at this length is a different claim — that is reach, and
    :func:`what_an_invention_buys` measures it — and mixing the two would let
    one unreachable task pay for any word at all.
    """
    with_words = {**given, **invented}
    the_word_costs = 0
    for name in invented:
        alone, _how, _done = the_shortest_way_to_say(
            _just(invented[name]), given, up_to=up_to, holes=holes, within=within
        )
        the_word_costs += alone if alone is not None else up_to + 1

    without = with_it = helped = counted = 0
    for says_it in tasks:
        old, _a, _b = the_shortest_way_to_say(
            says_it, given, up_to=up_to, holes=holes, within=within
        )
        new, _c, _d = the_shortest_way_to_say(
            says_it, with_words, up_to=up_to, holes=holes, within=within
        )
        if old is None or new is None:
            continue
        counted += 1
        without += old
        with_it += new
        helped += new < old
    return WhatItPaidFor(
        without=without,
        with_it=with_it,
        the_word_costs=the_word_costs,
        over=counted,
        helped=helped,
    )


def _just(word: Any) -> Callable[[Any], bool]:
    """Says-it for one word: anything behaving as it does, at the usual sizes."""

    def says_it(other: Any) -> bool:
        try:
            return all(
                int(other(at, size)) % size == int(word(at, size)) % size
                for size in (4, 5, 6)
                for at in range(size)
            )
        except (ArithmeticError, IndexError, TypeError, ValueError):
            return False

    return says_it
