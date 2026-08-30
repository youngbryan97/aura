"""Inventing a kind of thing, rather than another way of transforming things.

Every word she can invent so far is an operation: something that takes places
to places. That is a real language and it is a narrow one. A great deal of what
people mean by understanding something is not a new operation at all — it is a
new KIND of thing to have operations about.

Velocity is not an operation on positions. A gene is not an operation on
organisms. Prime, orbit, species, probability, latent variable: each of them
says that some things belong together and the difference between them stops
mattering. Once "even" and "odd" exist, arithmetic that was hopeless over the
numbers becomes trivial over the two of them.

The route to one is a quotient. When several cases behave identically for
everything she can currently ask, they are one thing wearing different clothes:

    x ~ y   when nothing she can say tells them apart

and the new kind of thing is the set of classes, X/~.

Which cases go together is not guessed. It is read off a failure: when the best
reading she has covers some of the cases and another covers exactly the rest,
that split IS the partition — it is the distinction her language is missing,
pointed at by the only thing that could point at it, which is what went wrong.

And the classifier is not given either. Given the two groups, the same
synthesis that writes a way of building words writes a term that says which
group a case is in. If no term does, there is no kind of thing here — only two
sets of cases that happen to differ, and naming that would be inventing a
distinction nobody can apply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "AKindOfThing",
    "a_way_of_building_over",
    "KINDS_OF_THING",
    "a_kind_of_thing_she_named",
    "read_back",
    "written_down",
]

logger = logging.getLogger("Aura.AKindOfThingSheNamed")


@dataclass(frozen=True)
class AKindOfThing:
    """A distinction she invented, and the test that applies it.

    ``tells`` is a term over a case that answers which class it is in, so the
    kind is usable rather than merely observed: anything she works out from
    here may ask it.
    """

    name: str
    tells: Any
    #: What each class is called, in the order the test numbers them.
    classes: tuple[str, ...] = ()
    #: How many cases were in each when it was named, so a class nothing was
    #: ever in can be told from one that is simply rare.
    seen: tuple[int, ...] = field(default_factory=tuple)

    def of(self, case: Sequence[Any]) -> int:
        """Which class this case is in.

        Whatever the test says, unaltered. Taking it modulo the number of
        classes would be putting the two-ness in by hand — the test has to
        produce the classes itself, or there is no distinction here that she
        found.
        """
        from core.cognition.one_algebra import run

        return int(run(self.tells, 0, len(case), ()))

    def describes(self) -> str:
        counts = ", ".join(
            f"{what} ({many})" for what, many in zip(self.classes, self.seen)
        )
        return f"{self.name}: {counts or 'two classes'}"


#: The kinds of thing she has named. Empty at import: nothing here is anybody's
#: idea of what a kind of thing should be.
KINDS_OF_THING: dict[str, AKindOfThing] = {}


def _the_split_the_failure_points_at(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    hypotheses: Sequence[Any],
) -> tuple[frozenset[int], frozenset[int]] | None:
    """The two groups of cases her language covers separately and not together.

    Nothing is guessed here. The best reading covers some cases; if another
    covers exactly the rest, those two groups are the distinction the language
    is missing, and the failure is what pointed at it.
    """
    covered: list[frozenset[int]] = []
    for one in hypotheses:
        read = getattr(one, "read", None)
        if not callable(read):
            continue
        right: set[int] = set()
        for at, (before, after) in enumerate(pairs):
            try:
                if read(before) == after:
                    right.add(at)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
                continue
        if right and len(right) < len(pairs):
            covered.append(frozenset(right))
    if not covered:
        return None
    covered.sort(key=len, reverse=True)
    everything = frozenset(range(len(pairs)))
    best = covered[0]
    rest = everything - best
    if not rest:
        return None
    for other in covered[1:]:
        if rest <= other:
            return best, rest
    return None


def a_kind_of_thing_she_named(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    hypotheses: Sequence[Any] | None = None,
    *,
    called: str = "",
) -> AKindOfThing | None:
    """Name the distinction a failure is pointing at, if a test can apply it.

    Returns nothing where the cases do not split that way, or where nothing in
    the algebra tells the two groups apart — a distinction she cannot apply is
    not a kind of thing, it is a remark about some examples.
    """
    from core.cognition.an_invented_kind import every_meaning
    from core.cognition.one_algebra import (
        _numbers_the_problem_shows,
        every_term,
        holes_in,
        run,
    )

    if hypotheses is None:
        hypotheses = list(every_meaning())
    split = _the_split_the_failure_points_at(pairs, hypotheses)
    if split is None:
        return None
    here, there = split

    # A term over the case that says which group it is in. Over the case, not
    # over a place in it: what is being asked is what KIND of thing this is.
    constants = _numbers_the_problem_shows(pairs)
    deepest = 2
    for term in every_term(constants, holes=1, deepest=deepest):
        if holes_in(term) > 0:
            # A test that needs a word is a test about places, not about kinds.
            continue
        try:
            said = {
                at: int(run(term, 0, len(pairs[at][0]), ()))
                for at in range(len(pairs))
            }
        except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
            continue
        # One value for everything in a group, a different one for the other,
        # and the term itself produces them. A test taken modulo two would
        # make anything look like two classes.
        if len({said[at] for at in here}) != 1 or len({said[at] for at in there}) != 1:
            continue
        if said[next(iter(here))] == said[next(iter(there))]:
            continue
        # And it has to be a distinction rather than a list of the cases seen.
        #
        # "How many there are" separates the four-long cases from the
        # five-long ones perfectly and gives a six-long case a class of its
        # own, which is not a kind of thing — it is the sizes she happened to
        # meet, written down. A kind holds on sizes she has not met.
        classes = {said[next(iter(here))], said[next(iter(there))]}
        if not _holds_on_sizes_it_never_saw(term, pairs, classes):
            continue
        name = str(called or f"a kind of thing: {term.name}")
        found = AKindOfThing(
            name=name,
            tells=term,
            classes=(
                f"{term.name} is {said[next(iter(here))]}",
                f"{term.name} is {said[next(iter(there))]}",
            ),
            seen=(len(here), len(there)),
        )
        KINDS_OF_THING[name] = found
        logger.info("she named a kind of thing — %s", found.describes())
        return found
    return None


def _holds_on_sizes_it_never_saw(
    term: Any,
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    classes: set[int],
) -> bool:
    """Whether the test still says one of those classes on unseen sizes.

    A test that invents a new class for every size it meets has not found a
    distinction; it has memorised the sizes in front of it. The check is the
    same held-out discipline as everywhere else, applied to the one thing a
    kind of thing has to do — hold on cases it was not shown.
    """
    from core.cognition.one_algebra import run

    met = {len(before) for before, _after in pairs}
    if not met:
        return False
    unseen = [size for size in range(2, max(met) + max(met) + 2) if size not in met]
    if not unseen:
        return False
    for size in unseen:
        try:
            if int(run(term, 0, size, ())) not in classes:
                return False
        except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
            return False
    return True


def a_way_of_building_over(kind: AKindOfThing) -> Any:
    """A way of building words that asks which kind of thing this is.

    The point of naming a distinction is that things can be said with it. This
    is the saying: one word for cases on one side of it and another for cases
    on the other, which is a thing her language could not express at all
    before the distinction existed.

    It is a term like any other, so everything downstream is unchanged and
    what she writes next may use it — including the next way of building.
    """
    from core.cognition.one_algebra import Term, as_a_maker

    first = kind.classes[0].rsplit(" is ", 1)[-1] if kind.classes else "0"
    try:
        which = int(first)
    except (TypeError, ValueError):
        which = 0
    branch = Term(
        "if",
        (
            Term("same as", (kind.tells, Term("fixed", value=which))),
            Term("hole", value=0),
            Term("hole", value=1),
        ),
    )
    return as_a_maker(branch), branch


def written_down(kind: AKindOfThing) -> dict[str, Any]:
    """The kind as plain data, so a distinction she drew survives a restart."""
    from core.cognition.one_algebra import written_down as term_data

    return {
        "name": kind.name,
        "tells": term_data(kind.tells),
        "classes": list(kind.classes),
        "seen": list(kind.seen),
    }


def read_back(row: Any) -> AKindOfThing | None:
    """A kind from what was written down, or nothing when it does not read."""
    from core.cognition.one_algebra import read_back as read_term

    if not isinstance(row, dict):
        return None
    tells = read_term(row.get("tells"))
    if tells is None:
        return None
    return AKindOfThing(
        name=str(row.get("name") or "a kind of thing"),
        tells=tells,
        classes=tuple(str(one) for one in row.get("classes") or ()),
        seen=tuple(int(one) for one in row.get("seen") or ()),
    )
