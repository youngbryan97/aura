"""One mechanism for growing the language, and for growing what grows it.

She can add a word. She can add a way of building words, which enlarges what
she can say about families she has never met. The obvious next thing is a way
of building ways of building, and then the one above that, and writing a
separate mechanism for each level is a tower with no top.

The tower is not needed. A way of building is a value, the same as a word is a
value, so one registry holds both and one mechanism extends either. What
distinguishes them is a number saying what they consume:

    level 0   a word: where a value comes from
    level 1   a maker: takes words, returns more words
    level 2   a maker of makers: takes level-1 things, returns more of them

Adding at level 2 uses the code that adds at level 0. That is the whole
content of the collapse, and it is why

    S(t+1) = U(S(t), H(t))

is the honest shape rather than an infinite regress: the thing that changes her
is written in the same terms as the thing it changes, and lives in the same
place.

What stays true at every level is the discipline. Something is admitted when it
makes sayable a family that was not, and rolled back when it does not, because
a level-2 addition that changes nothing has multiplied the search by a large
number in exchange for nothing at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

__all__ = [
    "Made",
    "as_a_way_of_building",
    "Maker",
    "REGISTRY",
    "everything_makeable",
    "grow_at",
    "grow_until_sayable",
    "how_far_up_it_goes",
    "twice_over",
    "what_it_reaches",
]

logger = logging.getLogger("Aura.GrowingAtAnyLevel")


@dataclass(frozen=True)
class Maker:
    """Something that makes things, at whatever level it makes them.

    ``level`` says what it consumes. Nothing else in this module cares which
    level anything is, which is the point of writing it this way.
    """

    name: str
    level: int
    make: Callable[[dict[str, Any]], dict[str, Any]]
    #: The term it came from, where it came from one. Provenance, and the
    #: thing that makes a maker at any level the same kind of object as a word.
    term: Any = None

    def describes(self) -> str:
        return f"{self.name!r} at level {self.level}"


@dataclass(frozen=True)
class Made:
    """A record of what was made and out of what, so it can be undone."""

    name: str
    level: int
    out_of: tuple[str, ...]


#: Every maker she has, at every level. Level 1 is seeded from the ways of
#: building she already earned, so this is a different view of one language
#: rather than a second language beside it.
REGISTRY: dict[str, Maker] = {}


def twice_over(makers: dict[str, Any]) -> dict[str, Any]:
    """Apply a way of building to what it already built.

    A level-2 maker, and a real one: composing pairs of words gives pairs;
    doing it again gives what three words in a row produce, which no amount of
    adding level-1 makers reaches. Written once here so that admitting it can
    be shown to run through the same path as admitting a word.
    """
    made: dict[str, Any] = {}
    for name, build in makers.items():

        def again(words: dict[str, Any], build: Any = build) -> dict[str, Any]:
            once = dict(words)
            once.update(build(words))
            return build(once)

        made[f"{name}, and again"] = again
    return made


def everything_makeable(level: int, *, top: int | None = None) -> dict[str, Any]:
    """Everything that exists at a level once every level above it has run.

    Read downwards, because that is the direction the levels act in: a maker of
    makers produces makers, and those makers produce words. Asking what words
    exist therefore means asking what makers exist, which means asking what
    makes those.

    Level 0 is the words. Level 1 is the ways of building words. Level 2 makes
    ways of building, and there is no ceiling written anywhere — the recursion
    stops at the highest level anything has actually been admitted at.
    """
    from core.cognition.an_invented_kind import WHERE_FROM

    ceiling = how_far_up_it_goes() if top is None else int(top)
    wanted = max(0, int(level))
    if wanted == 0:
        made: dict[str, Any] = dict(WHERE_FROM)
    else:
        made = {
            maker.name: maker.make
            for maker in REGISTRY.values()
            if maker.level == wanted
        }
    if wanted >= ceiling:
        return made
    for build in everything_makeable(wanted + 1, top=ceiling).values():
        try:
            made.update(build(dict(made)))
        except (TypeError, ValueError, KeyError):
            continue
    return made


def what_it_reaches(level: int) -> int:
    """How many distinct things exist at a level once everything above has run."""
    return len(everything_makeable(level))


def grow_at(
    level: int,
    name: str,
    make: Any,
    *,
    now_sayable: Callable[[], bool],
) -> Made | None:
    """Admit a maker at any level, on the same terms as a word.

    Refused when the family was already sayable, and rolled back when admitting
    it does not make it sayable. One function, whatever the level, which is
    what makes the tower a number rather than a pile of mechanisms.

    ``make`` may be a TERM. That is the gap this module left when it collapsed
    the levels: it made one registry hold every level and then took a Python
    callable at each of them, so the tower was a number and the thing that
    filled it was still mine. A term with a hole in it is a way of building
    words — `one_algebra.as_a_maker` is the whole of the conversion — so a
    term arriving here is turned into one and nothing else about the call
    changes. A callable is still accepted, and every caller that passes one is
    a test.
    """
    said = str(name or "").strip()
    if not said or said in REGISTRY:
        return None
    make = as_a_way_of_building(make)
    if make is None:
        return None
    if now_sayable():
        # Nothing needed it. A maker earns its place by making something
        # possible, never by being available when nothing was blocked.
        return None
    below = tuple(sorted(everything_makeable(max(0, int(level) - 1))))
    from core.cognition.one_algebra import Term as PositionalTerm

    REGISTRY[said] = Maker(
        name=said,
        level=int(level),
        make=make,
        term=getattr(make, "term", None),
    )
    _publish(int(level))
    if not now_sayable():
        REGISTRY.pop(said, None)
        _publish(int(level))
        return None
    logger.info(
        "grew at level %d: %r — the same step that adds a word, one level up",
        level,
        said,
    )
    return Made(name=said, level=int(level), out_of=below)


def as_a_way_of_building(make: Any) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """A term becomes a way of building; anything callable is already one."""
    from core.cognition.one_algebra import Term, as_a_maker

    if isinstance(make, Term):
        return as_a_maker(make)
    if callable(make):
        return make
    return None


def grow_until_sayable(
    candidates: Sequence[tuple[int, str, Callable[[dict[str, Any]], dict[str, Any]]]],
    *,
    now_sayable: Callable[[], bool],
) -> tuple[Made, ...]:
    """Admit makers up the levels until a family becomes sayable, then cut back.

    Admitting one at a time and rolling back whatever does not help on its own
    cannot ever build a stack, because a maker of makers needs makers to work
    on and the makers it needs look useless until it arrives. So they go in
    together and the question is asked once at the top.

    Then every one of them is taken out again in turn and put back only if its
    absence loses the answer. What remains is the smallest stack that works,
    which matters because each level multiplies the search.
    """
    if now_sayable():
        return ()
    ordered = sorted(candidates, key=lambda one: int(one[0]))
    put_in: list[tuple[int, str]] = []
    for level, name, make in ordered:
        said = str(name or "").strip()
        if not said or said in REGISTRY:
            continue
        built = as_a_way_of_building(make)
        if built is None:
            continue
        REGISTRY[said] = Maker(name=said, level=int(level), make=built)
        put_in.append((int(level), said))
        _publish(int(level))
    if not now_sayable():
        for level, said in reversed(put_in):
            _take_out(said, level)
        return ()
    kept: list[Made] = []
    for level, said in reversed(put_in):
        maker = REGISTRY.get(said)
        if maker is None:
            continue
        _take_out(said, level)
        if now_sayable():
            # It was not needed. Carrying it would multiply the search for
            # nothing, which is the cost every level charges whether or not it
            # is earning it.
            continue
        REGISTRY[said] = maker
        _publish(level)
        kept.append(
            Made(
                name=said,
                level=level,
                out_of=tuple(sorted(everything_makeable(max(0, level - 1)))),
            )
        )
    if kept:
        logger.info(
            "grew %d level(s) at once, up to level %d, and kept only what was needed",
            len(kept),
            max(one.level for one in kept),
        )
    return tuple(reversed(kept))


def _take_out(name: str, level: int) -> None:
    """Remove a maker, and with it everything it was putting into the language."""
    REGISTRY.pop(name, None)
    _publish(level)


def _publish(level: int) -> None:
    """Put the ways of building where the interpreter reads them.

    The levels touch the rest of her in exactly one place. Whatever the tower
    does, what comes out of it is a set of ways to build words, and that is
    what she makes rules out of.
    """
    from core.cognition.an_invented_kind import WAYS_TO_BUILD

    WAYS_TO_BUILD.clear()
    WAYS_TO_BUILD.update(everything_makeable(1))


def how_far_up_it_goes() -> int:
    """The highest level anything has been admitted at."""
    return max((maker.level for maker in REGISTRY.values()), default=0)
