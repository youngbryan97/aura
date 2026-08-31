"""A group of things that together license an act none of them licenses alone.

Asked what else was going on in the checkers game:

    using two pieces on either side of a gap as a pathway to "walk" up a line

Two pieces and an empty square between them. Anything stepping into the gap is
taken, so the gap is safe for her and not for them, and a line of those is a
corridor she can walk up. The thing to notice is that this is ONE thing to the
person playing. Not two pieces and a square — a pathway, seen whole, and seen
again somewhere else on the board without being worked out afresh.

That is what the chess studies found masters doing and novices not doing. De
Groot's masters rebuilt a position from five seconds of looking and could not
rebuild a random one; Chase and Simon's account is that what they saw was
configurations rather than pieces. The configurations are not given. They are
found, by having played.

Finding one here is an ablation and nothing cleverer. An act is safe. Take
away one part of the world at a time and ask whether it is still safe. What
the safety does not survive the loss of is what the safety rests on, and that
group, held together, is the thing worth naming.

Naming it by where its parts are would make it a fact about one board. Naming
it by where its parts are RELATIVE to the act makes it a shape, and a shape can
be looked for somewhere else — which is the whole difference between having
learned something and having remembered something.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AShape",
    "a_way_through",
    "what_makes_it_safe",
    "where_else_it_holds",
]


@dataclass(frozen=True)
class AShape:
    """What has to be around an act for the act to be safe, said as offsets."""

    around: tuple[tuple[tuple[int, ...], Any], ...]
    established: bool
    why: str

    @property
    def size(self) -> int:
        return len(self.around)

    def describe(self) -> str:
        if not self.around:
            return self.why
        where = ", ".join(f"{what} at {off}" for off, what in self.around)
        settled = "" if self.established else " (not checked on its own)"
        return f"{where}{settled}"


def _offset(there: Sequence[int], from_: Sequence[int]) -> tuple[int, ...] | None:
    if len(there) != len(from_):
        return None
    try:
        return tuple(int(a) - int(b) for a, b in zip(there, from_, strict=True))
    except (TypeError, ValueError):
        # not a failure: places that do not subtract cannot make a shape, and
        # saying so is the answer.
        return None


def what_makes_it_safe(
    state: Any,
    act: Any,
    *,
    safe: Callable[[Any, Any], bool],
    parts_of: Callable[[Any], Iterable[Any]],
    without: Callable[[Any, Any], Any],
    where_of: Callable[[Any], Sequence[int]],
    kind_of: Callable[[Any], Any],
    about: Callable[[Any], Sequence[int]],
) -> AShape:
    """The shape the safety of this act rests on.

    Each part is taken away in turn. The ones the safety does not survive are
    what it rests on. Then the group is checked alone — a world holding only
    those parts and the thing acting — because a set of individually necessary
    parts is not automatically a sufficient one, and a shape that has not been
    checked on its own is worth having but not worth trusting.
    """
    if not safe(state, act):
        return AShape((), False, "the act is not safe here to begin with")
    here = about(act)
    parts = list(parts_of(state))
    rests_on = [one for one in parts if not safe(without(state, one), act)]
    if not rests_on:
        return AShape((), True, "safe with nothing else around it")

    around: list[tuple[tuple[int, ...], Any]] = []
    for one in rests_on:
        off = _offset(where_of(one), here)
        if off is None:
            return AShape((), False, "these places do not make offsets")
        around.append((off, kind_of(one)))
    around.sort()

    alone = state
    keeping = set(map(id, rests_on))
    for one in parts:
        if id(one) not in keeping and _offset(where_of(one), here) != (0,) * len(here):
            alone = without(alone, one)
    established = safe(alone, act)
    return AShape(
        tuple(around),
        established,
        "these alone are enough" if established else "each is needed, together untested",
    )


def where_else_it_holds(
    state: Any,
    shape: AShape,
    *,
    places: Iterable[Sequence[int]],
    parts_of: Callable[[Any], Iterable[Any]],
    where_of: Callable[[Any], Sequence[int]],
    kind_of: Callable[[Any], Any],
) -> list[tuple[int, ...]]:
    """Everywhere the same shape is already standing.

    This is the part that makes it worth having found. A configuration learned
    in one corner is looked for in every other, and the ones it is found in are
    places she does not have to work anything out about.
    """
    if not shape.around:
        return []
    standing = {tuple(map(int, where_of(one))): kind_of(one) for one in parts_of(state)}
    found: list[tuple[int, ...]] = []
    for place in places:
        at = tuple(map(int, place))
        if all(
            standing.get(tuple(a + b for a, b in zip(at, off, strict=True))) == what
            for off, what in shape.around
        ):
            found.append(at)
    return found


def a_way_through(
    state: Any,
    shape: AShape,
    *,
    places: Iterable[Sequence[int]],
    parts_of: Callable[[Any], Iterable[Any]],
    where_of: Callable[[Any], Sequence[int]],
    kind_of: Callable[[Any], Any],
    next_to: Callable[[Sequence[int], Sequence[int]], bool],
) -> list[tuple[tuple[int, ...], ...]]:
    """Runs of touching places where the shape holds all the way along.

    One safe square is a square. Several in a row that touch is a corridor, and
    a corridor is a different kind of thing to have found: it says where she
    can get to, not only where she may step.
    """
    holds = where_else_it_holds(
        state, shape, places=places, parts_of=parts_of, where_of=where_of,
        kind_of=kind_of,
    )
    runs: list[list[tuple[int, ...]]] = []
    for at in holds:
        joined = [run for run in runs if any(next_to(at, was) for was in run)]
        if not joined:
            runs.append([at])
            continue
        first = joined[0]
        first.append(at)
        for other in joined[1:]:
            first.extend(other)
            runs.remove(other)
    return [tuple(sorted(run)) for run in runs if len(run) > 1]
