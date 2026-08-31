"""A property she commits to keeping true, and plans under.

Watched next to somebody good at a thing, the difference is not that they
search further. It is that most of what she considers, they never consider at
all.

A person clearing 2048 in 989 moves holds one thing true for the whole game:
the largest tile stays in its corner. That is not re-decided each move. It
excludes a whole direction from the start — press the wrong way and the corner
empties — so at every step they choose between two or three moves where she
weighs four. It also orders the rest: values descending away from the corner,
so that merging two things MAKES the next merge possible. Four and four become
eight beside an eight; that eight and its neighbour become sixteen beside a
sixteen. Each step manufactures the precondition for the next.

Scoring every move on its own merits cannot produce that, however deep it
looks. A ladder is not a sequence of individually good moves; it is a shape
held across hundreds of them.

So: a property is a term over what she can see, in the same algebra everything
else here is written in — nothing about corners or boards is written down. She
watches which properties are true when things go well and false when they do
not, commits to the one that predicts it best, and from then on a move that
would break it is not weighed against other moves. It is not offered.

And when something forces her to break it, restoring it is what she does next.
That is the whole of what a plan is here: not a list of moves, but a thing kept
true, and the moves that keep it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SomethingTrue",
    "what_to_hold_now",
    "every_property_of",
    "how_well_it_predicts",
    "the_one_worth_holding",
    "what_it_rules_out",
]

logger = logging.getLogger("Aura.SomethingSheKeepsTrue")


@dataclass(frozen=True)
class SomethingTrue:
    """A property of a state, and how well holding it went."""

    name: str
    #: Answers whether it holds of a state.
    holds: Callable[[Any], bool]
    #: How often things went well when it held, and when it did not.
    well_when_true: int = 0
    times_true: int = 0
    well_when_false: int = 0
    times_false: int = 0

    @property
    def tells_them_apart(self) -> float:
        """How much better things go when it holds. Nought means it says nothing.

        The difference of two rates rather than one rate, because a property
        true of every state she has ever been in explains nothing however well
        those went.
        """
        if not self.times_true or not self.times_false:
            return 0.0
        return (self.well_when_true / self.times_true) - (
            self.well_when_false / self.times_false
        )

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.well_when_true}/{self.times_true} went well when it "
            f"held, {self.well_when_false}/{self.times_false} when it did not "
            f"({self.tells_them_apart:+.2f})"
        )


def every_property_of(
    reading: Any, *, deepest: int = 2
) -> Iterator[tuple[str, Callable[[Any], bool]]]:
    """Properties a state can have, from the algebra rather than from a list.

    Nothing here knows what a board is, or a corner. What it knows is that a
    laid-out thing has places, the places hold values, and a property is a
    comparison over them — where the largest sits, whether values fall away
    from somewhere, how much room is left. Those are the same comparisons
    every other term in this codebase is built from.
    """
    # What a reading has to offer, asked of it rather than imported.
    #
    # Cognition may not reach into perception, and it does not need to: a
    # thing with places, each holding a value, is all this needs, and asking
    # the reading whether it is one keeps it usable for anything shaped that
    # way rather than for one class.
    yield from _properties_of_anything(reading)

    cells = getattr(reading, "cells", None)
    if not cells or not hasattr(reading, "rows") or not hasattr(reading, "columns"):
        return

    def value_of(cell: Any) -> float:
        try:
            got = cell.number()
        except (AttributeError, TypeError, ValueError):
            # not a failure: a place holding something that is not a number
            # holds nothing this can compare, which is an answer.
            return 0.0
        return float(got) if got is not None else 0.0

    corners = {
        "the near end of the first row": (0, 0),
        "the far end of the first row": (0, -1),
        "the near end of the last row": (-1, 0),
        "the far end of the last row": (-1, -1),
    }
    for said, (row, column) in corners.items():
        yield (
            f"the largest thing is at {said}",
            _the_largest_is_at(row, column, value_of),
        )
    for said, along in (("rows", True), ("columns", False)):
        yield (
            f"values fall away along its {said}",
            _values_fall_away(along, value_of),
        )
    # The one that actually held for a whole game.
    #
    # Watching a person clear 2048: the anchor was the top right corner for
    # four minutes, the board took it away from them, and they did NOT fight
    # to put it back. They kept the right EDGE and let the anchor slide down
    # it, rebuilding the ladder from wherever the big tile now sat, and
    # finished with it in the opposite corner. What held for all 989 moves was
    # not a corner. It was that the large values live on one edge.
    #
    # A corner is the special case of an edge where the anchor is at its end,
    # so both are here and which one tells the good states from the bad is
    # hers to find.
    for said, edge in (
        ("first row", ("row", 0)),
        ("last row", ("row", -1)),
        ("first column", ("column", 0)),
        ("last column", ("column", -1)),
    ):
        yield (
            f"the largest things live along its {said}",
            _the_largest_live_along(edge, value_of),
        )


def _the_largest_is_at(
    row: int, column: int, value_of: Callable[[Any], float]
) -> Callable[[Any], bool]:
    def holds(reading: Any) -> bool:
        cells = getattr(reading, "cells", ())
        if not cells:
            return False
        rows, columns = reading.rows, reading.columns
        want = ((row % rows) if rows else 0, (column % columns) if columns else 0)
        biggest = max(cells, key=value_of)
        return (biggest.row, biggest.column) == want and value_of(biggest) > 0

    return holds


def _values_fall_away(
    along_rows: bool, value_of: Callable[[Any], float]
) -> Callable[[Any], bool]:
    """Whether values never rise as you go along. An ordering that makes
    merges cascade rather than scatter."""

    def holds(reading: Any) -> bool:
        cells = getattr(reading, "cells", ())
        if len(cells) < 2:
            return False
        lines: dict[int, list[tuple[int, float]]] = {}
        for cell in cells:
            key = cell.row if along_rows else cell.column
            at = cell.column if along_rows else cell.row
            lines.setdefault(key, []).append((at, value_of(cell)))
        settled = 0
        for line in lines.values():
            in_order = [value for _at, value in sorted(line)]
            if len(in_order) < 2:
                continue
            settled += 1
            rising = all(a <= b for a, b in zip(in_order, in_order[1:]))
            falling = all(a >= b for a, b in zip(in_order, in_order[1:]))
            if not (rising or falling):
                return False
        return settled > 0

    return holds


def _places_in(thing: Any) -> list[tuple[Any, Any]]:
    """Whatever this is, said as places holding things.

    A laid-out thing has cells. A mapping has keys. Anything countable has
    positions. All three are places holding things, and the properties below
    only ever needed that much, so all three get them. Refusing everything
    that is not laid out in rows was a limit on the shapes of world she could
    hold anything true about, and nothing in the reasoning needed it.
    """
    cells = getattr(thing, "cells", None)
    if cells:
        out = []
        for cell in cells:
            where = (getattr(cell, "row", None), getattr(cell, "column", None))
            out.append((where, cell))
        return out
    if isinstance(thing, Mapping):
        return list(thing.items())
    if isinstance(thing, (str, bytes)):
        return []
    if isinstance(thing, (Set, Sequence)):
        return list(enumerate(thing))
    return []


def _held(one: Any) -> Any:
    """What is at a place, as something comparable."""
    asked = getattr(one, "number", None)
    if callable(asked):
        try:
            got = asked()
        except (AttributeError, TypeError, ValueError):
            # not a failure: a place holding something unreadable holds
            # nothing to compare, which is itself an answer.
            return None
        return got
    return one


def _properties_of_anything(thing: Any) -> Iterator[tuple[str, Callable[[Any], bool]]]:
    """Properties that need only places and what is in them.

    These are the ones that carry across shapes of world, and one of them is
    the reason any of this was built: two places holding the same thing is the
    precondition of every rule that combines, so wanting it is how a want for
    something far away gets turned into a want for something near.
    """
    if not _places_in(thing):
        return
    yield ("two of its places hold the same thing", _two_the_same)
    yield ("something in it can be told apart from the rest", _not_all_alike)
    yield ("more than half its places are empty", _mostly_empty)
    # And one for each thing actually in it.
    #
    # Somebody clearing 2048 did not want a 1024, they wanted two 512s, and
    # before that two 256s. That vocabulary is not a list anybody wrote down;
    # it is read off what is in front of them, and it is what lets a want for
    # something far away become a want for something near.
    for what in _things_in(thing):
        yield (f"it holds a {what}", _holds_at_least(what, 1))
        yield (f"it holds two {what}", _holds_at_least(what, 2))


def _things_in(thing: Any) -> list[Any]:
    """The distinct things it holds, in a settled order.

    Bounded by what is there, so it needs no cutoff chosen for it.
    """
    seen: dict[str, Any] = {}
    for _where, one in _places_in(thing):
        held = _held(one)
        if held is None or held == 0:
            continue
        seen.setdefault(str(held), held)
    return [seen[key] for key in sorted(seen)]


def _holds_at_least(what: Any, many: int) -> Callable[[Any], bool]:
    def holds(thing: Any) -> bool:
        found = 0
        for _where, one in _places_in(thing):
            if _held(one) == what:
                found += 1
                if found >= many:
                    return True
        return False

    return holds


def _two_the_same(thing: Any) -> bool:
    seen: set[Any] = set()
    for _where, one in _places_in(thing):
        held = _held(one)
        if held is None or held == 0:
            continue
        try:
            if held in seen:
                return True
            seen.add(held)
        except TypeError:
            # not a failure: something that cannot be put in a set cannot be
            # compared for sameness this way.
            continue
    return False


def _not_all_alike(thing: Any) -> bool:
    seen: set[Any] = set()
    for _where, one in _places_in(thing):
        held = _held(one)
        try:
            seen.add(held)
        except TypeError:
            # not a failure: see above.
            continue
        if len(seen) > 1:
            return True
    return False


def _the_largest_live_along(
    edge: tuple[str, int], value_of: Callable[[Any], float]
) -> Callable[[Any], bool]:
    """Whether the biggest things sit on one line of the thing.

    Not where the single largest is, which a bad turn can take away, but where
    the WEIGHT is — and weight moves slowly. That is why it survives a break
    that a corner does not.
    """
    which, at = edge

    def holds(reading: Any) -> bool:
        cells = getattr(reading, "cells", ())
        if len(cells) < 3:
            return False
        rows, columns = reading.rows, reading.columns
        line = (at % rows) if which == "row" else (at % columns)
        on_it = 0.0
        everywhere = 0.0
        for cell in cells:
            value = value_of(cell)
            everywhere += value
            where = cell.row if which == "row" else cell.column
            if where == line:
                on_it += value
        if everywhere <= 0:
            return False
        # More of the weight on that line than off it, which is what "the
        # large things live there" means and needs no number chosen for it.
        return on_it * 2 > everywhere

    return holds


def _mostly_empty(thing: Any) -> bool:
    """How much room is left, whatever kind of thing it is.

    A laid-out thing knows how many places it has whether or not they are
    filled, so its emptiness is measured against that. Anything else only has
    the places that are there, and emptiness there means places holding
    nothing.
    """
    rows, columns = getattr(thing, "rows", 0), getattr(thing, "columns", 0)
    if rows and columns:
        return len(getattr(thing, "cells", ())) * 2 < rows * columns
    places = _places_in(thing)
    if not places:
        return False
    empty = sum(1 for _where, one in places if _held(one) in (None, 0, "", ()))
    return empty * 2 > len(places)


def how_well_it_predicts(
    watched: Sequence[tuple[Any, bool]], *, deepest: int = 2
) -> list[SomethingTrue]:
    """Weigh every property against how things actually went.

    ``watched`` is what she saw and whether it went well from there. Well is
    the caller's to define, because what counts as going well is what she was
    asked for.
    """
    if not watched:
        return []
    # Read the properties off everything she saw rather than off whichever
    # came first. One empty state at the front used to mean no properties at
    # all, and a property that only some states even have is exactly the kind
    # worth weighing.
    found: dict[str, SomethingTrue] = {}
    for reading, _went in watched:
        for name, holds in every_property_of(reading, deepest=deepest):
            found.setdefault(name, SomethingTrue(name=name, holds=holds))
    weighed: list[SomethingTrue] = []
    for one in found.values():
        well_true = times_true = well_false = times_false = 0
        for reading, went in watched:
            try:
                if one.holds(reading):
                    times_true += 1
                    well_true += bool(went)
                else:
                    times_false += 1
                    well_false += bool(went)
            except (ArithmeticError, AttributeError, TypeError, ValueError):
                continue
        weighed.append(
            SomethingTrue(
                name=one.name,
                holds=one.holds,
                well_when_true=well_true,
                times_true=times_true,
                well_when_false=well_false,
                times_false=times_false,
            )
        )
    return sorted(weighed, key=lambda one: -one.tells_them_apart)


def the_one_worth_holding(
    watched: Sequence[tuple[Any, bool]], *, deepest: int = 2
) -> SomethingTrue | None:
    """The property most worth keeping true, or nothing if none tells them apart.

    Nothing is the honest answer where every property is true as often in the
    good states as the bad: committing to one then would be superstition, and
    it would exclude moves for no reason.
    """
    weighed = how_well_it_predicts(watched, deepest=deepest)
    if not weighed or weighed[0].tells_them_apart <= 0:
        return None
    logger.info("she is holding: %s", weighed[0])
    return weighed[0]


def what_it_rules_out(
    holding: SomethingTrue,
    reading: Any,
    acts: Sequence[str],
    *,
    expect: Callable[[Any, str], Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The acts that keep it true, and the acts that break it.

    This is where holding something costs a search rather than a thought. Of
    four directions, two evict the corner: excluding them halves the tree at
    every level, before any looking ahead has happened.

    Where it is already broken, everything is offered — restoring it is then
    what the acts are for, and refusing to move would be holding a thing that
    is not true.
    """
    if not acts:
        return (), ()
    try:
        standing = holding.holds(reading)
    except (ArithmeticError, AttributeError, TypeError, ValueError):
        return tuple(acts), ()
    keeps: list[str] = []
    breaks: list[str] = []
    for act in acts:
        try:
            went = expect(reading, act)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
            continue
        if went is None:
            continue
        try:
            (keeps if holding.holds(went) else breaks).append(act)
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            continue
    if not standing:
        # Broken already, so every act is worth weighing and the ones that
        # restore it are the ones that matter.
        return tuple(keeps or acts), ()
    return tuple(keeps), tuple(breaks)


def what_to_hold_now(
    holding: SomethingTrue | None,
    watched: Sequence[tuple[Any, bool]],
    *,
    deepest: int = 2,
) -> tuple[SomethingTrue | None, str]:
    """What she should be keeping true now, given how the last while went.

    The hardest part of holding something is knowing when to stop. Watching a
    person clear 2048: their anchor was the top right corner for four minutes,
    the board took it from them, and they did not fight to put it back. They
    kept the edge, let the anchor slide along it, rebuilt from where the big
    tile now was, and finished in the opposite corner.

    Two failures are possible and they are opposite. Holding something the
    board has made impossible is stubbornness, and every move spent restoring
    it is spent. Dropping it because one turn went badly is thrashing, and
    nothing is ever held long enough to pay.

    What separates them is not how long it has been broken but whether it still
    tells the good states from the bad. A property that has stopped predicting
    has stopped being worth the moves it costs, whatever it once did — and if
    something else predicts better on what she has seen since, that is what to
    hold. Both are the measurement she already makes, asked again.
    """
    weighed = how_well_it_predicts(watched, deepest=deepest)
    if not weighed:
        return holding, "nothing seen yet"
    best = weighed[0]
    if holding is None:
        if best.tells_them_apart <= 0:
            return None, "nothing here tells the good from the bad"
        return best, f"took up {best.name}"
    mine = next(
        (one for one in weighed if one.name == holding.name),
        None,
    )
    if mine is None:
        return holding, "nothing recent to weigh it against"
    if mine.tells_them_apart <= 0 and best.tells_them_apart > 0:
        # It has stopped saying anything and something else is saying
        # something. That is the moment to let go, and it is not the same
        # moment as a bad turn.
        return best, f"{holding.name} stopped telling them apart; took up {best.name}"
    if best.tells_them_apart > mine.tells_them_apart * 2:
        return best, f"{best.name} tells them apart far better now"
    return holding, "still worth holding"
