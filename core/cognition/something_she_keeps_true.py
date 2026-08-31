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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SomethingTrue",
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
    from core.perception.what_is_there import Arrangement

    if not isinstance(reading, Arrangement) or not reading.cells:
        return

    def value_of(cell: Any) -> float:
        got = cell.number()
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
    yield ("more than half its places are empty", _mostly_empty)


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


def _mostly_empty(reading: Any) -> bool:
    room = max(1, getattr(reading, "rows", 0) * getattr(reading, "columns", 0))
    return len(getattr(reading, "cells", ())) * 2 < room


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
    found: dict[str, SomethingTrue] = {}
    for reading, _went in watched:
        for name, holds in every_property_of(reading, deepest=deepest):
            found.setdefault(name, SomethingTrue(name=name, holds=holds))
        break
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
