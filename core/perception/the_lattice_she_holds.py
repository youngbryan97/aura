"""One lattice, held across time, that readings are fitted into.

A reading of a screen was being treated as a thing complete in itself: work
out its rows, work out its columns, hand back a grid. Do that twice and you
have two grids, and they are not the same grid. A four by four board whose top
row happens to be empty has three rows in it, so a tile that was in the second
row is now in the first, and the move between the two readings is unreadable —
not because either reading was wrong but because they were never in the same
frame of reference.

Which is the wrong picture of what she is looking at. A board is not a series
of pictures that happen to resemble each other. It is one thing, in one place,
that lasts, and the tiles move about inside it. The squares are there when
they are empty. They are there while she is not looking. What changes between
one glance and the next is what is IN them, and that is only sayable if the
them is the same them.

So the lattice is held rather than re-derived. It is worked out once from
where things have been seen to happen — accumulated over acts, so no single
sparse glance can shrink it — and after that every reading is placed into it.
An empty place is empty rather than absent. A row nothing has landed on this
turn is still a row.

It is not permanent, because things do get replaced: a new game, a window
resized, a different screen. What replaces it is evidence that the thing has
changed shape, not one reading that failed to fill it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.perception.what_is_there import Arrangement, Cell

__all__ = ["TheLatticeSheHolds"]

#: Two places count as the same line when they are nearer than this share of
#: the usual gap between lines. Half, because a place more than half way to the
#: next line is nearer to that one, which is what nearest means rather than
#: a tolerance anybody picked.
HALF_WAY = 0.5


def _lines_through(values: Sequence[float]) -> tuple[float, ...]:
    """The lines these places sit on, found from the spacing they show.

    Whatever the thing is, the gaps within one of its lines are smaller than
    the gap to the next line, so the values say where the lines are without
    being told how many there should be.
    """
    ordered = sorted(values)
    if not ordered:
        return ()
    gaps = sorted(b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a)
    if not gaps:
        return (ordered[0],)
    typical = gaps[len(gaps) // 2]
    lines: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - lines[-1][-1] > typical * HALF_WAY:
            lines.append([])
        lines[-1].append(value)
    return tuple(sum(one) / len(one) for one in lines if one)


@dataclass
class TheLatticeSheHolds:
    """Where the thing's places are, kept between glances."""

    down_at: tuple[float, ...] = ()
    across_at: tuple[float, ...] = ()
    #: How many acts the places it was built from were gathered over.
    from_acts: int = 0
    #: Readings that would not go into it, in a row. Several is evidence the
    #: thing has changed rather than that one glance was poor.
    would_not_fit: int = 0
    #: Readings in a row with two things landing on one of its places. A thing
    #: with places does not put two things in one of them, so something is
    #: over it.
    crowded_for: int = 0
    _built_from: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    #: How many readings in a row have to refuse to fit before she accepts the
    #: thing itself has changed. One is a bad glance. Two disagreeing with each
    #: other as well as with her is a different thing in front of her.
    CHANGED_AFTER = 2

    @property
    def held(self) -> bool:
        return bool(self.down_at and self.across_at)

    @property
    def rows(self) -> int:
        return len(self.down_at)

    @property
    def columns(self) -> int:
        return len(self.across_at)

    def built_from(self, places: Iterable[tuple[int, int]], acts: int = 0) -> bool:
        """Work the lattice out from where things have been seen to happen.

        ``places`` are hundredths of the window, the way they are counted
        everywhere else. Given the same places again this does nothing, so it
        can be called every turn without the grid moving underfoot.
        """
        held = frozenset(places)
        if len(held) < 2 or held == self._built_from:
            return False
        across = _lines_through([x / 100 for x, _y in held])
        down = _lines_through([y / 100 for _x, y in held])
        if not across or not down:
            return False
        self.across_at, self.down_at = across, down
        self._built_from, self.from_acts = held, acts
        self.would_not_fit = 0
        return True

    def fit(self, said: Sequence[tuple[float, float, str]]) -> Arrangement | None:
        """This reading placed into the lattice she is holding.

        Everything that lands on one of its places is in it. Everything else is
        furniture and is left out rather than allowed to define a row. The
        shape is the lattice's shape whatever this glance happened to contain,
        which is what holding one is for.

        None when nothing landed in it at all, which is a reading of something
        else rather than a poor reading of this.
        """
        if not self.held:
            return None
        room = _between(self.down_at)
        reach = _between(self.across_at)
        found: dict[tuple[int, int], Cell] = {}
        howfar: dict[tuple[int, int], float] = {}
        crowded = 0
        for y, x, text in said:
            words = str(text or "").strip()
            if not words:
                continue
            row = _nearest_to(y, self.down_at)
            column = _nearest_to(x, self.across_at)
            down = abs(self.down_at[row] - y)
            across = abs(self.across_at[column] - x)
            if down > room or across > reach:
                continue
            where = (row, column)
            # Nearest wins, rather than giving up on the whole reading.
            #
            # A real capture had a system dialog sitting over the board, and
            # its lines of prose landed on the board's places. Refusing the
            # reading threw away the tiles that were perfectly visible beside
            # it, every turn, and she read nothing for the whole run. A thing
            # centred on a place is what is in that place; a line of prose
            # that merely overlaps one is not, and the distance says which.
            away = down + across
            if where in found:
                crowded += 1
                if howfar.get(where, 0.0) <= away:
                    continue
            found[where] = Cell(row=row, column=column, says=words, at=(x, y))
            howfar[where] = away
        if not found:
            self.would_not_fit += 1
            return None
        self.would_not_fit = 0
        self.crowded_for = self.crowded_for + 1 if crowded else 0
        return Arrangement(
            rows=self.rows,
            columns=self.columns,
            cells=tuple(found.values()),
            down_at=self.down_at,
            across_at=self.across_at,
        )

    def has_changed(self) -> bool:
        """Whether what she is looking at is no longer the thing she measured."""
        return self.would_not_fit >= self.CHANGED_AFTER

    def looks_covered(self) -> bool:
        """Whether something is sitting over the thing.

        A thing with places does not put two things in one of them. One
        reading like that is a stray region; several in a row is a dialog, a
        banner, or her own window in front of what she is trying to read — and
        reading through it gives an answer that looks perfectly well formed and
        is wrong, which is worse than not reading at all.

        Measured on a real capture with a system permission dialog over the
        board: reading through it put a quarter of the cells right and the
        rule never formed. The same screen with the dialog gone put fifty-five
        of sixty right and the rule came out at a hundred percent.
        """
        return self.crowded_for >= self.CHANGED_AFTER

    def as_memory(self) -> dict[str, Any]:
        return {
            "down_at": list(self.down_at),
            "across_at": list(self.across_at),
            "from_acts": self.from_acts,
            "built_from": [f"{x},{y}" for x, y in sorted(self._built_from)],
        }

    @classmethod
    def from_memory(cls, held: Any) -> "TheLatticeSheHolds":
        """The lattice she had last time. Where a thing is tends to keep."""
        if not isinstance(held, dict):
            return cls()

        def edges(key: str) -> tuple[float, ...]:
            try:
                return tuple(float(one) for one in held.get(key) or ())
            except (TypeError, ValueError):
                # not a failure: a remembered edge that is not a number is not
                # an edge, and starting fresh is the honest answer.
                return ()

        places: set[tuple[int, int]] = set()
        for one in held.get("built_from") or ():
            try:
                x, y = (int(part) for part in str(one).split(","))
            except (TypeError, ValueError):
                continue
            places.add((x, y))
        return cls(
            down_at=edges("down_at"),
            across_at=edges("across_at"),
            from_acts=int(held.get("from_acts") or 0),
            _built_from=frozenset(places),
        )


def _between(lines: Sequence[float]) -> float:
    """How far from a line still counts as on it."""
    if len(lines) < 2:
        return 1.0
    gaps = sorted(b - a for a, b in zip(lines, lines[1:], strict=False) if b > a)
    return (gaps[len(gaps) // 2] if gaps else 1.0) * HALF_WAY


def _nearest_to(value: float, lines: Sequence[float]) -> int:
    return min(range(len(lines)), key=lambda one: abs(lines[one] - value))
