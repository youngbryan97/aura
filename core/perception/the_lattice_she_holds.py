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
    #: The places offered last time, so a set that has stopped changing can be
    #: told from one that is still growing.
    _offered: frozenset[tuple[int, int]] = field(default_factory=frozenset)

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
        # Built from places that have stopped changing, not from places seen
        # twice.
        #
        # The set grows as she watches, so building from it the moment it has
        # two entries builds from whatever the first two happened to be. That
        # was survivable while remembered places counted immediately, because
        # a sitting inherited a full set. Once they have to be seen again, a
        # short sitting genuinely has almost nothing of its own — and it built
        # a grid out of it and wrote that down. LIVE 2026-09-02, four sittings
        # running: five by nine, one by two, five by four, twelve by sixteen,
        # each one starting from the last one's wreckage.
        #
        # The same set twice running means the evidence has settled. Nothing
        # is chosen: it is offered whatever it is offered, and waits.
        if held != self._offered:
            self._offered = held
            return False
        # And enough of them to be a grid at all. Fewer places than lines
        # means every line rests on one place, which is not a grid — it is a
        # few things she has seen, drawn through.
        if len(held) <= len(_lines_through([x / 100 for x, _y in held])) + len(
            _lines_through([y / 100 for _x, y in held])
        ):
            return False
        # A place that already fits is not news about the shape.
        #
        # These places are gathered over acts, so the set grows as she watches
        # — and rebuilding from the larger set moves every line, because a
        # line is the mean of the places on it. The frame then changes under
        # her on the turn she learns something, and the two readings either
        # side of that turn are in different frames however alike they look.
        # LIVE 2026-08-31: a correct four-by-four lattice, and one comparison
        # in forty survived to teach her how the world moves.
        #
        # So places that land on the lines she holds, and that describe no
        # more lines than she holds, leave them exactly where they are. More
        # lines is a bigger view of the same thing and is taken: early on she
        # has seen two of a board's places, the gap between them is the whole
        # board, and everything fits inside it — a frame that could not grow
        # would freeze there.
        across = _lines_through([x / 100 for x, _y in held])
        down = _lines_through([y / 100 for _x, y in held])
        across = _lines_worth_having(across, [x / 100 for x, _y in held])
        down = _lines_worth_having(down, [y / 100 for _x, y in held])
        if not across or not down:
            return False
        if (
            self.held
            and (len(down), len(across)) == (self.rows, self.columns)
            and all(self._sits_on_a_line(x / 100, y / 100) for x, y in held)
        ):
            self._built_from, self.from_acts = held, acts
            return False
        self.across_at, self.down_at = across, down
        self._built_from, self.from_acts = held, acts
        # Both counts were about the grid this replaces. Two things landing on
        # one of the OLD grid's places says nothing about this one, and left
        # standing it goes on reporting that something is sitting over a thing
        # she can now read perfectly well.
        self.would_not_fit = 0
        self.crowded_for = 0
        return True

    def _sits_on_a_line(self, across: float, down: float) -> bool:
        """Whether a place lands on the lattice she is already holding."""
        if not self.held:
            return False
        return (
            abs(self.across_at[_nearest_to(across, self.across_at)] - across)
            <= _between(self.across_at)
            and abs(self.down_at[_nearest_to(down, self.down_at)] - down)
            <= _between(self.down_at)
        )

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


def _lines_worth_having(
    lines: tuple[float, ...], places: Sequence[float]
) -> tuple[float, ...]:
    """Drop lines with far fewer places on them than the rest.

    A grid is regular: every column of it holds as many places as the thing is
    tall, and every row as many as it is wide. So a line holding one place,
    beside lines holding four, is not one of the thing's lines — it is one
    place that got in.

    Which is how a score gets in. Its text is centred, so it moves sideways as
    its number lengthens, and a value that was a tile a moment ago turning up
    where the score now sits looks exactly like a tile that slid there. LIVE
    2026-09-02: a board four by four came back four by FIVE, and the fifth
    column had one thing in it.

    A line of a grid holds more than one place. A line holding exactly one,
    while the others hold several, is one place that got in — so that is the
    bar, and nothing else is. Comparing counts against each other looks more
    careful and is worse: the places are rounded to hundredths, so a column
    whose drift straddles a boundary is counted twice over and one that does
    not is counted once, and the counts come out fourteen, seven, fourteen,
    six for four columns that are all real.
    """
    if len(lines) < 3:
        return lines
    reach = _between(lines)
    on_each = [
        sum(1 for one in places if abs(one - line) <= reach) for line in lines
    ]
    if sorted(on_each)[len(on_each) // 2] < 2:
        # A thing this sparse has nothing to compare against.
        return lines
    kept = tuple(
        line for line, many in zip(lines, on_each, strict=True) if many > 1
    )
    return kept if len(kept) >= 2 else lines


def _between(lines: Sequence[float]) -> float:
    """How far from a line still counts as on it."""
    if len(lines) < 2:
        return 1.0
    gaps = sorted(b - a for a, b in zip(lines, lines[1:], strict=False) if b > a)
    return (gaps[len(gaps) // 2] if gaps else 1.0) * HALF_WAY


def _nearest_to(value: float, lines: Sequence[float]) -> int:
    return min(range(len(lines)), key=lambda one: abs(lines[one] - value))
