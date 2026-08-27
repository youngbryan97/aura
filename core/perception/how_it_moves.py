"""How this thing responds, worked out from watching herself act in it.

Every move she makes yields a triple — what was there, what she did, what was
there afterwards — and until now all three were thrown away after one glance.
So she could never try a move without making it, and every plan was a bet
placed blind.

This holds a small set of hypotheses about how a laid-out thing answers a
directional act, scores each against what actually happened, and uses the best
one while it keeps predicting. None of them is about any particular kind of
screen: they are the ways a set of things in rows and columns can respond to
being pushed. When none of them predicts, that is the answer, and she goes
back to acting and looking.

The scoring allows for what the world adds on its own. A board that deals a
new tile, a page that updates a clock, a list that gains a row from somebody
else — a rule is not wrong because something arrived that it never claimed to
know about. What it must get right is what it said would move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.perception.what_is_there import Arrangement, Cell

__all__ = ["HowItMoves", "Rule", "RULES", "shifted", "shifted_and_combined"]

logger = logging.getLogger("Aura.HowItMoves")

#: How many of her own moves a rule has to have survived before it is worth
#: predicting from. Below this the leader is whichever rule happened to match
#: the first thing she did.
ENOUGH_TO_TRUST = 4

#: The share of recent moves a rule has to have got right. A rule that is
#: wrong a third of the time is not a rule she can plan on.
OFTEN_ENOUGH = 0.7

#: How many of her own moves are kept. Enough to notice a rule going wrong
#: when the thing she is acting in changes under her.
REMEMBERED = 24

#: Directions a laid-out thing can be pushed, and what that does to an index.
_TOWARD: dict[str, tuple[int, int]] = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}


def _lines(arrangement: Arrangement, action: str) -> list[list[Cell | None]] | None:
    """The rows or columns a push runs along, in the order it runs."""
    step = _TOWARD.get(str(action or "").strip().lower())
    if step is None or not arrangement.rows or not arrangement.columns:
        return None
    down, across = step
    if across:
        lines = [list(arrangement.row_at(row)) for row in range(arrangement.rows)]
        return [line[::-1] if across > 0 else line for line in lines]
    lines = [list(arrangement.column_at(column)) for column in range(arrangement.columns)]
    return [line[::-1] if down > 0 else line for line in lines]


def _rebuild(
    arrangement: Arrangement, action: str, lines: list[list[str | None]]
) -> Arrangement:
    """Put worked-on lines back where they came from."""
    down, across = _TOWARD[action]
    cells: list[Cell] = []
    for index, line in enumerate(lines):
        restored = line[::-1] if (across > 0 or down > 0) else line
        for position, said in enumerate(restored):
            if said is None:
                continue
            row, column = (index, position) if across else (position, index)
            cells.append(Cell(row=row, column=column, says=said, at=(0.0, 0.0)))
    return Arrangement(rows=arrangement.rows, columns=arrangement.columns, cells=tuple(cells))


def _packed(line: Sequence[Cell | None]) -> list[str]:
    return [cell.says for cell in line if cell is not None]


def unchanged(arrangement: Arrangement, action: str) -> Arrangement | None:
    """The push did nothing. A real answer, and often the right one."""
    if _TOWARD.get(str(action or "").strip().lower()) is None:
        return None
    return arrangement


def shifted(arrangement: Arrangement, action: str) -> Arrangement | None:
    """Everything slides as far as it can go that way, and nothing else."""
    lines = _lines(arrangement, action)
    if lines is None:
        return None
    width = len(lines[0]) if lines else 0
    moved: list[list[str | None]] = []
    for line in lines:
        said = _packed(line)
        moved.append(said + [None] * (width - len(said)))
    return _rebuild(arrangement, str(action).strip().lower(), moved)


def shifted_and_combined(arrangement: Arrangement, action: str) -> Arrangement | None:
    """Things slide, and two equal neighbours become one thing worth both.

    What "worth both" means is read off the things themselves: where they are
    numbers, the pair becomes their sum. Where they are not, a pair of equals
    becomes one of them, which is what stacking or de-duplicating looks like.
    """
    lines = _lines(arrangement, action)
    if lines is None:
        return None
    width = len(lines[0]) if lines else 0
    moved: list[list[str | None]] = []
    for line in lines:
        said = _packed(line)
        joined: list[str] = []
        index = 0
        while index < len(said):
            if index + 1 < len(said) and said[index] == said[index + 1]:
                joined.append(_worth_both(said[index]))
                index += 2
            else:
                joined.append(said[index])
                index += 1
        moved.append(joined + [None] * (width - len(joined)))
    return _rebuild(arrangement, str(action).strip().lower(), moved)


def _worth_both(said: str) -> str:
    """What two of these together come to."""
    try:
        value = float(said.replace(",", ""))
    except (ValueError, AttributeError):
        return said
    doubled = value * 2
    return f"{doubled:g}"


@dataclass(frozen=True)
class Rule:
    """One way a thing might answer to being pushed."""

    name: str
    apply: Callable[[Arrangement, str], Arrangement | None]


#: Every way this knows of. Ordered from the strongest claim to the weakest,
#: so a tie goes to the rule that said the most.
RULES: tuple[Rule, ...] = (
    Rule("slides and combines", shifted_and_combined),
    Rule("slides", shifted),
    Rule("does not move", unchanged),
)


@dataclass
class HowItMoves:
    """What she has worked out about the thing she is acting in.

    Kept per run rather than per world, because a rule that held on one thing
    is a guess about the next one and she should find out rather than assume.
    """

    right: dict[str, int] = field(default_factory=dict)
    tried: dict[str, int] = field(default_factory=dict)
    seen: int = 0
    #: The last few moves, so a rule that stops working is noticed.
    recent: list[tuple[str, frozenset[str]]] = field(default_factory=list)
    #: Pairs of readings that could not be compared at all.
    unreadable: int = 0
    #: Places that sit still and change what they say — a score, a clock, a
    #: count of moves. They answer to her, so they are inside the part of the
    #: screen that responds, and no rule about what her act MOVES can predict
    #: one. Left in the comparison they make every hypothesis wrong forever.
    counters: set[tuple[int, int]] = field(default_factory=set)
    #: How often each place has been seen holding something, and how often
    #: what it holds has changed.
    _seen_at: dict[tuple[int, int], int] = field(default_factory=dict)
    _changed_at: dict[tuple[int, int], int] = field(default_factory=dict)
    #: Places she has seen empty, which settles them as part of the thing.
    _a_place: set[tuple[int, int]] = field(default_factory=set)
    _looks: int = 0

    # ── learning ─────────────────────────────────────────────────────────

    def _note_counters(self, before: Arrangement, after: Arrangement) -> None:
        """Places that sit still and change what they say.

        A score, a clock, a count of moves. They answer to her — they change
        when she acts — so the part of the screen that responds to her
        includes them, and every rule about what her act MOVES is wrong about
        one every single time. LIVE 2026-08-26: seventeen readings watched,
        every hypothesis discredited, on a board she was reading correctly.

        Nothing here knows what a score is. What it knows is that a thing
        which never goes anywhere and keeps saying something different is not
        a thing her moves move.
        """
        self._looks += 1
        now = {(cell.row, cell.column): cell.says for cell in after.cells}
        # Somewhere she has seen empty is a place, and never furniture again.
        #
        # One glimpse settles it. Everything in the thing she is acting on is
        # empty sometimes — that is what makes it a place rather than a
        # readout — and nothing that surrounds it ever is.
        for row in range(after.rows):
            for column in range(after.columns):
                if (row, column) not in now:
                    self._a_place.add((row, column))
                    self.counters.discard((row, column))
        if not self._a_place:
            # Nothing has been empty yet, so there is nothing to tell apart.
            return
        # How long a place has to have gone without ever being empty before
        # that means something: as many looks as the thing has places.
        #
        # A tile can sit in a corner for six moves running. It cannot sit
        # there for as many moves as there are squares while everything else
        # moves around it — and where it can, it is furniture in every sense
        # that matters to a rule about movement.
        enough = max(STILL_ENOUGH_TO_JUDGE, after.places())
        for place in now:
            if place in self._a_place:
                continue
            self._seen_at[place] = self._seen_at.get(place, 0) + 1
            times = self._seen_at[place]
            if times < enough or self._looks < enough:
                continue
            if times / max(1, self._looks) < ALWAYS_THERE:
                continue
            if place in self.counters:
                continue
            self.counters.add(place)
            if True:
                # What she learned was about a different thing.
                #
                # Every observation until now compared a board with a score
                # stuck to it against rules about a board. Those are not
                # evidence about the board, and left in the counts they hold a
                # correct rule below the bar for as long as they outnumber
                # what came after.
                logger.info(
                    "the thing itself is %s, not what surrounds it — starting again",
                    f"{len(self.counters)} place(s) smaller",
                )
                self.right.clear()
                self.tried.clear()
                self.seen = 0
                self.recent.clear()


    def watched(self, before: Arrangement, action: str, after: Arrangement) -> None:
        """One of her own moves, and what it did."""
        if not before.cells and not after.cells:
            return
        if (before.rows, before.columns) != (after.rows, after.columns):
            # Two readings that disagree about the shape of the thing cannot
            # be compared, and a comparison that cannot be made is not
            # evidence against anything. Scored as a miss, one reading that
            # dropped a faint tile discredits every hypothesis at once and the
            # model never forms. LIVE 2026-08-26: nineteen moves in, nothing
            # worked out, on a board she was reading perfectly well.
            self.unreadable += 1
            return
        self._note_counters(before, after)
        # The part that behaves like one thing, with its furniture cropped out.
        here, there = self.the_thing(before), self.the_thing(after)
        if not here.cells or (here.rows, here.columns) != (there.rows, there.columns):
            self.unreadable += 1
            return
        agreed: set[str] = set()
        for rule in RULES:
            predicted = rule.apply(here, action)
            if predicted is None:
                continue
            self.tried[rule.name] = self.tried.get(rule.name, 0) + 1
            if _near_enough(predicted, there):
                self.right[rule.name] = self.right.get(rule.name, 0) + 1
                agreed.add(rule.name)
        self.seen += 1
        self.recent.append((str(action), frozenset(agreed)))
        del self.recent[:-REMEMBERED]

    # ── using it ─────────────────────────────────────────────────────────

    def rule(self) -> Rule | None:
        """The one that has been right most often, once there is enough to say."""
        if self.seen < ENOUGH_TO_TRUST:
            return None
        best: tuple[float, Rule] | None = None
        for rule in RULES:
            tried = self.tried.get(rule.name, 0)
            if tried < ENOUGH_TO_TRUST:
                continue
            share = self.right.get(rule.name, 0) / tried
            if share < OFTEN_ENOUGH:
                continue
            if best is None or share > best[0]:
                best = (share, rule)
        return best[1] if best else None

    def confidence(self) -> float:
        """How often the rule she is using has been right."""
        rule = self.rule()
        if rule is None:
            return 0.0
        tried = self.tried.get(rule.name, 0)
        return (self.right.get(rule.name, 0) / tried) if tried else 0.0

    def the_thing(self, arrangement: Arrangement) -> Arrangement:
        """The part of a reading that behaves like one thing.

        Furniture surrounds a thing; it is not most of it. Where what has
        never been empty is most of what is there, nothing has been told apart
        yet and cropping would leave a handful of cells that every rule agrees
        about for want of anything to disagree over.
        """
        if not self.counters or len(self.counters) * 2 >= max(1, arrangement.occupied()):
            return arrangement
        return arrangement.without(self.counters)

    def expect(self, arrangement: Arrangement, action: str) -> Arrangement | None:
        """What this thing would look like after that, without doing it.

        ``None`` when she has not worked out how it moves, which is an answer
        and not a failure: she acts and looks instead.
        """
        rule = self.rule()
        if rule is None:
            return None
        return rule.apply(self.the_thing(arrangement), action)

    def expect_all(
        self, arrangement: Arrangement, actions: Sequence[str]
    ) -> dict[str, Arrangement]:
        """Every way this could go from here, for the moves really available."""
        futures: dict[str, Arrangement] = {}
        for action in actions:
            imagined = self.expect(arrangement, action)
            if imagined is not None:
                futures[action] = imagined
        return futures

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        """What she worked out, in a form that survives the process."""
        return {"right": dict(self.right), "tried": dict(self.tried), "seen": self.seen}

    @classmethod
    def from_memory(cls, held: dict[str, Any], trust: float = 1.0) -> "HowItMoves":
        """What she worked out last time, discounted.

        Under full trust on purpose. Something she worked out yesterday is
        evidence about today, not a fact about it, and a handful of moves that
        disagree should be able to overturn it — which they can only do if
        what came back is not overwhelming.
        """
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))

        def carried(counts: Any) -> dict[str, int]:
            if not isinstance(counts, dict):
                return {}
            return {
                str(name): int(round(float(value) * share))
                for name, value in counts.items()
                if isinstance(value, (int, float))
            }

        tried = carried(held.get("tried"))
        return cls(
            right=carried(held.get("right")),
            tried=tried,
            seen=int(round(float(held.get("seen") or 0) * share)),
        )

    def says(self) -> str:
        """What she has worked out, in a line, for whoever has to answer for it."""
        rule = self.rule()
        if rule is None:
            said = f"how this moves is not worked out yet ({self.seen} move(s) watched"
            return f"{said}, {self.unreadable} unreadable)" if self.unreadable else f"{said})"
        return f"this {rule.name} — right {self.confidence():.0%} of {self.tried.get(rule.name, 0)}"


def _near_enough(
    predicted: Arrangement, seen: Arrangement, ignoring: set[tuple[int, int]] | None = None
) -> bool:
    """Whether a rule got right the part it claimed to know about.

    A world that adds something of its own — a dealt tile, a new row, a clock
    ticking — has not falsified a rule about what her own act moves. What the
    rule must get right is every place it said something would be.
    """
    if predicted.rows != seen.rows or predicted.columns != seen.columns:
        return False
    skip = ignoring or set()
    claimed = {
        (cell.row, cell.column): cell.says
        for cell in predicted.cells
        if (cell.row, cell.column) not in skip
    }
    actual = {(cell.row, cell.column): cell.says for cell in seen.cells}
    for place, said in claimed.items():
        if actual.get(place) != said:
            return False
    # And nothing may vanish that the rule kept: an extra arrival is the
    # world's business, a disappearance would be the rule's mistake.
    return True


#: The least number of looks before anything can be called furniture, for a
#: thing so small that its own size would be a lower bar.
STILL_ENOUGH_TO_JUDGE = 6

#: How often a place has to be occupied before it counts as never empty. Not
#: quite one, because a reading can drop something faint for a frame and that
#: is not the same as the place being free.
ALWAYS_THERE = 0.95
