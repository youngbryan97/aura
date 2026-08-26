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
from typing import Callable, Sequence

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

    # ── learning ─────────────────────────────────────────────────────────

    def watched(self, before: Arrangement, action: str, after: Arrangement) -> None:
        """One of her own moves, and what it did."""
        if not before.cells and not after.cells:
            return
        agreed: set[str] = set()
        for rule in RULES:
            predicted = rule.apply(before, action)
            if predicted is None:
                continue
            self.tried[rule.name] = self.tried.get(rule.name, 0) + 1
            if _near_enough(predicted, after):
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

    def expect(self, arrangement: Arrangement, action: str) -> Arrangement | None:
        """What this thing would look like after that, without doing it.

        ``None`` when she has not worked out how it moves, which is an answer
        and not a failure: she acts and looks instead.
        """
        rule = self.rule()
        if rule is None:
            return None
        return rule.apply(arrangement, action)

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

    def says(self) -> str:
        """What she has worked out, in a line, for whoever has to answer for it."""
        rule = self.rule()
        if rule is None:
            return f"how this moves is not worked out yet ({self.seen} move(s) watched)"
        return f"this {rule.name} — right {self.confidence():.0%} of {self.tried.get(rule.name, 0)}"


def _near_enough(predicted: Arrangement, seen: Arrangement) -> bool:
    """Whether a rule got right the part it claimed to know about.

    A world that adds something of its own — a dealt tile, a new row, a clock
    ticking — has not falsified a rule about what her own act moves. What the
    rule must get right is every place it said something would be.
    """
    if predicted.rows != seen.rows or predicted.columns != seen.columns:
        return False
    claimed = {(cell.row, cell.column): cell.says for cell in predicted.cells}
    actual = {(cell.row, cell.column): cell.says for cell in seen.cells}
    for place, said in claimed.items():
        if actual.get(place) != said:
            return False
    # And nothing may vanish that the rule kept: an extra arrival is the
    # world's business, a disappearance would be the rule's mistake.
    return True
