"""Her model of a world, compiled, so a search over it runs at the speed of the world.

Looking ahead asks her rule what a situation becomes and her measure what that
is worth, one object at a time. Every question builds a new arrangement of
cells, finds each cell by scanning the list, and parses every number again.
Measured 2026-09-17 on a sliding board: one level of looking cost about a
tenth of a second, so the clock she was given bought one level, and a search
one move deep reaches a 256 tile. The same search three levels deep reaches
2048 in every one of four games. The judgement was right all along; it could
not afford to look.

A rule that acts along lines acts on each line on its own, and a line has few
states. So the rule is asked about each line the first time the search meets
it, and the answer is kept. A situation becomes a tuple of symbols, and what an
act makes of it is a lookup per line. The parts of her measure that are sums
over lines — whether things run in order, how near neighbours are in value —
are kept per line the same way.

Nothing is assumed about the world. The rule compiled is the rule she learned,
asked through its own ``apply``; the arrivals averaged over are the ones she
watched; the measure is her own terms with her own weights. Before a compiled
world is used, what it says each act does to the situation in front of her is
checked against the rule itself, and a world whose rule does not act line by
line is never compiled at all.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.perception.what_is_there import Arrangement, Cell

__all__ = ["CompiledWorld", "compiled", "search"]

#: The acts a line rule knows, and which way each runs.
_PUSHES = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}

#: A future less likely than this, along the whole line that reaches it, is not
#: worth the arithmetic. Averaging includes it at its share; below this its
#: share cannot move a choice.
_TOO_UNLIKELY = 1e-4

#: The most levels the clock will ever be asked for. Past this the world has
#: added more than any measure of the present can say anything about.
_DEEPEST = 8


@dataclass
class CompiledWorld:
    """A rule, the world's replies and her measure, over tuples of symbols."""

    rows: int
    columns: int
    rule: Any
    #: Symbol ids. Nought is an empty place.
    texts: list[str] = field(default_factory=lambda: [""])
    ids: dict[str, int] = field(default_factory=lambda: {"": 0})
    values: list[float | None] = field(default_factory=lambda: [None])
    moves: dict[str, dict[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)
    per_line: dict[tuple[int, ...], tuple[float, int, float, int]] = field(default_factory=dict)
    arrivals: tuple[tuple[int, float], ...] = ()
    how_often: float = 0.0
    line_places: dict[str, list[list[int]]] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)

    # ── symbols ──────────────────────────────────────────────────────────

    def symbol(self, said: str) -> int:
        text = str(said or "").strip()
        found = self.ids.get(text)
        if found is not None:
            return found
        self.ids[text] = len(self.texts)
        self.texts.append(text)
        self.values.append(Cell(0, 0, text, (0.0, 0.0)).number())
        return self.ids[text]

    def board(self, arrangement: Arrangement) -> tuple[int, ...]:
        places = [0] * (self.rows * self.columns)
        for cell in arrangement.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.column < self.columns:
                places[cell.row * self.columns + cell.column] = self.symbol(cell.says)
        return tuple(places)

    def arrangement(self, board: Sequence[int], like: Arrangement | None = None) -> Arrangement:
        cells = tuple(
            Cell(index // self.columns, index % self.columns, self.texts[symbol], (0.0, 0.0))
            for index, symbol in enumerate(board)
            if symbol
        )
        return Arrangement(
            self.rows,
            self.columns,
            cells,
            like.down_at if like is not None else (),
            like.across_at if like is not None else (),
        )

    # ── acting ───────────────────────────────────────────────────────────

    def _lines(self, action: str) -> list[list[int]]:
        known = self.line_places.get(action)
        if known is not None:
            return known
        down, _across = _PUSHES[action]
        if down == 0:
            made = [[row * self.columns + column for column in range(self.columns)] for row in range(self.rows)]
        else:
            made = [[row * self.columns + column for row in range(self.rows)] for column in range(self.columns)]
        self.line_places[action] = made
        return made

    def _line_after(self, action: str, line: tuple[int, ...]) -> tuple[int, ...]:
        table = self.moves.setdefault(action, {})
        known = table.get(line)
        if known is not None:
            return known
        down, _across = _PUSHES[action]
        length = len(line)
        if down == 0:
            alone = Arrangement(
                1, length,
                tuple(Cell(0, i, self.texts[s], (0.0, 0.0)) for i, s in enumerate(line) if s),
            )
        else:
            alone = Arrangement(
                length, 1,
                tuple(Cell(i, 0, self.texts[s], (0.0, 0.0)) for i, s in enumerate(line) if s),
            )
        after = self.rule.apply(alone, action)
        result = [0] * length
        if after is not None:
            for cell in after.cells:
                where = cell.column if down == 0 else cell.row
                if 0 <= where < length:
                    result[where] = self.symbol(cell.says)
        made = tuple(result)
        table[line] = made
        return made

    def act(self, board: tuple[int, ...], action: str) -> tuple[int, ...]:
        if action not in _PUSHES:
            return board
        after = list(board)
        for places in self._lines(action):
            line = tuple(board[i] for i in places)
            for index, symbol in zip(places, self._line_after(action, line), strict=False):
                after[index] = symbol
        return tuple(after)

    # ── judging ──────────────────────────────────────────────────────────

    def _line_terms(self, line: tuple[int, ...]) -> tuple[float, int, float, int]:
        """What one line contributes to order and to smoothness.

        The same arithmetic as the authored terms, done once per line: the
        share of its steps that run one way, whether it has steps at all, and
        how far apart its neighbours are in doublings, with how many pairs.
        """
        known = self.per_line.get(line)
        if known is not None:
            return known
        numbers = [self.values[s] for s in line if s and self.values[s] is not None]
        steps = [b - a for a, b in zip(numbers, numbers[1:], strict=False) if b != a]
        if len(numbers) > 1:
            up = sum(1 for step in steps if step > 0)
            order = (max(up, len(steps) - up) / len(steps)) if steps else 1.0
            ordered = 1
        else:
            order, ordered = 0.0, 0
        apart = 0.0
        pairs = 0
        for one, other in zip(numbers, numbers[1:], strict=False):
            if one and other and one > 0 and other > 0:
                apart += abs(math.log2(one) - math.log2(other))
                pairs += 1
        made = (order, ordered, apart, pairs)
        self.per_line[line] = made
        return made

    def terms(
        self,
        board: tuple[int, ...],
        *,
        toward: str,
        actions: Sequence[str],
        freedom: bool = True,
    ) -> dict[str, float]:
        """Her authored terms for a compiled situation. See how_good_is_this.

        ``freedom`` is left out when the caller is a search that works out
        where every act leads for itself.
        """
        target = self.targets.get(toward)
        if target is None:
            from core.agency.how_good_is_this import _target  # noqa: PLC0415

            target = _target(toward)
            self.targets[toward] = target
        order_sum = 0.0
        ordered = 0
        apart = 0.0
        pairs = 0
        for places in self._lines("left") + self._lines("up"):
            o, has, a, n = self._line_terms(tuple(board[i] for i in places))
            order_sum += o * has
            ordered += has
            apart += a
            pairs += n
        places = len(board)
        free = sum(1 for symbol in board if not symbol)
        numbers = [self.values[s] for s in board if s and self.values[s] is not None]
        biggest = max(numbers) if numbers else 0.0
        if target and biggest > 0:
            nearness = 1.0 if biggest >= target else max(0.0, min(1.0, math.log2(biggest) / math.log2(target)))
        else:
            nearness = 0.0
        said = {
            "nearness": nearness,
            "room": (free / places) if places else 0.0,
            "order": (order_sum / ordered) if ordered else 0.0,
            "smoothness": (1.0 / (1.0 + apart / pairs)) if pairs else 0.0,
        }
        if not freedom:
            return said
        reached: set[tuple[int, ...]] = set()
        stayed = False
        for action in actions:
            if action not in _PUSHES:
                stayed = True
                continue
            after = self.act(board, action)
            if after == board:
                stayed = True
            else:
                reached.add(after)
        options = len(reached) + (1 if stayed else 0)
        said["freedom"] = (len(reached) / options) if options else 0.0
        return said

    # ── the world's turn ─────────────────────────────────────────────────

    def replies(self, board: tuple[int, ...]) -> list[tuple[tuple[int, ...], float]]:
        """Every way the world might answer, with its share. All the room, not a sample."""
        room = [index for index, symbol in enumerate(board) if not symbol]
        if not room or not self.arrivals or self.how_often <= 0.0:
            return [(board, 1.0)]
        ways: list[tuple[tuple[int, ...], float]] = []
        each = self.how_often / len(room)
        for index in room:
            for symbol, share in self.arrivals:
                landed = list(board)
                landed[index] = symbol
                ways.append((tuple(landed), each * share))
        if self.how_often < 1.0:
            ways.append((board, 1.0 - self.how_often))
        return ways


def compiled(knows: Any, world: Any, state: Any, actions: Sequence[str]) -> CompiledWorld | None:
    """A compiled world for this rule and this situation, or None when it cannot be one.

    Only a rule that moves everything that can move, along lines, is a rule of
    lines. One that moves a single thing anywhere on the board decides which
    thing by looking at the whole of it, and compiling that line by line would
    be a different rule.
    """
    rule_of = getattr(knows, "rule", None)
    rule = rule_of() if callable(rule_of) else None
    if rule is None or not getattr(rule, "carries", "") or getattr(rule, "how_many", "") != "everything":
        return None
    if not isinstance(state, Arrangement) or state.rows < 1 or state.columns < 1:
        return None
    pushes = [action for action in actions if action in _PUSHES]
    if not pushes:
        return None
    made = CompiledWorld(rows=state.rows, columns=state.columns, rule=rule)
    here = made.board(state)
    # Checked against the rule itself, on what is in front of her.
    for action in pushes:
        expected = knows.expect(state, action)
        if expected is None:
            return None
        if made.arrangement(made.act(here, action)).as_text() != Arrangement(
            state.rows, state.columns, expected.cells
        ).as_text():
            return None
    worth = getattr(world, "worth_expecting", None)
    if callable(worth) and worth():
        made.how_often = float(world.how_often() or 0.0)
        made.arrivals = tuple(
            (made.symbol(said), float(share)) for said, share in world.what_arrives()[:4]
        )
    return made


class _OutOfTime(Exception):
    """A pass ran past what it was given."""


def search(
    world: CompiledWorld,
    state: Arrangement,
    actions: Sequence[str],
    *,
    budget_s: float,
    worth: Callable[[tuple[int, ...]], float],
    dead: float,
    fixed_depth: int = 0,
    no_deeper_than: int = 0,
) -> tuple[dict[str, tuple[float, tuple[int, ...]]], int]:
    """Every push available, scored by what it leads to, as deep as the clock allows.

    A move is worth what the situation is worth at the far end of the search:
    her best from each of the world's replies, averaged by how often each
    happens, and so on down. Only the far end is judged. Adding each level's
    worth to the next counts a situation that looks good early once for every
    level it appears at, and measured over the same sixteen games that was the
    difference between reaching 2048 in fifteen of them and in three of six.

    A situation she cannot leave is worth ``dead``, which is below anything a
    live situation can be worth, whatever else is true of it.

    Deepened one level at a time while a level can still finish, so what comes
    back is always a finished pass, and ``depth`` says how far it went.
    """
    started = time.monotonic()
    ends_at = started + max(0.0, float(budget_s))
    # A pass still running when the time is up is abandoned where it stands,
    # and the last finished pass is what she has.
    give_up_at = ends_at
    here = world.board(state)
    pushes = [action for action in actions if action in _PUSHES]
    ticks = [0]

    def best_from(board: tuple[int, ...], depth: int, likely: float, memo: dict) -> float:
        ticks[0] += 1
        if not ticks[0] % 16 and time.monotonic() > give_up_at:
            raise _OutOfTime
        key = (board, depth)
        known = memo.get(key)
        if known is not None:
            return known
        best: float | None = None
        for action in pushes:
            after = world.act(board, action)
            if after == board:
                continue
            value = what_it_leads_to(after, depth, likely, memo)
            best = value if best is None or value > best else best
        found = dead if best is None else best
        memo[key] = found
        return found

    def what_it_leads_to(after: tuple[int, ...], depth: int, likely: float, memo: dict) -> float:
        if depth <= 1:
            return worth(after)
        total = 0.0
        counted = 0.0
        for way, share in world.replies(after):
            if likely * share < _TOO_UNLIKELY:
                continue
            total += share * best_from(way, depth - 1, likely * share, memo)
            counted += share
        if counted <= 0.0:
            return worth(after)
        # What was too unlikely to follow is left out of the average rather
        # than counted as nothing.
        return total / counted

    scored: dict[str, tuple[float, tuple[int, ...]]] = {}
    finished = 0
    best_before = best_before_that = ""
    # As deep as the clock allows, and never deeper than her model of this
    # world has been measured to carry: past that, a level is fiction that
    # costs a level's time and looks surer for being deeper.
    deepest = max(1, int(fixed_depth)) if fixed_depth else _DEEPEST
    if no_deeper_than and not fixed_depth:
        deepest = min(deepest, max(1, int(no_deeper_than)))
    depth = max(1, int(fixed_depth)) if fixed_depth else 1
    while depth <= deepest:
        pass_began = time.monotonic()
        memo: dict = {}
        this_pass: dict[str, tuple[float, tuple[int, ...]]] = {}
        try:
            for action in pushes:
                # Between one act and the next, whatever happened inside the
                # last one. A pass that prunes hard reaches the count-based
                # check rarely, and a level meant to take three tenths of a
                # second ran for one and a quarter (live, 2026-09-17).
                if time.monotonic() > give_up_at:
                    raise _OutOfTime
                after = world.act(here, action)
                if after == here:
                    continue
                this_pass[action] = (what_it_leads_to(after, depth, 1.0, memo), after)
        except _OutOfTime:
            this_pass = {}
        if not this_pass:
            break
        # Enough, once the answer has stopped changing. The best move the
        # same at two depths running, from three on, is what a further level
        # would most likely say again; in real time the world is waiting on
        # it. Measured 2026-09-17: more time per move turned six wins in
        # eight into seven, and most of that time bought the move already
        # chosen.
        best_now = max(this_pass, key=lambda action: this_pass[action][0])
        settled = depth >= 3 and best_now == best_before == best_before_that
        best_before_that, best_before = best_before, best_now
        scored = this_pass
        finished = depth
        if fixed_depth or settled:
            break
        # A level is started while any time remains and abandoned when it
        # runs out, rather than refused because it looks too expensive to
        # finish. Guessing what the next level costs from what the last one
        # did left five sixths of the time unspent — 47 ms of 300, a third of
        # her moves decided one level shallower than the clock allowed — and
        # the time she saves is time she spends waiting for the world.
        # Nothing is lost when a level does not finish: what comes back is
        # the last one that did.
        if time.monotonic() >= ends_at:
            break
        depth += 1
    return scored, finished
