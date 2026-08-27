#!/usr/bin/env python3
"""Put the same machinery in several worlds and see what it works out.

A solver written for one thing is handed the environment, the state, the
actions, the transition rules, the noise model and the objective. The question
worth asking of a general one is not whether it can play the thing it was
tuned on. It is whether the same machinery, moved to something it has never
seen, discovers enough structure to become competent — without anyone writing
the solution for it.

So: several worlds that differ in shape rather than in subject, run through
the identical stack with nothing written for any of them.

    slides and combines   two equal neighbours become one thing worth both
    slides                things pack to one end and never merge
    still                 nothing she does moves anything
    swaps                 the two ends of every line exchange places
    one into the gap      a sliding-puzzle: one thing moves into the space

Four of these are compositions she can reach — not because anyone wrote them
down, but because they fall out of the three facts a push can turn on. The
fifth does not fall out of anything: exchanging two ends is not a distance, a
merge or a count, and no composition of those describes it. Refusing that one
honestly — saying how this moves is not worked out yet, and going back to
acting and looking — is a PASS. Claiming a rule that does not hold would be
the failure, and it is called out below when it happens.

    python tools/measure_generality.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agency.looking_ahead import look_ahead  # noqa: E402
from core.agency.what_i_can_do_here import WhatWorksHere  # noqa: E402
from core.agency.what_kind_of_problem import recognise  # noqa: E402
from core.perception.how_it_moves import (  # noqa: E402
    HowItMoves,
    shifted,
    shifted_and_combined,
)
from core.perception.what_is_there import Arrangement, Cell  # noqa: E402

MOVES = ("up", "down", "left", "right")


# ── the worlds, none of which she has been told anything about ───────────


def still(state: Arrangement, _action: str) -> Arrangement:
    """A world that does not answer at all."""
    return state


def swaps(state: Arrangement, action: str) -> Arrangement:
    """The two ends of every line exchange places."""
    across = action in ("left", "right")
    cells = []
    for cell in state.cells:
        if across and cell.column in (0, state.columns - 1):
            column = state.columns - 1 - cell.column
            cells.append(Cell(cell.row, column, cell.says, cell.at))
        elif not across and cell.row in (0, state.rows - 1):
            row = state.rows - 1 - cell.row
            cells.append(Cell(row, cell.column, cell.says, cell.at))
        else:
            cells.append(cell)
    return Arrangement(state.rows, state.columns, tuple(cells), state.down_at, state.across_at)


def one_into_the_gap(state: Arrangement, action: str) -> Arrangement:
    """A sliding puzzle: the one thing beside the space moves into it."""
    empty = [
        (r, c) for r in range(state.rows) for c in range(state.columns) if state.at(r, c) is None
    ]
    if not empty:
        return state
    row, column = empty[0]
    step = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}[action]
    from_row, from_column = row + step[0], column + step[1]
    moving = state.at(from_row, from_column)
    if moving is None:
        return state
    cells = [cell for cell in state.cells if cell is not moving]
    cells.append(Cell(row, column, moving.says, moving.at))
    return Arrangement(state.rows, state.columns, tuple(cells), state.down_at, state.across_at)


#: Each world, whether things arrive in it on their own, whether it starts
#: full, and whether she has a hypothesis that fits it.
WORLDS: dict[str, tuple[Callable, bool, bool, bool]] = {
    "slides and combines": (shifted_and_combined, True, False, True),
    "slides": (shifted, True, False, True),
    "still": (still, True, False, True),
    "swaps": (swaps, False, True, False),
    "one into the gap": (one_into_the_gap, False, True, True),
}


# ── the same machinery in each of them ───────────────────────────────────


def a_board(size: int = 4) -> Arrangement:
    return Arrangement(
        rows=size,
        columns=size,
        cells=(),
        down_at=tuple(0.2 + n * 0.15 for n in range(size)),
        across_at=tuple(0.2 + n * 0.15 for n in range(size)),
    )


def arrives(state: Arrangement, rng: random.Random) -> Arrangement:
    free = [
        (r, c) for r in range(state.rows) for c in range(state.columns) if state.at(r, c) is None
    ]
    if not free:
        return state
    r, c = rng.choice(free)
    return Arrangement(
        state.rows, state.columns, state.cells + (Cell(r, c, "2", (0.0, 0.0)),),
        state.down_at, state.across_at,
    )


def full_but_one(size: int = 4) -> Arrangement:
    """A board with one space in it, which is what a sliding puzzle is.

    Started with two things on sixteen places, almost nothing her acts do
    moves anything — and "does not move" is then a TRUE description of the
    world she met rather than a false claim about the one intended. A world
    has to be able to answer before refusing to model it means anything.
    """
    board = a_board(size)
    cells = tuple(
        Cell(row, column, str(row * size + column + 1), (0.0, 0.0))
        for row in range(size)
        for column in range(size)
        if (row, column) != (0, 0)
    )
    return Arrangement(size, size, cells, board.down_at, board.across_at)


def live_in(
    world: Callable,
    *,
    adds_things: bool,
    moves: int = 120,
    seed: int = 0,
    toward: str = "2048",
    starts_full: bool = False,
) -> dict[str, object]:
    """One life in one world, told nothing about it."""
    rng = random.Random(seed)
    knows = HowItMoves()
    can_do = WhatWorksHere(told=MOVES)
    state = full_but_one() if starts_full else arrives(arrives(a_board(), rng), rng)
    searched = 0

    for _ in range(moves):
        options = list(can_do.available() or MOVES)
        ahead = look_ahead(knows, state, options, toward=toward)
        if ahead:
            searched += 1
            move = max(ahead.items(), key=lambda row: row[1][0])[0]
        else:
            move = rng.choice(options)
        after = world(state, move) or state
        changed = after.as_text() != state.as_text()
        can_do.tried(move, changed)
        knows.watched(state, move, after)
        if changed and adds_things:
            after = arrives(after, rng)
        state = after

    suits = recognise(acts=options, knows_how_it_moves=knows, state=state, toward=toward)
    rule = knows.rule()
    return {
        "rule": rule.name if rule is not None else "",
        "confidence": knows.confidence(),
        "shape": suits.shape.named(),
        "searched": searched,
        "dead_acts": len(can_do.dead()),
        "best": max(state.numbers()) if state.numbers() else 0.0,
    }


def measure(lives: int, moves: int) -> None:
    print(f"{lives} lives in each world, {moves} moves each, told nothing about any of them\n")
    print(f"{'world':<21} {'what she worked out':<24} {'sure':>5} {'searched':>9}  what kind of thing")
    print("-" * 108)
    for name, (world, adds, full, covered) in WORLDS.items():
        lived = [
            live_in(world, adds_things=adds, starts_full=full, moves=moves, seed=seed)
            for seed in range(lives)
        ]
        rules = [str(run["rule"]) for run in lived]
        settled = statistics.mode(rules) if rules else ""
        sure = statistics.median(float(run["confidence"]) for run in lived)
        searched = statistics.median(int(run["searched"]) for run in lived)
        # The shape of a life that ended where most of them ended, rather
        # than of whichever happened to run first.
        typical = next(
            (run for run in lived if str(run["rule"]) == settled), lived[0]
        )
        worked = settled or "nothing — acts and looks"
        print(f"{name:<21} {worked:<24} {sure:>4.0%} {searched:>9}  {str(typical['shape'])[:44]}")
        if not covered and settled:
            print(f"{'':<21} ↑ claimed a rule for a shape it has no hypothesis for")
        if not covered and not settled:
            print(f"{'':<21} ✓ refused honestly, and went back to acting and looking")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--lives", type=int, default=5)
    ask.add_argument("--moves", type=int, default=120)
    said = ask.parse_args()
    measure(said.lives, said.moves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
