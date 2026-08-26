#!/usr/bin/env python3
"""Measure how well she chooses, against a null that chooses at random.

A score is not a number until something else has been scored the same way. So
this plays the same world with the same perception, the same transition model
and the same scoring she uses live, and changes only how the move is picked:

    random          the null. Any move that is available.
    her scoring     one level ahead, ranked by how good the result is.
    her line        the same, with an approach held — a line she has stated,
                    used as part of what makes a situation good.

Nothing here is a game. The world is a rule from core.perception.how_it_moves
plus something arriving after every act, which describes a board dealing a
tile, a queue gaining a job, or a page refreshing under her. Pass a different
rule and it measures the same three ways of choosing in a different world.

    python tools/measure_choosing.py --games 30 --moves 400
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agency.how_good_is_this import how_good  # noqa: E402
from core.agency.looking_ahead import look_ahead  # noqa: E402
from core.perception.how_it_moves import (  # noqa: E402
    HowItMoves,
    shifted_and_combined,
)
from core.perception.what_is_there import Arrangement, Cell  # noqa: E402

MOVES = ("up", "down", "left", "right")


def empty_board(size: int = 4) -> Arrangement:
    return Arrangement(rows=size, columns=size, cells=())


def arrives(state: Arrangement, rng: random.Random, says: str = "2") -> Arrangement:
    """What the world adds on its own after an act."""
    free = [
        (row, column)
        for row in range(state.rows)
        for column in range(state.columns)
        if state.at(row, column) is None
    ]
    if not free:
        return state
    row, column = rng.choice(free)
    return Arrangement(
        state.rows, state.columns, state.cells + (Cell(row, column, says, (0.0, 0.0)),)
    )


def stuck(state: Arrangement, rule: Callable) -> bool:
    """Nothing she can do changes anything, which is where a world ends."""
    return all(
        (rule(state, move) or state).as_text() == state.as_text() for move in MOVES
    )


def play(
    how: str,
    *,
    rule: Callable = shifted_and_combined,
    moves: int = 400,
    seed: int = 0,
    toward: str = "2048",
    approach: str = "",
) -> dict[str, float]:
    """One game, played the named way, in a world she is told nothing about."""
    rng = random.Random(seed)
    knows = HowItMoves()
    state = arrives(arrives(empty_board(), rng), rng)
    played = 0
    wasted = 0

    for _ in range(moves):
        if stuck(state, rule):
            break
        options = [m for m in MOVES if (rule(state, m) or state).as_text() != state.as_text()]
        if not options:
            break

        if how == "random":
            move = rng.choice(MOVES)
        else:
            line = approach if how == "her line" else ""
            ahead = look_ahead(knows, state, list(MOVES), toward=toward, approach=line)
            if ahead:
                move = max(ahead.items(), key=lambda row: row[1][0])[0]
            else:
                move = rng.choice(MOVES)

        after = rule(state, move) or state
        landed = after.as_text() != state.as_text()
        knows.watched(state, move, after)
        if not landed:
            wasted += 1
        else:
            after = arrives(after, rng)
        state = after
        played += 1

    numbers = state.numbers()
    return {
        "best": max(numbers) if numbers else 0.0,
        "sum": sum(numbers),
        "moves": played,
        "wasted": wasted,
        "knew": 1.0 if knows.rule() is not None else 0.0,
    }


def measure(games: int, moves: int, approach: str) -> None:
    ways = ("random", "her scoring", "her line")
    print(f"{games} games, up to {moves} moves each, same world and same perception\n")
    print(f"{'how she chose':<14} {'best tile':>11} {'total':>9} {'moves':>7} {'wasted':>8} {'learned':>8}")
    print("-" * 62)
    for how in ways:
        runs = [
            play(how, moves=moves, seed=seed, approach=approach) for seed in range(games)
        ]
        best = statistics.median(r["best"] for r in runs)
        top = max(r["best"] for r in runs)
        total = statistics.median(r["sum"] for r in runs)
        played = statistics.median(r["moves"] for r in runs)
        wasted = sum(r["wasted"] for r in runs) / max(1, sum(r["moves"] for r in runs))
        knew = sum(r["knew"] for r in runs) / len(runs)
        print(
            f"{how:<14} {best:>7.0f} (max {top:>4.0f}) {total:>8.0f} {played:>7.0f} "
            f"{wasted:>7.1%} {knew:>7.0%}"
        )


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--games", type=int, default=30)
    ask.add_argument("--moves", type=int, default=400)
    ask.add_argument(
        "--approach",
        default="keep the largest in the bottom-left corner",
        help="the line she states and holds, used as part of what makes a situation good",
    )
    said = ask.parse_args()
    measure(said.games, said.moves, said.approach)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
