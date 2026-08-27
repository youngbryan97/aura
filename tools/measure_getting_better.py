#!/usr/bin/env python3
"""Does she get better at a world by having been in it before?

Everything she works out about a world is kept and brought back discounted:
how it moves, which of her acts do anything, what the world does on its own,
which move suits a kind of position, and the line that worked. Each of those
was built and tested on its own. What none of them proves is the thing they
were all built for — that the fortieth run in a world starts better than the
first.

So: consecutive lives in one world, carrying everything forward, against the
same lives carrying nothing. Same world, same seeds, same everything else.

    python tools/measure_getting_better.py

Measured 2026-08-27, eight lives, up to 3000 moves each:

    starting fresh each time      first half 1881 → second half 1622
    carrying what she worked out  first half 2092 → second half 2736

Carrying it, she reached a 2048 tile twice and the share of moves answered on
sight — recognised rather than decided — went from 12% to 26%. Starting fresh,
neither happened.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agency import looking_ahead as _looking  # noqa: E402
from core.agency.looking_ahead import look_ahead  # noqa: E402
from core.agency.what_i_can_do_here import WhatWorksHere  # noqa: E402
from core.agency.what_worked_before import WhatWorkedBefore  # noqa: E402
from core.perception.how_it_moves import HowItMoves, shifted_and_combined  # noqa: E402
from core.perception.what_the_world_does import WhatTheWorldDoes  # noqa: E402
from core.runtime.what_she_learned import TRUST_CARRIED_OVER  # noqa: E402
from core.skills.screen_pursuit import A_LINE_HERE  # noqa: E402
from tools.measure_choosing import MOVES, arrives, empty_board, stuck  # noqa: E402

#: What a level of looking is taken to cost here, so both arms search to the
#: same depth. Close to what it really costs on this machine when nothing else
#: is running; the point is that it is the same in both.
A_LEVEL_COSTS = 0.008

#: A line to start from, so the first life is not deciding one from nothing.
#: What she carries forward replaces it as soon as one has earned its place.
LINE = "keep the largest in the bottom-left corner"


def a_life(
    seed: int,
    *,
    knows: HowItMoves,
    world: WhatTheWorldDoes,
    can_do: WhatWorksHere,
    skilled: WhatWorkedBefore,
    lines: WhatWorkedBefore,
    moves: int,
) -> dict[str, float]:
    """One life, using and adding to whatever it was handed."""
    # Every life starts from the same assumption about what a level of looking
    # costs. Left to carry over, the depth she can afford tracks how busy the
    # machine is, and the two arms of the comparison end up searching to
    # different depths — which is a measurement of the load, not of carrying.
    _looking._A_LEVEL["seconds"] = A_LEVEL_COSTS
    rng = random.Random(seed)
    state = arrives(arrives(empty_board(), rng), rng)
    line = lines.suggests(A_LINE_HERE) or LINE
    played = 0
    on_sight = 0

    for _ in range(moves):
        if stuck(state, shifted_and_combined):
            break
        options = list(can_do.available() or MOVES)
        kind = state.as_shape()
        known = skilled.suggests(kind, tuple(options))
        ahead = look_ahead(
            knows, state, options, toward="2048", approach=line, world=world
        )
        if ahead:
            move = max(ahead.items(), key=lambda row: row[1][0])[0]
            if known and known == move:
                on_sight += 1
        elif known:
            move, on_sight = known, on_sight + 1
        else:
            move = rng.choice(options)

        after = shifted_and_combined(state, move) or state
        changed = after.as_text() != state.as_text()
        can_do.tried(move, changed)
        knows.watched(state, move, after)
        better = _better(state, after)
        skilled.learned(kind, move, better)
        lines.learned(A_LINE_HERE, line, better)
        if changed:
            dealt = arrives(after, rng)
            world.watched(after, dealt)
            after = dealt
        state = after
        played += 1

    numbers = state.numbers()
    return {
        "best": max(numbers) if numbers else 0.0,
        "total": sum(numbers),
        "moves": played,
        "on_sight": on_sight / played if played else 0.0,
    }


def _better(before, after) -> bool:
    from core.agency.how_good_is_this import how_good

    return how_good(after, toward="2048", approach=LINE) >= how_good(
        before, toward="2048", approach=LINE
    )


def live_through(lives: int, moves: int, *, carrying: bool) -> list[dict[str, float]]:
    """Consecutive lives in one world, carrying everything or nothing."""
    held: dict[str, object] = {}
    out: list[dict[str, float]] = []
    for life in range(lives):
        knows = HowItMoves.from_memory(held.get("moves") or {}, TRUST_CARRIED_OVER)
        world = WhatTheWorldDoes.from_memory(held.get("world") or {}, TRUST_CARRIED_OVER)
        can_do = WhatWorksHere.from_memory(held.get("acts") or {}, told=MOVES)
        skilled = WhatWorkedBefore.from_memory(held.get("skill") or {}, TRUST_CARRIED_OVER)
        lines = WhatWorkedBefore.from_memory(held.get("lines") or {}, TRUST_CARRIED_OVER)
        out.append(
            a_life(
                life,
                knows=knows, world=world, can_do=can_do,
                skilled=skilled, lines=lines, moves=moves,
            )
        )
        if carrying:
            held = {
                "moves": knows.as_memory(),
                "world": world.as_memory(),
                "acts": can_do.as_memory(),
                "skill": skilled.as_memory(),
                "lines": lines.as_memory(),
            }
    return out


def measure(lives: int, moves: int) -> None:
    print(f"{lives} consecutive lives in one world, up to {moves} moves each\n")
    for carrying in (False, True):
        lived = live_through(lives, moves, carrying=carrying)
        held = "carrying what she worked out" if carrying else "starting fresh each time"
        print(f"{held}")
        print(f"  {'life':>4} {'best':>7} {'total':>8} {'moves':>7} {'on sight':>9}")
        for at, run in enumerate(lived):
            print(
                f"  {at:>4} {run['best']:>7.0f} {run['total']:>8.0f} "
                f"{run['moves']:>7.0f} {run['on_sight']:>8.0%}"
            )
        half = max(1, len(lived) // 2)
        early = statistics.median(run["total"] for run in lived[:half])
        late = statistics.median(run["total"] for run in lived[half:])
        print(f"  first half {early:.0f} → second half {late:.0f}\n")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--lives", type=int, default=8)
    ask.add_argument("--moves", type=int, default=3000)
    said = ask.parse_args()
    measure(said.lives, said.moves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
