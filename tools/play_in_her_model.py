"""Play whole games in her own model of a world, with her own search and judging.

A measuring instrument, not a player: nothing here decides a move. It loads
what she carries about a world — the rule she worked out for what her acts do,
her record of what arrives between them, what she judges a situation by and
the properties she invented — and lets her own search play from a start until
the world stops answering, the way `core/agency/rehearsing_in_her_model` does
for one change at a time.

It answers the question a live run is too slow and too costly to answer: how
far does her judgement get in this world, and does a change to it help.

    python tools/play_in_her_model.py --world 2048-game --games 4 --depth 3
    python tools/play_in_her_model.py --world 2048-game --without-invented
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _memory(world: str) -> dict:
    from core.runtime.state_ownership import state_root

    kept = state_root() / "state" / "worlds" / f"{world}.json"
    raw = json.loads(kept.read_text(encoding="utf-8"))
    return raw.get("data") or raw.get("payload") or raw


def _a_start(rows: int, columns: int, seed: int, like: object | None):
    """An empty place with what the world usually brings at the start of a game."""
    from core.perception.what_is_there import Arrangement, Cell

    roll = random.Random(seed)
    spots = roll.sample([(r, c) for r in range(rows) for c in range(columns)], 2)
    cells = tuple(
        Cell(row=r, column=c, says=roll.choice(("2", "2", "2", "4")), at=(0.0, 0.0))
        for r, c in sorted(spots)
    )
    return Arrangement(rows=rows, columns=columns, cells=cells, places_seen=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world", default="2048-game")
    parser.add_argument(
        "--rules-world", default="",
        help="where her rule comes from when it is not this world's own; she borrows from the kind of world",
    )
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3, help="her search depth; 0 = her own clock")
    parser.add_argument("--budget", type=float, default=0.3, help="seconds of search per move when depth is 0")
    parser.add_argument("--moves", type=int, default=3000, help="an upper bound, not a target")
    parser.add_argument("--toward", default="2048")
    parser.add_argument("--without-invented", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--budget-by-room", action="store_true",
        help="think as the live loop does: --budget on an open board, rising to 2 s as it fills",
    )
    parser.add_argument(
        "--lean", type=float, default=0.0,
        help="add the live loop's lean on what the world could swing, from -1 (avoid) to 1 (seek)",
    )
    args = parser.parse_args()

    from core.agency.a_world_compiled import compiled
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, INVENTED, forget, promote
    from core.agency.inventing_a_measure import measure_named
    from core.agency.looking_ahead import at_the_worlds_mercy, look_ahead
    from core.skills.screen_pursuit_decision_branches import _how_long_to_think
    from core.agency.what_makes_it_good_here import WhatMakesItGoodHere
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_the_world_does import WhatTheWorldDoes
    from core.runtime.what_she_learned import TRUST_CARRIED_OVER

    knew = _memory(args.world)
    borrowed = _memory(args.rules_world) if args.rules_world else knew
    rules = HowItMoves.from_memory(borrowed.get("moves") or {}, TRUST_CARRIED_OVER)
    # Whichever record of what arrives has seen more: this world's own can be
    # an empty record that is still a record.
    own, lent = knew.get("world") or {}, borrowed.get("world") or {}
    richer = own if int(own.get("acts") or 0) >= int(lent.get("acts") or 0) else lent
    world = WhatTheWorldDoes.from_memory(richer, TRUST_CARRIED_OVER)
    print(f"her rule: {rules.says()}")
    matters = WhatMakesItGoodHere.from_memory(knew.get("matters") or {}, TRUST_CARRIED_OVER)
    # The properties she invented, put back the way a sitting puts them back:
    # the world's own record names them, and only the recipes can make them.
    from core.agency.what_she_invented import recall

    recall()
    for name, worth in (knew.get("judging") or {}).items():
        found = measure_named(str(name))
        if found is not None:
            promote(found, float(worth))
    if args.without_invented:
        for name in list(INVENTED):
            forget(name)
    weights = dict(matters.weights() or AS_GOOD_A_GUESS_AS_ANY)
    acts = ["up", "down", "left", "right"]
    print(f"judging by: {', '.join(f'{k}={v:.2f}' for k, v in sorted(weights.items()))}")
    print(f"invented in play: {sorted(INVENTED) or 'none'}")

    reached: list[float] = []
    for game in range(args.games):
        start = _a_start(4, 4, args.seed + game, None)
        made = compiled(rules, world, start, acts)
        if made is None:
            print("her model of this world does not compile; nothing to play in")
            return 2
        roll = random.Random(10_000 + args.seed + game)
        board = made.board(start)
        state = start
        furthest, began, moves = 0.0, time.monotonic(), 0
        for moves in range(1, args.moves + 1):
            budget = (
                _how_long_to_think(state, least=args.budget, most=2.0)
                if args.budget_by_room
                else args.budget
            )
            scores = look_ahead(
                rules, state, acts, toward=args.toward, world=world, weights=weights,
                depth=args.depth, budget_s=budget,
            )
            if not scores:
                break
            if args.lean:
                # The same adjustment the live loop makes, from the same measure.
                exposed = at_the_worlds_mercy(
                    rules, state, acts, toward=args.toward, world=world
                )
                worths = [value for value, _why in scores.values()]
                spread = max(worths) - min(worths)
                if exposed and spread > 0.0:
                    scores = {
                        name: (value + args.lean * spread * exposed.get(name, 0.0), why)
                        for name, (value, why) in scores.items()
                    }
            act = max(scores, key=lambda name: scores[name][0])
            after = made.act(board, act)
            if after == board:
                moving = [name for name in acts if made.act(board, name) != board]
                if not moving:
                    break
                after = made.act(board, moving[0])
            ways = made.replies(after)
            total = sum(share for _way, share in ways)
            landed, pick = ways[-1][0], roll.random() * total
            for way, share in ways:
                pick -= share
                if pick <= 0.0:
                    landed = way
                    break
            board = landed
            state = made.arrangement(board, like=start)
            numbers = [v for v in (state.numbers() or ()) if v is not None]
            furthest = max([furthest, *numbers])
            if args.toward and furthest >= float(args.toward):
                break
        reached.append(furthest)
        took = time.monotonic() - began
        print(f"game {game + 1}: reached {furthest:g} in {moves} moves ({took:.1f}s, {took / max(1, moves) * 1000:.0f} ms a move)")
    middle = math.exp(statistics.mean(math.log(v) for v in reached if v > 0)) if reached else 0.0
    print(f"reached {', '.join(f'{v:g}' for v in reached)}; typical {middle:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
