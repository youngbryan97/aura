#!/usr/bin/env python3
"""Put her in worlds she has never seen, each with its own goal, and count how often she gets there.

`measure_generality.py` asks whether the same machinery works out how a new
world moves. That is half of it. Working out the rule and then losing is not
competence, and a player whose judging was tuned on one world can model a
second one perfectly and still play it badly, because what makes a position
good there is different.

So each world here comes with something to reach, and the same stack plays it
from nothing: her rule learned by watching, what the world adds learned the
same way, her own search over her own model, and her own judging, which starts
with every term alike and is worked out for this world by playing it out in
her head. Nothing here is written for any one world, and nothing about any of
them reaches her except what her acts show her.

The worlds differ in the ways that break a player tuned on one of them:

    four by four            things slide and two the same become one
    three by three          the same rule with little room
    five by five            the same rule with a lot of room
    three by five           a board that is not square
    threes arrive           what arrives is not what she has seen before
    keys that say nothing   her acts are called w, a, s and d
    keys the other way      "left" pushes everything right
    one step at a time      things move one place, not as far as they can
    a sliding puzzle        nothing combines or arrives; the goal is a layout

    python tools/measure_getting_there.py --lives 3
    python tools/measure_getting_there.py --only "three by three" --judging even
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.perception.how_it_moves import composed  # noqa: E402
from core.perception.what_is_there import Arrangement, Cell  # noqa: E402

ARROWS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class AWorld:
    """Everything that makes one world: its size, its rule, what turns up in it and what she is after."""

    rows: int
    columns: int
    #: What each of her acts is called, and which way it really pushes.
    acts: dict[str, str]
    #: A thing to make, by value, or a layout to make, written as rows.
    goal: float | str
    #: What turns up after an act that changed something, and how often each does.
    arrives: tuple[tuple[str, float], ...] = (("2", 0.9), ("4", 0.1))
    rule: Callable[[Arrangement, str], Arrangement | None] = field(
        default=composed("all the way", True, "everything").apply
    )
    #: Where a layout is the goal, how many random acts away from it a life
    #: starts. Every start is then one the world's own acts can undo.
    scrambled: int = 0

    @property
    def toward(self) -> str:
        return self.goal if isinstance(self.goal, str) else f"{self.goal:g}"

    def act(self, state: Arrangement, name: str) -> Arrangement:
        return self.rule(state, self.acts[name]) or state

    def something_turns_up(self, state: Arrangement, roll: random.Random) -> Arrangement:
        if not self.arrives:
            return state
        free = [
            (r, c) for r in range(state.rows) for c in range(state.columns) if state.at(r, c) is None
        ]
        if not free:
            return state
        row, column = roll.choice(free)
        pick, landed = roll.random(), self.arrives[-1][0]
        for said, share in self.arrives:
            pick -= share
            if pick <= 0.0:
                landed = said
                break
        return Arrangement(
            state.rows, state.columns, state.cells + (Cell(row, column, landed, (0.0, 0.0)),),
            places_seen=True,
        )

    def start(self, roll: random.Random) -> Arrangement:
        if isinstance(self.goal, str):
            rows = [line.split() for line in self.goal.split("/")]
            state = Arrangement(
                self.rows, self.columns,
                tuple(
                    Cell(r, c, said, (0.0, 0.0))
                    for r, line in enumerate(rows)
                    for c, said in enumerate(line)
                    if said != "_"
                ),
                places_seen=True,
            )
            names = sorted(self.acts)
            for _ in range(self.scrambled):
                state = self.act(state, roll.choice(names))
            return state
        empty = Arrangement(self.rows, self.columns, (), places_seen=True)
        return self.something_turns_up(self.something_turns_up(empty, roll), roll)

    def over(self, state: Arrangement) -> bool:
        return all(self.act(state, name).as_text() == state.as_text() for name in self.acts)


_SAME = {name: name for name in ARROWS}

#: Each goal is near the edge of what the rule and the room allow, because a
#: goal every way of judging reaches tells nothing about the judging.
WORLDS: dict[str, AWorld] = {
    "four by four": AWorld(4, 4, _SAME, goal=2048),
    "three by three": AWorld(3, 3, _SAME, goal=512),
    "five by five": AWorld(5, 5, _SAME, goal=4096),
    "three by five": AWorld(3, 5, _SAME, goal=1024),
    "threes arrive": AWorld(4, 4, _SAME, goal=3072, arrives=(("3", 0.9), ("6", 0.1))),
    "keys that say nothing": AWorld(
        4, 4, {"w": "up", "s": "down", "a": "left", "d": "right"}, goal=2048
    ),
    "keys the other way": AWorld(
        4, 4, {"left": "right", "right": "left", "up": "down", "down": "up"}, goal=2048
    ),
    "one step at a time": AWorld(
        4, 4, _SAME, goal=2048, rule=composed("one place", True, "everything").apply
    ),
    # Nothing combines, nothing arrives, nothing is a number worth reaching:
    # the goal is where things are.
    "a sliding puzzle": AWorld(
        3, 3, _SAME, goal="1 2 3 / 4 5 6 / 7 8 _", arrives=(), scrambled=40,
        rule=composed("one place", False, "one thing").apply,
    ),
}


def live_in(
    world: AWorld,
    *,
    seed: int,
    judging: str = "her own",
    think_s: float = 0.05,
    most_moves: int = 4000,
    within_s: float = 120.0,
    carried: dict[str, float] | None = None,
    working_out_s: float = 600.0,
    depth: int = 0,
) -> dict[str, object]:
    """One life in one world, told nothing about it but the names of her acts and what to reach.

    ``carried`` is what she worked out matters here in an earlier life, the
    way her memory of a world carries it into the next game. Her rule and her
    record of what arrives are learned again every life, so a world she
    cannot work out from scratch is not hidden behind what she remembers.

    ``depth`` holds her search at that many moves ahead instead of as far as
    ``think_s`` allows, so a result does not depend on how busy the machine is.
    """
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY
    from core.agency.looking_ahead import look_ahead
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_the_world_does import WhatTheWorldDoes

    roll = random.Random(seed)
    knows = HowItMoves()
    arrivals = WhatTheWorldDoes()
    names = list(world.acts)
    can_do = WhatWorksHere(told=tuple(names))
    from core.agency.what_she_is_after import goal_in

    toward = world.toward
    goal = goal_in(toward)
    weights = (
        {name: 1.0 for name in AS_GOOD_A_GUESS_AS_ANY}
        if judging in ("even", "her own")
        else dict(AS_GOOD_A_GUESS_AS_ANY)
    )
    if carried:
        weights = dict(carried)
    worked_out_at = -1 if carried or judging != "her own" else 0
    working_out: dict[str, object] = {}
    state = world.start(roll)
    furthest, nearest, searched, moves = 0.0, 0.0, 0, 0
    #: Every board she has stood on, as her search reads a board.
    been: set[str] = set()
    got_there = False
    began = time.monotonic()
    while moves < most_moves and time.monotonic() - began < within_s:
        furthest = max([furthest, *state.numbers()])
        nearest = max(nearest, goal.nearness(state))
        got_there = goal.reached(state)
        if got_there or world.over(state):
            break
        moves += 1
        if not worked_out_at:
            worked_out_at = _work_out_what_matters(
                knows, arrivals, state, names, toward, weights, working_out_s, working_out
            )
            if worked_out_at:
                began += float(working_out.get("took", 0.0))
        options = list(can_do.available() or names)
        been.add(state.as_text())
        ahead = look_ahead(
            knows, state, options, toward=toward, budget_s=think_s, weights=weights,
            world=arrivals if arrivals.worth_expecting() else None, depth=depth,
            been_before=been,
        )
        if ahead:
            searched += 1
            move = max(ahead, key=lambda name: ahead[name][0])
        else:
            move = roll.choice(options)
        after = world.act(state, move)
        changed = after.as_text() != state.as_text()
        can_do.tried(move, changed)
        knows.watched(state, move, after)
        if changed:
            after = world.something_turns_up(after, roll)
            arrivals.watched(knows.expect(state, move), after)
        state = after
    else:
        got_there = goal.reached(state)
        nearest = max(nearest, goal.nearness(state))
    rule = knows.rule()
    return {
        "got there": got_there,
        "furthest": furthest,
        "nearest": nearest,
        "moves": moves,
        "searched": searched,
        "rule": rule.name if rule is not None else "",
        "took": time.monotonic() - began,
        "weights": dict(weights),
        "worked out at": worked_out_at,
        "working out": working_out.get("said", ""),
    }


def _work_out_what_matters(
    knows: object, arrivals: object, state: Arrangement, names: list[str], toward: str,
    weights: dict[str, float], within_s: float, said: dict[str, object],
) -> int:
    """Once there is a model to play in, what matters here, by playing it out. The move it happened on, or nought."""
    from core.agency.working_out_what_matters import work_out_what_matters

    rule_of = getattr(knows, "rule", None)
    if not callable(rule_of) or rule_of() is None or not arrivals.worth_expecting():
        return 0
    found = work_out_what_matters(
        knows, arrivals, state, names, weights=weights, toward=toward, within_s=within_s
    )
    if found is None:
        return 0
    weights.clear()
    weights.update(found.weights)
    said.update(took=found.took_s, said=found.says())
    return max(1, int(getattr(knows, "seen", 0) or 1))


def measure(
    lives: int, judging: str, only: str, think_s: float, within_s: float, working_out_s: float
) -> None:
    print(f"{lives} lives in each world, judging: {judging}, {think_s:g}s a move\n", flush=True)
    print(f"{'world':<22} {'goal':>6} {'got there':>10} {'nearest (typical)':>19} {'moves':>6}  rule")
    print("-" * 96, flush=True)
    for name, world in WORLDS.items():
        if only and name != only:
            continue
        lived: list[dict[str, object]] = []
        carried: dict[str, float] | None = None
        for seed in range(lives):
            run = live_in(
                world, seed=seed, judging=judging, think_s=think_s, within_s=within_s,
                carried=carried, working_out_s=working_out_s,
            )
            lived.append(run)
            if run.get("working out"):
                print(f"{'':<22} {run['working out']}", flush=True)
                carried = dict(run["weights"])
        got = sum(1 for run in lived if run["got there"])
        typical = statistics.median(float(run["nearest"]) for run in lived)
        moves = statistics.median(int(run["moves"]) for run in lived)
        rules = statistics.mode(str(run["rule"]) or "none" for run in lived)
        print(
            f"{name:<22} {world.toward[:6]:>6} {got:>5}/{lives:<4} {typical:>19.2f} {moves:>6.0f}  {rules}",
            flush=True,
        )


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ask.add_argument("--lives", type=int, default=3)
    ask.add_argument("--judging", choices=("her own", "even", "standing"), default="her own")
    ask.add_argument("--only", default="")
    ask.add_argument("--think", type=float, default=0.05, help="seconds of search a move")
    ask.add_argument("--within", type=float, default=300.0, help="seconds a life may take")
    ask.add_argument(
        "--working-out", type=float, default=600.0,
        help="seconds she may spend playing a world out in her head before the first life goes on",
    )
    said = ask.parse_args()
    # What she tries while working out what matters, as she tries it.
    import logging

    trying = logging.getLogger("Aura.WorkingOutWhatMatters")
    trying.setLevel(logging.INFO)
    shown = logging.StreamHandler(sys.stdout)
    shown.setFormatter(logging.Formatter("        %(message)s"))
    trying.addHandler(shown)
    measure(said.lives, said.judging, said.only, said.think, said.within, said.working_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
