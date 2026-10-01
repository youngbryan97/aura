"""Does a world she solved lend her its way of moving in a world shaped like it?

Offline and deterministic. Nothing here touches a screen or a model: worlds are
simulated by the same rule space she learns in (`core.perception.how_it_moves`),
and she learns in them through the same learner she plays with.

One world is solved first: a four-by-four board of 2s where everything slides
and equals combine. Then three she has never met:

* **held out** — five by five, 3s instead of 2s, acts with names that say no
  direction. It moves exactly like the solved one.
* **distractor** — four by four, letters, one thing steps at a time, nothing
  combines. Shaped differently.
* **misleading** — five by five, 3s, looks like the held-out one at first
  sight, but things step one place instead of sliding. The solved world lends
  the wrong rule, and her own moves have to overturn it.

For each: whether the solved world clears its scrambled control, and how many
of her moves it takes to settle on the true rule with what is lent and
without it.

    python tools/check_the_world_it_is_most_like.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agency.the_world_it_is_most_like import (  # noqa: E402
    graph_from_memory,
    graph_to_memory,
    lend_to,
    lent,
    most_like,
    shape_of,
)
from core.perception.how_it_moves import HowItMoves, composed  # noqa: E402
from core.perception.what_is_there import Arrangement, Cell  # noqa: E402

WAYS = ("left", "right", "up", "down")

WORLDS = {
    "solved": {"size": 4, "things": ("2", "4"), "carries": "all the way", "combines": True,
               "how_many": "everything", "acts": WAYS},
    "held out": {"size": 5, "things": ("3", "6"), "carries": "all the way", "combines": True,
                 "how_many": "everything", "acts": ("a", "d", "w", "s")},
    "distractor": {"size": 4, "things": ("A", "B"), "carries": "one place", "combines": False,
                   "how_many": "one thing", "acts": ("j", "l", "i", "k")},
    "misleading": {"size": 5, "things": ("3", "6"), "carries": "one place", "combines": True,
                   "how_many": "everything", "acts": ("a", "d", "w", "s")},
}


def _arrival(state: Arrangement, things: tuple[str, ...], rng: random.Random) -> Arrangement:
    held = {(cell.row, cell.column) for cell in state.cells}
    room = [
        (row, column)
        for row in range(state.rows)
        for column in range(state.columns)
        if (row, column) not in held
    ]
    if not room:
        return state
    row, column = rng.choice(room)
    said = things[0] if rng.random() < 0.9 else things[1]
    return Arrangement(
        state.rows, state.columns, state.cells + (Cell(row, column, said, (0.0, 0.0)),)
    )


def _a_start(name: str, rng: random.Random, crowded: bool) -> Arrangement:
    """An opening board: a couple of things, or a crowded board met mid-way."""
    world = WORLDS[name]
    size = world["size"]
    state = Arrangement(size, size, ())
    if not crowded:
        for _ in range(max(2, size // 2)):
            state = _arrival(state, world["things"], rng)
        return state
    first = world["things"][0]
    if first.isdigit():
        kinds = [str(int(first) * 2**power) for power in range(5)]
    else:
        kinds = [chr(ord(first) + offset) for offset in range(5)]
    cells = tuple(
        Cell(row, column, rng.choice(kinds), (0.0, 0.0))
        for row in range(size)
        for column in range(size)
        if rng.random() < 0.7
    )
    return Arrangement(size, size, cells)


def play(
    name: str, seed: int, moves: int, *, look_after: int = 0, crowded: bool = False,
    at_the_look: Callable[[HowItMoves, Arrangement], None] | None = None,
) -> tuple[HowItMoves, Arrangement, int | None]:
    """Play ``moves`` random acts; the learner, a look, and when the truth settled.

    ``at_the_look`` is called at the move the look is taken, with the learner
    as it stands, which is where the pursuit asks what this world is like.
    """
    world = WORLDS[name]
    truth = composed(world["carries"], world["combines"], world["how_many"])
    rng = random.Random(f"{name}:{seed}")
    state = _a_start(name, rng, crowded)
    learner = HowItMoves()
    look = state
    settled: int | None = None
    way_of = dict(zip(world["acts"], WAYS, strict=True))
    for move in range(moves):
        if move == look_after:
            look = state
            if at_the_look is not None:
                at_the_look(learner, state)
        act = rng.choice(world["acts"])
        after = truth.apply(state, way_of[act]) or state
        if after.as_text() != state.as_text():
            after = _arrival(after, world["things"], rng)
        learner.watched(state, act, after)
        held = learner.rule()
        if held is not None and held.name == truth.name:
            if settled is None:
                settled = move + 1
        else:
            settled = None
        state = after
        if not any(
            (truth.apply(state, way) or state).as_text() != state.as_text() for way in WAYS
        ):
            # A dead board: start it again, as a person would.
            state = _a_start(name, rng, crowded)
    return learner, look, settled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--moves", type=int, default=60)
    parser.add_argument("--look-after", type=int, default=6)
    parser.add_argument("--solved-moves", type=int, default=300)
    parser.add_argument("--crowded", action="store_true",
                        help="meet every world mid-way, on a crowded board")
    args = parser.parse_args()
    crowded = args.crowded

    solved, solved_look, _ = play(
        "solved", 0, args.solved_moves, look_after=args.look_after, crowded=crowded)
    knew = {
        "moves": solved.as_memory(),
        "shape": graph_to_memory(shape_of(solved_look, WORLDS["solved"]["acts"])),
    }
    distractor, distractor_look, _ = play(
        "distractor", 0, args.solved_moves, look_after=args.look_after, crowded=crowded)
    knew_distractor = {
        "moves": distractor.as_memory(),
        "shape": graph_to_memory(
            shape_of(distractor_look, WORLDS["distractor"]["acts"])),
    }
    library = {
        "solved": (graph_from_memory(knew["shape"], "solved"), knew),
        "distractor solved": (
            graph_from_memory(knew_distractor["shape"], "distractor solved"),
            knew_distractor,
        ),
    }
    report: dict[str, object] = {"solved_rule": solved.says(), "distractor_rule": distractor.says()}
    for name in ("held out", "distractor", "misleading"):
        chosen: dict[str, int] = {}
        cleared: list[float] = []
        scratch: list[int] = []
        helped: list[int] = []
        for seed in range(1, args.seeds + 1):
            _learner, _look, alone = play(
                name, seed, args.moves, look_after=args.look_after, crowded=crowded)
            picked: list[str] = []

            def lend_at_the_look(
                learner: HowItMoves, look: Arrangement, name: str = name,
                picked: list[str] = picked, cleared: list[float] = cleared,
            ) -> None:
                here = shape_of(look, WORLDS[name]["acts"])
                likeness = most_like(
                    here, {world: graph for world, (graph, _k) in library.items()})
                picked.append(likeness.world if likeness else "none")
                if likeness is not None:
                    cleared.append(likeness.score - likeness.scrambled_at)
                    lend_to(learner, lent(library[likeness.world][1], likeness))

            # The same world, the same seed, the same moves — only what she is
            # lent at the look differs.
            _l, _look, with_it = play(
                name, seed, args.moves, look_after=args.look_after, crowded=crowded,
                at_the_look=lend_at_the_look,
            )
            for world in picked:
                chosen[world] = chosen.get(world, 0) + 1
            scratch.append(alone if alone is not None else args.moves + 1)
            helped.append(with_it if with_it is not None else args.moves + 1)
        report[name] = {
            "chosen": chosen,
            "margin_over_scrambled_median": statistics.median(cleared) if cleared else None,
            "moves_to_the_true_rule_alone_median": statistics.median(scratch),
            "moves_to_the_true_rule_lent_median": statistics.median(helped),
            "lent_faster": sum(1 for a, b in zip(scratch, helped, strict=True) if b < a),
            "lent_slower": sum(1 for a, b in zip(scratch, helped, strict=True) if b > a),
            "never_settled_alone": sum(1 for a in scratch if a > args.moves),
            "never_settled_lent": sum(1 for b in helped if b > args.moves),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
