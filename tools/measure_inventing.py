#!/usr/bin/env python3
"""Can she find the property nobody told her about, from her own play?

On 2026-08-29 a person noticed that her measure of a good situation said
nothing about whether neighbouring things were close in value, wrote that
property's arithmetic, swept its weight, and admitted it to her evaluator. It
doubled her play. Good engineering, and not her doing any of it.

This asks the only question that matters about that: could she have done it?

    take the property back out of her evaluator
    let her play, remembering each situation and how the game went from there
    find the pairs her remaining measure calls equal whose outcomes were not
    search the space of measures she can compose for one that separates them
    check it on pairs it was not chosen on
    then play again WITH it, against playing without, and see which is better

The last step is the whole thing. A property that explains what already
happened is a story. One that improves the play is a finding, and nothing is
promoted without it.

    python tools/measure_inventing.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agency import how_good_is_this as hg  # noqa: E402
from core.agency.looking_ahead import look_ahead  # noqa: E402
from core.agency.what_i_cannot_explain import WhatICannotExplain  # noqa: E402
from core.perception.how_it_moves import HowItMoves, shifted_and_combined  # noqa: E402
from core.perception.what_the_world_does import WhatTheWorldDoes  # noqa: E402
from tools.measure_choosing import MOVES, arrives, empty_board, stuck  # noqa: E402

LINE = "keep the largest in the bottom-left corner"


def played(seed: int, *, moves: int = 4000, remember: WhatICannotExplain | None = None):
    """One game. Returns what it came to, and remembers what she saw."""
    rng = random.Random(seed)
    knows = HowItMoves()
    world = WhatTheWorldDoes()
    state = arrives(arrives(empty_board(), rng), rng)
    seen: list[tuple[object, float, float]] = []
    for _ in range(moves):
        if stuck(state, shifted_and_combined):
            break
        ahead = look_ahead(knows, state, list(MOVES), toward="2048", approach=LINE, world=world)
        move = max(ahead.items(), key=lambda row: row[1][0])[0] if ahead else rng.choice(MOVES)
        if remember is not None:
            here = sum(state.numbers() or (0.0,))
            seen.append((state, hg.how_good(state, toward="2048", approach=LINE), here))
        after = shifted_and_combined(state, move) or state
        knows.watched(state, move, after)
        if after.as_text() != state.as_text():
            dealt = arrives(after, rng)
            world.watched(after, dealt)
            after = dealt
        state = after
    ended_at = sum(state.numbers() or (0.0,))
    if remember is not None and ended_at > 0:
        # How it turned out FROM there: what the game went on to gain, as a
        # share of everything it ever came to. Credit where the situation was,
        # not where the game finished.
        for situation, scored, here in seen:
            remember.been_here(situation, scored, (ended_at - here) / ended_at)
    numbers = state.numbers()
    return (max(numbers) if numbers else 0.0), ended_at


def without_the_property() -> dict[str, float]:
    kept = dict(hg.AS_GOOD_A_GUESS_AS_ANY)
    kept.pop("smoothness", None)
    return kept


def run(games: int, weight: float) -> None:
    print("Taking the property a person added back out, and asking whether she "
          "could have found it.\n")

    hg.AS_GOOD_A_GUESS_AS_ANY = without_the_property()
    hg.SMOOTHNESS_MATTERS = 0.0
    remember = WhatICannotExplain()
    blind = [played(seed, remember=remember) for seed in range(games)]
    print(f"played {games} games without it: median best tile "
          f"{statistics.median(best for best, _ in blind):.0f}, "
          f"median total {statistics.median(total for _, total in blind):.0f}")
    print(f"situations remembered: {len(remember.lived)}")
    pairs = remember.unexplained()
    print(f"pairs her measure calls equal that did not turn out equal: {len(pairs)}\n")

    found = remember.what_would_explain()
    if found is None:
        print("She found nothing that explains them. No property is promoted.")
        return
    measure, held_back, counted = found
    print(f"what she proposes: {measure.name!r}")
    print(f"  agrees with the outcome on {held_back:.0%} of the pairs it was "
          f"NOT chosen on, out of {counted}\n")

    # The only test that decides anything.
    hg.AS_GOOD_A_GUESS_AS_ANY = without_the_property()
    hg.promote(measure, weight)
    with_it = [played(seed) for seed in range(games)]
    print(f"played {games} games with it at {weight}: median best tile "
          f"{statistics.median(best for best, _ in with_it):.0f}, "
          f"median total {statistics.median(total for _, total in with_it):.0f}")

    before = statistics.median(total for _, total in blind)
    after = statistics.median(total for _, total in with_it)
    better = (after - before) / before if before else 0.0
    if better <= 0:
        hg.forget(measure.name)
    print(f"\n{'PROMOTED' if better > 0 else 'REFUSED'}: including it changed the "
          f"median total by {better:+.0%}")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--games", type=int, default=5)
    ask.add_argument("--weight", type=float, default=0.4)
    said = ask.parse_args()
    run(said.games, said.weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
