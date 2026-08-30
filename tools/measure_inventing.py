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


def run(games: int, weight: float, most: int = 3) -> None:
    import functools

    say = functools.partial(print, flush=True)
    say("Taking the property a person added back out, and asking whether she "
          "could have found it.\n")

    hg.AS_GOOD_A_GUESS_AS_ANY = without_the_property()
    hg.SMOOTHNESS_MATTERS = 0.0
    remember = WhatICannotExplain()
    blind = [played(seed, remember=remember) for seed in range(games)]
    say(f"played {games} games without it: median best tile "
          f"{statistics.median(best for best, _ in blind):.0f}, "
          f"median total {statistics.median(total for _, total in blind):.0f}")
    say(f"situations remembered: {len(remember.lived)}")
    pairs = remember.unexplained()
    say(f"pairs her measure calls equal that did not turn out equal: {len(pairs)}\n")

    shortlist = remember.worth_trying(most=most)
    if not shortlist:
        say("She found nothing that explains them. No property is promoted.")
        return
    say(f"{len(shortlist)} propert(ies) worth testing, best explanation first:")
    for measure, held_back, counted in shortlist:
        say(f"  {measure.name!r} — agrees on {held_back:.0%} of {counted // 2} "
            f"pairs it was not chosen on")
    say("")

    # The only test that decides anything. Explaining what already happened
    # and improving what happens next are different things: measured
    # 2026-08-29, the best explanation of her own unexplained pairs agreed
    # with the outcome 98% of the time and made her play 10% WORSE.
    before = statistics.median(total for _, total in blind)
    best: tuple[str, float, float] | None = None
    for measure, _held_back, _counted in shortlist:
        hg.AS_GOOD_A_GUESS_AS_ANY = without_the_property()
        hg.INVENTED.clear()
        hg.promote(measure, weight)
        tried = [played(seed) for seed in range(games)]
        after = statistics.median(total for _, total in tried)
        change = (after - before) / before if before else 0.0
        say(f"played with {measure.name!r} at {weight}: median total {after:.0f} "
            f"({change:+.0%})")
        if best is None or change > best[2]:
            best = (measure.name, after, change)
        hg.forget(measure.name)

    say("")
    if best is None or best[2] <= 0.0:
        say(f"REFUSED: nothing she proposed played better than the {before:.0f} "
            f"she manages without it. Her measure is unchanged.")
        return
    kept = next(m for m, _h, _c in shortlist if m.name == best[0])
    hg.AS_GOOD_A_GUESS_AS_ANY = without_the_property()
    hg.promote(kept, weight)
    say(f"PROMOTED {best[0]!r}: median total {before:.0f} -> {best[1]:.0f} "
        f"({best[2]:+.0%}), and it is now one of the things she judges a "
        f"situation by.")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--games", type=int, default=5)
    ask.add_argument("--weight", type=float, default=0.4)
    ask.add_argument("--most", type=int, default=3)
    said = ask.parse_args()
    run(said.games, said.weight, said.most)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
