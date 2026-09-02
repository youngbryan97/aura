#!/usr/bin/env python3
"""Does she write a better rule for what to try first — and does it hold on sealed tasks?

Experiment H, at the one part of the mechanism that is already an object of the
language it works on. `the_order_she_tries_them_in.THE_ORDER` is a floor term
taking five numbers and giving a score, and the search consults it to decide
which word to put in a hole first. Replacing it is the same call that replaces
a head.

What is measured
----------------
Not "the source changed". The order decides how far the search walks before it
reaches the word the answer is built from, so what a better order buys is a
lower rank for that word. `MetaCap` here is the mean rank of the winning word
across invention episodes, lower being better, and an episode is a family
together with the word its answer is built on.

The protocol
------------
Episodes are drawn at random and split before anything is synthesised. The new
order is written from the training half and scored on the sealed half, which it
never saw. Then it is lesioned — the authored order put back — and the gain has
to go with it.

Where the new order comes from
------------------------------
The same place a head comes from: shortest-first over the floor, judged by
whether it ranks the winning words higher than the order in force. No list of
scoring rules, and nothing is handed a target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from core.cognition.an_invented_kind import WHERE_FROM  # noqa: E402
from core.cognition.one_algebra import Made, Term, every_term, holes_in  # noqa: E402
from core.cognition.the_floor_she_stands_on import (  # noqa: E402
    Code,
    OutOfFuel,
    Stuck,
    every_code,
    how_long,
    run,
)
from core.cognition.the_order_she_tries_them_in import (  # noqa: E402
    THE_ORDER,
    forget_the_order,
    the_order_she_uses,
    the_order_she_wrote,
)


@dataclass
class Episode:
    """One invention: a family, and the word its answer is built on."""

    wanted: dict[int, tuple[int, ...]]
    winner: str
    #: How much each word already puts where it belongs, and how long it is.
    features: dict[str, tuple[int, int]]
    places: int


def _an_episode(rng: random.Random, terms: list[Term], names: list[str]) -> Episode | None:
    """A family whose answer is a term over two of her words.

    Two rather than one, and that is the difference between an experiment and
    a formality. Over one word, every word she has tells her the answer
    equally well — the feature the order reads is flat, so no order can
    separate them and there is nothing for a better one to do. Over two, the
    word that is actually inside the answer and the word that is not are
    separated by what they say about it, and an order can be right or wrong.
    """
    from core.cognition.one_algebra import _tells_her_the_answer  # noqa: PLC2701
    from core.cognition.what_it_costs_to_say import _symbols  # noqa: PLC2701

    term = rng.choice(terms)
    winner, beside = rng.sample(names, 2)
    words = (WHERE_FROM[winner], WHERE_FROM[beside])
    wanted: dict[int, tuple[int, ...]] = {}
    for size in (4, 5, 6, 7):
        here = []
        for at in range(size):
            try:
                here.append(int(Made(term=term, words=words)(at, size)) % size)
            except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
                return None
        if len(set(here)) < 2:
            return None
        wanted[size] = tuple(here)
    places = sum(len(one) for one in wanted.values())
    features: dict[str, tuple[int, int]] = {}
    for name, one in WHERE_FROM.items():
        try:
            told = int(_tells_her_the_answer(one, wanted))
        except (ArithmeticError, TypeError, ValueError):
            told = 0
        features[name] = (told, int(_symbols(name)))
    return Episode(wanted=wanted, winner=winner, features=features, places=places)


def _rank_of_the_winner(episode: Episode, order: Code) -> int:
    """Where the winning word sits under this order. Lower is better."""
    from core.cognition.how_she_learns_to_look import how_often_it_worked

    scored: list[tuple[int, int, str]] = []
    for name, (told, symbols) in episode.features.items():
        before = how_often_it_worked(name)
        try:
            work: Any = run(order, fuel=20_000)
            for one in (told, episode.places, before.won, before.of, symbols):
                work = run(work.body, (one, *work.env), fuel=20_000)
            said = int(work)
        except (OutOfFuel, Stuck, TypeError, ValueError, AttributeError):
            said = 0
        scored.append((-said, symbols, name))
    scored.sort()
    for at, (_score, _symbols, name) in enumerate(scored):
        if name == episode.winner:
            return at
    return len(scored)


def meta_capability(episodes: list[Episode], order: Code) -> float:
    """Mean rank of the winning word. Lower is a better order."""
    return statistics.mean(_rank_of_the_winner(one, order) for one in episodes)


def _an_order_she_writes(
    episodes: list[Episode], *, deepest: int, within: float
) -> Code | None:
    """Search the floor for an order that ranks the winners higher.

    Shortest first, over the five numbers an order is given, judged by the
    quantity the order exists to move. Nothing is handed a target and there is
    no list of scoring rules.
    """
    began = time.monotonic()
    best = meta_capability(episodes, THE_ORDER)
    found: Code | None = None
    for candidate in every_code(
        deepest=deepest, variables=5, constants=(0, 1, 2), also=()
    ):
        if time.monotonic() - began >= within:
            break
        closed: Code = candidate
        for _ in range(5):
            closed = Code("given a thing", parts=(closed,))
        try:
            got = meta_capability(episodes, closed)
        except (OutOfFuel, Stuck, TypeError, ValueError):
            continue
        if got < best:
            best, found = got, closed
    return found


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--episodes", type=int, default=60)
    ask.add_argument("--deepest", type=int, default=4)
    ask.add_argument("--within", type=float, default=60.0)
    ask.add_argument("--seed", type=int, default=3000)
    ask.add_argument("--out", default="")
    said = ask.parse_args()

    rng = random.Random(said.seed)
    terms = [
        one
        for one in itertools.islice(every_term((0, 1, 2), holes=2, deepest=2), 8000)
        if holes_in(one) == 2
    ]
    names = sorted(WHERE_FROM)
    episodes: list[Episode] = []
    while len(episodes) < said.episodes:
        made = _an_episode(rng, terms, names)
        if made is not None:
            episodes.append(made)
    training, sealed = episodes[0::2], episodes[1::2]

    forget_the_order()
    before_training = meta_capability(training, THE_ORDER)
    before_sealed = meta_capability(sealed, THE_ORDER)

    began = time.monotonic()
    written = _an_order_she_writes(training, deepest=said.deepest, within=said.within)
    took = time.monotonic() - began

    report: dict[str, Any] = {
        "episodes": len(episodes),
        "training": len(training),
        "sealed": len(sealed),
        "before": {"training": before_training, "sealed": before_sealed},
        "seconds": round(took, 1),
    }
    if written is None:
        report["wrote"] = None
        print("nothing better than the order she was given")
    else:
        the_order_she_wrote(written)
        report["wrote"] = {"symbols": how_long(written)}
        report["after"] = {
            "training": meta_capability(training, written),
            "sealed": meta_capability(sealed, written),
        }
        forget_the_order()
        report["lesioned"] = {
            "training": meta_capability(training, the_order_she_uses()),
            "sealed": meta_capability(sealed, the_order_she_uses()),
        }
        print(
            f"she wrote an order of {how_long(written)} symbols in {took:.0f}s\n"
            f"  training  before {before_training:.3f}  after "
            f"{report['after']['training']:.3f}\n"
            f"  SEALED    before {before_sealed:.3f}  after "
            f"{report['after']['sealed']:.3f}\n"
            f"  lesioned  sealed {report['lesioned']['sealed']:.3f}"
        )
        better = report["after"]["sealed"] < before_sealed
        back = abs(report["lesioned"]["sealed"] - before_sealed) < 1e-9
        print(f"  better on sealed: {better}   lesion restores it: {back}")
        report["better_on_sealed"] = better
        report["lesion_restores"] = back

    if said.out:
        from pathlib import Path

        Path(said.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
