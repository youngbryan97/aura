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
    A,
    Code,
    L,
    OutOfFuel,
    Stuck,
    V,
    build,
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


# ── experiment I: does the next invention use what the last one changed? ──


def _a_family_needing_a_head(episode: Episode) -> list[tuple[tuple, tuple]]:
    """The episode's correspondence, as before-and-after states."""
    made = []
    for size, places in sorted(episode.wanted.items()):
        before = tuple(range(100, 100 + size))
        made.append((before, tuple(before[one] for one in places)))
    return made


def _how_many_she_writes(episodes: list[Episode], *, within: float) -> int:
    """How many of these she writes a way of computing for, under the order in force.

    The order decides which words go in the holes first, so a better order is
    a search that reaches the answer sooner — and on a tight budget, sooner is
    the difference between an answer and nothing.
    """
    from core.cognition.a_way_of_computing_she_wrote import (
        a_way_of_computing_she_wrote,
    )

    got = 0
    for episode in episodes:
        found = a_way_of_computing_she_wrote(
            _a_family_needing_a_head(episode),
            now_sayable=lambda: False,
            words=dict(WHERE_FROM),
            within=within,
            by_recurrence=False,
        )
        got += 1 if found is not None else 0
    return got


def second_generation(
    training: list[Episode], sealed: list[Episode], *, deepest: int, within: float,
    each: float,
) -> dict[str, Any]:
    """Write an order, then invent with it, on families it never saw.

    Experiment I in the only form this repository can currently run it: the
    thing changed at the meta level is the order, and what is measured after
    is whether ORDINARY invention goes better with it — not the rank proxy the
    change was selected on.

    No source is edited between the two. The order is a value.
    """
    forget_the_order()
    written = _an_order_she_writes(training, deepest=deepest, within=within)
    if written is None:
        return {"wrote": None}
    before = _how_many_she_writes(sealed, within=each)
    the_order_she_wrote(written)
    after = _how_many_she_writes(sealed, within=each)
    forget_the_order()
    lesioned = _how_many_she_writes(sealed, within=each)
    return {
        "wrote": how_long(written),
        "sealed": len(sealed),
        "with_the_order_she_was_given": before,
        "with_the_order_she_wrote": after,
        "after_the_lesion": lesioned,
    }


# ── the same experiment, at the proposer rather than at the order ────────


def _where_the_answers_were(families: list[Any], *, leaves: int, upto: int) -> list[int]:
    """Which index of the proposal order each family's answer sits at.

    Read off the data rather than searched for, which is the discipline
    everywhere else here and the thing the first two versions of this
    experiment were missing. A reindexing needs to know WHERE to look, and
    where the answers are is observable: ask the default proposer for its k-th
    candidate and see which k solves the family.
    """
    from core.cognition.a_way_of_computing_she_wrote import (
        _computes,
        _here,
        as_a_head,
        what_each_part_says,
    )
    from core.cognition.one_algebra import _where_each_came_from  # noqa: PLC2701
    from core.cognition.the_proposer_she_can_replace import the_candidate_at

    names = sorted(WHERE_FROM)
    found: list[int] = []
    for family in families:
        wanted = _where_each_came_from(family)
        if not wanted:
            continue
        lengths = sorted(wanted)
        for first_name in names:
            for second_name in names:
                tables = {
                    size: (
                        what_each_part_says(WHERE_FROM[first_name], size),
                        what_each_part_says(WHERE_FROM[second_name], size),
                    )
                    for size in lengths
                }
                for which in range(upto):
                    body = the_candidate_at(which, leaves=leaves)
                    if body is None:
                        continue
                    if _computes(body, wanted, tables):
                        found.append(which)
                        break
                else:
                    continue
                break
            else:
                continue
            break
    return found


def _what_it_costs_to_propose_for(
    families: list[Any], *, budget: int, leaves: int = 10
) -> float:
    """Where in the proposal order each family's answer turns up. Lower is better.

    This counts the proposer and nothing else, and getting that wrong is what
    made the first three versions of this experiment report a negative.

    The obvious measure — candidates walked by the whole head search — is
    dominated by something the proposer does not control. The search tries
    word pairs in an outer loop and walks the whole proposal order inside each
    one, so a family whose answer needs the third pair costs two full orders
    before the proposer's own choices matter at all. Measured: answers turning
    up around candidate nineteen hundred when the proposer only offers seven
    hundred. Reindexing that is reindexing noise.

    So what is counted is the index at which the proposer itself offers the
    answer, with the words held fixed. That is exactly the quantity a proposal
    order decides, and nothing else is in it.
    """
    where = _where_the_answers_were(families, leaves=leaves, upto=budget)
    missing = len(families) - len(where)
    return (sum(where) + missing * budget) / max(1, len(families))


def _a_proposer_over(where_to_look: Code) -> Code:
    """Her own proposer, asked in a different order. Valid by construction.

    The first version of this searched floor terms freely and asked each one
    to be a proposer. Thirty-six of four hundred were proposers at all — the
    rest hand back a number, or a pair that is not a written term, and offer
    nothing at every index. Nine-tenths of the search was spent on things that
    were never candidates, and what came back was training noise.

    A proposal policy decides which candidate comes NEXT. So what is searched
    is where to look, and the proposer it is handed to is her own: every
    candidate is then a real proposer, and the space is the space of orders
    rather than the space of terms that might accidentally be one.

    The same move as solving for the step of a recurrence instead of searching
    for the fixed point, one level up.
    """
    from core.cognition.the_proposer_she_can_replace import THE_PROPOSER

    return build(
        L(
            "which",
            L(
                "leaves",
                A(THE_PROPOSER, where_to_look, V("leaves")),
            ),
        )
    )


def _the_numbers_the_answers_show(where: list[int]) -> tuple[int, ...]:
    """Constants a reindexing could be made of, taken from where the answers are.

    Nought, one and two are not enough and that was the defect. A stream whose
    answers sit past five hundred needs a five hundred to reach them, and a
    search over constants nobody measured will never contain one — so the
    experiment reported that no better proposer exists when what it had was no
    way to write one.
    """
    if not where:
        return (0, 1, 2)
    where = sorted(where)
    middle = where[len(where) // 2]
    smallest = where[0]
    return tuple(
        sorted({0, 1, 2, smallest, middle, max(0, smallest - 1), middle - smallest})
    )


def _a_proposer_she_writes(
    families: list[Any], *, budget: int, deepest: int, within: float,
    leaves: int = 10,
) -> Code | None:
    """Search for an order to ask her own proposer in."""
    from core.cognition.the_proposer_she_can_replace import (
        forget_the_proposer,
        the_proposer_she_wrote,
    )

    began = time.monotonic()
    forget_the_proposer()
    best = _what_it_costs_to_propose_for(families, budget=budget)
    constants = _the_numbers_the_answers_show(
        _where_the_answers_were(families, leaves=leaves, upto=_HOW_FAR_TO_LOOK)
    )
    found: Code | None = None
    #: Proposers that would not run, by why. A search reporting "nothing
    #: better" reads very differently when most of what it tried broke.
    broke: dict[str, int] = {}
    for where_to_look in every_code(
        deepest=deepest, variables=2, constants=constants
    ):
        if time.monotonic() - began >= within:
            break
        made = _a_proposer_over(where_to_look)
        the_proposer_she_wrote(made)
        try:
            got = _what_it_costs_to_propose_for(families, budget=budget)
        except Exception as exc:  # noqa: BLE001 - one that breaks proposes nothing
            # A proposer that breaks is a result and not a non-event. Counting
            # them separately is the difference between "the search found
            # nothing better" and "most of what it tried would not run".
            broke[type(exc).__name__] = broke.get(type(exc).__name__, 0) + 1
            continue
        finally:
            forget_the_proposer()
        if got < best:
            best, found = got, made
    forget_the_proposer()
    if broke:
        print(f"proposers that would not run: {broke}")
    return found


#: How far into the proposal order an answer is looked for when reading off
#: where the answers are. Read from the default proposer: it walks one
#: arithmetic head over ten leaves, so past seven hundred it repeats.
_HOW_FAR_TO_LOOK = 700


def _a_family_the_proposer_could_reach(
    rng: random.Random, *, band: tuple[int, int], leaves: int
) -> list[Any] | None:
    """A family whose answer FIRST TURNS UP inside the band.

    Where the generating body sits in the proposal order is not where the
    answer turns up, and the difference is what made the stream unlearnable.
    Several candidates compute the same thing — `plus(#3, #2)` and
    `plus(#2, #3)` among them — so a family built from a body at index six
    hundred is usually answered by a shorter one at index nine, and a stream
    clustered by generating body is not clustered in anything a proposal order
    can see.

    So the band is applied to the observable quantity: a family is kept when
    the index at which its answer first turns up falls inside it. That is what
    a stream with structure means here, and the control is the same draw with
    the band open.
    """
    from core.cognition.a_way_of_computing_she_wrote import (
        as_a_head,
        what_each_part_says,
    )
    from core.cognition.the_proposer_she_can_replace import the_candidate_at

    names = sorted(WHERE_FROM)
    for _ in range(120):
        which = rng.randrange(0, _HOW_FAR_TO_LOOK)
        body = the_candidate_at(which, leaves=leaves)
        if body is None:
            continue
        over = (rng.choice(names), rng.choice(names))
        first, second = WHERE_FROM[over[0]], WHERE_FROM[over[1]]
        closed = as_a_head(body)
        made: list[Any] = []
        ok = True
        for size in (4, 5, 6, 7):
            tables = (
                what_each_part_says(first, size),
                what_each_part_says(second, size),
            )
            here = (
                [int(first(at, size)) % size for at in range(size)],
                [int(second(at, size)) % size for at in range(size)],
            )
            places = []
            for at in range(size):
                said = _ask_a_head(closed, at, size, here[0][at], here[1][at], *tables)
                if said is None:
                    ok = False
                    break
                places.append(said % size)
            if not ok or len(set(places)) < 2:
                ok = False
                break
            before = tuple(range(100, 100 + size))
            made.append((before, tuple(before[one] for one in places)))
        if not ok or len(made) != 4:
            continue
        where = _where_the_answers_were([made], leaves=leaves, upto=_HOW_FAR_TO_LOOK)
        if not where:
            continue
        if band[0] <= where[0] < band[1]:
            return made
    return None


def _ask_a_head(
    closed: Code, at: int, size: int, here_first: int, here_second: int,
    first: Any, second: Any,
) -> int | None:
    from core.cognition.a_way_of_computing_she_wrote import _ask  # noqa: PLC2701

    return _ask(closed, at, size, here_first, here_second, first, second)


#: Where a shared stream's answers sit in the proposal order, and where a
#: control stream's are spread. Read off the default proposer rather than
#: chosen: it walks one arithmetic head over two leaves, so past seven hundred
#: it repeats, and a band late in that range is one the default reaches last.
_WHERE_A_SHARED_STREAM_SITS = (500, 700)
_WHERE_ANY_STREAM_SITS = (0, 700)


def a_better_proposer(
    *, seed: int, families: int, budget: int, deepest: int, within: float,
    stream: str = "shared",
) -> dict[str, Any]:
    """Experiment H, at the proposer, with the control that makes it mean something.

    Split before anything is written, scored on the half never seen, lesioned
    after. Two streams: one whose answers cluster in a band of the proposal
    order, and one whose answers are spread across it. A policy can learn the
    first and cannot learn the second, and the second is what says the gain on
    the first was about learning.
    """
    from core.cognition.the_proposer_she_can_replace import (
        THE_PROPOSER,
        forget_the_proposer,
        the_proposer_in_use,
        the_proposer_she_wrote,
    )

    rng = random.Random(seed)
    band = (
        _WHERE_A_SHARED_STREAM_SITS if stream == "shared" else _WHERE_ANY_STREAM_SITS
    )
    leaves = 10
    made: list[Any] = []
    while len(made) < families:
        forget_the_proposer()
        one = _a_family_the_proposer_could_reach(rng, band=band, leaves=leaves)
        if one is not None:
            made.append(one)
    training, sealed = made[0::2], made[1::2]

    forget_the_proposer()
    before = _what_it_costs_to_propose_for(sealed, budget=budget)
    written = _a_proposer_she_writes(
        training, budget=budget, deepest=deepest, within=within
    )
    if written is None:
        return {"seed": seed, "stream": stream, "wrote": None, "sealed": len(sealed)}
    the_proposer_she_wrote(written)
    after = _what_it_costs_to_propose_for(sealed, budget=budget)
    forget_the_proposer()
    lesioned = _what_it_costs_to_propose_for(sealed, budget=budget)
    assert the_proposer_in_use() == THE_PROPOSER
    return {
        "seed": seed,
        "stream": stream,
        "wrote": how_long(written),
        "sealed": len(sealed),
        "before": round(before, 1),
        "after": round(after, 1),
        "lesioned": round(lesioned, 1),
        "better_on_sealed": after < before,
        "lesion_restores": lesioned == before,
    }


def _a_family_needing_a_head_from(rng: random.Random) -> list[Any] | None:
    """A family drawn at random, as before-and-after states."""
    from tools.run_grown_against_reset_heads import _a_family

    made = _a_family(rng, [], 3)
    return None if made is None else made.transitions


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--episodes", type=int, default=60)
    ask.add_argument("--deepest", type=int, default=4)
    ask.add_argument("--within", type=float, default=60.0)
    ask.add_argument("--seed", type=int, default=3000)
    ask.add_argument("--seeds", type=int, default=1)
    ask.add_argument("--out", default="")
    ask.add_argument("--proposer", action="store_true",
                     help="write a better proposer rather than a better order")
    ask.add_argument("--budget", type=int, default=3000,
                     help="candidates one invention may walk")
    ask.add_argument("--second-generation", action="store_true",
                     help="invent with the order she wrote, on families it never saw")
    ask.add_argument("--each", type=float, default=2.0,
                     help="what one invention gets, in seconds")
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

    if said.proposer:
        rows = [
            a_better_proposer(
                seed=5000 + seed,
                families=said.episodes,
                budget=said.budget,
                deepest=said.deepest,
                within=said.within,
                stream=stream,
            )
            for stream in ("shared", "apart")
            for seed in range(said.seeds)
        ]
        for one in rows:
            print(one)
        if said.out:
            from pathlib import Path

            Path(said.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0

    if said.second_generation:
        found = second_generation(
            training, sealed, deepest=said.deepest, within=said.within, each=said.each
        )
        print(json.dumps(found, indent=2))
        if said.out:
            from pathlib import Path

            Path(said.out).write_text(json.dumps(found, indent=2), encoding="utf-8")
        return 0

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
