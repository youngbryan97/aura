"""Experiment H, at the one part of the mechanism that is already an object.

`the_order_she_tries_them_in.THE_ORDER` is a floor term taking five numbers
and giving a score, and the search consults it to decide which word to put in
a hole first. Replacing it is the same call that replaces a head. So the
question "can she improve the machinery she invents with" has, for this one
component, a place to be asked.

What is measured is not that the source changed. The order decides how far the
search walks before it reaches the word an answer is built on, so what a better
order buys is a lower rank for that word — and that is the number.

The protocol is the ordinary one. Episodes are drawn at random and split before
anything is written. The new order is written from the training half and scored
on the sealed half it never saw. Then the authored order is put back, and the
gain has to go with it.

The result, over five streams of sixty episodes:

    one stream    she wrote an eight-symbol order
                  sealed 2.000 -> 1.833, and the lesion returns it to 2.000
    four streams  nothing the search reached beat the order she was given,
                  at depth four or at depth five

One in five is the number, and it is the claim. A mechanism that can revise
itself and mostly finds nothing to revise is what this shows, and calling it
recursive self-improvement would be calling one demonstrated meta-change a
trend.
"""

from __future__ import annotations

import itertools
import random

import pytest

from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.one_algebra import every_term, holes_in
from core.cognition.the_floor_she_stands_on import how_long
from core.cognition.the_order_she_tries_them_in import (
    THE_ORDER,
    forget_the_order,
    the_order_she_uses,
    the_order_she_wrote,
)
from tools.run_meta_invention import (
    _an_episode,
    _an_order_she_writes,
    meta_capability,
)


@pytest.fixture(autouse=True)
def _restore():
    """Everything global that an order is scored against, put back.

    The order in force was already restored. What was not is the history the
    scoring reads: how often each word has appeared in a term that survived.
    That is global, every test here writes to it, and a search for a better
    order is judged against it — so a test passing alone and failing in the
    suite is not a flake, it is this.
    """
    from core.cognition.how_she_learns_to_look import (
        forget_what_worked,
        how_the_last_ones_looked,
    )
    from core.cognition.what_counts_as_better import forget_the_objective

    forget_what_worked()
    yield
    forget_the_order()
    forget_the_objective()
    forget_what_worked()


def _episodes(seed: int, how_many: int):
    rng = random.Random(seed)
    terms = [
        one
        for one in itertools.islice(every_term((0, 1, 2), holes=2, deepest=2), 8000)
        if holes_in(one) == 2
    ]
    names = sorted(WHERE_FROM)
    made = []
    while len(made) < how_many:
        one = _an_episode(rng, terms, names)
        if one is not None:
            made.append(one)
    return made[0::2], made[1::2]


def test_the_episodes_carry_signal_or_the_experiment_is_a_formality() -> None:
    """Over one word every word tells her the answer equally, so nothing can rank.

    Held as a test because the first version of this experiment used terms over
    one word, found nothing on any stream, and would have been reported as a
    negative — when what it actually showed was that the feature the order
    reads was flat and no order could have separated anything.
    """
    training, sealed = _episodes(3000, 40)
    spread = [
        len({told for told, _symbols in one.features.values()})
        for one in training + sealed
    ]
    assert sum(1 for one in spread if one > 1) > len(spread) // 2


def test_she_writes_one_that_holds_on_episodes_it_never_saw() -> None:
    training, sealed = _episodes(3000, 60)
    forget_the_order()
    before = meta_capability(sealed, THE_ORDER)

    written = _an_order_she_writes(training, deepest=4, within=45.0)
    assert written is not None, "nothing better on the stream this test is for"
    assert how_long(written) <= 24

    the_order_she_wrote(written)
    assert the_order_she_uses() == written
    after = meta_capability(sealed, written)
    assert after < before, (before, after)

    # The lesion. The gain has to go with the thing that made it.
    forget_the_order()
    assert the_order_she_uses() == THE_ORDER
    assert meta_capability(sealed, the_order_she_uses()) == before


def test_the_order_it_wrote_is_a_term_of_the_floor_and_nothing_else() -> None:
    from core.cognition.the_floor_she_stands_on import HOW_MANY_PARTS

    training, _sealed = _episodes(3000, 60)
    written = _an_order_she_writes(training, deepest=4, within=45.0)
    assert written is not None

    def heads(code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    assert set(heads(written)) <= set(HOW_MANY_PARTS)


@pytest.mark.slow
def test_on_most_streams_what_it_writes_does_not_hold_on_sealed() -> None:
    """The honest half of the result, and it is not a failure.

    Selection happens on the training half, so it can and does find something
    there. On four streams of five what it finds does not survive the half it
    never saw — `nought minus how long the word is`, which is a rule saying
    prefer longer words, and it holds on the half it was fitted to and nowhere
    else.

    That is the sealed half doing its job. A mechanism that can revise itself,
    usually finds something on the data it was given, and usually has that
    thrown out by the data it was not, is what this shows.
    """
    held = 0
    for seed in (3001, 3002):
        training, sealed = _episodes(seed, 40)
        before = meta_capability(sealed, THE_ORDER)
        written = _an_order_she_writes(training, deepest=4, within=20.0)
        if written is not None and meta_capability(sealed, written) < before:
            held += 1
    assert held == 0, "one of these streams held after all, so the number moved"


def test_selection_is_on_the_training_half_and_the_sealed_half_decides() -> None:
    """The protocol, as an assertion rather than a paragraph."""
    import inspect

    source = inspect.getsource(_an_order_she_writes)
    assert "episodes" in source
    training, sealed = _episodes(3001, 40)
    written = _an_order_she_writes(training, deepest=4, within=20.0)
    if written is None:
        pytest.skip("nothing was written on this stream, so there is nothing to judge")
    # It was chosen because it beat the authored order on what it saw.
    assert meta_capability(training, written) < meta_capability(training, THE_ORDER)
    # And the half it never saw is what decides whether that meant anything.
    assert meta_capability(sealed, written) >= meta_capability(sealed, THE_ORDER)
    assert the_order_she_uses() == THE_ORDER, "nothing is installed by writing it"


# ── experiment I: does the next invention use what the last one changed? ──


@pytest.mark.slow
def test_the_order_she_wrote_reaches_ordinary_invention() -> None:
    """The second generation, on families the order never saw.

    What was changed at the meta level is the order. What is measured after is
    whether ORDINARY invention goes better with it — not the rank the change
    was selected on, which is what makes this transfer rather than fit. No
    source is edited between the two: the order is a value.

    Three streams where an order was written, twenty sealed families each,
    every invention on the same tight budget:

        given 13 → wrote 16 → lesion 13
        given 16 → wrote 18 → lesion 16
        given 13 → wrote 12 → lesion 13

    Two of three better, one worse, and the lesion exact every time. The one
    that is worse is the no-free-lunch theorem arriving in practice rather
    than in a footnote, and it is why the claim names its stream.
    """
    from tools.run_meta_invention import second_generation

    found = second_generation(
        *_episodes(3000, 40), deepest=4, within=45.0, each=1.5
    )
    assert found["wrote"] is not None
    assert found["with_the_order_she_wrote"] > found["with_the_order_she_was_given"]
    assert found["after_the_lesion"] == found["with_the_order_she_was_given"]


@pytest.mark.slow
def test_the_lesion_is_exact_even_where_what_she_wrote_was_worse() -> None:
    """A stream where her order made ordinary invention worse.

    Kept as a test rather than left out. An update rule that improved on every
    stream would contradict a theorem this codebase already executes, and a
    result that only ever shows the good streams is the shape of a result
    nobody should believe.
    """
    from tools.run_meta_invention import second_generation

    found = second_generation(
        *_episodes(3003, 40), deepest=4, within=30.0, each=1.5
    )
    if found["wrote"] is None:
        pytest.skip("nothing written on this stream")
    assert found["after_the_lesion"] == found["with_the_order_she_was_given"]


# ── the same experiment, at the proposer rather than at the order ────────


@pytest.mark.slow
def test_at_the_proposer_the_lesion_is_exact_and_the_transfer_mostly_is_not() -> None:
    """Experiment H, one level up from the order, and it mostly does not work.

    The order decides which word goes in a hole first. The PROPOSER decides
    what to try at all, and it is the larger half of the machinery — so making
    it a term and replacing it is the stronger version of the same claim.
    Whether the replacement is any GOOD is a separate question, and here the
    answer is mostly no.

    Eight streams, split before anything is written, scored on the half never
    seen, lesioned after:

        5000  wrote 3 symbols   sealed cost 7,594 -> 8,033     worse
        5001  wrote 3 symbols   sealed cost 34,884 -> 26,634   better
        5002  nothing written
        5003  wrote 3 symbols   sealed cost 9,293 -> 9,730     worse
        5004  nothing written
        5005  wrote 3 symbols   sealed cost 1,207 -> 4,036     worse
        5006  nothing written
        5007  nothing written

    One better, three worse, four nothing. What holds in every written case is
    the LESION: the number returns exactly. What does not hold is transfer —
    selecting on four training families finds training noise more often than a
    better proposal policy.

    So this asserts what is true across the streams rather than what is true
    on the best of them. A passing test built on seed 5001 alone would be a
    result chosen after the fact.
    """
    from tools.run_meta_invention import a_better_proposer

    written = restored = 0
    for seed in (5000, 5001):
        found = a_better_proposer(
            # The configuration the result was measured at, which is not the
            # one this test was first written against. Eight families leaves
            # four to fit a proposer on, and four is below what generalises —
            # measured, and recorded in
            # artifacts/endogenous/a_better_proposer_20_episodes.txt. A test
            # registered against a configuration the experiment never used is
            # asking about a different experiment.
            seed=seed,
            families=20,
            budget=700,
            deepest=3,
            within=120.0,
        )
        if found["wrote"] is None:
            continue
        written += 1
        restored += 1 if found["lesion_restores"] else 0
    assert written > 0, "nothing was written on either stream"
    assert restored == written, "a lesion did not return the number exactly"



def test_a_proposer_is_selected_on_the_training_half_only() -> None:
    """The protocol, as an assertion rather than a paragraph."""
    import inspect

    from tools.run_meta_invention import a_better_proposer

    source = inspect.getsource(a_better_proposer)
    assert "training, sealed = made[0::2], made[1::2]" in source
    # Selection sees training; the number reported comes from sealed. Named
    # against what the functions are called now — the assertion was still
    # looking for _how_many_it_proposes_for, which stopped existing when the
    # cost measure changed to count the index the proposer offers the answer
    # at, and a source assertion that names something gone passes or fails on
    # the rename rather than on the protocol.
    assert source.index("_a_proposer_she_writes(\n        training") < source.index(
        "after = _what_it_costs_to_propose_for(sealed"
    )
