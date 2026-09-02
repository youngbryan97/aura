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
    yield
    forget_the_order()


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
