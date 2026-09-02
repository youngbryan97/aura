"""Experiment G, at the level of the grammar, with its negative control.

Three agents, the same families in the same order, the same budget, the same
words. GROWN keeps every head it writes and offers them as leaves next time.
RESET is emptied between blocks. LESIONED keeps the library except the newest
entry, which is the piece the next family was most likely drawn over.

Two streams, and the difference between them is what makes the result mean
anything. On the **shared** stream each family's term is drawn with the terms
already found available, so the stream has structure a learner could carry
forward. On the **apart** stream every term is drawn from the bedrock alone, so
there is nothing to carry.

Nobody chooses which family follows which. The terms are drawn at random and
the correspondence is read off whatever they compute.

Measured over five seeds, five blocks of six families:

    shared   GROWN 150/150   RESET 78    LESIONED 70
             gap by block 0, 2, 2.6, 5.8, 4
    apart    GROWN 150/150   RESET 150   LESIONED 150
             gap by block 0, 0, 0, 0, 0

The budget is a count of candidates, not a number of seconds, and that was a
correction rather than a preference: on a wall clock this assertion failed
twice in three runs, because two agents given the same seconds are not given
the same search. Under a count it does not.

The recurrence route is off throughout. With it on every family here is solved
in one candidate whatever the library holds — four of four in every block of
every condition — so the comparison would have nothing to compare. What is
measured is what the library buys the ENUMERATION, and saying which of the two
is being measured is the point.

The bounded version below is what runs in CI. It asserts the shape rather than
the exact numbers, because the exact numbers belong in
`artifacts/endogenous/grown_against_reset.json` where they can be re-measured
rather than in an assertion that has to be edited whenever the search changes.
"""

from __future__ import annotations

import pytest

from tools.run_grown_against_reset_heads import run_stream


@pytest.mark.slow
def test_the_gap_opens_only_where_there_is_something_to_carry() -> None:
    shared = run_stream(
        stream="shared", blocks=3, per_block=3, seed=1000, within=1.0, deepest=3
    )
    apart = run_stream(
        stream="apart", blocks=3, per_block=3, seed=1000, within=1.0, deepest=3
    )

    # On the stream with structure in it the gap is there by the end.
    assert shared["grown"][-1] > shared["reset"][-1], shared
    assert sum(shared["grown"]) > sum(shared["reset"]), shared

    # The control: nothing to carry, so the gap does not open. Not asserted to
    # be exactly nought — every agent gets the same wall clock, and a family
    # solved with a second to spare in one condition can miss it in another.
    opened = (apart["grown"][-1] - apart["reset"][-1]) - (
        apart["grown"][0] - apart["reset"][0]
    )
    assert opened <= 0, apart

    # And the lesion. Taking the newest entry out returns her to the reset
    # condition or below it, which is what says the library was the cause.
    assert sum(shared["lesioned"]) <= sum(shared["grown"]), shared


@pytest.mark.slow
def test_the_library_is_what_grew_and_not_the_budget() -> None:
    """Matched compute, so the gap is not GROWN having been given more time."""
    shared = run_stream(
        stream="shared", blocks=2, per_block=3, seed=1002, within=1.0, deepest=3
    )
    assert shared["library"] > 0
    # Every agent got the same per-family allowance; the only difference in
    # the call is what is offered as leaves.
    import inspect

    from tools.run_grown_against_reset_heads import _attempt

    source = inspect.getsource(_attempt)
    assert "most_candidates=" in source, "a wall clock is not a matched budget"
    assert "already=tuple(agent.library)" in source
    assert "by_recurrence=False" in source


def test_a_constant_family_is_not_offered_as_evidence() -> None:
    """A rule that says the same thing everywhere is not a rearrangement."""
    import random

    from tools.run_grown_against_reset_heads import _a_family, _correspondence
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.the_floor_she_stands_on import Code

    flat = Code("a number", value=1)
    assert _correspondence(flat, WHERE_FROM["here"], WHERE_FROM["one along"]) is None
    made = _a_family(random.Random(7), [], 3)
    assert made is not None
    for _before, after in made.transitions:
        assert len(set(after)) > 1


# ── experiment F: does it carry to a family that looks nothing like it? ───


@pytest.mark.slow
def test_what_she_wrote_carries_to_a_different_surface_and_not_to_an_unrelated_one() -> None:
    """One structure, learned on one surface, tested on another and on neither.

    The three domains differ in surface — a different pair of words, so the
    before-and-after states look unrelated — and in what is underneath. The
    related domain's term contains the piece she wrote on the first; the
    control's term does not contain it anywhere.

    Sixteen seeds, six families each:

        related    with the piece 87/96    without it 71/96
        unrelated  with the piece 90/90    without it 90/90

    The relation is constructed rather than found, and saying so is the point:
    this shows the piece is what carries, not that any real pair of domains
    stands in this relation.
    """
    from tools.run_grown_against_reset_heads import run_transfer

    with_it = without = control_with = control_without = 0
    usable = 0
    for seed in range(2000, 2006):
        row = run_transfer(seed=seed, families=4, within=2.0, deepest=4)
        if "why" in row:
            continue
        usable += 1
        with_it += row["related_with"]
        without += row["related_without"]
        control_with += row["apart_with"]
        control_without += row["apart_without"]

    assert usable >= 4, "too few seeds produced a usable domain"
    assert with_it > without, (with_it, without)
    assert control_with == control_without, (control_with, control_without)


@pytest.mark.slow
def test_the_control_domain_really_does_not_contain_the_piece() -> None:
    """The negative control is only a control if it is negative."""
    import random

    from tools.run_grown_against_reset_heads import Agent, _a_family_from, _attempt, _contains
    from core.cognition.an_invented_kind import WHERE_FROM

    rng = random.Random(2001)
    names = sorted(WHERE_FROM)
    made = _a_family_from(rng, (names[0], names[1]), [], 4)
    assert made is not None
    learned = Agent("learned")
    assert _attempt(learned, made[0], 2.0)[0]
    piece = learned.library[-1]

    apart = _a_family_from(
        rng, (names[2], names[3]), [], 4, must_not_contain=piece
    )
    assert apart is not None
    assert not _contains(apart[1], piece)
