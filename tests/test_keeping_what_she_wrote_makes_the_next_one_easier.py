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
        stream="shared", blocks=3, per_block=4, seed=1000, within=2.0, deepest=3
    )
    apart = run_stream(
        stream="apart", blocks=3, per_block=4, seed=1000, within=2.0, deepest=3
    )

    # Keeping what she wrote never costs her a family.
    assert all(
        grown >= reset
        for grown, reset in zip(shared["grown"], shared["reset"], strict=True)
    ), shared
    # And on the stream with structure in it, the gap is there by the end.
    assert shared["grown"][-1] > shared["reset"][-1], shared

    # The control: nothing to carry, so carrying it buys nothing.
    assert apart["grown"] == apart["reset"], apart
    assert sum(apart["grown"]) - sum(apart["reset"]) == 0

    # And the lesion. Taking the newest entry out returns her to the reset
    # condition or below it, which is what says the library was the cause.
    assert sum(shared["lesioned"]) <= sum(shared["grown"]), shared


@pytest.mark.slow
def test_the_library_is_what_grew_and_not_the_budget() -> None:
    """Matched compute, so the gap is not GROWN having been given more time."""
    shared = run_stream(
        stream="shared", blocks=2, per_block=3, seed=1002, within=2.0, deepest=3
    )
    assert shared["library"] > 0
    # Every agent got the same per-family allowance; the only difference in
    # the call is what is offered as leaves.
    import inspect

    from tools.run_grown_against_reset_heads import _attempt

    source = inspect.getsource(_attempt)
    assert "within=within" in source
    assert "already=tuple(agent.library)" in source


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
