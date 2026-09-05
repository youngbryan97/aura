"""A part whose removal changes nothing was not doing the work.

If the whole system passes, the system is what passed, and that is a real
result whatever the model underneath contributed. The narrower claim — that
the architecture caused the generality — needs the comparison against the
same model in a plain scaffold, and then the parts one at a time.

This file exists because the first ablations run here reported a gap of zero
for a part that was doing the work, and the harness was wrong rather than the
part: the lesion was applied around a gate that cleared the same state itself
on every world, so it removed nothing. A lesion that cannot be observed to
remove anything is not evidence that nothing was removed.
"""

from __future__ import annotations

import pytest

from tools.agi_gauntlet.ablations import THE_LESIONS, what_each_part_is_worth


def test_every_lesion_says_what_it_removes():
    for lesion in THE_LESIONS:
        assert lesion.what_it_removes
        assert lesion.can_be_applied or lesion.needs, (
            f"{lesion.name} can neither be applied nor says what it needs"
        )


def test_a_lesion_that_cannot_be_applied_here_is_declared_not_approximated():
    """A cheaper thing wearing its name is how a control stops controlling."""

    declared = [one for one in THE_LESIONS if not one.can_be_applied]
    assert declared, "the model-in-a-plain-scaffold comparison is not declared"
    for one in declared:
        assert "needs" not in one.name
        assert len(one.needs) > 40, "say what it would take, not that it is missing"


def test_a_lesion_actually_changes_what_the_run_sees():
    """The failure this file is named for: a lesion applied where the gate
    resets the same state itself removes nothing and reports zero."""

    from core.agency.what_matters_here import (
        ENOUGH_RUNS,
        for_this_world,
        forget_what_mattered,
        keep_nothing,
    )

    forget_what_mattered()
    mattered = for_this_world("a world")
    for _ in range(ENOUGH_RUNS):
        mattered.watched([{"nearness": 0.9}], went_well=True)
        mattered.watched([{"nearness": 0.1}], went_well=False)
    guess = {"nearness": 1.0, "newness": 1.0, "freedom": 1.0}
    learned = mattered.weights(guess)
    assert learned != guess, "nothing was learned, so nothing can be removed"

    keep_nothing(True)
    try:
        for_this_world("a world").watched([{"nearness": 0.9}], went_well=True)
        assert for_this_world("a world").runs == 0, "the lesion did not hold"
    finally:
        keep_nothing(False)
        forget_what_mattered()


def test_the_standing_guess_can_be_put_back():
    from core.agency.what_matters_here import (
        always_the_guess,
        for_this_world,
        forget_what_mattered,
    )

    forget_what_mattered()
    guess = {"nearness": 1.0, "newness": 1.0, "freedom": 1.0}
    always_the_guess(True)
    try:
        assert for_this_world("somewhere new").weights(guess) == guess
    finally:
        always_the_guess(False)
    assert for_this_world("somewhere new").weights(guess)["freedom"] == 0.0


def test_the_gaps_are_reported_for_every_number_both_runs_gave():
    def a_gate():
        return {"solved": 0.8, "worlds": 10, "passed": True, "note": "not a number"}

    got = what_each_part_is_worth(a_gate, lesions=())
    assert got["whole"]["solved"] == 0.8
    assert got["lesions"] == {}


def test_a_gap_of_zero_is_a_finding_rather_than_a_disappointment():
    """It is the only way to tell a component that matters from one that is
    present."""

    seen = {"n": 0}

    def a_gate():
        seen["n"] += 1
        return {"solved": 0.8}

    from tools.agi_gauntlet.ablations import Lesion

    def nothing():
        from contextlib import nullcontext

        return nullcontext()

    got = what_each_part_is_worth(
        a_gate, lesions=(Lesion("a part that does nothing", "nothing", apply=nothing),)
    )
    assert got["lesions"]["a part that does nothing"]["gap"]["solved"] == 0.0
    assert seen["n"] == 2
