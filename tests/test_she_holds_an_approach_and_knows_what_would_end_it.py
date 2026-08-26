"""Holding a line, and knowing in advance what would end it.

Predicting one move and grading it afterwards is reactive: it assumes the
world sits still and corrects once the world says otherwise. Anywhere the
world keeps moving while she works — a board that deals a new tile after
every move, a page that changes under her, anything with other people in it —
the missing middle is an approach: a line held across moves, with the
condition that would make it wrong named when she adopts it rather than
discovered when it fails.
"""
from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption
from core.agency.standing_strategy import (
    RECONSIDER_AFTER,
    Strategy,
    read_strategy,
    settle_on_an_approach,
    still_holds,
)

MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]

SPOKEN = (
    "Plan: build the big tiles in the bottom-left corner, because that keeps a "
    "clear row above. I will keep going while the 64 stays in the corner. "
    "Otherwise switch to the right edge."
)


def test_an_approach_carries_its_reason_and_its_ending():
    held = read_strategy(SPOKEN)
    assert held is not None
    assert "bottom-left corner" in held.approach
    assert "clear row above" in held.because
    assert "64 stays in the corner" in held.holds_while.describes
    assert held.otherwise == ("switch to the right edge",)


def test_an_approach_with_nothing_that_could_end_it_is_not_an_approach():
    """A preference is not a plan. Knowing in advance what would change her
    mind is the whole difference."""
    assert read_strategy("Plan: play well and keep the score going up.") is None


def test_what_it_depends_on_going_missing_ends_it():
    held = read_strategy(SPOKEN)
    holding, why = still_holds(held, "tiles 2 4 8 64 with 64 bottom left")
    assert holding and not why
    holding, why = still_holds(held, "tiles 2 4 8 16 32")
    assert not holding
    assert "64" in why


def test_a_condition_phrased_as_something_to_avoid_ends_it_when_it_happens():
    held = read_strategy("Plan: keep merging upward. I will keep going until no 1024 appears.")
    assert held is not None
    assert held.holds_while.absent == ("1024",)
    holding, why = still_holds(held, "board shows 512 256 1024")
    assert not holding
    assert "1024" in why


def test_an_approach_nobody_revisits_is_a_habit():
    held = Strategy(approach="a way", holds_while=read_strategy(SPOKEN).holds_while, adopted_on_move=0)
    holding, _why = still_holds(held, "64 is there", moves_made=RECONSIDER_AFTER - 1)
    assert holding
    holding, why = still_holds(held, "64 is there", moves_made=RECONSIDER_AFTER)
    assert not holding
    assert "fresh look" in why


def test_no_approach_yet_is_not_an_approach():
    holding, why = still_holds(None, "anything")
    assert not holding and why


@pytest.mark.asyncio
async def test_she_is_asked_for_the_line_not_for_the_next_move():
    asked = {}

    async def think(objective, evidence):
        asked["objective"] = objective
        asked["evidence"] = list(evidence)
        return SPOKEN

    held = await settle_on_an_approach("reach 4096", "a board", MOVES, think=think, moves_made=7)
    assert held is not None
    assert held.adopted_on_move == 7
    assert "not just the next move" in asked["objective"]
    assert any("what is visible now" in line.lower() for line in asked["evidence"])


@pytest.mark.asyncio
async def test_the_approach_she_left_behind_is_part_of_the_next_question():
    """Deciding again with no memory of what just failed is starting over."""
    asked = {}

    async def think(objective, evidence):
        asked["evidence"] = list(evidence)
        return SPOKEN

    before = read_strategy(SPOKEN)
    await settle_on_an_approach(
        "reach 4096", "a board", MOVES, think=think, previous=before, moves_made=3
    )
    assert any("The approach I was taking" in line for line in asked["evidence"])
    assert any("stopped being right" in line for line in asked["evidence"])


@pytest.mark.asyncio
async def test_language_out_of_reach_leaves_her_without_a_stated_approach():
    """Not a crash and not a made-up plan. She carries on choosing moves."""

    async def think(objective, evidence):
        raise TimeoutError("no answer in time")

    assert await settle_on_an_approach("reach 4096", "a board", MOVES, think=think) is None
    assert await settle_on_an_approach("reach 4096", "a board", MOVES, think=None) is None


def test_the_approach_reads_as_something_a_person_would_say():
    held = read_strategy(SPOKEN)
    said = held.narrate()
    assert said.startswith("Plan: ")
    assert "Watching for:" in said
