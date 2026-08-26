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


class _Recorded:
    """A consequence graph holding what a move has led to before."""

    def __init__(self, rows):
        self.rows = list(rows)

    def query_consequences(self, action, params=None):
        return self.rows


def test_a_move_the_world_answers_two_ways_is_named_as_such():
    """A single expected outcome is a bet that the world is deterministic.

    Where it is not, the record already holds the answer: the same action and
    more than one result.
    """
    from core.agency.deliberate_action import what_could_happen

    spread = _Recorded([{"success": i % 2 == 0, "outcome": f"o{i}"} for i in range(6)])
    said = what_could_happen("up", graph=spread)
    assert "more than one way" in said
    assert "3 of the last 6" in said


def test_a_move_that_has_always_gone_the_same_way_is_a_prediction_not_a_spread():
    from core.agency.deliberate_action import what_could_happen

    same = _Recorded([{"success": True, "outcome": "o"} for _ in range(6)])
    assert what_could_happen("up", graph=same) == ""


def test_too_little_record_says_nothing_rather_than_inventing_a_spread():
    from core.agency.deliberate_action import SPREAD_DEPTH, what_could_happen

    thin = _Recorded([{"success": bool(i), "outcome": "o"} for i in range(SPREAD_DEPTH - 1)])
    assert what_could_happen("up", graph=thin) == ""


def test_confidence_is_pulled_back_when_the_same_move_goes_two_ways():
    """However often it has worked, a move the world answers in more than one
    way is not a move she knows."""
    from core.agency.deliberate_action import UNTRIED_CONFIDENCE, confidence_from_history

    worked = ["up worked before: x"] * 4
    settled = confidence_from_history(worked)
    unsettled = confidence_from_history([*worked, "up has gone more than one way here: ..."])
    assert settled > unsettled > UNTRIED_CONFIDENCE


def test_a_spread_line_is_not_counted_as_a_grade():
    """It describes the record; it is not another entry in it."""
    from core.agency.deliberate_action import confidence_from_history

    only_spread = ["up has gone more than one way here: it worked 3 of the last 6 times"]
    assert confidence_from_history(only_spread) == pytest.approx(0.5)


def _board(*values) -> dict:
    layout = [
        {"text": str(value), "x": 0.3 + 0.1 * i, "y": 0.4, "center_x": 0.3 + 0.1 * i,
         "center_y": 0.4, "width": 0.06, "height": 0.05}
        for i, value in enumerate(values)
    ]
    layout.append(
        {"text": "SCORE", "x": 0.7, "y": 0.1, "center_x": 0.7, "center_y": 0.1,
         "width": 0.08, "height": 0.03}
    )
    layout.append(
        {"text": "1024", "x": 0.78, "y": 0.1, "center_x": 0.78, "center_y": 0.1,
         "width": 0.06, "height": 0.03}
    )
    return {"ok": True, "text": " ".join([*(str(v) for v in values), "SCORE 1024"]), "layout": layout}


def test_a_goal_described_in_words_still_notices_the_value_it_names():
    """Her own goal reader turns "play until you get a 128 tile" into "128",
    so the usual path never sees a sentence. A caller that passes the
    description straight through waited forever with the tile in front of
    her: 494 moves, a 128 on the board, and the run reported out of time.
    """
    from core.skills.screen_pursuit import goal_reached

    seen = _board(2, 64, 128, 4)
    assert goal_reached(seen, "a 128 tile is on the board")
    assert goal_reached(seen, "128")
    assert not goal_reached(seen, "a 4096 tile is on the board")


def test_the_value_still_has_to_be_a_thing_and_not_a_label_s_number():
    """The reason the strict path exists: "SCORE 1024" is not a 1024 tile."""
    from core.skills.screen_pursuit import goal_reached

    assert not goal_reached(_board(2, 4, 8), "a 1024 tile is on the board")
    assert not goal_reached(_board(2, 4, 8), "1024")


def test_a_description_naming_no_single_value_is_left_alone():
    """Only a condition naming exactly one value gets the second reading.
    Anything else means what it says."""
    from core.skills.screen_pursuit import goal_reached

    seen = _board(2, 64, 128, 4)
    assert not goal_reached(seen, "a 128 or 256 tile")
    assert not goal_reached(seen, "the word Congratulations appears")


def test_a_condition_that_already_matched_is_unaffected():
    from core.skills.screen_pursuit import goal_reached

    seen = _board(2, 4)
    seen["text"] = "You win! 2048"
    assert goal_reached(seen, "You win")


class _WithSituations:
    def __init__(self, rows):
        self.rows = list(rows)

    def query_consequences(self, action, params=None):
        return self.rows


def test_what_a_move_did_is_recalled_from_situations_like_this_one():
    """The same action has different consequences in different situations —
    that is what a situation IS. Recalling by action alone hands her the
    average of every screen she has ever seen.
    """
    from core.agency.deliberate_action import recall_consequences

    graph = _WithSituations([
        {"action": "left", "context": "a full board with no empty squares",
         "outcome": "nothing moved", "success": False},
        {"action": "left", "context": "2 4 8 with a 64 bottom left",
         "outcome": "merged into 128", "success": True},
    ])
    here = "2 4 8 with a 64 bottom left and a new 2"
    closest = recall_consequences("left", graph=graph, depth=1, like=here)
    assert closest == ["left worked before: merged into 128"]
    unfiltered = recall_consequences("left", graph=graph, depth=1)
    assert unfiltered == ["left did not work before: nothing moved"]


def test_two_situations_with_nothing_in_common_are_not_alike():
    from core.agency.deliberate_action import _how_alike

    assert _how_alike("2 4 8 with a 64 bottom left", "a full board with no empty squares") < 0.2
    assert _how_alike("2 4 8 with a 64 bottom left", "2 4 8 with a 64 bottom left") == 1.0
    assert _how_alike("", "anything") == 0.0


def test_recall_without_a_situation_behaves_as_it_always_did():
    """Nothing that works today changes: the ordering only applies when there
    is a situation to compare against."""
    from core.agency.deliberate_action import recall_consequences

    graph = _WithSituations([
        {"action": "up", "context": "one", "outcome": "first", "success": True},
        {"action": "up", "context": "two", "outcome": "second", "success": True},
    ])
    assert recall_consequences("up", graph=graph, depth=1) == ["up worked before: first"]
