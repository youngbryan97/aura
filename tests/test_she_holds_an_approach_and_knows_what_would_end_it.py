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

import asyncio
import logging

import pytest

from core.agency import standing_strategy

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


class _Blank:
    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_the_line_she_is_taking_reaches_the_moves_she_does_not_say():
    """An approach that only reaches the decisions she puts into words is not
    an approach, it is a remark. Most moves in a fast loop are decided from
    evidence, and a plan that cannot reach those cannot reach most of what
    she does.
    """
    from core.agency.deliberate_action import deliberate

    options = [ActionOption(name=name, detail=f"press {name}") for name in ("up", "down", "left", "right")]
    without = await deliberate(
        "reach 4096", "a board", options, think=None, lived=False, graph=_Blank()
    )
    with_plan = await deliberate(
        "reach 4096",
        "a board",
        options,
        think=None,
        lived=False,
        graph=_Blank(),
        approach="keep the largest tile in the bottom-left corner, pressing left and down",
    )
    assert with_plan.chosen.name in {"left", "down"}
    assert with_plan.chosen.name != without.chosen.name
    assert "describes what I am trying to do" in with_plan.rationale


def test_the_loop_hands_her_approach_to_every_decision():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert 'approach=plan["held"].approach if plan["held"] is not None else ""' in source


def test_throwing_away_the_work_so_far_is_not_reachable_from_a_ranking():
    """Most choices are fine to make from evidence — that is what keeps a
    loop fast. Some are not.

    Measured live with language out of reach: a run began the task again
    three times in a hundred seconds, each time from a ranking, and each
    restart erased the record the ranking was built on. With the costly ways
    out held behind words: no restarts, 66 moves graded, and she kept playing.
    """
    from core.agency.deliberate_action import choose_without_language

    moves = [ActionOption(name=name, detail=f"press {name}") for name in ("up", "down")]
    costly = ActionOption(name="start over", detail="begin again", needs_words=True)
    chosen, _why = choose_without_language([*moves, costly], wanted="begin again from the start")
    assert chosen.name != "start over"


def test_when_everything_available_needs_words_she_says_so():
    from core.agency.deliberate_action import choose_without_language

    only_costly = [ActionOption(name="start over", needs_words=True)]
    chosen, why = choose_without_language(only_costly)
    assert chosen is None
    assert "words" in why


def test_the_ways_out_of_a_stuck_run_are_the_ones_held_behind_words():
    from core.skills.screen_pursuit import ways_out

    seen = {
        "ok": True,
        "text": "New Game",
        "layout": [{"text": "New Game", "x": 0.5, "y": 0.2, "center_x": 0.5, "center_y": 0.2}],
    }
    offered = ways_out(seen)
    assert offered, "a stuck run was offered no way out at all"
    assert all(option.needs_words for option in offered)


def test_where_nothing_answers_the_way_out_needs_no_words():
    """Needing a reason in words protects live work from being thrown away on
    a ranking. Where nothing answers any more there is no live work to
    protect, and the alternative to choosing is pressing keys into something
    that has finished."""
    from core.skills.screen_pursuit import ways_out

    seen = {
        "ok": True,
        "text": "Game Over Play Again",
        "layout": [{"text": "Play Again", "x": 0.5, "y": 0.77, "center_x": 0.5, "center_y": 0.77}],
    }
    offered = ways_out(seen, ended=True)
    assert offered
    assert not any(option.needs_words for option in offered)


def test_a_move_into_something_finished_is_not_offered_as_a_choice():
    """LIVE 2026-08-26: thirty-nine moves after Game Over, each one costing a
    language pass because a screen with extra options counts as unusual. The
    move keys were beside the restart the whole time."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    where = source.index("if ended and out:")
    assert "available = out" in source[where : where + 600]


@pytest.mark.parametrize(
    ("said", "holds_to"),
    [
        ("I'll keep the 32 in the bottom-left and merge downward.", "32"),
        ("I'm going to stack toward the left edge and keep the 64 anchored there.", "64"),
        ("Keeping everything anchored on the 128 in the corner, merging down then left.", "128"),
        ("Plan: build around the 512, because that is where the room is.", "512"),
    ],
)
def test_an_approach_counts_however_she_words_it(said, holds_to):
    """Requiring both the plan and its ending in one templated sentence made
    having an approach depend on phrasing one. Measured live: she played a
    whole game and never once held an approach.

    She was asked how she is going about this, so the answer is the approach.
    What makes it one is that it names something to hold to.
    """
    held = read_strategy(said)
    assert held is not None, "she stated an approach and it was not counted"
    assert holds_to in held.holds_while.contains


@pytest.mark.parametrize(
    "said",
    ["Plan: play well and keep the score going up.", "left", "", "I'll do my best."],
)
def test_naming_nothing_to_hold_to_is_still_not_an_approach(said):
    """A plan with nothing that could end it is a preference, and knowing in
    advance what would change her mind is the whole value."""
    assert read_strategy(said) is None


BOARD = "2 4 2 4 16 16 2 32 64 2 8 8 16 8 2"


@pytest.mark.parametrize(
    "said",
    [
        "Keep the largest tile in the bottom-left corner and merge downward.",
        "I will build up along the left edge.",
        "Going left because that keeps the row clear",
    ],
)
def test_an_approach_that_names_no_value_is_bound_to_what_she_is_looking_at(said):
    """An approach often refers to something without naming it: "keep the
    largest tile in the corner", "protect the big one". None of that is
    checkable on its own, and refusing it meant she held no approach at all —
    measured live, she played whole games without one, because her plans were
    phrased the way people phrase plans.

    What she is referring to is in front of her.
    """
    held = read_strategy(said, MOVES, situation=BOARD)
    assert held is not None
    assert held.holds_while.contains == ("64",), "not bound to the biggest thing on the board"
    holds, _why = still_holds(held, BOARD)
    assert holds
    holds, why = still_holds(held, "2 4 8 16 32")
    assert not holds and "64" in why


@pytest.mark.parametrize("said", ["left", "play well", "Plan: play well.", "up"])
def test_a_move_or_a_preference_is_not_an_approach(said):
    """A bare option name is a move and two words of encouragement is a
    preference. Anchoring either to the board would dress it up as a plan."""
    assert read_strategy(said, MOVES, situation=BOARD) is None


class _ThinkingThatFails:
    """A mind that fails in a way nobody wrote a tuple for."""

    def __init__(self, raising: BaseException) -> None:
        self.raising = raising

    async def __call__(self, *_args, **_kwargs):
        raise self.raising


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [KeyError("gone"), OSError("no socket"), ZeroDivisionError("nonsense")],
)
async def test_a_thought_that_fails_any_way_at_all_is_still_no_plan(failure, caplog):
    """Whatever her thinking raises, the answer is no plan — and it is said."""
    with caplog.at_level(logging.INFO, logger="Aura.Strategy"):
        settled = await standing_strategy.settle_on_an_approach(
            "get to 256", "2 4 8", MOVES, think=_ThinkingThatFails(failure)
        )
    assert settled is None
    assert "could not settle on an approach" in caplog.text


@pytest.mark.asyncio
async def test_a_thought_that_ran_out_of_time_is_no_plan_not_a_teardown(caplog):
    with caplog.at_level(logging.INFO, logger="Aura.Strategy"):
        settled = await standing_strategy.settle_on_an_approach(
            "get to 256",
            "2 4 8",
            MOVES,
            think=_ThinkingThatFails(asyncio.CancelledError()),
        )
    assert settled is None
    assert "ran out of time" in caplog.text


@pytest.mark.asyncio
async def test_a_real_cancellation_keeps_travelling():
    """Tearing the task down must not be read as an ordinary failed thought."""
    thinking = asyncio.Event()

    async def never_answers(*_args, **_kwargs):
        thinking.set()
        await asyncio.Event().wait()

    task = asyncio.ensure_future(
        standing_strategy.settle_on_an_approach(
            "get to 256", "2 4 8", MOVES, think=never_answers
        )
    )
    await thinking.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
