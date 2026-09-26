"""What each thing about a situation is worth is found in each world, by playing it out.

The standing weights were 2048's: smoothness at 0.4 because six games at each
of four weights said so, room and order at 0.15 by choice. Every term now
starts level and `working_out_what_matters` finds a world's own weighting in
her model of that world.
"""

from __future__ import annotations

import random

import pytest

from core.agency import rehearsing_in_her_model, working_out_what_matters
from core.agency.rehearsing_in_her_model import Played
from core.agency.what_makes_it_good_here import WhatMakesItGoodHere
from core.agency.working_out_what_matters import (
    FEWEST_PAIRS,
    MOST_PAIRS,
    answer,
    every_term_alike,
    the_question,
    work_out_what_matters,
)
from core.perception.how_it_moves import HowItMoves
from core.perception.what_the_world_does import WhatTheWorldDoes
from tools.measure_getting_there import WORLDS

AUTHORED = ("nearness", "newness", "line", "room", "order", "smoothness", "freedom")


def _learned(world_name: str = "three by three", moves: int = 80, seed: int = 1):
    """Her rule and her record of what arrives, from some moves in a world."""
    world = WORLDS[world_name]
    roll = random.Random(seed)
    knows, arrivals = HowItMoves(), WhatTheWorldDoes()
    state = world.start(roll)
    names = list(world.acts)
    start = state
    for _ in range(moves):
        if world.over(state):
            state = start = world.start(roll)
        move = roll.choice(names)
        after = world.act(state, move)
        knows.watched(state, move, after)
        if after.as_text() != state.as_text():
            after = world.something_turns_up(after, roll)
            arrivals.watched(knows.expect(state, move), after)
        state = after
    return knows, arrivals, start, names


def test_before_a_world_is_played_every_term_is_worth_the_same():
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY

    assert {AS_GOOD_A_GUESS_AS_ANY[name] for name in AUTHORED} == {1.0}


def test_nothing_measured_on_one_world_is_left_in_the_standing_weights():
    from core.agency import how_good_is_this as hg

    named = {
        hg.ROOM_MATTERS, hg.ORDER_MATTERS, hg.SMOOTHNESS_MATTERS,
        hg.LINE_MATTERS, hg.FREEDOM_MATTERS, hg.NEWNESS_MATTERS,
    }
    assert named == {1.0}


def test_every_term_alike_is_what_she_starts_from():
    assert every_term_alike(["room", "order"]) == {"room": 1.0, "order": 1.0}


@pytest.fixture
def a_model(monkeypatch):
    """A world that compiles, and games whose outcome the test decides."""
    knows, arrivals, start, names = _learned()
    assert knows.rule() is not None and arrivals.worth_expecting()
    return knows, arrivals, start, names


def _a_game(furthest: float, goal: float) -> Played:
    """A game that got as far as ``furthest``, measured the way her own nearness measures it."""
    import math

    near = min(1.0, math.log2(furthest) / math.log2(goal)) if furthest > 1 else 0.0
    return Played(furthest=furthest, moves=50, reached=furthest >= goal, nearest=near)


def _games(monkeypatch, outcome):
    played: list[dict[str, float]] = []

    def played_out(_knows, _world, _start, _actions, *, weights, seed=0, **_how):
        played.append(dict(weights))
        return outcome(weights, seed)

    monkeypatch.setattr(rehearsing_in_her_model, "played_out", played_out)
    return played


def test_a_change_the_games_favour_is_kept(monkeypatch, a_model):
    knows, arrivals, start, names = a_model

    def outcome(weights, seed):
        # Room is what gets her further here, with a little luck either way.
        luck = random.Random(f"{seed} {weights.get('room', 0.0):.3f}").random()
        return _a_game(2.0 ** (5 + 2 * weights.get("room", 0.0) + 0.2 * luck), 4096)

    _games(monkeypatch, outcome)
    found = work_out_what_matters(knows, arrivals, start, names, toward="4096", within_s=60)
    assert found is not None
    assert found.weights["room"] > 1.0
    assert ("room", 1.0, 2.0, "kept") in [(t, w, n, v) for t, w, n, v, _p in found.tried]


def test_a_change_the_games_cannot_tell_from_luck_is_not_kept(monkeypatch, a_model):
    knows, arrivals, start, names = a_model

    def outcome(weights, seed):
        luck = random.Random(f"{seed} {sorted(weights.items())}").random()
        return _a_game(2.0 ** (5 + 3 * luck), 512)

    _games(monkeypatch, outcome)
    found = work_out_what_matters(knows, arrivals, start, names, toward="512", within_s=60)
    assert found is not None
    assert found.weights == every_term_alike(list(found.weights))
    assert not found.changed()
    assert "could not tell" in {verdict for *_rest, verdict, _p in found.tried}


def test_no_verdict_rests_on_fewer_pairs_than_can_show_a_spread(monkeypatch, a_model):
    knows, arrivals, start, names = a_model
    _games(monkeypatch, lambda weights, seed: Played(furthest=64.0 + seed, moves=50))
    found = work_out_what_matters(knows, arrivals, start, names, toward="512", within_s=60)
    assert found is not None
    assert all(FEWEST_PAIRS <= pairs <= MOST_PAIRS for *_rest, pairs in found.tried)


def test_the_games_played_the_current_way_are_played_once(monkeypatch, a_model):
    knows, arrivals, start, names = a_model
    played = _games(
        monkeypatch,
        lambda weights, seed: Played(furthest=2.0 ** (5 + random.Random(seed).random()), moves=50),
    )
    found = work_out_what_matters(knows, arrivals, start, names, toward="512", within_s=60)
    assert found is not None
    level = every_term_alike(list(found.weights))
    assert sum(1 for weights in played if weights == level) <= MOST_PAIRS


def test_a_term_that_reads_the_same_everywhere_is_not_tried(monkeypatch, a_model):
    knows, arrivals, start, names = a_model
    _games(monkeypatch, lambda weights, seed: Played(furthest=64.0, moves=50))
    found = work_out_what_matters(knows, arrivals, start, names, toward="512", within_s=60)
    assert found is not None
    tried = {term for term, *_rest in found.tried}
    # No line is held, and nothing on a board can say it has been visited.
    assert "line" not in tried and "newness" not in tried


def test_with_no_model_to_play_in_there_is_nothing_to_work_out():
    _knows, _arrivals, start, names = _learned()
    assert work_out_what_matters(HowItMoves(), WhatTheWorldDoes(), start, names) is None


def test_what_she_worked_out_is_kept_with_the_world_and_single_moves_do_not_move_it():
    matters = WhatMakesItGoodHere()
    settled = {**every_term_alike(AUTHORED), "room": 2.0, "smoothness": 0.5}
    matters.settled_by_playing_it_out(settled, "played out 40 game(s)")
    for _ in range(80):
        matters.watched({"room": 1.0, "smoothness": 0.0}, better=False)
    assert matters.weights() == settled
    back = WhatMakesItGoodHere.from_memory(matters.as_memory(), 1.0)
    assert back.played_out and back.weights() == settled
    assert "played out" in back.says()


def test_a_child_asked_the_question_answers_it_the_same_way():
    """What the pursuit sends to a process of its own, answered in this one."""
    knows, arrivals, start, names = _learned()
    question = the_question(
        knows, arrivals, start, names, weights=None, toward="64", within_s=3.0
    )
    said = answer(question)
    assert said["ok"] is True
    assert set(said["weights"]) >= set(AUTHORED)
    assert said["games"] >= 0 and said["stopped"]


def test_the_pursuit_works_it_out_in_a_process_of_its_own():
    import inspect

    from core.skills import screen_pursuit

    body = inspect.getsource(screen_pursuit._judge_what_she_judges_by_in_her_model)
    assert "in_a_process_of_its_own" in body
    assert "settled_by_playing_it_out" in body
    assert inspect.iscoroutinefunction(working_out_what_matters.in_a_process_of_its_own)


@pytest.mark.asyncio
async def test_a_child_whose_answer_nobody_waits_for_is_let_go(monkeypatch):
    """Cancelled mid-game, the child is killed rather than left to play on."""
    import asyncio
    import subprocess
    import sys

    started: list[subprocess.Popen] = []

    def a_child_that_plays_forever():
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys, time; sys.stdin.read(); time.sleep(600)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        started.append(child)
        return child

    monkeypatch.setattr(working_out_what_matters, "_a_child", a_child_that_plays_forever)
    knows, arrivals, start, names = _learned()
    asking = asyncio.ensure_future(
        working_out_what_matters.in_a_process_of_its_own(
            knows, arrivals, start, names, toward="64", within_s=600.0
        )
    )
    await asyncio.sleep(0.5)
    asking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asking
    assert started and started[0].poll() is not None
