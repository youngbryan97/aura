"""Reaching what she was asked for ends a line of thought, and where she has been counts.

Judged only at the far end, a line that reached the goal and kept moving was
worth wherever it wandered next, so the deeper she looked the more such lines
she preferred: a sliding puzzle solved in four of four at seven moves ahead
and one of four at ten (2026-09-25). And in a world where every act can be
undone, a search that remembers only the line it imagines walks the same few
boards for ever.
"""

from __future__ import annotations

import random

from core.agency.looking_ahead import arriving, look_ahead
from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell
from tools.measure_getting_there import WORLDS

PUZZLE = WORLDS["a sliding puzzle"]
GOAL = PUZZLE.toward


def board(rows: list[list[str]]) -> Arrangement:
    cells = tuple(
        Cell(r, c, said, (0.0, 0.0)) for r, row in enumerate(rows) for c, said in enumerate(row) if said
    )
    return Arrangement(len(rows), len(rows[0]), cells, places_seen=True)


def _knows_the_puzzle() -> HowItMoves:
    roll = random.Random(5)
    knows = HowItMoves()
    state = PUZZLE.start(roll)
    for _ in range(80):
        move = roll.choice(list(PUZZLE.acts))
        after = PUZZLE.act(state, move)
        knows.watched(state, move, after)
        state = after
    assert knows.rule() is not None and knows.rule().name == "one thing steps"
    return knows


ONE_AWAY = board([["1", "2", "3"], ["4", "5", "6"], ["7", "", "8"]])
SOLVED = board([["1", "2", "3"], ["4", "5", "6"], ["7", "8", ""]])


def test_the_move_that_arrives_is_the_best_however_far_she_looks():
    knows = _knows_the_puzzle()
    for depth in (1, 3, 6):
        ahead = look_ahead(knows, ONE_AWAY, list(PUZZLE.acts), toward=GOAL, depth=depth)
        best = max(ahead, key=lambda act: ahead[act][0])
        assert PUZZLE.act(ONE_AWAY, best).as_text() == SOLVED.as_text(), depth


def test_arriving_outscores_anything_short_of_it():
    weights = {"nearness": 1.0, "room": 1.0, "order": 1.0}
    assert arriving(weights) > sum(weights.values())


def test_somewhere_she_has_not_been_is_taken_over_somewhere_she_has():
    knows = _knows_the_puzzle()
    here = board([["1", "2", "3"], ["4", "", "6"], ["7", "5", "8"]])
    acts = list(PUZZLE.acts)
    ahead = look_ahead(knows, here, acts, toward=GOAL, depth=2)
    best = max(ahead, key=lambda act: ahead[act][0])
    stood_on = {knows.the_thing(PUZZLE.act(here, best)).as_text()}
    again = look_ahead(knows, here, acts, toward=GOAL, depth=2, been_before=stood_on)
    assert best not in again and again


def test_and_when_every_way_leads_back_she_goes_back_the_best_way():
    knows = _knows_the_puzzle()
    here = board([["1", "2", "3"], ["4", "", "6"], ["7", "5", "8"]])
    acts = list(PUZZLE.acts)
    everywhere = {PUZZLE.act(here, act).as_text() for act in acts}
    ahead = look_ahead(knows, here, acts, toward=GOAL, depth=2, been_before=everywhere)
    assert ahead == look_ahead(knows, here, acts, toward=GOAL, depth=2)


def test_a_compiled_world_ends_the_line_at_the_goal_too():
    """2048's own search: the merge that makes the goal is taken first."""
    from tools.measure_getting_there import WORLDS as ALL

    world = ALL["four by four"]
    roll = random.Random(2)
    knows = HowItMoves()
    state = world.start(roll)
    for _ in range(60):
        move = roll.choice(list(world.acts))
        after = world.act(state, move)
        knows.watched(state, move, after)
        if after.as_text() != state.as_text():
            after = world.something_turns_up(after, roll)
        state = after if not world.over(after) else world.start(roll)
    near = board([["1024", "1024", "", ""], ["2", "", "", ""], ["", "", "", "4"], ["", "", "", ""]])
    ahead = look_ahead(knows, near, list(world.acts), toward="2048", depth=3)
    best = max(ahead, key=lambda act: ahead[act][0])
    assert "2048" in {cell.says for cell in world.act(near, best).cells}
    assert ahead[best][0] >= arriving(None)
