"""Her model of a world, compiled, says exactly what the model says.

LIVE 2026-09-17: one level of looking cost a tenth of a second on a sliding
board, so the clock bought one level, and a search one move deep reached a 256
tile. The same judgement three levels deep reached 2048 in fifteen games of
sixteen. Compiling the rule she learned line by line is what buys the depth,
and it is only allowed if it changes nothing about what she believes: every
act on a compiled world has to be the act her rule describes, and every term
of her measure has to be the term she would have computed.
"""
from __future__ import annotations

import random

import pytest

from core.agency.a_world_compiled import compiled, search
from core.agency.how_good_is_this import terms as authored_terms
from core.agency.looking_ahead import look_ahead
from core.perception.how_it_moves import HowItMoves, composed, shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell

MOVES = ["up", "down", "left", "right"]
LINES = (0.2, 0.35, 0.5, 0.65)


def _board(values: list[int]) -> Arrangement:
    cells = tuple(
        Cell(index // 4, index % 4, str(value), (LINES[index % 4], LINES[index // 4]))
        for index, value in enumerate(values)
        if value
    )
    return Arrangement(4, 4, cells, LINES, LINES)


def _random_board(roll: random.Random) -> Arrangement:
    return _board([roll.choice([0, 0, 0, 2, 2, 4, 8, 16, 32]) for _ in range(16)])


class _Knows:
    """A rule, trusted, with nothing else about it."""

    def __init__(self, rule):
        self._rule = rule

    def rule(self):
        return self._rule

    def confidence(self):
        return 1.0

    def expect(self, arrangement, action):
        return self._rule.apply(arrangement, action)


@pytest.mark.parametrize(
    "rule",
    [
        composed("all the way", True, "everything"),
        composed("all the way", False, "everything"),
        composed("one place", True, "everything"),
        composed("one place", False, "everything"),
    ],
    ids=lambda rule: rule.name,
)
def test_every_act_on_the_compiled_world_is_the_act_her_rule_describes(rule):
    roll = random.Random(7)
    knows = _Knows(rule)
    for _ in range(60):
        here = _random_board(roll)
        made = compiled(knows, None, here, MOVES)
        if made is None:
            continue
        board = made.board(here)
        for action in MOVES:
            expected = rule.apply(here, action)
            got = made.arrangement(made.act(board, action))
            assert got.as_text() == Arrangement(4, 4, expected.cells).as_text(), (action, here.as_text())


def test_a_rule_that_moves_one_thing_on_the_whole_board_is_not_compiled():
    knows = _Knows(composed("all the way", True, "one thing"))
    assert compiled(knows, None, _board([2, 2, 0, 0] + [0] * 12), MOVES) is None


def test_the_measure_on_a_compiled_world_is_her_measure():
    roll = random.Random(11)
    rule = composed("all the way", True, "everything")
    knows = _Knows(rule)
    for _ in range(40):
        here = _random_board(roll)
        made = compiled(knows, None, here, MOVES)
        assert made is not None
        fast = made.terms(made.board(here), toward="2048", actions=MOVES)
        slow = authored_terms(here, toward="2048", knows=knows, acts=MOVES)
        for name, value in fast.items():
            assert value == pytest.approx(slow[name]), name


def test_a_situation_no_act_can_change_is_below_every_live_one():
    rule = composed("all the way", True, "everything")
    knows = _Knows(rule)
    # Two pushes change this: right and down. Each leaves one free place, and a
    # world that always puts a 4096 there leaves no two neighbours alike and no
    # push that changes anything.
    here = _board([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2, 4, 8, 16, 32, 0])
    made = compiled(knows, None, here, MOVES)
    assert made is not None
    made.arrivals = ((made.symbol("4096"), 1.0),)
    made.how_often = 1.0
    scored, _depth = search(
        made, here, MOVES, budget_s=0.2, worth=lambda board: 1.0, dead=-10.0, fixed_depth=2
    )
    assert set(scored) == {"right", "down"}
    for value, _after in scored.values():
        assert value == -10.0
    one_level, _ = search(
        made, here, MOVES, budget_s=0.2, worth=lambda board: 1.0, dead=-10.0, fixed_depth=1
    )
    assert all(value == 1.0 for value, _after in one_level.values())


def test_looking_ahead_takes_the_compiled_world_when_there_is_one():
    model = HowItMoves()
    state = _board([2, 4, 0, 8, 0, 2, 4, 0, 4, 0, 0, 2, 64, 2, 0, 4])
    for move in ("left", "up", "right", "down", "left", "up"):
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    seen = look_ahead(model, state, MOVES, toward="2048", budget_s=0.05)
    assert seen
    assert all(reason for _value, reason in seen.values())
