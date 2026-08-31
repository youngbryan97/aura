"""Everything downstream of the board reading, on a board that is correct.

Every live run of this has ended "how this moves is not worked out yet", and
each session found a new reason in perception. That left one question
unanswered for weeks: if she were handed a correct board, would the rest of it
work at all — the model, the confidence, the lookahead — or is that broken too?

It works. She learns the rule from watching, is certain of it, and scores
futures over it. Nothing downstream of the reading is what has been failing,
and this is the test that says so on every commit rather than once.
"""

from __future__ import annotations

import itertools

import pytest

from core.agency.looking_ahead import look_ahead
from core.perception.how_it_moves import HowItMoves
from core.perception.what_is_there import Arrangement, Cell


def _board(rows: list[list[int]]) -> Arrangement:
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=str(value), at=(c * 0.1, r * 0.1))
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value
        ),
    )


def _grid(board: Arrangement) -> list[list[int]]:
    out = [[0] * board.columns for _ in range(board.rows)]
    for cell in board.cells:
        out[cell.row][cell.column] = int(cell.says)
    return out


def _slid(row: tuple[int, ...]) -> tuple[int, ...]:
    kept = [one for one in row if one]
    out, at = [], 0
    while at < len(kept):
        if at + 1 < len(kept) and kept[at] == kept[at + 1]:
            out.append(kept[at] * 2)
            at += 2
        else:
            out.append(kept[at])
            at += 1
    return tuple(out + [0] * (len(row) - len(out)))


def _moved(board: Arrangement, act: str) -> Arrangement:
    grid = _grid(board)
    if act == "left":
        grid = [list(_slid(tuple(row))) for row in grid]
    elif act == "right":
        grid = [list(reversed(_slid(tuple(reversed(row))))) for row in grid]
    elif act == "up":
        columns = [list(_slid(tuple(one))) for one in zip(*grid)]
        grid = [list(one) for one in zip(*columns)]
    elif act == "down":
        columns = [list(reversed(_slid(tuple(reversed(one))))) for one in zip(*grid)]
        grid = [list(one) for one in zip(*columns)]
    return _board(grid)


ACTS = ["left", "right", "up", "down"]


def _after_watching(how_many: int = 40) -> tuple[HowItMoves, Arrangement]:
    rules = HowItMoves()
    here = _board([[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for act in itertools.islice(itertools.cycle(ACTS), how_many):
        went = _moved(here, act)
        rules.watched(here, act, went)
        here = went if went.as_text() != here.as_text() else _moved(here, "up")
    return rules, here


def test_she_works_out_how_the_board_moves_from_watching():
    rules, _here = _after_watching()
    rule = rules.rule()
    assert rule is not None, rules.says()
    assert "combines" in rule.name


def test_and_is_sure_of_it():
    """Certainty matters on its own: look_ahead refuses to plan at all while
    confidence is nought, so an uncertain model is the same as no model."""
    rules, _here = _after_watching()
    assert rules.confidence() > 0.9


def test_and_can_then_score_what_each_move_leads_to():
    rules, here = _after_watching()
    ahead = look_ahead(rules, here, ACTS, toward="", budget_s=1.0)
    assert ahead, "she has a rule and still offered no futures"
    assert all(isinstance(one[0], float) for one in ahead.values())


def test_a_move_that_changes_nothing_is_not_offered():
    """Scored like any other it collects the value of the situation it left
    alone, once at every level, so standing still outscores every move that
    costs something."""
    rules, _here = _after_watching()
    packed = _board([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    ahead = look_ahead(rules, packed, ACTS, toward="", budget_s=1.0)
    assert "left" not in ahead


def test_nothing_is_planned_before_the_rule_is_known():
    """The gate that has kept every live run from planning: no rule, no
    confidence, no futures. It is right to be there and it is why the board
    reading is the whole problem."""
    fresh = HowItMoves()
    here = _board([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert fresh.rule() is None
    assert look_ahead(fresh, here, ACTS, toward="", budget_s=0.5) == {}
