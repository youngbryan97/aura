"""How much of what happens next is the world's to decide, and which way to lean.

Looking ahead averages over what the world might do. That is right for working
out what a move is worth, and it throws away the thing a person plays for: a
position where every reply leaves her fine is not the same as one where the
average reply leaves her fine and one of them ruins her, and an average cannot
tell them apart.

That difference is what a grip is. A hand on a wrist does not improve the
position by itself and it is not free — it occupies a hand. What it buys is
that whatever the other side does next, she is still where she was. Keeping
the largest thing in a corner buys the same thing against a world that puts
something new down every turn.

Which way to lean on it is not a temperament. A fighter takes a shot at
nineteen seconds that he would not take at three minutes, and takes it behind
and not ahead. Both come off the run: how much of the budget is left, and what
she has been getting per act.
"""

from __future__ import annotations

from core.agency.looking_ahead import (
    at_the_worlds_mercy,
    whether_to_take_the_wide_option,
)
from core.perception.how_it_moves import RULES, HowItMoves
from core.perception.where_it_responds import what_is_there

ACTS = ("left", "right", "up", "down")


def _board(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


def _one_that_knows(rule_name: str = "slides and combines") -> HowItMoves:
    knows = HowItMoves()
    for rule in RULES:
        knows.tried[rule.name] = knows.tried_when_it_moved[rule.name] = 20
        got = 20 if rule.name == rule_name else 0
        knows.right[rule.name] = knows.right_when_it_moved[rule.name] = got
    knows.seen = knows.moved = 20
    return knows


class _AWorldThatPutsThingsDown:
    """A world that drops something into one of the free places each turn."""

    def __init__(self, spread: bool = True) -> None:
        self.spread = spread

    def might_do(self, state):
        free = [
            (row, column)
            for row in range(state.rows)
            for column in range(state.columns)
            if state.at(row, column) is None
        ]
        if not free:
            return ()
        from core.perception.what_is_there import Arrangement, Cell

        ways = []
        for row, column in free[:4]:
            cells = (
                *state.cells,
                Cell(row=row, column=column, says="2", at=(float(column), float(row))),
            )
            ways.append(
                (
                    Arrangement(
                        rows=state.rows,
                        columns=state.columns,
                        cells=cells,
                        down_at=state.down_at,
                        across_at=state.across_at,
                    ),
                    1.0 / len(free[:4]),
                )
            )
        return tuple(ways)


def test_a_position_the_world_can_swing_is_named_as_one():
    knows, world = _one_that_knows(), _AWorldThatPutsThingsDown()
    exposed = at_the_worlds_mercy(
        knows, _board("2 4 . .\n. . . .\n. . . .\n. . . 8"), ACTS, world=world
    )
    assert exposed, "she was told nothing about what the world could do to her"
    assert all(0.0 <= share <= 1.0 for share in exposed.values())


def test_with_no_world_model_she_claims_nothing():
    knows = _one_that_knows()
    assert at_the_worlds_mercy(knows, _board("2 4\n8 16"), ACTS, world=None) == {}
    assert at_the_worlds_mercy(None, _board("2 4\n8 16"), ACTS, world=_AWorldThatPutsThingsDown()) == {}


def test_a_world_with_nothing_to_do_cannot_swing_anything():
    """Every place taken: whatever happens next, it is not the world's doing."""
    knows, world = _one_that_knows(), _AWorldThatPutsThingsDown()
    full = _board("2 4 8 2\n16 32 64 4\n2 8 4 2\n4 2 8 16")
    assert at_the_worlds_mercy(knows, full, ACTS, world=world) == {}


def test_ahead_with_time_to_spare_she_steadies():
    assert whether_to_take_the_wide_option(left=0.9, gaining=0.5) < 0


def test_behind_with_the_clock_going_she_takes_the_wide_one():
    assert whether_to_take_the_wide_option(left=0.1, gaining=-0.5) > 0


def test_the_same_deficit_leans_harder_the_later_it_is():
    early = whether_to_take_the_wide_option(left=0.9, gaining=-0.5)
    late = whether_to_take_the_wide_option(left=0.1, gaining=-0.5)
    assert late > early > 0


def test_the_same_lead_matters_less_the_later_it_is():
    early = whether_to_take_the_wide_option(left=0.9, gaining=0.5)
    late = whether_to_take_the_wide_option(left=0.1, gaining=0.5)
    assert early < late < 0


def test_level_leans_neither_way():
    assert whether_to_take_the_wide_option(left=0.5, gaining=0.0) == 0.0


def test_it_stays_between_taking_and_avoiding():
    for left in (0.0, 0.25, 0.5, 0.75, 1.0):
        for gaining in (-2.0, 0.0, 2.0):
            assert -1.0 <= whether_to_take_the_wide_option(left, gaining) <= 1.0
