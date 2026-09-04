"""Acts that changed nothing cannot tell one rule from another.

Every rule there is agrees about an act that did nothing: they all predict
what is already there. So the raw counts, before enough acts have MOVED
something, elect whichever rule claims least — and on a board where two of
four directions do nothing from the opening position, "this does not move"
wins three of the first four comparisons and is adopted.

A rule that says nothing ever changes takes her search off the board: every
move looks identical, looking ahead returns nothing, and she plays blind for
the rest of the run.

LIVE 2026-09-04: "I can see what my moves do here now — this does not move —
right 75% of 4", on the fifth move of a game.
"""

from __future__ import annotations

from core.perception.how_it_moves import ENOUGH_TO_TRUST, HowItMoves, shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell


def _board(*rows: tuple[str, ...]) -> Arrangement:
    return Arrangement(
        rows=len(rows),
        columns=len(rows[0]),
        cells=tuple(
            Cell(row=r, column=c, says=said, at=(0.25 + 0.17 * c, 0.34 + 0.14 * r))
            for r, row in enumerate(rows)
            for c, said in enumerate(row)
            if said
        ),
    )


PACKED = _board(("2", "4", "8", "16"), ("4", "8", "16", "32"))
OPENING = _board(("2", "", "", ""), ("", "", "", "2"))


def test_a_world_that_has_moved_a_little_has_no_rule_yet():
    model = HowItMoves()
    # Three acts that did nothing, one that did — the opening of a game.
    for _ in range(3):
        model.watched(PACKED, "left", PACKED)
    model.watched(OPENING, "left", shifted_and_combined(OPENING, "left"))
    assert model.seen >= ENOUGH_TO_TRUST
    assert 0 < model.moved < ENOUGH_TO_TRUST
    assert model.rule() is None


def test_a_world_that_has_never_moved_says_so():
    model = HowItMoves()
    for _ in range(ENOUGH_TO_TRUST + 2):
        model.watched(PACKED, "left", PACKED)
    assert model.moved == 0
    rule = model.rule()
    assert rule is not None
    assert rule.name == "does not move"


def test_enough_acts_that_moved_something_settle_it():
    model = HowItMoves()
    state = _board(("2", "", "4", ""), ("", "8", "", "2"), ("4", "", "", ""))
    for step in range(10):
        move = ("left", "up", "right", "down")[step % 4]
        after = shifted_and_combined(state, move)
        model.watched(state, move, after)
        state = after
    assert model.moved >= ENOUGH_TO_TRUST
    assert model.rule() is not None
    assert model.rule().name != "does not move"


def test_nothing_is_worse_than_a_wrong_rule_here():
    """With no rule she acts and looks, which is what fills the counts."""
    model = HowItMoves()
    for _ in range(3):
        model.watched(PACKED, "left", PACKED)
    model.watched(OPENING, "left", shifted_and_combined(OPENING, "left"))
    assert model.expect(OPENING, "left") is None
    assert model.confidence() == 0.0
