"""A move chosen for what her search saw further on says so.

LIVE 2026-09-23: a line of her game read "Left." and nothing else. Nothing
separated left from the runner-up on the board right after the move, and her
search chose it for what came a few moves later — which went unsaid.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.perception.what_is_there import Arrangement, Cell
from core.skills.screen_pursuit_decision_branches import _what_she_says_as_she_moves


def _board(*cells):
    return Arrangement(
        rows=4, columns=4,
        cells=tuple(Cell(row=r, column=c, says=s, at=(0.0, 0.0)) for r, c, s in cells),
        places_seen=True,
    )


class _Rules:
    """Moves every tile to one side, and says every move leaves the same board."""

    def rule(self):
        return object()

    def expect(self, board, action):
        return board


def _say(ahead):
    board = _board((1, 3, "2"), (2, 1, "4"))
    return _what_she_says_as_she_moves(
        "left", board, SimpleNamespace(rules=_Rules()), ahead,
        weights={}, toward="2048", approach="", biggest_so_far=4.0,
    )


def test_a_move_ahead_of_the_runner_up_says_it_was_for_what_follows():
    said = _say({"left": (0.9, ""), "up": (0.7, "")})
    assert "played a few moves on" in said and "than up would" in said


def test_a_tie_says_it_is_a_tie():
    said = _say({"left": (0.7, ""), "up": (0.7, "")})
    assert "as well as up a few moves on, and no better" in said


def test_a_move_rated_below_the_runner_up_does_not_claim_the_search_as_its_reason():
    said = _say({"left": (0.5, ""), "up": (0.7, "")})
    assert "a few moves on" not in said
