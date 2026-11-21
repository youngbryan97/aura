"""How a position stands is said in what her commonest piece is worth, not her least.

LIVE 21 November 2026, after rehearsal had found one kind worth next to nothing: "I judge myself behind by about 48.1 of
my least piece".
"""
from __future__ import annotations

from types import SimpleNamespace

import core.agency.playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import FIRST, SECOND

START = {(1, c): (FIRST, "P") for c in range(8)} | {(0, 0): (FIRST, "R"), (0, 4): (FIRST, "K"),
                                                    (7, 4): (SECOND, "K")}


def test_a_kind_worth_next_to_nothing_is_not_the_unit() -> None:
    weights = SimpleNamespace(worth={"P": 100.0, "R": 500.0, "K": 0.0, "X": 2.0})
    said = playing._how_it_stands(SimpleNamespace(mate=False, score=-300), weights, START)
    assert said == "I judge myself behind by about 3.0 of my commonest piece"


def test_where_the_commonest_kind_has_no_worth_a_middling_piece_is_the_unit() -> None:
    weights = SimpleNamespace(worth={"P": 0.0, "R": 500.0, "N": 300.0})
    said = playing._how_it_stands(SimpleNamespace(mate=False, score=600), weights, START)
    assert said == "I judge myself ahead by about 1.2 of a middling piece of mine"
