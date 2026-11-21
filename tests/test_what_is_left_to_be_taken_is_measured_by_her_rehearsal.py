"""What she cannot afford to lose, as a feature of a position her rehearsal measures and keeps only where it wins.

A lesson from a recorded draughts game against a hard engine (core/cognition/what_she_cannot_afford_to_lose.py), until
now used only by her on-screen play. Tried first as a choice among moves nearly as good as the best, it cost the search
time and scored 0.375 +- 0.069 over 40 games against the plain search. As a feature (per kind, the pieces each side
leaves attacked and undefended) rehearsal earned it in the test world, and the weights it earned beat the same weights
without it 0.588 +- 0.041 over 120 games (21 November 2026).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.agency.searching_by_kind import described, first_weights, placed
from core.agency.what_a_kind_is_worth import GROUPS, fitted
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

#: A white knight on d5 that a black pawn on e6 attacks and nothing of white's defends.
HANGING = "4k3/8/4p3/3N4/8/8/8/4K3 w - -"
#: The same knight with a white pawn on c4 defending it.
DEFENDED = "4k3/8/4p3/3N4/2P5/8/8/4K3 w - -"


def _engine():
    import aura_board_search

    return aura_board_search


def test_an_exposed_piece_counts_against_its_side() -> None:
    engine = _engine()
    weights = first_weights(CHESS)
    with_it = replace(weights, exposed={"N": 50.0})
    plain = engine.evaluate(described(CHESS), placed(from_fen(HANGING)), weights.as_json())
    counted = engine.evaluate(described(CHESS), placed(from_fen(HANGING)), with_it.as_json())
    assert plain - counted == 50
    defended_plain = engine.evaluate(described(CHESS), placed(from_fen(DEFENDED)), weights.as_json())
    defended_counted = engine.evaluate(described(CHESS), placed(from_fen(DEFENDED)), with_it.as_json())
    assert defended_plain == defended_counted


def test_the_group_is_fitted_where_rehearsal_tries_it() -> None:
    assert "what is left to be taken" in GROUPS
    kinds = len(CHESS.kinds)
    squares = CHESS.rows * CHESS.columns
    width = 2 * kinds + 2 + kinds * squares + kinds
    rng = np.random.default_rng(3)
    xs = rng.normal(size=(400, width))
    exposed_at = 2 * kinds + 2 + kinds * squares
    # Outcomes that follow material and the other side's exposed pieces.
    signal = xs[:, :kinds].sum(axis=1) + 2.0 * xs[:, exposed_at:exposed_at + kinds].sum(axis=1)
    ys = (signal + rng.normal(scale=0.5, size=400) > 0).astype(float)
    weights = fitted(CHESS, xs, ys, ["worth", "what is left to be taken"])
    assert weights is not None and weights.exposed
    assert all(value > 0 for letter, value in weights.exposed.items() if not CHESS.kinds[letter].royal)
