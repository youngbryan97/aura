"""Her judgement of a situation is read as the chance it has turned out to be, where she plays.

LIVE 23 November 2026 she judged herself about even, and ahead, through games Stockfish was winning by six pieces from
the fifth move: nothing kept what her judgements went on to come to. Each judgement she makes in a game is now kept
with how the game ended, and a curve from judgement to outcome is fitted (core/cognition/how_far_her_judgement_holds.py).
"""
from __future__ import annotations

import math
import random

from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.playing_pieces_by_kind import _Game, _the_end
from core.cognition.how_far_her_judgement_holds import ENOUGH_TO_SAY, HowFarItHolds
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def test_the_curve_says_nothing_until_enough_games_have_ended() -> None:
    held = HowFarItHolds()
    for _ in range(ENOUGH_TO_SAY - 1):
        held.came_to([100.0, 50.0], 0.0)
    assert held.chance(100.0) is None and held.says(100.0) == ""


def test_judgements_that_lose_where_they_say_even_are_read_as_losing() -> None:
    """Against a far stronger side, her 'even' loses: the curve reads it so, whatever it says on her own scale."""
    rng = random.Random(3)
    held = HowFarItHolds()
    for _ in range(40):
        judgements = [rng.gauss(0.0, 80.0) for _ in range(20)]
        held.came_to(judgements, 0.0 if rng.random() < 0.9 else 1.0)
    assert held.chance(0.0) < 0.25
    assert "about" in held.says(0.0)


def test_where_her_judgement_tells_outcomes_apart_the_curve_rises_with_it() -> None:
    rng = random.Random(5)
    held = HowFarItHolds()
    for _ in range(60):
        lean = rng.gauss(0.0, 200.0)
        won = rng.random() < 1.0 / (1.0 + math.exp(-lean / 100.0))
        held.came_to([lean + rng.gauss(0.0, 20.0) for _ in range(10)], 1.0 if won else 0.0)
    assert held.chance(300.0) > 0.8 > 0.2 > held.chance(-300.0)
    again = HowFarItHolds.from_memory(held.as_memory())
    assert abs(again.chance(100.0) - held.chance(100.0)) < 0.02
    assert held.error() < 0.25 and held.held_out < 0.25


def test_a_curve_that_foretells_games_it_was_not_fitted_on_no_better_than_a_coin_says_nothing() -> None:
    """Suggestions of 23 November 2026: a curve read on the games it was fitted on flatters itself, most where they are
    few. Judgements that bear on nothing, in games that go either way, fit a curve; read on games each was not fitted
    on, it is no nearer than a coin, and it is not told."""
    rng = random.Random(11)
    held = HowFarItHolds()
    for game in range(ENOUGH_TO_SAY + 1):
        held.came_to([rng.gauss(0.0, 150.0) for _ in range(10)], float(game % 2))
    assert held.chance(100.0) is not None and held.held_out >= 0.25
    assert held.says(100.0) == ""


def test_a_game_that_ended_adds_what_her_judgements_came_to() -> None:
    game = _Game(learned=HowThePiecesMove.from_the_start(8, 8, dict(CHESS.start)), believed=CHESS.the_start())
    game.judged = [("a", "m", 40), ("b", "m", -20), ("c", "m", -300)]
    came = _the_end("won:1", game, lambda line: None)
    assert came["held"].undertakings == 1 and [o for _j, o, _u in came["held"].pairs] == [0.0, 0.0, 0.0]
