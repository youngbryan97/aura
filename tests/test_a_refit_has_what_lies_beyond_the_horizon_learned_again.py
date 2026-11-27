"""A refit of what a weighted sum holds has what lies beyond the horizon learned again on top of it
(core/agency/what_a_kind_is_worth.py ``rehearse``). What lies beyond the horizon is fitted with her search's score held
fixed; a refit fits the rest without it, and what it was worth is taken up by the rest.

LIVE 26 November 2026 her weights at 474 rounds held nothing for what lies beyond the horizon, though the group stood
earned, so it was never tried again. Put back, the 147-round values scored 0.156 +- 0.075 against Stockfish at depth 7,
where the same weights without them scored 0.312 +- 0.077: carried over a refit, they count twice.
"""
from __future__ import annotations

from dataclasses import replace

import core.agency.what_a_kind_is_worth as worth
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

#: A match the candidate wins 22 times, draws 5 and loses 13 of 40.
_MATCH = [1.0] * 22 + [0.5] * 5 + [0.0] * 13
BEYOND = [0.0, 9.75, 0.0, 0.0, 30.78]


class _Engine:
    def __init__(self) -> None:
        self.matched = 0

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float((i + n) % 3) for i in range(width + 64)] for n in range(60)]
        gathering = "nodes_per_move_b" not in options
        self.matched = 0 if gathering else self.matched + 1
        results = [0.5] * games if gathering else (list(_MATCH) if self.matched == 1 else [1.0 - p for p in _MATCH])
        return {"samples": [(f, 1.0 if n % 2 else 0.0) for n, f in enumerate(features)],
                "sample_scores": [400 if n % 2 else -400 for n in range(60)],
                "sample_games": [0] * 60, "plies": [80] * len(results), "results": results,
                "final_counts": [[0.0] * len(CHESS.kinds)] * len(results)}


def test_a_refit_kept_has_what_lies_beyond_the_horizon_learned_again(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    # A refit is fitted afresh: nothing of what lies beyond the horizon is in it.
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.weights = replace(rehearsal.weights, beyond=list(BEYOND))
    rehearsal.earned = ["worth", "turn and safety", "moves", "what is left to be taken", worth.BEYOND_THE_HORIZON]
    rehearsal.teacher = worth.HER_LONGER_LOOK
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=40, nodes=10)
    assert rehearsal.last["kept"] and rehearsal.last["tried"] == "where each stands", rehearsal.last
    assert rehearsal.weights.beyond == [], "fitted for the judgement before, it would count twice"
    assert worth.BEYOND_THE_HORIZON not in rehearsal.earned, "it is learned again on top of the judgement now kept"
    assert worth.what_to_try(rehearsal)[0] is not None
