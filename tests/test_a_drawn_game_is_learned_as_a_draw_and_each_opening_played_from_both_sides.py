"""A game her rules ended level is learned as a draw; only one stopped unfinished has its material read as an estimate.

From the suggestions of 23 November 2026: rehearsal read every level game's result from the material where it
stopped, so a draw by repetition or by count became a target of 0.791 or 0.339, teaching that holding material wins
where it drew. And a candidate's match drew a fresh opening for each game, so which opening a game began from was
part of its result; each opening is now played twice, the candidate on each side of it.
"""
from __future__ import annotations

import core.agency.what_a_kind_is_worth as worth
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

QUEEN = list(CHESS.kinds).index("Q")


def _counts(queens: int) -> list[float]:
    counts = [0.0] * len(CHESS.kinds)
    counts[QUEEN] = float(queens)
    return counts


class _Engine:
    """Two games gathered, both level: the first ended by her rules at ply 80, a queen up; the second ran to the most
    plies a game is played to, a queen down. Then a match, played as it is asked for."""

    def __init__(self, first_sides_points=None) -> None:
        self.matches: list[dict] = []
        self.first_sides_points = first_sides_points

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float((i + n) % 3) for i in range(width + 64)] for n in range(4)]
        if "nodes_per_move_b" not in options:
            return {"samples": [(f, 0.5) for f in features], "sample_games": [0, 0, 1, 1],
                    "plies": [80, worth.GAME_PLIES_AT_MOST - 1], "results": [0.5, 0.5],
                    "final_counts": [_counts(1), _counts(-1)]}
        self.matches.append({"a": weights_a, "b": weights_b, "games": games, **options})
        # The first named side moves first in the even games.
        first_side = list(self.first_sides_points(games))
        return {"samples": [], "sample_games": [], "plies": [80] * games,
                "results": [p if game % 2 == 0 else 1.0 - p for game, p in enumerate(first_side)],
                "final_counts": [_counts(0)] * games}


def test_a_game_her_rules_ended_level_is_learned_as_a_draw_and_an_unfinished_one_by_its_material(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: None)
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), worth.a_first_rehearsal(CHESS), games=2, nodes=10)
    assert rehearsal.samples_y[:2] == [0.5, 0.5], "drawn by her rules: a draw"
    assert all(label < 0.5 for label in rehearsal.samples_y[2:]), "unfinished a queen down: read as worse than even"


def test_what_practice_holds_is_bounded_by_its_size_and_the_latest_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: None)
    rehearsal = worth.a_first_rehearsal(CHESS)
    width = len(worth._columns(CHESS, list(worth.GROUPS))) + 64
    monkeypatch.setattr(worth, "SAMPLES_HELD_AT_MOST_BYTES", 6 * width * 4)
    for seed in range(3):
        rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=2, nodes=10, seed=seed)
    assert len(rehearsal.samples_y) == 6 and len(rehearsal.samples_x) == 6


def test_each_opening_of_a_match_is_played_with_the_candidate_on_each_side(monkeypatch) -> None:
    def first_to_move_wins(games):
        return [1.0] * games

    engine = _Engine(first_to_move_wins)
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), worth.a_first_rehearsal(CHESS), games=40, nodes=10)
    ours, swapped = engine.matches[-2:]
    assert ours["seed"] == swapped["seed"] and ours["games"] == swapped["games"] == 20
    assert (ours["a"], ours["b"]) == (swapped["b"], swapped["a"])
    # Moving first is worth a win, and each opening was played from both sides: the candidate is no better.
    assert rehearsal.last["scored"] == 0.5 and not rehearsal.last["kept"]
