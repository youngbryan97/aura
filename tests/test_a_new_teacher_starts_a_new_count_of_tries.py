"""A group taught by a new teacher is a new candidate: the pairs the old teacher's candidates played do not count in
its test (core/agency/what_a_kind_is_worth.py ``_decided``).

LIVE 22 November 2026 her rehearsal at En Croissant had tried where each kind stands 71 times, nearly all taught by how
her games ended, and her longer look then taught candidates scoring up to 0.646 +- 0.072. Judged with the results-taught
candidates' record, a new teacher's candidates answer for another teacher's.
"""
from __future__ import annotations

import core.agency.what_a_kind_is_worth as worth
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

#: A match the candidate wins 22 times, draws 5 and loses 13 of 40: 0.613 +- 0.073, about one and a half standard
#: errors above even.
_MATCH = [1.0] * 22 + [0.5] * 5 + [0.0] * 13


class _Engine:
    """Games gathered, then a match the candidate wins by about one and a half standard errors."""

    def __init__(self) -> None:
        self.matched = 0

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float((i + n) % 3) for i in range(width + 64)] for n in range(60)]
        # A match paces its second side; gathering does not. A match is played twice, the candidate named first and
        # then second, and its points are the first named side's.
        gathering = "nodes_per_move_b" not in options
        self.matched = 0 if gathering else self.matched + 1
        results = [0.5] * games if gathering else (list(_MATCH) if self.matched == 1 else [1.0 - p for p in _MATCH])
        return {"samples": [(f, 1.0 if n % 2 else 0.0) for n, f in enumerate(features)],
                "sample_scores": [400 if n % 2 else -400 for n in range(60)],
                "sample_games": [0] * 60, "plies": [80] * len(results), "results": results,
                "final_counts": [[0.0] * len(CHESS.kinds)] * len(results)}


def _rehearsal(teacher: str) -> worth.Rehearsal:
    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth", "turn and safety", "moves", "what is left to be taken"]
    rehearsal.tries = {"where each stands": 71}
    rehearsal.teacher = teacher
    return rehearsal


def test_a_candidate_her_longer_look_taught_is_judged_as_a_first_try(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), _rehearsal(worth.HER_LONGER_LOOK), games=40, nodes=10)
    assert rehearsal.last["kept"], rehearsal.last
    assert "where each stands" in rehearsal.earned
    assert rehearsal.tries[worth.tried_as("where each stands", worth.HER_LONGER_LOOK)] == 1
    assert rehearsal.tries["where each stands"] == 71


def test_the_pairs_one_teacher_s_candidates_lost_count_against_that_teacher_alone(monkeypatch) -> None:
    lost = [0.0] * 60
    for teacher, kept in ((worth.HER_LONGER_LOOK, True), (worth.RESULTS, False)):
        monkeypatch.setattr(worth, "_engine", lambda: _Engine())
        monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
        rehearsal = _rehearsal(teacher)
        rehearsal.trials = {"where each stands": list(lost)}
        rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=40, nodes=10)
        assert rehearsal.last["kept"] is kept, rehearsal.last
