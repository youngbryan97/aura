"""A group taught by a new teacher is a new candidate: the tries of the old one do not raise its bar.

Each try of a group is one more chance for a group that does not help to win its match by luck, so each asks more of it
(Bonferroni). LIVE 22 November 2026 her rehearsal at En Croissant had tried where each kind stands 71 times, nearly all
taught by how her games ended, and her longer look then taught candidates scoring up to 0.646 +- 0.072: two standard
errors above even, against a bar of nearly three that the results-taught candidates had raised.
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
        self.calls = 0

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        self.calls += 1
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float((i + n) % 3) for i in range(width + 64)] for n in range(60)]
        gathering = self.calls % 2 == 1
        results = [0.5] * games if gathering else list(_MATCH)
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


def test_the_same_score_on_the_seventy_second_try_of_one_teacher_is_not_enough(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), _rehearsal(worth.RESULTS), games=40, nodes=10)
    assert not rehearsal.last["kept"], rehearsal.last
    assert "where each stands" not in rehearsal.earned
