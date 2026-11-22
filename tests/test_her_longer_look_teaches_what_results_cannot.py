"""What results cannot teach her, her longer look does.

Her rehearsal fitted what positions are worth to how her games against herself ended. Games between two weak players
say what a kind of piece holds is worth and little more: measured 22 November 2026 on 141,547 of her positions, where
each piece stands added nothing to how well a fit foretold results it had not seen, and a candidate fitted to them lost
its match 0.107 +- 0.048. Taught by what a search twenty times as long found each position worth, the same group was
earned on its first try (0.657 +- 0.071). A longer look cannot teach what a kind holds is worth: it reads that from the
weights it searches with, and begun from her first guess eight rounds of it earned nothing. So results teach first,
and a group they fail to earn twice, once worth is earned, is taught by her longer look.
"""
from __future__ import annotations

import math
import random

import numpy as np

import core.agency.what_a_kind_is_worth as worth
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


class _Engine:
    """Her compiled search as rehearsal meets it: games gathered, then a match that the candidate does not win."""

    def __init__(self, *, has_longer_looks: bool = True) -> None:
        self.asked: list[dict] = []
        self.has_longer_looks = has_longer_looks

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        if "deep_nodes" in options and not self.has_longer_looks:
            raise TypeError("self_play() got an unexpected keyword argument 'deep_nodes'")
        self.asked.append(options)
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float(i % 3) for i in range(width + 64)] for _ in range(60)]
        return {"samples": [(f, 1.0 if n % 2 else 0.0) for n, f in enumerate(features)],
                "sample_scores": [400 if n % 2 else -400 for n in range(60)],
                "sample_games": [0] * 60, "plies": [80] * games, "results": [0.5] * games,
                "final_counts": [[0.0] * len(CHESS.kinds)] * games}


def _a_rehearsal_with_worth_earned() -> worth.Rehearsal:
    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth"]
    return rehearsal


def test_her_scores_are_read_on_the_scale_her_results_show() -> None:
    rng = random.Random(3)
    scores = [rng.uniform(-900, 900) for _ in range(4000)]
    results = [1.0 if rng.random() < 1 / (1 + math.exp(-v / 150)) else 0.0 for v in scores]
    assert 100 < worth.calibrated_scale(scores, results) < 230


def test_a_group_results_fail_twice_is_taught_by_her_longer_look(monkeypatch) -> None:
    engine = _Engine()
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = _a_rehearsal_with_worth_earned()
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=4, nodes=10, seed=1)
    assert rehearsal.teacher == worth.RESULTS
    assert "deep_nodes" not in engine.asked[0]
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=4, nodes=10, seed=2)
    assert rehearsal.teacher == worth.HER_LONGER_LOOK
    # What results labelled is let go: her longer look teaches on positions of its own.
    assert rehearsal.samples_y == []
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=4, nodes=10, seed=3)
    gathered = engine.asked[-2]
    assert gathered["deep_nodes"] == worth.LONGER_LOOK_NODES
    # Each position is labelled by what the longer look found, read as a chance to win: a position it found good above
    # even, a bad one below, and none as certain as a result.
    labels = np.array(rehearsal.samples_y)
    assert (labels[1::2] > 0.5).all() and (labels[0::2] < 0.5).all()
    assert 0.0 < labels.min() and labels.max() < 1.0


def test_results_keep_teaching_until_worth_is_earned(monkeypatch) -> None:
    engine = _Engine()
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = worth.a_first_rehearsal(CHESS)
    for seed in range(4):
        rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=4, nodes=10, seed=seed)
    assert rehearsal.teacher == worth.RESULTS
    assert all("deep_nodes" not in asked for asked in engine.asked)


def test_a_build_without_longer_looks_is_taught_by_results(monkeypatch) -> None:
    engine = _Engine(has_longer_looks=False)
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: worth.first_weights(family))
    rehearsal = _a_rehearsal_with_worth_earned()
    rehearsal.teacher = worth.HER_LONGER_LOOK
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=4, nodes=10, seed=1)
    assert set(rehearsal.samples_y) <= {0.0, 1.0}
