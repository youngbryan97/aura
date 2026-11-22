"""How the pieces stand together is learned from what her first look missed of what her longer look found.

Once the groups a weighted sum can hold are earned, her rehearsal keeps the quiet positions her longer look searched and
learns a judgement over which kind stands where, from the side to move. It is tried as a group once it foretells
positions it was not fitted on better than her first look, and kept only where it wins its match. Measured 22 November
2026 on 84,299 such positions against Stockfish held to depth 5, 40 games on each of four seeds: 0.447, 0.474, 0.550
and 0.538 with it, 0.412, 0.450, 0.350 and 0.425 without.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import core.agency.what_a_kind_is_worth as worth
from core.agency.pieces_by_their_kind import FIRST, after, moves
from core.agency.searching_by_kind import Weights, described, first_weights, placed
from core.cognition.a_quick_judgement_from_slow_ones import learned_from_slow_ones
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

engine = pytest.importorskip("aura_board_search")


def _played(order: list[str], cells: list[list[int]]) -> dict:
    n = len(cells)
    return {"sample_cells": cells, "sample_static": [10] * n, "sample_quiet": [True] * n, "sample_turns": [0] * n,
            "sample_scores": [200 if i % 2 else -200 for i in range(n)], "kind_order": order,
            "sample_games": list(range(n)), "samples": [([0.0], 1.0 if i % 2 else 0.0) for i in range(n)]}


def test_a_position_is_kept_by_its_kinds_whatever_order_a_round_counts_them_in() -> None:
    kept = worth.LookedAtLonger()
    squares = 4
    # Round one counts kinds A then B; round two B then A, with a kind C it had not seen.
    kept.add(_played(["A", "B"], [[0 * 8 + 1, 1 * 8 + 4 + 2]]), 1, 2, 2)
    kept.add(_played(["B", "C", "A"], [[0 * 8 + 1, 2 * 8 + 4 + 2]]), 2, 2, 2)
    assert kept.kinds == ["A", "B", "C"]
    a_own_1 = 0 * 2 * squares + 1
    b_theirs_2 = (1 * 2 + 1) * squares + 2
    b_own_1 = 1 * 2 * squares + 1
    a_theirs_2 = (0 * 2 + 1) * squares + 2
    assert sorted(kept.parts[0].tolist()) == [a_own_1, b_theirs_2]
    assert sorted(kept.parts[1].tolist()) == [a_theirs_2, b_own_1]


def test_what_she_gathered_comes_back_from_disk_the_same() -> None:
    kept = worth.LookedAtLonger()
    kept.add(_played(["A", "B"], [[1, 12], [3]]), 1, 2, 2)
    back = worth.LookedAtLonger.from_bytes(kept.as_bytes())
    assert back.kinds == kept.kinds and back.squares == kept.squares and back.columns == kept.columns
    assert [p.tolist() for p in back.parts] == [p.tolist() for p in kept.parts]
    assert back.slow == kept.slow and back.game == kept.game


class _Engine:
    """Games gathered, then a match the candidate wins 30 of 40."""

    def __init__(self) -> None:
        self.calls = 0

    def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
        self.calls += 1
        width = len(worth._columns(CHESS, list(worth.GROUPS)))
        features = [[float((i + n) % 3) for i in range(width + 64)] for n in range(60)]
        results = [0.5] * games if self.calls % 2 else [1.0] * 30 + [0.0] * 10
        return {"samples": [(f, 1.0 if n % 2 else 0.0) for n, f in enumerate(features)],
                "sample_scores": [400 if n % 2 else -400 for n in range(60)],
                "sample_games": [0] * 60, "plies": [80] * len(results), "results": results,
                "final_counts": [[0.0] * len(CHESS.kinds)] * len(results)}


def _with_every_weighted_group_earned() -> worth.Rehearsal:
    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.earned = [group for group in worth.GROUPS if group != worth.TOGETHER]
    rehearsal.teacher = worth.HER_LONGER_LOOK
    return rehearsal


def test_until_something_learned_foretells_better_nothing_is_tried(monkeypatch) -> None:
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth.LookedAtLonger, "judgement", lambda self, seed=0: None)
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), _with_every_weighted_group_earned(), games=40, nodes=10)
    assert rehearsal.last["kept"] is False and "foretells" in rehearsal.last["why"]
    assert not any(worth.TOGETHER in key for key in rehearsal.tries)


def test_a_learned_judgement_that_wins_its_match_is_kept_and_a_refit_keeps_it(monkeypatch) -> None:
    learned = {"hidden": 1, "columns": 8, "own": {}, "theirs": {}, "first_bias": [0.0], "out": [0.0], "out_bias": 0.0,
               "held_out": [0.01, 0.009]}
    monkeypatch.setattr(worth, "_engine", lambda: _Engine())
    monkeypatch.setattr(worth.LookedAtLonger, "judgement", lambda self, seed=0: learned)
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), _with_every_weighted_group_earned(), games=40, nodes=10)
    assert rehearsal.last["kept"], rehearsal.last
    assert worth.TOGETHER in rehearsal.earned and rehearsal.weights.learned == learned
    monkeypatch.setattr(worth, "fitted", lambda family, xs, ys, groups: first_weights(family))
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=40, nodes=10, seed=2)
    assert rehearsal.weights.learned == learned


def test_one_that_foretells_no_better_than_the_last_tried_is_not_tried(monkeypatch) -> None:
    def learned_with(after: float) -> dict:
        return {"hidden": 1, "columns": 8, "own": {}, "theirs": {}, "first_bias": [0.0], "out": [0.0],
                "out_bias": 0.0, "held_out": [0.01, after]}

    class _LosesEveryMatch(_Engine):
        def self_play(self, *args, **options):
            played = super().self_play(*args, **options)
            if self.calls % 2 == 0:
                played["results"] = [0.0] * 40
            return played

    engine = _LosesEveryMatch()
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    rehearsal = _with_every_weighted_group_earned()
    for after, tried in ((0.009, True), (0.009, False), (0.0095, False), (0.008, True)):
        monkeypatch.setattr(worth.LookedAtLonger, "judgement", lambda self, seed=0, a=after: learned_with(a))
        rehearsal = worth.rehearse(CHESS, CHESS.the_start(), rehearsal, games=40, nodes=10)
        assert ("scored" in rehearsal.last) is tried, rehearsal.last
    assert rehearsal.tries[worth.tried_as(worth.TOGETHER, worth.HER_LONGER_LOOK)] == 2


def test_her_compiled_search_judges_as_the_learned_layer_does() -> None:
    rng = np.random.default_rng(1)
    kinds = sorted(CHESS.kinds)
    squares = CHESS.rows * CHESS.columns
    hidden = 4
    first = rng.normal(0, 20, (len(kinds) * 2 * squares, hidden))
    bias, out, out_bias = rng.normal(0, 10, hidden), rng.normal(0, 0.5, hidden), 3.0
    learned = {"hidden": hidden, "columns": CHESS.columns, "first_bias": bias.tolist(), "out": out.tolist(),
               "out_bias": out_bias,
               "own": {k: first[(i * 2) * squares:(i * 2 + 1) * squares].tolist() for i, k in enumerate(kinds)},
               "theirs": {k: first[(i * 2 + 1) * squares:(i * 2 + 2) * squares].tolist() for i, k in enumerate(kinds)}}
    plain = first_weights(CHESS)
    with_it = Weights(**{**json.loads(plain.as_json()), "learned": learned})
    position = CHESS.the_start()
    for move in [moves(CHESS, position)[3], None]:
        if move is not None:
            position = after(CHESS, position, move)
        h = bias.copy()
        for (row, column), (side, kind) in position.cells.items():
            r = row if position.turn == FIRST else CHESS.rows - 1 - row
            h += first[(kinds.index(kind) * 2 + (side != position.turn)) * squares + r * CHESS.columns + column]
        expected = out_bias + float(np.maximum(h, 0) @ out)
        description = described(CHESS)
        added = (engine.evaluate(description, placed(position), with_it.as_json())
                 - engine.evaluate(description, placed(position), plain.as_json()))
        assert abs(added - expected) <= 0.5 + 1e-6, (added, expected)


def test_a_layer_learns_what_a_sum_of_parts_misses() -> None:
    # Parts 0 and 1 together are worth much more than either alone: a sum of parts cannot say it, a layer can.
    rng = np.random.default_rng(4)
    parts = np.full((6000, 2), -1, dtype=np.int64)
    present = rng.random((6000, 2)) < 0.5
    parts[present[:, 0], 0] = 0
    parts[present[:, 1], 1] = 1
    slow = np.where(present.all(axis=1), 300.0, 0.0) + rng.normal(0, 20, 6000)
    learned = learned_from_slow_ones(parts, np.zeros(6000), slow, np.arange(6000) // 20, inputs=2, scale=200.0)
    assert learned is not None and learned.held_out_after < learned.held_out_before / 2
    both, one = learned.of([np.array([0, 1])])[0], learned.of([np.array([0])])[0]
    assert both > 200 and abs(one) < 100


def test_how_the_pieces_stand_together_is_kept_beside_her_record_and_comes_back() -> None:
    import core.skills.screen_pursuit_board_of_pieces as pursuit
    from core.perception.pieces_by_their_look import Looks
    from core.runtime.what_she_learned import recall

    learned = {"hidden": 1, "columns": 8, "own": {"K": [[1.5]] * 64}, "theirs": {}, "first_bias": [0.0],
               "out": [1.0], "out_bias": 0.0}
    rehearsal = _with_every_weighted_group_earned()
    rehearsal.earned.append(worth.TOGETHER)
    rehearsal.weights.learned = learned
    rehearsal.looked_at_longer.add(_played(["K", "P"], [[1, 140], [3]]), 1, 8, 8)
    pursuit._keep_what_she_learned("A Board App", Looks(), None, rehearsal)
    # The record itself stays small: the learned judgement and its positions are beside it.
    assert "learned" not in recall(pursuit._her_record("A Board App"))["worth"]["weights"]
    back = pursuit._what_she_kept("A Board App")["rehearsal"]
    assert back.weights.learned == learned
    assert [p.tolist() for p in back.looked_at_longer.parts] == [p.tolist() for p in rehearsal.looked_at_longer.parts]


def test_lent_to_a_board_seen_from_its_other_side_each_row_of_squares_turns_round() -> None:
    rows = [[float(square)] for square in range(4)]
    learned = {"hidden": 1, "columns": 2, "own": {"a": rows}, "theirs": {"a": rows}, "first_bias": [0.0],
               "out": [1.0], "out_bias": 0.0}
    lent = Weights(worth={"a": 1.0}, learned=learned).under_other_names({"a": "x"}, turned_round=True)
    # Squares 0 1 / 2 3 seen from the other side read 1 0 / 3 2.
    assert lent.learned["own"]["x"] == [[1.0], [0.0], [3.0], [2.0]]
    assert lent.learned["theirs"]["x"] == [[1.0], [0.0], [3.0], [2.0]]
    assert Weights(worth={"a": 1.0}, learned=learned).under_other_names({"a": "x"}).learned["own"]["x"] == rows


def test_a_slower_judgement_is_matched_with_the_time_it_costs(monkeypatch) -> None:
    # A game is on a clock: a judgement that slows her search looks at fewer positions a move in its match.
    import time

    class _Paced(_Engine):
        def __init__(self) -> None:
            super().__init__()
            self.asked: list[tuple] = []

        def search(self, description, begin, weights, seconds, deepest, hash_mb, nodes):
            time.sleep(0.02 if '"learned": null' in weights else 0.04)
            return {"nodes": nodes}

        def self_play(self, description, begin, weights_a, weights_b, games, nodes, **options):
            self.asked.append((nodes, options.get("nodes_per_move_b", 0)))
            return super().self_play(description, begin, weights_a, weights_b, games, nodes, **options)

    engine = _Paced()
    learned = {"hidden": 1, "columns": 8, "own": {}, "theirs": {}, "first_bias": [0.0], "out": [0.0], "out_bias": 0.0,
               "held_out": [0.01, 0.009]}
    monkeypatch.setattr(worth, "_engine", lambda: engine)
    monkeypatch.setattr(worth.LookedAtLonger, "judgement", lambda self, seed=0: learned)
    rehearsal = worth.rehearse(CHESS, CHESS.the_start(), _with_every_weighted_group_earned(), games=40, nodes=1000)
    candidates, current = engine.asked[-1]
    assert current == 1000 and 400 < candidates < 650, engine.asked
    assert 0.4 < rehearsal.last["pace"] < 0.65
