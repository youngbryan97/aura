"""What lies beyond her search's horizon is counted from the rules she learned and taught by how her games ended.

core/agency/what_lies_beyond_the_horizon.py. Bryan asked on 22 November 2026 for a judgement that sees what her search
cannot reach, in the most general form there is. These check the counting on positions whose answer is plain, and that
the fit learns only what results show beyond her search's own score.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.what_a_kind_is_worth import BEYOND_THE_HORIZON, GROUPS, TOGETHER
from core.agency.what_lies_beyond_the_horizon import BEYOND, WhatGamesCameTo, beyond_at
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, DRAUGHTS, from_fen


def _games(n: int, worth_of_first: float, seed: int = 0) -> tuple[dict, list[float]]:
    """Positions whose results depend on her search's score and, by ``worth_of_first``, on the first quantity beyond."""
    rng = np.random.default_rng(seed)
    at = beyond_at(CHESS)
    samples, scores, games, came_to = [], [], [], []
    for i in range(n):
        beyond = rng.normal(0, 1, len(BEYOND))
        score = float(rng.normal(0, 150))
        p = 1 / (1 + np.exp(-(score / 150 + worth_of_first * beyond[0])))
        result = float(rng.random() < p)
        samples.append(([0.0] * at + list(beyond), result))
        scores.append(score)
        games.append(i // 10)
        came_to.append(result)
    return {"samples": samples, "sample_scores": scores, "sample_games": games}, came_to


def test_what_results_show_beyond_her_search_is_learned() -> None:
    store = WhatGamesCameTo()
    played, came_to = _games(6_000, worth_of_first=0.8)
    store.add(played, came_to, 0, beyond_at(CHESS))
    worth, why = store.worth()
    assert worth is not None, why
    assert worth[0] > 0 and all(abs(worth[0]) > 3 * abs(w) for w in worth[1:]), worth


def test_nothing_is_learned_where_results_show_nothing_beyond_her_search() -> None:
    store = WhatGamesCameTo()
    played, came_to = _games(6_000, worth_of_first=0.0, seed=3)
    store.add(played, came_to, 0, beyond_at(CHESS))
    worth, why = store.worth()
    assert worth is None and "no quantity foretells" in why, (worth, why)


def test_what_her_games_came_to_is_kept_whole() -> None:
    store = WhatGamesCameTo()
    played, came_to = _games(400, worth_of_first=0.5)
    store.add(played, came_to, 2, beyond_at(CHESS))
    again = WhatGamesCameTo.from_bytes(store.as_bytes())
    assert len(again.result) == len(store.result) == 400
    assert np.allclose(np.array(again.beyond), np.array(store.beyond)) and again.game == store.game


def test_it_is_tried_once_how_the_pieces_stand_together_is_earned() -> None:
    assert GROUPS.index(BEYOND_THE_HORIZON) == GROUPS.index(TOGETHER) + 1


def test_the_compiled_search_counts_an_unopposed_runner_and_judges_it() -> None:
    engine = pytest.importorskip("aura_board_search")
    from core.agency.searching_by_kind import described, first_weights, placed

    # A pawn of the first side on its sixth row with nothing ahead of it, kings apart.
    free = from_fen("4k3/8/P7/8/8/8/8/4K3 w - -")
    blocked = from_fen("4k3/p7/P7/8/8/8/8/4K3 w - -")
    weights = first_weights(CHESS)
    plain = engine.evaluate(described(CHESS), placed(free), weights.as_json())
    weights.beyond = [400.0, 0.0, 0.0, 0.0, 0.0]
    judged = engine.evaluate(described(CHESS), placed(free), weights.as_json())
    # Five rows of seven come: (5/7)^2 of the worth.
    assert judged - plain == round(400 * (5 / 7) ** 2), (plain, judged)
    stopped = engine.evaluate(described(CHESS), placed(blocked), weights.as_json())
    stopped_plain = engine.evaluate(described(CHESS), placed(blocked), first_weights(CHESS).as_json())
    assert stopped == stopped_plain, "a runner with the other side's runner ahead of it is not unopposed"


def test_draughts_men_are_runners_and_kings_are_not() -> None:
    engine = pytest.importorskip("aura_board_search")
    from core.agency.pieces_by_their_kind import FIRST, Position
    from core.agency.searching_by_kind import described, first_weights, placed

    alone = Position(cells={(5, 1): (FIRST, "M"), (0, 0): (1 - FIRST, "K")}, turn=FIRST, unmoved=frozenset())
    weights = first_weights(DRAUGHTS)
    plain = engine.evaluate(described(DRAUGHTS), placed(alone), weights.as_json())
    weights.beyond = [700.0, 0.0, 0.0, 0.0, 0.0]
    assert engine.evaluate(described(DRAUGHTS), placed(alone), weights.as_json()) - plain == round(700 * (5 / 7) ** 2)


def test_what_lies_beyond_is_in_the_weights_she_plays_by() -> None:
    # Earned in rehearsal and left out of the weights her moves are searched with, it would never be played.
    import core.agency.playing_pieces_by_kind as playing
    from core.agency.what_a_kind_is_worth import a_first_rehearsal

    rehearsal = a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth", BEYOND_THE_HORIZON]
    rehearsal.weights.beyond = [30.0, -10.0, 5.0, 3.0, 5.0]
    assert playing._Rehearsing(rehearsal=rehearsal).weights_for(CHESS).beyond == [30.0, -10.0, 5.0, 3.0, 5.0]


def test_a_weight_on_each_feature_is_added_where_her_search_reads_it() -> None:
    # A weight learned on where each kind stands, added to her weights, changes her compiled judgement of a position by
    # exactly those weights summed over its pieces, each from its own side's frame.
    engine = pytest.importorskip("aura_board_search")
    from core.agency.pieces_by_their_kind import FIRST
    from core.agency.searching_by_kind import described, first_weights, placed
    from core.agency.what_a_kind_is_worth import added_to

    position = from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -")
    weights = first_weights(CHESS)
    letters, squares = list(CHESS.kinds), CHESS.rows * CHESS.columns
    base = 2 * len(letters) + 2
    extra = np.zeros(beyond_at(CHESS) + len(BEYOND))
    extra[base:base + len(letters) * squares] = np.random.default_rng(0).integers(-5, 6, len(letters) * squares)
    more = added_to(weights, CHESS, extra)
    expected = 0.0
    for (row, column), (side, letter) in position.cells.items():
        own_row = row if side == FIRST else CHESS.rows - 1 - row
        value = extra[base + letters.index(letter) * squares + own_row * CHESS.columns + column]
        expected += value if side == position.turn else -value
    judged = engine.evaluate(described(CHESS), placed(position), more.as_json())
    before = engine.evaluate(described(CHESS), placed(position), weights.as_json())
    assert judged - before == round(expected), (judged, before, expected)


def test_groups_taught_by_how_games_ended_take_turns() -> None:
    from core.agency.what_a_kind_is_worth import SLOW_AND_DECIDING, TAUGHT_BY_RESULTS

    assert set(TAUGHT_BY_RESULTS) == {BEYOND_THE_HORIZON, SLOW_AND_DECIDING}
    assert GROUPS[-2:] == (BEYOND_THE_HORIZON, SLOW_AND_DECIDING)


def test_what_changes_slowly_through_her_games_is_found_in_their_positions() -> None:
    rng = np.random.default_rng(7)
    at = beyond_at(CHESS)
    width = at + len(BEYOND)
    samples, scores, games, came_to = [], [], [], []
    for game in range(300):
        slow = rng.normal() + np.cumsum(rng.normal(0, 0.05, 20))
        won = float(rng.random() < 1 / (1 + np.exp(-2.5 * slow.mean())))
        for step in range(20):
            row = rng.normal(0, 1, width)
            row[at + 3] = slow[step]
            samples.append((list(row), won))
            # Her search's score says a little of how the game ends, as a real one does, and nothing of the slow number.
            scores.append(float(rng.normal(0, 100) + 120 * (won - 0.5)))
            games.append(game)
            came_to.append(won)
    store = WhatGamesCameTo()
    store.add({"samples": samples, "sample_scores": scores, "sample_games": games}, came_to, 0, at)
    found, why = store.slow_and_deciding()
    assert found is not None, why
    assert found[at + 3] > 0 and abs(found[at + 3]) > 5 * np.median(np.abs(found)), found[at:]


def test_once_the_rest_are_earned_how_the_pieces_stand_together_is_learned_again_in_turn() -> None:
    from core.agency.what_a_kind_is_worth import SLOW_AND_DECIDING, a_first_rehearsal, what_to_try

    rehearsal = a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth", "turn and safety"]
    assert what_to_try(rehearsal) == ("moves", False)
    rehearsal.earned = [g for g in GROUPS if g not in (BEYOND_THE_HORIZON, SLOW_AND_DECIDING)]
    tried = []
    for rounds in range(6):
        rehearsal.rounds = rounds
        tried.append(what_to_try(rehearsal))
    assert {t for t, _again in tried} == {BEYOND_THE_HORIZON, SLOW_AND_DECIDING, TOGETHER}
    assert all(again == (t == TOGETHER) for t, again in tried)
