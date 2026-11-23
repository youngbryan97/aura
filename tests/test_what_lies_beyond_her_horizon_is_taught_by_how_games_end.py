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
    assert worth is None and "no better" in why, (worth, why)


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
