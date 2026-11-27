"""Her self-play counts a position quiet where a look at the takes alone changes her judgement by no more than one
quiet move could (rust_extensions/aura_board_search ``self_play``, ``sample_quiet``): only quiet positions teach her how
the pieces stand together (core/agency/what_a_kind_is_worth.py ``LookedAtLonger``).

LIVE-like 26 November 2026 a position counted as quiet only where the takes left her judgement exactly as it was, and
compared her judgement uncorrected with a look that began from it corrected. With her kept weights, whose learned layer
is not a plain sum of its parts, three in a hundred positions of her model of chess were quiet; with her first guess,
nine in ten. What she learned how the pieces stand together from was what was left.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from core.agency.searching_by_kind import _compiled, described, first_weights, placed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME

pytestmark = pytest.mark.unit

engine = _compiled()


def _with_a_layer(family):
    """Her first guess with a learned layer over where each kind stands, of some size: not a plain sum of its parts."""
    rng = random.Random(7)
    squares = family.rows * family.columns
    rows = {letter: [[rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0)] for _ in range(squares)]
            for letter in family.kinds}
    learned = {"hidden": 2, "columns": family.columns, "own": rows, "theirs": rows, "first_bias": [10.0, -10.0],
               "out": [1.0, -1.0], "out_bias": 0.0}
    return replace(first_weights(family), learned=learned)


def _quiet_share(family, weights) -> float:
    played = engine.self_play(described(family), placed(family.the_start()), weights.as_json(), weights.as_json(), 6,
                              1500, random_plies=4, seed=3, max_plies=200, sample_every=1)
    quiet = played["sample_quiet"]
    return sum(quiet) / len(quiet)


@pytest.mark.skipif(engine is None, reason="the compiled search is not built")
@pytest.mark.parametrize("family", [CHESS, NOBODYS_GAME], ids=["chess", "nobody's game"])
def test_a_judgement_that_is_not_a_plain_sum_leaves_most_positions_quiet(family) -> None:
    plain = _quiet_share(family, first_weights(family))
    layered = _quiet_share(family, _with_a_layer(family))
    assert plain > 0.6
    assert layered > 0.6
