"""The surprise counter never left zero.

`was_surprising` was the composite error against a fixed 0.4. The composite is
`0.3 * valence_error + 0.4 * drive_error + 0.3 * focus_error`, and the last two
are exactly zero whenever she names her dominant drive and her focus source
correctly. Her dominant drive is stable for thousands of turns, so the composite
lived at three tenths of a bounded valence error and could not reach four
tenths. Over a 320-turn recording the surprise count and the surprise rate were
both exactly zero while the confirmation count moved: the self domain carried
one tail of its own error distribution and a constant where the other should be.

Confirmation in that file is already the lower tail read against her own
history. This is the upper one.
"""

from __future__ import annotations

import pytest

from core.consciousness.self_prediction import SelfPredictionLoop


class _Error:
    def __init__(self, composite: float) -> None:
        self.composite_error = composite


def _loop(history: list[float]) -> SelfPredictionLoop:
    loop = SelfPredictionLoop.__new__(SelfPredictionLoop)
    from collections import deque

    loop._error_history = deque(
        (_Error(one) for one in history), maxlen=SelfPredictionLoop._HISTORY_SIZE
    )
    return loop


def test_before_she_has_a_distribution_the_old_bar_answers():
    short = _loop([0.1] * (SelfPredictionLoop._ENOUGH_ERRORS - 1))
    assert short._is_surprising(0.5) is True
    assert short._is_surprising(0.3) is False


def test_an_error_like_her_usual_ones_is_not_a_surprise():
    loop = _loop([0.20, 0.22, 0.18, 0.21, 0.19, 0.23, 0.20, 0.21, 0.19, 0.20])
    assert loop._is_surprising(0.21) is False


def test_an_error_well_past_her_spread_is_a_surprise():
    loop = _loop([0.20, 0.22, 0.18, 0.21, 0.19, 0.23, 0.20, 0.21, 0.19, 0.20])
    assert loop._is_surprising(0.40) is True


def test_a_small_error_can_surprise_her_if_she_is_usually_exact():
    """The bar is hers. Being wrong by a tenth is news to someone never wrong."""
    loop = _loop([0.0] * 20)
    assert loop._is_surprising(0.1) is True


def test_a_large_error_is_not_news_when_she_is_always_that_wrong():
    loop = _loop([0.9] * 20)
    assert loop._is_surprising(0.9) is False


def test_the_counter_can_reach_the_composite_the_old_bar_could_not():
    """Three tenths of a bounded valence error never reached four tenths."""
    below_the_old_bar = 0.3 * 0.8
    assert below_the_old_bar < SelfPredictionLoop._SURPRISE_THRESHOLD
    loop = _loop([0.03, 0.02, 0.04, 0.03, 0.02, 0.05, 0.03, 0.04, 0.02, 0.03])
    assert loop._is_surprising(below_the_old_bar) is True


def test_both_tails_now_move_over_a_run():
    """Neither a constant zero nor a surprise on every turn."""
    import random

    rng = random.Random(5)
    loop = _loop([])
    fired = 0
    for _ in range(400):
        composite = min(1.0, max(0.0, rng.gauss(0.25, 0.08)))
        if loop._is_surprising(composite):
            fired += 1
        loop._error_history.append(_Error(composite))
    assert 0 < fired < 400
    assert 0.05 < fired / 400 < 0.35
