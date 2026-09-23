"""Displacing how good she feels, by her own valence weights and for as long as a turn.

The battery's affect writer raises every feeling, fear as much as joy, and a
push of the feelings is gone by the end of the turn (seed 7, 22 September). The
report experiment and the content run move how good she feels instead, with
these three helpers from core/subject/perturbation.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.phases.affect_update import _NEGATIVE_AFFECT_WEIGHTS, _POSITIVE_AFFECT_WEIGHTS
from core.subject.perturbation import ordinary_span, shift_reference, towards_good

pytestmark = pytest.mark.unit


def test_towards_good_raises_what_she_counts_as_good_and_lowers_what_she_counts_as_bad() -> None:
    moved = towards_good({"joy": 0.3, "fear": 0.3, "a_feeling_nobody_weighs": 0.3}, 0.1)
    assert "joy" in _POSITIVE_AFFECT_WEIGHTS and "fear" in _NEGATIVE_AFFECT_WEIGHTS
    assert moved == pytest.approx({"joy": 0.4, "fear": 0.2, "a_feeling_nobody_weighs": 0.3})


def test_towards_bad_is_the_same_move_the_other_way_and_stays_in_range() -> None:
    moved = towards_good({"joy": 0.05, "fear": 0.98}, -0.1)
    assert moved == pytest.approx({"joy": 0.0, "fear": 1.0})


def test_the_reference_point_moves_against_the_sign_so_the_same_feelings_feel_better() -> None:
    state = SimpleNamespace(affect=SimpleNamespace(mood_baselines={"joy": 0.2, "fear": 0.2, "unweighed": 0.2}))
    assert shift_reference(state, 0.05)
    assert state.affect.mood_baselines == pytest.approx({"joy": 0.15, "fear": 0.25, "unweighed": 0.2})


def test_no_baselines_is_no_shift() -> None:
    assert not shift_reference(SimpleNamespace(affect=SimpleNamespace(mood_baselines={})), 0.05)
    assert not shift_reference(SimpleNamespace(affect=SimpleNamespace()), 0.05)


def test_the_span_is_each_columns_middle_ninety_percent_averaged() -> None:
    rows = np.linspace(0.0, 1.0, 101)
    block = np.column_stack([rows, 2.0 * rows])
    assert ordinary_span(block) == pytest.approx((0.9 + 1.8) / 2)


def test_a_block_too_small_to_have_a_span_has_none() -> None:
    assert ordinary_span(np.zeros((1, 3))) == 0.0
    assert ordinary_span(np.zeros((5, 0))) == 0.0
