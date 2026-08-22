"""A NaN in the substrate became full intensity, not an error.

`clamp` was `max(lo, min(hi, x))`. Python's `min` returns its first argument
whenever the comparison is False, and every comparison with NaN is False, so
`min(1.0, nan)` is `1.0` and `clamp(nan)` was `1.0`. A NaN arriving on an
affect channel did not surface as a NaN. It surfaced as that channel at
maximum, which is the worst available answer and the hardest to notice.

Every numpy path in this repository already guarded for this —
`sqlite_vector_store` and `rl_glue` both call `np.nan_to_num(..., nan=0.0)`.
The substrate is scalar and dict-shaped, so it never picked up the same
convention.
"""
from __future__ import annotations

import math

from core.phenomenal_substrate.maths import (
    add,
    bound01,
    bound_signed,
    clamp,
    clamp_signed,
    finite,
    l2,
    mix,
    normalize_sum,
    sigmoid,
    tanh,
    weighted_error,
)

NAN = float("nan")
INF = float("inf")


def test_the_old_behaviour_is_named():
    """The comparison that produced it, so the fix is not mistaken for style."""
    assert min(1.0, NAN) == 1.0
    assert max(0.0, min(1.0, NAN)) == 1.0


def test_a_nan_intensity_becomes_neutral_not_maximal():
    assert clamp(NAN) == 0.0
    assert clamp_signed(NAN) == 0.0


def test_infinities_still_saturate_at_the_bound():
    """An infinity carries direction; only a NaN carries nothing."""
    assert clamp(INF) == 1.0
    assert clamp(-INF) == 0.0
    assert clamp_signed(INF) == 1.0
    assert clamp_signed(-INF) == -1.0


def test_ordinary_values_are_untouched():
    assert clamp(0.25) == 0.25
    assert clamp_signed(-0.5) == -0.5
    assert clamp(1.5) == 1.0
    assert clamp(-0.5) == 0.0


def test_a_nan_does_not_propagate_through_a_vector():
    bounded = bound01({"arousal": NAN, "valence": 0.4})
    assert bounded == {"arousal": 0.0, "valence": 0.4}

    signed = bound_signed({"valence": NAN, "dominance": -2.0})
    assert signed == {"valence": 0.0, "dominance": -1.0}


def test_a_nan_does_not_poison_an_aggregate():
    assert l2({"a": NAN, "b": 3.0, "c": 4.0}) == 5.0
    assert not math.isnan(l2({"a": NAN}))


def test_a_nan_does_not_poison_a_blend():
    blended = mix({"a": 1.0}, {"a": NAN}, 0.5)
    assert blended["a"] == 0.5
    # A non-finite rate falls back to no blending rather than an undefined one.
    assert mix({"a": 1.0}, {"a": 0.0}, NAN)["a"] == 1.0


def test_a_nan_does_not_poison_a_distribution():
    distribution = normalize_sum({"a": NAN, "b": 1.0, "c": 1.0})
    assert distribution == {"a": 0.0, "b": 0.5, "c": 0.5}
    assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_a_nan_does_not_poison_prediction_error():
    error = weighted_error({"a": NAN}, {"a": 1.0}, {"a": 2.0})
    assert error["a"] == 2.0


def test_the_squashing_functions_stay_in_range():
    assert 0.0 <= sigmoid(NAN) <= 1.0
    assert sigmoid(NAN) == sigmoid(0.0)
    assert tanh(NAN) == 0.0
    assert -1.0 <= tanh(INF) <= 1.0


def test_a_non_numeric_value_is_treated_as_missing():
    assert finite("not a number") == 0.0
    assert finite(None) == 0.0
    assert finite("0.5") == 0.5


def test_addition_of_two_nans_is_neutral_not_nan():
    assert add({"a": NAN}, {"a": NAN}) == {"a": 0.0}
