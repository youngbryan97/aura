"""Exact integer scores preserve rational sums, order and outward bounds."""

import math
import random
from fractions import Fraction

import pytest

from core.learning.dyadic_scores import DyadicScores


@pytest.mark.parametrize("seed", range(20))
def test_all_linear_combinations_keep_exact_rational_meaning(seed):
    rng = random.Random(seed)
    values = [math.ldexp(rng.uniform(-1, 1), rng.randrange(-1000, 1001)) for _ in range(30)]
    scale = DyadicScores.from_values(values)
    assert scale.denominator & (scale.denominator - 1) == 0
    assert [Fraction(x, scale.denominator) for x in scale.integers] == list(map(Fraction, values))
    ordered = []
    for _ in range(50):
        coefficients = [rng.randrange(-10, 11) for _ in values]
        actual = sum(a * b for a, b in zip(coefficients, scale.integers, strict=True))
        expected = sum((a * Fraction(b) for a, b in zip(coefficients, values, strict=True)), Fraction(0))
        assert Fraction(actual, scale.denominator) == expected
        bound = scale.upper_float(actual)
        assert bound == math.inf or Fraction(bound) >= expected
        ordered.append((actual, expected))
    assert sorted(ordered) == sorted(ordered, key=lambda item: item[1])


def test_cancellation_subnormals_overflow_and_signed_zero():
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    minimum = float.fromhex("0x0.0000000000001p-1022")
    scale = DyadicScores.from_values([maximum, -maximum, minimum, .1, -0.])
    a, b, c, _, zero = scale.integers
    assert a + b == 0 and c == 1 and zero == 0
    assert scale.upper_float(a + b + c) == minimum
    assert scale.upper_float(2 * a) == math.inf
    assert scale.upper_float(2 * b) == -maximum
    # The exact sum is between two adjacent binary64 values. Round upward.
    bound = scale.upper_float(a + c)
    assert bound == math.inf


@pytest.mark.parametrize("value", [True, math.inf, -math.inf, math.nan])
def test_invalid_score_cannot_be_encoded(value):
    with pytest.raises(ValueError):
        DyadicScores.from_values([value])


def test_empty_space_is_exact_zero():
    scale = DyadicScores.from_values([])
    assert scale.integers == () and scale.denominator == 1
    assert scale.upper_float(0) == 0.


@pytest.mark.parametrize("integers,denominator", [((1,), 0), ((1,), 3),
    ((1,), True), ((True,), 1), ((1.5,), 2), ([1], 2)])
def test_direct_construction_cannot_break_the_exact_scale(integers, denominator):
    with pytest.raises(ValueError):
        DyadicScores(integers, denominator)


@pytest.mark.parametrize("value", [True, 1., .5])
def test_bound_requires_scaled_integer(value):
    with pytest.raises(ValueError):
        DyadicScores.from_values([.5]).upper_float(value)
