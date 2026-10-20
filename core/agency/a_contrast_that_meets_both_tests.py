"""Choose a contrast for the same two admission gates its consumer applies.

A larger statistic with a negligible effect cannot hide a pair passing
both gates. Ranking changes neither measurements nor admission thresholds.
The normalized minimum is a ranking margin, not a probability.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

from core.verify import invariant


def contrast_rank(pair: tuple, *, statistic_over: float, effect_over: float, strict: bool = False) -> tuple:
    statistic, effect = pair
    qualified = (statistic > statistic_over and effect > effect_over if strict
                 else statistic >= statistic_over and effect >= effect_over)
    return qualified, min(statistic/statistic_over, effect/effect_over), statistic, effect


def strongest_eligible_contrast(pairs: Iterable, *, statistic_over: float, effect_over: float,
                                strict: bool = False) -> tuple[float, float]:
    """Prefer a jointly qualified pair; unavailable measurements never qualify."""
    if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
            for v in (statistic_over, effect_over))):
        raise ValueError('contrast thresholds must be finite and positive')
    valid = []
    for pair in pairs:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in pair)):
            continue
        try:
            values = tuple(float(v) for v in pair)
        except (ValueError, OverflowError):
            continue
        if all(math.isfinite(v) and v >= 0 for v in values):
            valid.append(values)
    return max(valid, key=lambda pair: contrast_rank(pair, statistic_over=statistic_over,
               effect_over=effect_over, strict=strict), default=(0., 0.))


@invariant('agency.contrast_ranking_preserves_joint_admission', scope='agency',
           owner='core/agency/a_contrast_that_meets_both_tests.py', observational=False)
def _joint_contrast_invariant() -> tuple:
    import itertools

    failed, qualified = (21.8003, 13.8009), (20.4512, 72.3741)
    for pairs in itertools.permutations((failed, qualified, (1000., 1.))):
        assert strongest_eligible_contrast(pairs, statistic_over=20., effect_over=20.) == qualified
    assert strongest_eligible_contrast([(20., 20.)], statistic_over=20., effect_over=20.) == (20., 20.)
    assert not contrast_rank((20., 20.), statistic_over=20., effect_over=20., strict=True)[0]
    return ()
