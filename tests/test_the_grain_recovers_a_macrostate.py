"""A positive control for the grain: a system whose causal state is coarser than its parts.

Twelve micro variables, and interventional futures that depend on them only
through two weighted sums. The micro description has twelve dimensions and the
causal state has two, and knowing the two makes the micro detail useless for
prediction. A grain estimator that reads the system rather than the matrix it
was handed has to find two by both estimators, span the macrostate far better
than chance, and leave the micro variables nothing to add on held-out futures.

The futures are linear in the macrostate. The grain is a linear estimate of
the predictive state's dimension, and a nonlinear response of a two-dimensional
state has a linear rank above two: bi-cross-validation read ten off a tanh
response, correctly, which made that version a test of the response function
rather than of the grain.
"""

from __future__ import annotations

import numpy as np

#: The v25 runner's own bar for "raw history adds nothing beyond the grain".
SUFFICIENCY_TOLERANCE = 0.05

#: The parallel analysis reads its rank against the 0.99 quantile of a
#: permutation null; the span check uses the same quantile of its own null.
NULL_QUANTILE = 0.99


def _macro_system(*, anchors: int = 80, micro: int = 12, macro: int = 2, columns: int = 48, seed: int = 21):
    rng = np.random.default_rng(seed)
    micro_state = rng.normal(size=(anchors, micro))
    coarse = rng.normal(size=(macro, micro))
    macrostate = micro_state @ coarse.T
    probes = rng.normal(size=(columns, macro))
    signatures = macrostate @ probes.T + 0.05 * rng.normal(size=(anchors, columns))
    heldout_probes = rng.normal(size=(columns, macro))
    heldout = macrostate @ heldout_probes.T + 0.05 * rng.normal(size=(anchors, columns))
    return micro_state, macrostate, signatures, heldout


def _explained(state: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack([state, np.ones(len(state))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float((1.0 - residual.var(axis=0) / target.var(axis=0)).min())


def test_the_grain_has_the_macrostate_dimension() -> None:
    from core.subject.intrinsic_v25 import bicross_validated_rank, fit_predictive_grain

    micro_state, _, signatures, _ = _macro_system()
    assert micro_state.shape[1] == 12
    assert fit_predictive_grain(signatures).rank == 2
    assert bicross_validated_rank(signatures) == 2


def test_the_grain_spans_the_macrostate_far_better_than_chance() -> None:
    from core.subject.intrinsic_v25 import fit_predictive_grain

    _, macrostate, signatures, _ = _macro_system()
    state = fit_predictive_grain(signatures).transform(signatures)
    observed = _explained(state, macrostate)
    rng = np.random.default_rng(5)
    null = [_explained(state[rng.permutation(len(state))], macrostate) for _ in range(200)]
    assert observed > float(np.quantile(null, NULL_QUANTILE)), (observed, float(np.quantile(null, NULL_QUANTILE)))


def test_the_micro_detail_reaches_no_direction_the_grain_does_not() -> None:
    """The micro variables determine the macrostate exactly, so they improve on
    any noisy estimate of it inside the grain's own directions. What they must
    not do is predict the held-out futures along a direction the grain misses."""
    from core.subject.intrinsic_v25 import fit_predictive_grain, heldout_new_direction_gain

    micro_state, _, signatures, heldout = _macro_system()
    grain = fit_predictive_grain(signatures)
    gain = heldout_new_direction_gain(micro_state, grain.transform(signatures), heldout)
    assert gain <= SUFFICIENCY_TOLERANCE, gain
