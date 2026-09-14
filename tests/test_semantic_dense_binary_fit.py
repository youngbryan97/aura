"""Dense fitting preserves the weighted, regularized binary objective."""

import numpy as np
import pytest

from core.learning.semantic_program_transducer import _fit_binary_head


@pytest.mark.parametrize("weighted", [False, True])
def test_dense_solver_matches_liblinear_including_regularized_bias(weighted):
    rng = np.random.default_rng(271828)
    x = rng.normal(size=(160, 12)).astype(np.float32)
    y = (x[:, 0] + 0.2 * x[:, 1] + rng.normal(size=160) > 0.8).astype(np.int8)
    sample_weight = rng.uniform(0.2, 2.0, len(y)) if weighted else None
    kwargs = {"sample_weight": sample_weight, "tolerance": 1e-9}
    legacy, old_bias = _fit_binary_head(x, y, **kwargs)
    dense, bias = _fit_binary_head(x, y, solver="lbfgs", **kwargs)
    np.testing.assert_allclose(dense, legacy, atol=2e-5, rtol=2e-5)
    assert bias == pytest.approx(old_bias, abs=2e-5)


def test_dense_solver_refuses_unfinished_fit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 12)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int8)
    with pytest.warns(Warning), pytest.raises(ValueError, match="did not converge"):
        _fit_binary_head(x, y, solver="lbfgs", max_iter=1)


def test_unknown_solver_does_not_silently_choose_a_different_fit():
    with pytest.raises(ValueError, match="unsupported semantic binary solver"):
        _fit_binary_head(np.ones((2, 1)), np.asarray([0, 1]), solver="other")
