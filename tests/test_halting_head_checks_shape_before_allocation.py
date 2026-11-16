"""A corrupt declared width cannot allocate a head before its arrays are checked."""
from __future__ import annotations

import numpy as np
import pytest

from core.learning.adaptive_halting import HaltingHead


@pytest.mark.parametrize("declared_width", [0, -1, 2**40])
def test_a_mismatched_width_is_refused_before_constructing_a_head(tmp_path, declared_width):
    path = tmp_path / "bad-width.npz"
    np.savez(
        path, weight=np.zeros((4, 1), dtype=np.float32), bias=np.zeros(1, dtype=np.float32),
        threshold=np.array([0.7], dtype=np.float32), hidden_size=np.array([declared_width]),
    )
    constructed = []

    class WatchedHead(HaltingHead):
        def __init__(self, hidden_size, **kwargs):
            constructed.append(hidden_size)
            raise AssertionError("an unchecked artifact reached the allocation boundary")

    with pytest.raises(ValueError, match="wrong shapes"):
        WatchedHead.load(path)
    assert not constructed


def test_a_valid_head_is_constructed_only_after_its_arrays_match(tmp_path):
    path = tmp_path / "head.npz"
    weight = np.arange(4, dtype=np.float32).reshape(4, 1)
    bias = np.array([0.25], dtype=np.float32)
    np.savez(path, weight=weight, bias=bias, threshold=np.array([0.7]), hidden_size=np.array([4]))
    constructed = []

    class WatchedHead(HaltingHead):
        def __init__(self, hidden_size, **kwargs):
            constructed.append(hidden_size)
            super().__init__(hidden_size, **kwargs)

    loaded = WatchedHead.load(path)
    assert constructed == [4]
    assert loaded.threshold == pytest.approx(0.7)
    np.testing.assert_array_equal(np.asarray(loaded.weight), weight)
    np.testing.assert_array_equal(np.asarray(loaded.bias), bias)


def test_the_shape_boundary_has_a_registered_invariant_and_validation():
    from core.learning.adaptive_halting import _halting_head_shape_invariant
    from core.organism.model_validation import ValidationSuite, _install_repair_executor_claims

    assert _halting_head_shape_invariant() == ()
    suite = ValidationSuite()
    _install_repair_executor_claims(suite)
    check = next(test for test in suite.tests() if test.name == "halting_artifact_shape_guard")
    assert check.predict(None) is True
    assert any(claim.test == check.name for claim in suite.claims())
