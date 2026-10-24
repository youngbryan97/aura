"""A large common-input covariance value is not labeled causal evidence."""
from __future__ import annotations

import numpy as np

from core.consciousness.iit_surrogate import RIIU


def test_covariance_diagnostic_does_not_claim_to_have_measured_causal_integration():
    estimator = RIIU(neuron_count=8)
    estimator._partitions = [(list(range(4)), list(range(4, 8)))]
    rng = np.random.default_rng(17)
    driver = rng.normal(size=(128, 1))
    independent = rng.normal(size=(128, 8))
    common_input = driver + rng.normal(scale=0.2, size=(128, 8))
    assert estimator._compute_phi_internal(common_input) > estimator._compute_phi_internal(independent)
    stats = estimator.get_stats()
    assert stats["measurement_kind"] == "observational_covariance_dependence"
    assert stats["causal_integration"] is None
    assert not stats["citable_as_causal_evidence"]
    assert stats["measured_dependency"] is None  # No live sampling loop ran.
