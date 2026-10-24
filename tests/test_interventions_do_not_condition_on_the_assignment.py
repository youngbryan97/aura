"""A common cause is not a response to the intervention it correlates with."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.consciousness.causal_transition_response import CausalTransitionKernel, weakest_response
from core.consciousness.phi_core import PhiCore


def _kernel(next_state, background):
    transitions = np.zeros((4, 4))
    for state in range(4):
        transitions[state, next_state(state)] = 1
    return CausalTransitionKernel(transitions, background, "structural_equations", "known-two-bit-system")


def test_shared_cause_does_not_become_a_causal_response():
    # B persists and drives A. At equilibrium A=B; forcing A leaves B alone.
    kernel = _kernel(lambda s: 3 if s & 2 else 0, [0.5, 0.0, 0.0, 0.5])
    value, _, _, _ = weakest_response(kernel)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_bidirectional_coupling_changes_the_other_side_of_the_cut():
    kernel = _kernel(lambda s: ((s & 1) << 1) | ((s >> 1) & 1), [0.25] * 4)
    value, left, right, cuts = weakest_response(kernel)
    assert value == pytest.approx(math.log(2))
    assert left == (0,) and right == (1,) and len(cuts) == 1


def test_observational_history_is_never_substituted_for_a_causal_kernel(monkeypatch):
    core = PhiCore.__new__(PhiCore)

    def forbidden():
        raise AssertionError("an observational TPM is not the intervention kernel")

    monkeypatch.setattr(core, "build_tpm", forbidden)
    assert core.compute_interventional_phi() is None


def test_kernel_rejects_large_geometry_before_materializing_it():
    class Huge:
        shape = (65536, 65536)

        def __array__(self, *args, **kwargs):
            raise AssertionError("large matrix was materialized")

    with pytest.raises(ValueError, match="two to eight"):
        CausalTransitionKernel(Huge(), [], "structural_equations", "too-wide")


def test_identification_and_normalization_are_required():
    with pytest.raises(ValueError, match="observations alone"):
        CausalTransitionKernel(np.eye(4), [0.25] * 4, "observational", "correlated-history")
    with pytest.raises(ValueError, match="normalized"):
        CausalTransitionKernel(np.zeros((4, 4)), [0.25] * 4, "structural_equations", "broken")


def test_result_retains_its_bound_and_identification():
    kernel = _kernel(lambda s: ((s & 1) << 1) | ((s >> 1) & 1), [0.25] * 4)
    core = PhiCore.__new__(PhiCore)
    result = core.compute_interventional_phi(causal_kernel=kernel)
    assert result.phi_s > 0
    assert result.node_count == 2
    assert result.sampling == "intervention_response_proxy"
    assert "known-two-bit-system" in result.grounding
    assert not result.citable_as_evidence


@pytest.mark.parametrize("nodes", [3, 4])
def test_ring_coupling_crosses_every_cut_while_independent_bits_do_not(nodes):
    size = 2**nodes
    identity = np.eye(size)
    transition = np.zeros((size, size))
    for state in range(size):
        following = ((state << 1) & (size - 1)) | (state >> (nodes - 1))
        transition[state, following] = 1
    background = [1 / size] * size
    coupled = CausalTransitionKernel(transition, background, "structural_equations", "binary-ring")
    separate = CausalTransitionKernel(identity, background, "structural_equations", "independent-bits")
    assert weakest_response(coupled)[0] == pytest.approx(math.log(2))
    assert weakest_response(separate)[0] == pytest.approx(0.0, abs=1e-12)
