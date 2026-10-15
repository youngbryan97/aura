"""Shared time-step matrices retain the full Gaussian mixture's estimates."""
import math

import numpy as np
import pytest

from core.perception.motion_kept_between_pictures import (
    MotionBetweenPictures,
    MotionEstimate,
    _MotionLaw,
    _prediction_matrices,
)

pytestmark = pytest.mark.unit


class _FullMatrixLaw(_MotionLaw):
    def predicted(self, dt):
        if self.order == 2:
            transition = np.array(((1., dt), (0., 1.)))
            process = np.array(((dt**3/3, dt**2/2), (dt**2/2, dt)))
        else:
            transition = np.array(((1., dt, dt**2/2), (0., 1., dt), (0., 0., 1.)))
            process = np.array(((dt**5/20, dt**4/8, dt**3/6),
                                (dt**4/8, dt**3/3, dt**2/2), (dt**3/6, dt**2/2, dt)))
        return transition @ self.state, transition @ self.covariance @ transition.T + process*self.noise


class _FullMatrixBelief(MotionBetweenPictures):
    def _start(self, owner, basis, at, position, variance, status):
        super()._start(owner, basis, at, position, variance, status)
        self._laws = [_FullMatrixLaw(position, variance, law.order, law.noise) for law in self._laws]

    def forecast(self, at):
        if self.at is None or not 0 <= at-self.at <= self.longest_gap:
            return None
        states, covariances = [], []
        for law in self._laws:
            state, covariance = law.predicted(at-self.at)
            full, uncertainty = np.zeros(6), np.eye(6)*4.
            flat = state.ravel()
            full[:len(flat)] = flat
            uncertainty[:len(flat), :len(flat)] = np.kron(covariance, np.eye(2))
            states.append(full)
            covariances.append(uncertainty)
        mean = sum(w*s for w, s in zip(self._weights, states, strict=True))
        covariance = sum(w*(c+np.outer(s-mean, s-mean))
                         for w, s, c in zip(self._weights, states, covariances, strict=True))
        error = np.sqrt(np.maximum(0., np.diag(covariance)))*self.scale
        values = mean*self.scale
        return MotionEstimate(self.owner, self.basis, float(at), self.at, tuple(values[:2]),
            tuple(values[2:4]), tuple(values[4:6]), float(np.linalg.norm(error[:2])),
            float(np.linalg.norm(error[2:4])), self.observations, self.status, tuple(self._weights))


@pytest.mark.parametrize('scale', [1., 240., 2560.])
def test_current_and_future_estimates_match_the_full_covariance_reference(scale):
    actual, reference = MotionBetweenPictures(scale=scale), _FullMatrixBelief(scale=scale)
    rng, at = np.random.default_rng(313), 1.
    for step in range(80):
        at += float(rng.choice([.05, .1, .167, .268130064]))
        position = tuple(scale*(np.array((.15+.04*at, .3+.03*math.sin(at))) + rng.normal(0., .002, 2)))
        owner, basis = (1, 0.) if step < 50 else (2, at), ('pixels', step//30)
        for belief in (actual, reference):
            belief.observe(owner, basis, at, position, error=.002*scale)
        for ahead in (0., .05, .3, 1.):
            first, second = actual.forecast(at+ahead), reference.forecast(at+ahead)
            if first is None or second is None:
                assert first is second is None
                continue
            assert (first.owner, first.basis, first.observed_at, first.observations, first.status) == (
                second.owner, second.basis, second.observed_at, second.observations, second.status)
            np.testing.assert_array_equal(first.position+first.velocity+first.acceleration+first.weights,
                                          second.position+second.velocity+second.acceleration+second.weights)
            assert first.position_error == second.position_error
            assert first.velocity_error == second.velocity_error


def test_shared_matrices_are_bounded_and_cannot_be_written_by_a_prediction():
    _prediction_matrices.cache_clear()
    try:
        transition, process = _prediction_matrices(3, .03, .1)
        with pytest.raises(ValueError):
            transition[0, 0] = 2.
        with pytest.raises(ValueError):
            process[0, 0] = 2.
        for index in range(300):
            _prediction_matrices(2, .03, .001*(index+1))
        assert _prediction_matrices.cache_info().currsize <= 256
        law = _MotionLaw(np.array((1., 2.)), 1., 3, .03)
        state, covariance = law.predicted(0.)
        state[:] = 99.
        covariance[:] = 99.
        assert law.state[0, 0] == 1. and law.covariance[0, 0] == 1.
    finally:
        _prediction_matrices.cache_clear()


def test_shared_prediction_arithmetic_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.motion_kept_between_pictures import _shared_motion_arithmetic_invariant

    assert _shared_motion_arithmetic_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'shared_motion_arithmetic_keeps_its_owned_state')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
