"""Keep an object's motion and uncertainty between measured pictures.

Position measurements correct competing velocity and acceleration models.
Their disagreement remains part of the uncertainty. Missing pictures permit
a forecast, never a new observation or a control-identity measurement.

This uses Kalman's prediction/update equations (1960), the Joseph covariance
update, and Gaussian model evidence. Coordinates are normalized to the view
so a change of resolution does not change the physical prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from core.verify import invariant

# Smooth and responsive laws compete at each derivative order. These are
# normalized process-noise priors, not measured limits on a world's motion.
_LAWS = tuple((order, noise) for order in (2, 3) for noise in (.001, .03, 1.))


@lru_cache(maxsize=256)
def _prediction_matrices(order: int, noise: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Reuse immutable time-step arithmetic across objects observed in one picture."""
    if order == 2:
        transition = np.array(((1., dt), (0., 1.)))
        process = np.array(((dt ** 3 / 3, dt ** 2 / 2), (dt ** 2 / 2, dt)))
    else:
        transition = np.array(((1., dt, dt ** 2 / 2), (0., 1., dt), (0., 0., 1.)))
        process = np.array(((dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6),
                            (dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2),
                            (dt ** 3 / 6, dt ** 2 / 2, dt)))
    process *= noise
    transition.flags.writeable = process.flags.writeable = False
    return transition, process

def _finite(*values: Any) -> bool:
    return all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class MotionEstimate:
    owner: tuple
    basis: tuple
    at: float
    observed_at: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    position_error: float
    velocity_error: float
    observations: int
    status: str
    weights: tuple[float, ...]

    @property
    def current(self) -> bool:
        return self.at == self.observed_at and self.observations >= 3 and self.status == 'observed'


@dataclass(frozen=True)
class MeasuredMotionBox:
    """The visible box belonging to one measured state, before later projections."""

    state: MotionEstimate
    box: tuple[float, float, float, float]


@dataclass(init=False, eq=False)
class _MotionLaw:
    """One small Gaussian state; all measurements name position only."""

    order: int
    noise: float
    state: np.ndarray
    covariance: np.ndarray

    def __init__(self, position: np.ndarray, variance: float, order: int, noise: float) -> None:
        self.order, self.noise = order, noise
        self.state = np.zeros((order, 2))
        self.state[0] = position
        # Axes have equal measurement noise and independent innovations.
        # One derivative covariance serves both, avoiding two matrix solves.
        self.covariance = np.eye(order) * 4.
        self.covariance[0, 0] = variance

    def predicted(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if dt == 0.:
            return self.state.copy(), self.covariance.copy()
        transition, process = _prediction_matrices(self.order, self.noise, dt)
        return transition @ self.state, transition @ self.covariance @ transition.T + process

    def corrected(self, position: np.ndarray, variance: float, dt: float) -> tuple:
        state, covariance = self.predicted(dt)
        residual = position - state[0]
        innovation = float(covariance[0, 0] + variance)
        if not math.isfinite(innovation) or innovation <= 0:
            return None
        apart = float(residual @ residual) / innovation
        gain = covariance[:, 0] / innovation
        state += gain[:, None] * residual
        remaining = np.eye(len(state))
        remaining[:, 0] -= gain
        covariance = remaining @ covariance @ remaining.T + variance * np.outer(gain, gain)
        return state, (covariance + covariance.T) / 2, apart, -.5 * apart - math.log(innovation)


@dataclass(init=False, eq=False)
class MotionBetweenPictures:
    """A bounded object history whose predictions retain their observation time."""

    scale: float
    longest_gap: float
    owner: tuple | None
    basis: tuple | None
    at: float | None
    observations: int
    status: str
    _laws: list[_MotionLaw]
    _weights: np.ndarray

    def __init__(self, *, scale: float = 1., longest_gap: float = 1.) -> None:
        if not _finite(scale, longest_gap) or scale <= 0 or not 0 < longest_gap <= 5.:
            raise ValueError('motion needs a finite view scale and bounded forecast interval')
        self.scale, self.longest_gap = float(scale), float(longest_gap)
        self.owner: tuple | None = None
        self.basis: tuple | None = None
        self.at: float | None = None
        self.observations = 0
        self.status = 'unobserved'
        self._laws: list[_MotionLaw] = []
        self._weights = np.full(len(_LAWS), 1. / len(_LAWS))

    def _start(self, owner: tuple, basis: tuple, at: float, position: np.ndarray,
               variance: float, status: str) -> None:
        self.owner, self.basis, self.at = owner, basis, float(at)
        self.observations, self.status = 1, status
        self._laws = [_MotionLaw(position, variance, order, noise) for order, noise in _LAWS]
        self._weights = np.full(len(_LAWS), 1. / len(_LAWS))

    def observe(self, owner: tuple, basis: tuple, at: float, position: tuple,
                *, error: float = 1.) -> MotionEstimate | None:
        """Update from current measured positions in one identity and coordinate origin."""
        if (not isinstance(owner, tuple) or not isinstance(basis, tuple)
                or not isinstance(position, (tuple, list, np.ndarray)) or len(position) != 2
                or not _finite(at, *position, error) or at < 0 or error <= 0):
            return None
        if self.at is not None and at <= self.at:
            return None
        measured = np.asarray(position, dtype=float) / self.scale
        variance = (error / self.scale) ** 2
        if (self.at is None or owner != self.owner or basis != self.basis or at - self.at > self.longest_gap):
            self._start(owner, basis, at, measured, variance, 'warming')
            return self.forecast(at)
        dt = float(at - self.at)
        corrected = [law.corrected(measured, variance, dt) for law in self._laws]
        if any(row is None for row in corrected):
            return None
        # 99.9 percent of two-dimensional Gaussian innovations lie inside
        # this contour. A jump outside both laws starts a new motion regime;
        # it cannot silently pull an old velocity toward another object.
        if min(row[2] for row in corrected) > 13.816:
            self._start(owner, basis, at, measured, variance, 'motion changed')
            return self.forecast(at)
        # A law can regain support when motion changes. Mixing by elapsed
        # time avoids tying that chance to the rendering frequency.
        change = -math.expm1(-dt)
        prior = self._weights * (1 - change) + change / len(_LAWS)
        scores = np.log(prior) + np.asarray([row[3] for row in corrected])
        evidence = np.exp(scores - scores.max())
        self._weights = evidence / evidence.sum()
        for law, (state, covariance, _apart, _score) in zip(self._laws, corrected, strict=True):
            law.state, law.covariance = state, covariance
        self.at, self.observations, self.status = float(at), self.observations + 1, 'observed'
        return self.forecast(at)

    def forecast(self, at: float) -> MotionEstimate | None:
        """A forecast is a copy; looking ahead cannot refresh or train the history."""
        if (self.at is None or not _finite(at) or not 0 <= at - self.at <= self.longest_gap):
            return None
        states, diagonals = [], []
        for law in self._laws:
            state, covariance = law.predicted(float(at - self.at))
            full, diagonal = np.zeros(6), np.full(6, 4.)
            flat = state.ravel()
            full[:len(flat)] = flat
            diagonal[:len(flat)] = np.repeat(np.diag(covariance), 2)
            states.append(full)
            diagonals.append(diagonal)
        mean = sum(weight * state for weight, state in zip(self._weights, states, strict=True))
        variance = sum(weight * (diagonal + (state - mean) ** 2)
                       for weight, state, diagonal in zip(self._weights, states, diagonals, strict=True))
        error = np.sqrt(np.maximum(0., variance)) * self.scale
        values = mean * self.scale
        return MotionEstimate(self.owner, self.basis, float(at), self.at, tuple(values[:2]),
            tuple(values[2:4]), tuple(values[4:6]), float(np.linalg.norm(error[:2])),
            float(np.linalg.norm(error[2:4])), self.observations, self.status, tuple(self._weights))


def measured_motion_reference(thing: Any, at: float) -> tuple | None:
    """Name a measured origin independently of an object's current outline."""
    if getattr(thing, 'motion_position', None) is None:
        return ('box',)
    reference = getattr(thing, 'visible_reference', None)
    if (reference is not None and reference.born == thing.born and reference.at == at
            and reference.position == thing.motion_position and _finite(reference.reference_at)
            and reference.born <= reference.reference_at <= at):
        return 'points', reference.reference_at
    return None


def _measured_basis(thing: Any, at: float, shape: tuple) -> tuple | None:
    reference = measured_motion_reference(thing, at)
    return (tuple(shape), *reference) if reference is not None else None


def observe_thing_motion(thing: Any, at: float, shape: tuple) -> None:
    """Keep a belief only from the same current evidence used to learn controls."""
    from core.perception.position_from_the_same_points import measured_position
    from core.perception.translation_from_visible_support import motion_was_measured

    if (not motion_was_measured(thing, at) or len(shape) != 2 or min(shape) <= 0
            or not _finite(thing.born, thing.x, thing.y, thing.w, thing.h)):
        return
    basis = _measured_basis(thing, at, shape)
    if basis is None:
        thing.motion_estimate = None
        return
    belief = getattr(thing, 'motion_belief', None)
    if belief is None or belief.scale != max(shape):
        belief = thing.motion_belief = MotionBetweenPictures(scale=float(max(shape)))
    previous = getattr(thing, 'motion_estimate', None)
    # A changed extent adds uncertainty to a box centre. Owned point paths
    # keep their separate origin and do not borrow a changing box's centre.
    error = 1.
    if thing.motion_position is None and getattr(thing, 'dims', None) and len(thing.dims) > 1:
        error += .5 * math.dist(tuple(thing.dims[-1]), tuple(thing.dims[-2]))
    estimate = belief.observe((thing.number, float(thing.born)), basis, at, measured_position(thing), error=error)
    if estimate is not None:
        thing.motion_estimate = estimate
        thing.measured_motion_box = MeasuredMotionBox(estimate, (thing.x, thing.y, thing.w, thing.h))
    elif previous is not None and previous.observed_at != at:
        thing.motion_estimate = None


def current_motion_estimate(thing: Any, at: float) -> MotionEstimate | None:
    """A forecast cannot substitute for a current owned observation."""
    from core.perception.translation_from_visible_support import motion_was_measured

    estimate = getattr(thing, 'motion_estimate', None)
    if (isinstance(estimate, MotionEstimate) and estimate.current and estimate.observed_at == at
            and estimate.owner == (getattr(thing, 'number', None), getattr(thing, 'born', None))
            and estimate.basis == _measured_basis(thing, at, estimate.basis[0])
            and motion_was_measured(thing, at)):
        return estimate
    return None


def measured_velocity(thing: Any, at: float) -> tuple[float, float, float | None]:
    """One current state policy for input identification and physical controllers."""
    estimate = current_motion_estimate(thing, at)
    if estimate is not None and estimate.velocity_error <= max(2., .5 * math.hypot(*estimate.velocity)):
        return *estimate.velocity, estimate.velocity_error
    return float(getattr(thing, 'vx', 0.)), float(getattr(thing, 'vy', 0.)), None


def predicted_box_from_motion(thing: Any, at: float) -> tuple | None:
    """A measured trajectory can propose correspondence, never prove it.

    Use the last measured box even if a missing picture projected the public
    box elsewhere. Reject a forecast whose uncertainty exceeds the body size.
    Existing appearance, support and shape checks must still admit the match.
    """
    belief = getattr(thing, 'motion_belief', None)
    anchor = getattr(thing, 'measured_motion_box', None)
    if belief is None or anchor is None:
        return None
    estimate = anchor.state
    if (not estimate.current or estimate.owner != (thing.number, thing.born)
            or estimate.owner != belief.owner or estimate.basis != belief.basis
            or estimate.observed_at != belief.at):
        return None
    from core.perception.translation_from_visible_support import motion_was_measured

    # A current measured reference change invalidates the old history. A
    # missing or unmeasured observation may retain only its bounded proposal.
    if (motion_was_measured(thing, thing.seen)
            and estimate.basis != _measured_basis(thing, thing.seen, estimate.basis[0])):
        return None
    if math.hypot(*estimate.velocity) < 1. and math.hypot(*estimate.acceleration) < 1.:
        return None
    key = at, anchor
    cached = getattr(thing, 'motion_proposal', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    future = belief.forecast(at)
    if future is None or future.position_error > max(anchor.box[2:]):
        thing.motion_proposal = key, None
        return None
    x, y = anchor.box[:2]
    dx, dy = np.subtract(future.position, estimate.position)
    proposal = float(x + dx), float(y + dy), 3. * future.position_error
    thing.motion_proposal = key, proposal
    return proposal


@invariant('perception.motion_forecasts_do_not_renew_observations', scope='perception',
           owner='core/perception/motion_kept_between_pictures.py', observational=False)
def _motion_belief_invariant() -> tuple:
    belief = MotionBetweenPictures(scale=100.)
    for at in (1., 1.1, 1.2, 1.3):
        belief.observe((1, 0.), ('one view',), at, (20 * at, 10.), error=.5)
    current = belief.forecast(1.3)
    predicted = belief.forecast(1.6)
    assert current.current and not predicted.current
    assert predicted.observed_at == current.observed_at == belief.at == 1.3
    assert predicted.position_error > current.position_error
    assert belief.observe((1, 0.), ('one view',), 1.3, (200., 10.)) is None
    assert belief.observe((1, 1.4), ('one view',), 1.4, (30., 10.)).observations == 1
    return ()


@invariant('perception.shared_motion_arithmetic_keeps_its_owned_state', scope='perception',
           owner='core/perception/motion_kept_between_pictures.py', observational=False)
def _shared_motion_arithmetic_invariant() -> tuple:
    transition, process = _prediction_matrices(3, .03, .1)
    assert not transition.flags.writeable and not process.flags.writeable
    assert _prediction_matrices(3, .03, .1)[0] is transition
    assert _prediction_matrices.cache_info().currsize <= 256
    law = _MotionLaw(np.array((1., 2.)), 1., 3, .03)
    state, covariance = law.predicted(0.)
    state[:] = covariance[:] = 99.
    assert law.state[0, 0] == 1. and law.covariance[0, 0] == 1.
    return ()


@invariant('perception.motion_proposals_keep_the_last_measured_box', scope='perception',
           owner='core/perception/motion_kept_between_pictures.py', observational=False)
def _motion_proposal_invariant() -> tuple:
    from types import SimpleNamespace

    belief = MotionBetweenPictures(scale=100.)
    for at in np.arange(1., 2., .1):
        state = belief.observe((1, 0.), ((100, 100), 'box'), float(at), (40. * at, 10.))
    thing = SimpleNamespace(number=1, born=0., seen=belief.at, motion_measured_at=belief.at,
                            motion_evidence='measured', motion_position=None, motion_belief=belief,
                            motion_estimate=state, path=[(belief.at, *state.position)], w=10., h=12.)
    thing.measured_motion_box = MeasuredMotionBox(state, (*state.position, 10., 12.))
    before = predicted_box_from_motion(thing, belief.at + .2)
    thing.x, thing.y = -900., 900.
    assert predicted_box_from_motion(thing, belief.at + .2) == before
    assert before[0] > state.position[0] + 7.
    assert thing.seen == thing.motion_measured_at == belief.at
    assert predicted_box_from_motion(thing, belief.at + 1.1) is None
    return ()
