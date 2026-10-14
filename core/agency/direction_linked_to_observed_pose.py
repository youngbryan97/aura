"""Learn whether a measured angular phase predicts a direction observed during travel."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field

from core.verify import invariant


def _wrap(angle: float) -> float:
    return (angle+math.pi) % (2.*math.pi)-math.pi


@dataclass(frozen=True)
class _Pair:
    phase: float
    direction: float
    uncertainty: float


@dataclass
class DirectionFromPose:
    """Three agreeing, spread observations qualify a relation within one pose basis."""

    context: Hashable | None = None
    at: float | None = None
    samples: deque[_Pair] = field(default_factory=lambda: deque(maxlen=16))
    current: float | None = None

    def observe(self, context: Hashable, at: float, phase: float, uncertainty: float,
                *, direction: float | None, sample_phase: float | None = None,
                sample_uncertainty: float | None = None) -> None:
        try:
            hash(context)
        except TypeError:
            self.current = None
            self.samples.clear()
            return
        if context != self.context:
            self.context, self.at = context, None
            self.samples.clear()
        if (isinstance(at, (int, float)) and not isinstance(at, bool) and math.isfinite(at)
                and self.at is not None and at <= self.at):
            return
        self.current = None
        if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (at, phase, uncertainty)) or not 0. <= uncertainty <= .12
                or direction is not None and (not isinstance(direction, (int, float))
                                             or isinstance(direction, bool) or not math.isfinite(direction))):
            return
        self.at = at
        if direction is not None:
            sampled = phase if sample_phase is None else sample_phase
            error = uncertainty if sample_uncertainty is None else sample_uncertainty
            if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in (sampled, error)) or not 0. <= error <= .12):
                return
            self._keep_sample(_Pair(sampled, direction, error))
        if len(self.samples) < 3:
            return
        phases = [sample.phase for sample in self.samples]
        if max(abs(_wrap(a-b)) for a in phases for b in phases) < math.pi/6.:
            return
        offsets = [_wrap(sample.direction-sample.phase) for sample in self.samples]
        offset = math.atan2(sum(math.sin(v) for v in offsets), sum(math.cos(v) for v in offsets))
        if any(abs(_wrap(value-offset)) > .05+sample.uncertainty
               for sample, value in zip(self.samples, offsets, strict=True)):
            return
        self.current = _wrap(phase+offset)

    def _keep_sample(self, sample: _Pair) -> None:
        """Keep measured phase coverage; repeated nearby poses cannot crowd it out.

        The latest observation always stays, including a contradiction. A
        replacement removes no more than one older sample from the fixed
        budget. The existing agreement test still checks every retained pair.
        """
        values = [*self.samples, sample]
        if len(values) > self.samples.maxlen:
            def coverage_without(index):
                remaining = [p for i, p in enumerate(values) if i != index]
                spread = max(abs(_wrap(a.phase-b.phase)) for a in remaining for b in remaining)
                return spread, values[index].uncertainty, -index
            remove = max(range(len(values)-1), key=coverage_without)
            del values[remove]
        self.samples.clear()
        self.samples.extend(values)


@invariant('agency.a_pose_direction_requires_observed_travel_agreement', scope='agency',
           owner='core/agency/direction_linked_to_observed_pose.py', observational=False)
def _pose_direction_invariant() -> tuple:
    relation, animation = DirectionFromPose(), DirectionFromPose()
    for i, phase in enumerate((0., .4, .8)):
        relation.observe('body', 1.+i, phase, .01, direction=phase+.3)
        animation.observe('body', 1.+i, phase, .01, direction=.3)
    assert animation.current is None
    relation.observe('body', 4., 1.2, .01, direction=None)
    assert math.isclose(relation.current, 1.5)
    relation.observe('new-body', 5., 1.3, .01, direction=None)
    assert relation.current is None
    return ()


@invariant('agency.pose_calibration_keeps_measured_phase_coverage', scope='agency',
           owner='core/agency/direction_linked_to_observed_pose.py', observational=False)
def _pose_coverage_invariant() -> tuple:
    relation = DirectionFromPose()
    for i, phase in enumerate((0., .6, 1., *([.9]*32))):
        relation.observe('body', 1.+i, phase, .01, direction=phase+.3)
    assert len(relation.samples) == 16 and math.isclose(relation.current, 1.2)
    relation.observe('body', 40., .9, .01, direction=2.2)
    assert relation.current is None and relation.samples[-1].direction == 2.2
    return ()
