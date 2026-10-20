"""Learn acceleration from dated material displacements and delivered inputs.

Each velocity belongs to the midpoint of its observed interval. Precision
includes the pixel fit and capture-clock widths. Independent physical holds
remain the trials; gaps split fits without creating another trial. The noise
estimates are neither calibrated probabilities nor bounds on tracking bias.
"""
from __future__ import annotations

import copy
import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.material_displacement_from_named_pixels import MaterialDisplacement
from core.verify import invariant
from core.agency.a_contrast_that_meets_both_tests import strongest_eligible_contrast


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CaptureBounds:
    at: float
    lower: float
    upper: float
    source: str
    surface: str
    epoch: int
    sequence: int

    @classmethod
    def from_frame(cls, frame: Any) -> CaptureBounds | None:
        if (not isinstance(frame, MeasuredFrameTime)
                or not all(_finite(v) for v in (float(frame), frame.lower, frame.upper, frame.source_at))
                or not 0 <= frame.lower <= frame <= frame.upper
                or not isinstance(frame.source, str) or not frame.source
                or not isinstance(getattr(frame, 'surface', None), str) or not frame.surface
                or type(frame.epoch) is not int or frame.epoch < 0
                or type(getattr(frame, 'sequence', None)) is not int or frame.sequence < 0):
            return None
        return cls(float(frame), float(frame.lower), float(frame.upper), frame.source,
                   frame.surface, frame.epoch, frame.sequence)


@dataclass(frozen=True)
class VelocityInterval:
    key: str
    began: float
    owner: tuple[int, float, float]
    before: float
    at: float
    velocity: tuple[float, float]
    error: float
    time_error: float = 0.

    def __post_init__(self) -> None:
        if (not isinstance(self.key, str) or not isinstance(self.owner, tuple) or len(self.owner) != 3
                or type(self.owner[0]) is not int or self.owner[0] < 0
                or not isinstance(self.velocity, tuple) or len(self.velocity) != 2
                or not all(_finite(v) for v in (*self.owner[1:], self.began, self.before, self.at,
                                                *self.velocity, self.error, self.time_error))
                or not 0 <= self.owner[1] <= self.owner[2] <= self.before
                or not 0 <= self.began < self.before < self.at or self.at-self.before > .35
                or self.error < 0 or self.time_error < 0):
            raise ValueError('velocity intervals need one owned reference and finite dated precision')

    @property
    def middle(self) -> float:
        return (self.before+self.at)/2


class CapturedIntervals:
    """Keep only bounded capture metadata actually consumed by this observer."""

    def __init__(self) -> None:
        self.frames: dict[float, CaptureBounds] = {}
        self.latest = -math.inf

    def remember(self, frame: Any) -> None:
        bounds = CaptureBounds.from_frame(frame)
        if bounds is None:
            return
        if bounds.at <= self.latest:
            # A replayed or conflicting receipt cannot replace old ownership.
            return
        previous = next(reversed(self.frames.values()), None) if self.frames else None
        if previous and (bounds.source, bounds.surface, bounds.epoch) != (previous.source, previous.surface, previous.epoch):
            self.frames.clear()
        self.latest = bounds.at
        self.frames[bounds.at] = bounds
        while len(self.frames) > 64:
            del self.frames[next(iter(self.frames))]

    def measured(self, thing: Any, frame: Any, held_at: Any, *, response_s: float) -> VelocityInterval | None:
        step = getattr(thing, 'material_displacement', None)
        if (not isinstance(step, MaterialDisplacement) or step.at != frame
                or step.number != thing.number or step.born != thing.born or step.at != thing.seen
                or step.end != getattr(thing, 'motion_position', None)
                or getattr(thing, 'motion_measured_at', None) != step.at
                or not _finite(response_s) or response_s < 0):
            return None
        first, last = self.frames.get(step.before), self.frames.get(step.at)
        if (first is None or last is None or CaptureBounds.from_frame(frame) != last
                or (first.source, first.surface, first.epoch) != (last.source, last.surface, last.epoch)
                or first.sequence >= last.sequence or first.upper >= last.lower):
            return None
        held = held_at(first.lower)
        if (held is None or not isinstance(held, tuple) or len(held) != 3
                or not _finite(held[0]) or not isinstance(held[1], str) or type(held[2]) is not bool
                or held != held_at(last.upper) or first.lower-held[0] < response_s):
            return None
        dt = step.at-step.before
        width = first.upper-first.lower+last.upper-last.lower
        if dt <= width:
            return None
        velocity = tuple((b-a)/dt for a, b in zip(step.start, step.end, strict=True))
        error = step.error/dt + math.hypot(*velocity)*width/(dt-width)
        if not all(_finite(v) for v in (*velocity, error)):
            return None
        return VelocityInterval(held[1], held[0], (step.number, step.born, step.reference_at),
                                step.before, step.at, velocity, error, width/2)


@dataclass(frozen=True)
class AccelerationEstimate:
    acceleration: tuple[float, float]
    error: float
    samples: int


def acceleration_from_intervals(samples: list[VelocityInterval]) -> AccelerationEstimate | None:
    if (len(samples) < 2 or samples[-1].middle-samples[0].middle < .15
            or any((s.key, s.began, s.owner) != (samples[0].key, samples[0].began, samples[0].owner)
                   for s in samples)
            or any(a.at != b.before for a, b in zip(samples, samples[1:], strict=False))):
        return None
    middle = statistics.fmean(s.middle for s in samples)
    spread = sum((s.middle-middle)**2 for s in samples)
    if not math.isfinite(spread) or spread <= 0:
        return None
    average = tuple(statistics.fmean(s.velocity[axis] for s in samples) for axis in (0, 1))
    acceleration = tuple(sum((s.middle-middle)*s.velocity[axis] for s in samples)/spread for axis in (0, 1))
    residual = max(math.dist(s.velocity, tuple(average[a]+acceleration[a]*(s.middle-middle) for a in (0, 1)))
                   for s in samples)
    error = (sum(abs(s.middle-middle)*s.error for s in samples)/spread
             + 2*residual/(samples[-1].middle-samples[0].middle))
    time_leverage = sum(abs(s.middle-middle)*s.time_error for s in samples)/spread
    if time_leverage >= 1.:
        return None
    # Capture widths also perturb the midpoint on the fit's time axis.
    # Solve E <= sensor_error + (|a_measured|+E)*time_leverage.
    error = (error+math.hypot(*acceleration)*time_leverage)/(1.-time_leverage)
    if not all(_finite(v) for v in (*acceleration, error)):
        return None
    return AccelerationEstimate(acceleration, error, len(samples))


class IntervalResponses:
    """One precision-qualified fit per physical hold, with bounded storage."""

    def __init__(self) -> None:
        from core.agency.motion_models_on_the_same_intervals import MotionModelsOnTheSameIntervals

        self.families = MotionModelsOnTheSameIntervals()
        self.samples: dict[tuple[str, float], list[VelocityInterval]] = {}
        self.best: dict[tuple[str, float], AccelerationEstimate] = {}
        self.best_owners: dict[tuple[str, float], tuple] = {}
        self.through: dict[tuple[str, float], float] = {}
        self.ended: set[tuple[str, float]] = set()
        self.invalid: set[tuple[str, float]] = set()
        self.forgot_before = -math.inf

    def add(self, sample: VelocityInterval | None) -> None:
        if (not isinstance(sample, VelocityInterval) or not isinstance(sample.key, str)
                or not all(_finite(v) for v in (sample.began, sample.before, sample.at, sample.error, *sample.velocity))
                or sample.error < 0 or not sample.began < sample.before < sample.at
                or sample.began <= self.forgot_before):
            return
        press = sample.key, sample.began
        if press in self.ended or press in self.invalid or sample.at <= self.through.get(press, -math.inf):
            return
        self.families.add(sample)
        self.through[press] = sample.at
        samples = self.samples.setdefault(press, [])
        if samples and (samples[-1].at != sample.before or samples[-1].owner != sample.owner):
            samples.clear()
        samples.append(sample)
        del samples[:-128]
        estimate = acceleration_from_intervals(samples)
        old = self.best.get(press)
        if estimate is not None and (old is None or self.best_owners.get(press) != sample.owner or estimate.error <= old.error):
            self.best[press] = estimate
            self.best_owners[press] = sample.owner
        self._trim()

    def end(self, press: tuple[str, float]) -> None:
        if self._remember_press(press):
            self.families.end(press)
            self.ended.add(press)
            self.samples.pop(press, None)

    def unseen(self, press: tuple[str, float], *, contradicted: bool = False) -> None:
        self.families.unseen(press, contradicted=contradicted)
        self.samples.pop(press, None)
        if contradicted and press not in self.ended and self._remember_press(press):
            self.invalid.add(press)
            self.best.pop(press, None)
            self.best_owners.pop(press, None)

    def _remember_press(self, press: tuple[str, float]) -> bool:
        if (not isinstance(press, tuple) or len(press) != 2 or not isinstance(press[0], str)
                or not _finite(press[1]) or press[1] < 0 or press[1] <= self.forgot_before):
            return False
        self.through.setdefault(press, -math.inf)
        self._trim()
        return press in self.through

    def _trim(self) -> None:
        while len(self.through) > 80:
            old = min(self.through, key=lambda value: value[1])
            self.forgot_before = max(self.forgot_before, old[1])
            for kept in (self.through, self.samples, self.best, self.best_owners):
                kept.pop(old, None)
            self.ended.discard(old)
            self.invalid.discard(old)

    def merge(self, other: IntervalResponses) -> None:
        if other.families.through > self.families.through:
            self.families = copy.deepcopy(other.families)
        for name in ('samples', 'best', 'best_owners', 'through'):
            getattr(self, name).update(copy.deepcopy(getattr(other, name)))
        self.ended.update(other.ended)
        self.invalid.update(other.invalid)
        self.forgot_before = max(self.forgot_before, other.forgot_before)
        for press in tuple(self.best):
            if press in self.invalid or press[1] <= self.forgot_before:
                self.best.pop(press, None)
                self.best_owners.pop(press, None)
                self.samples.pop(press, None)
        self._trim()

    def groups(self, *, owner: tuple | None = None) -> dict[str, list[AccelerationEstimate]]:
        groups = defaultdict(list)
        for (key, _began), estimate in self.best.items():
            if owner is not None and self.best_owners.get((key, _began)) != owner:
                continue
            groups[key].append(estimate)
        return {key: values for key, values in groups.items() if len(values) >= 2}

    def contrast(self, *, keys: tuple[str, str] | None = None, owner: tuple | None = None,
                 statistic_over: float = 20., effect_over: float = 20., strict: bool = False) -> tuple[float, float]:
        groups = self.groups(owner=owner)
        results = [(0., 0.)]
        for first, second in itertools.combinations(groups, 2):
            if keys is not None and {first, second} != set(keys):
                continue
            a, b = groups[first], groups[second]
            means = [tuple(statistics.fmean(v.acceleration[axis] for v in g) for axis in (0, 1)) for g in (a, b)]
            grand = tuple(statistics.fmean(v.acceleration[axis] for v in a+b) for axis in (0, 1))
            between = sum(len(g)*math.dist(mean, grand)**2 for g, mean in zip((a, b), means, strict=True))
            within = sum(math.dist(v.acceleration, mean)**2 for g, mean in zip((a, b), means, strict=True)
                         for v in g)/(len(a)+len(b)-2)
            noise = statistics.fmean(v.error**2 for v in a+b)
            error = sum(statistics.fmean(v.error for v in g) for g in (a, b))
            effect = max(0., math.dist(*means)-error)*.5
            results.append((between/(within+noise+.25), effect))
        return strongest_eligible_contrast(results, statistic_over=statistic_over, effect_over=effect_over, strict=strict)

    def push(self, key: str, *, statistic_over: float, effect_over: float) -> tuple[float, float] | None:
        groups = self.groups()
        if key not in groups:
            return None
        comparisons = [self.contrast(keys=(key, other), statistic_over=statistic_over, effect_over=effect_over, strict=True) for other in groups
                       if other != key and (not key or not other)]
        statistic, effect = strongest_eligible_contrast(comparisons, statistic_over=statistic_over, effect_over=effect_over, strict=True)
        if statistic <= statistic_over or effect <= effect_over:
            return None
        values = groups[key]
        return tuple(statistics.median(v.acceleration[axis] for v in values) for axis in (0, 1)) if values else None


@invariant('agency.measured_intervals_keep_one_physical_trial', scope='agency',
           owner='core/agency/responses_from_measured_intervals.py', observational=False)
def _measured_interval_trials_invariant() -> tuple:
    responses = IntervalResponses()
    for began in (1., 3.):
        for before, at in ((began+.1, began+.3), (began+.3, began+.5)):
            responses.add(VelocityInterval('a', began, (1, 0., 0.), before, at, (0., 60.*(before+at)/2), .1))
    assert len(responses.best) == 2 and responses.contrast() == (0., 0.)
    responses.unseen(('a', 1.), contradicted=True)
    assert len(responses.best) == 1
    return ()
