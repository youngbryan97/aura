"""Compare frozen motion laws against the same later measured intervals.

One completed physical hold supplies one comparison. Fits use earlier
holds only; a hold cannot train the model predicting itself. Scores are
normalized residuals, not calibrated probabilities. Missing observations
do not vote, and large unexplained residuals remain in both model scores.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class ForwardFamilyReading:
    selected: bool | None
    reason: str
    owner: tuple | None
    at: float | None
    holds: int
    pairs: int
    acceleration_loss: float | None
    direct_loss: float | None
    acceleration_rms: float | None
    direct_rms: float | None
    contrary_holds: int


class MotionModelsOnTheSameIntervals:
    """Retain one current material basis, bounded fits and independent votes."""

    def __init__(self) -> None:
        self.owner: tuple | None = None
        self.samples: dict[tuple, list] = {}
        self.models: dict[tuple, tuple | None] = {}
        self.losses: dict[tuple, list] = {}
        self.closed: set[tuple] = set()
        self.selected: bool | None = None
        self.challenger: bool | None = None
        self.contrary = 0
        self.votes = 0
        self.last: tuple | None = None
        self.reason = 'needs_forward_observations'
        self.through = -math.inf
        self._push_admitted = False

    def add(self, sample: Any) -> None:
        from core.agency.responses_from_measured_intervals import VelocityInterval

        if not isinstance(sample, VelocityInterval) or sample.at <= self.through:
            return
        if self.owner != sample.owner:
            self.__init__()
            self.owner = sample.owner
        self.through = sample.at
        press = sample.key, sample.began
        if press in self.closed:
            return
        if press not in self.samples:
            self.models[press] = self._fit(sample.key, sample.began)
        previous = self.samples.setdefault(press, [])
        if previous and previous[-1].at == sample.before and self.models[press] is not None:
            old = previous[-1]
            dt = sample.middle-old.middle
            acceleration, pace, fit_error = self.models[press]
            if dt > old.time_error+sample.time_error:
                predicted = tuple(old.velocity[a]+acceleration[a]*dt for a in (0, 1))
                residuals = math.dist(predicted, sample.velocity), math.dist(pace, sample.velocity)
                noise = max(1., old.error+sample.error+fit_error*dt
                            + math.hypot(*acceleration)*(old.time_error+sample.time_error))
                self.losses.setdefault(press, []).append(
                    (sample.at, dt, *(r*r/noise**2 for r in residuals), *(r*r for r in residuals)))
                del self.losses[press][:-128]
        previous.append(sample)
        del previous[:-128]
        while len(self.samples) > 80:
            oldest = min(self.samples, key=lambda value: value[1])
            for store in (self.samples, self.models, self.losses):
                store.pop(oldest, None)
            self.closed.discard(oldest)

    def _fit(self, key: str, before: float) -> tuple | None:
        from core.agency.responses_from_measured_intervals import acceleration_from_intervals

        trained = []
        for press, samples in self.samples.items():
            if press[0] != key or press not in self.closed or not samples or samples[-1].at >= before:
                continue
            fit = acceleration_from_intervals(samples)
            if fit is not None:
                trained.append((fit, tuple(statistics.fmean(s.velocity[a] for s in samples) for a in (0, 1))))
        if len(trained) < 2:
            return None
        return (tuple(statistics.median(f.acceleration[a] for f, _ in trained) for a in (0, 1)),
                tuple(statistics.median(p[a] for _, p in trained) for a in (0, 1)),
                statistics.median(f.error for f, _ in trained))

    def end(self, press: tuple) -> None:
        if press in self.closed or press not in self.samples:
            return
        self.closed.add(press)
        pairs = self.losses.get(press, ())
        if len(pairs) < 2 or sum(p[1] for p in pairs) < .15:
            return
        losses = tuple(statistics.fmean(p[a] for p in pairs) for a in range(2, 6))
        self.votes += 1
        self.last = pairs[-1][0], len(pairs), *losses
        a, d, _raw_a, _raw_d = losses
        winner = True if a+1. < d and a*1.5 < d else False if d+1. < a and d*1.5 < a else None
        if winner is None:
            self.contrary, self.challenger = 0, None
            self.reason = 'forward_predictions_ambiguous'
            return
        if winner == self.selected:
            self.contrary, self.challenger = 0, None
            self.reason = 'forward_predictions_agree'
            return
        self.contrary = self.contrary+1 if self.challenger == winner else 1
        self.challenger = winner
        self.reason = 'needs_independent_confirmation'
        if self.contrary >= 2:
            self.selected = winner
            self.contrary, self.challenger = 0, None
            self.reason = 'forward_predictions_select_acceleration' if winner else 'forward_predictions_select_direct'

    def unseen(self, press: tuple, *, contradicted: bool = False) -> None:
        # Loss of material correspondence cannot train or score a physical
        # law. Already scored, completed holds keep their historical receipt.
        if press not in self.closed:
            self.samples.pop(press, None)
            self.models.pop(press, None)
            self.losses.pop(press, None)
        if contradicted:
            self.selected = None
            self.challenger, self.contrary = None, 0
            self.reason = 'material_correspondence_contradicted'

    def read(self, *, admitted_acceleration: bool, qualified_push: bool) -> ForwardFamilyReading:
        # Initial acceleration authority still needs the existing independent
        # physical contrast and effect tests. Later pace F values cannot
        # revoke it; two forward physical comparisons can.
        if self.owner is not None and self.selected is None and admitted_acceleration and qualified_push:
            self.selected, self.reason = True, 'retained_admitted_acceleration'
        selected = self.selected
        if selected is True and not self._push_admitted:
            self._push_admitted = qualified_push
            if not self._push_admitted:
                selected = None
        if qualified_push:
            self._push_admitted = True
        at, pairs, a, d, ra, rd = self.last or (None, 0, None, None, None, None)
        return ForwardFamilyReading(selected, self.reason, self.owner, at, self.votes, pairs, a, d,
                                    math.sqrt(ra) if ra is not None else None,
                                    math.sqrt(rd) if rd is not None else None, self.contrary)


@invariant('agency.motion_models_share_forward_heldout_observations', scope='agency',
           owner='core/agency/motion_models_on_the_same_intervals.py', observational=False)
def _forward_motion_models_invariant() -> tuple:
    from core.agency.responses_from_measured_intervals import VelocityInterval

    models = MotionModelsOnTheSameIntervals()
    for began in (1., 3., 5., 7.):
        dates = [round(began+.1+j*.1, 6) for j in range(7)]
        for before, at in zip(dates, dates[1:]):
            models.add(VelocityInterval('a', began, (1, 0., 0.), before, at,
                                       (40.*(before+at)/2, 0.), .1))
        models.end(('a', began))
    reading = models.read(admitted_acceleration=False, qualified_push=True)
    assert reading.selected is True and reading.holds == 2 and reading.pairs == 5
    assert reading.acceleration_rms < reading.direct_rms
    return ()
