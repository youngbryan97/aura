"""Measure travel direction from an owned pose window without borrowing filtered velocity."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field


def _wrap(value: float) -> float:
    return (value+math.pi) % (2.*math.pi)-math.pi


@dataclass(frozen=True)
class _Seen:
    at: float
    phase: float
    uncertainty: float
    position: tuple[float, float]
    position_error: float


@dataclass(frozen=True)
class DirectionPair:
    phase: float
    direction: float
    uncertainty: float


@dataclass
class DirectionFromPosePath:
    context: Hashable | None = None
    seen: deque[_Seen] = field(default_factory=lambda: deque(maxlen=32))

    def observe(self, context: Hashable, at: float, phase: float, uncertainty: float,
                position: tuple[float, float], position_error: float, *, ready_after: float) -> DirectionPair | None:
        try:
            hash(context)
        # not a failure: a context that cannot be compared starts afresh
        except TypeError:
            self.seen.clear()
            self.context = None
            return None
        if context != self.context:
            self.context = context
            self.seen.clear()
        if (not isinstance(position, tuple) or len(position) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                           for v in (at, phase, uncertainty, position_error, ready_after, *position))
                or min(uncertainty, position_error) < 0. or at < ready_after
                or self.seen and at <= self.seen[-1].at):
            return None
        if self.seen and at-self.seen[-1].at > .35:
            self.seen.clear()
        current = _Seen(at, phase, uncertainty, position, position_error)
        self.seen.append(current)
        candidates = [s for s in self.seen if s.at >= ready_after and .15 <= at-s.at <= 1.2]
        pairs = []
        for before in candidates:
            distance = math.dist(before.position, position)
            if distance < 8.:
                continue
            window = [s for s in self.seen if s.at >= before.at]
            phases = [before.phase]
            for a, b in zip(window, window[1:], strict=False):
                step = _wrap(b.phase-a.phase)
                if abs(step) > math.pi/3.:
                    break
                phases.append(phases[-1]+step)
            if len(phases) != len(window):
                continue
            if max(phases)-min(phases) > math.pi/2.:
                continue
            middle = (before.at+at)/2.
            for i, (a, b) in enumerate(zip(window, window[1:], strict=False)):
                if a.at <= middle <= b.at:
                    phase_mid = phases[i]+(phases[i+1]-phases[i])*(middle-a.at)/(b.at-a.at)
                    error = math.atan2(before.position_error+position_error, distance)
                    pairs.append(DirectionPair(_wrap(phase_mid), math.atan2(position[1]-before.position[1],
                                                                           position[0]-before.position[0]),
                                               error+max(a.uncertainty, b.uncertainty)))
                    break
        return min(pairs, key=lambda pair: pair.uncertainty, default=None)
