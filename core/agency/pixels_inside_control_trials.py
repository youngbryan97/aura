"""Bind independent visual trials to measured captures and delivered inputs.

The experiment creates its randomized schedule before seeing any pixels.
One acknowledged press owns one frozen image pair. A responsive region is a
search hint; this module cannot select a body or deliver controls.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.verify import invariant


@dataclass(frozen=True)
class FrozenPixelFrame:
    surface: str
    epoch: int
    sequence: int
    lower: float
    upper: float
    arrived: float
    source: str
    width: int
    height: int
    pixels: bytes
    digest: str

    def valid(self) -> bool:
        return (isinstance(self.surface, str) and 0 < len(self.surface) <= 256
                and type(self.epoch) is int and self.epoch > 0
                and type(self.sequence) is int and self.sequence > 0
                and all(type(v) in (int, float) and math.isfinite(v) for v in (self.lower, self.upper, self.arrived))
                and 0 <= self.lower <= self.upper <= self.arrived
                and isinstance(self.source, str)
                and self.source in {"canvas-bracket", "browser-monotonic", "browser-wall", "screenshot-bracket"}
                and type(self.width) is int and type(self.height) is int and min(self.width, self.height) >= 24
                and self.width <= 320 and self.height <= 240
                and type(self.pixels) is bytes and len(self.pixels) == self.width * self.height * 3
                and self.digest == hashlib.sha256(self.pixels).hexdigest())

    def rgb(self) -> np.ndarray:
        return np.frombuffer(self.pixels, np.uint8).reshape(self.height, self.width, 3)


def freeze_frame(picture: np.ndarray, at: MeasuredFrameTime) -> FrozenPixelFrame | None:
    """Copy actual pixels and their capture contract within a fixed memory bound."""
    if (not isinstance(at, MeasuredFrameTime) or at.lower is None
            or not isinstance(picture, np.ndarray) or picture.dtype != np.uint8
            or picture.ndim != 3 or picture.shape[2] != 3 or min(picture.shape[:2]) < 24
            or picture.size > 32 * 1024 * 1024):
        return None
    surface, sequence = getattr(at, "surface", None), getattr(at, "sequence", None)
    height, width = picture.shape[:2]
    scale = min(1., 320 / width, 240 / height)
    if scale < 1:
        width, height = int(width * scale), int(height * scale)
        picture = zoom(picture, (height / picture.shape[0], width / picture.shape[1], 1.), order=1, prefilter=False)
    pixels = np.ascontiguousarray(picture).tobytes()
    frame = FrozenPixelFrame(surface, at.epoch, sequence, at.lower, at.upper, at.arrived,
                             at.source, width, height, pixels, hashlib.sha256(pixels).hexdigest())
    return frame if frame.valid() else None


@dataclass(frozen=True)
class ScheduledPixelPress:
    ordinal: int
    key: str
    cohort: str


@dataclass(frozen=True)
class PixelPress:
    schedule: ScheduledPixelPress
    down_dispatch: float
    down_ack: float
    up_dispatch: float
    up_ack: float
    neutral_began: float
    before: FrozenPixelFrame
    after: FrozenPixelFrame

    def refusal(self) -> str | None:
        if (not isinstance(self.schedule, ScheduledPixelPress)
                or not isinstance(self.before, FrozenPixelFrame) or not isinstance(self.after, FrozenPixelFrame)
                or not self.before.valid() or not self.after.valid()):
            return "capture_values"
        a, b = self.before, self.after
        if (a.surface, a.epoch, a.source, a.width, a.height) != (b.surface, b.epoch, b.source, b.width, b.height):
            return "changed_capture"
        times = (self.neutral_began, self.down_dispatch, self.down_ack, self.up_dispatch, self.up_ack)
        if not all(type(t) in (int, float) and math.isfinite(t) for t in times):
            return "delivery_values"
        if not 0 <= self.neutral_began <= self.down_dispatch <= self.down_ack < self.up_dispatch <= self.up_ack:
            return "delivery_order"
        if not self.neutral_began <= a.lower <= a.upper <= self.down_dispatch or self.down_dispatch - a.upper > .75:
            return "before_outside_neutral"
        if not self.down_ack + .02 <= b.lower <= b.upper <= self.up_dispatch:
            return "after_outside_hold"
        if not .02 <= self.up_dispatch - self.down_ack <= 1.5 or not .02 <= b.lower - a.lower <= 1.5:
            return "unmatched_or_long_window"
        if a.sequence >= b.sequence:
            return "capture_order"
        return None


class RandomizedPixelExperiment:
    """Own a prespecified discovery cohort and a disjoint confirmation cohort."""

    def __init__(self, keys: tuple[str, str], *, seed: int, per_key: int = 12) -> None:
        if (type(keys) is not tuple or len(keys) != 2 or len(set(keys)) != 2
                or any(not isinstance(k, str) or not k or len(k) > 64 for k in keys)
                or type(seed) is not int or type(per_key) is not int or not 10 <= per_key <= 20 or per_key % 2):
            raise ValueError("a bounded experiment needs two distinct keys and two balanced cohorts")
        rng, schedule = random.Random(seed), []
        for cohort in ("discovery", "confirmation"):
            entries = [key for key in keys for _ in range(per_key // 2)]
            rng.shuffle(entries)
            start = len(schedule)
            schedule.extend(ScheduledPixelPress(start + i, key, cohort) for i, key in enumerate(entries))
        self._schedule = tuple(schedule)
        self._keys = keys
        self._next = 0
        self._frames: set[tuple] = set()
        self._last_release = -math.inf
        self._context: tuple | None = None
        self._trials: list[PixelPress] = []
        self._windows: tuple[float, float] | None = None
        self.refused: str | None = None

    @property
    def schedule(self) -> tuple[ScheduledPixelPress, ...]:
        return self._schedule

    @property
    def keys(self) -> tuple[str, str]:
        return self._keys

    @property
    def trials(self) -> tuple[PixelPress, ...]:
        return tuple(self._trials)

    @property
    def complete(self) -> bool:
        return self.refused is None and self._next == len(self.schedule)

    def retire(self, reason: str) -> None:
        """Preserve the first refusal and the measured prefix."""
        if self.refused is None:
            self.refused = str(reason)[:128] or "unmeasured_trial"

    @property
    def next_press(self) -> ScheduledPixelPress | None:
        return self.schedule[self._next] if self._next < len(self.schedule) and self.refused is None else None

    def record(self, press: PixelPress) -> bool:
        """A failed or substituted trial retires the experiment; it cannot be silently omitted."""
        if self.refused is not None or self.next_press is None:
            return False
        reason = "wrong_schedule" if not isinstance(press, PixelPress) or press.schedule != self.next_press else press.refusal()
        if reason is None:
            a, b = press.before, press.after
            context = (a.surface, a.epoch, a.source, a.width, a.height)
            identities = {(a.surface, a.epoch, a.sequence), (b.surface, b.epoch, b.sequence)}
            if self._context is not None and self._context != context:
                reason = "experiment_changed_capture"
            elif identities & self._frames:
                reason = "reused_capture"
            elif press.neutral_began < self._last_release:
                reason = "overlapping_trials"
            elif self._windows is not None and any(abs(now - first) > .08 for now, first in zip(
                    (b.lower - a.lower, b.lower - press.down_ack), self._windows, strict=True)):
                reason = "unmatched_exposure"
        if reason is not None:
            self.retire(reason)
            return False
        self._frames.update(identities)
        self._context = context
        self._last_release = press.up_ack
        if self._windows is None:
            self._windows = (b.lower - a.lower, b.lower - press.down_ack)
        self._trials.append(press)
        self._next += 1
        return True


@invariant("agency.pixel_trials_keep_prespecified_input_and_capture", scope="agency",
           owner="core/agency/pixels_inside_control_trials.py", observational=False)
def _pixel_trial_custody_invariant() -> tuple:
    experiment = RandomizedPixelExperiment(("a", "b"), seed=41)
    schedule = experiment.schedule
    assert len(schedule) == 24 and all(sum(p.key == k and p.cohort == c for p in schedule) == 6
                                      for k in experiment.keys for c in ("discovery", "confirmation"))
    assert freeze_frame(np.zeros((40, 60, 3), np.uint8), 1.) is None
    return ()
