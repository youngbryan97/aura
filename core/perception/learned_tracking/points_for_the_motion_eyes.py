"""Publish learned point correspondences through the existing measured motion geometry.

New texture gets a bounded share of query slots. Continuing moving marks
keep their identities; hidden points supply no velocity or control evidence.
The classical path's object support and rigid-motion checks remain in use.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.learned_tracking.contracts import PointSeed, TrackingFrame
from core.perception.picture_arithmetic import shrink
from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction, _new_marks
from core.perception.points_that_keep_their_motion import (
    FollowingPoints,
    SeenPoint,
    _grey,
    _sample,
)
from core.runtime.optional_compute import OptionalComputeBusy
from core.verify import invariant


def _same_capture(result: Any, frame: TrackingFrame) -> bool:
    return (result.epoch == frame.epoch and result.frame_number == frame.number
            and result.at == frame.at and result.shape == frame.shape)


class PointsForMotionEyes(FollowingPoints):
    """One worker-owned causal tracker, expressed in the motion eyes' working pixels."""

    def __init__(self, backend: Any, *, limit: int = 64, pixel_budget: int = 256_000,
                 longest_gap: float = 1., fallback_when_busy: bool = False) -> None:
        super().__init__(limit=limit, longest_gap=longest_gap)
        if not 3 <= self.limit <= 384 or type(pixel_budget) is not int or not 256 <= pixel_budget <= 1_000_000:
            raise ValueError('learned point observation needs bounded point and pixel budgets')
        self._backend = backend
        self.pixel_budget = pixel_budget
        self._epoch, self._sequence = 0, 0
        self._geometry: tuple | None = None
        self._previous_pixels: np.ndarray | None = None
        self._native_visible: dict[int, SeenPoint] = {}
        self._slots: tuple[PointSeed, ...] = ()
        self._births: dict[int, float] = {}
        self._window_positions: dict[int, tuple[float, float]] = {}
        self._moving_window_marks: set[int] = set()
        self.status = 'not started'
        if type(fallback_when_busy) is not bool:
            raise ValueError('a point fallback needs an explicit boolean choice')
        self._fallback = PointsBeforeReduction(limit=limit, pixel_budget=pixel_budget) if fallback_when_busy else None

    def clear(self) -> None:
        super().clear()
        self._epoch += 1
        self._geometry, self._previous_pixels = None, None
        self._native_visible.clear()
        self._slots, self._births = (), {}
        self._window_positions.clear()
        self._moving_window_marks.clear()
        self.status = 'reset'
        if self._fallback is not None:
            self._fallback.clear()

    def close(self) -> None:
        close = getattr(self._backend, 'close', None)
        if callable(close):
            close()
        self.clear()

    def _new_seed(self, x: float, y: float, at: float) -> PointSeed:
        seed = PointSeed(self._number, x, y)
        self._number += 1
        self._births[seed.number] = at
        return seed

    def see_source(self, source: np.ndarray, working: np.ndarray, at: float) -> tuple[SeenPoint, ...]:
        """Only a current visible result can extend a point's observed path."""
        if (source.ndim != 3 or source.shape[2] < 3 or working.ndim != 3 or source.dtype != np.uint8
                or min(source.shape[:2]) < 16 or min(working.shape[:2]) < 16 or not math.isfinite(at)):
            self.clear()
            return ()
        original_shape = source.shape[:2]
        fast = self._fallback.see_source(source, working, at) if self._fallback is not None else ()
        factor = min(1., math.sqrt(self.pixel_budget / (source.shape[0] * source.shape[1])))
        if factor < 1.:
            source = shrink(source, int(source.shape[1] * factor), int(source.shape[0] * factor))
        source = np.ascontiguousarray(source[..., :3], dtype=np.uint8)
        geometry = original_shape, source.shape[:2], working.shape[:2]
        if self._geometry is not None and (geometry != self._geometry or not 0. < at - self._at <= self.longest_gap):
            self.clear()
        self._geometry = geometry
        self._sequence += 1
        frame = TrackingFrame(source, at, self._epoch, self._sequence)
        grey = _grey(source)
        try:
            fresh = self._new_window_queries(source, grey)
            if not self._slots or fresh:
                seeds = fresh or self._corner_queries(grey, [], self.limit)
                if len(seeds) < 3:
                    self.points, self.status = (), 'insufficient visible texture'
                    self._at, self._previous_pixels = at, source.copy()
                    self._remember_points()
                    return ()
                self._slots = tuple(self._new_seed(x, y, at) for x, y in seeds)
                self._native_visible.clear()
                self._window_positions = {seed.number: (seed.x, seed.y) for seed in self._slots}
                self._moving_window_marks.clear()
                result = self._backend.start(frame, self._slots)
            else:
                replacements = self._new_texture_queries(source, grey, at)
                result = (self._backend.track(frame, replacements=replacements) if replacements
                          else self._backend.track(frame))
                if replacements:
                    self._slots = tuple(replacements.get(index, seed) for index, seed in enumerate(self._slots))
            if not _same_capture(result, frame):
                raise ValueError('a learned point result belongs to another capture')
            self._publish(result, grey, working.shape[:2], at)
        except OptionalComputeBusy as why:
            if self._fallback is None:
                self.clear()
                raise
            self._slots = ()
            self._native_visible.clear()
            self._births.clear()
            self._window_positions.clear()
            self._moving_window_marks.clear()
            self.status = f'point model deferred: {why}'
            self.points = self._fallback_marks(fast)
        except BaseException:
            self.clear()
            raise
        self._at, self._previous_pixels = at, source.copy()
        self._births = {seed.number: self._births[seed.number] for seed in self._slots}
        if not self.points and fast:
            self.points = self._fallback_marks(fast)
        self._remember_points()
        return self.points

    @staticmethod
    def _fallback_marks(points: tuple) -> tuple:
        # Distinct namespaces prevent a fallback mark inheriting a model
        # query's identity when the available compute changes.
        return tuple(SeenPoint(p.number + (1 << 52), p.x, p.y, p.old_x, p.old_y, p.born,
                               p.before, p.at, p.error, p.pictures) for p in points)

    def _new_window_queries(self, source: np.ndarray, grey: np.ndarray) -> list[tuple[float, float]]:
        """A finished window can admit new texture when its old marks only describe the background."""
        if (getattr(self._backend, 'query_window_frames', None) is None or self.status != 'observed'
                or self._previous_pixels is None):
            return []
        moving = [p for p in self._native_visible.values() if math.hypot(p.x - p.old_x, p.y - p.old_y) >= .5]
        if len(moving) >= 3 or len(self._moving_window_marks) >= 3:
            return []
        changed = np.max(abs(source.astype(np.int16) - self._previous_pixels.astype(np.int16)), axis=2) > 12
        if not changed.any() or changed.mean() > .6:
            return []
        exposed = ndimage.maximum_filter(changed, 7)
        corners = self._corner_queries(grey, moving,
            min(1536, max(3 * self.limit, source.shape[0] * source.shape[1] // 64)), exposed)
        fresh = [(x, y) for x, y in corners if exposed[int(round(y)), int(round(x))]][:self.limit]
        return fresh if len(fresh) >= 3 else []

    def _new_texture_queries(self, source: np.ndarray, grey: np.ndarray, at: float) -> dict[int, PointSeed]:
        """Replace hidden or resting slots with currently exposed changing corners."""
        if self._previous_pixels is None or not callable(getattr(self._backend, '_replace_queries', None)):
            return {}
        if not self._native_visible and self.status in ('seeded', 'warming'):
            return {}
        changed = np.max(abs(source.astype(np.int16) - self._previous_pixels.astype(np.int16)), axis=2) > 12
        if not changed.any() or changed.mean() > .6:
            return {}
        exposed = ndimage.maximum_filter(changed, 7)
        continuing_motion = [p for p in self._native_visible.values() if math.hypot(p.x - p.old_x, p.y - p.old_y) >= .5]
        candidates = self._corner_queries(grey, continuing_motion,
            min(1536, max(3 * self.limit, source.shape[0] * source.shape[1] // 64)), exposed)
        candidates = [(x, y) for x, y in candidates if exposed[int(round(y)), int(round(x))]]
        if not candidates:
            return {}
        slots = []
        for index, seed in enumerate(self._slots):
            point = self._native_visible.get(seed.number)
            if point is None or (at - self._births[seed.number] >= .5
                                and math.hypot(point.x - point.old_x, point.y - point.old_y) < .5):
                slots.append((point is not None, self._births[seed.number], index))
        slots.sort()
        count = min(len(slots), len(candidates), max(1, len(self._slots) // 8))
        return {slot[2]: self._new_seed(*place, at) for slot, place in zip(slots[:count], candidates[:count], strict=True)}

    def _corner_queries(self, grey: np.ndarray, occupied: list, budget: int,
                        activity: np.ndarray | None = None) -> list[tuple[float, float]]:
        """A query budget is shared in the geometry that measures object support."""
        native, working = self._geometry[1:]
        scale = working[1] / native[1], working[0] / native[0]
        return _new_marks(ndimage.gaussian_filter(grey, .6), occupied, budget, scale, activity)

    def _publish(self, result: Any, grey: np.ndarray, shape: tuple[int, int], at: float) -> None:
        current: dict[int, SeenPoint] = {}
        previous_grey = _grey(self._previous_pixels) if self._previous_pixels is not None else None
        for track in result.tracks if result.status == 'observed' else ():
            if not track.visible:
                continue
            before = self._native_visible.get(track.number)
            continued = before is not None and before.at == self._at and track.number in self._births
            error = 0.
            if continued and previous_grey is not None:
                earlier = _sample(previous_grey, np.asarray([[before.x, before.y]]))
                now = _sample(grey, np.asarray([[track.x, track.y]]))
                error = float(np.mean(abs(earlier - now)))
            current[track.number] = SeenPoint(track.number, track.x, track.y,
                before.x if continued else track.x, before.y if continued else track.y,
                self._births[track.number], self._at if continued else at, at, error,
                before.pictures + 1 if continued else 1)
        scale = shape[1] / grey.shape[1], shape[0] / grey.shape[0]
        if result.status == 'observed' and getattr(self._backend, 'query_window_frames', None) is not None:
            # This protects useful query slots. Window displacement never
            # substitutes for consecutive visible motion in the public path.
            self._moving_window_marks = {number for number, p in current.items()
                if number in self._window_positions and math.dist((p.x, p.y), self._window_positions[number]) >= .5}
            self._window_positions = {number: (p.x, p.y) for number, p in current.items()}
        self.points = tuple(SeenPoint(p.number, p.x * scale[0], p.y * scale[1], p.old_x * scale[0], p.old_y * scale[1],
                           p.born, p.before, p.at, p.error, p.pictures) for p in current.values())
        self._native_visible, self.status = current, result.status


@invariant('perception.learned_points_keep_their_capture', scope='perception',
           owner='core/perception/learned_tracking/points_for_the_motion_eyes.py', observational=False)
def _learned_point_capture_invariant() -> tuple:
    from dataclasses import replace

    from core.perception.learned_tracking.contracts import PointResult

    frame = TrackingFrame(np.zeros((20, 30, 3), np.uint8), 1., 2, 3)
    result = PointResult('invented', 2, 3, 1., frame.shape, (), .01)
    assert _same_capture(result, frame)
    for changed in (replace(result, epoch=3), replace(result, frame_number=4),
                    replace(result, at=1.1), replace(result, shape=(21, 30))):
        assert not _same_capture(changed, frame)
    return ()


@invariant('perception.queries_keep_their_first_observation_and_live_budget', scope='perception',
           owner='core/perception/learned_tracking/points_for_the_motion_eyes.py', observational=False)
def _query_lifetime_invariant() -> tuple:
    from core.perception.learned_tracking.continuity_backend import ContinuityBackend

    pixels = np.full((64, 80, 3), 25, np.uint8)
    texture = np.random.default_rng(942).integers(30, 230, (24, 24, 3), dtype=np.uint8)
    pixels[20:44, 20:44] = texture
    points = PointsForMotionEyes(ContinuityBackend(max_points=192), limit=192)
    assert not points.see_source(pixels, pixels, 1.)
    names = {s.number for s in points._slots}
    moved = np.full_like(pixels, 25)
    moved[20:44, 23:47] = texture
    assert points.see_source(moved, moved, 1.1)
    assert {s.number for s in points._slots} == names
    replacements = points._new_texture_queries(pixels, _grey(pixels), 1.8)
    assert len(replacements) <= max(1, len(points._slots)//8)
    points.close()
    return ()
