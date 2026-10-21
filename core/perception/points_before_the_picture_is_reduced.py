"""Keep visible marks before a reduced picture loses their detail.

An outline can be read small while its points need the pixels it was drawn
with. Tracking uses a bounded source picture; all exposed positions return
to the same working coordinates. Local rotation and colour support are
measured in both directions before a point keeps its name.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.points_that_keep_their_motion import (
    FollowingPoints,
    SeenPoint,
    _follow,
    _pyramid,
    _sample,
)
from core.verify import invariant

_OFFSETS = np.mgrid[-3:4, -3:4].reshape(2, -1).T[:, ::-1].astype(float)


def _samples(picture: np.ndarray, places: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(picture, [places[..., 1], places[..., 0]], order=1, mode='constant', cval=0.)


def _colours(picture: np.ndarray) -> np.ndarray:
    return ndimage.gaussian_filter(picture[..., :3].astype(np.float32) / 255., (.6, .6, 0.))


def _refine(before: np.ndarray, after: np.ndarray, points: np.ndarray,
            initial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit each rigid patch to its own convergence; measure every final colour residual."""
    if not len(points):
        return initial.copy(), np.empty(0), np.empty(0), np.empty(0, dtype=bool)
    gradients = [np.gradient(after[..., c]) for c in range(3)]
    templates = np.stack([_samples(before[..., c], points[:, None] + _OFFSETS) for c in range(3)], axis=-1)
    shift, angles, useful = initial - points, np.zeros(len(points)), np.ones(len(points), dtype=bool)
    active = np.arange(len(points))
    for _ in range(12):
        if not len(active):
            break
        c, s = np.cos(angles[active]), np.sin(angles[active])
        rotated = np.stack((c[:, None] * _OFFSETS[:, 0] - s[:, None] * _OFFSETS[:, 1],
                            s[:, None] * _OFFSETS[:, 0] + c[:, None] * _OFFSETS[:, 1]), axis=-1)
        places = points[active, None] + shift[active, None] + rotated
        current = np.stack([_samples(after[..., k], places) for k in range(3)], axis=-1)
        dx = np.stack([_samples(gx, places) for _gy, gx in gradients], axis=-1)
        dy = np.stack([_samples(gy, places) for gy, _gx in gradients], axis=-1)
        rotation = -rotated[..., 1, None] * dx + rotated[..., 0, None] * dy
        jacobian = np.stack((dx, dy, rotation), axis=-1).reshape(len(active), -1, 3)
        residual = (templates[active] - current).reshape(len(active), -1)
        normal = np.einsum('nki,nkj->nij', jacobian, jacobian)
        valid = np.linalg.eigvalsh(normal)[:, 0] > 1e-6
        useful[active] &= valid
        delta = np.zeros((len(active), 3))
        rhs = np.einsum('nki,nk->ni', jacobian, residual)
        delta[valid] = np.linalg.solve(normal[valid], rhs[valid, :, None])[..., 0]
        delta[:, :2] = np.clip(delta[:, :2], -2., 2.)
        delta[:, 2] = np.clip(delta[:, 2], -.12, .12)
        shift[active] += delta[:, :2]
        angles[active] += delta[:, 2]
        converged = (np.linalg.norm(delta[:, :2], axis=1) < .015) & (abs(delta[:, 2]) < .002)
        active = active[valid & ~converged]
    c, s = np.cos(angles), np.sin(angles)
    rotated = np.stack((c[:, None] * _OFFSETS[:, 0] - s[:, None] * _OFFSETS[:, 1],
                        s[:, None] * _OFFSETS[:, 0] + c[:, None] * _OFFSETS[:, 1]), axis=-1)
    places = points[:, None] + shift[:, None] + rotated
    current = np.stack([_samples(after[..., k], places) for k in range(3)], axis=-1)
    error = np.mean(abs(templates - current), axis=(1, 2))
    return points + shift, error, angles, useful


@invariant('perception.local_patch_fits_finish_independently', scope='perception',
           owner='core/perception/points_before_the_picture_is_reduced.py', observational=False)
def _independent_patch_fit_invariant() -> tuple:
    before = ndimage.gaussian_filter(np.random.default_rng(37).uniform(.05, .95, (64, 64, 3)), (.6, .6, 0.))
    after = ndimage.shift(before, (.4, -.7, 0.), order=1, mode='constant')
    after[:25, :25] = before[:25, :25]
    points = np.array(((12., 12.), (44., 44.)))
    initial = points + np.array(((0., 0.), (6., 4.)))
    grouped = _refine(before, after, points, initial)
    for index in range(len(points)):
        alone = _refine(before, after, points[index:index+1], initial[index:index+1])
        assert all(np.allclose(value[index:index+1], one, rtol=0., atol=1e-10)
                   for value, one in zip(grouped, alone, strict=True))
    return ()


def _new_marks(image: np.ndarray, occupied: list[SeenPoint], limit: int, scale: tuple,
               activity: np.ndarray | None = None, *, priority: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Spread the source corners by working-picture cells within the fixed point budget."""
    gy, gx = np.gradient(image)
    xx, xy, yy = (ndimage.uniform_filter(a * b, 3) for a, b in ((gx, gx), (gx, gy), (gy, gy)))
    strength = .5 * (xx + yy - np.sqrt((xx - yy) ** 2 + 4 * xy ** 2))
    peaks = (strength == ndimage.maximum_filter(strength, 5)) & (strength > max(.00015, .01 * float(strength.max())))
    peaks[:4] = peaks[-4:] = False
    peaks[:, :4] = peaks[:, -4:] = False
    for point in occupied:
        x, y = int(round(point.x)), int(round(point.y))
        peaks[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = False
    ys, xs = np.nonzero(peaks)
    cells: dict[tuple, list] = {}
    for y, x in zip(ys, xs, strict=True):
        cell = int(y * scale[1]) // 24, int(x * scale[0]) // 24
        cells.setdefault(cell, []).append((bool(activity[y, x]) if activity is not None else False,
                                          float(strength[y, x]), float(x), float(y)))
    for values in cells.values():
        values.sort(reverse=True)
    ranks = max((len(values) for values in cells.values()), default=0)

    def take(wanted: bool | None, budget: int) -> list[tuple[float, float]]:
        found = []
        if budget <= 0:
            return found
        for active_first in ((True, False) if activity is not None else (False,)):
            for rank in range(ranks):
                layer = (v[rank] for v in cells.values() if len(v) > rank and v[rank][0] == active_first
                         and (wanted is None or bool(priority[int(v[rank][3]), int(v[rank][2])]) == wanted))
                for _active, _strength, x, y in sorted(layer, reverse=True):
                    found.append((x, y))
                    if len(found) >= budget:
                        return found
        return found

    if priority is None:
        return take(None, limit)
    preferred = take(True, max(1, limit // 3))
    background = take(False, limit - len(preferred))
    if len(preferred) + len(background) < limit:
        preferred = take(True, limit - len(background))
    return preferred + background


class PointsBeforeReduction(FollowingPoints):
    """Source-resolution marks published in the existing working coordinate system."""

    def __init__(self, *, pixel_budget: int = 256_000, **kwargs) -> None:
        super().__init__(**kwargs)
        if type(pixel_budget) is not int or not 256 <= pixel_budget <= 1_000_000:
            raise ValueError('source tracking needs a bounded pixel budget')
        self.pixel_budget = pixel_budget
        self._native_points: tuple[SeenPoint, ...] = ()
        self._colours: np.ndarray | None = None
        self._geometry: tuple | None = None
        self._activity: np.ndarray | None = None
        self._priority: np.ndarray | None = None
        self.sampling_hint_receipt: Any = None

    def clear(self) -> None:
        super().clear()
        self._native_points, self._colours, self._geometry = (), None, None
        self._activity = None
        self._priority = None
        self.sampling_hint_receipt = None

    def motion_of(self, thing: Any, at: float) -> Any:
        """Publish the tracked grid's cell size in the same units as its coordinates."""
        motion = super().motion_of(thing, at)
        if motion is None or self._geometry is None:
            return motion
        _original, tracked, working = self._geometry
        size = max(working[axis]/tracked[axis] for axis in (0, 1))
        return replace(motion, pixel_size=size)

    def see_source(self, source: np.ndarray, working: np.ndarray, at: float, *,
                   priority_mask: np.ndarray | None = None, search_hint: Any = None) -> tuple[SeenPoint, ...]:
        """A source and its reduced view must belong to this one capture and geometry."""
        self.sampling_hint_receipt = None
        sampling_owner = None
        if search_hint is not None:
            from core.perception.a_search_hint_for_these_pixels import frame_priority_mask

            priority_mask = frame_priority_mask(search_hint, source, at)
            sampling_owner = search_hint if priority_mask is not None else None
            at = float(at)
        if (source.ndim != 3 or working.ndim != 3 or min(source.shape[:2]) < 16
                or min(working.shape[:2]) < 16 or min(source.shape[2], working.shape[2]) < 3
                or not math.isfinite(at) or self.limit <= 0):
            self.clear()
            return ()
        source_shape = source.shape[:2]
        factor = min(1., math.sqrt(self.pixel_budget / (source.shape[0] * source.shape[1])))
        if factor < 1.:
            from core.perception.picture_arithmetic import shrink

            width, height = int(source.shape[1] * factor), int(source.shape[0] * factor)
            if min(width, height) < 16:
                self.clear()
                return ()
            source = shrink(source, width, height)
        geometry = source_shape, source.shape[:2], working.shape[:2]
        if self._geometry is not None and (geometry != self._geometry or not 0 < at - self._at <= self.longest_gap):
            self.clear()
        self._geometry = geometry
        self._priority = None
        if (isinstance(priority_mask, np.ndarray) and priority_mask.dtype == bool
                and priority_mask.shape == source_shape and 0 < np.count_nonzero(priority_mask) <= priority_mask.size // 2):
            height, width = source.shape[:2]
            ys = np.minimum(((np.arange(height) + .5) * source_shape[0] / height).astype(int), source_shape[0] - 1)
            xs = np.minimum(((np.arange(width) + .5) * source_shape[1] / width).astype(int), source_shape[1] - 1)
            self._priority = priority_mask[np.ix_(ys, xs)].copy()
            if not self._priority.any():
                self._priority = None
        scale = working.shape[1] / source.shape[1], working.shape[0] / source.shape[0]
        colours = _colours(source)
        image = _pyramid(colours.mean(axis=2))
        self._activity = None
        if self._colours is not None and self._colours.shape == colours.shape:
            changed = np.max(abs(colours - self._colours), axis=2) > 12 / 255.
            if changed.any() and changed.mean() <= .6:
                self._activity = ndimage.maximum_filter(changed, 7)
        current = self._continued_source(image, colours, at) if self._image is not None and self._native_points else []
        if self._priority is None:
            self._priority = self.still_needed.priority(current, scale, image[0].shape, at)
        marks = (_new_marks(image[0], current, self.limit, scale, self._activity) if self._priority is None
                 else _new_marks(image[0], current, self.limit, scale, self._activity, priority=self._priority))
        for x, y in marks:
            current.append(SeenPoint(self._number, x, y, x, y, at, at, at, 0., 1))
            self._number += 1
        self._native_points = self.still_needed.keep(current, self._spread(current, scale), at, self.limit)
        self.points = tuple(SeenPoint(p.number, p.x * scale[0], p.y * scale[1], p.old_x * scale[0], p.old_y * scale[1],
                            p.born, p.before, p.at, p.error, p.pictures) for p in self._native_points)
        self._image, self._colours, self._at = image, colours, at
        self._remember_points()
        if sampling_owner is not None and self._priority is not None:
            from core.perception.a_search_hint_for_these_pixels import SamplingHintReceipt

            preferred = []
            for point in self._native_points:
                x, y = int(round(point.x)), int(round(point.y))
                if 0 <= y < self._priority.shape[0] and 0 <= x < self._priority.shape[1] and self._priority[y, x]:
                    preferred.append(point)
            hint = sampling_owner
            self.sampling_hint_receipt = SamplingHintReceipt(
                hint.surface, hint.epoch, hint.sequence, hint.lower, hint.upper, hint.arrived, hint.source,
                hint.source_at, hint.shape, hint.digest, hint.region, True, int(np.count_nonzero(self._priority)),
                sum(point.pictures == 1 and point.born == at and point.before == at for point in preferred),
                len(preferred), len(self._native_points) - len(preferred))
        return self.points

    def _spread(self, points: list[SeenPoint], scale: tuple) -> tuple[SeenPoint, ...]:
        if self._priority is not None:
            from core.perception.points_shared_with_changing_regions import share_point_capacity

            preferred, background = [], []
            for point in points:
                x, y = int(round(point.x)), int(round(point.y))
                inside = 0 <= y < self._priority.shape[0] and 0 <= x < self._priority.shape[1] and self._priority[y, x]
                (preferred if inside else background).append(point)
            kept = list(self._spatial_share(preferred, scale, max(1, self.limit // 3)))
            room = self.limit - len(kept)
            kept.extend(share_point_capacity(background, self._activity, scale, room, quiet_budget=self.limit // 3)
                        if self._activity is not None else self._spatial_share(background, scale, room))
            if len(kept) < self.limit:
                names = {point.number for point in kept}
                unused = [point for point in points if point.number not in names]
                kept.extend(self._spatial_share(unused, scale, self.limit - len(kept)))
            return tuple(kept)
        if self._activity is not None:
            from core.perception.points_shared_with_changing_regions import share_point_capacity

            return share_point_capacity(points, self._activity, scale, self.limit)
        return self._spatial_share(points, scale, self.limit)

    @staticmethod
    def _spatial_share(points: list[SeenPoint], scale: tuple, limit: int) -> tuple[SeenPoint, ...]:
        cells: dict[tuple, list] = {}
        for point in points:
            cells.setdefault((int(point.y * scale[1]) // 24, int(point.x * scale[0]) // 24), []).append(point)
        for values in cells.values():
            values.sort(key=lambda p: (-p.pictures, p.error, p.number))
        kept = []
        for rank in range(max((len(values) for values in cells.values()), default=0)):
            kept.extend(sorted((v[rank] for v in cells.values() if len(v) > rank), key=lambda p: (-p.pictures, p.error, p.number)))
            if len(kept) >= limit:
                break
        return tuple(kept[:limit])

    def _continued_source(self, image: list, colours: np.ndarray, at: float) -> list[SeenPoint]:
        positions = np.asarray([(p.x, p.y) for p in self._native_points], dtype=float)
        before = self._colours
        rests = np.stack([abs(_sample(before[..., c], positions) - _sample(colours[..., c], positions)) for c in range(3)], axis=-1)
        unchanged = rests.max(axis=(1, 2)) <= 1 / 255.
        found, errors, valid = positions.copy(), rests.mean(axis=(1, 2)), unchanged.copy()
        changed = np.flatnonzero(~unchanged)
        if len(changed):
            initial, forwards = _follow(self._image, image, positions[changed])
            ends, error, angle, fitted = _refine(before, colours, positions[changed], initial)
            reverse, backwards = _follow(image, self._image, ends)
            starts, back_error, back_angle, back_fitted = _refine(colours, before, ends, reverse)
            valid[changed] = (forwards & backwards & fitted & back_fitted
                & (np.linalg.norm(starts - positions[changed], axis=1) < .5)
                & (error < .025) & (back_error < .025) & (abs(angle + back_angle) < .08))
            found[changed], errors[changed] = ends, error
        height, width = image[0].shape
        valid &= (np.isfinite(found).all(axis=1) & (found[:, 0] >= 3) & (found[:, 0] < width - 3)
                  & (found[:, 1] >= 3) & (found[:, 1] < height - 3))
        measured = [SeenPoint(p.number, float(found[i, 0]), float(found[i, 1]), p.x, p.y, p.born,
                             self._at, at, float(errors[i]), p.pictures + 1)
                    for i, p in enumerate(self._native_points) if valid[i]]
        if valid.all():
            return measured
        from core.perception.patches_compared_on_a_common_grid import recover_source_marks

        return recover_source_marks(self, image, colours, at, measured)


@invariant("perception.sampling_hint_receipt_comes_from_kept_points", scope="perception",
           owner="core/perception/points_before_the_picture_is_reduced.py", observational=False)
def _sampling_hint_receipt_invariant() -> tuple:
    from core.perception.a_search_hint_for_these_pixels import (
        make_search_hint,
        sampling_hint_matches,
    )
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    picture = np.random.default_rng(41).integers(20, 240, (48, 64, 3), dtype=np.uint8)
    at = CanvasFrameTime(1., lower=1., upper=1.01, arrived=1.02, source="canvas-bracket",
                         source_at=2., epoch=3, surface="document/canvas", sequence=4)
    hint = make_search_hint(picture, at, normalized_region=(0., 0., .5, .5))
    observer = PointsBeforeReduction(limit=12)
    observer.see_source(picture, picture, at, search_hint=hint)
    receipt = observer.sampling_hint_receipt
    assert sampling_hint_matches(receipt, hint)
    assert receipt.kept_priority_points + receipt.background_points == len(observer.points)
    assert receipt.fresh_priority_points == receipt.kept_priority_points
    assert all(type(point.at) is float for point in observer.points)
    observer.see_source(picture, picture, 1.1)
    assert observer.sampling_hint_receipt is None
    return ()


@invariant("perception.pixel_search_keeps_quiet_point_capacity", scope="perception",
           owner="core/perception/points_before_the_picture_is_reduced.py", observational=False)
def _pixel_search_capacity_invariant() -> tuple:
    observer = PointsBeforeReduction(limit=9)
    observer._priority = np.zeros((64, 64), bool)
    observer._priority[:20, :20] = True
    observer._activity = np.zeros((64, 64), bool)
    observer._activity[30:40] = True
    preferred = [SeenPoint(i, 5. + i, 5., 5. + i, 5., 1., 1., 1., 0., 1) for i in range(6)]
    changed = [SeenPoint(i, 5. + i, 35., 5. + i, 35., 1., 1., 1., 0., 1) for i in range(6, 12)]
    quiet = [SeenPoint(i, 5. + i, 55., 5. + i, 55., 0., .9, 1., 0., 20) for i in range(12, 18)]
    kept = observer._spread(preferred + changed + quiet, (1., 1.))
    assert len(kept) == 9 and sum(point.number < 6 for point in kept) == 3
    assert sum(point.number >= 12 for point in kept) == 3
    assert all(point.pictures == 1 and point.before == point.at for point in kept if point.number < 6)
    return ()


@invariant('perception.source_marks_return_to_one_working_geometry', scope='perception',
           owner='core/perception/points_before_the_picture_is_reduced.py', observational=False)
def _source_point_geometry_invariant() -> tuple:
    point = SeenPoint(1, 60., 30., 58., 28., 0., 1., 2., .01, 3)
    scale = .25, .5
    exposed = SeenPoint(point.number, point.x * scale[0], point.y * scale[1],
                        point.old_x * scale[0], point.old_y * scale[1], point.born, point.before,
                        point.at, point.error, point.pictures)
    assert (exposed.x, exposed.y, exposed.old_x, exposed.old_y) == (15., 15., 14.5, 14.)
    assert exposed.number == point.number and exposed.before == point.before and exposed.at == point.at
    return ()


@invariant('perception.point_resolution_keeps_the_tracked_grids_units', scope='perception',
           owner='core/perception/points_before_the_picture_is_reduced.py', observational=False)
def _source_point_precision_invariant() -> tuple:
    from types import SimpleNamespace

    image = np.full((120, 160, 3), 25, np.uint8)
    image[35:65, 40:64] = np.random.default_rng(47).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    observer = PointsBeforeReduction(limit=96)
    working = image[::2, ::2].copy()
    observer.see_source(image, working, 1.)
    observer.see_source(image, working, 1.1)
    body = SimpleNamespace(x=26., y=25., w=12., h=15., seen=1., support=np.ones((15, 12), bool))
    motion = observer.motion_of(body, 1.1)
    assert motion is not None and motion.pixel_size == .5
    return ()
