"""Follow visible texture points independently of object outlines.

Shi and Tomasi's corner measure supplies points; pyramidal Lucas-Kanade
follows them. A reverse journey and a comparison of the sampled patches
reject bad matches. Point identities survive changing object proposals.
Numpy and scipy keep this usable in the desktop process without OpenCV.

References: Shi and Tomasi, Good Features to Track (1994); Lucas and
Kanade, An Iterative Image Registration Technique (1981).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.position_from_the_same_points import PointPose, last_observed_outline
from core.perception.rigid_motion_from_named_points import RigidMotion, rigid_motion
from core.verify import invariant

NAMED_POINT_GAP_S = .35


@dataclass(frozen=True)
class SeenPoint:
    number: int
    x: float
    y: float
    old_x: float
    old_y: float
    born: float
    before: float
    at: float
    error: float
    pictures: int


@dataclass(frozen=True)
class PointMotion:
    dx: float
    dy: float
    error: float
    support: int
    before: float
    at: float
    points: tuple[int, ...] = ()
    locations: tuple[tuple[float, float], ...] = ()
    rigid: RigidMotion | None = None
    pixel_size: float = 1.

    def has_rigid_pose(self) -> bool:
        pose = self.rigid
        return (pose is not None and len(pose.offset) == 2
                and isinstance(self.pixel_size, (int, float)) and not isinstance(self.pixel_size, bool)
                and math.isfinite(self.pixel_size) and self.pixel_size > 0.
                and all(math.isfinite(v) for v in (pose.angle, *pose.offset, pose.error))
                and 0. <= pose.error <= .75 and self.support >= 3 and self.support == len(self.points) == len(self.locations)
                and len(set(self.points)) == self.support
                and all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in self.locations))

    def inside(self, blob: dict) -> bool:
        """A proposed outline must still contain most of the texture that supplied its motion."""
        mask = blob.get("support")
        if not self.locations or mask is None:
            return True
        left, top = blob["x"] - blob["w"] / 2, blob["y"] - blob["h"] / 2
        count = sum(0 <= (x := int(round(px - left))) < mask.shape[1]
                    and 0 <= (y := int(round(py - top))) < mask.shape[0] and bool(mask[y, x])
                    for px, py in self.locations)
        return count >= 3 and count >= .6 * len(self.locations)


def _grey(picture: np.ndarray) -> np.ndarray:
    return ndimage.gaussian_filter(np.asarray(picture[..., :3], dtype=np.float32).mean(axis=2) / 255., .6)


def _pyramid(picture: np.ndarray) -> list[np.ndarray]:
    images = [picture]
    for _ in range(2):
        images.append(ndimage.gaussian_filter(images[-1], .8)[::2, ::2])
    return images


def _sample(picture: np.ndarray, points: np.ndarray) -> np.ndarray:
    offsets = np.mgrid[-3:4, -3:4].reshape(2, -1).T[:, ::-1]
    places = points[:, None, :] + offsets[None, :, :]
    return ndimage.map_coordinates(picture, [places[..., 1], places[..., 0]], order=1, mode="constant", cval=0.)


def _follow(before: list[np.ndarray], after: list[np.ndarray], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.zeros_like(points)
    valid = np.ones(len(points), dtype=bool)
    for level in range(len(before) - 1, -1, -1):
        original, current = before[level], after[level]
        centers = points / (2 ** level)
        displacement *= 2 if level != len(before) - 1 else 1
        gy, gx = np.gradient(original)
        template = _sample(original, centers)
        x, y = _sample(gx, centers), _sample(gy, centers)
        xx, xy, yy = (np.sum(a * b, axis=1) for a, b in ((x, x), (x, y), (y, y)))
        det = xx * yy - xy * xy
        useful = det > 1e-7
        for _ in range(8):
            error = template - _sample(current, centers + displacement)
            bx, by = np.sum(x * error, axis=1), np.sum(y * error, axis=1)
            step = np.column_stack(((yy * bx - xy * by) / np.maximum(det, 1e-9),
                                    (xx * by - xy * bx) / np.maximum(det, 1e-9)))
            step[~useful] = 0
            displacement += np.clip(step, -2., 2.)
            if np.max(np.linalg.norm(step[useful], axis=1), initial=0.) < .02:
                break
        if level == 0:
            valid &= useful
    found = points + displacement
    height, width = after[0].shape
    valid &= np.all(np.isfinite(found), axis=1)
    valid &= (found[:, 0] >= 3) & (found[:, 0] < width - 3) & (found[:, 1] >= 3) & (found[:, 1] < height - 3)
    return found, valid


def _new_points(picture: np.ndarray, occupied: list[SeenPoint], budget: int) -> list[tuple[float, float]]:
    """Spread corners across the picture so one textured patch cannot take every slot."""
    if budget <= 0:
        return []
    gy, gx = np.gradient(picture)
    xx, xy, yy = (ndimage.uniform_filter(a * b, 3) for a, b in ((gx, gx), (gx, gy), (gy, gy)))
    strength = .5 * (xx + yy - np.sqrt((xx - yy) ** 2 + 4 * xy ** 2))
    peaks = (strength == ndimage.maximum_filter(strength, 5)) & (strength > max(.00015, .01 * float(strength.max())))
    peaks[:4] = peaks[-4:] = False
    peaks[:, :4] = peaks[:, -4:] = False
    for point in occupied:
        x, y = int(round(point.x)), int(round(point.y))
        peaks[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = False
    ys, xs = np.nonzero(peaks)
    cells: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for y, x in zip(ys, xs, strict=True):
        cells.setdefault((int(y) // 24, int(x) // 24), []).append((float(strength[y, x]), int(x), int(y)))
    for cell in cells.values():
        cell.sort(reverse=True)
    found = []
    for rank in range(max((len(cell) for cell in cells.values()), default=0)):
        layer = sorted((cell[rank] for cell in cells.values() if len(cell) > rank), reverse=True)
        for _strength, x, y in layer:
            found.append((float(x), float(y)))
            if len(found) == budget:
                return found
    return found


def _spread_points(points: list[SeenPoint], limit: int) -> tuple[SeenPoint, ...]:
    """Share slots across occupied cells, retaining old identities within each cell.

    Existing scenery must leave room for new corners elsewhere. Old points
    are removed only when their cell would consume another cell's share.
    """
    cells: dict[tuple[int, int], list[SeenPoint]] = {}
    for point in points:
        cells.setdefault((int(point.y) // 24, int(point.x) // 24), []).append(point)
    for cell in cells.values():
        cell.sort(key=lambda p: (-p.pictures, p.error, p.number))
    ordered = []
    for rank in range(max((len(cell) for cell in cells.values()), default=0)):
        layer = [cell[rank] for cell in cells.values() if len(cell) > rank]
        layer.sort(key=lambda p: (-p.pictures, p.error, p.number))
        ordered.extend(layer)
        if len(ordered) >= limit:
            break
    return tuple(ordered[:limit])


class FollowingPoints:
    """A bounded set of point identities, each backed by consecutive measured pictures."""

    def __init__(self, *, limit: int = 192, longest_gap: float = NAMED_POINT_GAP_S) -> None:
        self.limit = max(0, min(384, int(limit)))
        self.longest_gap = longest_gap
        self.points: tuple[SeenPoint, ...] = ()
        self._image: list[np.ndarray] | None = None
        self._at = -math.inf
        self._number = 0
        self._point_history: dict[float, dict[int, SeenPoint]] = {}

    def clear(self) -> None:
        self.points = ()
        self._image = None
        self._at = -math.inf
        self._point_history.clear()

    def see(self, picture: np.ndarray, at: float) -> tuple[SeenPoint, ...]:
        """Track points in a current RGB picture; a gap, resize or reset starts new identities."""
        if picture.ndim != 3 or picture.shape[2] < 3 or min(picture.shape[:2]) < 16 or not math.isfinite(at):
            self.clear()
            return ()
        image = _pyramid(_grey(picture))
        current: list[SeenPoint] = []
        if (self._image is not None and self._image[0].shape == image[0].shape
                and 0 < at - self._at <= self.longest_gap and self.points):
            current = self._continued(image, at)
        else:
            self._point_history.clear()
        for x, y in _new_points(image[0], current, self.limit):
            current.append(SeenPoint(self._number, x, y, x, y, at, at, at, 0., 1))
            self._number += 1
        self.points = _spread_points(current, self.limit)
        self._image, self._at = image, at
        self._remember_points()
        return self.points

    def _remember_points(self, *, replacing_current: bool = False) -> None:
        """Retain bounded measured coordinates while an outline can briefly be absent."""
        if self._point_history and (self._at < next(reversed(self._point_history))
                or self._at == next(reversed(self._point_history)) and not replacing_current):
            self._point_history.clear()
        self._point_history[self._at] = {point.number: point for point in self.points}
        while self._point_history and (len(self._point_history) > 32
                or self._at-next(iter(self._point_history)) > min(self.longest_gap, NAMED_POINT_GAP_S)):
            del self._point_history[next(iter(self._point_history))]

    def _points_since(self, seen: float, at: float) -> tuple[SeenPoint, ...]:
        """An older object support can own only marks seen at every intervening picture."""
        if not isinstance(seen, (int, float)) or not isinstance(at, (int, float)) or not math.isfinite(seen+at):
            return ()
        if (self._at != at or not 0 < at-seen <= min(self.longest_gap, NAMED_POINT_GAP_S)
                or seen not in self._point_history):
            return self.points
        frames = [(when, points) for when, points in self._point_history.items() if seen <= when <= at]
        if len(frames) < 3 or frames[0][0] != seen or frames[-1][0] != at:
            return self.points
        result = [point for point in self.points if point.before == seen and point.at == at]
        for point in self.points:
            if point.before == seen:
                continue
            chain = [points.get(point.number) for _when, points in frames]
            if (any(p is None for p in chain)
                    or any(p.born != point.born or not p.born <= seen or p.at != when
                           or p.pictures < 1 or p.pictures != chain[0].pictures+index
                           or not math.isfinite(p.error) or p.error < 0
                           or (index and (p.before != frames[index-1][0]
                               or (p.old_x, p.old_y) != (chain[index-1].x, chain[index-1].y)))
                           for index, ((when, _points), p) in enumerate(zip(frames, chain, strict=True)))):
                continue
            result.append(replace(point, old_x=chain[0].x, old_y=chain[0].y, before=seen,
                                  error=sum(p.error for p in chain[1:])))
        return tuple(result)

    def _continued(self, image: list[np.ndarray], at: float) -> list[SeenPoint]:
        positions = np.array([(p.x, p.y) for p in self.points], dtype=np.float32)
        original = _sample(self._image[0], positions)
        at_rest = np.abs(original - _sample(image[0], positions))
        unchanged = at_rest.max(axis=1) <= 1 / 255.
        found, valid = positions.copy(), unchanged.copy()
        error = at_rest.mean(axis=1)
        # Most scenery points have exactly the same local pixels. They have
        # a measured zero shift without eight iterations of registration.
        changing = np.flatnonzero(~unchanged)
        if len(changing):
            moved, forward_valid = _follow(self._image, image, positions[changing])
            back, reverse_valid = _follow(image, self._image, moved)
            round_trip = np.linalg.norm(back - positions[changing], axis=1)
            patch_error = np.mean(np.abs(original[changing] - _sample(image[0], moved)), axis=1)
            found[changing], error[changing] = moved, patch_error
            valid[changing] = forward_valid & reverse_valid & (round_trip < .7) & (patch_error < .06)
        return [SeenPoint(old.number, float(found[i, 0]), float(found[i, 1]), old.x, old.y,
                          old.born, self._at, at, float(error[i]), old.pictures + 1)
                for i, old in enumerate(self.points) if valid[i]]

    def motion_of(self, thing: Any, at: float) -> PointMotion | None:
        """The common displacement of points spread through this last observed object box."""
        bounds = last_observed_outline(thing)
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        center = (left+right)/2, (top+bottom)/2
        points = [p for p in self._points_since(thing.seen, at) if p.pictures >= 2 and p.before == thing.seen and p.at == at
                  and left <= p.old_x <= right and top <= p.old_y <= bottom]
        mask = getattr(thing, "support", None)
        if mask is None or mask.ndim != 2 or not mask.size:
            return None
        points = [p for p in points if 0 <= (x := int(round(p.old_x - left))) < mask.shape[1]
                  and 0 <= (y := int(round(p.old_y - top))) < mask.shape[0] and mask[y, x]]
        if isinstance(getattr(thing, 'point_pose', None), PointPose):
            return _motion_of_retained_marks(thing, points, center, at)
        if len(points) < 3:
            return None
        differences = np.array([(p.x - p.old_x, p.y - p.old_y) for p in points])
        middle = np.median(differences, axis=0)
        coherent = np.linalg.norm(differences - middle, axis=1) < 1.
        rigid = None
        if coherent.sum() < 3 or coherent.mean() < .7:
            starts = np.array([(p.old_x, p.old_y) for p in points])
            ends = np.array([(p.x, p.y) for p in points])
            rigid = rigid_motion(starts, ends)
            if rigid is None:
                return None
            coherent = np.zeros(len(points), dtype=bool)
            coherent[list(rigid.inliers)] = True
        support = [p for p, keep in zip(points, coherent, strict=True) if keep]
        places = np.array([(p.old_x, p.old_y) for p in support])
        if np.max(np.ptp(places, axis=0) / np.maximum((thing.w, thing.h), 1.)) < .3:
            return None
        middle = np.median(differences[coherent], axis=0)
        if rigid is not None:
            middle = rigid.apply(center) - center
        return PointMotion(float(middle[0]), float(middle[1]), max(p.error for p in support),
                           len(support), thing.seen, at, tuple(p.number for p in support),
                           tuple((p.x, p.y) for p in support), rigid)


def _motion_of_retained_marks(thing: Any, points: list[SeenPoint], center: tuple, at: float) -> PointMotion | None:
    """Previously owned marks choose the motion before a joined outline can add other pixels."""
    pose = thing.point_pose
    if (pose.at != thing.seen or pose.born != thing.born or not 0 < at-thing.seen <= NAMED_POINT_GAP_S
            or not isinstance(pose.anchors, dict) or not 3 <= len(pose.anchors) <= 256
            or not isinstance(pose.visible_marks, dict)):
        return None
    owned = [p for p in points if p.number in pose.anchors and p.number in pose.visible_marks
             and math.dist((p.old_x, p.old_y), pose.visible_marks[p.number]) <= 1e-6]
    if (len(owned) < max(3, math.ceil(.6*len(pose.anchors)))
            or len({p.number for p in owned}) != len(owned)):
        return None
    starts = np.asarray([(p.old_x, p.old_y) for p in owned])
    ends = np.asarray([(p.x, p.y) for p in owned])
    fitted = rigid_motion(starts, ends)
    if fitted is None:
        return None
    # Other corners can join only through this observed transform. A larger
    # stationary consensus elsewhere in the outline cannot replace it.
    predicted = fitted.apply(np.asarray([(p.old_x, p.old_y) for p in points]))
    coherent = np.linalg.norm(predicted-np.asarray([(p.x, p.y) for p in points]), axis=1) <= .75
    continuing = [p for p, agrees in zip(points, coherent, strict=True) if agrees and p.number in pose.anchors]
    extra = [p for p, agrees in zip(points, coherent, strict=True) if agrees and p.number not in pose.anchors]
    used = continuing + extra[:256-len(continuing)]
    retained = set(pose.anchors) & {p.number for p in used}
    if len(retained) < max(3, math.ceil(.6*len(pose.anchors))):
        return None
    measured = rigid_motion(np.asarray([(p.old_x, p.old_y) for p in used]),
                            np.asarray([(p.x, p.y) for p in used]))
    if measured is None or np.max(np.linalg.norm(measured.apply(starts)-fitted.apply(starts), axis=1)) > .75:
        return None
    support = [used[index] for index in measured.inliers]
    if (len(set(pose.anchors) & {p.number for p in support}) < max(3, math.ceil(.6*len(pose.anchors)))
            or len({p.number for p in support}) != len(support)):
        return None
    shift = measured.apply(center)-center
    return PointMotion(float(shift[0]), float(shift[1]), max(p.error for p in support), len(support),
                       thing.seen, at, tuple(p.number for p in support), tuple((p.x, p.y) for p in support), measured)


@invariant('perception.retained_material_precedes_a_joined_outline_majority', scope='perception',
           owner='core/perception/points_that_keep_their_motion.py', observational=False)
def _owned_material_motion_invariant() -> tuple:
    from types import SimpleNamespace

    marks = dict(enumerate(((15., 15.), (25., 15.), (15., 25.), (25., 25.))))
    pose = PointPose(0., 1., (20., 20.), marks.copy(), (20., 20.), visible_marks=marks.copy())
    body = SimpleNamespace(point_pose=pose, born=0., seen=1.)
    points = [SeenPoint(n, x+4., y, x, y, 0., 1., 1.1, .01, 3) for n, (x, y) in marks.items()]
    points.extend(SeenPoint(n, x, y, x, y, 0., 1., 1.1, .01, 3)
                  for n, (x, y) in enumerate(((x, y) for x in (35., 45., 55.) for y in (35., 45., 55.)), 4))
    measured = _motion_of_retained_marks(body, points, (30., 30.), 1.1)
    assert measured is not None and abs(measured.dx-4.) < 1e-9 and set(measured.points) == set(marks)
    assert _motion_of_retained_marks(body, points[2:], (30., 30.), 1.1) is None
    return ()


@invariant('perception.a_missing_outline_requires_continuous_observed_marks', scope='perception',
           owner='core/perception/points_that_keep_their_motion.py', observational=False)
def _point_history_invariant() -> tuple:
    from types import SimpleNamespace

    tracker = FollowingPoints()
    marks = ((10., 10.), (20., 10.), (10., 20.), (20., 20.))
    for step, at in enumerate((1., 1.1, 1.2)):
        tracker.points = tuple(SeenPoint(name, x+step*2., y, x+max(0, step-1)*2., y,
                               1., at if step == 0 else (1., 1.1, 1.2)[step-1], at, .01, step+1)
                               for name, (x, y) in enumerate(marks))
        tracker._at = at
        tracker._remember_points()
    body = SimpleNamespace(x=15., y=15., w=20., h=20., seen=1., support=np.ones((20, 20), bool))
    motion = tracker.motion_of(body, 1.2)
    assert motion is not None and motion.dx == 4. and motion.before == 1.
    tracker._point_history[1.1].clear()
    assert tracker.motion_of(body, 1.2) is None
    return ()


@invariant("perception.point_identity_needs_current_pixels", scope="perception",
           owner="core/perception/points_that_keep_their_motion.py", observational=False)
def _point_identity_invariant() -> tuple:
    image = np.full((48, 64, 3), 25, np.uint8)
    image[12:36, 16:40] = np.random.default_rng(71).integers(30, 240, (24, 24, 3), dtype=np.uint8)
    points = FollowingPoints(limit=12)
    first = points.see(image, 1.)
    assert first and len(first) <= 12
    continued = points.see(image, 1.1)
    assert any(p.pictures == 2 and p.born == 1. for p in continued)
    assert not points.see(np.full_like(image, 25), 1.2)
    after_gap = points.see(image, 2.)
    assert not set(p.number for p in first) & set(p.number for p in after_gap)
    points.clear()
    assert not points.points and points._image is None
    return ()
