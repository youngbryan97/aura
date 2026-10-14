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
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.rigid_motion_from_named_points import RigidMotion, rigid_motion
from core.verify import invariant


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

    def __init__(self, *, limit: int = 192, longest_gap: float = .35) -> None:
        self.limit = max(0, min(384, int(limit)))
        self.longest_gap = longest_gap
        self.points: tuple[SeenPoint, ...] = ()
        self._image: list[np.ndarray] | None = None
        self._at = -math.inf
        self._number = 0

    def clear(self) -> None:
        self.points = ()
        self._image = None
        self._at = -math.inf

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
        for x, y in _new_points(image[0], current, self.limit):
            current.append(SeenPoint(self._number, x, y, x, y, at, at, at, 0., 1))
            self._number += 1
        self.points = _spread_points(current, self.limit)
        self._image, self._at = image, at
        return self.points

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
        points = [p for p in self.points if p.pictures >= 2 and p.before == thing.seen and p.at == at
                  and abs(p.old_x - thing.x) <= thing.w / 2 and abs(p.old_y - thing.y) <= thing.h / 2]
        mask = getattr(thing, "support", None)
        if mask is None or mask.ndim != 2 or not mask.size:
            return None
        left, top = thing.x - thing.w / 2, thing.y - thing.h / 2
        points = [p for p in points if 0 <= (x := int(round(p.old_x - left))) < mask.shape[1]
                  and 0 <= (y := int(round(p.old_y - top))) < mask.shape[0] and mask[y, x]]
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
            middle = rigid.apply((thing.x, thing.y)) - (thing.x, thing.y)
        return PointMotion(float(middle[0]), float(middle[1]), max(p.error for p in support),
                           len(support), thing.seen, at, tuple(p.number for p in support),
                           tuple((p.x, p.y) for p in support), rigid)


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
