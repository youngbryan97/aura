"""Measure bounded pixel motion before an object has a track or a name.

Correspondences come from current RGB patches in both directions. A spatially
spread translation consensus supplies a scene reference, without claiming
that the reference is a camera or that any residual belongs to a body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.perception.picture_arithmetic import shrink
from core.perception.points_before_the_picture_is_reduced import _colours, _refine
from core.perception.points_that_keep_their_motion import _follow, _grey, _new_points, _pyramid
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant

MAX_WIDTH = 320
MAX_HEIGHT = 240
MAX_SOURCE_PIXELS = 8_388_608
MAX_POINTS = 256
#: Working-pixel excursion below which a match cannot propose a moving layer.
ANGULAR_MOVEMENT_MIN = .6
#: The largest local correspondence residual admitted to a motion-layer fit.
LAYER_ERROR_MAX = .75


@dataclass(frozen=True)
class PixelCorrespondence:
    before: tuple[float, float]
    after: tuple[float, float]
    pixel_error: float
    round_trip_error: float


@dataclass(frozen=True)
class CameraTranslation:
    """A majority scene translation; its material ownership remains unknown."""

    dx: float
    dy: float
    uncertainty: float
    support: int
    fraction: float
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PixelMotionSignature:
    """Immutable search measurements, with missing fits marked by zero support.

    Features are row-major tile values in working pixels: x/y shift and
    tile width times fitted local rotation in radians. A measured reference
    is subtracted only for ``distributed_translation``. The optional
    strain holds x/y expansion and symmetric shear, also scaled by tile width.
    A zero feature without support is unavailable, not measured stillness.
    Translation support counts distinct verified points in the fixed local
    neighborhood; ``point_counts`` retains counts inside each tile itself.
    Angular support counts the actual matches retained by its motion layer.
    """

    source_shape: tuple[int, int]
    working_shape: tuple[int, int]
    shape: tuple[int, int]
    tile: int
    features: tuple[tuple[float, float, float], ...]
    support: tuple[int, ...]
    point_counts: tuple[int, ...]
    angular_support: tuple[int, ...]
    strain: tuple[tuple[float, float, float], ...]
    fit_error: tuple[float | None, ...]
    camera: CameraTranslation | None
    reference: str
    correspondences: tuple[PixelCorrespondence, ...]


def _pictures(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if (not isinstance(before, np.ndarray) or not isinstance(after, np.ndarray)
            or before.ndim != 3 or before.shape[2] != 3 or before.shape != after.shape
            or before.dtype != np.uint8 or after.dtype != np.uint8
            or min(before.shape[:2]) < 32 or before.shape[0] * before.shape[1] > MAX_SOURCE_PIXELS):
        return None
    height, width = before.shape[:2]
    scale = min(1., MAX_WIDTH / width, MAX_HEIGHT / height)
    wide, tall = int(width * scale), int(height * scale)
    if min(wide, tall) < 32:
        return None
    if scale < 1.:
        return shrink(before, wide, tall), shrink(after, wide, tall)
    return before, after


def _correspondences(before: np.ndarray, after: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, tuple]:
    first, last = _pyramid(_grey(before)), _pyramid(_grey(after))
    starts = np.asarray(_new_points(first[0], [], limit), dtype=float).reshape(-1, 2)
    if not len(starts):
        return starts, starts.copy(), ()
    initial, forward = _follow(first, last, starts)
    first_rgb, last_rgb = _colours(before), _colours(after)
    ends, pixel_error, angle, fitted = _refine(first_rgb, last_rgb, starts, initial)
    initial_back, reverse = _follow(last, first, ends)
    returned, back_error, back_angle, back_fitted = _refine(last_rgb, first_rgb, ends, initial_back)
    round_trip = np.linalg.norm(returned - starts, axis=1)
    height, width = before.shape[:2]
    valid = (forward & reverse & fitted & back_fitted
             & np.isfinite(ends).all(axis=1) & np.isfinite(pixel_error) & np.isfinite(back_error)
             & (round_trip < .7) & (pixel_error < .035) & (back_error < .035)
             & (abs(angle + back_angle) < .08)
             & (ends[:, 0] >= 4) & (ends[:, 0] < width - 4)
             & (ends[:, 1] >= 4) & (ends[:, 1] < height - 4))
    receipts = tuple(PixelCorrespondence(tuple(float(v) for v in start), tuple(float(v) for v in end),
                                         float(max(error, reverse_error)), float(retraced))
                     for start, end, error, reverse_error, retraced in zip(
                         starts[valid], ends[valid], pixel_error[valid], back_error[valid], round_trip[valid], strict=True))
    return starts[valid], ends[valid], receipts


def _camera(starts: np.ndarray, ends: np.ndarray, shape: tuple[int, int]) -> CameraTranslation | None:
    if len(starts) < 12:
        return None
    shifts = ends - starts
    shift = np.median(shifts, axis=0)
    inside = np.linalg.norm(shifts - shift, axis=1) <= .75
    if inside.sum() < 12 or inside.mean() < .65:
        return None
    shift = np.median(shifts[inside], axis=0)
    inside = np.linalg.norm(shifts - shift, axis=1) <= .75
    if inside.sum() < 12 or inside.mean() < .65:
        return None
    height, width = shape
    places = starts[inside]
    if np.any(np.ptp(places, axis=0) < np.asarray((width, height)) * .55):
        return None
    cells = tuple(sorted({(min(2, int(3 * y / height)), min(2, int(3 * x / width))) for x, y in places}))
    if len(cells) < 6 or len({row for row, _ in cells}) < 3 or len({column for _, column in cells}) < 3:
        return None
    centered = places - places.mean(axis=0)
    covariance = centered.T @ centered
    if np.linalg.det(covariance) <= .05 * covariance[0, 0] * covariance[1, 1]:
        return None
    residual = np.linalg.norm(shifts[inside] - shift, axis=1)
    return CameraTranslation(float(shift[0]), float(shift[1]), max(.25, float(residual.max())),
                             int(inside.sum()), float(inside.mean()), cells)


def _affine_fit(starts: np.ndarray, residual: np.ndarray, center: np.ndarray) -> tuple | None:
    if len(starts) < 4:
        return None
    centered = starts - center
    if np.linalg.svd(centered - centered.mean(axis=0), compute_uv=False)[-1] < 2.:
        return None
    design = np.column_stack((centered, np.ones(len(starts))))
    fitted, _error, rank, _singular = np.linalg.lstsq(design, residual, rcond=None)
    if rank != 3:
        return None
    error = float(np.max(np.linalg.norm(design @ fitted - residual, axis=1)))
    if error > LAYER_ERROR_MAX:
        return None
    transform = np.eye(2) + fitted[:2].T
    if np.linalg.det(transform) <= 0:
        return None
    return fitted, transform, error


def _local_affine(starts: np.ndarray, residual: np.ndarray, center: np.ndarray, tile: int) -> tuple | None:
    fit = _affine_fit(starts, residual, center)
    if fit is None:
        return None
    fitted, transform, error = fit
    angle = math.atan2(float(transform[1, 0] - transform[0, 1]), float(np.trace(transform)))
    strain = tile * fitted[0, 0], tile * fitted[1, 1], tile * (fitted[0, 1] + fitted[1, 0]) / 2
    return float(tile * angle), tuple(float(v) for v in strain), error


def _angular_motion_layer(starts: np.ndarray, residual: np.ndarray, center: np.ndarray, tile: int, *,
                          owned: np.ndarray | None = None) -> tuple | None:
    """Measure a coherent moving layer among nearby still marks.

    The 0.6-pixel excursion threshold separates proposals, not point validity.
    Stillness can be measured when all matches stay below it. Competing moving
    layers require a 70-percent rigid consensus, with at least four actual
    noncollinear inliers. The target tile must contain a retained moving
    match. A tied response remains unavailable.
    """
    if not 4 <= len(starts) <= MAX_POINTS:
        return None
    if owned is None:
        owned = np.ones(len(starts), dtype=bool)
    if (not isinstance(owned, np.ndarray) or owned.shape != (len(starts),)
            or owned.dtype != bool or not owned.any()):
        return None
    moving = np.linalg.norm(residual, axis=1) > ANGULAR_MOVEMENT_MIN
    if not moving.any():
        measured = _local_affine(starts, residual, center, tile)
        return (*measured, len(starts)) if measured is not None else None
    starts, residual, owned = starts[moving], residual[moving], owned[moving]
    if len(starts) < 4 or not owned.any():
        return None
    measured = _local_affine(starts, residual, center, tile)
    if measured is not None:
        return (*measured, len(starts))
    # The shared rigid fitter uses at most 120 deterministic point-pair
    # hypotheses; only current correspondences can join its final consensus.
    pose = rigid_motion(starts, starts + residual, error=LAYER_ERROR_MAX)
    if pose is None or len(pose.inliers) < 4:
        return None
    inside = np.asarray(pose.inliers, dtype=int)
    if not owned[inside].any():
        return None
    measured = _local_affine(starts[inside], residual[inside], center, tile)
    return (*measured, len(inside)) if measured is not None else None


def _local_translation(starts: np.ndarray, motion: np.ndarray, center: np.ndarray) -> tuple | None:
    """Pool distinct local matches only while one measured motion describes them."""
    if not len(starts):
        return None
    shift = np.median(motion, axis=0)
    error = float(np.max(np.linalg.norm(motion - shift, axis=1)))
    if error <= .75:
        return shift, len(starts)
    fit = _affine_fit(starts, motion, center)
    # The center must lie within observed local extents; an affine fit cannot
    # supply an extrapolated translation elsewhere on the screen.
    if fit is None or np.any(center < starts.min(axis=0)) or np.any(center > starts.max(axis=0)):
        return None
    return fit[0][2], len(starts)


def pixel_motion_signature(before_rgb: np.ndarray, after_rgb: np.ndarray, *,
                           tile: int = 12, point_limit: int = 192) -> PixelMotionSignature | None:
    """Measure a pair of RGB captures; custody and input assignment belong to the caller.

    Working geometry is at most 320 by 240, with at most 256 correspondences.
    Without a majority translation across distributed current pixel support,
    raw screen motion keeps its unknown reference. At least four verified
    correspondences are required to return either kind of signature.
    """
    if type(tile) is not int or not 6 <= tile <= 48 or type(point_limit) is not int or not 12 <= point_limit <= MAX_POINTS:
        return None
    pictures = _pictures(before_rgb, after_rgb)
    if pictures is None:
        return None
    before, after = pictures
    starts, ends, receipts = _correspondences(before, after, point_limit)
    if len(receipts) < 4:
        return None
    camera = _camera(starts, ends, before.shape[:2])
    reference = "screen_unreferenced" if camera is None else "distributed_translation"
    residual = ends - starts
    if camera is not None:
        residual = residual - (camera.dx, camera.dy)
    height, width = before.shape[:2]
    rows, columns = math.ceil(height / tile), math.ceil(width / tile)
    features, support, point_counts, angular_support, strain, errors = [], [], [], [], [], []
    for row in range(rows):
        for column in range(columns):
            left, top = column * tile, row * tile
            right, bottom = min(width, left + tile), min(height, top + tile)
            own = ((starts[:, 0] >= left) & (starts[:, 0] < right)
                   & (starts[:, 1] >= top) & (starts[:, 1] < bottom))
            center = np.asarray(((left + right) / 2., (top + bottom) / 2.))
            # This neighborhood is fixed before pixels or input labels are
            # read. Shared matches remain the same points across nearby tiles.
            pooled = np.all(abs(starts - center) <= tile, axis=1)
            translation = _local_translation(starts[pooled], residual[pooled], center)
            if translation is None:
                translation = _local_translation(starts[own], residual[own], center)
            displacement, count = translation if translation is not None else ((0., 0.), 0)
            nearby = np.all(abs(starts - center) <= 1.5 * tile, axis=1)
            fitted = _angular_motion_layer(starts[nearby], residual[nearby], center, tile, owned=own[nearby])
            angle, deformation, error, angular_count = fitted if fitted is not None else (0., (0., 0., 0.), None, 0)
            features.append((float(displacement[0]), float(displacement[1]), angle))
            support.append(count)
            point_counts.append(int(own.sum()))
            angular_support.append(angular_count)
            strain.append(deformation)
            errors.append(error)
    return PixelMotionSignature(tuple(before_rgb.shape[:2]), tuple(before.shape[:2]), (rows, columns), tile,
                                tuple(features), tuple(support), tuple(point_counts), tuple(angular_support),
                                tuple(strain), tuple(errors), camera, reference, receipts)


@invariant("perception.pixel_motion_proposals_have_bounded_measured_support", scope="perception",
           owner="core/perception/pixel_motion_before_object_identity.py", observational=False)
def _pixel_motion_bound_invariant() -> tuple:
    blank = np.full((64, 96, 3), 30, dtype=np.uint8)
    assert pixel_motion_signature(blank, blank) is None
    assert pixel_motion_signature(blank, blank, point_limit=MAX_POINTS + 1) is None
    return ()
