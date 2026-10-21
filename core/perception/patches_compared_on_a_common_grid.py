"""Compare two resampled patches in a shared intermediate coordinate frame.

One-sided interpolation changes edge sharpness with raster phase. Both images
are sampled here, with a rigid half-warp each way. This is a bounded fallback
for lost visible marks; local repetition still prevents a new correspondence.
"""
from __future__ import annotations

import numpy as np

from core.perception.points_before_the_picture_is_reduced import _OFFSETS, _samples
from core.perception.points_that_keep_their_motion import SeenPoint, _follow
from core.verify import invariant

RECOVERY_LIMIT = 32
_ALTERNATIVES = np.asarray([(x, y) for y in range(-8, 9) for x in range(-8, 9) if x*x+y*y >= 9], float)


def _empty_fit() -> tuple:
    return np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0, dtype=bool)


def _usable(before: np.ndarray, after: np.ndarray, points: np.ndarray, initial: np.ndarray) -> bool:
    return (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)
        and before.shape == after.shape and before.ndim == 3 and before.shape[2] == 3
        and min(before.shape[:2]) >= 16 and before.shape[0]*before.shape[1] <= 1_000_000
        and before.dtype.kind == after.dtype.kind == 'f'
        and isinstance(points, np.ndarray) and isinstance(initial, np.ndarray)
        and points.shape == initial.shape and points.ndim == 2 and points.shape[1:] == (2,)
        and points.dtype.kind == initial.dtype.kind == 'f' and 1 <= len(points) <= 3*RECOVERY_LIMIT
        and all(np.isfinite(v).all() for v in (before, after, points, initial))
        and min(float(before.min()), float(after.min())) >= 0.
        and max(float(before.max()), float(after.max())) <= 1.
        and max(float(abs(points).max()), float(abs(initial).max())) <= 4096.)


def _warped(points: np.ndarray, shift: np.ndarray, angles: np.ndarray, lattice: np.ndarray, indices: np.ndarray) -> tuple:
    half = angles[indices]/2
    c, s = np.cos(half)[:, None], np.sin(half)[:, None]
    offsets = lattice[indices]-points[indices, None]-shift[indices, None]/2
    x, y = offsets[..., 0], offsets[..., 1]
    old = np.stack((c*x+s*y, -s*x+c*y), axis=-1)
    new = np.stack((c*x-s*y, s*x+c*y), axis=-1)
    return old, new, points[indices, None]+old, points[indices, None]+shift[indices, None]+new


def _inside(places: np.ndarray, shape: tuple) -> np.ndarray:
    return ((places[..., 0] >= 1) & (places[..., 0] <= shape[1]-2)
            & (places[..., 1] >= 1) & (places[..., 1] <= shape[0]-2)).all(axis=1)


def fit_common_grid(before: np.ndarray, after: np.ndarray, points: np.ndarray, initial: np.ndarray) -> tuple:
    """Fit translation and rotation with independent stopping and final RGB error."""
    if not _usable(before, after, points, initial):
        return _empty_fit()
    old_gradients = [np.gradient(before[..., c]) for c in range(3)]
    new_gradients = [np.gradient(after[..., c]) for c in range(3)]
    shift, angles = initial-points, np.zeros(len(points))
    lattice = np.round(points+shift/2)[:, None]+_OFFSETS
    useful, active = np.ones(len(points), dtype=bool), np.arange(len(points))
    for _ in range(12):
        if not len(active):
            break
        old, new, old_places, new_places = _warped(points, shift, angles, lattice, active)
        old_values = np.stack([_samples(before[..., k], old_places) for k in range(3)], axis=-1)
        new_values = np.stack([_samples(after[..., k], new_places) for k in range(3)], axis=-1)
        old_dx = np.stack([_samples(gx, old_places) for _gy, gx in old_gradients], axis=-1)
        old_dy = np.stack([_samples(gy, old_places) for gy, _gx in old_gradients], axis=-1)
        new_dx = np.stack([_samples(gx, new_places) for _gy, gx in new_gradients], axis=-1)
        new_dy = np.stack([_samples(gy, new_places) for gy, _gx in new_gradients], axis=-1)
        c, s = np.cos(angles[active]/2)[:, None, None], np.sin(angles[active]/2)[:, None, None]
        dx = new_dx*(1-c/2)-new_dy*s/2+old_dx*c/2-old_dy*s/2
        dy = new_dx*s/2+new_dy*(1-c/2)+old_dx*s/2+old_dy*c/2
        rotation = .5*(-new[..., 1, None]*new_dx+new[..., 0, None]*new_dy
                        -old[..., 1, None]*old_dx+old[..., 0, None]*old_dy)
        jacobian = np.stack((dx, dy, rotation), axis=-1).reshape(len(active), -1, 3)
        normal = np.einsum('nki,nkj->nij', jacobian, jacobian)
        valid = (np.linalg.eigvalsh(normal)[:, 0] > 1e-6) & _inside(old_places, before.shape) & _inside(new_places, after.shape)
        useful[active] &= valid
        rhs = np.einsum('nki,nk->ni', jacobian, (old_values-new_values).reshape(len(active), -1))
        delta = np.zeros((len(active), 3))
        delta[valid] = np.linalg.solve(normal[valid], rhs[valid, :, None])[..., 0]
        delta[:, :2] = np.clip(delta[:, :2], -2., 2.)
        delta[:, 2] = np.clip(delta[:, 2], -.12, .12)
        shift[active] += delta[:, :2]
        angles[active] += delta[:, 2]
        converged = (np.linalg.norm(delta[:, :2], axis=1) < .015) & (abs(delta[:, 2]) < .002)
        active = active[valid & ~converged]
    _old, _new, old_places, new_places = _warped(points, shift, angles, lattice, np.arange(len(points)))
    old_values = np.stack([_samples(before[..., k], old_places) for k in range(3)], axis=-1)
    new_values = np.stack([_samples(after[..., k], new_places) for k in range(3)], axis=-1)
    error = abs(old_values-new_values).mean(axis=(1, 2))
    useful &= _inside(old_places, before.shape) & _inside(new_places, after.shape)
    useful &= np.isfinite(shift).all(axis=1) & np.isfinite(angles) & np.isfinite(error)
    return points+shift, error, angles, useful


def distinct_patches(before: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Veto a patch when a nearby separate repeat also fits its pixels.

    Three low-cost alternatives in the bounded surrounding grid are refined.
    This veto does not establish global uniqueness or controlled-body identity.
    """
    if not _usable(before, before, points, points) or len(points) > RECOVERY_LIMIT:
        return np.empty(0, dtype=bool)
    templates = np.stack([_samples(before[..., k], points[:, None]+_OFFSETS) for k in range(3)], axis=-1)
    offsets = np.mgrid[-11:12, -11:12].transpose(1, 2, 0)[..., ::-1]
    sampled = np.stack([_samples(before[..., k], points[:, None, None]+offsets) for k in range(3)], axis=-1)
    windows = np.lib.stride_tricks.sliding_window_view(sampled, (7, 7), axis=(1, 2))
    scores = abs(windows-templates.reshape(-1, 7, 7, 3).transpose(0, 3, 1, 2)[:, None, None]).mean(axis=(3, 4, 5))
    scores = scores[:, _ALTERNATIVES[:, 1].astype(int)+8, _ALTERNATIVES[:, 0].astype(int)+8]
    centers = points[:, None]+_ALTERNATIVES
    scores[(centers[..., 0] < 4) | (centers[..., 0] > before.shape[1]-5)
           | (centers[..., 1] < 4) | (centers[..., 1] > before.shape[0]-5)] = np.inf
    ranked = np.argsort(scores, axis=1, kind='stable')[:, :3]
    starts = np.repeat(points, 3, axis=0)
    ends, error, _angle, valid = fit_common_grid(before, before, starts, starts+_ALTERNATIVES[ranked.reshape(-1)])
    repeated = valid & (error < .025) & (np.linalg.norm(ends-starts, axis=1) >= 3.) & (abs(ends-starts) <= 10.).all(axis=1)
    return ~repeated.reshape(-1, 3).any(axis=1)


def _one_sided_error(before: np.ndarray, after: np.ndarray, starts: np.ndarray, ends: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    rotated = np.stack((c*_OFFSETS[:, 0]-s*_OFFSETS[:, 1], s*_OFFSETS[:, 0]+c*_OFFSETS[:, 1]), axis=-1)
    template = np.stack([_samples(before[..., k], starts[:, None]+_OFFSETS) for k in range(3)], axis=-1)
    current = np.stack([_samples(after[..., k], ends[:, None]+rotated) for k in range(3)], axis=-1)
    return abs(template-current).mean(axis=(1, 2))


def recover_source_marks(tracker: object, image: list, colours: np.ndarray, at: float, measured: list) -> list:
    """Keep successful fits unchanged; retry at most 32 currently missing marks."""
    visible = {p.number: p for p in measured}
    missing = [p for p in tracker._native_points if p.number not in visible]
    missing.sort(key=lambda p: (tracker.still_needed.requests.get((p.number, p.born), -1.) <= at,
                                -p.pictures, p.error, p.number))
    missing = missing[:RECOVERY_LIMIT]
    if not missing:
        return measured
    starts = np.asarray([(p.x, p.y) for p in missing])
    initial, forwards = _follow(tracker._image, image, starts)
    ends, error, angle, fitted = fit_common_grid(tracker._colours, colours, starts, initial)
    if not len(ends):
        return measured
    reverse, backwards = _follow(image, tracker._image, ends)
    returned, back_error, back_angle, back_fitted = fit_common_grid(colours, tracker._colours, ends, reverse)
    if not len(returned):
        return measured
    valid = forwards & backwards & fitted & back_fitted & (error < .025) & (back_error < .025)
    valid &= (np.linalg.norm(returned-starts, axis=1) < .5) & (abs(angle+back_angle) < .08)
    discrepancy = np.maximum(_one_sided_error(tracker._colours, colours, starts, ends, angle),
                             _one_sided_error(colours, tracker._colours, ends, returned, back_angle))
    # Retain the original sampling discrepancy in the published uncertainty.
    error = np.maximum.reduce((error, back_error, discrepancy))
    valid &= error <= .055
    candidates = np.flatnonzero(valid)
    if len(candidates):
        valid[candidates] &= distinct_patches(tracker._colours, starts[candidates])
    for index, point in enumerate(missing):
        if valid[index]:
            visible[point.number] = SeenPoint(point.number, float(ends[index, 0]), float(ends[index, 1]), point.x, point.y,
                point.born, tracker._at, at, float(error[index]), point.pictures+1)
    return [visible[p.number] for p in tracker._native_points if p.number in visible]


@invariant('perception.common_grid_recovery_rejects_local_repetition', scope='perception',
           owner='core/perception/patches_compared_on_a_common_grid.py', observational=False)
def _common_grid_recovery_invariant() -> tuple:
    tile = np.random.default_rng(37).uniform(.05, .95, (8, 8, 3))
    repeated = np.tile(tile, (8, 8, 1))
    assert not distinct_patches(repeated, np.array(((24., 24.),))).any()
    flat = np.full((32, 32, 3), .5)
    assert not fit_common_grid(flat, flat, np.array(((16., 16.),)), np.array(((16., 16.),)))[3].any()
    assert not len(fit_common_grid(flat, flat, np.zeros((97, 2)), np.zeros((97, 2)))[0])
    return ()
