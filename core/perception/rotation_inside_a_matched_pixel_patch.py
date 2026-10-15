"""Retain local patch orientation without claiming a rigid object's pose.

RGB residuals and the translation/rotation normal matrix measure fit
uncertainty. A patch can turn around its centre while that centre stays put.
"""
from __future__ import annotations

import numpy as np

from core.perception.points_before_the_picture_is_reduced import _OFFSETS, _samples
from core.verify import invariant


def patch_angle_uncertainty(before: np.ndarray, after: np.ndarray, starts: np.ndarray,
                            ends: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Fit uncertainty from the current RGB patch, including translation nuisance."""
    if not isinstance(starts, np.ndarray) or starts.ndim != 2 or starts.shape[1] != 2 or len(starts) > 256:
        return np.empty(0)
    unknown = np.full(len(starts), np.inf)
    if (not len(starts) or not isinstance(ends, np.ndarray) or ends.shape != starts.shape
            or not isinstance(angles, np.ndarray) or angles.shape != (len(starts),)
            or starts.dtype.kind not in 'fiu' or ends.dtype.kind not in 'fiu' or angles.dtype.kind not in 'fiu'
            or not isinstance(before, np.ndarray) or not isinstance(after, np.ndarray)
            or before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3
            or min(before.shape[:2]) < 8 or before.shape[0] * before.shape[1] > 320 * 240
            or before.dtype.kind not in 'fiu' or after.dtype.kind not in 'fiu'
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any((before < 0) | (before > 1)) or np.any((after < 0) | (after > 1))):
        return unknown
    extent = np.asarray((before.shape[1], before.shape[0]))
    rows = np.flatnonzero(np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1) & np.isfinite(angles)
                         & np.all((starts >= 4) & (starts < extent - 4), axis=1)
                         & np.all((ends >= 4) & (ends < extent - 4), axis=1) & (abs(angles) <= .5))
    if not len(rows):
        return unknown
    starts, ends, angles = starts[rows], ends[rows], angles[rows]
    gradients = [np.gradient(after[..., c]) for c in range(3)]
    cosine, sine = np.cos(angles), np.sin(angles)
    rotated = np.stack((cosine[:, None] * _OFFSETS[:, 0] - sine[:, None] * _OFFSETS[:, 1],
                        sine[:, None] * _OFFSETS[:, 0] + cosine[:, None] * _OFFSETS[:, 1]), axis=-1)
    positions = ends[:, None] + rotated
    template = np.stack([_samples(before[..., c], starts[:, None] + _OFFSETS) for c in range(3)], axis=-1)
    measured = np.stack([_samples(after[..., c], positions) for c in range(3)], axis=-1)
    dx = np.stack([_samples(gx, positions) for _gy, gx in gradients], axis=-1)
    dy = np.stack([_samples(gy, positions) for gy, _gx in gradients], axis=-1)
    angular = -rotated[..., 1, None] * dx + rotated[..., 0, None] * dy
    design = np.stack((dx, dy, angular), axis=-1).reshape(len(starts), -1, 3)
    normal = np.einsum('nki,nkj->nij', design, design)
    eigenvalues = np.linalg.eigvalsh(normal)
    finite = np.isfinite(normal).all(axis=(1, 2)) & (eigenvalues[:, 0] > 1e-6)
    finite &= eigenvalues[:, -1] < 1e6 * eigenvalues[:, 0]
    if finite.any():
        residual = (template - measured).reshape(len(starts), -1)
        variance = np.sum(residual[finite] ** 2, axis=1) / (residual.shape[1] - 3)
        covariance = np.linalg.inv(normal[finite])
        unknown[rows[finite]] = np.maximum(.01, np.sqrt(np.maximum(0., variance * covariance[:, 2, 2])))
    return unknown


@invariant('perception.patch_rotation_uncertainty_needs_rgb_gradient_rank', scope='perception',
           owner='core/perception/rotation_inside_a_matched_pixel_patch.py', observational=False)
def _patch_rotation_uncertainty_invariant() -> tuple:
    flat = np.full((24, 32, 3), .5, dtype=np.float32)
    points = np.asarray(((12., 12.),))
    assert np.isinf(patch_angle_uncertainty(flat, flat, points, points, np.zeros(1))).all()
    return ()
