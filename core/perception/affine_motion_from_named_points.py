"""A redundant affine observation for material that changes shape in the image.

Rigid pose remains a separate measurement. This map can explain limited
scale and shear without calling that change a body turn. It does not create
correspondences, object identity or control authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.verify import invariant


@dataclass(frozen=True)
class AffineMotion:
    linear: tuple[tuple[float, float], tuple[float, float]]
    offset: tuple[float, float]
    inliers: tuple[int, ...]
    error: float
    source_center: tuple[float, float]
    information_inverse: tuple[tuple[float, ...], ...]
    residual_variance: float

    def valid(self) -> bool:
        try:
            matrix = np.asarray(self.linear, dtype=float)
            information = np.asarray(self.information_inverse, dtype=float)
            scales = np.linalg.svd(matrix, compute_uv=False)
            return bool(matrix.shape == (2, 2) and information.shape == (3, 3)
                and np.isfinite(matrix).all() and np.isfinite(information).all()
                and len(self.offset) == len(self.source_center) == 2
                and all(math.isfinite(v) for v in (*self.offset, *self.source_center, self.error, self.residual_variance))
                and 0 <= self.error <= .75 and self.residual_variance >= 0
                and len(self.inliers) >= 6 and len(set(self.inliers)) == len(self.inliers)
                and np.linalg.det(matrix) > 0 and scales.min() >= .5 and scales.max() <= 2.
                and np.linalg.eigvalsh(information).min() > 0)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return False

    def apply(self, places) -> np.ndarray:
        return np.asarray(places) @ np.asarray(self.linear).T + self.offset

    def inverse(self, places) -> np.ndarray:
        return (np.asarray(places)-self.offset) @ np.linalg.inv(np.asarray(self.linear)).T

    def target_error(self, target, *, resolution: float) -> float:
        if not math.isfinite(resolution) or resolution <= 0:
            return math.inf
        try:
            target = np.asarray(target, dtype=float)
        except (TypeError, ValueError):
            return math.inf
        if target.shape != (2,) or not np.isfinite(target).all():
            return math.inf
        delta = target-self.source_center
        location = np.array((delta[0], delta[1], 1.))
        leverage = float(location @ np.asarray(self.information_inverse) @ location)
        variance = max(self.residual_variance, resolution**2)
        if not math.isfinite(leverage) or not math.isfinite(variance) or leverage <= 0 or variance <= 0:
            return math.inf
        return math.sqrt(2.*variance*leverage)


def affine_motion(starts, ends, *, error: float = .75) -> AffineMotion | None:
    """Fit redundant named pairs with bounded strain and leave-one-out prediction.

    At least six correspondences and 70 percent of the supplied pairs must
    remain. A least-trimmed fit removes only the worst residual at each step,
    with a fixed iteration budget. Reflections, collapse, collinearity and
    predictions that depend on one mark remain unknown.
    """
    try:
        starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    except (TypeError, ValueError):
        return None
    if (starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1:] != (2,)
            or not 6 <= len(starts) <= 256 or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or not math.isfinite(error) or error <= 0):
        return None
    indices = np.arange(len(starts))
    minimum = max(6, math.ceil(.7*len(starts)))
    for _ in range(8):
        source, observed = starts[indices], ends[indices]
        center = source.mean(axis=0)
        centered = source-center
        if np.linalg.svd(centered, compute_uv=False)[-1] < 1.:
            return None
        design = np.column_stack((centered, np.ones(len(source))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        if rank != 3:
            return None
        residual = np.linalg.norm(design@coefficients-observed, axis=1)
        if residual.max() <= error:
            return _qualified(source, observed, indices, center, design, coefficients, residual, error)
        if len(indices) <= minimum:
            return None
        indices = np.delete(indices, int(np.argmax(residual)))
    return None


def _qualified(source, observed, indices, center, design, coefficients, residual, error):
    linear = coefficients[:2].T
    scales = np.linalg.svd(linear, compute_uv=False)
    if np.linalg.det(linear) <= 0 or scales.min() < .5 or scales.max() > 2.:
        return None
    information = np.linalg.inv(design.T@design)
    leverage = np.einsum('ij,jk,ik->i', design, information, design)
    if np.any(1.-leverage < 1e-6) or np.max(residual/(1.-leverage)) > 2.*error:
        return None
    # An inverse derived independently from the same admitted pairs checks
    # that noisy compression has not made the reverse observation unstable.
    reverse_design = np.column_stack((observed-observed.mean(axis=0), np.ones(len(observed))))
    reverse = np.linalg.lstsq(reverse_design, source, rcond=None)[0]
    forward = design@coefficients
    round_trip = np.column_stack((forward-observed.mean(axis=0), np.ones(len(forward))))@reverse
    if np.linalg.norm(round_trip-source, axis=1).max() > .5:
        return None
    offset = coefficients[2]-linear@center
    variance = float(np.sum(residual**2)/max(1, 2*len(source)-6))
    return AffineMotion(tuple(tuple(float(v) for v in row) for row in linear), tuple(float(v) for v in offset),
        tuple(int(i) for i in indices), float(residual.max()), tuple(float(v) for v in center),
        tuple(tuple(float(v) for v in row) for row in information), variance)


@invariant('perception.affine_material_requires_redundant_geometry', scope='perception',
           owner='core/perception/affine_motion_from_named_points.py', observational=False)
def _affine_material_invariant() -> tuple:
    starts = np.array([(x, y) for x in (0., 6., 12.) for y in (0., 6., 12.)])
    ends = starts @ np.array(((1., .05), (0., .8))).T+(3., 2.)
    fit = affine_motion(starts, ends)
    assert fit is not None and np.allclose(fit.apply(starts), ends)
    assert np.allclose(fit.inverse(ends), starts)
    assert fit.target_error((6., 6.), resolution=.25) < fit.target_error((500., 500.), resolution=.25)
    assert affine_motion(starts[:5], ends[:5]) is None
    assert affine_motion(starts, starts*(-1., 1.)) is None
    return ()
