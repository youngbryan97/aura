"""Fit rotation and translation to continuing visible point correspondences."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from core.verify import invariant


@dataclass(frozen=True)
class RigidMotion:
    angle: float
    offset: tuple[float, float]
    inliers: tuple[int, ...]
    error: float

    def apply(self, places: np.ndarray | tuple) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.asarray(places) @ np.array(((c, s), (-s, c))) + self.offset

    def inverse(self, places: np.ndarray | tuple) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (np.asarray(places) - self.offset) @ np.array(((c, -s), (s, c)))


def _fit(starts: np.ndarray, ends: np.ndarray) -> RigidMotion:
    middle_s, middle_e = starts.mean(axis=0), ends.mean(axis=0)
    a, b = starts - middle_s, ends - middle_e
    angle = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a * b)))
    rotation = RigidMotion(angle, (0., 0.), (), 0.)
    offset = middle_e - rotation.apply(middle_s)
    return RigidMotion(angle, tuple(float(v) for v in offset), (), 0.)


def angular_uncertainty(starts: np.ndarray, ends: np.ndarray, fitted: RigidMotion, *, resolution: float = .25) -> float:
    """Estimate angular RMS error from the measured consensus and quarter-pixel resolution.

    This estimate does not bound correlated correspondence errors or drift.
    """
    if (not isinstance(resolution, (int, float)) or isinstance(resolution, bool)
            or not math.isfinite(resolution) or resolution <= 0.):
        return math.inf
    inside = np.asarray(starts)[list(fitted.inliers)]
    observed = np.asarray(ends)[list(fitted.inliers)]
    span = float(np.sqrt(np.mean(np.sum((inside-inside.mean(axis=0))**2, axis=1))))
    residual = fitted.apply(inside)-observed
    error = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return math.atan2(resolution+error, span)


def rigid_motion(starts: np.ndarray, ends: np.ndarray, *, error: float = .75) -> RigidMotion | None:
    """A bounded consensus of at least three noncollinear correspondences.

    Two named points propose a proper rotation. At least 70 percent of all
    correspondences must agree; a noncollinear consensus supplies the fit.
    Scaling, reflection and deformation beyond the error bound are rejected.
    """
    try:
        starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    except (TypeError, ValueError):
        return None
    if (starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1:] != (2,)
            or not 3 <= len(starts) <= 256 or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or not math.isfinite(error) or error <= 0):
        return None
    if np.linalg.svd(starts - starts.mean(axis=0), compute_uv=False)[-1] < 1.:
        return None
    fitted = _fit(starts, ends)
    residual = np.linalg.norm(fitted.apply(starts)-ends, axis=1)
    if residual.max() <= error:
        return RigidMotion(fitted.angle, fitted.offset, tuple(range(len(starts))), float(residual.max()))
    # Spread a fixed hypothesis budget across the correspondence list.
    selected = np.unique(np.linspace(0, len(starts) - 1, min(len(starts), 16)).round().astype(int))
    best = None
    for i, j in itertools.combinations(selected, 2):
        if np.linalg.norm(starts[i] - starts[j]) < 2.:
            continue
        proposed = _fit(starts[[i, j]], ends[[i, j]])
        residual = np.linalg.norm(proposed.apply(starts) - ends, axis=1)
        inside = residual <= error
        rank = int(inside.sum()), -float(np.median(residual[inside])) if inside.any() else -math.inf
        if best is None or rank > best[0]:
            best = rank, inside
    if best is None or best[0][0] < max(3, math.ceil(.7 * len(starts))):
        return None
    inside = best[1]
    if np.linalg.svd(starts[inside] - starts[inside].mean(axis=0), compute_uv=False)[-1] < 1.:
        return None
    fitted = _fit(starts[inside], ends[inside])
    residual = np.linalg.norm(fitted.apply(starts) - ends, axis=1)
    inside &= residual <= error
    if inside.sum() < max(3, math.ceil(.7 * len(starts))):
        return None
    if np.linalg.svd(starts[inside] - starts[inside].mean(axis=0), compute_uv=False)[-1] < 1.:
        return None
    return RigidMotion(fitted.angle, fitted.offset, tuple(int(i) for i in np.flatnonzero(inside)),
                       float(residual[inside].max()))


@invariant("perception.rigid_point_motion_requires_spread_consensus", scope="perception",
           owner="core/perception/rigid_motion_from_named_points.py", observational=False)
def _rigid_motion_invariant() -> tuple:
    starts = np.array(((0., 0.), (12., 0.), (0., 12.), (12., 12.)))
    truth = RigidMotion(.4, (4., -3.), (), 0.)
    fit = rigid_motion(starts, truth.apply(starts))
    assert fit is not None and np.allclose(fit.apply((6., 6.)), truth.apply((6., 6.)))
    assert rigid_motion(starts, starts * 1.8) is None
    assert rigid_motion(starts, starts * (-1., 1.)) is None
    assert rigid_motion(starts[:2], truth.apply(starts[:2])) is None
    return ()
