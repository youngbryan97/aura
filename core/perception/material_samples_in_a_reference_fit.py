"""Validate redundant material samples before a reference fit supplies a binding."""
from __future__ import annotations

import math

import numpy as np

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance


def material_samples(appearance: ReferenceAppearance) -> tuple | None:
    if (not isinstance(appearance, ReferenceAppearance)
            or not isinstance(appearance.source_points, (tuple, list)) or not 6 <= len(appearance.source_points) <= 256
            or not isinstance(appearance.observed_points, (tuple, list))
            or len(appearance.observed_points) != len(appearance.source_points)):
        return None
    try:
        starts = np.asarray(appearance.source_points, float)
        ends = np.asarray(appearance.observed_points, float)
        linear, offset = np.asarray(appearance.linear, float), np.asarray(appearance.offset, float)
    except (TypeError, ValueError):
        return None
    if (starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1:] != (2,)
            or not 6 <= len(starts) <= 256 or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or linear.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(linear).all()
            or not np.isfinite(offset).all() or np.linalg.det(linear) <= 0
            or len(np.unique(starts, axis=0)) < 6 or type(appearance.error) not in (int, float)
            or not math.isfinite(appearance.error) or appearance.error < 0
            or np.linalg.svd(starts-starts.mean(axis=0), compute_uv=False)[-1] < 1.
            or np.max(np.linalg.norm(starts @ linear.T+offset-ends, axis=1)) > appearance.error+1e-5):
        return None
    return starts, ends, linear, offset
