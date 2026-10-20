"""Estimate local displacement precision without counting shared origin error twice.

The point observer supplies current visible correspondence. A common
reference offset cancels under translation. Rotation or deformation can
turn that offset into displacement error. These are precision estimates,
not bounds on correlated tracking errors or physical object identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.points_that_keep_their_motion import NAMED_POINT_GAP_S, PointMotion
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and math.isfinite(value))


def _point(value: Any) -> bool:
    return (isinstance(value, tuple) and len(value) == 2 and all(_finite(v) for v in value))


@dataclass(frozen=True)
class MaterialDisplacement:
    number: int
    born: float
    reference_at: float
    before: float
    at: float
    start: tuple[float, float]
    end: tuple[float, float]
    error: float
    support: int

    def __post_init__(self) -> None:
        if (type(self.number) is not int or self.number < 0 or type(self.support) is not int
                or not 3 <= self.support <= 256 or not _point(self.start) or not _point(self.end)
                or not all(_finite(v) for v in (self.born, self.reference_at, self.before, self.at, self.error))
                or not 0 <= self.born <= self.reference_at <= self.before < self.at
                or self.at-self.before > NAMED_POINT_GAP_S or self.error < 0):
            raise ValueError('material displacement needs a current owned reference and finite precision')
        for key in ('born', 'reference_at', 'before', 'at', 'error'):
            object.__setattr__(self, key, float(getattr(self, key)))
        for key in ('start', 'end'):
            object.__setattr__(self, key, tuple(float(v) for v in getattr(self, key)))


def _valid_pose(pose: Any) -> bool:
    if (not isinstance(pose, PointPose) or not _point(pose.position)
            or not all(_finite(v) for v in (pose.born, pose.at, pose.reference_at,
                                           pose.reference_error, pose.pixel_size))
            or not 0 <= pose.born <= pose.reference_at <= pose.at
            or pose.reference_error < 0 or pose.pixel_size <= 0
            or not isinstance(pose.anchors, dict) or not 3 <= len(pose.anchors) <= 256
            or not isinstance(pose.visible_marks, dict) or not 3 <= len(pose.visible_marks) <= 256):
        return False
    return all(type(name) is int and name >= 0 and _point(point)
               for marks in (pose.anchors, pose.visible_marks) for name, point in marks.items())


def displacement_from_named_pixels(number: int, previous: PointPose, current: PointPose,
                                   motion: Any) -> MaterialDisplacement | None:
    """Remeasure the local transform of old owned names and transport its uncertainty.

    A caller must already have qualified these marks against current pixels.
    This helper cannot replace missing correspondence or confer identity.
    """
    if (type(number) is not int or number < 0 or not _valid_pose(previous) or not _valid_pose(current)
            or previous.born != current.born or previous.reference_at != current.reference_at
            or not 0 < current.at-previous.at <= NAMED_POINT_GAP_S
            or not isinstance(motion, PointMotion) or getattr(motion, 'before', None) != previous.at
            or getattr(motion, 'at', None) != current.at):
        return None
    rigid = callable(getattr(motion, 'has_rigid_pose', None)) and motion.has_rigid_pose()
    affine = callable(getattr(motion, 'has_affine_pose', None)) and motion.has_affine_pose()
    if not rigid and not affine:
        return None
    names = sorted(previous.anchors.keys() & previous.visible_marks.keys() & current.visible_marks.keys())
    if len(names) < 3:
        return None
    located = dict(zip(motion.points, motion.locations, strict=True))
    if any(name not in located or math.dist(located[name], current.visible_marks[name]) > 1e-6 for name in names):
        return None
    starts = np.asarray([previous.visible_marks[name] for name in names])
    ends = np.asarray([current.visible_marks[name] for name in names])
    fitted = rigid_motion(starts, ends)
    if fitted is None and affine:
        fitted = affine_motion(starts, ends)
    if fitted is None:
        return None
    # The support mask has working-pixel resolution even when the tracker
    # sampled a finer grid. Keep that raster floor in this precision estimate.
    size = max(1., previous.pixel_size, current.pixel_size)
    local = PointPose(previous.born, previous.at, previous.position, {}, previous.position,
                      reference_error=0., pixel_size=size)
    error = _reference_error(local, starts, ends, fitted, size)
    if hasattr(fitted, 'linear'):
        linear = np.asarray(fitted.linear)
    else:
        c, s = math.cos(fitted.angle), math.sin(fitted.angle)
        linear = np.asarray(((c, -s), (s, c)))
    error += float(np.linalg.norm(linear-np.eye(2), ord=2))*previous.reference_error
    error += math.dist(tuple(fitted.apply(previous.position)), current.position)
    if not math.isfinite(error):
        return None
    return MaterialDisplacement(number, previous.born, previous.reference_at, previous.at,
        current.at, previous.position, current.position, error, len(fitted.inliers))


@invariant('perception.material_displacement_separates_shared_reference_error', scope='perception',
           owner='core/perception/material_displacement_from_named_pixels.py', observational=False)
def _material_displacement_precision_invariant() -> tuple:
    marks = dict(enumerate(((0., 0.), (20., 0.), (0., 20.), (20., 20.))))
    previous = PointPose(0., 1., (10., 10.), marks, (10., 10.), reference_at=1.,
                         visible_marks=marks.copy(), reference_error=100.)
    current = PointPose(0., 1.1, (12., 11.), marks, (12., 11.), reference_at=1.,
                        visible_marks={name: (x+2., y+1.) for name, (x, y) in marks.items()})
    starts = np.asarray(list(marks.values()))
    motion = PointMotion(2., 1., 0., 4, 1., 1.1, tuple(marks), tuple(current.visible_marks.values()),
                         rigid=rigid_motion(starts, starts+(2., 1.)))
    observed = displacement_from_named_pixels(1, previous, current, motion)
    assert observed is not None and observed.error < 1. and observed.support == 4
    current.reference_at = 1.1
    assert displacement_from_named_pixels(1, previous, current, motion) is None
    return ()
