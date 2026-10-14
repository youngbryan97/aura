"""Read a current angular phase from the same measured material marks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.position_from_the_same_points import PointPose
from core.perception.rigid_motion_from_named_points import angular_uncertainty, rigid_motion
from core.verify import invariant


@dataclass(frozen=True)
class AngularPhase:
    at: float
    reference_at: float
    angle: float
    uncertainty: float
    position: tuple[float, float]
    position_error: float


def current_angular_phase(thing: Any, at: float) -> AngularPhase | None:
    """The phase is image geometry; it has no facing or control authority."""
    pose = getattr(thing, 'point_pose', None)
    if (not isinstance(pose, PointPose) or pose.at != at or getattr(thing, 'seen', None) != at
            or pose.born != getattr(thing, 'born', None) or getattr(thing, 'motion_measured_at', None) != at
            or getattr(thing, 'motion_evidence', None) not in {'points', 'support'} or pose.reference_at is None
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (at, pose.reference_at)) or pose.reference_at > at
            ):
        return None
    return _phase_of_pose(pose)


def _phase_of_pose(pose: PointPose) -> AngularPhase | None:
    reference_at = getattr(pose, 'angular_reference_at', None)
    if (reference_at is None or not isinstance(pose.pixel_size, (int, float)) or isinstance(pose.pixel_size, bool)
            or not math.isfinite(pose.pixel_size) or pose.pixel_size <= 0.
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                      for v in (reference_at, pose.born, pose.at, pose.angular_offset, pose.angular_error))
            or not pose.born <= reference_at <= pose.at or not 0. <= pose.angular_error <= .12
            or not isinstance(pose.anchors, dict) or not isinstance(getattr(pose, 'visible_marks', None), dict)):
        return None
    names = pose.anchors.keys() & pose.visible_marks.keys()
    if (not 3 <= len(names) <= 256 or len(names) < .6 * len(pose.anchors)
            or any(type(n) is not int or n < 0 for n in names)):
        return None
    names = sorted(names)
    try:
        starts = np.asarray([pose.anchors[n] for n in names], dtype=float)
        ends = np.asarray([pose.visible_marks[n] for n in names], dtype=float)
    except (TypeError, ValueError):
        return None
    fitted = rigid_motion(starts, ends)
    if fitted is None:
        return None
    inside = starts[list(fitted.inliers)]
    span = float(np.sqrt(np.mean(np.sum((inside-inside.mean(axis=0))**2, axis=1))))
    if span <= 0.:
        return None
    if (not isinstance(pose.position, tuple) or len(pose.position) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in pose.position)):
        return None
    return AngularPhase(pose.at, reference_at, (fitted.angle+pose.angular_offset+math.pi) % (2.*math.pi)-math.pi,
                        math.hypot(pose.angular_error, angular_uncertainty(starts, ends, fitted,
                            resolution=.25*pose.pixel_size)),
                        pose.position, .25*pose.pixel_size+fitted.error)


@invariant('perception.angular_phase_requires_current_owned_material_marks', scope='perception',
           owner='core/perception/orientation_from_continuing_marks.py', observational=False)
def _angular_phase_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.rigid_motion_from_named_points import RigidMotion

    marks = {i: p for i, p in enumerate(((0., 0.), (12., 0.), (0., 12.), (12., 12.)))}
    fit = RigidMotion(.4, (20., 20.), (), 0.)
    pose = PointPose(0., 1., (6., 6.), marks, tuple(fit.apply((6., 6.))), reference_at=.5,
                     visible_marks={i: tuple(fit.apply(p)) for i, p in marks.items()}, angular_reference_at=.5)
    thing = SimpleNamespace(born=0., seen=1., motion_measured_at=1., motion_evidence='points', point_pose=pose)
    phase = current_angular_phase(thing, 1.)
    assert phase is not None and math.isclose(phase.angle, .4) and phase.uncertainty > 0.
    assert current_angular_phase(thing, 1.1) is None
    pose.visible_marks = {i: (-p[0], p[1]) for i, p in marks.items()}
    assert current_angular_phase(thing, 1.) is None
    return ()
