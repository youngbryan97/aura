"""Keep a measured origin where two independent observations overlap.

Offline 2026-10-13 a point path ended while the whole patch was still seen
translating. Its coordinate reference can cross that interval through the
observed displacement of matching pixels and visible exterior edges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.boundary_support_for_translation import visible_boundary
from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.position_from_the_same_points import last_observed_outline, position_from_points
from core.perception.translation_from_visible_support import (
    PIXEL_UNCERTAINTY,
    matching_support,
    translation_status,
)
from core.verify import invariant


@dataclass(frozen=True)
class VisibleReference:
    born: float
    at: float
    position: tuple[float, float]
    box: tuple[float, float]
    reference_at: float | None = None


def still_material_observation(thing: Any, picture: np.ndarray, motion: Any, at: float) -> dict | None:
    """Propose a still-visible patch only with current agreeing named motion.

    The ordinary point and reference checks still decide whether its old
    origin can continue. A matching box alone supplies no point path.
    """
    if (getattr(thing, 'motion_position', None) is None or not isinstance(motion, PointMotion)
            or isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at)
            or motion.before != thing.seen or motion.at != at or motion.support < 3
            or not 0 < at-thing.seen <= .35
            or not all(math.isfinite(v) for v in (motion.dx, motion.dy))
            or math.hypot(motion.dx, motion.dy) > PIXEL_UNCERTAINTY
            or not isinstance(picture, np.ndarray) or picture.dtype != np.uint8
            or picture.ndim != 3 or picture.shape[2] < 3):
        return None
    outline = last_observed_outline(thing)
    if outline is None:
        return None
    left, top, right, bottom = (int(round(v)) for v in outline)
    height, width = picture.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    patch = picture[top:bottom, left:right]
    support = getattr(thing, 'support', None)
    if (not isinstance(support, np.ndarray) or support.dtype != bool
            or support.shape != patch.shape[:2] or patch.shape != thing.patch.shape):
        return None
    blob = dict(x=(left+right)/2, y=(top+bottom)/2, w=thing.w, h=thing.h,
                look=thing.look, colour=thing.colour, patch=patch.copy(), support=support.copy(),
                boundary=visible_boundary(picture, left, top, right, bottom, support))
    return blob if motion.inside(blob) and matching_support(thing, blob) else None


def point_receipt_for_candidate(thing: Any, blob: dict, motion: Any, at: float) -> Any:
    """Off-patch marks yield only to independently observed, agreeing patch motion.

    A corner can leave the foreground mask through raster sampling. Missing
    ownership cannot itself justify association. Matching pixels and visible
    exterior edges must independently support the same displacement.
    """
    if (motion is None or motion.before != thing.seen or motion.at != at or motion.inside(blob)
            or not 0 < at - thing.seen <= .35):
        return motion
    previous = thing.path[-1][1:] if thing.path else (thing.x, thing.y)
    shift = blob['x'] - previous[0], blob['y'] - previous[1]
    if math.dist(shift, (motion.dx, motion.dy)) <= PIXEL_UNCERTAINTY and matching_support(thing, blob):
        return None
    return motion


def position_from_visible_support(thing: Any, blob: dict, motion: Any, at: float) -> tuple:
    """Return a measured path and its evidence without changing coordinate origins."""
    previous = getattr(thing, 'visible_reference', None)
    had_path = getattr(thing, 'motion_position', None) is not None
    status = translation_status(thing, blob, at, motion)
    path = position_from_points(thing, blob, motion, at)
    if path is not None:
        status = 'points'
        reference_at = thing.point_pose.reference_at
    elif had_path and _current_reference(previous, thing, at) and status == 'measured' and matching_support(thing, blob):
        position = tuple(p + blob[k] - b for p, b, k in zip(previous.position, previous.box, ('x', 'y'), strict=True))
        path, status = (previous.position, position), 'support'
        reference_at = previous.reference_at
        seed = getattr(thing, 'point_pose', None)
        if seed is not None and seed.born == thing.born and seed.at == at:
            # Current owned marks start in the independently continued origin.
            seed.origin = seed.position = position
            seed.reference_at = reference_at
    thing.visible_reference = (VisibleReference(thing.born, at, path[1], (blob['x'], blob['y']), reference_at)
                               if path is not None else None)
    return path, status


def _current_reference(reference: VisibleReference | None, thing: Any, at: float) -> bool:
    """A reference must belong to this birth and the preceding measured observation."""
    if reference is None or reference.born != thing.born or reference.at != thing.seen:
        return False
    if getattr(thing, 'motion_measured_at', None) != thing.seen or not 0 < at - thing.seen <= .35:
        return False
    box = tuple(thing.path[-1][1:]) if thing.path else (thing.x, thing.y)
    return (reference.position == thing.motion_position and reference.box == box
            and all(math.isfinite(v) for v in (reference.born, reference.at, *reference.position, *reference.box, at)))


@invariant('perception.a_reference_bridge_needs_independent_visible_support', scope='perception',
           owner='core/perception/a_reference_kept_by_visible_support.py', observational=False)
def _reference_bridge_invariant() -> tuple:
    from types import SimpleNamespace

    import numpy as np

    from core.perception.boundary_support_for_translation import visible_boundary
    from core.perception.points_that_keep_their_motion import PointMotion

    image = np.full((80, 100, 3), 30, np.uint8)
    image[30:50, 30:50] = 200
    mask, patch = np.ones((20, 20), bool), image[30:50, 30:50].copy()
    edge = visible_boundary(image, 30, 30, 50, 50, mask)
    reference = VisibleReference(0., 1., (39., 40.), (40., 40.))
    thing = SimpleNamespace(born=0., seen=1., x=40., y=40., w=20., h=20., point_pose=None,
                            motion_position=(39., 40.), motion_measured_at=1., visible_reference=reference,
                            path=[(1., 40., 40.)], patch=patch, support=mask, boundary=edge)
    blob = dict(x=44., y=40., w=20., h=20., patch=patch, support=mask, boundary=edge)
    assert position_from_visible_support(thing, blob, None, 1.1) == (((39., 40.), (43., 40.)), 'support')
    marks = PointMotion(4., 0., .01, 4, 1., 1.1, (1, 2, 3, 4),
                        ((32., 34.), (42., 34.), (32., 46.), (42., 46.)))
    assert not marks.inside(blob) and point_receipt_for_candidate(thing, blob, marks, 1.1) is None
    thing.visible_reference, blob['boundary'] = reference, None
    assert position_from_visible_support(thing, blob, None, 1.1)[0] is None
    assert point_receipt_for_candidate(thing, blob, marks, 1.1) is marks
    return ()


@invariant('perception.resting_material_keeps_a_measured_origin', scope='perception',
           owner='core/perception/a_reference_kept_by_visible_support.py', observational=False)
def _resting_material_invariant() -> tuple:
    from types import SimpleNamespace
    from dataclasses import replace

    picture = np.full((80, 100, 3), 30, np.uint8)
    picture[30:50, 30:50] = 200
    support = np.ones((20, 20), bool)
    thing = SimpleNamespace(born=0., seen=1., x=40., y=40., w=20., h=20.,
        motion_position=(39., 40.), path=[(1., 40., 40.)], patch=picture[30:50, 30:50].copy(),
        support=support, look=np.zeros(64), colour=(200, 200, 200),
        boundary=visible_boundary(picture, 30, 30, 50, 50, support))
    motion = PointMotion(0., 0., .01, 4, 1., 1.1, (1, 2, 3, 4),
                         ((34., 34.), (46., 34.), (34., 46.), (46., 46.)))
    assert still_material_observation(thing, picture, motion, 1.1) is not None
    assert still_material_observation(thing, picture, None, 1.1) is None
    assert still_material_observation(thing, picture, replace(motion, dx=8.), 1.1) is None
    assert still_material_observation(thing, np.zeros_like(picture), motion, 1.1) is None
    return ()
