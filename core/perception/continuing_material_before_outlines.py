"""Continuing owned marks can retain an object while its outline joins or splits.

An outline's centre and aspect can change without its material changing.
The previous measured pose and the current point receipt must agree on a
rigid, spread-out set of marks. Both sides of the association must be unique.
This preserves observation identity; it supplies no control attribution.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from core.perception.position_from_the_same_points import PointPose
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


def material_matches(things: dict[int, Any], blobs: list[dict], motions: dict, at: float) -> dict[int, int]:
    """Return only distinct matches supported by the preceding owned material pose."""
    pairs = []
    for number, thing in things.items():
        marks = _continuing_marks(thing, motions.get(number), at)
        if marks is None:
            continue
        previous, current = marks
        for index, blob in enumerate(blobs):
            names = [name for name, place in current.items() if _owns(blob, place)]
            if len(names) < max(3, math.ceil(.6*len(previous)), math.ceil(.6*len(current))):
                continue
            starts = np.asarray([previous[name] for name in names])
            ends = np.asarray([current[name] for name in names])
            if rigid_motion(starts, ends) is not None:
                pairs.append((number, index))
    by_thing, by_blob = Counter(n for n, _i in pairs), Counter(i for _n, i in pairs)
    return {n: i for n, i in pairs if by_thing[n] == by_blob[i] == 1}


def _continuing_marks(thing: Any, motion: Any, at: float) -> tuple[dict, dict] | None:
    pose = getattr(thing, 'point_pose', None)
    if (not isinstance(pose, PointPose) or pose.at != thing.seen or pose.born != thing.born
            or motion is None or motion.before != pose.at or motion.at != at
            or not all(math.isfinite(v) for v in (at, thing.seen, thing.born, motion.dx, motion.dy, motion.error))
            or motion.error < 0.
            or thing.born > thing.seen or not 0. < at-thing.seen <= .35
            or (motion.rigid is not None and not motion.has_rigid_pose())):
        return None
    names, locations = tuple(motion.points), tuple(motion.locations)
    if (not 3 <= len(names) <= 256 or motion.support != len(names) or len(names) != len(locations)
            or len(set(names)) != len(names) or any(type(name) is not int for name in names)
            or not isinstance(pose.visible_marks, dict) or not 3 <= len(pose.visible_marks) <= 256):
        return None
    previous = {name: place for name, place in pose.visible_marks.items()
                if type(name) is int and _owns_thing(thing, place)}
    current = {name: place for name, place in zip(names, locations, strict=True) if name in previous}
    if (len(previous) != len(pose.visible_marks)
            or len(current) < max(3, math.ceil(.6*len(previous)), math.ceil(.6*len(names)))):
        return None
    return previous, current


def _owns_thing(thing: Any, place: tuple) -> bool:
    return _owns(dict(x=thing.x, y=thing.y, w=thing.w, h=thing.h, support=getattr(thing, 'support', None)), place)


def _owns(blob: dict, place: tuple) -> bool:
    mask = blob.get('support')
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or len(place) != 2
            or not all(math.isfinite(v) for v in (*place, blob['x'], blob['y'], blob['w'], blob['h']))
            or min(blob['w'], blob['h']) <= 0 or mask.shape != (round(blob['h']), round(blob['w']))):
        return False
    col, row = (int(round(place[axis]-blob[key]+blob[size]/2.))
                for axis, key, size in ((0, 'x', 'w'), (1, 'y', 'h')))
    return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col])


@invariant('perception.material_identity_needs_unique_current_geometry', scope='perception',
           owner='core/perception/continuing_material_before_outlines.py', observational=False)
def _material_identity_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion

    marks = {n: p for n, p in enumerate(((16., 28.), (24., 28.), (16., 34.), (24., 34.)))}
    pose = PointPose(0., 1., (10., 20.), {n: (x-10., y) for n, (x, y) in marks.items()},
                     (20., 20.), reference_at=.8, visible_marks=marks)
    body = SimpleNamespace(x=20., y=20., w=12., h=36., born=0., seen=1., point_pose=pose,
                           support=np.ones((36, 12), bool), motion_measured_at=1.,
                           motion_position=(20., 20.), motion_evidence='points')
    blob = dict(x=23., y=31., w=12., h=10., support=np.ones((10, 12), bool))
    motion = PointMotion(3., 0., .01, 4, 1., 1.1, tuple(marks), tuple((x+3., y) for x, y in marks.values()))
    assert material_matches({1: body}, [blob], {1: motion}, 1.1) == {1: 0}
    assert material_matches({1: body}, [blob, dict(blob)], {1: motion}, 1.1) == {}
    assert material_matches({1: body, 2: body}, [blob], {1: motion, 2: motion}, 1.1) == {}
    assert material_matches({1: body}, [blob], {1: motion}, 1.2) == {}
    return ()
