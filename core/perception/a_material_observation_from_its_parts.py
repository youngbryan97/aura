"""Reassemble current visible pieces only through their continuing named material."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from core.perception.boundary_support_for_translation import visible_boundary
from core.perception.continuing_material_before_outlines import (
    _continuing_marks,
    _owns,
    material_matches,
)
from core.perception.position_from_the_same_points import _reference_error
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


def material_observations_from_parts(picture: np.ndarray, blobs: list[dict], things: dict,
                                    motions: dict, at: float, *, outline_cost: Any) -> list[dict]:
    """Join unclaimed pieces through one unique, current rigid material witness.

    Each piece must own two continuing marks. Their union must own most of
    the current material receipt and locate its retained reference. A valid
    outline match of another object, or another material owner claiming a
    piece, refuses the union. The observed mask keeps gaps between pieces.
    """
    if picture.ndim != 3 or picture.shape[2] != 3 or not 2 <= len(blobs) <= 4096:
        return blobs
    established = material_matches(things, blobs, motions, at)
    proposals = []
    for number, thing in things.items():
        if len(proposals) >= 16:
            break
        if number in established:
            continue
        marks = _continuing_marks(thing, motions.get(number), at)
        if marks is None:
            continue
        previous, current = marks
        parts = [(index, {name for name, place in current.items() if _owns(blob, place)})
                 for index, blob in enumerate(blobs)]
        parts = [(index, names) for index, names in parts if len(names) >= 2]
        if not 2 <= len(parts) <= 8:
            continue
        names = sorted(set().union(*(names for _index, names in parts)))
        if len(names) < max(3, math.ceil(.6*len(current))):
            continue
        starts, ends = np.asarray([previous[n] for n in names]), np.asarray([current[n] for n in names])
        fitted = rigid_motion(starts, ends)
        pose = thing.point_pose
        if (fitted is None or _reference_error(pose, starts, ends, fitted,
                pose.pixel_size, target=pose.position) > 1.):
            continue
        indices = tuple(index for index, _names in parts)
        if any(outline_cost(other, blobs[index], at) < math.inf
               for other_number, other in things.items() if other_number != number for index in indices):
            continue
        compound = _visible_union(picture, [blobs[i] for i in indices], thing.size)
        if compound is not None:
            proposals.append((indices, compound))
    uses = Counter(index for indices, _blob in proposals for index in indices)
    accepted = [(indices, blob) for indices, blob in proposals if all(uses[i] == 1 for i in indices)]
    removed = {index for indices, _blob in accepted for index in indices}
    return [blob for index, blob in enumerate(blobs) if index not in removed] + [blob for _, blob in accepted]


def _visible_union(picture: np.ndarray, parts: list[dict], previous_area: float) -> dict | None:
    bounds = [(round(b['x']-b['w']/2), round(b['y']-b['h']/2),
               round(b['x']+b['w']/2), round(b['y']+b['h']/2)) for b in parts]
    left, top = min(b[0] for b in bounds), min(b[1] for b in bounds)
    right, bottom = max(b[2] for b in bounds), max(b[3] for b in bounds)
    area = (right-left)*(bottom-top)
    if (not 0 <= left < right <= picture.shape[1] or not 0 <= top < bottom <= picture.shape[0]
            or area > min(4.*previous_area, .2*picture.shape[0]*picture.shape[1])):
        return None
    mask = np.zeros((bottom-top, right-left), bool)
    for blob, (x0, y0, x1, y1) in zip(parts, bounds, strict=True):
        if blob['support'].shape != (y1-y0, x1-x0):
            return None
        mask[y0-top:y1-top, x0-left:x1-left] |= blob['support']
    if mask.sum() < 3:
        return None
    from core.perception.what_moves_in_the_picture import _look_of

    patch = picture[top:bottom, left:right].copy()
    pixels = patch[mask]
    return dict(x=(left+right)/2, y=(top+bottom)/2, w=float(right-left), h=float(bottom-top),
                patch=patch, support=mask, look=_look_of(pixels),
                colour=tuple(int(c) for c in np.median(pixels, axis=0)),
                boundary=visible_boundary(picture, left, top, right, bottom, mask))


@invariant('perception.material_parts_need_one_current_owner', scope='perception',
           owner='core/perception/a_material_observation_from_its_parts.py', observational=False)
def _material_parts_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion
    from core.perception.position_from_the_same_points import PointPose

    marks = dict(enumerate(((16., 28.), (24., 28.), (16., 34.), (24., 34.))))
    pose = PointPose(0., 1., (20., 20.), marks.copy(), (20., 20.), visible_marks=marks.copy())
    body = SimpleNamespace(x=20., y=20., w=12., h=36., born=0., seen=1., size=432,
                           point_pose=pose, support=np.ones((36, 12), bool))
    picture = np.full((80, 100, 3), 170, np.uint8)
    blobs = [dict(x=23., y=y, w=12., h=3., support=np.ones((3, 12), bool)) for y in (28.5, 34.5)]
    motion = PointMotion(3., 0., .01, 4, 1., 1.1, tuple(marks), tuple((x+3., y) for x, y in marks.values()))
    def unclaimed(*_args):
        return math.inf
    result = material_observations_from_parts(picture, blobs, {1: body}, {1: motion}, 1.1, outline_cost=unclaimed)
    assert len(result) == 1 and result[0]['support'].sum() == 72 and not result[0]['support'][3:6].any()
    assert len(material_observations_from_parts(picture, blobs, {1: body, 2: body},
        {1: motion, 2: motion}, 1.1, outline_cost=unclaimed)) == 2
    return ()
