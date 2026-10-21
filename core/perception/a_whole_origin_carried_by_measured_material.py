"""Date a whole artwork origin's displacement through shared observed point names.

Visual bounds, source coordinates and a response to input remain separate.
This supplies the current material interval; it grants no player, contact or
physical-unit authority.
"""
from __future__ import annotations

import math

import numpy as np

from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.material_displacement_from_named_pixels import displacement_from_named_pixels
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.points_that_keep_their_motion import NAMED_POINT_GAP_S, PointMotion
from core.perception.position_from_the_same_points import PointPose
from core.perception.rigid_motion_from_named_points import RigidMotion, rigid_motion
from core.verify import invariant


def _last_origin(whole: object) -> tuple | None:
    path = whole.origin_path
    if not isinstance(path, tuple) or not path:
        return None
    last = path[-1]
    if (not isinstance(last, tuple) or len(last) != 4 or last[0] != whole.seen
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in last) or last[3] < 0):
        return None
    return last


def whole_material_displacement(previous: object, current: object, marks: MaterialMarks):
    """Only the same source frame and current uninterrupted named material can join."""
    from core.perception.whole_referents_from_source_material import WholeSourceReferent

    if (not isinstance(previous, WholeSourceReferent) or not isinstance(current, WholeSourceReferent)
            or not isinstance(marks, MaterialMarks) or marks.pixels != current.pixels
            or marks.epoch != previous.mark_epoch or marks.epoch != current.mark_epoch
            or not 0 < current.seen-previous.seen <= NAMED_POINT_GAP_S
            or (previous.number, previous.born, previous.kind, previous.source, previous.reference_sha256, previous.character_id)
                != (current.number, current.born, current.kind, current.source, current.reference_sha256, current.character_id)):
        return None
    before, after = _last_origin(previous), _last_origin(current)
    if before is None or after is None:
        return None
    shared = previous.anchors & current.anchors
    points = sorted((p for p in marks.points if (p.number, p.born) in shared and p.before == previous.seen
                     and p.at == current.seen and p.pictures >= 2 and p.error <= .055), key=lambda p: p.number)
    if len(points) < 3:
        return None
    starts = np.asarray([(p.old_x, p.old_y) for p in points])
    ends = np.asarray([(p.x, p.y) for p in points])
    fitted = rigid_motion(starts, ends) or (affine_motion(starts, ends) if len(points) >= 6 else None)
    if fitted is None:
        return None
    old_places = {p.number: (p.old_x, p.old_y) for p in points}
    new_places = {p.number: (p.x, p.y) for p in points}
    old_pose = PointPose(previous.born, previous.seen, before[1:3], old_places, before[1:3],
        reference_at=previous.born, visible_marks=old_places.copy(), reference_error=before[3], pixel_size=marks.pixel_size)
    new_pose = PointPose(current.born, current.seen, before[1:3], old_places.copy(), after[1:3],
        reference_at=previous.born, visible_marks=new_places, reference_error=after[3], pixel_size=marks.pixel_size)
    step = np.asarray(after[1:3])-before[1:3]
    rigid = isinstance(fitted, RigidMotion)
    motion = PointMotion(float(step[0]), float(step[1]), fitted.error, len(points), previous.seen, current.seen,
        tuple(p.number for p in points), tuple(new_places.values()), rigid=fitted if rigid else None,
        pixel_size=marks.pixel_size, affine=None if rigid else fitted)
    return displacement_from_named_pixels(current.number, old_pose, new_pose, motion)


@invariant('perception.whole_motion_cannot_default_to_direct_state', scope='perception',
           owner='core/perception/a_whole_origin_carried_by_measured_material.py', observational=False)
def _whole_motion_origin_invariant() -> tuple:
    from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
    from core.perception.translation_from_visible_support import motion_was_measured, told_directly
    from core.perception.whole_referents_from_source_material import WholeSourceReferent

    item = WholeSourceReferent(1, 1., 2, 1., 10., 10., 8., 8., 'source', '0'*64,
        PixelsConsumed(1., (32, 32, 3), '1'*64), '2'*32, frozenset(((1, 0.), (2, 0.), (3, 0.))))
    assert not told_directly(item) and not motion_was_measured(item, 1.)
    assert item.material_displacement is None and whole_material_displacement(None, item, None) is None
    return ()
