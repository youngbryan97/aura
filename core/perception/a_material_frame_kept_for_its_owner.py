"""Keep a frame's pixels and material basis for work that finishes after play advances."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from core.perception.names_for_observed_instances import InstanceQuery, query_for
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed, consumed_pixels
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.position_from_the_same_points import PointPose, last_observed_outline
from core.verify import invariant


@dataclass(frozen=True)
class MaterialThingTaken:
    number: int
    born: float
    seen: float
    kind: int
    x: float
    y: float
    w: float
    h: float
    support: np.ndarray | None = field(repr=False, compare=False)
    point_pose: PointPose | None = field(repr=False, compare=False)

    def box(self) -> tuple[float, float, float, float]:
        return self.x-self.w/2, self.y-self.h/2, self.x+self.w/2, self.y+self.h/2


@dataclass(frozen=True)
class MaterialFrameTaken:
    """A detached observation retains the original owner's instance receipts."""

    at: float
    scale: float
    pixels_consumed: PixelsConsumed
    picture: np.ndarray = field(repr=False, compare=False)
    things: Mapping[int, MaterialThingTaken] = field(repr=False, compare=False)
    queries: Mapping[int, InstanceQuery] = field(repr=False, compare=False)
    material_marks: MaterialMarks | None = field(default=None, repr=False, compare=False)

    def query_for(self, thing: Any, at: float) -> InstanceQuery | None:
        if (type(at) not in (int, float) or not math.isfinite(at) or at != self.at
                or self.things.get(getattr(thing, 'number', None)) is not thing):
            return None
        return self.queries.get(thing.number)


def query_in_observation(thing: Any, observer: Any, at: float) -> InstanceQuery | None:
    return observer.query_for(thing, at) if isinstance(observer, MaterialFrameTaken) else query_for(thing, observer, at)


def keep_material_frame(observer: Any, picture: np.ndarray, at: float) -> MaterialFrameTaken | None:
    """Copy only a consumed capture and bounded current material state.

    The owned copy carries the original instance receipts. Its Python object
    identity cannot become a new observer, and future live geometry still has
    to prove continuity independently. A changed capture refuses the copy.
    """
    if (not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or picture.ndim != 3
            or picture.shape[2] != 3 or picture.size > 24_000_000):
        return None
    owned = picture.copy()
    receipt = consumed_pixels(owned, at)
    if receipt is None or receipt != getattr(observer, 'pixels_consumed', None):
        return None
    scale = getattr(observer, 'scale', None)
    things = getattr(observer, 'things', None)
    if (type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 8
            or not isinstance(things, Mapping) or len(things) > 4096):
        return None
    kept, queries, support_bytes, material_marks = {}, {}, 0, 0
    for thing in things.values():
        query = query_for(thing, observer, at)
        if query is None:
            continue
        bounds = last_observed_outline(thing)
        if bounds is None or thing.number in kept:
            return None
        bounds = tuple(float(v) for v in bounds)
        left, top, right, bottom = bounds
        mask = getattr(thing, 'support', None)
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2:
                return None
            support_bytes += mask.nbytes
            if support_bytes > 16*1024*1024:
                return None
            mask = mask.copy()
            mask.setflags(write=False)
        pose = getattr(thing, 'point_pose', None)
        if isinstance(pose, PointPose):
            if (not isinstance(pose.anchors, dict) or not isinstance(pose.visible_marks, dict)
                    or len(pose.anchors) > 256 or len(pose.visible_marks) > 256):
                return None
            material_marks += len(pose.anchors)+len(pose.visible_marks)
            if material_marks > 8192:
                return None
            pose = copy.deepcopy(pose)
        else:
            pose = None
        kept[thing.number] = MaterialThingTaken(thing.number, float(thing.born), float(thing.seen), thing.kind,
            (left+right)/2, (top+bottom)/2, right-left, bottom-top, mask, pose)
        queries[thing.number] = replace(query, box=bounds, picture_sha256=receipt.sha256)
    if receipt != getattr(observer, 'pixels_consumed', None):
        return None
    owned.setflags(write=False)
    marks = getattr(observer, 'material_marks', None)
    marks = marks if isinstance(marks, MaterialMarks) and marks.pixels == receipt and marks.scale == scale else None
    return MaterialFrameTaken(at, float(scale), receipt, owned, MappingProxyType(kept), MappingProxyType(queries), marks)


@invariant('perception.detached_material_keeps_its_original_observer', scope='perception',
           owner='core/perception/a_material_frame_kept_for_its_owner.py', observational=False)
def _material_frame_owner_invariant() -> tuple:
    from types import SimpleNamespace

    picture = np.zeros((40, 40, 3), np.uint8)
    thing = SimpleNamespace(number=1, born=0., seen=1., kind=0, x=20., y=20., w=20., h=20.,
                            support=np.ones((20, 20), bool), point_pose=None, box=lambda: (10., 10., 30., 30.))
    observer = SimpleNamespace(things={1: thing}, scale=1., pixels_consumed=consumed_pixels(picture, 1.))
    frame = keep_material_frame(observer, picture, 1.)
    assert frame is not None and not frame.picture.flags.writeable
    assert query_in_observation(frame.things[1], frame, 1.).key == query_for(thing, observer, 1.).key
    assert query_in_observation(thing, frame, 1.) is None
    thing.support[:] = False
    assert frame.things[1].support.all()
    picture[0, 0] = 1
    assert keep_material_frame(observer, picture, 1.) is None
    return ()
