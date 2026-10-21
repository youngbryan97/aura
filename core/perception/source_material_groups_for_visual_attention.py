"""Keep current source-supported material available to visual inspection.

The box is a proposed inspection window around current named points. Their
measured span remains separate. Neither supplies a contact surface or object
outline. Full-reference pixel reconciliation remains independent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from core.perception.a_source_point_on_visible_marks import SourcePointOnVisibleMarks, VisibleSourcePoint
from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed, consumed_pixels
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.perception.whole_referents_from_source_material import _material_parts
from core.verify import invariant


@dataclass(frozen=True)
class SourceMaterialGroup:
    number: int
    born: float
    kind: int
    seen: float
    x: float
    y: float
    w: float
    h: float
    source: str
    character_id: int
    pixels: PixelsConsumed
    mark_epoch: str
    anchors: frozenset[tuple[int, float]]
    material_bounds: tuple[float, float, float, float]
    observed_parts: tuple[tuple[int, float, int], ...] = ()
    moved: bool = False

    def __post_init__(self) -> None:
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in
                (self.born, self.seen, self.x, self.y, self.w, self.h))
                or not 0 <= self.born <= self.seen or min(self.w, self.h) <= 0
                or any(type(v) is not int or v < 0 for v in (self.number, self.kind, self.character_id))
                or not isinstance(self.pixels, PixelsConsumed) or self.pixels.at != self.seen
                or not isinstance(self.source, str) or not self.source
                or not isinstance(self.anchors, frozenset) or not 3 <= len(self.anchors) <= 128
                or not isinstance(self.material_bounds, tuple) or len(self.material_bounds) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.material_bounds)
                or self.material_bounds[2] <= self.material_bounds[0] or self.material_bounds[3] <= self.material_bounds[1]
                or not isinstance(self.observed_parts, tuple) or len(self.observed_parts) > 32):
            raise ValueError('visual attention needs current owned material without an inferred outline')

    @property
    def geometry_qualified(self) -> bool:
        return False

    @property
    def role(self) -> None:
        return None

    @property
    def contact_verified(self) -> None:
        return None

    def box(self) -> tuple[float, float, float, float]:
        return self.x-self.w/2, self.y-self.h/2, self.x+self.w/2, self.y+self.h/2


def _current_support(observer: Any, reference: Any, binding: Any, point: Any, at: float) -> tuple:
    marks = observer.material_marks
    if (not isinstance(reference, RegisteredVisualReference) or not isinstance(binding, SourcePointOnVisibleMarks)
            or not isinstance(point, VisibleSourcePoint) or reference.source != binding.source
            or reference.coordinates.reference_sha256 != binding.reference_sha256
            or _digest(reference.pixels) != binding.reference_sha256
            or binding.inspect(observer, at, source=reference.source).current != point):
        return ()
    owners = {(a[0], a[1]) for a in binding.anchors}
    support = tuple(p for p in marks.points if (p.number, p.born) in owners and p.error <= .055)
    if len(support) < 3:
        return ()
    xy = np.asarray([(p.x, p.y) for p in support])
    if np.linalg.matrix_rank(xy-xy.mean(axis=0), tol=.05) < 2:
        return ()
    return support


def _inspection_window(point: VisibleSourcePoint, support: tuple, marks: MaterialMarks) -> tuple:
    measured = (min(p.x for p in support), min(p.y for p in support),
                max(p.x for p in support), max(p.y for p in support))
    region = point.sampling_region
    if (not isinstance(region, tuple) or len(region) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in region)):
        return measured, measured
    # A proposed source window helps ask about the whole visible appearance.
    # It does not become the measured extent merely because it can be drawn.
    height, width = marks.grid.working
    window = (max(0., min(region[0], measured[0])), max(0., min(region[1], measured[1])),
              min(float(width), max(region[2], measured[2])), min(float(height), max(region[3], measured[3])))
    return window, measured


def _material_questions(candidates: dict) -> tuple[tuple, tuple]:
    """Combine directly shared current material; an overlap chain is insufficient."""
    pending, result, refused = set(candidates), [], []
    while pending:
        seed = min(pending, key=lambda key: (key[0], key[1], sorted(key[2])))
        component, todo = {seed}, [seed]
        pending.remove(seed)
        while todo:
            key = todo.pop()
            neighbours = {other for other in pending if other[:2] == key[:2] and len(other[2] & key[2]) >= 3}
            component.update(neighbours)
            pending.difference_update(neighbours)
            todo.extend(neighbours)
        owners = tuple(sorted(component, key=lambda key: sorted(key[2])))
        if any(len(a[2] & b[2]) < 3 for a in owners for b in owners):
            refused.append({'source': seed[0], 'character_id': seed[1], 'reason': 'ambiguous_material_owner'})
            continue
        anchors = frozenset().union(*(key[2] for key in owners))
        parts = { (p.number, p.born, p.kind): p for key in owners for p in candidates[key][2] }
        if len(anchors) > 128 or len(parts) > 32:
            refused.append({'source': seed[0], 'character_id': seed[1], 'reason': 'material_question_capacity_unavailable'})
            continue
        rectangles = []
        for field in (0, 1):
            rectangles.append(tuple((min if axis < 2 else max)(candidates[key][field][axis] for key in owners)
                                    for axis in range(4)))
        result.append((*seed[:2], anchors, *rectangles, tuple(parts.values())))
    return tuple(result), tuple(refused)


class SourceMaterialGroupsInView:
    """Separate bounded identities for current inspection, outside physical tracks."""

    def __init__(self) -> None:
        self.records: dict[int, SourceMaterialGroup] = {}
        self.scope: tuple = ()
        self.at = -math.inf
        # Disjoint from physical and whole-reference identities; exact in JSON.
        self._next = (1 << 51)+((uuid4().int & ((1 << 29)-1)) << 20)
        self._kinds: dict[tuple, int] = {}
        self.readings: tuple = ()

    def observe(self, observer: Any, picture: np.ndarray, at: float, matches: tuple,
                *, wholes: dict | None = None) -> dict[int, SourceMaterialGroup]:
        marks = getattr(observer, 'material_marks', None)
        pixels = getattr(observer, 'pixels_consumed', None)
        self.readings = ()
        if (not isinstance(marks, MaterialMarks) or marks.pixels != pixels
                or pixels != consumed_pixels(picture, at) or not isinstance(matches, tuple) or len(matches) > 64):
            self.records, self.scope = {}, ()
            self.readings = ({'reason': 'current_material_unavailable'},)
            return {}
        scope = marks.epoch, marks.grid, marks.scale
        if scope != self.scope or at <= self.at:
            self.records, self._kinds = {}, {}
        self.scope, self.at = scope, at
        candidates = {}
        for reference, binding, point in matches:
            support = _current_support(observer, reference, binding, point, at)
            if not support:
                self.readings += ({'reason': 'source_material_not_current'},)
                continue
            source, character = reference.source, reference.declaration.character_id
            anchors = frozenset((p.number, p.born) for p in support)
            parts = _material_parts(observer, anchors, at)
            window, measured = _inspection_window(point, support, marks)
            low, high = measured[:2], measured[2:]
            if len(parts) > 32 or min(high[a]-low[a] for a in (0, 1)) <= 0:
                continue
            # Do not ask twice when one current qualified observation already
            # contains this same material. Proximity alone never links owners.
            complete = any(item.pixels == pixels and item.source == source and item.character_id == character
                           and len(item.anchors & anchors) >= 3 for item in (wholes or {}).values())
            if complete or any(p.x-p.w/2 <= low[0] and p.x+p.w/2 >= high[0]
                    and p.y-p.h/2 <= low[1] and p.y+p.h/2 >= high[1] for p in parts):
                continue
            # All extents remain proposals. Shared named material can combine
            # inspection questions without qualifying either outline.
            key = source, character, anchors
            previous = candidates.get(key)
            if previous is not None:
                window = tuple((min if axis < 2 else max)(window[axis], previous[0][axis]) for axis in range(4))
            candidates[key] = (window, measured, parts)
        questions, refused = _material_questions(candidates)
        self.readings += refused
        qualified = []
        for source, character, anchors, window, measured, parts in questions:
            peers = [old for old in self.records.values() if (old.source, old.character_id) == (source, character)
                     and len(old.anchors & anchors) >= 3]
            if len(peers) > 1:
                self.readings += ({'source': source, 'character_id': character, 'reason': 'ambiguous_material_owner'},)
                continue
            qualified.append((source, character, anchors, window, measured, parts, peers[0] if peers else None))
        claims = [old.number for *_rest, old in qualified if old is not None]
        result = {}
        for source, character, anchors, window, measured, parts, old in qualified:
            if old is not None and claims.count(old.number) != 1:
                continue
            key = source, character
            if key not in self._kinds:
                self._next += 1
                self._kinds[key] = self._next
            if old is None:
                self._next += 1
            number, born = (old.number, old.born) if old is not None else (self._next, at)
            low, high = window[:2], window[2:]
            item = SourceMaterialGroup(number, born, self._kinds[key], at,
                (low[0]+high[0])/2, (low[1]+high[1])/2, high[0]-low[0], high[1]-low[1],
                source, character, pixels, marks.epoch, anchors, measured,
                tuple((p.number, p.born, p.kind) for p in parts))
            result[number] = item
        # A missing current constellation cannot lend an old semantic identity
        # after reappearance. Continuing anchors are required on every frame.
        self.records = result
        retained = {item.kind for item in result.values()}
        self._kinds = {key: value for key, value in self._kinds.items() if value in retained}
        return result


@invariant('perception.source_material_attention_has_no_extent_authority', scope='perception',
           owner='core/perception/source_material_groups_for_visual_attention.py', observational=False)
def _material_attention_invariant() -> tuple:
    picture = np.zeros((20, 20, 3), np.uint8)
    item = SourceMaterialGroup(7, 1., 8, 1., 10., 10., 4., 6., 'source', 9,
        consumed_pixels(picture, 1.), '0'*32, frozenset((n, 1.) for n in range(3)), (9., 8., 11., 12.))
    assert item.box() == (8., 7., 12., 13.) and not item.geometry_qualified
    assert item.role is None and item.contact_verified is None
    return ()
