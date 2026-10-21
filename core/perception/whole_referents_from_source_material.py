"""Observe whole source artwork without borrowing a fragment's track or role.

Continuing reference marks supply a measured coordinate frame. Current
opaque pixels must also match across that artwork. Published referents have
separate identities and remain outside body/control tracking. Their extents
are visual geometry, with collision and task roles still unknown.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import ndimage

from core.perception.a_source_point_on_visible_marks import SourcePointOnVisibleMarks, VisibleSourcePoint
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed, consumed_pixels
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.verify import invariant


@dataclass(frozen=True)
class WholeSourceReferent:
    number: int
    born: float
    kind: int
    seen: float
    x: float
    y: float
    w: float
    h: float
    source: str
    reference_sha256: str
    pixels: PixelsConsumed
    mark_epoch: str
    anchors: frozenset[tuple[int, float]]
    moved: bool = False
    contact_verified: bool | None = None
    role: str | None = None
    origin_path: tuple[tuple[float, float, float, float], ...] = ()
    vx: float | None = None
    vy: float | None = None
    velocity_error: float | None = None
    character_id: int | None = None
    observed_parts: tuple[tuple[int, float, int], ...] = ()
    # Fraction of opaque reference pixel centres in this capture's footprint.
    visible_fraction: float = 1.
    clipped_edges: tuple[bool, bool, bool, bool] = (False, False, False, False)

    def __post_init__(self) -> None:
        values = self.born, self.seen, self.x, self.y, self.w, self.h
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= self.born <= self.seen or min(self.w, self.h) <= 0
                or any(type(v) is not int or v < 0 for v in (self.number, self.kind))
                or not isinstance(self.pixels, PixelsConsumed) or self.pixels.at != self.seen
                or not isinstance(self.source, str) or not self.source
                or not re.fullmatch('[0-9a-f]{64}', self.reference_sha256)
                or not re.fullmatch('[0-9a-f]{32}', self.mark_epoch)
                or not isinstance(self.anchors, frozenset) or not 3 <= len(self.anchors) <= 128
                or self.role is not None or self.contact_verified is not None
                or type(self.visible_fraction) not in (int, float) or not math.isfinite(self.visible_fraction)
                or not 0 < self.visible_fraction <= 1.
                or not isinstance(self.clipped_edges, tuple) or len(self.clipped_edges) != 4
                or any(type(v) is not bool for v in self.clipped_edges)
                or not isinstance(self.origin_path, tuple) or len(self.origin_path) > 16
                or not isinstance(self.observed_parts, tuple) or len(self.observed_parts) > 32):
            raise ValueError('a whole visual referent needs owned geometry and unknown role/contact')
        motion = self.vx, self.vy, self.velocity_error
        if any(v is not None for v in motion) and (any(type(v) not in (int, float) or not math.isfinite(v) for v in motion)
                                                  or self.velocity_error < 0):
            raise ValueError('whole-referent velocity needs its finite uncertainty')

    @property
    def size(self) -> float:
        return self.w*self.h

    @property
    def path(self) -> tuple:
        return tuple((at, x, y) for at, x, y, _error in self.origin_path)

    def where_at(self, after_s: float) -> tuple:
        if self.vx is None or self.vy is None:
            raise ValueError('whole-referent motion remains unmeasured')
        return self.x+self.vx*after_s, self.y+self.vy*after_s

    def box(self) -> tuple[float, float, float, float]:
        return self.x-self.w/2, self.y-self.h/2, self.x+self.w/2, self.y+self.h/2


@dataclass(frozen=True)
class WholeExtentReading:
    reason: str
    bounds: tuple | None = None
    mean_colour_error: float | None = None
    maximum_edge_error: float | None = None
    samples: int | None = None
    visible_fraction: float | None = None
    clipped_edges: tuple[bool, bool, bool, bool] = (False, False, False, False)


def _whole_extent(reference: RegisteredVisualReference, point: VisibleSourcePoint,
                  picture: np.ndarray, marks: MaterialMarks) -> WholeExtentReading:
    """Verify all in-view artwork; clipped pixels cannot supply an object edge."""
    rgba = reference.pixels
    if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4
            or not 16 <= min(rgba.shape[:2]) <= max(rgba.shape[:2]) <= 512
            or reference.coordinates.reference_sha256 != reference.key[2]
            or len(point.sampling_corners) != 4 or marks.grid is None):
        return WholeExtentReading('reference_frame_unresolved')
    from core.perception.a_visual_reference_in_the_picture import _digest

    if _digest(rgba) != reference.coordinates.reference_sha256:
        return WholeExtentReading('changed_reference_bytes')
    corners = np.asarray(point.sampling_corners, float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return WholeExtentReading('reference_transform_unresolved')
    height, width = rgba.shape[:2]
    linear = np.column_stack(((corners[1]-corners[0])/width, (corners[2]-corners[0])/height))
    if (abs(np.linalg.det(linear)) < 1e-8 or np.linalg.cond(linear) > 100
            or np.linalg.norm(corners[3]-(corners[1]+corners[2]-corners[0])) > max(.05, point.error)):
        return WholeExtentReading('reference_transform_unresolved')
    yy, xx = np.nonzero(rgba[..., 3] >= 250)
    if len(xx) < 16:
        return WholeExtentReading('opaque_support_unavailable')
    positions = marks.grid.to_image(np.column_stack((xx, yy)) @ linear.T+corners[0])
    image_linear = linear*np.asarray(marks.grid.image[::-1])[:, None]/np.asarray(marks.grid.working[::-1])[:, None]
    half_cell = abs(image_linear) @ np.array((.5, .5))
    support_low, support_high = positions.min(axis=0)-half_cell, positions.max(axis=0)+half_cell
    clipped = (bool(support_low[0] < -.5-1e-6), bool(support_low[1] < -.5-1e-6),
               bool(support_high[0] > picture.shape[1]-.5+1e-6), bool(support_high[1] > picture.shape[0]-.5+1e-6))
    visible = ((positions[:, 0] >= -.5) & (positions[:, 1] >= -.5)
               & (positions[:, 0] <= picture.shape[1]-.5) & (positions[:, 1] <= picture.shape[0]-.5))
    fraction = float(visible.mean())
    if int(visible.sum()) < 16:
        return WholeExtentReading('visible_source_support_unavailable', visible_fraction=fraction, clipped_edges=clipped)
    xx, yy = xx[visible], yy[visible]
    # Uniform foreground cannot independently confirm a whole reference.
    colours = rgba[yy, xx, :3]
    if float(np.std(colours.astype(float), axis=0).max()) < 4.:
        return WholeExtentReading('uniform_reference_unresolved')
    edges = (xx == xx.min(), xx == xx.max(), yy == yy.min(), yy == yy.max())
    edge_samples = [np.flatnonzero(edge) for edge in edges]
    take = np.unique(np.concatenate((np.linspace(0, len(xx)-1, min(2048, len(xx)), dtype=int),
        *(indices[np.linspace(0, len(indices)-1, min(128, len(indices)), dtype=int)] for indices in edge_samples))))
    local = np.column_stack((xx[take], yy[take]))
    actual = marks.grid.to_image(local @ linear.T+corners[0])
    # Boundary pixels own their half-pixel footprint. Clamping there cannot
    # exclude a contradicted last row or column from the material check.
    actual = np.clip(actual, (0., 0.), (picture.shape[1]-1., picture.shape[0]-1.))
    errors = np.zeros(len(actual))
    for channel in range(3):
        observed = ndimage.map_coordinates(picture[..., channel].astype(float),
            [actual[:, 1], actual[:, 0]], order=1, mode='constant', cval=-255.)
        errors += abs(observed-colours[take, channel])/(3*255.)
    mean, edge_error = float(errors.mean()), max(float(errors[edge[take]].max()) for edge in edges)
    metrics = dict(mean_colour_error=mean, maximum_edge_error=edge_error, samples=len(errors),
                   visible_fraction=fraction, clipped_edges=clipped)
    if mean > .055 or np.count_nonzero(errors > .12) > .05*len(errors):
        return WholeExtentReading('whole_pixel_support_unresolved', **metrics)
    if edge_error > .12:
        return WholeExtentReading('whole_edge_support_unresolved', **metrics)
    # The visible opaque support, including disconnected parts, sets width.
    # No hull is enlarged to fit a body, request or expected collider.
    edge = np.asarray(((xx.min()-.5, yy.min()-.5), (xx.max()+.5, yy.min()-.5),
                       (xx.min()-.5, yy.max()+.5), (xx.max()+.5, yy.max()+.5)))
    extent = edge @ linear.T+corners[0]
    low, high = extent.min(axis=0), extent.max(axis=0)
    if any(clipped):
        boundary = marks.grid.from_image(np.array(((-.5, -.5), (picture.shape[1]-.5, picture.shape[0]-.5))))
        low, high = np.maximum(low, boundary[0]), np.minimum(high, boundary[1])
    if np.any(high <= low):
        return WholeExtentReading('visible_source_support_unavailable', **metrics)
    return WholeExtentReading('qualified_visible_part' if any(clipped) else 'qualified',
                              tuple(map(float, (*((low+high)/2), *(high-low)))), **metrics)


def _material_parts(observer: Any, anchors: frozenset, at: float) -> tuple:
    result = []
    numbers = {name for name, _born in anchors}
    for thing in getattr(observer, 'things', {}).values():
        pose = getattr(thing, 'point_pose', None)
        visible = getattr(pose, 'visible_marks', None)
        if (getattr(thing, 'seen', None) == at and getattr(pose, 'at', None) == at
                and getattr(pose, 'born', None) == getattr(thing, 'born', None)
                and isinstance(visible, dict) and len(numbers & visible.keys()) >= 3):
            marks = getattr(observer, 'material_marks', None)
            points = {p.number: (p.x, p.y) for p in marks.points}
            agreeing = sum(name in points and math.dist(points[name], visible[name]) <= .05
                          for name in numbers & visible.keys())
            if agreeing >= 3:
                result.append(thing)
    return tuple(result)


class WholeReferentsInView:
    """Bounded whole-appearance identities in one continuing material stream."""

    def __init__(self) -> None:
        self.records: dict[int, WholeSourceReferent] = {}
        self.scope: tuple = ()
        self.at = -math.inf
        # Keep ordinary JSON number fidelity and a disjoint practical range
        # from the observer's bounded, sequential physical-track numbers.
        self._next = (1 << 52)+((uuid4().int & ((1 << 30)-1)) << 20)
        self._kinds: dict[tuple, int] = {}
        self.readings: tuple = ()

    def observe(self, observer: Any, picture: np.ndarray, at: float, matches: tuple) -> dict:
        marks = getattr(observer, 'material_marks', None)
        pixels = getattr(observer, 'pixels_consumed', None)
        self.readings = ()
        if (not isinstance(marks, MaterialMarks) or marks.pixels != pixels
                or pixels != consumed_pixels(picture, at) or marks.grid is None):
            self.records, self.scope = {}, ()
            self.readings = ({'reason': 'current_material_capture_unavailable'},)
            return {}
        if not isinstance(matches, tuple) or len(matches) > 64:
            self.readings = ({'reason': 'whole_reference_capacity_unavailable'},)
            return {}
        scope = marks.epoch, marks.grid, marks.scale
        if scope != self.scope or at <= self.at:
            self.records, self._kinds = {}, {}
        self.scope, self.at = scope, at
        candidates = []
        for reference, binding, point in matches:
            if (not isinstance(reference, RegisteredVisualReference) or not isinstance(binding, SourcePointOnVisibleMarks)
                    or not isinstance(point, VisibleSourcePoint) or point.at != at
                    or reference.source != binding.source or binding.epoch != marks.epoch
                    or reference.coordinates.reference_sha256 != binding.reference_sha256):
                self.readings += ({'source': getattr(reference, 'source', None), 'reason': 'reference_material_unavailable'},)
                continue
            current = binding.inspect(observer, at, source=reference.source).current
            if current != point:
                self.readings += ({'source': reference.source, 'character_id': reference.declaration.character_id,
                                   'reason': 'source_material_not_current'},)
                continue
            reading = _whole_extent(reference, point, picture, marks)
            visible = {(p.number, p.born) for p in marks.points}
            anchors = frozenset((a[0], a[1]) for a in binding.anchors if (a[0], a[1]) in visible)
            self.readings += ({'source': reference.source, 'character_id': reference.declaration.character_id,
                'mark_epoch': marks.epoch, 'material_anchors': tuple(sorted(anchors)),
                'captured_at': at, 'picture_sha256': pixels.sha256,
                'reason': reading.reason, 'bounds': reading.bounds, 'mean_colour_error': reading.mean_colour_error,
                'maximum_edge_error': reading.maximum_edge_error, 'samples': reading.samples,
                'visible_fraction': reading.visible_fraction, 'clipped_edges': reading.clipped_edges},)
            if reading.bounds is None:
                continue
            extent = reading.bounds
            parts = _material_parts(observer, anchors, at)
            x, y, w, h = extent
            if len(parts) > 32:
                self.readings += ({'source': reference.source, 'reason': 'material_parent_capacity_unavailable'},)
                continue
            # A complete existing observation already offers this geometry.
            # Shared named material, not proximity, establishes this overlap.
            if any(thing.x-thing.w/2 <= x-w/2+point.error and thing.x+thing.w/2 >= x+w/2-point.error
                   and thing.y-thing.h/2 <= y-h/2+point.error and thing.y+thing.h/2 >= y+h/2-point.error for thing in parts):
                self.readings += ({'source': reference.source, 'reason': 'whole_already_observed_as_one_instance'},)
                continue
            peers = [old for old in self.records.values() if old.source == reference.source
                     and old.reference_sha256 == binding.reference_sha256 and old.mark_epoch == marks.epoch
                     and len(old.anchors & anchors) >= 3]
            if len(peers) > 1:
                continue
            candidates.append((reference, binding, point, reading, anchors, parts, peers[0] if peers else None))
        # Two current whole measurements cannot borrow one previous identity.
        claims = [old.number for _r, _b, _p, _e, _a, _parts, old in candidates if old is not None]
        result = {}
        for reference, binding, point, reading, anchors, parts, old in candidates:
            if old is not None and claims.count(old.number) != 1:
                continue
            x, y, w, h = reading.bounds
            kind_key = reference.source, reference.declaration.character_id, binding.reference_sha256
            if kind_key not in self._kinds:
                self._next += 1
                self._kinds[kind_key] = self._next
            if old is None:
                self._next += 1
            number, born = (old.number, old.born) if old is not None else (self._next, at)
            item = WholeSourceReferent(number, born, self._kinds[kind_key], at, x, y, w, h,
                reference.source, binding.reference_sha256, pixels, marks.epoch, anchors,
                character_id=reference.declaration.character_id,
                visible_fraction=reading.visible_fraction, clipped_edges=reading.clipped_edges,
                observed_parts=tuple((thing.number, thing.born, thing.kind) for thing in parts))
            path = tuple(p for p in old.origin_path if 0 < at-p[0] <= 1.) if old is not None else ()
            current = at, *point.position, point.error
            if path:
                before = path[0]
                seconds = at-before[0]
                vx, vy = tuple((current[a]-before[a])/seconds for a in (1, 2))
                error = (point.error+before[3])/seconds
                item = replace(item, vx=vx, vy=vy, velocity_error=error, moved=math.hypot(vx, vy) > error)
            item = replace(item, origin_path=(*path[-15:], current))
            result[number] = item
            self.records[number] = item
        self.records = dict(sorted(self.records.items(), key=lambda pair: pair[1].seen, reverse=True)[:128])
        retained_kinds = {item.kind for item in self.records.values()}
        self._kinds = {key: kind for key, kind in self._kinds.items() if kind in retained_kinds}
        return result


def scene_referents(observer: Any) -> tuple:
    """Current material can be named; each type keeps its own geometry authority."""
    from collections.abc import Mapping
    from core.perception.source_material_groups_for_visual_attention import SourceMaterialGroup

    things = getattr(observer, 'things', {})
    result = list(things.values()) if isinstance(things, Mapping) else []
    extra = getattr(observer, 'source_referents', {})
    if isinstance(extra, Mapping):
        known = {getattr(item, 'number', None) for item in result}
        result.extend(item for item in extra.values() if isinstance(item, WholeSourceReferent)
                      and item.number not in known and item.pixels == getattr(observer, 'pixels_consumed', None))
    material = getattr(observer, 'source_material_groups', {})
    if isinstance(material, Mapping):
        known = {getattr(item, 'number', None) for item in result}
        result.extend(item for item in material.values() if isinstance(item, SourceMaterialGroup)
                      and item.number not in known and item.pixels == getattr(observer, 'pixels_consumed', None))
    return tuple(result)


@invariant('perception.whole_source_referents_do_not_create_body_or_contact_authority', scope='perception',
           owner='core/perception/whole_referents_from_source_material.py', observational=False)
def _whole_source_referent_invariant() -> tuple:
    from types import SimpleNamespace

    picture = np.zeros((20, 20, 3), np.uint8)
    pixels = consumed_pixels(picture, 1.)
    item = WholeSourceReferent(99, 1., 99, 1., 10., 10., 6., 8., 'source', '0'*64, pixels, '0'*32,
                              frozenset((n, 1.) for n in range(3)))
    observer = SimpleNamespace(things={}, source_referents={99: item}, pixels_consumed=pixels)
    assert scene_referents(observer) == (item,) and not observer.things
    assert item.role is None and item.contact_verified is None and item.box() == (7., 6., 13., 14.)
    observer.pixels_consumed = replace(pixels, at=2.)
    assert not scene_referents(observer)
    return ()
