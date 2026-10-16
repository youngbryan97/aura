"""Carry a calibrated source point through an existing observed material basis.

Reference appearance can locate a point without selecting a controlled body.
Attaching that point requires a unique owner of the matched pixels and a
current named-point basis. The existing motion origin remains unchanged.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance
from core.perception.a_material_frame_kept_for_its_owner import query_in_observation
from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.continuing_material_before_outlines import _owns_thing
from core.perception.names_for_observed_instances import InstanceQuery, query_for
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.the_pixel_grid_a_mark_uses import SampleGrid
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


@dataclass(frozen=True)
class SourcePointSeen:
    number: int
    born: float
    at: float
    position: tuple[float, float]
    error: float


@dataclass(frozen=True)
class SourcePointSample:
    current: SourcePointSeen | None
    reason: str


@dataclass(frozen=True)
class SourcePointOnTrack:
    """Geometry in a continuing track's working coordinates, without a task role."""

    query: InstanceQuery
    source: str
    reference_sha256: str
    source_point: tuple[float, float]
    basis_at: float
    basis_point: tuple[float, float]
    registration_error: float
    scale: float
    grid: SampleGrid | None = None

    def __post_init__(self) -> None:
        vectors = (self.source_point, self.basis_point)
        if (not isinstance(self.query, InstanceQuery) or not self.query.picture_sha256
                or not isinstance(self.source, str) or not self.source
                or not isinstance(self.reference_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.reference_sha256)
                or any(not isinstance(v, (tuple, list)) or len(v) != 2 for v in vectors)
                or any(type(n) not in (int, float) or not math.isfinite(n) for v in vectors for n in v)
                or any(type(n) not in (int, float) or not math.isfinite(n)
                       for n in (self.basis_at, self.registration_error, self.scale))
                or not self.query.born <= self.basis_at <= self.query.at
                or not 0 <= self.registration_error <= 1 or not 0 < self.scale <= 8
                or self.grid is not None and not isinstance(self.grid, SampleGrid)):
            raise ValueError('a source point needs a captured instance and finite measured registration')
        object.__setattr__(self, 'source_point', tuple(self.source_point))
        object.__setattr__(self, 'basis_point', tuple(self.basis_point))

    def current(self, observer: Any, at: float, *, source: str) -> SourcePointSeen | None:
        """Return only a current point in the same source, observer and measured basis."""
        return self.inspect(observer, at, source=source).current

    def inspect(self, observer: Any, at: float, *, source: str) -> SourcePointSample:
        """Keep each refused connection visible without granting stale geometry."""
        if type(at) not in (int, float) or not math.isfinite(at):
            return SourcePointSample(None, 'capture_not_measured')
        if source != self.source:
            return SourcePointSample(None, 'source_changed')
        if type(getattr(observer, 'scale', None)) not in (int, float) or observer.scale != self.scale:
            return SourcePointSample(None, 'scale_changed')
        if at < self.query.at:
            return SourcePointSample(None, 'before_attachment')
        consumed = getattr(observer, 'pixels_consumed', None)
        if not isinstance(consumed, PixelsConsumed) or consumed.at != at:
            return SourcePointSample(None, 'capture_not_observed')
        if self.grid is not None:
            marks = getattr(observer, 'material_marks', None)
            if not isinstance(marks, MaterialMarks) or marks.pixels != consumed or marks.grid != self.grid:
                return SourcePointSample(None, 'point_sampling_changed')
        if at == self.query.at and consumed.sha256 != self.query.picture_sha256:
            return SourcePointSample(None, 'observer_pixels_do_not_match')
        thing = getattr(observer, 'things', {}).get(self.query.number)
        if thing is None:
            return SourcePointSample(None, 'track_missing')
        query = query_in_observation(thing, observer, at)
        if query is None:
            return SourcePointSample(None, 'track_not_current')
        # A visual class may change while the same named material continues.
        # Observer epoch, instance number and birth still must match.
        if query.key[:4] != self.query.key[:4]:
            return SourcePointSample(None, 'track_replaced')
        basis = _current_basis(thing, at)
        if basis is None:
            return SourcePointSample(None, 'no_current_point_basis')
        pose, starts, ends, fit = basis
        if pose.reference_at != self.basis_at:
            return SourcePointSample(None, 'point_basis_replaced')
        error = math.hypot(self.registration_error,
                           _reference_error(pose, starts, ends, fit, pose.pixel_size, target=self.basis_point))
        if not math.isfinite(error) or error > 1.:
            return SourcePointSample(None, 'reference_point_precision')
        point = tuple(float(v) for v in fit.apply(self.basis_point))
        return SourcePointSample(SourcePointSeen(thing.number, thing.born, at, point, error), 'qualified')


@dataclass(frozen=True)
class SourcePointAttachment:
    binding: SourcePointOnTrack | None
    reason: str
    owner_support: tuple[tuple[int, int], ...] = ()


def _current_basis(thing: Any, at: float) -> tuple | None:
    pose = getattr(thing, 'point_pose', None)
    if (not isinstance(pose, PointPose) or type(pose.at) not in (int, float) or pose.at != at
            or type(pose.born) not in (int, float) or pose.born != thing.born
            or type(pose.reference_at) not in (int, float) or not math.isfinite(pose.reference_at)
            or not thing.born <= pose.reference_at <= at
            or type(pose.pixel_size) not in (int, float) or not math.isfinite(pose.pixel_size) or pose.pixel_size <= 0
            or type(pose.reference_error) not in (int, float) or not math.isfinite(pose.reference_error)
            or pose.reference_error < 0 or not isinstance(pose.anchors, dict) or not isinstance(pose.visible_marks, dict)
            or not 3 <= len(pose.anchors) <= 256 or not 3 <= len(pose.visible_marks) <= 256):
        return None
    names = pose.anchors.keys() & pose.visible_marks.keys()
    if len(names) < 3 or any(type(name) is not int for name in names):
        return None
    names = sorted(names)
    try:
        starts = np.asarray([pose.anchors[n] for n in names], float)
        ends = np.asarray([pose.visible_marks[n] for n in names], float)
    except (TypeError, ValueError):
        return None
    if (starts.shape != (len(names), 2) or ends.shape != starts.shape
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or not all(_owns_thing(thing, tuple(point)) for point in ends)):
        return None
    fit = rigid_motion(starts, ends)
    if fit is None and pose.deformed:
        fit = affine_motion(starts, ends)
    return (pose, starts, ends, fit) if fit is not None else None


def attach_source_point(coordinates: Any, source_point: tuple, appearance: ReferenceAppearance,
                        picture: np.ndarray, reference: np.ndarray, observer: Any, at: float,
                        *, source: str) -> SourcePointAttachment:
    """Bind a source point only where matched material has one current track owner.

    At least six matched points and 70 percent of all matched material must
    belong to that owner. Any competing ownership refuses attachment. A
    precision estimate above one working pixel remains unqualified.
    """
    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates

    refused = SourcePointAttachment
    if not isinstance(coordinates, SourceImageCoordinates) or not isinstance(appearance, ReferenceAppearance):
        return refused(None, 'reference_not_current')
    scale = getattr(observer, 'scale', None)
    if (type(at) not in (int, float) or not math.isfinite(at)
            or type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 8):
        return refused(None, 'capture_not_measured')
    samples = material_samples(appearance)
    if samples is None:
        return refused(None, 'invalid_correspondence')
    starts, ends, linear, offset = samples
    point = coordinates.project(source_point, appearance, picture, reference, source=source)
    if point is None:
        return refused(None, 'reference_not_current')
    consumed = getattr(observer, 'pixels_consumed', None)
    if (not isinstance(consumed, PixelsConsumed) or consumed.at != at or consumed.shape != picture.shape
            or consumed.sha256 != appearance.picture_sha256):
        return refused(None, 'observer_pixels_do_not_match')
    owners = []
    marks = getattr(observer, 'material_marks', None)
    grid = marks.grid if isinstance(marks, MaterialMarks) and marks.pixels == consumed and marks.scale == scale else None
    observed = grid.from_image(ends) if grid is not None else ends*scale
    for thing in getattr(observer, 'things', {}).values():
        query = query_in_observation(thing, observer, at)
        if query is not None:
            owned = frozenset(index for index, place in enumerate(observed) if _owns_thing(thing, tuple(place)))
            if owned:
                owners.append((thing, query, owned))
    support = tuple((thing.number, len(owned)) for thing, _, owned in owners)
    candidates = [item for item in owners if len(item[2]) >= max(6, math.ceil(.7*len(ends)))]
    if len(candidates) != 1:
        return refused(None, 'no_unique_material_owner', support)
    thing, query, owned = candidates[0]
    if any(owned & other for peer, _, other in owners if peer is not thing):
        return refused(None, 'ambiguous_material_ownership', support)
    basis = _current_basis(thing, at)
    if basis is None:
        return refused(None, 'no_current_point_basis', support)
    pose, _, _, fit = basis
    located = grid.from_image(point) if grid is not None else np.asarray(point)*scale
    basis_point = tuple(float(v) for v in fit.inverse(located))
    registration_error = math.hypot(appearance.error,
        coordinates.calibration_error*float(np.linalg.norm(linear, ord=2)))*(grid.working_scale if grid is not None else scale)
    if registration_error > 1.:
        return refused(None, 'reference_point_precision', support)
    binding = SourcePointOnTrack(query.with_picture(picture), source, appearance.reference_sha256,
        tuple(source_point), pose.reference_at, basis_point, registration_error, scale, grid)
    if binding.current(observer, at, source=source) is None:
        return refused(None, 'reference_point_precision', support)
    return SourcePointAttachment(binding, 'qualified', support)


@invariant('perception.source_points_need_their_current_material_basis', scope='perception',
           owner='core/perception/a_source_point_on_current_material.py', observational=False)
def _source_point_basis_invariant() -> tuple:
    from types import SimpleNamespace

    marks = {0: (10., 10.), 1: (30., 10.), 2: (10., 30.), 3: (30., 30.)}
    pose = PointPose(0., 1., (20., 20.), marks.copy(), (20., 20.), reference_at=1., visible_marks=marks.copy())
    thing = SimpleNamespace(number=1, born=0., seen=1., kind=0, point_pose=pose, support=np.ones((40, 40), bool),
                            x=20., y=20., w=40., h=40., box=lambda: (0., 0., 40., 40.))
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels

    picture = np.zeros((40, 40, 3), np.uint8)
    observer = SimpleNamespace(things={1: thing}, scale=1., pixels_consumed=consumed_pixels(picture, 1.))
    query = query_for(thing, observer, 1.).with_picture(picture)
    binding = SourcePointOnTrack(query, 'source', '0'*64, (0., 0.), 1., (16., 14.), .01, 1.)
    assert math.dist(binding.current(observer, 1., source='source').position, (16., 14.)) < .001
    assert pose.origin == (20., 20.)
    pose.reference_at = .9
    assert binding.current(observer, 1., source='source') is None
    return ()
