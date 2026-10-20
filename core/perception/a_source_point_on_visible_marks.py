"""Carry reference geometry through visible named pixels before object grouping.

This locates a source frame. It creates no object track, controlled body,
semantic role or collision surface. Every retained mark needs current pixels
and uninterrupted point identity; new corners cannot replace lost anchors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.a_visual_reference_in_the_picture import ReferenceAppearance
from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.perception.the_pixel_grid_a_mark_uses import SampleGrid
from core.verify import invariant


@dataclass(frozen=True)
class VisibleSourcePoint:
    at: float
    position: tuple[float, float]
    error: float
    support: int
    sampling_region: tuple[float, float, float, float] | None = None
    sampling_corners: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class VisibleSourceSample:
    current: VisibleSourcePoint | None
    reason: str


@dataclass(frozen=True)
class SourcePointOnVisibleMarks:
    """Source geometry supported by a pixel stream, without an object identity."""

    source: str
    reference_sha256: str
    epoch: str
    sequence: int
    at: float
    picture_sha256: str
    shape: tuple
    scale: float
    pixel_size: float
    origin: tuple[float, float]
    registration_error: float
    anchors: tuple[tuple[int, float, int, float, float], ...]
    grid: SampleGrid
    sampling_corners: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        import re

        PixelsConsumed(self.at, self.shape, self.picture_sha256)
        if not isinstance(self.grid, SampleGrid) or self.grid.image != self.shape[:2]:
            raise ValueError('a visible reference needs its measured point sampling grid')
        if (not isinstance(self.source, str) or not self.source
                or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                       for v in (self.reference_sha256, self.picture_sha256))
                or not isinstance(self.epoch, str) or not re.fullmatch('[0-9a-f]{32}', self.epoch)
                or type(self.sequence) is not int or self.sequence < 1
                or not isinstance(self.anchors, tuple) or not 3 <= len(self.anchors) <= 128
                or any(not isinstance(anchor, tuple) or len(anchor) != 5
                       or type(anchor[0]) is not int or anchor[0] < 0 or type(anchor[2]) is not int or anchor[2] < 1
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in (anchor[1], *anchor[3:]))
                       or not 0 <= anchor[1] <= self.at for anchor in self.anchors)
                or len({anchor[0] for anchor in self.anchors}) != len(self.anchors)
                or not isinstance(self.origin, tuple) or len(self.origin) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in
                       (self.at, self.scale, self.pixel_size, self.registration_error, *self.origin))
                or self.at < 0 or not 0 < self.scale <= 8 or not 0 < self.pixel_size <= 32
                or not 0 <= self.registration_error <= 1):
            raise ValueError('a visible reference needs bounded measured original marks and coordinates')
        if self.sampling_corners and (len(self.sampling_corners) != 4 or any(len(p) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in self.sampling_corners)):
            raise ValueError('a sampling proposal needs finite captured corners')

    def current(self, observer: Any, at: float, *, source: str) -> VisibleSourcePoint | None:
        return self.inspect(observer, at, source=source).current

    def inspect(self, observer: Any, at: float, *, source: str) -> VisibleSourceSample:
        refused = VisibleSourceSample
        marks = getattr(observer, 'material_marks', None)
        if source != self.source:
            return refused(None, 'source_changed')
        if (not isinstance(marks, MaterialMarks) or marks.pixels != getattr(observer, 'pixels_consumed', None)
                or type(at) not in (int, float) or not math.isfinite(at) or marks.pixels.at != at):
            return refused(None, 'marks_not_current')
        if marks.epoch != self.epoch:
            return refused(None, 'marks_epoch_changed')
        if (marks.scale != self.scale or marks.scale != getattr(observer, 'scale', None)
                or marks.pixels.shape != self.shape or marks.pixel_size != self.pixel_size or marks.grid != self.grid):
            return refused(None, 'marks_sampling_changed')
        if (marks.sequence < self.sequence or at < self.at
                or (marks.sequence == self.sequence and (at != self.at or marks.pixels.sha256 != self.picture_sha256))):
            return refused(None, 'before_mark_attachment')
        current = {point.number: point for point in marks.points}
        steps = marks.sequence-self.sequence
        retained = [(anchor, current[anchor[0]]) for anchor in self.anchors if anchor[0] in current
                    and current[anchor[0]].born == anchor[1]
                    and current[anchor[0]].pictures == anchor[2]+steps and current[anchor[0]].error <= .055]
        if len(retained) < max(3, math.ceil(.7*len(self.anchors))):
            return refused(None, 'original_marks_missing')
        starts = np.asarray([anchor[3:] for anchor, _point in retained])
        ends = np.asarray([(point.x, point.y) for _anchor, point in retained])
        fit = rigid_motion(starts, ends)
        needed = 3
        if fit is None and len(retained) >= 6:
            fit, needed = affine_motion(starts, ends), 6
        if fit is None or len(fit.inliers) < max(needed, math.ceil(.7*len(retained))):
            return refused(None, 'marks_do_not_share_a_frame')
        pose = PointPose(self.at, self.at, self.origin, {}, self.origin,
                         pixel_size=self.pixel_size, reference_error=self.registration_error)
        error = _reference_error(pose, starts, ends, fit, marks.pixel_size)
        if not math.isfinite(error) or error > 1.:
            return refused(None, 'source_point_precision')
        region, measured_corners = None, ()
        if self.sampling_corners:
            corners = fit.apply(np.asarray(self.sampling_corners))
            low, high = corners.min(axis=0), corners.max(axis=0)
            region = (*map(float, low), *map(float, high))
            measured_corners = tuple(tuple(float(v) for v in p) for p in corners)
        return refused(VisibleSourcePoint(at, tuple(float(v) for v in fit.apply(self.origin)), error,
                                         len(fit.inliers), region, measured_corners), 'qualified')


@dataclass(frozen=True)
class VisibleMarkAttachment:
    binding: SourcePointOnVisibleMarks | None
    reason: str
    support: int = 0


def _supported_marks(picture: np.ndarray, reference: np.ndarray, appearance: ReferenceAppearance,
                     marks: MaterialMarks) -> tuple:
    """Require each named patch's actual colour and opaque reference support."""
    eligible = tuple(point for point in marks.points if point.error <= .055)
    if not eligible:
        return ()
    linear, offset = np.asarray(appearance.linear), np.asarray(appearance.offset)
    if (linear.shape != (2, 2) or not np.isfinite(linear).all() or not np.isfinite(offset).all()
            or np.linalg.det(linear) <= 1e-8 or np.linalg.cond(linear) > 100):
        return ()
    offsets = np.asarray([(x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.)])*marks.pixel_size
    working = np.asarray([(point.x, point.y) for point in eligible])[:, None, :]+offsets
    actual = marks.grid.to_image(working)
    expected = (actual-offset) @ np.linalg.inv(linear).T
    alpha = ndimage.map_coordinates(reference[..., 3], [expected[..., 1], expected[..., 0]],
                                   order=1, mode='constant', cval=0.)
    difference = np.zeros(alpha.shape, float)
    for channel in range(3):
        a = ndimage.map_coordinates(picture[..., channel], [actual[..., 1], actual[..., 0]], order=1, mode='constant')
        b = ndimage.map_coordinates(reference[..., channel], [expected[..., 1], expected[..., 0]], order=1, mode='constant')
        difference += abs(a.astype(float)-b.astype(float))/(3*255)
    supported = (alpha.min(axis=1) >= 250) & (difference.mean(axis=1) <= .055)
    return tuple(point for point, good in zip(eligible, supported, strict=True) if good)[:128]


def attach_source_point_to_visible_marks(coordinates: SourceImageCoordinates, source_point: tuple,
        appearance: ReferenceAppearance, picture: np.ndarray, reference: np.ndarray, observer: Any,
        at: float, *, source: str) -> VisibleMarkAttachment:
    """Attach only measured reference pixels; an outline or object ID is unnecessary."""
    refused = VisibleMarkAttachment
    if not isinstance(coordinates, SourceImageCoordinates) or not isinstance(appearance, ReferenceAppearance):
        return refused(None, 'reference_not_current')
    if material_samples(appearance) is None:
        return refused(None, 'invalid_correspondence')
    point = coordinates.project(source_point, appearance, picture, reference, source=source)
    if point is None:
        return refused(None, 'reference_not_current')
    marks = getattr(observer, 'material_marks', None)
    if (not isinstance(marks, MaterialMarks) or marks.pixels != getattr(observer, 'pixels_consumed', None)
            or marks.pixels.at != at or marks.pixels.sha256 != appearance.picture_sha256
            or marks.scale != getattr(observer, 'scale', None)):
        return refused(None, 'marks_not_current')
    supported = _supported_marks(picture, reference, appearance, marks)
    # Three non-collinear marks redundantly determine a rigid 2-D frame.
    # The initial appearance still needs six verified correspondences;
    # subsequent affine deformation also needs six retained marks.
    if len(supported) < 3:
        return refused(None, 'reference_marks_missing', len(supported))
    positions = np.asarray([(p.x, p.y) for p in supported])
    if np.linalg.svd(positions-positions.mean(axis=0), compute_uv=False)[-1] < marks.pixel_size:
        return refused(None, 'reference_marks_not_spread', len(supported))
    error = math.hypot(appearance.error, coordinates.calibration_error*float(np.linalg.norm(appearance.linear, ord=2)))*marks.grid.working_scale
    if not math.isfinite(error) or error > 1.:
        return refused(None, 'source_point_precision', len(supported))
    binding = SourcePointOnVisibleMarks(source, coordinates.reference_sha256, marks.epoch, marks.sequence, at,
        marks.pixels.sha256, marks.pixels.shape, marks.scale, marks.pixel_size,
        tuple(float(v) for v in marks.grid.from_image(point)), error,
        tuple((p.number, p.born, p.pictures, p.x, p.y) for p in supported), marks.grid,
        tuple(tuple(float(v) for v in marks.grid.from_image(appearance.project(corner))) for corner in
              ((0., 0.), (reference.shape[1], 0.), (0., reference.shape[0]), (reference.shape[1], reference.shape[0]))))
    sample = binding.inspect(observer, at, source=source)
    return refused(binding if sample.current is not None else None, sample.reason, len(supported))


@invariant('perception.reference_marks_do_not_create_object_identity', scope='perception',
           owner='core/perception/a_source_point_on_visible_marks.py', observational=False)
def _visible_reference_identity_invariant() -> tuple:
    point = VisibleSourcePoint(1., (2., 3.), .1, 6)
    assert not any(hasattr(point, name) for name in ('number', 'kind', 'controlled', 'role'))
    assert VisibleSourceSample(None, 'original_marks_missing').current is None
    return ()
