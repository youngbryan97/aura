"""Remeasure source geometry in the consumed frame, independently of object grouping."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from core.perception.a_source_point_on_visible_marks import VisibleSourcePoint
from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed, consumed_pixels
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.reference_material_near_a_measured_pose import near_reference_appearances
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.visual_references_for_current_sources import _belongs, current_visual_sources
from core.verify import invariant


@dataclass(frozen=True)
class CurrentSourceReference:
    """A fresh pixel measurement, with no continuing object or controlled identity."""

    source: str
    character_id: int
    exported_name: str
    reference_sha256: str
    pixels: PixelsConsumed
    point: VisibleSourcePoint
    hypotheses: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.source, str) or not self.source or type(self.character_id) is not int
                or self.character_id < 0 or not isinstance(self.exported_name, str) or len(self.exported_name) > 256
                or not isinstance(self.reference_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.reference_sha256)
                or not isinstance(self.pixels, PixelsConsumed) or not isinstance(self.point, VisibleSourcePoint)
                or self.point.at != self.pixels.at or type(self.point.support) is not int or not 6 <= self.point.support <= 256
                or type(self.hypotheses) is not int or not 1 <= self.hypotheses <= 32
                or len(self.point.position) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in (*self.point.position, self.point.error)) or not 0 <= self.point.error <= 1.):
            raise ValueError('fresh source geometry needs current redundant pixels and bounded precision')


def _enclosed_measurements(measurements: tuple) -> tuple:
    """Enclose agreeing source origins; hypotheses do not count physical objects."""
    groups = []
    for item in measurements:
        for group in groups:
            first = group[0]
            if ((first.source, first.character_id, first.reference_sha256, first.pixels)
                    != (item.source, item.character_id, item.reference_sha256, item.pixels)):
                continue
            candidates = (*group, item)
            centre = tuple(math.fsum(p.point.position[axis] for p in candidates)/len(candidates) for axis in (0, 1))
            if max(math.dist(centre, p.point.position)+p.point.error for p in candidates) <= 1.:
                group.append(item)
                break
        else:
            groups.append([item])
    result = []
    for group in groups:
        first = group[0]
        centre = tuple(math.fsum(p.point.position[axis] for p in group)/len(group) for axis in (0, 1))
        error = max(math.dist(centre, p.point.position)+p.point.error for p in group)
        point = VisibleSourcePoint(first.point.at, centre, error, min(p.point.support for p in group))
        result.append(CurrentSourceReference(first.source, first.character_id, first.exported_name,
            first.reference_sha256, first.pixels, point, sum(p.hypotheses for p in group)))
    return tuple(result)


def _current_grid(picture, at, state):
    marks = state.get('material_marks')
    receipt = state.get('pixels_consumed')
    if (not isinstance(marks, MaterialMarks) or marks.pixels != receipt or marks.grid is None
            or receipt != consumed_pixels(picture, at)):
        return None
    return marks.grid


def _current_point(reference, pose, picture, at, grid):
    samples = material_samples(pose)
    point = reference.coordinates.project((0., 0.), pose, picture, reference.pixels, source=reference.source)
    if samples is None or point is None:
        return None
    starts, observed, _linear, _offset = samples
    predicted = starts @ np.asarray(pose.linear).T+pose.offset
    starts, ends = grid.from_image(predicted), grid.from_image(observed)
    fit = rigid_motion(starts, ends) or affine_motion(starts, ends)
    if fit is None or len(fit.inliers) < max(6, math.ceil(.7*len(starts))):
        return None
    position = tuple(float(v) for v in grid.from_image(point))
    ratio = min(1., 480/max(picture.shape[:2]))
    viewed = tuple(max(1, round(v*ratio)) for v in picture.shape[:2])
    size = max(grid.working[axis]/viewed[axis] for axis in (0, 1))
    calibration = reference.coordinates.calibration_error*float(np.linalg.norm(pose.linear, ord=2))*grid.working_scale
    owner = PointPose(at, at, position, {}, position, pixel_size=size, reference_error=calibration)
    error = _reference_error(owner, starts, ends, fit, size)
    if not math.isfinite(error) or error > 1.:
        return None
    return VisibleSourcePoint(at, position, error, len(fit.inliers))


def _reference_to_measure(keys: tuple, covered: set, cursor: int):
    """Priority hints spend work on missing geometry; regular audits stay fair."""
    wanted = tuple(key for key in keys if key not in covered)
    if not covered or not wanted:
        return keys[cursor % len(keys)]
    if cursor % 4 == 3:
        return keys[(cursor//4) % len(keys)]
    return wanted[(cursor-cursor//4) % len(wanted)]


class SourceGeometryInCurrentPixels:
    """Worker-owned optional measurements: one registered reference per frame."""

    def __init__(self) -> None:
        self.scope = ()
        self.records: dict = {}
        self.cursor = 0
        self.at = -math.inf

    def measure(self, picture, at, state, proposals: tuple, tables: tuple, covered: tuple = ()) -> tuple:
        scope = current_visual_sources(tables)
        if scope != self.scope or at <= self.at:
            self.scope, self.records, self.cursor = scope, {}, 0
        self.at = at
        if not scope or not self.records and not proposals:
            return (), ()
        if not isinstance(proposals, tuple) or len(proposals) > 32:
            return (), ('fresh_reference_proposal_capacity_unavailable',)
        grid = _current_grid(picture, at, state)
        if not scope or grid is None:
            return (), ('fresh_reference_capture_unavailable',) if scope else ()
        unavailable = []
        for proposal in proposals:
            if not isinstance(proposal, SourceReferenceProposal) or proposal.capture.at > at:
                unavailable.append('fresh_reference_proposal_not_current')
                continue
            reference = proposal.reference
            old = self.records.get(reference.key)
            if old is not None and old[2] >= proposal.revision:
                continue
            if not any(_belongs(reference, table) for table in scope):
                unavailable.append('fresh_reference_source_changed')
                continue
            if old is None and len(self.records) >= 32:
                unavailable.append('fresh_reference_catalog_capacity_unavailable')
                continue
            poses = tuple(proposal.appearances)
            if old is not None:
                probes = ((0., 0.), (reference.pixels.shape[1]-1., 0.), (0., reference.pixels.shape[0]-1.))
                poses = (*old[1], *(pose for pose in poses if not any(
                    max(math.dist(pose.project(p), prior.project(p)) for p in probes) < 1. for prior in old[1])))
                if len(poses) > 4 and len(proposal.appearances) <= 4:
                    # These are old search hints, without continuing object
                    # identity. A bounded fresh bank can replace them while
                    # the unsearched alternatives remain explicit.
                    poses = proposal.appearances
                    unavailable.append('fresh_reference_prior_search_retired')
            self.records[reference.key] = reference, poses, proposal.revision
        if not self.records:
            return (), tuple(unavailable)
        hints = {key for key in covered if isinstance(key, tuple) and len(key) == 3
                 and isinstance(key[0], str) and type(key[1]) is int and isinstance(key[2], str)} \
            if isinstance(covered, tuple) and len(covered) <= 32 else set()
        key = _reference_to_measure(tuple(self.records), hints, self.cursor)
        self.cursor += 1
        reference, poses, revision = self.records[key]
        if not 1 <= len(poses) <= 4:
            return (), (*unavailable, 'fresh_reference_pose_capacity_unavailable')
        found = near_reference_appearances(picture, reference.pixels, poses)
        if found:
            self.records[key] = reference, found, revision
        measured = []
        for pose in found:
            point = _current_point(reference, pose, picture, at, grid)
            if point is None:
                unavailable.append('fresh_reference_precision_unavailable')
            else:
                measured.append(CurrentSourceReference(reference.source, reference.declaration.character_id,
                    reference.declaration.name, reference.coordinates.reference_sha256, state['pixels_consumed'], point))
        return _enclosed_measurements(tuple(measured)), tuple(unavailable)


@invariant('perception.fresh_source_geometry_needs_the_consumed_pixels', scope='perception',
           owner='core/perception/source_geometry_in_current_pixels.py', observational=False)
def _fresh_source_capture_invariant() -> tuple:
    from core.perception.the_pixel_grid_a_mark_uses import SampleGrid

    picture = np.zeros((32, 32, 3), np.uint8)
    receipt = consumed_pixels(picture, 1.)
    marks = MaterialMarks('0'*32, 1, receipt, 1., 1., (), SampleGrid((32, 32), (32, 32), (32, 32)))
    assert _current_grid(picture, 1., dict(material_marks=marks, pixels_consumed=receipt)) == marks.grid
    assert _current_grid(picture, 2., dict(material_marks=marks, pixels_consumed=receipt)) is None
    return ()


@invariant('perception.reference_attention_keeps_fair_audits', scope='perception',
           owner='core/perception/source_geometry_in_current_pixels.py', observational=False)
def _reference_attention_invariant() -> tuple:
    keys = ('a', 'b', 'c', 'd', 'e')
    selected = tuple(_reference_to_measure(keys, {'a'}, i) for i in range(20))
    assert set(selected) == set(keys)
    assert all(selected[i] == keys[i//4] for i in range(3, 20, 4))
    assert all(key != 'a' for i, key in enumerate(selected) if i % 4 != 3)
    return ()
