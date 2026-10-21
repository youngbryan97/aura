"""Remeasure source geometry in the consumed frame, independently of object grouping."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np

from core.perception.a_source_point_on_visible_marks import VisibleSourcePoint, attach_source_point_to_visible_marks
from core.perception.a_search_hint_for_these_pixels import _capture, make_search_hint
from core.perception.affine_motion_from_named_points import affine_motion
from core.perception.material_samples_in_a_reference_fit import material_samples
from core.perception.named_material_in_a_consumed_frame import MaterialMarks
from core.perception.pixels_consumed_by_the_observer import PixelsConsumed, consumed_pixels
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.reference_material_near_a_measured_pose import near_reference_appearances
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.perception.source_reference_proposals import SourceReferenceProposal
from core.perception.visual_references_for_current_sources import _belongs, current_visual_sources
from core.runtime.optional_work_shares_time import OptionalWorkShare
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
    wanted = tuple(key for key in keys if (key[:3] if isinstance(key, tuple) and len(key) == 4 else key) not in covered)
    if not covered or not wanted:
        return keys[cursor % len(keys)]
    if cursor % 4 == 3:
        return keys[(cursor//4) % len(keys)]
    return wanted[(cursor-cursor//4) % len(wanted)]


def _refitted_pose_bank(poses: tuple, index: int, found: tuple) -> tuple:
    """Update the searched hint without erasing alternatives not searched this frame."""
    return (*poses[:index], found[0], *poses[index+1:]) if found else poses


class SourceGeometryInCurrentPixels:
    """Worker-owned optional measurements: one pose of one reference per frame."""

    def __init__(self) -> None:
        self.scope = ()
        self.records: dict = {}
        self.cursor = 0
        self.pose_cursors: dict = {}
        self.search_cursors: dict = {}
        self.at = -math.inf
        self.work_share = OptionalWorkShare()
        self._essential_seconds = 0.
        self._prepared = None
        self.material_frames = ()
        self.material_status = ()
        self._searched = None
        self._pending_seconds = None
        self.refinement_readings = ()

    def finish_sampling(self):
        """Charge completed optional work even when observation was abandoned."""
        if self._pending_seconds is not None:
            seconds, self._pending_seconds = self._pending_seconds, None
            self.work_share.spent(seconds)

    def sampling_before_observation(self, picture, at, proposals: tuple, tables: tuple,
                                    covered: tuple = (), *, priority: tuple = ()):
        """Fresh correspondence can request corners before object grouping exists.

        Prior mandatory work buys one current refit. Its result is reused for
        geometry after observation; neither the hint nor a fit grants identity.
        """
        self.finish_sampling()
        self.refinement_readings = ()
        self._prepared = None
        self.material_frames = ()
        self.material_status = ()
        if _capture(at) is None:
            return None
        capture = consumed_pixels(picture, float(at))
        if capture is None:
            return None
        self._prepared = capture, (), ()
        scope = current_visual_sources(tables)
        if scope != self.scope:
            self.scope, self.records, self.cursor, self.pose_cursors = scope, {}, 0, {}
            self.search_cursors.clear()
        cost, self._essential_seconds = self._essential_seconds, 0.
        if not proposals and not self.records:
            return None
        if not self.work_share.reserve(cost):
            self._prepared = capture, (), ('fresh_reference_work_deferred',)
            return None
        began = time.perf_counter()
        try:
            fits, unavailable = self._fits(picture, float(at), proposals, tables, covered, one_position=True, priority=priority)
            self._prepared = capture, fits, unavailable
            hint = None
            if fits:
                _reference, pose = fits[0]
                height, width = picture.shape[:2]
                x, y, w, h = pose.box
                region = (max(0., x/width), max(0., y/height),
                          min(1., (x+w)/width), min(1., (y+h)/height))
                hint = make_search_hint(picture, at, normalized_region=region)
        except BaseException:
            self._prepared = None
            self.work_share.spent(time.perf_counter()-began)
            raise
        self._pending_seconds = time.perf_counter()-began
        return hint

    def measure_if_budget(self, picture, at, state, proposals: tuple, tables: tuple, covered: tuple = (),
                          *, essential_seconds: float, priority: tuple = ()) -> tuple:
        """Repay prior optional refits before delaying another essential picture."""
        self.material_frames = ()
        self.material_status = ()
        if self._prepared is not None:
            capture, fits, unavailable = self._prepared
            self._prepared = None
            self._essential_seconds = essential_seconds
            began = time.perf_counter()
            try:
                grid = _current_grid(picture, at, state)
                if (grid is None or capture != state.get('pixels_consumed')
                        or current_visual_sources(tables) != self.scope):
                    return (), ('fresh_reference_capture_unavailable',)
                return self._measured(fits, picture, at, state, grid, unavailable)
            finally:
                if self._pending_seconds is not None:
                    self._pending_seconds += time.perf_counter()-began
                    self.finish_sampling()
        self._essential_seconds = 0.
        self.refinement_readings = ()
        if not proposals and not self.records:
            return self.measure(picture, at, state, proposals, tables, covered, one_position=True, priority=priority)
        if not self.work_share.reserve(essential_seconds):
            return (), ('fresh_reference_work_deferred',)
        began = time.perf_counter()
        try:
            return self.measure(picture, at, state, proposals, tables, covered, one_position=True, priority=priority)
        finally:
            self.work_share.spent(time.perf_counter()-began)

    def measure(self, picture, at, state, proposals: tuple, tables: tuple, covered: tuple = (),
                *, one_position: bool = False, priority: tuple = ()) -> tuple:
        self.material_frames = ()
        self.material_status = ()
        grid = _current_grid(picture, at, state)
        if grid is None:
            return (), ('fresh_reference_capture_unavailable',)
        fits, unavailable = self._fits(picture, at, proposals, tables, covered, one_position=one_position, priority=priority)
        return self._measured(fits, picture, at, state, grid, unavailable)

    def _measured(self, fits, picture, at, state, grid, unavailable):
        from types import SimpleNamespace

        unavailable = list(unavailable)
        measured = []
        current = SimpleNamespace(**state)
        for reference, pose in fits:
            point = _current_point(reference, pose, picture, at, grid)
            if point is None:
                unavailable.append('fresh_reference_precision_unavailable')
            else:
                measured.append(CurrentSourceReference(reference.source, reference.declaration.character_id,
                    reference.declaration.name, reference.coordinates.reference_sha256, state['pixels_consumed'], point))
                attached = attach_source_point_to_visible_marks(reference.coordinates, (0., 0.), pose,
                    picture, reference.pixels, current, at, source=reference.source)
                if attached.binding is not None:
                    self.material_frames = (*self.material_frames, (reference, attached.binding))
                else:
                    self.material_status = (*self.material_status, attached.reason)
        if measured and self._searched is not None:
            key, selected = self._searched
            self.search_cursors[key][selected] = 0
        return _enclosed_measurements(tuple(measured)), tuple(unavailable)

    def _fits(self, picture, at, proposals, tables, covered, *, one_position, priority=()):
        self.refinement_readings = ()
        self._searched = None
        if type(one_position) is not bool:
            return (), ('fresh_reference_position_capacity_unavailable',)
        scope = current_visual_sources(tables)
        if scope != self.scope or at <= self.at:
            self.scope, self.records, self.cursor, self.pose_cursors = scope, {}, 0, {}
            self.search_cursors.clear()
        self.at = at
        if not scope or not self.records and not proposals:
            return (), ()
        if not isinstance(proposals, tuple) or len(proposals) > 32:
            return (), ('fresh_reference_proposal_capacity_unavailable',)
        unavailable = []
        for proposal in proposals:
            if not isinstance(proposal, SourceReferenceProposal) or proposal.capture.at > at:
                unavailable.append('fresh_reference_proposal_not_current')
                continue
            reference = proposal.reference
            old = self.records.get(reference.variant_key)
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
            self.records[reference.variant_key] = reference, poses, proposal.revision
        if not self.records:
            return (), tuple(unavailable)
        hints = {key for key in covered if isinstance(key, tuple) and len(key) == 3
                 and isinstance(key[0], str) and type(key[1]) is int and isinstance(key[2], str)} \
            if isinstance(covered, tuple) and len(covered) <= 32 else set()
        from core.perception.a_source_extent_requested_in_play import extent_reference_to_measure

        priority = priority if isinstance(priority, tuple) and len(priority) <= 4 else ()
        key = extent_reference_to_measure(tuple(self.records), hints, self.cursor, priority, _reference_to_measure)
        self.cursor += 1
        reference, poses, revision = self.records[key]
        if any(other[:2] == key[:2] for other in self.records if other != key):
            unavailable.append('fresh_reference_variants_unsearched')
        if not 1 <= len(poses) <= 4:
            return (), (*unavailable, 'fresh_reference_pose_capacity_unavailable')
        pose_cursor = self.pose_cursors.get(key, 0)
        selected = pose_cursor % len(poses)
        self.pose_cursors[key] = pose_cursor+1
        if len(poses) > 1:
            unavailable.append('fresh_reference_alternatives_unsearched')
        search = self.search_cursors.setdefault(key, [0]*4)
        position = search[selected]
        if one_position:
            unavailable.append('fresh_reference_positions_unsearched')
        options = {'position': position} if one_position else {}
        found = near_reference_appearances(picture, reference.pixels, (poses[selected],), **options)
        if found:
            from core.perception.a_reference_pose_refined_in_current_pixels import refine_reference_pose

            refined = refine_reference_pose(picture, reference.pixels, found[0])
            self.refinement_readings = (dict(source=reference.source, character_id=reference.declaration.character_id,
                reference_sha256=reference.key[2], captured_at=at, picture_sha256=found[0].picture_sha256,
                **refined.receipt()),)
            if refined.pose is not None:
                found = (refined.pose, *found[1:])
        if found:
            self.records[key] = reference, _refitted_pose_bank(poses, selected, found), revision
            if len(found) > 1:
                unavailable.append('fresh_reference_additional_matches_not_retained')
        if one_position:
            search[selected] = (position+1) % 6
            self._searched = key, selected
        return tuple((reference, pose) for pose in found), tuple(unavailable)


@invariant('perception.optional_geometry_repays_observation_time', scope='perception',
           owner='core/perception/source_geometry_in_current_pixels.py', observational=False)
def _optional_reference_share_invariant() -> tuple:
    quota, started = OptionalWorkShare(), []
    for picture in range(9):
        if quota.reserve(.09):
            started.append(picture)
            assert not quota.reserve(.09)
            quota.spent(.09)
    assert started == [0, 3, 6] and not quota.active
    assert quota.credit <= quota.cap
    return ()


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
    bank = ('pose-a', 'pose-b', 'pose-c', 'pose-d')
    for index in range(len(bank)):
        updated = _refitted_pose_bank(bank, index, ('current', 'another current match'))
        assert len(updated) == len(bank) and updated[index] == 'current'
        assert all(updated[j] == bank[j] for j in range(len(bank)) if j != index)
    assert _refitted_pose_bank(bank, 0, ()) == bank
    return ()


@invariant('perception.source_sampling_rechecks_current_pixels', scope='perception',
           owner='core/perception/source_geometry_in_current_pixels.py', observational=False)
def _reference_sampling_invariant() -> tuple:
    from core.perception.a_visual_reference_in_the_picture import _digest, reference_appearances
    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime
    from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, FlashVisualAssets
    from core.perception.visual_references_for_current_sources import RegisteredVisualReference

    rgba = np.random.default_rng(873).integers(0, 255, (32, 40, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    picture = np.zeros((96, 144, 3), np.uint8)
    picture[24:56, 40:80] = rgba[..., :3]
    source = 'swf:'+'3'*64
    asset = DeclaredVisualAsset('unfamiliar', 12, 'sprite', 0, 39, (10,))
    coordinates = SourceImageCoordinates(source, _digest(rgba), '0'*64,
        (40, 32), ((1., 0.), (0., 1.)), (0., 0.), .01)
    reference = RegisteredVisualReference(source, asset, coordinates, rgba)
    poses = reference_appearances(picture, rgba, scales=(1.,))
    proposal = SourceReferenceProposal(reference, consumed_pixels(picture, 1.), poses, 1)
    table = FlashVisualAssets(source, (asset,))
    at = CanvasFrameTime(2., lower=2., upper=2.01, arrived=2.02, source='canvas-bracket',
        source_at=2., epoch=1, surface='document/canvas', sequence=1)
    worker = SourceGeometryInCurrentPixels()
    worker._essential_seconds = .3
    hint = worker.sampling_before_observation(picture, at, (proposal,), (table,))
    assert hint is not None and not hasattr(hint, 'number') and not hasattr(hint, 'role')
    worker = SourceGeometryInCurrentPixels()
    worker._essential_seconds = .3
    assert worker.sampling_before_observation(picture*0, at, (proposal,), (table,)) is None
    return ()
