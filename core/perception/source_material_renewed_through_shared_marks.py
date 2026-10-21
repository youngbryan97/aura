"""Renew source-coordinate support through a currently verified material majority.

An old frame must qualify before any new mark is considered. Current opaque
reference patches can add a minority of visible points through that same
projection. Capture lineage and accumulated coordinate error remain explicit.
This creates no object, semantic role or control attribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.perception.a_source_point_on_visible_marks import (
    SourcePointOnVisibleMarks, marks_supported_by_reference_projection,
)
from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.visual_references_for_current_sources import RegisteredVisualReference
from core.verify import invariant


@dataclass(frozen=True)
class SourceMaterialRenewal:
    binding: SourcePointOnVisibleMarks | None
    reason: str
    retained: int = 0
    added: int = 0


def renew_source_material(reference: RegisteredVisualReference, binding: SourcePointOnVisibleMarks,
        observer: Any, picture: np.ndarray, at: float) -> SourceMaterialRenewal:
    """Keep a verified source origin while original features gradually leave it."""
    refused = SourceMaterialRenewal
    if (not isinstance(reference, RegisteredVisualReference) or not isinstance(binding, SourcePointOnVisibleMarks)
            or reference.source != binding.source or reference.key[2] != binding.reference_sha256
            or _digest(reference.pixels) != binding.reference_sha256):
        return refused(None, 'renewal_source_changed')
    sample = binding.inspect(observer, at, source=reference.source)
    if sample.current is None:
        return refused(None, sample.reason)
    marks, point = observer.material_marks, sample.current
    if consumed_pixels(picture, at) != marks.pixels:
        return refused(None, 'renewal_pixels_changed')
    if marks.sequence <= binding.sequence or not 0 < at-binding.at <= .5:
        return refused(None, 'renewal_interval_unavailable')
    if len(binding.renewal_lineage) >= 32 or len(point.sampling_corners) != 4:
        return refused(None, 'renewal_lineage_capacity_unavailable')
    corners = marks.grid.to_image(np.asarray(point.sampling_corners))
    height, width = reference.pixels.shape[:2]
    if np.linalg.norm(corners[3]-(corners[1]+corners[2]-corners[0])) > 1e-5:
        return refused(None, 'renewal_projection_unavailable')
    linear = np.column_stack(((corners[1]-corners[0])/width, (corners[2]-corners[0])/height))
    supported = marks_supported_by_reference_projection(picture, reference.pixels, linear, corners[0], marks)
    previous = {(a[0], a[1]): a[2] for a in binding.anchors}
    steps = marks.sequence-binding.sequence
    retained = [p for p in supported if (p.number, p.born) in previous
                and p.pictures == previous[p.number, p.born]+steps]
    if len(retained) < max(3, math.ceil(.7*len(binding.anchors))):
        return refused(None, 'renewal_shared_pixel_majority_missing', len(retained))
    fresh = [p for p in supported if (p.number, p.born) not in previous and p.pictures >= 2]
    # Retained material owns at least two thirds of each new attachment.
    fresh = sorted(fresh, key=lambda p: (p.error, -p.pictures, p.number))[:min(len(retained)//2, 128-len(retained))]
    if not fresh and len(retained) == len(binding.anchors):
        return refused(None, 'renewal_not_needed', len(retained))
    chosen = (*retained, *fresh)
    places = np.asarray([(p.x, p.y) for p in chosen])
    if np.linalg.svd(places-places.mean(axis=0), compute_uv=False)[-1] < marks.pixel_size:
        return refused(None, 'renewal_marks_not_spread', len(retained))
    renewed = replace(binding, sequence=marks.sequence, at=at, picture_sha256=marks.pixels.sha256,
        origin=point.position, registration_error=max(binding.registration_error, point.error),
        anchors=tuple((p.number, p.born, p.pictures, p.x, p.y) for p in chosen),
        sampling_corners=point.sampling_corners,
        renewal_lineage=(*binding.renewal_lineage, (binding.sequence, binding.at, binding.picture_sha256)))
    checked = renewed.inspect(observer, at, source=reference.source)
    return refused(renewed if checked.current is not None else None, 'renewed' if checked.current is not None else checked.reason,
                   len(retained), len(fresh))


def renew_current_source_frames(frames: tuple, observer: Any, picture: np.ndarray, at: float,
        tables: tuple) -> tuple:
    """Spend at most four local pixel checks on already qualified source frames."""
    from core.perception.visual_references_for_current_sources import _belongs, current_visual_sources

    if not isinstance(frames, tuple) or len(frames) > 64:
        return (), ({'reason': 'renewal_frame_capacity_unavailable'},)
    scope, candidates = current_visual_sources(tables), []
    marks = getattr(observer, 'material_marks', None)
    for index, item in enumerate(frames):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        reference, binding = item
        if (not isinstance(reference, RegisteredVisualReference) or not isinstance(binding, SourcePointOnVisibleMarks)
                or not any(_belongs(reference, table) for table in scope) or binding.at >= at):
            continue
        sample = binding.inspect(observer, at, source=reference.source)
        if sample.current is not None:
            missing = 1.-sample.current.support/len(binding.anchors)
            candidates.append((-missing, (index-marks.sequence) % len(frames), reference, binding))
    offered, readings = [], []
    for _loss, _turn, reference, binding in sorted(candidates, key=lambda entry: entry[:2])[:4]:
        result = renew_source_material(reference, binding, observer, picture, at)
        readings.append(dict(source=reference.source, character_id=reference.declaration.character_id,
            reference_sha256=reference.key[2], captured_at=at, picture_sha256=observer.pixels_consumed.sha256,
            reason=result.reason, retained=result.retained, added=result.added,
            lineage=list(result.binding.renewal_lineage) if result.binding is not None else [],
            registration_error=result.binding.registration_error if result.binding is not None else None))
        if result.binding is not None:
            offered.append((reference, result.binding))
    return tuple(offered), tuple(readings)


@invariant('perception.source_material_renewal_cannot_start_without_shared_pixels', scope='perception',
           owner='core/perception/source_material_renewed_through_shared_marks.py', observational=False)
def _source_material_renewal_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
    from core.perception.named_material_in_a_consumed_frame import MaterialMarks
    from core.perception.points_that_keep_their_motion import SeenPoint
    from core.perception.the_pixel_grid_a_mark_uses import SampleGrid
    from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset

    pixels = np.random.default_rng(481).integers(10, 230, (64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    picture = np.zeros((96, 96, 3), np.uint8)
    picture[12:76, 13:77] = pixels[..., :3]
    coordinates = SourceImageCoordinates('source', _digest(pixels), '0'*64, (64, 64), ((1., 0.), (0., 1.)), (0., 0.), .01)
    reference = RegisteredVisualReference('source', DeclaredVisualAsset('art', 17, 'sprite', 0, 39, (10,)), coordinates, pixels)
    places = ((20., 20.), (35., 20.), (55., 20.), (20., 45.), (55., 45.), (35., 60.))
    anchors = tuple((i, .9, 2, x, y) for i, (x, y) in enumerate(places))
    grid = SampleGrid((96, 96), (96, 96), (96, 96))
    binding = SourcePointOnVisibleMarks('source', coordinates.reference_sha256, '1'*32, 1, 1., '0'*64,
        picture.shape, 1., 1., (10., 10.), .01, anchors, grid, ((10., 10.), (74., 10.), (10., 74.), (74., 74.)))
    points = tuple(SeenPoint(i, x+3., y+2., x, y, .9, 1., 1.1, .001, 3)
                   for i, (x, y) in enumerate(places) if i)
    points += (SeenPoint(99, 44., 48., 41., 46., 1., 1., 1.1, .001, 2),)
    receipt = consumed_pixels(picture, 1.1)
    marks = MaterialMarks('1'*32, 2, receipt, 1., 1., points, grid)
    observer = SimpleNamespace(material_marks=marks, pixels_consumed=receipt, scale=1.)
    result = renew_source_material(reference, binding, observer, picture, 1.1)
    assert result.binding is not None and result.retained == 5 and result.added == 1
    assert result.binding.origin == (13., 12.) and result.binding.registration_error >= binding.registration_error
    observer.material_marks = replace(marks, points=tuple(replace(p, number=p.number+1000) for p in points))
    assert renew_source_material(reference, binding, observer, picture, 1.1).binding is None
    return ()
