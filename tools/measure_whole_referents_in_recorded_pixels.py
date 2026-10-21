#!/usr/bin/env python3
"""Measure source-parent observations in retained pixels, without input or models.

This runs the ordinary classical observer, registered reference matcher and
whole-material producer. A recording supplies pixels and capture times only.
It does not replay actions, ask for names, bind goals or establish live play.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _frames(directory: Path, start: int, count: int):
    directory = directory.resolve()
    with (directory/'events.jsonl').open() as stream:
        for _ in range(100_000):
            line = stream.readline(1_000_001)
            if not line:
                break
            if len(line) > 1_000_000:
                raise ValueError('recording event exceeds its byte budget')
            event = json.loads(line)
            if event.get('kind') != 'frame' or event.get('sequence', -1) < start:
                continue
            path = (directory/event['file']).resolve()
            if not path.is_relative_to(directory) or path.stat().st_size > 32*1024*1024:
                raise ValueError('a recorded image needs a bounded contained file')
            with Image.open(path) as image:
                if image.width*image.height > 8_000_000 or image.mode != 'RGB':
                    raise ValueError('a retained RGB capture is required')
                picture = np.array(image, dtype=np.uint8)
            if (list(picture.shape) != event['shape']
                    or hashlib.sha256(picture.tobytes()).hexdigest() != event['raw_sha256']):
                raise ValueError('recorded pixels no longer match their receipt')
            yield event, picture
            count -= 1
            if count == 0:
                break


def measure(recording: Path, source: Path, catalog_root: Path, *, start: int, frames: int, match_every: int,
            sample_matched_appearances: bool = False, request_material_extents: bool = False) -> dict:
    from core.runtime.classical_compute_boundary import current_classical_boundary
    from core.runtime.model_worker_policy import model_workers_allowed

    boundary = current_classical_boundary()
    if boundary is None or model_workers_allowed():
        raise RuntimeError('recorded-pixel measurement requires classical imports and model workers prohibited')
    boundary.require_intact()
    from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
    from core.perception.source_references_while_play_advances import _matched, _with_current_material_frames, retention_for_source_frames
    from core.perception.source_geometry_in_current_pixels import SourceGeometryInCurrentPixels
    from core.perception.source_reference_proposals import SourceReferenceProposal
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime
    from core.perception.play_observation_without_holding_the_loop import _seen
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from core.perception.visual_references_for_current_sources import acquire_visual_references, reference_variant_coverage
    from core.perception.what_moves_in_the_picture import WhatMoves
    from core.perception.whole_referents_from_source_material import WholeReferentsInView
    from core.perception.source_material_groups_for_visual_attention import SourceMaterialGroupsInView
    from core.perception.what_things_look_like import LookingAtThings
    from core.perception.a_source_extent_requested_in_play import SourceExtentRequest
    from core.perception.source_layers_with_copied_positions import source_layer_proposals
    from core.perception.flash_action_sources import read_flash_action_streams
    from core.perception.composed_source_artwork_for_pixel_search import with_source_layer_proposals, composition_record
    from core.perception.visual_references_for_current_sources import ReferencesAcquired, interleaved_reference_variants

    if not 1 <= frames <= 100 or not 1 <= match_every <= 100 or start < 0:
        raise ValueError('replay needs bounded frame and match counts')
    if request_material_extents and not sample_matched_appearances:
        raise ValueError('extent requests require the ordinary current-sampling path')
    if not 8 <= source.stat().st_size <= 40*1024*1024:
        raise ValueError('source bytes exceed their bound')
    raw = source.read_bytes()
    table = read_flash_visual_assets(raw)
    catalog = acquire_visual_references((table,), root=catalog_root)
    layers = source_layer_proposals(read_flash_action_streams(raw), (table,))
    references, refused_layers = with_source_layer_proposals(catalog.references, layers)
    catalog = ReferencesAcquired(interleaved_reference_variants(references), (*catalog.unavailable, *refused_layers))
    observer, wholes = WhatMoves(), WholeReferentsInView()
    material_groups = SourceMaterialGroupsInView()
    geometry, proposals = SourceGeometryInCurrentPixels(), {}
    bindings, searches, rows = (), [], []
    requests = ()
    retention = None
    began = time.monotonic()
    for index, (event, picture) in enumerate(_frames(recording, start, frames)):
        at = float(event['capture']['at'])
        state = {}
        if sample_matched_appearances:
            capture = dict(event['capture'])
            capture.pop('at')
            options = dict(whole_referents=wholes, material_groups=material_groups,
                           reference_frames=bindings, extent_requests=requests) if request_material_extents else {}
            _, state = _seen(observer, picture, CanvasFrameTime(at, **capture), fresh_references=geometry,
                reference_proposals=tuple(proposals.values()), reference_tables=(table,), material_retention=retention, **options)
        else:
            observer.see(picture, at, material_retention=retention)
        registered = {r.variant_key: r for r in catalog.references}
        offered = tuple((registered[key], binding) for key, binding in state.get('fresh_source_material_frames', ()))
        bindings, _ = _with_current_material_frames(bindings, offered, observer, at, (table,))
        if index % match_every == 0:
            captured = keep_material_frame(observer, picture, at)
            if captured is None:
                raise ValueError('observer did not retain this exact source capture')
            found = []
            for reference in catalog.references:
                result = _matched(reference, captured)
                searches.append(dict(sequence=event['sequence'], source=reference.source,
                    reference_frame_number=reference.frame_number, reference_sha256=reference.key[2],
                    character_id=reference.declaration.character_id, appearances=len(result.appearances),
                    object_origins=len(result.bindings), material_frames=len(result.frames), refused=list(result.refused)))
                if result.appearances:
                    proposals[reference.variant_key] = SourceReferenceProposal(reference, result.capture,
                        result.appearances, index+1)
                found.extend((reference, binding) for binding in result.frames)
            bindings = (*bindings, *found)[-64:]
        current, kept, frame_readings = [], [], []
        for reference, binding in bindings:
            sample = binding.inspect(observer, at, source=reference.source)
            point = sample.current
            frame_readings.append(dict(source=reference.source, character_id=reference.declaration.character_id,
                reference_sha256=reference.key[2], reference_frame_number=reference.frame_number,
                attached_at=binding.at, reason=sample.reason, initial_anchors=len(binding.anchors),
                current_support=point.support if point is not None else None,
                current_error=point.error if point is not None else None))
            if point is not None:
                current.append((reference, binding, point))
                kept.append((reference, binding))
        bindings = tuple(kept)
        retention = retention_for_source_frames(observer, at, bindings)
        started = time.perf_counter()
        if request_material_extents:
            # Broad search results become available to the following frame.
            # The same worker call both consumes the demand and measures it.
            seen = state.get('source_referents', {})
            observer.source_material_groups = state.get('source_material_groups', {})
        else:
            seen = wholes.observe(observer, picture, at, tuple(current))
            observer.source_material_groups = material_groups.observe(observer, picture, at, tuple(current), wholes=seen)
        observer.source_referents = seen
        requests = tuple(SourceExtentRequest(g) for g in list(observer.source_material_groups.values())[:4]) \
            if request_material_extents else ()
        eligible = LookingAtThings()._to_look_at(observer, None, set(), at=at)
        elapsed = time.perf_counter()-started
        rows.append(dict(sequence=event['sequence'], file=event['file'], at=at,
            picture_sha256=observer.pixels_consumed.sha256, material_marks=len(observer.material_marks.points),
            source_frames=len(current), producer_seconds=elapsed, readings=list(wholes.readings),
            source_frame_readings=frame_readings,
            source_material_renewals=state.get('source_material_renewal_readings', ()),
            retained_marks_admitted=getattr(observer, 'material_retention_admitted', None),
            material_groups=[dict(number=g.number, born=g.born, kind=g.kind, source=g.source,
                character_id=g.character_id, box=list(g.box()), anchors=len(g.anchors),
                material_bounds=list(g.material_bounds),
                geometry_qualified=g.geometry_qualified, role=g.role, contact_verified=g.contact_verified,
                observed_parts=list(g.observed_parts)) for g in observer.source_material_groups.values()],
            material_group_readings=list(material_groups.readings),
            extent_receipts=list(state.get('source_extent_receipts', ())),
            extent_requests_offered=len(requests),
            material_groups_eligible_for_naming=[q.number for q in eligible if q.number in observer.source_material_groups],
            sampling_hint=asdict(observer.sampling_hint_receipt) if observer.sampling_hint_receipt else None,
            fresh_geometry=len(state.get('fresh_source_reference_frames', ())),
            fresh_geometry_status=state.get('fresh_source_reference_status', ()),
            fresh_material_frames=len(state.get('fresh_source_material_frames', ())),
            fresh_material_status=state.get('fresh_source_material_status', ()),
            pose_refinements=state.get('fresh_source_pose_refinements', ()),
            referents=[dict(number=w.number, born=w.born, kind=w.kind, source=w.source,
                character_id=w.character_id, reference_sha256=w.reference_sha256,
                box=list(w.box()), vx=w.vx, vy=w.vy, velocity_error=w.velocity_error,
                observed_parts=list(w.observed_parts), contact_verified=w.contact_verified, role=w.role,
                visible_fraction=w.visible_fraction, clipped_edges=list(w.clipped_edges),
                pixels=asdict(w.pixels)) for w in seen.values()]))
    return dict(schema='aura.recorded_source_parent_measurement.v1', source=table.source,
        recording=str(recording.resolve()), requested_frames=frames, measured_frames=len(rows),
        source_complete=table.complete, catalog_references=len(catalog.references),
        reference_variant_coverage=[asdict(c) for c in reference_variant_coverage(catalog.references)],
        source_layer_proposals=[asdict(layer) for layer in layers],
        composed_reference_proposals=sum(bool(getattr(r, 'components', ())) for r in catalog.references),
        composed_reference_records=[record for r in catalog.references if (record := composition_record(r)) is not None],
        unavailable=list(catalog.unavailable), searches=searches, frames=rows,
        frames_with_whole_referents=sum(bool(row['referents']) for row in rows),
        frames_with_material_attention=sum(bool(row['material_groups']) for row in rows),
        frames_with_material_naming_candidates=sum(bool(row['material_groups_eligible_for_naming']) for row in rows),
        extent_request_origin='diagnostic_current_material' if request_material_extents else None,
        extent_requests_delivered=sum(len(row['extent_receipts']) for row in rows),
        distinct_whole_ids=len({w['number'] for row in rows for w in row['referents']}),
        frames_with_sampling_hints=sum(bool(row['sampling_hint']) for row in rows),
        frames_with_current_source_material=sum(bool(row['source_frames']) for row in rows),
        duration_s=time.monotonic()-began, model_workers=False, input_delivered=False,
        sample_matched_appearances=sample_matched_appearances,
        classical_boundary=boundary.report(), goal_bound=False, contact_proven=False, native_play=False, demo_ready=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--catalog-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-sequence', type=int, default=0)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--match-every', type=int, default=8)
    parser.add_argument('--sample-matched-appearances', action='store_true')
    parser.add_argument('--request-material-extents', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a fresh output path is required')
    from core.runtime.classical_compute_boundary import classical_compute_boundary

    with classical_compute_boundary():
        result = measure(args.recording, args.source, args.catalog_root,
            start=args.start_sequence, frames=args.frames, match_every=args.match_every,
            sample_matched_appearances=args.sample_matched_appearances,
            request_material_extents=args.request_material_extents)
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('recorded_source_parent_measurement'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.output.parent, source='recorded_source_parent_measurement')
        gateway.write_text(args.output, json.dumps(result, indent=2)+'\n', source='recorded_source_parent_measurement')
    print(json.dumps({key: result[key] for key in ('measured_frames', 'frames_with_sampling_hints',
        'frames_with_current_source_material', 'frames_with_whole_referents',
        'frames_with_material_attention', 'frames_with_material_naming_candidates',
        'extent_requests_delivered',
        'distinct_whole_ids', 'duration_s', 'model_workers', 'input_delivered', 'demo_ready')}))


if __name__ == '__main__':
    main()
