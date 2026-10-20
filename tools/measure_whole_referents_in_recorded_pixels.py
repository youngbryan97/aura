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


def measure(recording: Path, source: Path, catalog_root: Path, *, start: int, frames: int, match_every: int) -> dict:
    from core.runtime.classical_compute_boundary import current_classical_boundary
    from core.runtime.model_worker_policy import model_workers_allowed

    boundary = current_classical_boundary()
    if boundary is None or model_workers_allowed():
        raise RuntimeError('recorded-pixel measurement requires classical imports and model workers prohibited')
    boundary.require_intact()
    from core.perception.a_material_frame_kept_for_its_owner import keep_material_frame
    from core.perception.source_references_while_play_advances import _matched
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from core.perception.visual_references_for_current_sources import acquire_visual_references
    from core.perception.what_moves_in_the_picture import WhatMoves
    from core.perception.whole_referents_from_source_material import WholeReferentsInView

    if not 1 <= frames <= 100 or not 1 <= match_every <= 100 or start < 0:
        raise ValueError('replay needs bounded frame and match counts')
    if not 8 <= source.stat().st_size <= 40*1024*1024:
        raise ValueError('source bytes exceed their bound')
    raw = source.read_bytes()
    table = read_flash_visual_assets(raw)
    catalog = acquire_visual_references((table,), root=catalog_root)
    observer, wholes = WhatMoves(), WholeReferentsInView()
    bindings, searches, rows = (), [], []
    began = time.monotonic()
    for index, (event, picture) in enumerate(_frames(recording, start, frames)):
        at = float(event['capture']['at'])
        observer.see(picture, at)
        if index % match_every == 0:
            captured = keep_material_frame(observer, picture, at)
            if captured is None:
                raise ValueError('observer did not retain this exact source capture')
            found = []
            for reference in catalog.references:
                result = _matched(reference, captured)
                searches.append(dict(sequence=event['sequence'], source=reference.source,
                    character_id=reference.declaration.character_id, appearances=len(result.appearances),
                    object_origins=len(result.bindings), material_frames=len(result.frames), refused=list(result.refused)))
                found.extend((reference, binding) for binding in result.frames)
            bindings = (*bindings, *found)[-64:]
        current, kept = [], []
        for reference, binding in bindings:
            point = binding.current(observer, at, source=reference.source)
            if point is not None:
                current.append((reference, binding, point))
                kept.append((reference, binding))
        bindings = tuple(kept)
        started = time.perf_counter()
        seen = wholes.observe(observer, picture, at, tuple(current))
        elapsed = time.perf_counter()-started
        rows.append(dict(sequence=event['sequence'], file=event['file'], at=at,
            picture_sha256=observer.pixels_consumed.sha256, material_marks=len(observer.material_marks.points),
            source_frames=len(current), producer_seconds=elapsed, readings=list(wholes.readings),
            referents=[dict(number=w.number, born=w.born, kind=w.kind, source=w.source,
                character_id=w.character_id, reference_sha256=w.reference_sha256,
                box=list(w.box()), vx=w.vx, vy=w.vy, velocity_error=w.velocity_error,
                observed_parts=list(w.observed_parts), contact_verified=w.contact_verified, role=w.role,
                pixels=asdict(w.pixels)) for w in seen.values()]))
    return dict(schema='aura.recorded_source_parent_measurement.v1', source=table.source,
        recording=str(recording.resolve()), requested_frames=frames, measured_frames=len(rows),
        source_complete=table.complete, catalog_references=len(catalog.references),
        unavailable=list(catalog.unavailable), searches=searches, frames=rows,
        frames_with_whole_referents=sum(bool(row['referents']) for row in rows),
        distinct_whole_ids=len({w['number'] for row in rows for w in row['referents']}),
        duration_s=time.monotonic()-began, model_workers=False, input_delivered=False,
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a fresh output path is required')
    from core.runtime.classical_compute_boundary import classical_compute_boundary

    with classical_compute_boundary():
        result = measure(args.recording, args.source, args.catalog_root,
            start=args.start_sequence, frames=args.frames, match_every=args.match_every)
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('recorded_source_parent_measurement'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.output.parent, source='recorded_source_parent_measurement')
        gateway.write_text(args.output, json.dumps(result, indent=2)+'\n', source='recorded_source_parent_measurement')
    print(json.dumps({key: result[key] for key in ('measured_frames', 'frames_with_whole_referents',
        'distinct_whole_ids', 'duration_s', 'model_workers', 'input_delivered', 'demo_ready')}))


if __name__ == '__main__':
    main()
