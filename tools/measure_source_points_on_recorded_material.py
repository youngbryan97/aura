"""Measure source-point attachment through recorded pixels and the ordinary tracker.

This replay sends no input, reads no game state and loads no model. Process
completion records a measurement; qualification and gameplay remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _contained(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('a receipt must name a contained relative file')
    path = root/name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('a receipt file must stay inside its recording or reference bundle')
    return path


def _reference(bundle: Path):
    from core.perception.a_visual_reference_in_the_picture import _digest
    from core.perception.coordinates_in_a_visual_reference import SourceImageCoordinates
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from tools.export_program_visual_reference import _inventory

    receipt = bundle/'receipt.json'
    if receipt.stat().st_size > 1_000_000:
        raise ValueError('reference receipt exceeds the byte budget')
    record = json.loads(receipt.read_text())
    if (record.get('schema') != 'aura.program_visual_reference.v1'
            or record.get('status') != 'rendered' or len(record.get('images', [])) != 1):
        raise ValueError('one successfully rendered static reference is required')
    program = bundle/'source.swf'
    if program.stat().st_size > 64*1024*1024:
        raise ValueError('source exceeds the byte budget')
    table = read_flash_visual_assets(program.read_bytes())
    declaration = record['declaration']
    asset = table.named(declaration['name'])
    if (table.source != record['source'] or asset is None or asset.character_id != declaration['character_id']
            or asset.definition_offset != declaration['definition_offset']):
        raise ValueError('source declaration no longer matches the reference receipt')
    if _inventory(bundle/'rendered', asset.character_id) != record['images']:
        raise ValueError('rendered inventory no longer matches the declared reference')
    image = record['images'][0]
    path = _contained(bundle/'rendered', image['path'])
    if path.stat().st_size > 8_000_000 or hashlib.sha256(path.read_bytes()).hexdigest() != image['file_sha256']:
        raise ValueError('reference image bytes changed')
    with Image.open(path) as opened:
        if min(opened.size) < 16 or max(opened.size) > 512:
            raise ValueError('reference dimensions exceed the budget')
        pixels = np.array(opened.convert('RGBA'))
    registration = record['registration']
    fields = {**registration['coordinates'], 'size': tuple(registration['coordinates']['size'])}
    coordinates = SourceImageCoordinates(**fields)
    if (coordinates.source != table.source or coordinates.reference_sha256 != _digest(pixels)
            or coordinates.size != (pixels.shape[1], pixels.shape[0])):
        raise ValueError('coordinate registration no longer matches the source and reference pixels')
    svg = _contained(bundle, registration['svg'])
    if svg.stat().st_size > 2_000_000 or hashlib.sha256(svg.read_bytes()).hexdigest() != coordinates.svg_sha256:
        raise ValueError('source coordinate declaration bytes changed')
    return coordinates, pixels


def _frames(recording: Path, through: int, history: int, follow: int = 0) -> tuple[list[dict], list[dict]]:
    path = recording/'events.jsonl'
    if path.stat().st_size > 64*1024*1024:
        raise ValueError('recording index exceeds the byte budget')
    frames, following, previous = deque(maxlen=history), [], -1
    with path.open() as stream:
        for line in stream:
            if len(line) > 1_000_000:
                raise ValueError('recording event exceeds the byte budget')
            event = json.loads(line)
            sequence = event.get('sequence')
            if type(sequence) is not int or sequence <= previous:
                raise ValueError('recording sequences must increase')
            previous = sequence
            if sequence > through:
                if not follow:
                    break
                if event.get('kind') == 'frame':
                    following.append(event)
                    if len(following) >= follow:
                        break
                continue
            if event.get('kind') == 'frame':
                frames.append(event)
    if not frames or frames[-1]['sequence'] != through:
        raise ValueError('the requested sequence must identify a recorded frame')
    return list(frames), following


def _pixels(recording: Path, event: dict) -> np.ndarray:
    path = _contained(recording, event['file'])
    with Image.open(path) as image:
        if min(image.size) < 16 or max(image.size) > 4096:
            raise ValueError('recorded picture exceeds the pixel budget')
        picture = np.array(image.convert('RGB'))
    if (list(picture.shape) != event['shape'] or event['dtype'] != 'uint8'
            or hashlib.sha256(picture.tobytes()).hexdigest() != event['raw_sha256']):
        raise ValueError('recorded RGB bytes changed')
    return picture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--through-sequence', type=int, required=True)
    parser.add_argument('--history', type=int, default=20)
    parser.add_argument('--follow-frames', type=int, default=0)
    parser.add_argument('--reference-bundle', type=Path, required=True)
    parser.add_argument('--source-point', type=float, nargs=2, default=(0., 0.))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 2 <= args.history <= 256 or not 0 <= args.follow_frames <= 256
            or args.through_sequence < 0 or args.out.exists()
            or not all(math.isfinite(v) for v in args.source_point)):
        parser.error('use a fresh output, finite source point and bounded frame history')
    from core.governance_context import local_internal_governed_scope
    from core.perception.a_visual_reference_in_the_picture import reference_appearances
    from core.perception.what_moves_in_the_picture import WhatMoves, prepare_play_vision
    from core.runtime.file_write_gateway import get_file_write_gateway

    coordinates, reference = _reference(args.reference_bundle)
    frames, following = _frames(args.recording, args.through_sequence, args.history, args.follow_frames)
    prepare_play_vision()
    observer, began = WhatMoves(), time.perf_counter()
    previous_at, surface = -math.inf, None
    for event in frames:
        capture = event['capture']
        if (type(capture.get('at')) not in (int, float) or not math.isfinite(capture['at'])
                or capture['at'] <= previous_at or not capture.get('surface')
                or surface is not None and capture['surface'] != surface):
            raise ValueError('replay needs increasing captures from one measured surface')
        previous_at, surface = capture['at'], capture['surface']
        picture = _pixels(args.recording, event)
        observer.see(picture, previous_at)
    scales = tuple(map(float, np.geomspace(.5, 4., 32)))
    appearances = reference_appearances(picture, reference, scales=scales)
    results, bindings = [], []
    for appearance in appearances:
        result = coordinates.on_track(tuple(args.source_point), appearance, picture, reference, observer,
                                      previous_at, source=coordinates.source)
        current = result.binding.current(observer, previous_at, source=coordinates.source) if result.binding else None
        results.append(dict(reason=result.reason, owner_support=result.owner_support,
                            matched_points=len(appearance.source_points),
                            current=asdict(current) if current else None))
        bindings.append(result.binding)
    tracks = [dict(number=t.number, born=t.born, seen=t.seen, kind=t.kind,
                   point_basis_at=getattr(t.point_pose, 'reference_at', None),
                   point_pose_at=getattr(t.point_pose, 'at', None),
                   material_marks=len(getattr(t.point_pose, 'visible_marks', {}))) for t in observer.things.values()]
    tracked, attachment_at = [], previous_at
    for event in following:
        capture = event['capture']
        if (type(capture.get('at')) not in (int, float) or not math.isfinite(capture['at'])
                or capture['at'] <= previous_at or capture.get('surface') != surface):
            tracked.append(dict(sequence=event['sequence'], reason='capture_scope_changed', current=None))
            break
        previous_at = capture['at']
        picture = _pixels(args.recording, event)
        observer.see(picture, previous_at)
        fresh = reference_appearances(picture, reference, scales=scales)
        fresh_point = (coordinates.project(tuple(args.source_point), fresh[0], picture, reference,
                                           source=coordinates.source) if len(fresh) == 1 else None)
        estimates = []
        for binding in bindings:
            sample = binding.inspect(observer, previous_at, source=coordinates.source) if binding else None
            current = sample.current if sample else None
            difference = math.dist(current.position, np.asarray(fresh_point)*observer.scale) if current and fresh_point else None
            estimates.append(dict(reason=sample.reason if sample else 'initial_attachment_refused',
                                  current=asdict(current) if current else None, independent_difference=difference,
                                  within_working_pixel=difference <= 1. if difference is not None else None))
        tracked.append(dict(sequence=event['sequence'], at=previous_at, independent_appearances=len(fresh),
                            estimates=estimates))
    record = dict(schema='aura.recorded_source_point_material.v1', frames=len(frames), source=coordinates.source,
        through_sequence=args.through_sequence, at=attachment_at, followed_until=previous_at, scale=observer.scale, tracks=tracks,
        source_point=args.source_point, results=results, following=tracked, seconds=time.perf_counter()-began,
        task_passed=None, presentation_qualified=False,
        evidence_scope='recorded pixel replay and current material attachment; no role, action or gameplay proof')
    with local_internal_governed_scope('recorded_source_point_material'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='recorded_source_point_material')
        if not gateway.write_bytes_if_absent(args.out, json.dumps(record, indent=2).encode(),
                                            source='recorded_source_point_material'):
            raise ValueError('another replay owns this output')
    print(json.dumps(dict(appearances=len(appearances), reasons=[r['reason'] for r in results],
                         seconds=record['seconds'], task_passed=None, presentation_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
