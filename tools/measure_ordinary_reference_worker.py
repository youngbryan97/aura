"""Replay recorded pixels through ordinary observation and its reference worker.

This sends no game inputs and starts no model. It measures the ordinary
asynchronous publication path, without certifying live gameplay.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def measure(args: argparse.Namespace) -> dict:
    from core.cognition.a_guide_to_a_place import Guide, THE_GUIDE
    from core.perception.play_observation_without_holding_the_loop import open_play_observation
    from core.perception.reading_a_program import read_program
    from tools.publish_program_visual_reference import publish_reference_bundle
    from core.perception.what_moves_in_the_picture import WhatMoves
    from tools.measure_source_points_on_recorded_material import _frames, _pixels

    published = await asyncio.to_thread(publish_reference_bundle, args.reference_bundle)
    raw = await asyncio.to_thread((args.reference_bundle/'source.swf').read_bytes)
    read = await asyncio.to_thread(read_program, raw)
    guide = Guide(place='recorded source and pixels')
    guide.take_in_program(read)
    token = THE_GUIDE.set(guide)
    observation = await open_play_observation(factory=WhatMoves)
    frames, following = _frames(args.recording, args.through_sequence, args.history, args.follow_frames)
    previous, surface, measured, lags, pulse, maximum, running = -math.inf, None, [], [], time.monotonic(), 0., True

    async def heartbeat() -> None:
        nonlocal pulse, maximum
        while running:
            await asyncio.sleep(.01)
            now = time.monotonic()
            maximum = max(maximum, now-pulse)
            pulse = now

    ticking = asyncio.create_task(heartbeat())
    began = time.monotonic()
    frame_started = began
    try:
        for event in (*frames, *following):
            capture = event['capture']
            at = capture.get('at')
            if (type(at) not in (int, float) or not math.isfinite(at) or at <= previous
                    or not capture.get('surface') or surface is not None and surface != capture['surface']):
                raise ValueError('increasing source captures on one surface are required')
            # Decode on a worker too: the loop timing includes observation and
            # reference publication, with no synchronous PNG reader in between.
            picture = await asyncio.to_thread(_pixels, args.recording, event)
            if args.source_pace and previous > -math.inf:
                await asyncio.sleep(max(0., at-previous-(time.monotonic()-frame_started)))
            previous, surface, frame_started = at, capture['surface'], time.monotonic()
            events = await observation.see(picture, at)
            worker = observation._references
            measured.append(dict(sequence=event['sequence'], at=at, observed=events is not None,
                scale=float(observation.scale), rgb_sha256=event['raw_sha256'],
                sample_grid=asdict(observation.material_marks.grid) if observation.material_marks is not None else None,
                status=observation.source_reference_status,
                points=[asdict(point) for point in observation.source_reference_points],
                source_frames=[asdict(frame) for frame in observation.source_reference_frames],
                fresh_source_frames=[asdict(frame) for frame in observation.fresh_source_reference_frames],
                fresh_source_status=list(observation.fresh_source_reference_status),
                material_retention_admitted=getattr(observation, 'material_retention_admitted', None),
                unavailable=list(worker.unavailable) if worker else [], refused=list(worker.refused) if worker else []))
            lags.append(time.monotonic()-frame_started)
        return dict(schema='aura.ordinary_reference_worker_replay.v1', index=str(published),
            recording=str(args.recording.resolve()), reference_bundle=str(args.reference_bundle.resolve()),
            frames=measured, seconds=time.monotonic()-began, max_frame_call_s=max(lags, default=None),
            max_loop_pulse_gap_s=maximum, source_pace=args.source_pace,
            task_passed=None, presentation_qualified=False,
            evidence_scope='ordinary asynchronous observation over recorded pixels; no inputs or live gameplay')
    finally:
        running = False
        await observation.close()
        await ticking
        THE_GUIDE.reset(token)


def check_published_points(result: dict, *, source_frames_only: bool = False) -> dict:
    """Measure published points against fresh pixels, without retained tracking state."""
    import numpy as np

    from core.perception.a_visual_reference_in_the_picture import reference_appearances
    from core.perception.visual_references_for_current_sources import registered_reference, acquire_visual_references
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from core.perception.the_pixel_grid_a_mark_uses import SampleGrid
    from core.perception.pixels_consumed_by_the_observer import consumed_pixels
    from tools.measure_source_points_on_recorded_material import _frames, _pixels

    recording = Path(result['recording'])
    reference = registered_reference(Path(result['reference_bundle']))
    references = {reference.key: reference}
    if result.get('index'):
        table = read_flash_visual_assets((Path(result['reference_bundle'])/'source.swf').read_bytes())
        catalog = acquire_visual_references((table,), root=Path(result['index']).parent.parent)
        references.update((item.key, item) for item in catalog.references)
    fields = ('source_frames', 'fresh_source_frames') if source_frames_only else ('points', 'source_frames', 'fresh_source_frames')
    measured = [frame for frame in result['frames'] if any(frame.get(field) for field in fields)]
    if not measured:
        return dict(checks=[], measured=0, maximum_difference=None, unmeasured=0)
    events, _ = _frames(recording, measured[-1]['sequence'], 256)
    by_sequence = {event['sequence']: event for event in events}
    checks, began = [], time.monotonic()
    for frame in measured:
        event = by_sequence[frame['sequence']]
        if frame['at'] != event['capture']['at'] or frame['rgb_sha256'] != event['raw_sha256']:
            raise ValueError('verification must use the same recorded capture bytes')
        picture = _pixels(recording, event)
        cache = {}
        for field in fields:
            for point in frame.get(field, []):
                capture_verified = None
                if field == 'fresh_source_frames':
                    published = point.get('pixels', {})
                    actual = consumed_pixels(picture, frame['at'])
                    capture_verified = (actual is not None and isinstance(published, dict)
                        and isinstance(published.get('shape'), (list, tuple)) and published.get('at') == actual.at
                        and published.get('sha256') == actual.sha256
                        and tuple(published.get('shape', ())) == actual.shape
                        and point.get('point', {}).get('at') == actual.at)
                    if not capture_verified:
                        raise ValueError('fresh reference geometry does not name the consumed RGB capture')
                key = (point['source'], point['character_id'], point['reference_sha256'])
                if key not in references:
                    raise ValueError('a different reference cannot verify this published point')
                reference = references[key]
                if key not in cache:
                    cache[key] = reference_appearances(picture, reference.pixels, scales=tuple(map(float, np.geomspace(.5, 4., 32))))
                poses = cache[key]
                position = (reference.coordinates.project((0., 0.), poses[0], picture, reference.pixels, source=reference.source)
                            if len(poses) == 1 else None)
                grid = SampleGrid(**{k: tuple(v) for k, v in frame['sample_grid'].items()}) if frame.get('sample_grid') else None
                expected = None
                if position is not None:
                    expected = grid.from_image(position) if grid is not None else np.asarray(position)*frame['scale']
                difference = math.dist(point['point']['position'], expected) if expected is not None else None
                checks.append(dict(sequence=frame['sequence'], at=frame['at'], kind=field,
                    number=point['point'].get('number'), character_id=point['character_id'],
                    capture_receipt_verified=capture_verified,
                    fresh_appearances=len(poses), independent_difference=difference,
                    within_working_pixel=difference <= 1. if difference is not None else None))
    differences = [check['independent_difference'] for check in checks if check['independent_difference'] is not None]
    return dict(checks=checks, measured=len(differences), maximum_difference=max(differences, default=None),
                unmeasured=len(checks)-len(differences), seconds=time.monotonic()-began,
                evidence_scope='fresh zero-angle raster correspondence; missing or competing appearances excluded')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--reference-bundle', type=Path, required=True)
    parser.add_argument('--through-sequence', type=int, required=True)
    parser.add_argument('--history', type=int, default=20)
    parser.add_argument('--follow-frames', type=int, default=40)
    parser.add_argument('--source-pace', action='store_true')
    parser.add_argument('--check-published', action='store_true', help='check fresh raster correspondences after the replay ends')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not 2 <= args.history <= 256 or not 0 <= args.follow_frames <= 256
            or args.through_sequence < 0 or args.out.exists()):
        parser.error('use bounded frame counts and a fresh output')
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    result = asyncio.run(measure(args))
    if args.check_published:
        result['independent_pixels'] = check_published_points(result)
    with local_internal_governed_scope('ordinary_reference_worker_replay'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='ordinary_reference_worker_replay')
        if not gateway.write_bytes_if_absent(args.out, json.dumps(result, indent=2).encode(),
                                            source='ordinary_reference_worker_replay'):
            raise ValueError('another measurement owns this output')
    print(json.dumps(dict(frames=len(result['frames']),
        frames_with_points=sum(bool(frame['points']) for frame in result['frames']),
        seconds=result['seconds'], max_loop_pulse_gap_s=result['max_loop_pulse_gap_s'],
        task_passed=None, presentation_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
