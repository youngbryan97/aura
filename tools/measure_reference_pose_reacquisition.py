"""Measure fresh reference correspondence over recorded captures, without input."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(args: argparse.Namespace) -> dict:
    from core.perception.a_visual_reference_in_the_picture import reference_appearances
    from core.perception.reference_material_near_a_measured_pose import near_reference_appearances
    from core.perception.visual_references_for_current_sources import registered_reference
    from tools.measure_source_points_on_recorded_material import _frames, _pixels

    reference = registered_reference(args.reference_bundle)
    history, following = _frames(args.recording, args.through_sequence, args.history, args.follow_frames)
    scales = tuple(sorted({1., *map(float, np.geomspace(.25, 4., 31))}))
    previous, at_before, surface, phase, frames = (), -math.inf, None, 0, []
    began = time.monotonic()
    for event in (*history, *following):
        capture = event['capture']
        at = capture['at']
        if (type(at) not in (int, float) or not math.isfinite(at) or at <= at_before
                or not capture.get('surface') or surface is not None and capture['surface'] != surface):
            raise ValueError('the recording needs increasing captures on one surface')
        at_before, surface = at, capture['surface']
        picture = _pixels(args.recording, event)
        start = time.perf_counter()
        poses = near_reference_appearances(picture, reference.pixels, previous) if previous else ()
        near_s = time.perf_counter()-start
        from_near, broad_s = bool(poses), 0.
        if not poses:
            angle = (phase*math.pi/8+math.pi) % (2*math.pi)-math.pi
            phase = (phase+1) % 16
            start = time.perf_counter()
            poses = reference_appearances(picture, reference.pixels, scales=scales, angles=(angle,))
            broad_s = time.perf_counter()-start
        if poses:
            previous = poses
        points = [reference.coordinates.project((0., 0.), pose, picture, reference.pixels,
                                               source=reference.source) for pose in poses]
        frames.append(dict(sequence=event['sequence'], at=at, rgb_sha256=event['raw_sha256'],
            appearances=len(poses), source_points=points, from_near=from_near,
            near_s=near_s, broad_s=broad_s,
            correspondence_support=[len(pose.source_points) for pose in poses]))
    return dict(schema='aura.reference_pose_reacquisition.v1', source=reference.source,
        reference_sha256=reference.coordinates.reference_sha256, frames=frames,
        recording=str(args.recording.resolve()), reference_bundle=str(args.reference_bundle.resolve()),
        seconds=time.monotonic()-began, task_passed=None, presentation_qualified=False,
        evidence_scope='fresh raster reference fits on recorded captures; no temporal object identity or game input')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--reference-bundle', type=Path, required=True)
    parser.add_argument('--through-sequence', type=int, required=True)
    parser.add_argument('--history', type=int, default=20)
    parser.add_argument('--follow-frames', type=int, default=80)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 2 <= args.history <= 256 or not 0 <= args.follow_frames <= 256 or args.out.exists():
        parser.error('use bounded frame counts and a fresh output')
    result = measure(args)
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('reference_pose_reacquisition'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='reference_pose_reacquisition')
        if not gateway.write_bytes_if_absent(args.out, json.dumps(result, indent=2).encode(),
                                            source='reference_pose_reacquisition'):
            raise ValueError('another measurement owns this output')
    print(json.dumps(dict(frames=len(result['frames']),
        frames_with_reference=sum(bool(row['appearances']) for row in result['frames']),
        near_frames=sum(row['from_near'] for row in result['frames']), seconds=result['seconds'],
        task_passed=None, presentation_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
