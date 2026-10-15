"""Replay recorded play pixels through current eyes, without a browser or model.

This diagnoses perception. It does not recreate the actor's private decisions,
establish control identity or qualify gameplay. Input events remain transport
receipts, and drawing scenes omitted by the recorder cannot be reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def point_diagnostics():
    """Observe the reference handoff without changing any returned evidence."""
    from core.perception import a_reference_kept_by_visible_support as references
    from core.perception.position_from_the_same_points import _owned_marks

    original, records = references.position_from_points, []

    def measured(thing, blob, motion, at):
        previous = thing.point_pose
        current = _owned_marks(thing, blob, motion, at)
        before = set(previous.anchors) if previous is not None else set()
        shared = before & current.keys() if current is not None else set()
        result = original(thing, blob, motion, at)
        records.append(dict(number=thing.number, at=at, dt=at-thing.seen,
            previous_anchors=len(before), current_owned=len(current) if current else 0, shared_anchors=len(shared),
            supplied_support=motion.support if motion is not None else 0,
            continued=result is not None, previous_reference=previous.reference_at if previous else None,
            current_reference=thing.point_pose.reference_at if thing.point_pose else None))
        return result

    references.position_from_points = measured
    try:
        yield records
    finally:
        references.position_from_points = original


def recorded_picture(directory: Path, event: dict) -> np.ndarray:
    name = event['file']
    if Path(name).name != name:
        raise ValueError('a frame must be inside its recording directory')
    with Image.open(directory / name) as image:
        picture = np.array(image, dtype=np.uint8)
    if (list(picture.shape) != event['shape'] or event['dtype'] != 'uint8'
            or hashlib.sha256(picture.tobytes()).hexdigest() != event['raw_sha256']):
        raise ValueError('recorded pixels no longer match their receipt')
    return picture


def control_histories(events: list[dict]) -> dict[int, list[tuple]]:
    """Only actual actor snapshots can label an input as an independent trial."""
    streams, stream = {}, 0
    for event in events:
        if event['kind'] == 'stream_started':
            stream += 1
            streams[stream] = {}
        elif event['kind'] == 'control_observation' and stream:
            since, key, trying = event['held_since'], event['held'], event['trying']
            if since is not None:
                if (isinstance(since, bool) or not isinstance(since, (int, float)) or not math.isfinite(since)
                        or not isinstance(key, str) or type(trying) is not bool):
                    raise ValueError('input ownership snapshot is unmeasured')
                prior = streams[stream].setdefault(since, (key, trying))
                if prior != (key, trying):
                    raise ValueError('conflicting input ownership snapshots')
    if not streams or any(not history for history in streams.values()):
        raise ValueError('control replay requires actual actor ownership snapshots in every stream')
    if any(e['kind'] == 'input' and e['method'] not in ('up', 'down') for e in events):
        raise ValueError('this control replay does not reconstruct combined or pointer input')
    return {n: [(at, key, trying) for at, (key, trying) in sorted(history.items())] for n, history in streams.items()}


def replay(directory: Path, limit: int = 1000, *, replay_control: bool = False) -> dict:
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime
    from core.perception.translation_from_visible_support import motion_status
    from core.perception.what_moves_in_the_picture import WhatMoves, prepare_play_vision

    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'aura.pixel_input_recording.v1':
        raise ValueError('unknown recording schema')
    if replay_control and not manifest.get('complete'):
        raise ValueError('control replay requires a complete input recording')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    sequences = [e['sequence'] for e in events]
    if any(b <= a for a, b in zip(sequences, sequences[1:], strict=False)):
        raise ValueError('recording event order is invalid')
    histories = control_histories(events) if replay_control else None
    prepare_play_vision()
    moves, hers, streams, frames, statuses, inputs = None, None, 0, [], Counter(), []
    for event in events:
        if event['kind'] == 'stream_started':
            moves = WhatMoves(kinds=moves.kinds if moves else None)
            streams += 1
            if histories is not None:
                from core.agency.which_one_answers_to_her import WhichIsHers

                hers = WhichIsHers()
                for when, key, trying in histories[streams]:
                    hers.holding(key, when, trying=trying)
        elif event['kind'] == 'input':
            inputs.append(event)
        elif event['kind'] == 'frame':
            if len(frames) >= limit:
                break
            if moves is None:
                raise ValueError('a frame has no stream start')
            picture = recorded_picture(directory, event)
            capture = dict(event['capture'])
            at = (CanvasFrameTime(**capture) if 'surface' in capture and 'sequence' in capture
                  else float(capture['at']))
            began = time.perf_counter()
            happened = moves.see(picture, at)
            if hers is not None:
                hers.saw(moves, happened, at)
            duration = time.perf_counter() - began
            things = []
            for thing in moves.things.values():
                status = motion_status(thing, float(at))
                statuses[status] += 1
                pose = thing.point_pose
                things.append(dict(number=thing.number, kind=thing.kind, born=thing.born, seen=thing.seen,
                    box=list(thing.box()), position=[thing.x, thing.y], velocity=[thing.vx, thing.vy],
                    status=status, evidence=thing.motion_evidence, measured_at=thing.motion_measured_at,
                    material_position=list(thing.motion_position) if thing.motion_position is not None else None,
                    moved=thing.moved, path_length=len(thing.path),
                    owned_marks=len(pose.anchors) if pose is not None else 0))
            frames.append(dict(sequence=event['sequence'], file=event['file'], at=float(at), stream=streams,
                drawing_scene_omitted=event.get('drawing_scene_present', False), duration_s=duration,
                working_shape=list(moves.shape), things=things,
                events=[{k: v for k, v in e.items() if k in ('what', 'thing', 'kind', 'at')} for e in happened],
                control_selected=hers.number if hers is not None else None,
                control_evidence=[dict(number=int(n), ratio=list(s.ratio()),
                    pushes={key: len(values) for key, values in s.pushes().items()})
                    for n, s in hers._by_thing.items()] if hers is not None else None,
                points=len(moves.points.points), retained_points=sum(p.pictures >= 2 for p in moves.points.points)))
    return dict(scope='pixel-only replay on recorded controls; changed policy and gameplay outcomes are not replayed',
                recording=manifest, replayed_frames=len(frames), streams=streams,
                replayed_actor_control_flags=replay_control,
                statuses=dict(statuses), frames=frames, inputs=inputs, presentation_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--replay-control', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.limit <= 10000:
        parser.error('use a fresh output and a frame limit between one and ten thousand')
    with point_diagnostics() as points:
        result = replay(args.recording, args.limit, replay_control=args.replay_control)
        result['point_diagnostics'] = points
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('play_pixel_replay'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='play_pixel_replay')
        gateway.write_text(args.out, json.dumps(result, indent=2), source='play_pixel_replay')
    print(json.dumps({k: v for k, v in result.items() if k not in ('frames', 'inputs', 'recording', 'point_diagnostics')}))


if __name__ == '__main__':
    main()
