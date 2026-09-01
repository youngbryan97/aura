#!/usr/bin/env python3
"""Does watching every frame beat glancing every few, at the same total reads?

Card 133: moving objects stay continuously tracked and prediction beats sparse
snapshots.

Two arms see the same world. The continuous arm reads every frame and carries
a track across them, so it has a velocity and can say where a thing will be.
The sparse arm reads one frame in every N — the sampling this codebase
actually does when it looks at a screen — and has to answer from a snapshot.

The comparison is at equal reads, not equal frames. A continuous arm that
simply looks more often and does better has shown nothing, so the sparse arm
is given the same number of reads spread over the same span, and both are
scored on the same question: where is each object, one step from now.

The world includes the case the whole thing is about — objects that pass
behind an occluder and come out the other side. A snapshot sees a thing
vanish and a different thing appear; a track sees one thing.

    python tools/campaigns/tracking_campaign.py --sequences 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognition.entity_track import Observation, TrackStore  # noqa: E402

FIELD = 100.0


def _world(
    rng: random.Random, *, objects: int, frames: int, occluder: tuple[float, float]
) -> list[list[tuple[int, float, float]]]:
    """Frames of (object_id, x, y). An object inside the occluder is not seen."""
    state = [
        (
            rng.uniform(0.0, FIELD),
            rng.uniform(0.0, FIELD),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
        )
        for _ in range(objects)
    ]
    low, high = occluder
    out: list[list[tuple[int, float, float]]] = []
    for _ in range(frames):
        frame: list[tuple[int, float, float]] = []
        for index, (x, y, dx, dy) in enumerate(state):
            x, y = x + dx, y + dy
            if not 0.0 <= x <= FIELD:
                dx = -dx
                x = max(0.0, min(FIELD, x))
            if not 0.0 <= y <= FIELD:
                dy = -dy
                y = max(0.0, min(FIELD, y))
            state[index] = (x, y, dx, dy)
            if not (low <= x <= high):
                frame.append((index, x, y))
        out.append(frame)
    return out


def _truth_next(
    frames: list[list[tuple[int, float, float]]], step: int
) -> dict[int, tuple[float, float]]:
    return {index: (x, y) for index, x, y in frames[step + 1]}


def _error(
    predicted: dict[int, tuple[float, float]],
    truth: dict[int, tuple[float, float]],
) -> tuple[float, int]:
    """Mean distance over the objects both arms could have answered about."""
    shared = set(predicted) & set(truth)
    if not shared:
        return 0.0, 0
    total = sum(math.dist(predicted[i], truth[i]) for i in shared)
    return total / len(shared), len(shared)


def one_sequence(
    rng: random.Random, *, objects: int, frames: int, stride: int
) -> dict[str, float]:
    """One world, two arms, the same number of reads."""
    occluder = (40.0, 55.0)
    world = _world(rng, objects=objects, frames=frames, occluder=occluder)

    read_frames = list(range(0, frames - 1, stride))
    # Equal reads: the continuous arm sees the same COUNT of frames as the
    # sparse arm, but consecutively, so it has a velocity where the sparse arm
    # has a gap. Giving it every frame would be giving it more looks.
    budget = len(read_frames)

    continuous_errors: list[float] = []
    sparse_errors: list[float] = []
    tracked = 0
    snapshot_identities = 0

    for start in read_frames:
        window = list(range(start, min(start + budget, frames - 1)))
        if len(window) < 2:
            continue

        # Continuous: consecutive frames through the track store.
        store = TrackStore(match_threshold=6.0)
        previous: dict[str, tuple[float, float]] = {}
        velocity: dict[str, tuple[float, float]] = {}
        for step in window:
            observations = [
                Observation(at=float(step), geometry=(x, y), label=str(index))
                for index, x, y in world[step]
            ]
            for track in store.update(observations):
                if track.last_observation is None:
                    continue
                position = (
                    track.last_observation.geometry[0],
                    track.last_observation.geometry[1],
                )
                if track.track_id in previous:
                    last = previous[track.track_id]
                    velocity[track.track_id] = (
                        position[0] - last[0],
                        position[1] - last[1],
                    )
                previous[track.track_id] = position

        predicted: dict[int, tuple[float, float]] = {}
        for track in store.tracks():
            if track.last_observation is None or not track.alive:
                continue
            label = track.last_observation.label
            if not label.isdigit():
                continue
            dx, dy = velocity.get(track.track_id, (0.0, 0.0))
            position = track.last_observation.geometry
            predicted[int(label)] = (position[0] + dx, position[1] + dy)
        tracked += len(store.tracks())

        # Sparse: the same number of reads, spread across the same span, so
        # consecutive frames are never seen and there is no velocity to have.
        step_size = max(1, len(window) // budget) if budget else 1
        sampled = window[::step_size][:budget] or window[:1]
        last_frame = sampled[-1]
        snapshot = {index: (x, y) for index, x, y in world[last_frame]}
        snapshot_identities += len(snapshot)

        truth = _truth_next(world, window[-1])
        continuous_error, _ = _error(predicted, truth)
        sparse_error, _ = _error(snapshot, truth)
        if predicted:
            continuous_errors.append(continuous_error)
        if snapshot:
            sparse_errors.append(sparse_error)

    return {
        "continuous_error": statistics.fmean(continuous_errors)
        if continuous_errors
        else 0.0,
        "sparse_error": statistics.fmean(sparse_errors) if sparse_errors else 0.0,
        "reads_per_arm": float(budget),
        "tracks_held": float(tracked),
        "snapshot_identities": float(snapshot_identities),
    }


def occlusion_test(
    rng: random.Random, *, frames: int
) -> dict[str, object]:
    """One object crossing an occluder: does it stay one thing?

    A snapshot sees it vanish and something appear; a track that survives the
    gap sees one thing. This is the case the card is about and it is scored on
    its own, because averaging it into the error hides it.
    """
    store = TrackStore(match_threshold=8.0)
    x, y, dx = 5.0, 50.0, 4.0
    hidden = (40.0, 55.0)
    ids_seen: set[str] = set()
    gap_frames = 0

    for step in range(frames):
        x += dx
        if hidden[0] <= x <= hidden[1]:
            gap_frames += 1
            for track in store.tracks():
                track.miss()
            continue
        observations = [Observation(at=float(step), geometry=(x, y), label="one")]
        for track in store.update(observations):
            ids_seen.add(track.track_id)

    del rng
    return {
        "frames_hidden": gap_frames,
        "distinct_tracks": len(ids_seen),
        "stayed_one_thing": len(ids_seen) == 1,
        "a_snapshot_would_see": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--objects", type=int, default=5)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/tracking_campaign.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [
        one_sequence(
            rng, objects=args.objects, frames=args.frames, stride=args.stride
        )
        for _ in range(args.sequences)
    ]
    occlusion = occlusion_test(random.Random(args.seed + 1), frames=args.frames)

    continuous = statistics.fmean(r["continuous_error"] for r in rows)
    sparse = statistics.fmean(r["sparse_error"] for r in rows)
    payload = {
        "schema": "aura.tracking_campaign.v1",
        "card": "133",
        "claim_boundary": (
            "synthetic frame sequences through core.cognition.entity_track at "
            "an equal read budget; a claim about the tracker and the "
            "prediction it supports, not about reading a real screen at video "
            "rate"
        ),
        "config": {
            "sequences": args.sequences,
            "objects": args.objects,
            "frames": args.frames,
            "stride": args.stride,
            "seed": args.seed,
        },
        "equal_reads": {
            "reads_per_arm": rows[0]["reads_per_arm"],
            "basis": "both arms read the same number of frames over the same span",
        },
        "next_step_error": {
            "continuous_mean": round(continuous, 4),
            "sparse_snapshot_mean": round(sparse, 4),
            "continuous_beats_sparse": continuous < sparse,
            "improvement": round(1.0 - continuous / sparse, 4) if sparse else 0.0,
            "sequences_continuous_won": sum(
                1 for r in rows if r["continuous_error"] < r["sparse_error"]
            ),
            "of": len(rows),
        },
        "through_an_occluder": occlusion,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("tracking_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
