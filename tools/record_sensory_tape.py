#!/usr/bin/env python3
"""Cut a tape of what her senses actually carried, for the battery to replay.

Perception's own domain is measured on a stream the perception stack never
produced. A condition writes one percept of a declared kind with a salience
drawn from the run's own generator, and nothing in it came from a sensor.

This reads the five real channels — screen, operating system, audio, host
telemetry, user events — once a frame, and writes what they held. The battery
plays it back with `--sensory-tape`, at the turn index rather than at a wall
clock, so both arms of a paired trial see the same frame of the same world and
the difference between them stays the intervention.

    python tools/record_sensory_tape.py --frames 240 --every 0.5
    python tools/record_sensory_tape.py --frames 60 --only screen,host

Nothing here takes a screenshot or asks for a grant. It reads what perception
already captured, because a battery that captures the host is changing the
world it is about to measure. On a machine with no live instance most channels
open with nothing on them, and the tape says so rather than passing that off as
perception: `carried` names the channels that put a percept on the tape, and it
is the only claim a run is entitled to make.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument(
        "--every", type=float, default=0.5, help="seconds between frames"
    )
    parser.add_argument(
        "--only", default="", help="comma-separated channel names; default is all"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "artifacts" / "subject_core" / "sensory_tape.json"
    )
    args = parser.parse_args()

    from core.subject.perception_replay import record_channels
    from core.subject.sensory_channels import channel_names

    only = tuple(n.strip() for n in args.only.split(",") if n.strip()) or None
    if only:
        unknown = sorted(set(only) - set(channel_names()))
        if unknown:
            raise SystemExit(f"no such channel: {unknown}; known: {list(channel_names())}")

    print(f"recording {args.frames} frames every {args.every}s from {only or channel_names()}")
    started = time.monotonic()
    tape = record_channels(
        args.frames, wait=lambda: time.sleep(max(0.0, args.every)), only=only
    )
    coverage = tape.coverage()
    coverage["recorded_over_seconds"] = round(time.monotonic() - started, 2)
    tape.notes["recorded_over_seconds"] = coverage["recorded_over_seconds"]

    from core.governance_context import local_internal_governed_scope
    from core.subject.archive import write_json

    with local_internal_governed_scope("subject_core.record_sensory_tape"):
        write_json(args.out.parent, args.out.name, tape.as_dict())

    print(f"\n{'channel':18} {'carried':>8} {'silent':>8} {'refused':>8}")
    for name, row in coverage["declared"].items():
        print(f"{name:18} {row['live']:>8} {row['silent']:>8} {row['refused']:>8}")
        if row.get("reason"):
            print(f"{'':18}   {row['reason']}")
    print()
    print(f"{coverage['percepts']} percepts over {coverage['frames']} frames")
    print(f"carried by: {', '.join(coverage['carried']) or 'nothing'}")
    if not coverage["carried"]:
        print(
            "\nNo channel put a percept on this tape. Replaying it gives the "
            "organism no perception at all, which is a worse experiment than "
            "the scripted one — record while a live instance is perceiving."
        )
        print(f"written anyway to {args.out}")
        return 1
    print(f"\nwritten to {args.out}")
    print(json.dumps({"carried": coverage["carried"], "frames": coverage["frames"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
