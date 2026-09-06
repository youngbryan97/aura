#!/usr/bin/env python3
"""Run the whole-mind activity prediction benchmark on a recording.

ZAPBench's task, on Aura: given C frames of context predict the next H, MAE per
step, one condition held out of training entirely. The arms differ in which
other cells each cell is allowed to see, and the one that matters is the
comparison between the real connectome and a degree-preserving rewiring of it.

    python tools/run_zapbench.py --data artifacts/connectome
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "artifacts" / "connectome")
    parser.add_argument("--contexts", default="4,256")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--signal", default="calcium", choices=("calcium", "spikes"))
    parser.add_argument("--hold-out", default="", help="condition kept out of training")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import numpy as np

    from core.connectome.activity import ActivityTrace
    from core.connectome.volume import VolumeReconstructor
    from core.connectome.zapbench import BenchmarkConfig, run_benchmark

    manifest_path = args.data / "activity_manifest.json"
    matrix_path = args.data / "activity.npz"
    if not manifest_path.exists() or not matrix_path.exists():
        print(f"no recording under {args.data}; run tools/record_connectome_activity.py first")
        return 2

    manifest = json.loads(manifest_path.read_text())
    spikes = np.load(matrix_path)["spikes"]
    trace = ActivityTrace(
        uids=tuple(manifest["uids"]),
        conditions=tuple(manifest["conditions"]),
        spikes=[list(map(float, row)) for row in spikes],
        frame_seconds=float(manifest.get("frame_seconds", 0.914)),
    )
    print(f"recording: {trace.n_frames} frames x {trace.n_cells} cells", flush=True)

    started = time.time()
    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    snapshot = reconstructor.build()
    print(f"connectome: {snapshot.cell_count()} cells in {time.time() - started:.1f}s", flush=True)

    held_out = args.hold_out
    if not held_out:
        counts: dict[str, int] = {}
        for condition in trace.conditions:
            counts[condition] = counts.get(condition, 0) + 1
        # Hold out the smallest condition that still has enough frames to score,
        # which keeps the training set as large as possible while still testing
        # generalisation to a condition never trained on.
        eligible = [
            name
            for name, count in sorted(counts.items(), key=lambda kv: kv[1])
            if count >= args.horizon + 8
        ]
        held_out = eligible[0] if eligible else ""

    config = BenchmarkConfig(
        contexts=tuple(int(c) for c in args.contexts.split(",") if c.strip()),
        horizon=args.horizon,
        bootstrap=args.bootstrap,
        held_out_condition=held_out,
        signal=args.signal,
    )
    report = run_benchmark(trace, snapshot, config)
    payload = report.as_json()

    print(json.dumps(payload["dataset"], indent=2))
    print(json.dumps(payload["adjacency"], indent=2))
    print(f"{'arm':<16}{'ctx':>5}{'MAE':>12}{'step1':>12}{'step32':>12}")
    for arm in sorted(payload["arms"], key=lambda a: (a["context"], a["mae"])):
        print(
            f"{arm['arm']:<16}{arm['context']:>5}{arm['mae']:>12.6f}"
            f"{arm['mae_step_1']:>12.6f}{arm['mae_step_32']:>12.6f}"
        )
    print(json.dumps(payload["structure_test"], indent=2))
    if payload["held_out"]:
        print(json.dumps(payload["held_out"], indent=2))

    target = args.out or (args.data / "zapbench_report.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
