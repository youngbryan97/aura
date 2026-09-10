#!/usr/bin/env python3
"""tools/compare_to_human_cortex.py — hold her activity against a human recording.

Operationally: runs the mesh, takes the trace, and reports for each published
statistic of human cortical activity whether it holds and by how much it misses.
Nothing is fitted. The mesh's constants were set from anatomy and the dynamics
come out where they come out.

Usage:

    tools/compare_to_human_cortex.py --ticks 3000
    tools/compare_to_human_cortex.py --ticks 3000 --percentile 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_human_dynamics_logs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--percentile", type=float, default=25.0)
    parser.add_argument(
        "--units",
        type=int,
        default=60,
        help="how many units the recording reads, as an electrode array reads a slice",
    )
    parser.add_argument(
        "--unit-percentile",
        type=float,
        default=0.0,
        help="threshold on the raster; 0 counts any spike, which is the raster's own definition",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drive", type=float, default=0.1)
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    import numpy as np

    from core.connectome.human_dynamics import compare_to_human_cortex
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    config = MeshConfig()
    mesh = NeuralMesh(config)
    width = config.sensory_end * config.neurons_per_column
    rng = np.random.default_rng(arguments.seed)

    print(f"running {arguments.ticks} ticks", flush=True)
    # The spike raster, not the mean. Beggs and Plenz counted electrodes that
    # fired, and a cascade is a run of bins in which SOMETHING fired. A mean
    # over 64 columns is above its own quartile three quarters of the time, so
    # it has no silent bins and reports the whole recording as one cascade.
    trace: list[float] = []
    raster: list[np.ndarray] = []
    previous = np.concatenate(
        [column.last_spike_time.copy() for column in mesh.columns]
    )
    for _ in range(arguments.ticks):
        mesh.inject_sensory(
            rng.standard_normal(width).astype(np.float32) * arguments.drive
        )
        mesh._tick_inner()
        stamps = np.concatenate([column.last_spike_time for column in mesh.columns])
        fired = (stamps != previous).astype(np.float64)
        previous = stamps.copy()
        raster.append(fired)
        trace.append(float(fired.sum()))

    spikes = np.vstack(raster)
    # Sample it the way a recording samples tissue.
    #
    # Beggs and Plenz read 60 electrodes and Shriki 102 MEG sensors, out of a
    # cortex with billions of cells. Reading all 4,096 units gives a spike in
    # 83% of bins, so there is almost no silence and the cascades merge into
    # each other — the exponents that come out of that describe the sampling,
    # not the mesh. Subsampling is what makes the comparison a comparison, and
    # it is also why the branching ratio is estimated by multistep regression
    # rather than by counting.
    if 0 < arguments.units < spikes.shape[1]:
        chosen = np.random.default_rng(arguments.seed + 7).choice(
            spikes.shape[1], size=arguments.units, replace=False
        )
        spikes = spikes[:, np.sort(chosen)]
        trace = list(spikes.sum(axis=1))
    print(
        f"{int(spikes.sum()):,} spikes over {arguments.ticks} ticks from "
        f"{spikes.shape[1]:,} of {sum(c.n for c in mesh.columns):,} units; "
        f"{100 * float((spikes.sum(axis=1) > 0).mean()):.1f}% of bins had one",
        flush=True,
    )
    report = compare_to_human_cortex(
        trace,
        percentile=arguments.percentile,
        per_unit=spikes,
        per_unit_percentile=arguments.unit_percentile,
    )
    print(f"\n{report['verdict']}")
    print(f"cascades: {report['avalanches']}")
    print(f"size fit: {report['size_fit']}")
    print(f"duration fit: {report['duration_fit']}")
    print(
        f"crackling: measured {report['statistics'][2]['hers']} against "
        f"{report['predicted_crackling']} predicted by the two exponents\n"
    )
    for entry in report["statistics"]:
        mark = "holds" if entry["holds"] else "no"
        print(
            f"  {entry['name']:30s} human {entry['human']:>5}  hers {entry['hers']:>8}  "
            f"{mark}"
        )
        if entry["reason"]:
            print(f"      {entry['reason']}")
        print(f"      {entry['recorded_in']} — {entry['source']}")
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.json}")
    return 0 if report["held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
