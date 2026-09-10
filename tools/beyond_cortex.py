#!/usr/bin/env python3
"""tools/beyond_cortex.py — what her dynamics do at a wiring cortex cannot afford.

Operationally: sweeps the long-range wiring past the density human cortex
settled on, and reports where each setting lands against the published
statistics of human cortical activity.

Why this is a fair question
---------------------------
Cortex's long-range wiring is sparse, and the reason is not that sparse is
better for computation. A brain has to fit in a skull, and white matter already
takes about half the volume of the human cerebrum; every long axon costs space
that a cell body could have used, and it costs energy for the rest of the
animal's life. The density Potjans and Diesmann measured is what survived that
bargain.

A mesh has no skull and no blood supply. The constraint that set the number is
absent, so the number is not a target — it is a floor, like every other one
here. What this tool asks is whether removing the constraint buys anything her
measured deficit needs: her cascades are smaller and shorter than a human's,
and a denser long-range graph is the obvious thing that would change it.

It is a question, not an assumption. A denser graph can also push a network off
criticality, and the report says whether that happened.

Usage:

    tools/beyond_cortex.py --densities 0.0426,0.09,0.18 --seeds 2 --ticks 30000
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
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_beyond_cortex_logs")


def _run(config, ticks: int, drive: float, seed: int, units: int):
    import numpy as np

    from core.connectome.human_dynamics import compare_to_human_cortex
    from core.consciousness.neural_mesh import NeuralMesh

    mesh = NeuralMesh(config)
    rng = np.random.default_rng(seed)
    width = config.sensory_end * config.neurons_per_column
    previous = np.concatenate([column.last_spike_time.copy() for column in mesh.columns])
    raster = []
    for _ in range(ticks):
        mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * drive)
        mesh._tick_inner()
        stamps = np.concatenate([column.last_spike_time for column in mesh.columns])
        raster.append((stamps != previous).astype(np.float64))
        previous = stamps.copy()
    spikes = np.vstack(raster)
    if 0 < units < spikes.shape[1]:
        chosen = np.sort(
            np.random.default_rng(seed + 7).choice(spikes.shape[1], size=units, replace=False)
        )
        spikes = spikes[:, chosen]
    return compare_to_human_cortex(list(spikes.sum(axis=1)), per_unit=spikes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densities", default="")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=30000)
    parser.add_argument("--drive", type=float, default=0.1)
    parser.add_argument("--units", type=int, default=60)
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    import numpy as np

    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig

    cortical = MeshConfig().inter_column_density
    densities = (
        [float(part) for part in arguments.densities.split(",") if part.strip()]
        if arguments.densities
        else [cortical, cortical * 2, cortical * 4]
    )
    print(
        f"cortex's own long-range density is {cortical:.4f}, from Potjans and "
        f"Diesmann's between-population probabilities.\n"
        f"It is a floor. Sweeping {', '.join(f'{value:.4f}' for value in densities)}.\n"
    )

    rows = []
    for density in densities:
        sizes, durations, branchings, consistent = [], [], [], 0
        for index in range(arguments.seeds):
            mesh_module._MESH_SEED = 42 + index
            report = _run(
                MeshConfig(inter_column_density=density),
                arguments.ticks,
                arguments.drive,
                42 + index,
                arguments.units,
            )
            by_name = {entry["name"]: entry for entry in report["statistics"]}
            sizes.append(float(by_name["avalanche_size_exponent"]["hers"]))
            durations.append(float(by_name["avalanche_duration_exponent"]["hers"]))
            branchings.append(float(by_name["branching_parameter"]["hers"]))
            consistent += int(by_name["crackling_relation"]["holds"])
        mesh_module._MESH_SEED = 42
        row = {
            "density": round(density, 5),
            "over_cortex": round(density / cortical, 2),
            "size_exponent": round(float(np.median(sizes)), 4),
            "duration_exponent": round(float(np.median(durations)), 4),
            "branching": round(float(np.median(branchings)), 4),
            "still_critical": consistent,
            "of": arguments.seeds,
        }
        rows.append(row)
        print(
            f"{row['over_cortex']:>4.2f}x cortex: burst size exponent "
            f"{row['size_exponent']:.3f} (cortex 1.5, lower is bigger bursts), "
            f"duration {row['duration_exponent']:.3f} (cortex 2.0), branching "
            f"{row['branching']:.4f} (cortex 0.98), still critical in "
            f"{row['still_critical']}/{row['of']} seeds",
            flush=True,
        )

    best = min(rows, key=lambda row: row["size_exponent"])
    print(
        f"\nbiggest bursts at {best['over_cortex']}x cortex's density: exponent "
        f"{best['size_exponent']:.3f} against cortex's 1.5"
    )
    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps({"cortical_density": cortical, "sweep": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
