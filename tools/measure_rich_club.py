#!/usr/bin/env python3
"""tools/measure_rich_club.py — do her hubs form a club, and does it help?

Operationally: sweeps how strongly hub columns are coupled, and for each
setting reports whether a rich club exists against degree-preserving rewiring
of her own graph, and what happens to the cascades. The human connectome says a
club EXISTS and how big it is; what coupling produces one in a 64-column graph
is a property of this graph, so it is measured rather than assumed.

The sweep goes past the human setting on purpose. Cortex is a floor.

Usage:

    tools/measure_rich_club.py --couplings 1,3,6,10,20 --seeds 5
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
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_rich_club_logs")


def _cascades(mesh, config, ticks: int, drive: float, seed: int, units: int):
    import numpy as np

    from core.connectome.human_dynamics import compare_to_human_cortex

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
    parser.add_argument("--couplings", default="1,3,6,10,20")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=0, help="0 skips the cascade arm")
    parser.add_argument("--drive", type=float, default=0.1)
    parser.add_argument("--units", type=int, default=60)
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    import numpy as np

    from core.connectome.rich_club import HUMAN_RICH_CLUB, measure_rich_club
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    print(f"{HUMAN_RICH_CLUB['source']}\n{HUMAN_RICH_CLUB['recorded_in']}")
    print(
        f"one node in {round(1 / HUMAN_RICH_CLUB['hub_fraction'])} belongs to the club, "
        f"and it is denser than degree-preserving rewiring gives\n"
    )

    rows = []
    for raw in [part.strip() for part in arguments.couplings.split(",") if part.strip()]:
        coupling = float(raw)
        peaks, degrees, exponents = [], [], []
        for index in range(arguments.seeds):
            mesh_module._MESH_SEED = 42 + index
            config = MeshConfig(hub_coupling=coupling)
            mesh = NeuralMesh(config)
            measurement = measure_rich_club(mesh._inter_W, seed=13 + index, nulls=20)
            peaks.append(measurement.peak)
            present = np.abs(mesh._inter_W) > 0
            degrees.append(float((present.sum(axis=0) + present.sum(axis=1)).max()))
            if arguments.ticks:
                report = _cascades(
                    mesh, config, arguments.ticks, arguments.drive, 42 + index, arguments.units
                )
                by_name = {entry["name"]: entry for entry in report["statistics"]}
                exponents.append(float(by_name["avalanche_size_exponent"]["hers"]))
        mesh_module._MESH_SEED = 42
        row = {
            "coupling": coupling,
            "median_peak": round(float(np.median(peaks)), 4),
            "clubs": int(sum(1 for value in peaks if value > 1.0)),
            "of": len(peaks),
            "max_degree": round(float(np.median(degrees)), 1),
            "median_size_exponent": (
                round(float(np.median(exponents)), 4) if exponents else None
            ),
        }
        rows.append(row)
        print(
            f"coupling {coupling:5.1f}: club in {row['clubs']}/{row['of']} seeds, "
            f"median normalised {row['median_peak']:.3f}, busiest column has "
            f"{row['max_degree']:.0f} long-range edges"
            + (
                f", cascade exponent {row['median_size_exponent']:.3f} "
                f"(cortex 1.5, lower is bigger cascades)"
                if row["median_size_exponent"] is not None
                else ""
            ),
            flush=True,
        )

    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps({"source": HUMAN_RICH_CLUB, "sweep": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
