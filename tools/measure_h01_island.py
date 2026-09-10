#!/usr/bin/env python3
"""tools/measure_h01_island.py — what changes when a column is wired like cortex.

Operationally: runs the mesh with the same seed and the same drive, once with
every column drawn from a Gaussian and once with some columns wired from H01's
measured contact-multiplicity law, and reports what moved.

What is being compared
----------------------
Not accuracy at anything. The mesh has no task here. What it has is a dynamical
regime, and the measurement that says where a network sits in it is the
multistep-regression branching ratio: one means each unit of activity begets
one more, below one the network forgets its input, above one it saturates.
Cortex sits just under one and this mesh's regulator steers for that, so the
question a wiring change has to answer is which side of it the change moves to
and whether the estimate is any good.

The heavier tail is the whole point of the change, so it is reported too: how
much stronger the heaviest connection is than the median one, under each
wiring.

Usage:

    tools/measure_h01_island.py --islands 0,8,32,64 --ticks 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_h01_island_logs")


def _run(islands: int, ticks: int, seed: int, *, tiered: bool = True) -> dict[str, Any]:
    import numpy as np

    from core.connectome.criticality import branching_ratio_mr
    from core.consciousness import neural_mesh as mesh_module
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    # The uniform arm is the mesh's own fallback path, not a second
    # implementation: an empty tier table sends every column back to the global
    # figures it used before the layers were read.
    mesh_module._TIER_CONSTANTS = None if tiered else {}

    config = MeshConfig(human_island_columns=islands)
    mesh = NeuralMesh(config)
    mesh._rng = np.random.default_rng(seed)

    drive = np.random.default_rng(seed + 1)
    activity: list[float] = []
    for _ in range(ticks):
        # The full sensory input width. Sixteen values reached sixteen of the
        # 1,024 unit inputs and left the rest at zero, so the mesh was being
        # measured almost undriven.
        mesh.inject_sensory(
            drive.standard_normal(
                config.sensory_end * config.neurons_per_column
            ).astype(np.float32)
            * 0.1
        )
        mesh._tick_inner()
        activity.append(float(np.mean(np.abs(mesh.column_activations))))

    estimate = branching_ratio_mr(activity)
    strengths = np.concatenate(
        [np.abs(column.W[column.W != 0]).ravel() for column in mesh.columns]
    )
    return {
        "islands": float(islands),
        "branching_ratio": float(estimate.m),
        "fit": float(estimate.r_squared),
        "mean_activity": float(np.mean(activity)),
        "regime": estimate.regime,
        "heaviest_over_median": float(np.max(strengths) / np.median(strengths)),
        "connections": float(strengths.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--islands", default="0,8,32,64")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tiers",
        default="both",
        choices=("both", "tiered", "uniform"),
        help="whether each tier takes its own layers' wiring or the global average",
    )
    arguments = parser.parse_args()

    from core.connectome.island import (
        candidate_laws,
        probability_of_the_observed_maximum,
        surviving_law,
    )

    print("The laws that could have produced H01's single-contact fraction:")
    for law in candidate_laws():
        print(
            f"  {law.name}: fitted to {len(law.fitted_to)} of its numbers, "
            f"predicts {law.p_at_least(4) * 100:.4f}% of pairs at four or more contacts "
            f"(measured 0.0920%), and a fifty-contact pair turns up in "
            f"{probability_of_the_observed_maximum(law) * 100:.4g}% of volumes this size"
        )
    chosen = surviving_law()
    print(f"  -> wiring from: {chosen.name}\n")

    arms = (
        (True, False) if arguments.tiers == "both" else (arguments.tiers == "tiered",)
    )
    rows = []
    for tiered in arms:
        for raw in [part.strip() for part in arguments.islands.split(",") if part.strip()]:
            row = _run(int(raw), arguments.ticks, arguments.seed, tiered=tiered)
            row["tiered"] = tiered
            rows.append(row)
            print(
                f"{'per-tier' if tiered else ' uniform'} wiring, "
                f"{int(row['islands']):3d} human columns: branching ratio "
                f"{row['branching_ratio']:.4f} (fit {row['fit']:.3f}), mean activity "
                f"{row['mean_activity']:.5f}, heaviest connection "
                f"{row['heaviest_over_median']:.1f}x the median, "
                f"{int(row['connections']):,} connections, {row['regime']}",
                flush=True,
            )

    print()
    for tiered in arms:
        arm = [row for row in rows if row["tiered"] is tiered]
        if len(arm) < 2:
            continue
        base = arm[0]
        for row in arm[1:]:
            moved = abs(row["branching_ratio"] - 1.0) - abs(base["branching_ratio"] - 1.0)
            direction = "nearer" if moved < 0 else "further from"
            print(
                f"{'per-tier' if tiered else 'uniform'}: {int(row['islands'])} human "
                f"columns put the mesh {abs(moved):.4f} {direction} critical, with the "
                f"fit at {row['fit']:.3f} against {base['fit']:.3f}"
            )
    if len(arms) == 2:
        for islands in sorted({int(row["islands"]) for row in rows}):
            pair = {row["tiered"]: row for row in rows if int(row["islands"]) == islands}
            if len(pair) != 2:
                continue
            moved = abs(pair[True]["branching_ratio"] - 1.0) - abs(
                pair[False]["branching_ratio"] - 1.0
            )
            direction = "nearer" if moved < 0 else "further from"
            print(
                f"at {islands} human columns, giving each tier its own layers' wiring "
                f"puts the mesh {abs(moved):.4f} {direction} critical"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
