#!/usr/bin/env python3
"""tools/grow_the_mesh.py — the mesh arrives by growing, over several seeds.

Operationally: runs the developmental trajectory a number of times and reports
whether pruning by what a synapse carried beats cutting the same number at
random. One run cannot answer that; the difference is small and the mesh is
stochastic, so the answer is a paired comparison across seeds.

Usage:

    tools/grow_the_mesh.py --seeds 5 --ticks 800
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
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_mesh_growth_logs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=800)
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    import numpy as np

    from core.connectome.mesh_growth import HUTTENLOCHER, grow_mesh

    print(f"{HUTTENLOCHER['source']}\n{HUTTENLOCHER['recorded_in']}\n")

    runs = []
    for index in range(arguments.seeds):
        _, result = grow_mesh(ticks=arguments.ticks, seed=42 + index)
        record = result.as_json()
        runs.append(record)
        print(
            f"seed {42 + index}: overshoot {record['overshoot']}, "
            f"grew {record['grown']:,} and cut {record['pruned']:,}; the executive band "
            f"carries {record['carried_by_use']:.6f} against {record['carried_at_random']:.6f} "
            f"at random ({record['carried_over_random']:.3f}x)",
            flush=True,
        )

    ratios = np.array([run["carried_over_random"] for run in runs], dtype=float)
    better = int((ratios > 1.0).sum())
    print(
        f"\npruning by what a synapse carried beat the same cut made at random in "
        f"{better} of {len(ratios)} runs; median {float(np.median(ratios)):.3f}x"
    )

    landed = all(
        abs(run["adult_density"][band] - run["target_density"][band]) < 0.002
        for run in runs
        for band in run["adult_density"]
    )
    print(
        "every band landed on the density its anatomy specifies"
        if landed
        else "a band did not land on its anatomical density"
    )

    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps(
                {
                    "source": HUTTENLOCHER,
                    "runs": runs,
                    "beat_the_null": better,
                    "of": len(ratios),
                    "median_ratio": round(float(np.median(ratios)), 4),
                    "landed_on_anatomy": landed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.json}")
    return 0 if better * 2 > len(ratios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
