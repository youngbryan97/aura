#!/usr/bin/env python3
"""Ask a recording whether the consciousness ring closes, and how reproducibly.

The question is the one the workspace theories make answerable: does influence
run from interoception through affect, the global workspace, higher-order
monitoring, the self model and planning to action, and back to interoception —
in the recording, not on the wiring diagram. A structural path proves the ring
could close. Only a recording says whether it did.

Reading it from one condition is not enough. A single ring closing is an
observation about one workload; the same links closing across conditions that
share no content is a coalition. So this runs the test per condition and then
asks how often each link carried, which is what ``reproducibility`` is for.

The graph tested is proofread first. A static reconstruction cannot see a call
made through a service lookup or a dispatch table, and those are exactly the
edges a station-to-station link runs on, so testing the raw reconstruction
answers a question about the reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(recording: Path) -> tuple[Any, Any]:
    """The trace and the observed edges from one recording directory."""
    import numpy as np

    from core.connectome.activity import ActivityTrace, ObservedEdges

    manifest = json.loads((recording / "activity_manifest.json").read_text())
    trace = ActivityTrace(
        uids=tuple(manifest["uids"]),
        conditions=tuple(manifest["conditions"]),
        spikes=[],
        array=np.load(recording / "activity.npz")["spikes"],
        frame_seconds=float(manifest.get("frame_seconds", 0.05)),
    )
    observed = ObservedEdges()
    payload = json.loads((recording / "observed_edges.json").read_text())
    for key, count in payload.get("counts", {}).items():
        pre, _, post = key.partition(">")
        observed.counts[(pre, post)] = int(count)
    observed.unresolved = int(payload.get("summary", {}).get("unresolved", 0))
    return trace, observed.without_self_pairs()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        default=REPO / "artifacts" / "connectome" / "turn",
        help="a directory holding activity.npz, its manifest and observed_edges.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nulls", type=int, default=200)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--influence-nulls", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument(
        "--structural",
        action="store_true",
        help="also run the weaker structural test, for comparison",
    )
    args = parser.parse_args()
    out = args.out or (args.recording / "coalition_analysis.json")

    from core.connectome.coalition import (
        assign_stations,
        measure_ring,
        reproducibility,
        test_closure,
    )
    from core.connectome.effective import predictive_influence
    from core.connectome.layers import extract_layers
    from core.connectome.proofreading import repair_observed_splits
    from core.connectome.volume import VolumeReconstructor

    started = time.monotonic()
    trace, observed = _load(args.recording)
    reconstructor = VolumeReconstructor(REPO)
    reconstructor.scan()
    raw = reconstructor.build()
    ledger = repair_observed_splits(raw, observed)
    snapshot = ledger.apply(raw)
    multilayer = extract_layers(snapshot, REPO)
    stations = assign_stations(snapshot)
    recorded = list(trace.uids)
    print(
        json.dumps(
            {
                "joins_written": len(ledger),
                "cells": len(snapshot.units),
                "recorded_cells": len(recorded),
                "stations": {name: len(st.cells) for name, st in stations.items()},
                "recorded_per_station": {
                    name: len(set(recorded).intersection(st.cells))
                    for name, st in stations.items()
                },
                "seconds": round(time.monotonic() - started, 1),
            },
            indent=2,
        ),
        flush=True,
    )

    # Only cells in a station can answer a question about stations, and the
    # influence test is nine regressions an edge. Narrowing it here is the
    # difference between minutes and hours, and it changes no result: an edge
    # with neither end in a station is never read by the closure test.
    in_stations: set[str] = set()
    for station in stations.values():
        in_stations.update(station.cells)

    report: dict[str, Any] = {
        "recording": {
            "path": str(args.recording),
            "frames": len(trace.conditions),
            "cells": len(trace.uids),
            "frame_seconds": trace.frame_seconds,
            "conditions": sorted(set(trace.conditions)),
        },
        "proofreading": {"joins_written": len(ledger)},
        "stations": {name: len(st.cells) for name, st in stations.items()},
        "conditions": {},
    }
    closures = []
    for condition in sorted(set(trace.conditions)):
        began = time.monotonic()
        effective = predictive_influence(
            trace,
            snapshot,
            condition,
            lags=args.lags,
            nulls=args.influence_nulls,
            only_cells=sorted(in_stations),
        )
        closure = test_closure(
            snapshot,
            effective,
            stations,
            nulls=args.nulls,
            multilayer=multilayer,
            max_hops=args.max_hops,
            recorded=recorded,
        )
        closures.append(closure)
        # The link-by-link answer, and then where the ring sits among every
        # cycle through the same stations. The first is about coupling; only
        # the second is about the ring being a ring.
        ring = measure_ring(trace, condition, stations, seed=1)
        report["conditions"][condition] = {
            "effective": effective.summary(),
            "closure": closure.as_json(),
            "ring": ring.as_json(),
            "seconds": round(time.monotonic() - began, 1),
        }
        print(
            json.dumps(
                {
                    "condition": condition,
                    "links_present": closure.links_present,
                    "links_carrying": ring.links_carrying,
                    "ring_percentile": round(ring.percentile, 3),
                    "ring_z": round(ring.z, 2),
                    "verdict": ring.as_json()["verdict"],
                    "seconds": round(time.monotonic() - began, 1),
                }
            ),
            flush=True,
        )
    report["reproducibility"] = reproducibility(closures)

    if args.structural:
        structural = test_closure(
            snapshot,
            None,
            stations,
            nulls=args.nulls,
            use_effective=False,
            multilayer=multilayer,
            max_hops=args.max_hops,
        )
        report["structural"] = structural.as_json()

    report["seconds"] = round(time.monotonic() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["reproducibility"], indent=2), flush=True)
    print(f"written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
