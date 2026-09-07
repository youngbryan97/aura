#!/usr/bin/env python3
"""Reconstruct Aura's connectome and run every analysis over it.

One command, one JSON artifact, every number in this package with the published
value it is compared against. Sections can be selected because the whole battery
takes a couple of minutes and most questions want one part of it.

    python tools/connectome_report.py --sections all --out artifacts/connectome

With ``--observed`` pointing at a recording made by
``tools/record_connectome_activity.py``, the reconstruction is also scored
against ground truth and the agglomeration threshold is swept.
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

SECTIONS = (
    "reconstruction",
    "layers",
    "likewise",
    "stereotypy",
    "pathology",
    "prefetch",
    "synaptology",
    "topology",
    "celltypes",
    "microcircuit",
    "spine",
    "whorls",
    "delays",
    "criticality",
    "ground_truth",
)


def _load_observed(path: Path) -> Any:
    from core.connectome.activity import ObservedEdges

    payload = json.loads(path.read_text())
    observed = ObservedEdges()
    for key, count in payload.get("counts", {}).items():
        pre, _, post = key.partition(">")
        observed.counts[(pre, post)] = int(count)
    observed.unresolved = int(payload.get("summary", {}).get("unresolved", 0))
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "connectome")
    parser.add_argument("--observed", type=Path, default=None)
    parser.add_argument("--nulls", type=int, default=4)
    parser.add_argument("--triad-sample", type=int, default=1500)
    parser.add_argument("--roots", default="core,interface,skills,security,llm,executors")
    parser.add_argument(
        "--neuroglancer",
        action="store_true",
        help="also write the viewer state and segment properties",
    )
    args = parser.parse_args()

    wanted = set(SECTIONS) if args.sections == "all" else {
        s.strip() for s in args.sections.split(",") if s.strip()
    }
    args.out.mkdir(parents=True, exist_ok=True)

    from core.connectome.volume import ReconstructionConfig, VolumeReconstructor

    started = time.time()
    reconstructor = VolumeReconstructor(
        REPO, ReconstructionConfig(roots=tuple(r.strip() for r in args.roots.split(",")))
    )
    reconstructor.scan()
    snapshot = reconstructor.build()
    report: dict[str, Any] = {
        "built_at": time.time(),
        "repo": str(REPO),
        "reconstruction": {**snapshot.summary(), **snapshot.attrs},
    }
    print(f"reconstructed in {time.time() - started:.1f}s", flush=True)

    observed = _load_observed(args.observed) if args.observed and args.observed.exists() else None

    if "synaptology" in wanted:
        from core.connectome.synaptology import (
            compartment_profile,
            ei_report,
            gate_dominated_cells,
            measure_multiplicity,
            strong_connections,
        )

        report["synaptology"] = {
            "multiplicity": measure_multiplicity(snapshot).as_json(),
            "strongest": [c.as_json() for c in strong_connections(snapshot, limit=15)],
            "compartments": compartment_profile(snapshot).as_json(),
            "gate_dominated": gate_dominated_cells(snapshot, limit=15),
            "excitation_inhibition": ei_report(snapshot),
        }
        print("synaptology done", flush=True)

    if "topology" in wanted:
        from core.connectome.topology import analyse

        result = analyse(snapshot, nulls=args.nulls, triad_sample=args.triad_sample)
        report["topology"] = result.as_json()
        report["topology"]["significant_motifs"] = result.significant_motifs()
        print("topology done", flush=True)

    if "celltypes" in wanted:
        from core.connectome.celltypes import stability

        report["celltypes"] = stability(snapshot, rounds=1, repeats=3)
        print("cell types done", flush=True)

    assignment = None
    if {"microcircuit", "whorls"} & wanted:
        from core.connectome.microcircuit import assign_layers

        assignment = assign_layers(snapshot)

    if "microcircuit" in wanted and assignment is not None:
        from core.connectome.microcircuit import compare_to_cortex, connection_probabilities

        matrix = connection_probabilities(snapshot, assignment)
        report["microcircuit"] = {
            "assignment": assignment.summary(),
            "matrix": [[round(v, 8) for v in row] for row in matrix],
            "against_cortex": compare_to_cortex(matrix),
        }
        print("microcircuit done", flush=True)

    multilayer = None
    if {"layers", "pathology"} & wanted:
        from core.connectome.layers import extract_layers, layer_report

        multilayer = extract_layers(snapshot, REPO)
        if "layers" in wanted:
            report["layers"] = layer_report(snapshot, multilayer)
            print("layers done", flush=True)

    if "pathology" in wanted:
        from core.connectome.integration import record_pathology
        from core.connectome.microcircuit import assign_layers
        from core.connectome.pathology import diagnose

        laminar = assignment if assignment is not None else assign_layers(snapshot)
        diagnosis = diagnose(
            snapshot, multilayer=multilayer, observed=observed, laminar=laminar
        )
        report["pathology"] = diagnosis.as_json(limit=120)
        record_pathology(diagnosis)
        print(
            f"pathology done: {len(diagnosis.findings)} findings, "
            f"{len(diagnosis.confirmed())} confirmed",
            flush=True,
        )

    if "prefetch" in wanted and args.observed:
        manifest_path = args.observed.parent / "activity_manifest.json"
        matrix_path = args.observed.parent / "activity.npz"
        if manifest_path.exists() and matrix_path.exists():
            import numpy as np

            from core.connectome.activity import ActivityTrace
            from core.connectome.prefetch import evaluate_prefetch

            manifest = json.loads(manifest_path.read_text())
            spikes = np.load(matrix_path)["spikes"]
            trace = ActivityTrace(
                uids=tuple(manifest["uids"]),
                conditions=tuple(manifest["conditions"]),
                spikes=[list(map(float, row)) for row in spikes],
            )
            report["prefetch"] = evaluate_prefetch(trace, snapshot, hops=1).as_json()
            print("prefetch done", flush=True)

    if "spine" in wanted:
        from core.connectome.spine import analyse_spine, descending_directness

        report["spine"] = analyse_spine(snapshot).as_json()
        report["spine"]["descending_directness"] = descending_directness(snapshot)
        print("spine done", flush=True)

    if "likewise" in wanted and args.observed:
        manifest_path = args.observed.parent / "activity_manifest.json"
        matrix_path = args.observed.parent / "activity.npz"
        if manifest_path.exists() and matrix_path.exists():
            import numpy as np

            from core.connectome.activity import ActivityTrace
            from core.connectome.likewise import test_like_to_like

            manifest = json.loads(manifest_path.read_text())
            trace = ActivityTrace(
                uids=tuple(manifest["uids"]),
                conditions=tuple(manifest["conditions"]),
                spikes=[],
                array=np.load(matrix_path)["spikes"],
            )
            report["likewise"] = test_like_to_like(trace, snapshot).as_json()
            print("like-to-like done", flush=True)

    if "stereotypy" in wanted:
        from core.connectome.celltypes import refine_types, serial_homology

        report["serial_homology"] = serial_homology(
            snapshot, refine_types(snapshot, rounds=1)
        )
        print("serial homology done", flush=True)

    if "whorls" in wanted:
        from core.connectome.beyond import explain_whorls, whorl_census

        whorls = whorl_census(snapshot, limit=20)
        report["whorls"] = {
            "census": [w.as_json() for w in whorls],
            "explained": explain_whorls(snapshot, whorls, limit=5),
        }
        print("whorls done", flush=True)

    if "delays" in wanted:
        from core.connectome.beyond import compile_delays

        report["delays"] = compile_delays(snapshot).as_json()
        print("delays done", flush=True)

    if "ground_truth" in wanted and observed is not None:
        from core.connectome.development import compare_pruning
        from core.connectome.proofreading import focused_queue
        from core.connectome.segmentation import score_against_observation, sweep_threshold

        report["ground_truth"] = {
            "observed": {
                "pairs": len(observed.counts),
                "calls": sum(observed.counts.values()),
            },
            "base_score": score_against_observation(snapshot, observed).as_json(),
            "threshold_sweep": sweep_threshold(
                snapshot, reconstructor.ambiguous_sites, observed
            ),
            "focused_queue": [c.to_json() for c in focused_queue(snapshot, observed, limit=20)],
            "pruning": compare_pruning(snapshot, observed),
        }
        print("ground truth done", flush=True)

    if "criticality" in wanted and args.observed:
        manifest = args.observed.parent / "activity_manifest.json"
        matrix_path = args.observed.parent / "activity.npz"
        if manifest.exists() and matrix_path.exists():
            import numpy as np

            from core.connectome.criticality import assess

            spikes = np.load(matrix_path)["spikes"]
            population = spikes.sum(axis=1).tolist()
            report["criticality"] = assess(population).as_json()
            print("criticality done", flush=True)

    if args.neuroglancer:
        from core.connectome.microcircuit import assign_layers
        from core.connectome.neuroglancer import write_export

        laminar = assignment if assignment is not None else assign_layers(snapshot)
        report["neuroglancer"] = write_export(snapshot, args.out / "neuroglancer", laminar)
        print("neuroglancer export done", flush=True)

    target = args.out / "connectome_report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {target} in {time.time() - started:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
