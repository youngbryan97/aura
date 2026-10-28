#!/usr/bin/env python3
"""Construct a collision in Aura's sketch and certify a declared-read repair.

This is an offline constructed counterexample, not a claimed pair of live Aura
states. The repair covers the declared read only, not unknown runtime reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output path")

    import numpy as np

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.observation_basis import sufficient_basis
    from core.subject.sketch import _directions, sketch

    size = 16
    directions = _directions(size)
    constraints = np.vstack([np.ones(size - 2), directions[:, 1:-1]])
    _u, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > 1e-12).sum())
    states = np.zeros((2, size))
    states[:, 0], states[:, -1] = -3., 3.
    states[0, 1:-1], states[1, 1:-1] = vt[rank], vt[rank + 1]
    observed = [sketch(state) for state in states]
    error = max(abs(observed[0][key] - observed[1][key]) for key in observed[0])
    coordinate = int(np.argmax(np.abs(states[0] - states[1])))
    reader = np.eye(size)[[coordinate]]
    basis = sufficient_basis(np.vstack([np.ones(size), directions]), reader)
    expected = states @ reader.T
    reconstructed = basis.reconstruct_reads(basis.observe(states))
    verified = error < 1e-12 and basis.reconstruction_error < 1e-12 and abs(expected[0, 0] - expected[1, 0]) > .1
    output = {
        "constructed_counterexample": True, "verified": bool(verified),
        "actual_aura_transition_witness": False, "states": states.tolist(),
        "production_sketches": observed, "maximum_sketch_difference": error,
        "state_distance": float(np.linalg.norm(states[0] - states[1])),
        "declared_read_coordinate": coordinate, "different_futures": expected[:, 0].tolist(),
        "read_reconstruction_error": basis.reconstruction_error,
        "reconstructed_futures": reconstructed[:, 0].tolist(),
        "original_linear_rank": basis.previous_rank, "added_rank": basis.added_rank,
        "repair_axes": basis.axes.tolist(), "reader_coordinates": basis.reader_coordinates.tolist(),
        "proof": "The interior vectors are orthogonal unit vectors in the nullspace of the mean and four projections. Fixed extrema preserve min and max; equal norms preserve SD. Adding the read's missing row-space direction recovers that read for every state, to the numerical certificate's tolerance.",
        "limits": "This proves loss of sufficiency is possible in the production observation family. It does not prove it caused the observed seed-73 gain. Runtime adapters must supply and witness complete transition reads before this basis can qualify an actual boundary.",
        "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                          for name in ("core/subject/sketch.py", "core/subject/observation_basis.py")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "verified", "maximum_sketch_difference", "state_distance", "different_futures",
        "read_reconstruction_error", "original_linear_rank", "added_rank",
    )}, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
