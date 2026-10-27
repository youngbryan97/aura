#!/usr/bin/env python3
"""Compare the nonlinear closure candidate on a saved recording, without a runtime.

Output is a diagnostic, never an official closure verdict. Inputs and output
must be explicit. All recorded live targets and held predictors are retained.
Bound the process externally and use one numerical thread on a shared host.
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2", help="explicit diagnostic fold indices")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output path")

    import numpy as np

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.closure import (
        _as_increments,
        _named_clock,
        _pooled_bounds,
        environment_of_transitions,
        forward_folds,
    )
    from core.subject.closure_candidate import compare
    from core.subject.recording import load_recording

    recording = load_recording(args.run).by_turn()
    source = args.run / "closure_inputs.npz"
    with np.load(source, allow_pickle=True) as saved:
        names = tuple(str(name) for name in saved["names"])
        periphery = saved["periphery"]
    if periphery.shape[0] != recording.x.shape[0] or periphery.shape[1] != len(names):
        raise ValueError("periphery and turn recording are not aligned")
    held = np.asarray([name.startswith("held.") for name in names])
    clocks = np.asarray([_named_clock(name) for name in names])
    moving = periphery.std(axis=0) > 1e-9
    live = recording.live_columns()
    current = recording.x[:-1][:, live]
    target = recording.x[1:][:, live] - current
    own = np.hstack([current, periphery[:-1, held & moving & ~clocks]])
    env, _ = environment_of_transitions(recording)
    if env.shape[0] != own.shape[0]:
        raise ValueError("the recorded environment is not aligned with transitions")
    own = np.hstack([own, env])
    outside = periphery[:-1, ~held & moving & ~clocks]
    outside = _as_increments(outside, np.ones(outside.shape[1], dtype=bool))
    folds = forward_folds(len(own))
    requested = [int(part) for part in args.folds.split(",")]
    if not requested or any(index not in range(len(folds)) for index in requested):
        raise ValueError("unknown forward fold")
    records = []
    for index in requested:
        train, validate, test = folds[index]
        print(f"fold {index}: {own.shape[1]} own, {outside.shape[1]} outside, {target.shape[1]} targets", flush=True)
        result = compare(own, outside, target, train=train, validate=validate, test=test, seed=args.seed)
        baseline = np.asarray(result.narrow.row_sse)
        lower, upper = _pooled_bounds(
            [baseline - np.asarray(result.wide.row_sse)], [baseline], seed=args.seed + index,
        )
        records.append({
            "fold": index, "test": [test.start, test.stop], "gain": result.gain,
            "lower": lower, "upper": upper, "narrow": result.narrow.as_dict(),
            "wide": result.wide.as_dict(), "own_basis_width": result.own_width,
            "added_basis_width": result.added_width,
        })
        print(json.dumps(records[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps({
        "candidate_only": True, "official_verdict": None,
        "run": str(args.run.resolve()), "seed": args.seed, "folds": records,
        "closure_inputs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limits": "No surrogate floor, planted resolution or full distributional calibration was run here; a small gain alone cannot certify closure.",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
