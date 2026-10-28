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
import math
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
    parser.add_argument("--feature-limit", type=float, help="candidate-only bound in training standard deviations")
    parser.add_argument("--surrogates", type=int, default=0, help="diagnostic temporal controls; zero does not measure a floor")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output path")
    if args.feature_limit is not None and (not math.isfinite(args.feature_limit) or args.feature_limit <= 0.):
        parser.error("feature-limit must be finite and positive")
    if args.surrogates < 0:
        parser.error("surrogates cannot be negative")

    import numpy as np

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.closure import (
        _as_increments,
        _named_clock,
        _pooled_bounds,
        _surrogate,
        environment_of_transitions,
        forward_folds,
    )
    from core.subject.closure_candidate import compare
    from core.subject.estimate import _FLAT
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
    target_names = np.asarray(recording.columns)[live]
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
    metadata = {
        "candidate_only": True, "official_verdict": None,
        "run": str(args.run.resolve()), "seed": args.seed,
        "feature_limit": args.feature_limit, "surrogates_requested": args.surrogates,
        "folds_requested": requested,
        "input_sha256": {
            name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
            for name in ("closure_inputs.npz", "core_state.npz", "core_state_manifest.json")
        },
        "source_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in ("core/subject/estimate.py", "core/subject/closure_candidate.py",
                         "core/subject/closure.py", "core/subject/recording.py", "tools/diagnose_closure_candidate.py")
        },
        "limits": "Diagnostic only. Few surrogates give a coarse floor. No planted resolution or distributional calibration was run; gain alone cannot certify closure, and unscored moving targets leave coverage incomplete.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(complete: bool = False) -> None:
        atomic_write_text(args.out, json.dumps({**metadata, "complete": complete, "folds": records}, indent=2) + "\n")

    checkpoint()
    for index in requested:
        train, validate, test = folds[index]
        print(f"fold {index}: {own.shape[1]} own, {outside.shape[1]} outside, {target.shape[1]} targets", flush=True)
        result = compare(
            own, outside, target, train=train, validate=validate, test=test,
            seed=args.seed, feature_limit=args.feature_limit,
        )
        unscored = target[train].std(axis=0) <= _FLAT
        # A target can change to a different constant increment in the test
        # window. Test-window variance alone would miss that activation.
        newly_moving = unscored & (
            np.max(np.abs(target[test] - target[train].mean(axis=0)), axis=0) > _FLAT
        )
        baseline = np.asarray(result.narrow.row_sse)
        lower, upper = _pooled_bounds(
            [baseline - np.asarray(result.wide.row_sse)], [baseline], seed=args.seed + index,
        )
        records.append({
            "fold": index, "test": [test.start, test.stop], "gain": result.gain,
            "lower": lower, "upper": upper, "narrow": result.narrow.as_dict(),
            "wide": result.wide.as_dict(), "own_basis_width": result.own_width,
            "added_basis_width": result.added_width,
            "unscored_targets": target_names[unscored].tolist(),
            "unscored_but_moving_in_test": target_names[newly_moving].tolist(),
            "target_coverage_complete": not bool(newly_moving.any()),
        })
        print(json.dumps(records[-1]), flush=True)
        controls = records[-1]["surrogate_gains"] = []
        checkpoint()
        rng = np.random.default_rng(args.seed + index)
        for draw in range(args.surrogates):
            shuffled = compare(
                own, _surrogate(outside, rng), target, train=train, validate=validate, test=test,
                seed=args.seed, feature_limit=args.feature_limit,
            )
            controls.append(shuffled.gain)
            print(f"fold {index}, surrogate {draw + 1}: {shuffled.gain}", flush=True)
            checkpoint()
        measured = bool(controls) and all(value is not None and math.isfinite(value) for value in controls)
        records[-1]["surrogate_floor_measured"] = measured
        records[-1]["surrogate_q95"] = float(np.quantile(controls, .95)) if measured else None
        records[-1]["surrogate_tail_fraction"] = (
            (1 + sum(value >= result.gain for value in controls)) / (len(controls) + 1)
            if measured and result.gain is not None else None
        )
        checkpoint()
    checkpoint(complete=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
