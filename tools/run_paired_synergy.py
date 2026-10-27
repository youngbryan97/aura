#!/usr/bin/env python3
"""Whether two domains move a third jointly, by intervention rather than by information.

The synergy line reads the four declared triples off a recording, through an
estimator. This asks the organism directly, by ISC-v6's design and bars
(docs/ISC_V6_PREREGISTRATION.md, core/subject/interventional_synergy.py): from
each moment, four untouched forks and, per triple, the first source displaced,
the second and both; the interaction against a sham of its own form, in the
target with the conjunction registers out; one direction, with whole vectors
flipped in its null; and a tenth of the larger main effect. A quick look on a
short recording, beside the campaign's own reading; it counts toward no total.

    python tools/run_paired_synergy.py --rounds 4 --trials 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "paired_synergy")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from core.subject.causal import DEFAULT_DELTA
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.interventional_synergy import collect, line, sample_arrays, summarise, target_columns
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.paired_displacement import displacement_metric
    from core.subject.provenance import environment, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.state import domain_slices
    from core.subject.synergy import TRIPLES

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(run_dir / "logs"))
    isolate_state(run_dir)
    started = time.monotonic()
    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}")
    await start_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)
    slices = domain_slices()
    try:
        frames = []
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        # ISC-v6's design and bars (core/subject/interventional_synergy.py):
        # registers out of each target, four shared untouched forks, a sham of
        # the interaction's form, whole vectors flipped for the direction.
        targets = {y: target_columns(list(recording.columns), slices[y]) for _a, _b, y in TRIPLES}
        metrics = {y: displacement_metric(recording.x[:, targets[y]]) for _a, _b, y in TRIPLES}
        # As the battery does before its interventions: the free-running loops
        # stop, so two forks from one moment see the same computation.
        await quiesce_organism(runtime)
        held = await collect(
            runtime, CONDITIONS, metrics=metrics, targets=targets, trials=args.trials,
            turns=args.turns, delta=DEFAULT_DELTA, seed=args.seed, on_progress=_log,
        )
    finally:
        await quiesce_organism(runtime)
    evidence = {
        "environment": environment(),
        "design": {"rounds": args.rounds, "trials": args.trials, "turns": args.turns, "delta": DEFAULT_DELTA,
                   "bars": "ISC-v6"},
        "triples": {key: summarise(samples, seed=args.seed) for key, samples in held.items()},
        "seconds": round(time.monotonic() - started, 1),
    }
    evidence["line"] = line(evidence["triples"])
    out = run_dir / "paired_synergy.json"
    from core.subject.latch import refuse_if_latched

    refuse_if_latched("the paired synergy reading")
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    # Every snapshot's vectors and the target columns they are in, so a triple
    # that fails can be read column by column.
    np.savez_compressed(
        run_dir / "paired_synergy_samples.npz", **sample_arrays(held, list(recording.columns), targets)
    )
    for key, row in evidence["triples"].items():
        print(f"  {key:10s} interaction {row['interaction']:.4f} sham {row['sham']:.4f} "
              f"fraction {row['fraction']:.3f} p {row['p_above_sham']:.4f} "
              f"direction p {row['p_one_direction']:.4f} -> {'passes' if row['passes'] else 'fails'}")
    _log(f"wrote {out} in {evidence['seconds']}s; the v6 line {'passes' if evidence['line'] else 'fails'}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
