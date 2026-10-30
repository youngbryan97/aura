#!/usr/bin/env python3
"""Her own runtime with every channel between her domains cut: does anything still read as one?

The battery's nulls are toys: twenty small systems built to have the parts and
lack the organisation. A reader asked for the null the toys cannot be, her
(the outside reads of 30 October, Q35). This is that null. From one saved
moment, each of her ten domains lives alone: the other nine are held where the
moment left them, so it evolves on its own pipeline with no information from
any other domain. The ten recordings are composed column by column into one
life of hers with every crossing removed. Each domain's own dynamics are hers,
so its drift, its rhythms and its autocorrelation are real, which is what the
toys cannot give a measure to be fooled by.

It is the lesion of `tools/run_subject_core.py` taken to every partition at
once: there each side of the cheapest cut ran with the other held, here each
domain runs with all the others held, through the same clamp and the same
composition (core/subject/clamp.py).

Read beside two intact lives from the same moment, the second giving the drift
floor:

  * irreducibility (phi_do, the weakest cut) and its standard error;
  * each synergy triple, and whether it passes its line;
  * every held column that moved anyway, which would be a channel the cut did
    not sever.

What it predicts, written before it runs: the disconnected life reads no
integration (phi_do no more than its own standard error above zero) and passes
no synergy triple, and no held column moves. Integration read on it would mean
the estimators find unity in her marginals alone, and every line built on them
would have to say so.

    python tools/run_disconnected_aura.py --seed 101 --out ~/subject-core-runs/disconnected
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("AURA_TESTING", "1")

REPORT_FILE = "disconnected_aura.json"


def _triple(report: Any) -> str:
    return f"{','.join(report.sources)}->{report.target}"


def compose_alone(alone: dict[str, list[Any]], domains: list[str]) -> list[Any]:
    """One recording in which each domain's columns come from the life it lived alone."""
    from core.subject.clamp import compose

    composed = alone[domains[0]]
    done = [domains[0]]
    for domain in domains[1:]:
        composed = compose(composed, alone[domain], done)
        done.append(domain)
    return composed


def verdict(intact: dict[str, Any], drift: dict[str, Any], cut: dict[str, Any], leaks: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Whether the disconnected life behaved as predicted, and why."""
    integrated = cut["phi_do"] > max(cut["standard_error"], 0.0)
    passing = [name for name, passed in cut["synergy_passes"].items() if passed]
    leaked = {domain: moved for domain, moved in leaks.items() if moved}
    holds = not integrated and not passing and not leaked
    reasons = []
    if integrated:
        reasons.append(f"phi_do {cut['phi_do']:.5f} sits above its standard error {cut['standard_error']:.5f}")
    if passing:
        reasons.append(f"synergy passes on {passing}")
    if leaked:
        reasons.append(f"held columns moved in {sorted(leaked)}: channels the cut did not sever")
    return {
        "as_predicted": holds,
        "why": "; ".join(reasons) if reasons else (
            "cut apart, her domains read no integration and no synergy, and nothing held moved"
        ),
        "intact_phi_do": intact["phi_do"],
        "drift_phi_do": drift["phi_do"],
        "disconnected_phi_do": cut["phi_do"],
    }


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warm", type=int, default=6, help="rounds of ordinary life before the saved moment")
    parser.add_argument("--rounds", type=int, default=6, help="rounds of each life, eight conditions a round")
    args = parser.parse_args(argv)

    import numpy as np

    from core.subject.clamp import clamped, still
    from core.subject.driver import CONDITIONS, build_runtime, calibrate_clock, quiesce_organism, start_organism
    from core.subject.irreducibility import MIN_TRANSITIONS, phi_do
    from core.subject.isolation import isolate_state
    from core.subject.provenance import evidence_source, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS
    from core.subject.synergy import synergy_suite

    rounds = max(args.rounds, -(-(MIN_TRANSITIONS + 1) // max(1, len(CONDITIONS))))
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))
    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    isolate_state(run_dir)
    started = time.monotonic()
    source = evidence_source(args.seed)
    runtime = build_runtime(run_dir, seed=args.seed)
    await start_organism(runtime)
    await quiesce_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)
    for _ in range(args.warm):
        for condition in CONDITIONS:
            await runtime.turn_once(condition)
    await quiesce_organism(runtime)
    start = runtime.snapshot()

    async def live() -> list[Any]:
        frames: list[Any] = []
        for _ in range(rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        return frames

    def read(label: str, frames: list[Any]) -> dict[str, Any]:
        recording = build_recording(frames, notes={"arm": label}).by_turn()
        report = phi_do(recording)
        suite = synergy_suite(recording, seed=args.seed)
        return {
            "phi_do": float(report.phi),
            "standard_error": float(report.standard_error),
            "best_cut": ["".join(report.best_cut[0]), "".join(report.best_cut[1])],
            "degenerate": bool(report.degenerate),
            "synergy": {_triple(s): round(float(s.normalised), 5) for s in suite},
            "synergy_passes": {_triple(s): bool(s.passes) for s in suite},
        }

    try:
        intact = read("intact", await live())
        runtime.restore(start)
        drift = read("drift", await live())
        alone: dict[str, list[Any]] = {}
        leaks: dict[str, dict[str, float]] = {}
        for domain in DOMAINS:
            runtime.restore(start)
            held = [other for other in DOMAINS if other != domain]
            with clamped(runtime, held):
                frames = await live()
            alone[domain] = frames
            leaks[domain] = still(frames, held)
            print(f"[disconnected] {domain} alone: {len(frames)} frames, held columns moved: {len(leaks[domain])}", flush=True)
        cut = read("disconnected", compose_alone(alone, list(DOMAINS)))
    finally:
        await quiesce_organism(runtime)

    evidence = {
        "evidence_source": source,
        "seed": args.seed,
        "rounds": rounds,
        "conditions": [c.name for c in CONDITIONS],
        "intact": intact,
        "drift": drift,
        "disconnected": cut,
        "held_columns_that_moved": leaks,
        "verdict": verdict(intact, drift, cut, leaks),
        "seconds": round(time.monotonic() - started, 1),
    }
    out = run_dir / REPORT_FILE
    out.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"[disconnected] {json.dumps(evidence['verdict'])}", flush=True)
    print(f"[disconnected] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
