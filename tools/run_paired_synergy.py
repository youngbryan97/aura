#!/usr/bin/env python3
"""Whether two domains move a third jointly, by intervention rather than by information.

The synergy line reads the four declared triples off a recording, through an
estimator. A reviewer showed on 21 October 2026 that a nonlinear effect on a
target can fail that estimator's bars, and that a pass can come from the
target reading a register of the coupling rather than from its effect. This
asks the organism directly. From one moment, five forks: neither source
displaced, each alone, both, and neither again. The interaction is

    Y(both) - Y(first) - Y(second) + Y(neither)

in the target's state turns later, each column in its own baseline spread, and
the floor is the same arithmetic's size between the two forks with nothing
done. A triple whose interaction clears the floor across trials, by paired sign
flips, is a joint effect in her dynamics. Diagnostic beside the frozen line;
it counts toward no total.

    python tools/run_paired_synergy.py --rounds 4 --trials 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Sign flips for the paired test. The smallest reachable p is 1 / (draws + 1).
DRAWS = 20_000


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def _arm(runtime: Any, snapshot: Any, condition: Any, displace: tuple[str, ...], delta: float, turns: int) -> np.ndarray:
    """One fork: the named domains displaced together at the top of the first turn."""
    from core.subject.causal import SUSTAINED
    from core.subject.perturbation import perturb, perturb_organs

    runtime.restore(snapshot)
    sustained = any(domain in SUSTAINED for domain in displace)
    applied = {"done": not displace}

    async def hit(rt: Any) -> None:
        if not displace or (applied["done"] and not sustained):
            return
        for domain in displace:
            in_state = perturb(rt.state, domain, delta, ontogeny=rt.ontogeny)
            in_organ = await perturb_organs(rt.organs, domain, delta, state=rt.state)
            applied["done"] = applied["done"] or bool(in_state or in_organ)

    frames = []
    for turn in range(turns):
        frames.extend(
            await runtime.turn_once(
                condition,
                perturb_at=0 if (turn == 0 and displace) or sustained else None,
                perturb=hit,
                sustain=hit if sustained else None,
            )
        )
    return np.asarray(frames[-1].vector(), dtype=np.float64)


def sign_flip_p(values: np.ndarray, *, seed: int, draws: int = DRAWS) -> float:
    """One-sided p that the mean of paired differences is above zero."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 1.0
    observed = values.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, values.size))
    flipped = (signs * values).mean(axis=1)
    return float((np.sum(flipped >= observed) + 1) / (draws + 1))


def summarise(contrasts: list[np.ndarray], floors: list[np.ndarray], *, seed: int) -> dict[str, Any]:
    """The interaction's size against the floor's, trial by trial, and whether it keeps a direction."""
    size = np.array([np.linalg.norm(c) for c in contrasts])
    floor = np.array([np.linalg.norm(f) for f in floors])
    out: dict[str, Any] = {
        "trials": int(size.size),
        "interaction": round(float(size.mean()), 6) if size.size else 0.0,
        "floor": round(float(floor.mean()), 6) if floor.size else 0.0,
        "p_above_floor": round(sign_flip_p(size - floor, seed=seed), 6),
    }
    if len(contrasts) >= 3:
        # Each trial projected on the mean direction of the others, so the
        # direction is not fitted to the trial it scores.
        stacked = np.stack(contrasts)
        projections = []
        for index in range(len(contrasts)):
            rest = np.delete(stacked, index, axis=0).mean(axis=0)
            norm = float(np.linalg.norm(rest))
            projections.append(float(stacked[index] @ rest / norm) if norm > 0 else 0.0)
        out["p_consistent_direction"] = round(sign_flip_p(np.array(projections), seed=seed + 1), 6)
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "paired_synergy")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from core.subject.causal import DEFAULT_DELTA
    from core.subject.driver import CONDITIONS, build_runtime, calibrate_clock, quiesce_organism, start_organism
    from core.subject.isolation import isolate_state, state_leaks
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
    rng = random.Random(args.seed)
    slices = domain_slices()
    held: dict[str, dict[str, list[np.ndarray]]] = {
        f"{a},{b}->{y}": {"contrast": [], "floor": []} for a, b, y in TRIPLES
    }
    try:
        frames = []
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        spread = recording.x.std(axis=0)
        safe = np.where(spread > 1e-12, spread, np.inf)
        for trial in range(args.trials):
            for condition in CONDITIONS:
                await runtime.turn_once(condition)
                runtime.freeze_host()
                snapshot = runtime.snapshot()
                for a, b, y in TRIPLES:
                    arms = [("neither", ()), ("first", (a,)), ("second", (b,)), ("both", (a, b)), ("again", ())]
                    rng.shuffle(arms)
                    read = {}
                    for name, displace in arms:
                        vector = await _arm(runtime, snapshot, condition, displace, DEFAULT_DELTA, args.turns)
                        read[name] = vector[slices[y]] / safe[slices[y]]
                    key = f"{a},{b}->{y}"
                    held[key]["contrast"].append(read["both"] - read["first"] - read["second"] + read["neither"])
                    held[key]["floor"].append(read["again"] - read["neither"])
                runtime.restore(snapshot)
                runtime.thaw_host()
            _log(f"trial {trial + 1}/{args.trials}")
    finally:
        await quiesce_organism(runtime)
    evidence = {
        "environment": environment(),
        "design": {"rounds": args.rounds, "trials": args.trials, "turns": args.turns, "delta": DEFAULT_DELTA},
        "triples": {key: summarise(v["contrast"], v["floor"], seed=args.seed) for key, v in held.items()},
        "seconds": round(time.monotonic() - started, 1),
    }
    out = run_dir / "paired_synergy.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    for key, row in evidence["triples"].items():
        print(f"  {key:10s} interaction {row['interaction']:.4f} floor {row['floor']:.4f} "
              f"p {row['p_above_floor']:.4f} direction p {row.get('p_consistent_direction')}")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
