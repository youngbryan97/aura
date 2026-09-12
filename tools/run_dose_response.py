#!/usr/bin/env python3
"""Whether an edge is a channel or an artefact of one intervention size.

Every edge in the battery is measured at one displacement. If a channel only
appears at that displacement — absent below it, absent above it — then what was
measured is a threshold in the harness rather than a channel in the organism.
The way to tell is to turn the dial.

This runs the same paired intervention at several sizes and reports, per edge:
the effect at each, whether it grows with the dose, and whether it exists at
only one. An edge that appears at a single size and nowhere near it is named,
because that is the finding, and an effect that does not grow at all is named
too — a channel that saturates immediately is carrying a flag rather than a
quantity.

Both signs are run where the writer allows it. A displacement that produces the
same effect pushed either way is not carrying a direction, and that is worth
knowing about a channel the graph is treating as one.

The deltas are fixed here and the smallest is first, so the run cannot be
stopped once a flattering one has been seen.

    python tools/run_dose_response.py --trials 3
    python tools/run_dose_response.py --quick
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
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger("dose_response")

#: The sizes, smallest first, fixed before the run. The battery's own
#: displacement is in the middle of them so the curve brackets it.
DELTAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

#: An effect this far below the edge bar is absent for the purpose of asking
#: whether a channel appears at only one size.
ABSENT: float = 0.5


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _effects(results: Any) -> dict[str, float]:
    """Peak effect per source-to-target pair, floored by the sham arm."""
    from core.subject.state import DOMAINS

    out: dict[str, float] = {}
    for source in sorted({t.source for t in results.trials}):
        mine = [t for t in results.trials if t.source == source]
        for target in DOMAINS:
            if target == source:
                continue
            values = [
                t.effect.get(target, 0.0) - t.floor.get(target, 0.0) for t in mine
            ]
            if values:
                out[f"{source}->{target}"] = round(float(np.mean(values)), 5)
    return out


def _curve(by_delta: dict[float, dict[str, float]], bar: float) -> dict[str, Any]:
    """Per edge: the effect at each dose, and what the shape of that says."""
    deltas = sorted(by_delta)
    edges = sorted({key for row in by_delta.values() for key in row})
    rows: dict[str, Any] = {}
    for edge in edges:
        values = [by_delta[delta].get(edge, 0.0) for delta in deltas]
        present = [value >= bar for value in values]
        # Monotone in the dose, allowing the noise one step back. A channel
        # that grows with what is pushed into it is a channel; one that is flat
        # across a fourfold change in dose is carrying a flag.
        grows = values[-1] > values[0]
        rank = float(np.corrcoef(np.arange(len(values)), values)[0, 1]) if len(values) > 2 else 0.0
        rows[edge] = {
            "effects": values,
            "present": present,
            "present_at": [deltas[i] for i, flag in enumerate(present) if flag],
            "grows_with_dose": bool(grows),
            "dose_correlation": round(rank if rank == rank else 0.0, 4),
            # The failure this experiment exists to find.
            "only_at_one_dose": bool(sum(present) == 1),
            # And its neighbour: an effect that is there at every dose and the
            # same size at all of them.
            "flat_across_the_range": bool(
                all(present)
                and max(values) > 0
                and (max(values) - min(values)) / max(values) < 0.1
            ),
        }
    return {
        "deltas": deltas,
        "bar": bar,
        "edges": rows,
        "edges_at_one_dose_only": sorted(k for k, v in rows.items() if v["only_at_one_dose"]),
        "edges_flat_across_the_range": sorted(
            k for k, v in rows.items() if v["flat_across_the_range"]
        ),
        "edges_growing_with_dose": sorted(k for k, v in rows.items() if v["grows_with_dose"]),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "dose_response")
    parser.add_argument("--rounds", type=int, default=12, help="baseline turns per condition")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--both-signs", action="store_true",
        help="run the negative displacement too, so a channel that does not "
             "carry a direction is visible",
    )
    args = parser.parse_args()

    if args.quick:
        args.rounds, args.trials, args.turns = 3, 1, 1

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.battery import THRESHOLDS
    from core.subject.causal import run_interventions
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.provenance import environment, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Its own state root, before the organism is built. Sharing one with other
    # runs made every run start from what the ones before it had trained; see
    # core.subject.isolation.
    from core.subject.isolation import isolate_state, state_leaks

    isolate_state(run_dir)
    started = time.monotonic()
    _log(f"dose-response run {run_dir.name}")

    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(
            f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}"
        )
    await start_organism(runtime)
    await calibrate_clock(runtime, CONDITIONS)

    evidence: dict[str, Any] = {
        "environment": environment(),
        "design": {
            "deltas": list(DELTAS),
            "trials": args.trials,
            "turns_per_arm": args.turns,
            "both_signs": bool(args.both_signs),
            "note": (
                "The sizes are fixed here and the smallest runs first, so the "
                "run cannot be stopped once a flattering one has been seen."
            ),
        },
    }

    try:
        _log(f"baseline: {args.rounds} rounds over {len(CONDITIONS)} conditions")
        frames = []
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
        recording = build_recording(frames)
        scale = {
            key: recording.x[:, recording.slices[key]].std(axis=0) for key in DOMAINS
        }

        signs = (1.0, -1.0) if args.both_signs else (1.0,)
        by_sign: dict[str, dict[float, dict[str, float]]] = {}
        for sign in signs:
            label = "positive" if sign > 0 else "negative"
            by_delta: dict[float, dict[str, float]] = {}
            for delta in DELTAS:
                _log(f"{label} displacement at delta {delta}")
                results = await run_interventions(
                    runtime, CONDITIONS, scale=scale, trials=args.trials,
                    turns=args.turns, delta=sign * delta, seed=args.seed,
                )
                by_delta[delta] = _effects(results)
                reached = sum(
                    1 for v in by_delta[delta].values() if v >= THRESHOLDS["edge_effect"]
                )
                _log(f"  {reached} pair(s) at or above the edge bar")
            by_sign[label] = by_delta

        bar = float(THRESHOLDS["edge_effect"])
        evidence["curves"] = {
            label: _curve(by_delta, bar) for label, by_delta in by_sign.items()
        }
        if len(signs) > 1:
            evidence["direction"] = _direction(by_sign, bar)
        evidence["verdict"] = _verdict(evidence)
    finally:
        await quiesce_organism(runtime)

    evidence["seconds"] = round(time.monotonic() - started, 1)
    out = run_dir / "dose_response.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")

    _log("")
    for line in _lines(evidence):
        print(line)
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0 if evidence["verdict"]["no_edge_exists_at_only_one_dose"] else 1


def _direction(by_sign: dict[str, dict[float, dict[str, float]]], bar: float) -> dict[str, Any]:
    """Edges that answer the same way whichever way they are pushed.

    A displacement that produces the same effect in both directions is not
    carrying a direction, which is worth knowing about a channel the graph is
    treating as one.
    """
    out: dict[str, Any] = {}
    positive, negative = by_sign.get("positive", {}), by_sign.get("negative", {})
    shared = sorted(set(positive) & set(negative))
    for edge in sorted({k for d in positive.values() for k in d}):
        up = [positive[d].get(edge, 0.0) for d in shared]
        down = [negative[d].get(edge, 0.0) for d in shared]
        if not up or not down:
            continue
        out[edge] = {
            "positive": up,
            "negative": down,
            "kept_either_way": bool(max(up) >= bar and max(down) >= bar),
            "one_direction_only": bool((max(up) >= bar) != (max(down) >= bar)),
        }
    return out


def _verdict(evidence: dict[str, Any]) -> dict[str, Any]:
    curves = evidence.get("curves", {})
    at_one = sorted({e for c in curves.values() for e in c["edges_at_one_dose_only"]})
    flat = sorted({e for c in curves.values() for e in c["edges_flat_across_the_range"]})
    growing = sorted({e for c in curves.values() for e in c["edges_growing_with_dose"]})
    return {
        "no_edge_exists_at_only_one_dose": not at_one,
        "edges_at_one_dose_only": at_one,
        "edges_flat_across_the_range": flat,
        "edges_growing_with_dose": growing,
        "note": (
            "An edge present at one displacement and absent at the sizes either "
            "side of it is a threshold in the harness rather than a channel in "
            "the organism. One that is the same size across a fourfold change "
            "in dose is carrying a flag rather than a quantity."
        ),
    }


def _lines(evidence: dict[str, Any]) -> list[str]:
    verdict = evidence["verdict"]
    lines = [
        "dose-response:",
        f"  edges growing with dose:        {len(verdict['edges_growing_with_dose'])}",
        f"  edges at one dose only:         {verdict['edges_at_one_dose_only'] or 'none'}",
        f"  edges flat across the range:    {verdict['edges_flat_across_the_range'] or 'none'}",
        f"  no edge exists at only one dose: {verdict['no_edge_exists_at_only_one_dose']}",
    ]
    for label, curve in evidence.get("curves", {}).items():
        strongest = sorted(
            curve["edges"].items(), key=lambda kv: -max(kv[1]["effects"])
        )[:6]
        lines.append(f"  {label}, the six strongest:")
        for edge, row in strongest:
            shape = " ".join(f"{v:7.3f}" for v in row["effects"])
            lines.append(f"    {edge:10} {shape}   grows: {row['grows_with_dose']}")
    return lines


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
