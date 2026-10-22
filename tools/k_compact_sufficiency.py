#!/usr/bin/env python3
"""Whether the eight numbers K reads of an array are enough to say what it does next.

K reads each of her large arrays (the substrate's state and weights, the mesh,
the field, the self-predictor's Gram matrix) through `core.subject.sketch`: four
moments and four fixed projections. A reviewer showed on 21 October 2026 that
two weight matrices with the same eight numbers can send the same state to
next states 1.72 apart. Whether that matters for her is a measurement, not an
argument, so this makes it in her organism.

For each sketched array: two forks of one moment, the array moved by +e v in
one and -e v in the other, where v is orthogonal to the sketch's projections,
to the array itself and to its constant, and zero at its extremes. The two
forks read identically through the sketch (checked, not assumed) and differ in
the array. One turn later, how far apart is K? Against two references taken
from the same moment: two forks with nothing moved (the floor), and two forks
moved by the same amount along a direction the sketch does see (the scale).
The size e v is one turn's ordinary change of that array, measured on the
baseline, so the question is whether a turn's learning the sketch cannot see
changes what she does.

A hidden divergence at the floor says the sketch carried what this array does
in a turn. One near the visible divergence says K is blind to most of it, and
the closure test's leaks from that organ are K's blindness rather than a cause
from outside.

    python tools/k_compact_sufficiency.py --rounds 3 --pairs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _resolve(organs: Any, source: str) -> tuple[Any, str, Any]:
    organ_name, _, path = source[len("organ:"):].partition(".")
    holder = getattr(organs, organ_name, None)
    parts = path.split(".")
    for part in parts[:-1]:
        holder = getattr(holder, part, None) if holder is not None else None
    value = getattr(holder, parts[-1], None) if holder is not None else None
    return holder, parts[-1], value


#: How many directions the sketch projects on.
SKETCH_LOOKS = 4


def hidden_direction(array: np.ndarray, rng: np.random.Generator, axes: Any = None) -> np.ndarray:
    """A unit direction the sketch cannot see, zero at the array's extremes.

    Orthogonal to the directions the sketch projects on: the calibrated axes
    when the run fixed some for this array, the frozen random ones otherwise.
    """
    from core.subject.sketch import _directions

    flat = np.asarray(array, dtype=np.float64).ravel()
    size = flat.size
    # Every entry at the minimum or the maximum, ties included: a buffer that
    # is mostly zeros has thousands at its minimum, and moving any of them
    # moves the minimum the sketch reads.
    extremes = np.flatnonzero((flat == flat.min()) | (flat == flat.max()))
    free = np.setdiff1d(np.arange(size), extremes)
    if free.size <= SKETCH_LOOKS + 2:
        return np.zeros(np.shape(array))
    looks = np.asarray(axes[1]) if axes is not None else _directions(size)
    basis = np.vstack([*looks, np.ones(size), flat - flat.mean()])[:, free]
    v = np.zeros(size)
    draw = rng.standard_normal(free.size)
    draw -= basis.T @ np.linalg.lstsq(basis @ basis.T, basis @ draw, rcond=None)[0]
    v[free] = draw
    norm = float(np.linalg.norm(v))
    return (v / norm).reshape(np.shape(array)) if norm > 0 else v.reshape(np.shape(array))


def visible_direction(array: np.ndarray, axes: Any = None) -> np.ndarray:
    """A unit direction the sketch does see: its first projection, kept off the extremes."""
    from core.subject.sketch import _directions

    flat = np.asarray(array, dtype=np.float64).ravel()
    d = np.array(np.asarray(axes[1])[0] if axes is not None else _directions(flat.size)[0], dtype=np.float64)
    d[(flat == flat.min()) | (flat == flat.max())] = 0.0
    norm = float(np.linalg.norm(d))
    return (d / norm).reshape(np.shape(array)) if norm > 0 else d.reshape(np.shape(array))


def _room(array: np.ndarray, direction: np.ndarray) -> float:
    """The largest step either way along `direction` that leaves the array's extremes where they are."""
    return min(_room_one_way(array, direction), _room_one_way(array, -np.asarray(direction)))


def _room_one_way(array: np.ndarray, direction: np.ndarray) -> float:
    flat, step = np.asarray(array, dtype=np.float64).ravel(), np.asarray(direction).ravel()
    low, high = flat.min(), flat.max()
    moving = np.abs(step) > 1e-15
    if not moving.any():
        return 0.0
    up = np.where(step[moving] > 0, (high - flat[moving]) / step[moving], (flat[moving] - low) / -step[moving])
    return float(np.min(up))


def _set(holder: Any, attr: str, value: np.ndarray) -> None:
    """Write the array back by assignment, so a cache keyed on its identity refreshes."""
    setattr(holder, attr, np.array(value, dtype=np.asarray(getattr(holder, attr)).dtype, copy=True))


def _writable(holder: Any, attr: str) -> bool:
    """Whether the attribute can be written back: a computed property cannot."""
    found = getattr(type(holder), attr, None)
    return not (isinstance(found, property) and found.fset is None)


async def _fork_reading(runtime: Any, snapshot: Any, condition: Any, change: Any = None) -> np.ndarray:
    runtime.restore(snapshot)
    if change is not None:
        change()
    frames = await runtime.turn_once(condition)
    return np.asarray(frames[-1].vector(), dtype=np.float64)


def _distance(a: np.ndarray, b: np.ndarray, spread: np.ndarray) -> float:
    safe = np.where(spread > 1e-12, spread, np.inf)
    return float(np.sqrt(np.mean(((a - b) / safe) ** 2)))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "k_compact_sufficiency")
    parser.add_argument("--rounds", type=int, default=3, help="baseline rounds over the conditions")
    parser.add_argument("--pairs", type=int, default=3, help="moments probed per array")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--calibrated-axes", action="store_true",
        help="fix each array's leading axes on a pass before probing, as a run with the option does",
    )
    args = parser.parse_args()

    from core.subject.calibrated_axes import axes_for, sketched_sources
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.isolation import isolate_state, state_leaks
    from core.subject.provenance import environment, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.sketch import sketch

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
    rng = np.random.default_rng(args.seed)
    sources = sketched_sources()
    evidence: dict[str, Any] = {"environment": environment(), "sources": sources, "arrays": {}}
    try:
        _log(f"baseline: {args.rounds} rounds over {len(CONDITIONS)} conditions")
        frames, history = [], {source: [] for source in sources}
        for _ in range(args.rounds):
            for condition in CONDITIONS:
                frames.extend(await runtime.turn_once(condition))
                for source in sources:
                    _, _, value = _resolve(runtime.organs, source)
                    if isinstance(value, np.ndarray) and value.size:
                        history[source].append(np.array(value, dtype=np.float64, copy=True))
        recording = build_recording(frames)
        spread = recording.x.std(axis=0)
        if args.calibrated_axes:
            from core.subject.calibrated_axes import calibration_pass

            evidence["calibrated_axes"] = await calibration_pass(runtime, CONDITIONS)
        # As the battery does before its interventions: the free-running loops
        # stop, so two forks from one moment see the same computation.
        evidence["stopped_loops"] = await quiesce_organism(runtime)
        for source in sources:
            seen = history[source]
            steps = [float(np.linalg.norm(b - a)) for a, b in zip(seen, seen[1:], strict=False) if a.shape == b.shape]
            turn_change = float(np.median(steps)) if steps else 0.0
            row: dict[str, Any] = {"size": int(seen[-1].size) if seen else 0, "turn_change": round(turn_change, 8)}
            evidence["arrays"][source] = row
            if not seen or turn_change <= 0.0:
                row["verdict"] = "does not change between turns"
                continue
            moments = []
            for pair in range(args.pairs):
                condition = CONDITIONS[pair % len(CONDITIONS)]
                await runtime.turn_once(condition)
                runtime.freeze_host()
                snapshot = runtime.snapshot()
                holder, attr, value = _resolve(runtime.organs, source)
                if not _writable(holder, attr):
                    runtime.thaw_host()
                    moments.append({"skipped": "computed from other state, so it cannot be moved on its own"})
                    break
                base = np.array(value, dtype=np.float64, copy=True)
                hidden = hidden_direction(base, rng, axes_for(source))
                visible = visible_direction(base, axes_for(source))
                size = min(turn_change, _room(base, hidden), _room(base, visible))
                if size <= 0.0:
                    runtime.thaw_host()
                    moments.append({"skipped": "no room inside the extremes"})
                    continue
                plus, minus = base + size * hidden, base - size * hidden
                unseen = max(abs(sketch(plus)[k] - sketch(minus)[k]) for k in sketch(plus))
                floor = _distance(
                    await _fork_reading(runtime, snapshot, condition),
                    await _fork_reading(runtime, snapshot, condition),
                    spread,
                )
                hidden_gap = _distance(
                    await _fork_reading(runtime, snapshot, condition, partial(_set, holder, attr, plus)),
                    await _fork_reading(runtime, snapshot, condition, partial(_set, holder, attr, minus)),
                    spread,
                )
                visible_gap = _distance(
                    await _fork_reading(runtime, snapshot, condition, partial(_set, holder, attr, base + size * visible)),
                    await _fork_reading(runtime, snapshot, condition, partial(_set, holder, attr, base - size * visible)),
                    spread,
                )
                runtime.restore(snapshot)
                runtime.thaw_host()
                moments.append({
                    "condition": condition.name,
                    "step": round(size, 8),
                    "sketch_difference": unseen,
                    "floor": round(floor, 6),
                    "hidden": round(hidden_gap, 6),
                    "visible": round(visible_gap, 6),
                })
                _log(f"{source}: floor {floor:.4f} hidden {hidden_gap:.4f} visible {visible_gap:.4f} sketch diff {unseen:.2e}")
            row["moments"] = moments
            measured = [m for m in moments if "hidden" in m]
            if measured:
                hidden_mean = float(np.mean([m["hidden"] for m in measured]))
                floor_max = float(max(m["floor"] for m in measured))
                visible_mean = float(np.mean([m["visible"] for m in measured]))
                row["hidden_over_visible"] = round(hidden_mean / visible_mean, 4) if visible_mean > 0 else None
                row["verdict"] = (
                    "the sketch carried what a turn of this array does"
                    if hidden_mean <= floor_max
                    else "K is blind to some of what this array does in a turn"
                )
    finally:
        await quiesce_organism(runtime)
    evidence["seconds"] = round(time.monotonic() - started, 1)
    out = run_dir / "k_compact_sufficiency.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    for source, row in evidence["arrays"].items():
        print(f"  {source:40s} {row.get('verdict', '')}  hidden/visible {row.get('hidden_over_visible')}")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
