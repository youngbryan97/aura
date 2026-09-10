#!/usr/bin/env python3
"""Displace one domain and say, column by column, where the effect died.

The battery answers whether an edge is retained. When it is not, the next
question is always the same: did the displacement move the source at all, did
anything downstream move, and which column carried it. A full run costs
seventy-five minutes and answers that once. This costs a few minutes and
answers it for the domains named on the command line, which is what a repair
loop needs.

It is a diagnostic, not evidence. The trial count is small, no multiplicity
correction is applied, and nothing here decides whether an edge exists. Read
the sign of a channel from it; read the existence of an edge from a run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

import numpy as np  # noqa: E402


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="*", help="domains to displace (default: all)")
    parser.add_argument("--rounds", type=int, default=6, help="baseline turns per condition")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conditions", default="", help="comma-separated subset")
    parser.add_argument("--columns", type=int, default=6, help="columns to name per target")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--where",
        action="store_true",
        help="with --census, name the phase at which each column first diverges",
    )
    parser.add_argument(
        "--census",
        action="store_true",
        help="run two shams and rank the columns that move without any displacement",
    )
    args = parser.parse_args()

    out = args.out or (REPO / "artifacts" / "subject_core" / "probe")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(out / "logs"))

    from core.subject.causal import DEFAULT_DELTA, _arm
    from core.subject.driver import CONDITIONS, build_runtime, start_organism
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS, feature_names

    delta = DEFAULT_DELTA if args.delta is None else args.delta
    conditions = list(CONDITIONS)
    if args.conditions:
        wanted = {name.strip() for name in args.conditions.split(",") if name.strip()}
        conditions = [c for c in conditions if c.name in wanted] or list(CONDITIONS)
    sources = [s for s in (args.sources or list(DOMAINS)) if s in DOMAINS]

    _log(f"building the offline organism in {out}")
    runtime = build_runtime(out / "runtime", seed=args.seed)
    organism = await start_organism(runtime, quiet=True)
    _log(f"organism up: {len(organism['up'])} layers, {len(organism['down'])} down")

    _log(f"recording {args.rounds} rounds over {len(conditions)} conditions")
    frames = []
    for _ in range(args.rounds):
        for condition in conditions:
            frames.extend(await runtime.turn_once(condition))
    recording = build_recording(frames, notes={"rounds": args.rounds})
    scale = {key: recording.domain(key).std(axis=0) for key in DOMAINS}
    names = {key: feature_names(key) for key in DOMAINS}

    if args.census:
        return await _census(runtime, conditions, scale, names, args)

    gathered: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in sources}
    for index in range(args.trials):
        for condition in conditions:
            await runtime.turn_once(condition)
            runtime.freeze_host()
            snapshot = runtime.snapshot()
            for source in sources:
                arms = {}
                for name, displace in (
                    ("pert", (source, delta)),
                    ("sham_a", None),
                    ("sham_b", None),
                ):
                    arms[name] = await _arm(
                        runtime, snapshot, condition, turns=args.turns, displace=displace, at=0
                    )
                if not arms["pert"]:
                    continue
                gathered[source].append(
                    {
                        "effect": _columns(arms["pert"], arms["sham_a"], scale),
                        "floor": _columns(arms["sham_a"], arms["sham_b"], scale),
                    }
                )
            runtime.restore(snapshot)
            runtime.thaw_host()
            _log(f"  trial {index + 1}/{args.trials} {condition.name}")

    print()
    for source in sources:
        runs = gathered[source]
        if not runs:
            print(f"{source}: no writer bit — the domain is unwritable here\n")
            continue
        print(f"{source} displaced by {delta:+.3f}, {len(runs)} trials")
        for target in DOMAINS:
            effect = np.mean([r["effect"][target] for r in runs], axis=0)
            floor = np.mean([r["floor"][target] for r in runs], axis=0)
            peak = float(effect.max()) if effect.size else 0.0
            base = float(floor.max()) if floor.size else 0.0
            tag = "self" if target == source else "    "
            note = ""
            if target != source:
                if peak <= base + 1e-9:
                    note = "   the sham moved it as much"
                elif peak - base >= 0.30:
                    note = "   above the bar"
            print(f"  {tag} {source}->{target}: {peak:8.4f} against floor {base:8.4f}{note}")
            if target == source or peak < 1e-6:
                continue
            label = names.get(target, ())
            order = np.argsort(-(effect - floor))[: args.columns]
            rendered = ", ".join(
                f"{label[i] if i < len(label) else i}={effect[i]:.2f}/{floor[i]:.2f}"
                for i in order
                if effect[i] > 1e-6
            )
            if rendered:
                print(f"         {rendered}")
        print()
    return 0



async def _census(runtime, conditions, scale, names, args) -> int:
    """What moves when nothing is displaced.

    Every effect the battery reports is a displaced arm against a sham, minus
    what two shams do to each other. A column that differs between two shams
    is nondeterminism in the fork, and it enters the threshold of every edge
    into that column. Ranked here so the largest one can be found and removed
    rather than averaged into a floor.
    """
    from core.subject.causal import _arm
    from core.subject.state import DOMAINS

    runs = []
    first: list[dict[str, str]] = []
    for index in range(args.trials):
        for condition in conditions:
            await runtime.turn_once(condition)
            runtime.freeze_host()
            snapshot = runtime.snapshot()
            a = await _arm(runtime, snapshot, condition, turns=args.turns, displace=None, at=0)
            b = await _arm(runtime, snapshot, condition, turns=args.turns, displace=None, at=0)
            runs.append(_columns(a, b, scale))
            if args.where:
                first.append(_first_divergence(a, b, scale, names))
            runtime.restore(snapshot)
            runtime.thaw_host()
            _log(f"  sham pair {index + 1}/{args.trials} {condition.name}")

    print()
    print("what two identical arms do to each other, largest first")
    rows = []
    for domain in DOMAINS:
        mean = np.mean([r[domain] for r in runs], axis=0)
        label = names.get(domain, ())
        for i, value in enumerate(mean):
            rows.append((float(value), f"{label[i] if i < len(label) else i}"))
    rows.sort(key=lambda kv: -kv[0])
    for value, name in rows[:40]:
        print(f"  {value:8.4f}  {name}")
    print()
    for domain in DOMAINS:
        mean = np.mean([r[domain] for r in runs], axis=0)
        print(f"  {domain}: peak {float(mean.max()) if mean.size else 0.0:.4f}")
    if first:
        print()
        print("the phase after which each column first differs between two shams")
        counted: dict[str, dict[str, int]] = {}
        for run in first:
            for column, tag in run.items():
                counted.setdefault(column, {}).setdefault(tag, 0)
                counted[column][tag] += 1
        ranked = {name: value for value, name in rows}
        for column in sorted(counted, key=lambda c: -ranked.get(c, 0.0))[:24]:
            where = ", ".join(
                f"{tag}x{n}" for tag, n in sorted(counted[column].items(), key=lambda kv: -kv[1])
            )
            print(f"  {ranked.get(column, 0.0):8.4f}  {column}: {where}")
    return 0


def _first_divergence(left, right, scale, names) -> dict[str, str]:
    """For every column that ever differs, the frame tag where it first does."""
    from core.subject.causal import SCALE_FLOOR
    from core.subject.state import DOMAINS

    out: dict[str, str] = {}
    span = min(len(left), len(right))
    for domain in DOMAINS:
        unit = np.asarray(scale.get(domain, np.zeros(0)), dtype=np.float64)
        if unit.size == 0:
            continue
        label = names.get(domain, ())
        live = unit > SCALE_FLOOR
        for index in range(span):
            gap = np.abs(left[index].domain(domain) - right[index].domain(domain))
            moved = np.where(live & (gap > 1e-9))[0]
            for i in moved:
                key = f"{label[i] if i < len(label) else i}"
                out.setdefault(key, str(getattr(left[index], "tag", index)))
    return out


def _columns(left, right, scale: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Largest per-column displacement across the arm, in units of ordinary spread."""
    from core.subject.causal import DIVERGENCE_CEILING, SCALE_FLOOR
    from core.subject.state import DOMAINS

    out: dict[str, np.ndarray] = {}
    span = min(len(left), len(right))
    for domain in DOMAINS:
        unit = np.asarray(scale.get(domain, np.zeros(0)), dtype=np.float64)
        if unit.size == 0:
            out[domain] = unit
            continue
        live = unit > SCALE_FLOOR
        peak = np.zeros(unit.size)
        for index in range(span):
            gap = np.abs(left[index].domain(domain) - right[index].domain(domain))
            scaled = np.zeros(unit.size)
            scaled[live] = np.clip(gap[live] / unit[live], 0.0, DIVERGENCE_CEILING)
            peak = np.maximum(peak, scaled)
        out[domain] = peak
    return out


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
