#!/usr/bin/env python3
"""The shape of her internal content space, and nothing about how it feels.

Phenomenal structuralism says a quality is its complete position in the web of
relations to every other quality: red is closer to orange than to green, and
that whole pattern of closeness is what red is. The testable half of that is
the geometry — whether the internal causal distances between contents predict
the system's own discriminations, and whether moving the geometry moves the
discriminations in the direction predicted before the move.

This measures the internal half. It presents matched percept classes, estimates
each one's causal-state distribution, and computes Fisher–Rao distances between
them. Then it asks whether those distances predict what the system does with
the classes, and perturbs the manifold with a direction registered first.

What it cannot do is say the geometry is felt. Two phenomenal assignments
differing only by a relabelling that preserves every relation produce identical
observations, so absolute quale labels are gauge — identifiable only up to
structure-preserving isomorphism, and not at all from third-person data. Every
number here is internal content geometry. The report says so in those words.

    python tools/run_content_geometry.py --repeats 8
    python tools/run_content_geometry.py --quick
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

logger = logging.getLogger("content_geometry")

#: The classes, and what arrives for each. Fixed before the run: a class list
#: chosen after seeing which pairs came out close would be drawing the map
#: around the territory. Three sensory kinds and three contents each, so within
#: class and between class are both measurable.
CLASSES: dict[str, tuple[str, ...]] = {
    "greeting": (
        "Bryan said hello",
        "Bryan said good morning",
        "Bryan said hi there",
    ),
    "question": (
        "Bryan asked how it works",
        "Bryan asked what happens next",
        "Bryan asked why that follows",
    ),
    "alarm": (
        "the disk is nearly full",
        "a process stopped responding",
        "the temperature is climbing",
    ),
}

#: What the percepts are delivered as, so the class is the content and not the
#: channel. All three arrive the same way.
KIND: str = "interaction"
SOURCE: str = "chat"

#: Which domains the content reaches. The geometry is measured over what the
#: contents reach rather than over perception alone, because a content that
#: only moved the percept stream has not become content.
READ: tuple[str, ...] = ("P", "A", "G", "M", "W")

#: How many times a class is split against itself to build the floor, and which
#: quantile of that is the bar. A discriminator on twelve rows a side reads
#: anywhere from 0.59 to 1.59 with nothing to find, so a handful of draws bounds
#: none of it.
FLOOR_DRAWS: int = 24
FLOOR_QUANTILE: float = 0.95

#: The perturbation, and the direction registered before it is applied. Raising
#: what the workspace is attending to towards one class should pull that class
#: towards the attended one and leave the pair that does not involve it alone.
PREREGISTERED: str = (
    "attending to the alarm class should shorten alarm-to-other distances "
    "relative to the greeting-to-question distance, which involves neither"
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _vector(runtime: Any) -> np.ndarray:
    from core.subject.state import read_core_state

    reading = read_core_state(
        runtime.state, ontogeny=runtime.ontogeny, organs=runtime.organs
    )
    return np.concatenate([np.asarray(reading.values[key]) for key in READ])


async def _present(
    runtime: Any, condition: Any, content: str, *, attend: str | None = None
) -> np.ndarray:
    """Deliver one content and read what the core did with it."""
    from core.state.percepts import emit_percept, reweight_stream

    emit_percept(
        runtime.state.world, KIND, content=content, intensity=0.5, source=SOURCE
    )
    if attend:
        # The manipulation: the workspace is holding this, so an arriving
        # percept sharing content with it is biased upward. Same mechanism the
        # runtime uses, driven here rather than waiting for the competition to
        # land on it.
        reweight_stream(runtime.state.world, {"content": attend, "priority": 0.9})
    await runtime.turn_once(condition)
    return _vector(runtime)


async def _bank(
    runtime: Any,
    condition: Any,
    snapshot: Any,
    *,
    repeats: int,
    attend: str | None = None,
) -> dict[str, np.ndarray]:
    """One sample block per class, every block taken from the same fork."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in CLASSES}
    for repeat in range(repeats):
        for name, contents in CLASSES.items():
            runtime.restore(snapshot)
            content = contents[repeat % len(contents)]
            out[name].append(await _present(runtime, condition, content, attend=attend))
    return {name: np.vstack(rows) for name, rows in out.items()}


def _geometry(bank: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    """Fisher–Rao distance between every pair of classes, and within each one."""
    from core.subject.intrinsic_v25 import crossfit_fisher_rao

    names = sorted(bank)
    # Both comparisons at the same sample size. A discriminator finds more
    # separation with more rows, so a floor estimated on half the data and a
    # signal estimated on all of it are not comparable: two of three classes
    # drawn from one distribution cleared a floor built that way. The floor
    # splits a class in half, so the signal uses halves too.
    half = min(len(rows) for rows in bank.values()) // 2
    between: dict[str, float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            estimate = crossfit_fisher_rao(
                bank[left][:half], bank[right][:half], seed=seed
            )
            between[f"{left}|{right}"] = round(float(estimate.distance), 6)

    # The floor, as a distribution rather than as three draws. A class split in
    # half against itself is the same law twice, so what the estimator reads
    # there is its own noise — and at twelve rows a side that noise ran from
    # 0.59 to 1.59 in twenty draws. Three samples of a spread like that bound
    # nothing, and two of three classes from one distribution cleared a ceiling
    # built from them. The split is redrawn many times and the bar is the high
    # quantile of what comes out.
    rng = np.random.default_rng(seed + 1)
    draws: list[float] = []
    within: dict[str, float] = {}
    for name, rows in bank.items():
        if half < 3:
            continue
        here: list[float] = []
        for draw in range(FLOOR_DRAWS):
            order = rng.permutation(len(rows))
            left_rows = rows[order[:half]]
            right_rows = rows[order[half : 2 * half]]
            here.append(
                float(
                    crossfit_fisher_rao(
                        left_rows, right_rows, seed=int(seed + 2 + draw)
                    ).distance
                )
            )
        draws.extend(here)
        within[name] = round(float(np.quantile(here, FLOOR_QUANTILE)), 6)
    # The largest within-class distance, not the mean of them. Half of a set of
    # noisy numbers sits above their own mean, so "further apart than the mean
    # of the floor" is a coin flip: three classes drawn from one distribution
    # all cleared it. The ceiling is what the estimator does when there is
    # nothing to find.
    floor = round(float(np.quantile(draws, FLOOR_QUANTILE)), 6) if draws else 0.0
    return {
        "between": between,
        "within": within,
        "floor": floor,
        "floor_rule": (
            f"the {FLOOR_QUANTILE:.0%} quantile of {FLOOR_DRAWS} within-class "
            "splits per class, at the same sample size as the between-class "
            "distances: what the estimator reads with nothing to find"
        ),
        "floor_draws": len(draws),
        "samples_per_side": half,
        "separated": sorted(k for k, v in between.items() if v > floor),
    }


def _behaviour(bank: dict[str, np.ndarray]) -> dict[str, Any]:
    """What the system does with each class, read off the state rather than words.

    Not a report and not a label: the mean reading each class leaves across the
    domains it reaches. Two classes that leave the same trace are two classes
    the system did not discriminate, whatever it would say about them.
    """
    names = sorted(bank)
    centres = {name: bank[name].mean(axis=0) for name in names}
    out: dict[str, float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            a, b = centres[left], centres[right]
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            out[f"{left}|{right}"] = round(
                float(1.0 - (a @ b) / norm) if norm > 1e-12 else 0.0, 6
            )
    return out


def _agreement(geometry: dict[str, float], behaviour: dict[str, float]) -> dict[str, Any]:
    """Whether the causal distances order the pairs the way the traces do.

    Rank agreement rather than a fit. Three classes give three pairs, which is
    too few for a slope to mean anything and enough to ask whether the ordering
    is the same one.
    """
    shared = sorted(set(geometry) & set(behaviour))
    if len(shared) < 3:
        return {"pairs": len(shared), "note": "too few pairs to order"}
    causal = [geometry[k] for k in shared]
    traced = [behaviour[k] for k in shared]
    order_a = np.argsort(np.argsort(causal))
    order_b = np.argsort(np.argsort(traced))
    agree = int((order_a == order_b).sum())
    return {
        "pairs": len(shared),
        "same_order": agree == len(shared),
        "ranks_matching": agree,
        "causal": dict(zip(shared, causal, strict=True)),
        "trace": dict(zip(shared, traced, strict=True)),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "content_geometry")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.repeats, args.warmup = 6, 2

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        quiesce_organism,
        start_organism,
    )
    from core.subject.provenance import environment, next_run_directory

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Its own state root, before the organism is built. Sharing one with other
    # runs made every run start from what the ones before it had trained; see
    # core.subject.isolation.
    from core.subject.isolation import isolate_state, state_leaks

    isolate_state(run_dir)
    started = time.monotonic()
    _log(f"content geometry run {run_dir.name}")

    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(
            f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}"
        )
    await start_organism(runtime)
    condition = CONDITIONS[0]

    evidence: dict[str, Any] = {
        "environment": environment(),
        "design": {
            "classes": {k: list(v) for k, v in CLASSES.items()},
            "read_over": list(READ),
            "repeats": args.repeats,
            "preregistered_direction": PREREGISTERED,
        },
        "what_this_is": (
            "internal content geometry. Every number here is a distance between "
            "causal-state distributions inside one system. None of it is "
            "evidence that the geometry is felt, and no third-person "
            "measurement could be: two phenomenal assignments differing only by "
            "a relabelling that preserves every relation produce identical "
            "observations, so absolute quale labels are gauge."
        ),
    }

    try:
        _log(f"warming up {args.warmup} turns")
        for turn in range(args.warmup):
            await runtime.turn_once(CONDITIONS[turn % len(CONDITIONS)])
        fork = runtime.snapshot()

        _log(f"presenting {len(CLASSES)} classes x {args.repeats} repeats")
        bank = await _bank(runtime, condition, fork, repeats=args.repeats)
        geometry = _geometry(bank, seed=args.seed)
        behaviour = _behaviour(bank)
        evidence["geometry"] = geometry
        evidence["trace"] = behaviour
        evidence["agreement"] = _agreement(geometry["between"], behaviour)
        for pair, value in sorted(geometry["between"].items()):
            _log(f"  {pair:22} {value:.5f}   (floor {geometry['floor']:.5f})")

        _log("perturbing the manifold in the direction registered first")
        runtime.restore(fork)
        attended = await _bank(
            runtime, condition, fork, repeats=args.repeats, attend=CLASSES["alarm"][0]
        )
        moved = _geometry(attended, seed=args.seed)
        evidence["perturbed"] = moved
        evidence["prediction"] = _prediction(geometry["between"], moved["between"])
        _log(
            "  the registered direction held: "
            f"{evidence['prediction']['the_registered_direction_held']}"
        )
    finally:
        await quiesce_organism(runtime)

    evidence["seconds"] = round(time.monotonic() - started, 1)
    out = run_dir / "content_geometry.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0


def _prediction(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """Did the geometry move the way it was said it would, before it moved?

    Attending to the alarm class should shorten the distances that involve it
    relative to the one that does not. The pair that involves neither is the
    control: if it moved as much, what moved was the measurement.
    """
    involving = [k for k in before if "alarm" in k]
    control = [k for k in before if "alarm" not in k]
    if not involving or not control:
        return {"note": "the class list does not contain the registered pair"}
    shift = {k: round(after.get(k, before[k]) - before[k], 6) for k in before}
    mean_involving = float(np.mean([shift[k] for k in involving]))
    mean_control = float(np.mean([shift[k] for k in control]))
    return {
        "registered": PREREGISTERED,
        "shift": shift,
        "mean_shift_involving_alarm": round(mean_involving, 6),
        "mean_shift_of_the_control_pair": round(mean_control, 6),
        "the_registered_direction_held": bool(mean_involving < mean_control),
        "note": (
            "The control pair involves neither attended class. If it moved as "
            "much, what moved was the measurement rather than the geometry."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
