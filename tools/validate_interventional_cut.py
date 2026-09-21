#!/usr/bin/env python3
"""The interventional cut, run on systems whose answer is known.

The battery's irreducibility is a regression: it hides one side's columns from
a model and reads what prediction loses. Held-out predictive loss scores an
input-driven pipeline integrated by construction at 0.003 to 0.016 at every
coupling strength tried, the same as one with no cross-domain coupling, because
each turn's fresh input fills the loss the cut is divided by. v25's cut does not
hide columns. From one anchor state it runs two untouched arms (the second is
the sham), the left side evolving with the right held at its anchor values, the
right with the left held, and composes the two free halves, the way
core/subject/v25_runtime.py cuts the organism. Every arm from one anchor draws
the same noise and the same inputs, which is what restoring a snapshot does, so
the input is common to both sides of every comparison and cancels. The decision
is core.subject.v25_cut.decide_cut, unchanged.

    python tools/validate_interventional_cut.py --anchors 12 --lag 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject.nulls import architecture  # noqa: E402
from core.subject.state import DOMAINS  # noqa: E402
from core.subject.v25_cut import decide_cut  # noqa: E402
from core.subject.v25_runtime import bipartitions  # noqa: E402

WIDTH = 4

Step = Callable[[dict, np.ndarray, np.ndarray], dict]


def toy_step(system) -> tuple[Step, Callable]:
    def step(state, noise, _input):
        rng = _Fixed(noise)
        return system.step(state, rng)
    return step, system.start


class _Fixed:
    """A generator that hands out pre-drawn normals, so arms share their noise."""

    def __init__(self, draws: np.ndarray) -> None:
        self._draws = list(draws)

    def normal(self, size=None, scale=1.0, loc=0.0):
        n = int(np.prod(size)) if size is not None else 1
        out = np.array([self._draws.pop(0) for _ in range(n)])
        return loc + scale * (out.reshape(size) if size is not None else out[0])


def pipeline_step(kind: str, coupling: float, seed: int) -> tuple[Step, Callable]:
    rng = np.random.default_rng(seed)
    order = list(DOMAINS)
    nb = {k: [order[(i + s) % 10] for s in (1, 2, 3, -1, -2, -3)] for i, k in enumerate(order)}
    inputs_to = {k: rng.normal(scale=1 / np.sqrt(WIDTH), size=(WIDTH, WIDTH)) for k in order}
    couplings = {k: rng.normal(scale=coupling / np.sqrt(6 * WIDTH), size=(WIDTH, 6 * WIDTH)) for k in order}

    def start(r):
        return {k: r.normal(scale=0.3, size=WIDTH) for k in order}

    def step(state, noise, e):
        x = {k: v.copy() for k, v in state.items()}
        draws = iter(noise.reshape(10, WIDTH))
        for k in order:
            n = next(draws)
            if k == "P":
                x[k] = 0.8 * x[k] + e + 0.05 * n
                continue
            around = np.concatenate([x[j] for j in nb[k]])
            if kind == "additive":
                new = 0.8 * x[k] + np.tanh(couplings[k] @ around) + inputs_to[k] @ x["P"]
            elif kind == "modulated":
                new = 0.8 * x[k] + (1.0 + np.tanh(couplings[k] @ around)) * (inputs_to[k] @ x["P"])
            else:
                new = 0.8 * x[k] + inputs_to[k] @ e
            x[k] = np.tanh(new) + 0.05 * n
        return x

    return step, start


def run(step: Step, start_state: dict, noises, inputs, held: tuple[str, ...], lag: int) -> dict:
    state = {k: v.copy() for k, v in start_state.items()}
    anchor = {k: v.copy() for k, v in start_state.items()}
    for index in range(lag):
        state = step(state, noises[index], inputs[index])
        for key in held:
            state[key] = anchor[key].copy()
    return state


def vec(state: dict) -> np.ndarray:
    return np.concatenate([state[k] for k in DOMAINS])


def sweep(step: Step, start: Callable, noise_width: int, *, anchors: int, lag: int, cuts=None, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    state = start(rng)
    held_states = []
    for t in range(400 + anchors * 25):
        state = step(state, rng.normal(size=noise_width), rng.normal(size=WIDTH))
        if t >= 400 and (t - 400) % 25 == 0 and len(held_states) < anchors:
            held_states.append({k: v.copy() for k, v in state.items()})
    arms = []
    for anchor in held_states:
        noises = [rng.normal(size=noise_width) for _ in range(lag)]
        inputs = [rng.normal(size=WIDTH) for _ in range(lag)]
        arms.append((anchor, noises, inputs))
    decided, weakest = 0, None
    chosen = list(bipartitions(DOMAINS)) if cuts is None else list(cuts)
    for left, right in chosen:
        samples = {k: [] for k in ("context", "intact", "cut", "sham_a", "sham_b")}
        for anchor, noises, inputs in arms:
            intact = run(step, anchor, noises, inputs, (), lag)
            left_free = run(step, anchor, noises, inputs, tuple(right), lag)
            right_free = run(step, anchor, noises, inputs, tuple(left), lag)
            cut = {k: (left_free[k] if k in left else right_free[k]) for k in DOMAINS}
            samples["context"].append(vec(anchor))
            samples["intact"].append(vec(intact))
            samples["sham_a"].append(vec(intact))
            samples["sham_b"].append(vec(intact) + 1e-6 * np.random.default_rng(len(samples["sham_b"])).normal(size=len(vec(intact))))
            samples["cut"].append(vec(cut))
        arrays = {k: np.vstack(v) for k, v in samples.items()}
        estimate, lower, _p, _q = decide_cut(arrays, tau_seconds=float(lag), draws=60, permutation_draws=39)
        if lower > 0.0:
            decided += 1
        if weakest is None or estimate.excess_rate < weakest[0]:
            weakest = (estimate.excess_rate, lower, "".join(left) + "|" + "".join(right))
    return {"decided": decided, "cuts": len(chosen), "weakest_excess": weakest[0], "weakest_lower": weakest[1], "weakest_cut": weakest[2]}



ARCHITECTURES_CHECKED: tuple[str, ...] = ("recurrent", "independent", "star", "hub", "ring", "one_way", "common_driver")
PIPELINES_CHECKED: tuple[str, ...] = ("additive", "modulated", "independent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=int, default=12)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--systems", default="", help="comma-separated names; default is every control")
    args = parser.parse_args(argv)
    wanted = {name for name in args.systems.split(",") if name}
    out = {}
    for name in ARCHITECTURES_CHECKED:
        label = f"null:{name}"
        if wanted and label not in wanted:
            continue
        system = architecture(name, seed=7)
        step, start = toy_step(system)
        out[label] = sweep(step, start, sum(system.widths.values()) + system.hub_width, anchors=args.anchors, lag=args.lag)
        print(label, json.dumps(out[label]), flush=True)
    for kind in PIPELINES_CHECKED:
        label = f"pipeline:{kind}"
        if wanted and label not in wanted:
            continue
        step, start = pipeline_step(kind, 0.6, 1)
        out[label] = sweep(step, start, 10 * WIDTH, anchors=args.anchors, lag=args.lag)
        print(label, json.dumps(out[label]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
