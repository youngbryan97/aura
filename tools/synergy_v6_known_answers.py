#!/usr/bin/env python3
"""ISC-v6's synergy bars on simulated forks whose answer is known.

docs/ISC_V6_PREREGISTRATION.md states what each world must come out as before
the line is read on her. Each dataset is 48 snapshots, as a campaign gives:
four untouched forks and the three displaced ones of one triple, in eight
whitened target columns.

    python tools/synergy_v6_known_answers.py --datasets 1000 --out artifacts/synergy_v6_known_answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WIDTH, SNAPSHOTS, CONDITIONS = 8, 48, 8
#: What each world must do, as the preregistration states it: the most or the
#: fewest of its datasets that may pass.
STATED = {
    "noise": ("at most", 0.03),
    "additive": ("at most", 0.03),
    "one_source_threshold": ("at most", 0.03),
    "shared_fork_noise": ("at most", 0.03),
    "deterministic_divergence": ("at most", 0.03),
    "product": ("at least", 0.95),
    "deterministic_product": ("at least", 0.95),
    "product_turning_with_the_situation": ("reported", None),
}


def world_samples(kind: str, rng: np.random.Generator):
    """One dataset of a world: the forks of 48 snapshots."""
    from core.subject.interventional_synergy import Samples

    out = Samples()
    first_effect, second_effect = rng.normal(size=WIDTH), rng.normal(size=WIDTH)
    direction = rng.normal(size=WIDTH)
    direction /= np.linalg.norm(direction)
    turned = [rng.normal(size=WIDTH) for _ in range(CONDITIONS)]
    turned = [v / np.linalg.norm(v) for v in turned]
    for index in range(SNAPSHOTS):
        base = rng.normal(size=WIDTH)
        deterministic = kind in ("deterministic_divergence", "deterministic_product")
        shared = rng.normal(size=WIDTH) if kind == "shared_fork_noise" else np.zeros(WIDTH)

        def noise() -> np.ndarray:
            return np.zeros(WIDTH) if deterministic else rng.normal(size=WIDTH) + shared

        untouched = [base + noise() for _ in range(4)]
        first, second, both = base + noise(), base + noise(), base + noise()
        if kind in ("additive", "shared_fork_noise"):
            first, second, both = first + first_effect, second + second_effect, both + first_effect + second_effect
        elif kind == "one_source_threshold":
            step = 2.0 * first_effect / (1.0 + np.exp(-4.0 * float(base[0])))
            first, both = first + step, both + step
        elif kind == "deterministic_divergence":
            # Every displacement sends the run somewhere else; how far and which
            # way has nothing to do with the two sources meeting.
            first, second, both = (base + 5.0 * rng.normal(size=WIDTH) for _ in range(3))
        elif kind == "product":
            # Size 5 against unit noise in each of eight columns: a little under
            # the size of the four forks' noise in the contrast (2 sqrt 8).
            first, second = first + first_effect, second + second_effect
            both = both + first_effect + second_effect + 5.0 * direction
        elif kind == "deterministic_product":
            first, second = first + first_effect, second + second_effect
            both = both + first_effect + second_effect + 1.0 * direction
        elif kind == "product_turning_with_the_situation":
            first, second = first + first_effect, second + second_effect
            both = both + first_effect + second_effect + 5.0 * turned[index % CONDITIONS]
        out.add(untouched=untouched, first=first, second=second, both=both)
    return out


def run(datasets: int, draws: int, seed: int = 20261023) -> dict:
    from core.subject.interventional_synergy import summarise

    rng = np.random.default_rng(seed)
    result = {}
    for kind, (side, bound) in STATED.items():
        passed = sum(
            summarise(world_samples(kind, rng), seed=seed + index, draws=draws)["passes"]
            for index in range(datasets)
        )
        rate = passed / datasets
        result[kind] = {
            "passed": passed, "datasets": datasets, "rate": round(rate, 4),
            "stated": side if bound is None else f"{side} {bound}",
            "as_stated": True if bound is None else (rate <= bound if side == "at most" else rate >= bound),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(args.datasets, args.draws)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if all(row["as_stated"] for row in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
