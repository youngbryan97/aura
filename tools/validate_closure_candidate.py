#!/usr/bin/env python3
"""Offline calibration of a prospective closure predictor on known worlds.

Known effects come from the data-generating equation, never from the fitted
gain. This measures detection frequency and its exact binomial interval; it
does not certify closure on Aura or change any official run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=30)
    parser.add_argument("--rows", type=int, default=2960)
    parser.add_argument("--outside-width", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.worlds < 1 or args.rows < 200:
        parser.error("use a new output path, positive world count, and at least 200 rows")

    import numpy as np
    from scipy.stats import binomtest

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.closure import CLOSURE_MARGIN, _pooled_bounds
    from core.subject.closure_candidate import compare, gaussian_world
    from core.subject.estimate import split_rows

    records = []
    for effect in (0., CLOSURE_MARGIN / 2, CLOSURE_MARGIN, 2*CLOSURE_MARGIN, .1):
        worlds = []
        for seed in range(args.seed_start, args.seed_start + args.worlds):
            own, outside, target = gaussian_world(
                seed=seed, rows=args.rows, effect=effect, outside_width=args.outside_width,
            )
            train, validate, test = split_rows(args.rows)
            comparison = compare(own, outside, target, train=train, validate=validate, test=test, seed=1)
            base = np.asarray(comparison.narrow.row_sse)
            gain = base - np.asarray(comparison.wide.row_sse)
            lower, upper = _pooled_bounds([gain], [base], seed=seed, draws=500)
            worlds.append({"seed": seed, "gain": comparison.gain, "lower": lower, "upper": upper,
                           "detected": lower > 0.})
        hits = sum(item["detected"] for item in worlds)
        interval = binomtest(hits, args.worlds).proportion_ci(confidence_level=.95, method="exact")
        record = {
            "oracle_effect": effect, "detected": hits, "worlds": args.worlds,
            "frequency": hits / args.worlds, "interval_95": [interval.low, interval.high],
            "mean_gain": float(np.mean([item["gain"] for item in worlds])),
            "readings": worlds,
        }
        records.append(record)
        print(f"oracle {effect:.6f}: {hits}/{args.worlds}, interval [{interval.low:.3f}, {interval.high:.3f}]", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps({
        "candidate_only": True, "rows": args.rows, "outside_width": args.outside_width,
        "known_world_equation": "Y=K^2+sqrt(rho)*P0+sqrt(1-rho)*epsilon; independent standard Gaussian K,P,epsilon",
        "decision": "one-sided 95% block-bootstrap lower bound on held-out predictive gain above zero",
        "limits": "This calibration covers these Gaussian worlds and this feature width, not Aura's full distribution or the official closure rule.",
        "records": records,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
