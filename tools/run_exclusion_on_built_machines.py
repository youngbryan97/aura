#!/usr/bin/env python3
"""J*'s exclusion rule on four machines built with a known number of subjects (core/subject/built_candidates.py).

Preregistered in docs/ISC_V6_PREREGISTRATION.md (addendum of 5 November 2026, the exclusion rule on built machines):
every machine on every seed must come out as written in `EXPECTED`, or the run says which did not and why.

    python tools/run_exclusion_on_built_machines.py --out ~/subject-core-runs/exclusion-built --seeds 0,1,2,3,4

Exit 0 when every machine on every seed came out as predicted, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject import built_candidates as bc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seeds; each builds every machine anew")
    parser.add_argument("--machines", default=",".join(bc.MACHINES))
    parser.add_argument("--anchors", type=int, default=bc.ANCHORS, help="a smaller number is a smoke run, not a reading")
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    unknown = [m for m in machines if m not in bc.MACHINES]
    if unknown:
        parser.error(f"no machine named {unknown}; there are {list(bc.MACHINES)}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for seed in seeds:
        for name in machines:
            began = time.monotonic()
            row = bc.measure(name, seed=seed, trials=args.trials, anchors=args.anchors)
            row["seconds"] = round(time.monotonic() - began, 1)
            rows.append(row)
            print(f"seed {seed} {name}: expected {row['expected']}, found {row['found']}, "
                  f"{'as predicted' if row['as_predicted'] else 'NOT as predicted'} ({row['seconds']} s)", flush=True)
    report = {
        "settings": {"seeds": seeds, "machines": machines, "anchors": args.anchors, "trials": args.trials,
                     "width": bc.WIDTH, "noise": bc.NOISE, "radius": bc.RADIUS, "horizons": list(bc.HORIZONS)},
        "authoritative": args.anchors == bc.ANCHORS and machines == list(bc.MACHINES),
        "as_predicted": sum(1 for r in rows if r["as_predicted"]),
        "of": len(rows),
        "all_as_predicted": all(r["as_predicted"] for r in rows),
        "seconds": round(time.monotonic() - started, 1),
        "rows": rows,
    }
    (args.out / "exclusion_on_built_machines.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['as_predicted']} of {report['of']} as predicted; wrote {args.out / 'exclusion_on_built_machines.json'}")
    return 0 if report["all_as_predicted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
