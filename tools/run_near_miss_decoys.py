"""Read every integration line on the near misses and the agent, through the same rows the campaign builds for nulls.

Each near miss is the reference with one property taken out (core/subject/near_miss.py). The prediction, written in
docs/ISC_V6_PREREGISTRATION.md before this ran, is that each fails on exactly the line it is named for. The agent is
read the same way, and also with no input, where nothing in it may move. The reference is read beside them, because
a run in which the reference fails a line has measured the instrument, not the near misses.

    python tools/run_near_miss_decoys.py --out ~/subject-core-runs/near-miss-s7 --seed 7 --null-draws 8 --trials 6

Rows are computed by `core.subject.null_table.null_rows`, the function the campaign's null stage calls, and judged
on the thresholds the null conjunction judges by.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject import near_miss  # noqa: E402
from core.subject.battery import THRESHOLDS  # noqa: E402
from core.subject.null_table import null_rows  # noqa: E402
from core.subject.nulls import architecture  # noqa: E402
from core.subject.state import DOMAINS  # noqa: E402

#: The integration lines a null is judged on, by the name the prediction uses for each.
LINES = {
    "strong connectivity": lambda r: bool(r.get("one_component")),
    "vertex connectivity": lambda r: float(r.get("vertex_connectivity", 0)) >= THRESHOLDS["vertex_connectivity"],
    "re-entry": lambda r: bool(r.get("reentry")),
    "closure": lambda r: bool(r.get("closed", True)),
    "differentiation": lambda r: float(r.get("largest_component_share", 1)) < THRESHOLDS["component_share"]
    and float(r.get("d_eff_normalised", 1)) < THRESHOLDS["d_eff_normalised"],
    "synergy": lambda r: all(bool(x) for x in (r.get("synergy_v3_passes") or [True])),
    "irreducibility": lambda r: float(r.get("phi_do", 0)) > THRESHOLDS["phi_do"],
}

#: What each near miss is predicted to fail, as a set of the names above.
PREDICTED = {
    "near_miss_no_synergy": {"synergy"},
    "near_miss_driven": {"closure"},
    "near_miss_one_cut_vertex": {"vertex connectivity"},
    "near_miss_one_way": {"strong connectivity", "vertex connectivity", "re-entry"},
    "near_miss_one_shock": {"differentiation"},
    "near_miss_weakly_coupled": {"irreducibility"},
}


def _idle_motion(seed: int, steps: int = 500) -> float:
    agent = architecture(near_miss.AGENT, seed=seed)
    agent.input_rate = 0.0
    rng = np.random.default_rng(seed)
    start = agent.start(rng)
    state = {key: value.copy() for key, value in start.items()}
    for _ in range(steps):
        state = agent.step(state, rng)
    return float(max(np.abs(state[key] - start[key]).max() for key in DOMAINS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--null-draws", type=int, default=8)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    names = ["recurrent", *near_miss.NEAR_MISSES, near_miss.AGENT]
    rows = {}
    for name, row, notes in null_rows(
        names, seed=args.seed, null_draws=args.null_draws, trials=args.trials, domains=list(DOMAINS),
        workers=args.workers,
    ):
        for note in notes:
            print(note, flush=True)
        failed = sorted(line for line, holds in LINES.items() if not holds(row))
        rows[name] = {"row": row, "fails": failed}
        print(f"{name}: fails {failed or 'nothing'}", flush=True)

    verdicts = {}
    for name, lines in PREDICTED.items():
        failed = set(rows[name]["fails"])
        verdicts[name] = {
            "predicted": sorted(lines),
            "failed": sorted(failed),
            "as_predicted": failed == lines,
            "caught": bool(failed),
            "missed_its_own_line": sorted(lines - failed),
            "also_failed": sorted(failed - lines),
        }
    report = {
        "seed": args.seed,
        "null_draws": args.null_draws,
        "trials": args.trials,
        "reference_fails": rows["recurrent"]["fails"],
        "instrument_ok": not rows["recurrent"]["fails"],
        "near_misses": verdicts,
        "agent": {"fails": rows[near_miss.AGENT]["fails"], "idle_motion": _idle_motion(args.seed)},
        "rows": {name: entry["row"] for name, entry in rows.items()},
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "near_miss_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
