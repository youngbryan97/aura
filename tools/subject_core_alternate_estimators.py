#!/usr/bin/env python3
"""Whether a run's irreducibility and synergy verdicts survive other estimators.

    python tools/subject_core_alternate_estimators.py artifacts/subject_core/run_033 [...]

For each run, on the recording sampled once per turn as the battery samples it:

- irreducibility at every pairing of 3, 4 and 6 components with 5 and 8 folds,
  each read against the battery's bar on its own lower bound;
- synergy for the four declared triples on the target's change, with the
  Kraskov nearest-neighbour estimator and its own shifted null.

The report says, per criterion, what the battery decided and whether every
alternative decided the same. A verdict that flips under one of these is
reported as depending on the estimator, whichever way the battery went.
See core/subject/alternates.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: The settings irreducibility is re-scored at. The battery's own, four and
#: five, is one of them, so the table shows the primary reading beside the rest.
COMPONENT_COUNTS: tuple[int, ...] = (3, 4, 6)
FOLD_COUNTS: tuple[int, ...] = (5, 8)


def alternates_for(run: Path, *, draws: int, seed: int) -> dict[str, Any]:
    from core.subject.alternates import irreducibility_under, ksg_synergy
    from core.subject.battery import THRESHOLDS
    from core.subject.recording import load_recording
    from core.subject.synergy import TRIPLES

    report = json.loads((run / "subject_core_report.json").read_text()) if (run / "subject_core_report.json").exists() else {}
    turns = load_recording(run).by_turn()
    bar = float(THRESHOLDS["phi_do"])

    irreducibility = [
        irreducibility_under(turns, components=components, folds=folds).as_dict()
        for components in COMPONENT_COUNTS
        for folds in FOLD_COUNTS
    ]
    for row in irreducibility:
        row["passes"] = row["lower_bound"] > bar
    battery_phi = (report.get("phi") or {}).get("lower_bound")
    battery_passes = None if battery_phi is None else float(battery_phi) > bar

    synergy = [ksg_synergy(turns, a, b, y, of="change", draws=draws, seed=seed).as_dict() for a, b, y in TRIPLES]
    battery_synergy = [bool(row.get("passes")) for row in report.get("synergy_v2") or []]
    underpowered = any(row["underpowered"] for row in synergy)

    def verdict(primary: bool | None, alternates: list[bool]) -> dict[str, Any]:
        agree = primary is not None and all(value == primary for value in alternates)
        return {
            "battery": primary,
            "alternates": alternates,
            "survives": bool(agree),
            "reading": (
                "no battery verdict to compare" if primary is None
                else "every alternative agrees" if agree
                else "depends on the estimator"
            ),
        }

    return {
        "run": str(run),
        "rows": int(turns.frames),
        "irreducibility": {"bar": bar, "settings": irreducibility, **verdict(battery_passes, [r["passes"] for r in irreducibility])},
        "synergy_change": {
            "estimator": "Kraskov-Stoegbauer-Grassberger, k=4, shifted null",
            "triples": synergy,
            "underpowered": underpowered,
            **(
                {"battery": all(battery_synergy) if battery_synergy else None, "alternates": [], "survives": False,
                 "reading": "too few rows for the nearest-neighbour estimate to decide"}
                if underpowered
                else verdict(
                    all(battery_synergy) if battery_synergy else None,
                    [all(row["clears_its_null"] for row in synergy)],
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--draws", type=int, default=200, help="shifted-null draws for the nearest-neighbour synergy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    results = [alternates_for(run, draws=args.draws, seed=args.seed) for run in args.runs]
    for result in results:
        print(f"{result['run']} ({result['rows']} turns)")
        irr = result["irreducibility"]
        print(f"  irreducibility: battery {irr['battery']}, {irr['reading']}")
        for row in irr["settings"]:
            print(f"    components {row['components']} folds {row['folds']}: lower bound {row['lower_bound']:+.4f} passes {row['passes']}")
        syn = result["synergy_change"]
        print(f"  synergy on change: battery {syn['battery']}, {syn['reading']}")
        for row in syn["triples"]:
            print(
                f"    {'+'.join(row['sources'])}->{row['target']}: fraction {row['synergy_fraction']:.3f} "
                f"null q99 {row['null_q99_fraction']:.3f} raw {row['synergy']:.4f} raw q99 {row['raw_null_q99']:.4f} "
                f"clears {row['clears_its_null']}"
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
