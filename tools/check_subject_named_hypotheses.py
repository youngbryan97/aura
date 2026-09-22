#!/usr/bin/env python3
"""The ties and synergies Bryan named, tested on one campaign's side record.

Reads `named_readings.npz` from a campaign run directory, written beside the
recording by tools/run_subject_core.py, and runs core.subject.named_readings's
two tests: each named pair against its slid surrogate, and each named triple
against the synergy estimator's shifted null. Writes
`subject_named_hypotheses.json` beside it. See docs/SUBJECT_NAMED_HYPOTHESES.md.

    python tools/check_subject_named_hypotheses.py --run RUN_DIR
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.named_readings import check_named

    held = np.load(args.run / "named_readings.npz", allow_pickle=True)
    rows, names = held["rows"], tuple(str(name) for name in held["names"])
    manifest = json.loads((args.run / "core_state_manifest.json").read_text(encoding="utf-8"))
    rounds = int((manifest.get("notes") or {}).get("rounds") or 1)
    cycle = max(1, rows.shape[0] // max(1, rounds))
    report = check_named(rows, names, cycle=cycle, seed=args.seed)
    report["cycle_frames"] = cycle
    out = args.run / "subject_named_hypotheses.json"
    atomic_write_text(out, json.dumps(report, indent=1))
    for tie in report["ties"]:
        print(f"{'PASS' if tie['passes'] else 'FAIL'}  tie       {tie['said']}: {tie['observed']} vs {tie['null_bar']}")
    for item in report["together"]:
        print(f"{'PASS' if item['passes'] else 'FAIL'}  together  {item['said']}: {item['fraction']} vs {item['null_q99']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
