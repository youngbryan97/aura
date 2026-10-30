#!/usr/bin/env python3
"""Which rival accounts of a decisive reading its own files rule out (core/subject/alternatives.py).

    python tools/rule_out_alternatives.py --run ~/subject-core-runs/decisive-v9-<commit>-s101 \\
        [--confirmation passed|failed]

Reads the run's verdict, nulls, carrier, interventional closure and reports,
and beside them, when the run has them, the declared report controls
(reports-steering-off, reports-untuned) and the disconnected-Aura control.
Writes alternatives.json into the run directory and prints one line a rival.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _newest(root: Path, pattern: str) -> dict[str, Any] | None:
    found = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return json.loads(found[-1].read_text()) if found else None


def gather(run: Path) -> dict[str, Any]:
    """The evidence a decisive run directory holds, by the names rule_out reads."""
    return {
        "verdict": _newest(run, "campaign/run_*/subject_core_v5_verdict.json"),
        "nulls": _newest(run, "campaign/run_*/nulls.json") or {},
        "carrier": _newest(run, "carrier/launch_*/coordinator/run_*/subject_core_v25_report.json") or {},
        "interventional": _newest(run, "interventional/interventional_closure.json") or {},
        "reports": _newest(run, "reports/run_*/report_grounding.json") or {},
        "controls": {
            name: _newest(run, f"reports-{name}/run_*/report_grounding.json")
            for name in ("steering-off", "untuned")
            if _newest(run, f"reports-{name}/run_*/report_grounding.json")
        },
        "disconnected": _newest(run, "disconnected/run_*/disconnected_aura.json"),
    }


def main(argv: list[str] | None = None) -> int:
    from core.subject.alternatives import null_calibration, rule_out

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--disconnected", type=Path, default=None, help="a disconnected_aura.json read elsewhere")
    parser.add_argument(
        "--confirmation", choices=("passed", "failed"), default=None,
        help="this run is the confirmatory seed named in its addendum, and how it read",
    )
    args = parser.parse_args(argv)
    evidence = gather(args.run)
    if args.disconnected:
        evidence["disconnected"] = json.loads(args.disconnected.read_text())
    if args.confirmation:
        evidence["confirmation"] = {"read": True, "passed": args.confirmation == "passed", "why": str(args.run.name)}
    if evidence["verdict"] is None:
        raise SystemExit(f"refusing: no campaign verdict under {args.run}")
    out = {"run": str(args.run), "alternatives": rule_out(evidence), "null_calibration": null_calibration(evidence["nulls"])}
    (args.run / "alternatives.json").write_text(json.dumps(out, indent=2, default=str))
    for row in out["alternatives"]["rivals"]:
        print(f"{row['status']:<9} {row['rival']:<24} {row['why']}")
    print(out["alternatives"]["summary"])
    for row in out["null_calibration"]["grounds"]:
        print(f"ground {row['ground']}: passing nulls {row['passing']}, upper bound {row['pass_rate_upper_95']}")
    print(f"null conjunction passed by {out['null_calibration']['conjunction_passing'] or 'no null'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
