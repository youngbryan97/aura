#!/usr/bin/env python3
"""A campaign's verdict under ISC-v5: its own lines, with irreducibility read by intervention.

The campaign report supplies every line v3 reads. The v5 sweep, a run of
tools/run_subject_core_v25.py with `--v5` on the same commit and seed, supplies
the two lines v5 replaces. A null that passes every other v3 line on this
campaign needs a v5 sweep of its own, passed as NAME=REPORT; without one the
null line fails and says which is missing.

    python tools/score_isc_v5.py --campaign RUN_DIR --sweep V25_RUN_DIR

It refuses a sweep that was not run to the preregistered design rather than
scoring it, and writes subject_core_v5_verdict.json beside the campaign report.
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


def _report(path: Path, name: str) -> dict[str, Any]:
    target = path / name if path.is_dir() else path
    return json.loads(target.read_text(encoding="utf-8"))


def _deciding_sweep(v25: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    from core.subject.isc_v5 import DECIDING_LAG

    cuts = v25.get("cuts") or {}
    sweep = cuts.get(f"lag_{DECIDING_LAG}")
    if sweep is None:
        return None, f"the sweep scored no horizon of {DECIDING_LAG} frames"
    return sweep, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True, help="a campaign run directory or its report")
    parser.add_argument("--sweep", type=Path, required=True, help="a v25 run directory or its report, run with --v5")
    parser.add_argument("--null-sweep", action="append", default=[], help="NAME=REPORT for a null that passes the rest")
    args = parser.parse_args(argv)

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.battery import assemble
    from core.subject.isc_v5 import nulls_passing_the_rest

    campaign = _report(args.campaign, "subject_core_report.json")
    v25 = _report(args.sweep, "subject_core_v25_report.json")
    sweep, missing = _deciding_sweep(v25)
    if sweep is None:
        raise SystemExit(f"refusing: {missing}")
    if campaign.get("campaign", {}).get("frozen", {}).get("seed") not in (None, v25.get("campaign", {}).get("frozen", {}).get("seed")):
        raise SystemExit("refusing: the sweep and the campaign ran on different seeds")
    null_sweeps: dict[str, Any] = {}
    for item in args.null_sweep:
        # A null architecture's sweep is a line of tools/validate_interventional_cut.py,
        # run with the v5 looks, level and draws, saved as JSON.
        name, _, where = item.partition("=")
        null_sweeps[name] = json.loads(Path(where).read_text(encoding="utf-8"))

    evidence = {key: value for key, value in campaign.items() if key != "verdict"}
    if "synergy_v4" not in evidence:
        # A campaign recorded before ISC-v5 read synergy with the clocks out.
        # The line is computed from the campaign's own recording, as the
        # campaign would have: turn rows, the target's change, the run's seed.
        from core.subject.recording import load_recording
        from core.subject.synergy import synergy_suite

        run_dir = args.campaign if args.campaign.is_dir() else args.campaign.parent
        seed = int((campaign.get("campaign", {}).get("frozen", {}) or {}).get("seed", 0) or 0)
        turns = load_recording(run_dir).by_turn()
        evidence["synergy_v4"] = [
            item.as_dict() for item in synergy_suite(turns, seed=seed, of="change", clocks_out=True)
        ]
    evidence["interventional_cut"] = {
        "sweep": sweep,
        "null_sweeps": null_sweeps,
        "nulls_that_pass": nulls_passing_the_rest((campaign.get("nulls") or {}).get("detail") or {}),
        "sweep_report": str(args.sweep),
        "sweep_commit": v25.get("campaign", {}).get("commit"),
        "campaign_commit": campaign.get("campaign", {}).get("commit"),
    }
    verdict = assemble(evidence).as_dict()
    out = (args.campaign if args.campaign.is_dir() else args.campaign.parent) / "subject_core_v5_verdict.json"
    atomic_write_text(
        out, json.dumps({"verdict": verdict, "interventional_cut": evidence["interventional_cut"]}, indent=1, default=str)
    )
    for line in verdict["v5_criteria"]:
        print(f"{'PASS' if line['passed'] else 'FAIL'}  {line['criterion']:<28} {json.dumps(line['value'], default=str)}")
    print(f"ISC-v5 on this seed: {verdict['isc_v5_on_this_seed']}   {verdict['v5_passed']}/24 criteria")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
