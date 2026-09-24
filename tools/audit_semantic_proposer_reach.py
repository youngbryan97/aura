#!/usr/bin/env python3
"""Profile signed construction-held proposal banks without changing their receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def audit_directory(directory: Path) -> dict:
    from tools.probe_semantic_proposer_crossfit import _digest, proposal_reach_profile

    report = json.loads((directory / "report.json").read_bytes())
    plan = json.loads((directory / "plan.json").read_bytes())
    if (report.get("schema") != "aura.semantic_proposer_crossfit.v1"
            or report.get("receipt_sha256") != _digest({
                key: value for key, value in report.items() if key != "receipt_sha256"})
            or plan.get("schema") != "aura.semantic_proposer_crossfit_plan.v1"
            or plan.get("plan_sha256") != _digest({
                key: value for key, value in plan.items() if key != "plan_sha256"})
            or report.get("plan_sha256") != plan["plan_sha256"]
            or sorted(report["row_receipts"]) != sorted(plan["held_ids"])
            or report["held_population"] != len(plan["held_ids"])):
        raise ValueError("construction-held report and plan differ")
    rows = []
    for source in plan["held_ids"]:
        row = json.loads((directory / "rows" / f"{source}.json").read_bytes())
        body = {key: value for key, value in row.items() if key != "receipt_sha256"}
        if (row.get("receipt_sha256") != _digest(body)
                or report["row_receipts"][source] != row["receipt_sha256"]
                or row.get("source") != source
                or row.get("plan_sha256") != plan["plan_sha256"]):
            raise ValueError("construction-held row differs from its signed report")
        rows.append(row)
    profile = proposal_reach_profile(rows)
    if profile["observed_reachable"] != report["correct_reachable"]:
        raise ValueError("profile differs from archived candidate reach")
    return {"fold": plan["fold"], "report_receipt_sha256": report["receipt_sha256"],
            "profile": profile, "serving_authority": False,
            "qualification_evidence": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit_directory(path) for path in args.directory], sort_keys=True))


if __name__ == "__main__":
    main()
