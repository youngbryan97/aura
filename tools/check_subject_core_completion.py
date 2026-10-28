#!/usr/bin/env python3
"""Read explicit evidence files and require ISC-v6, J*, and parity together.

This is an offline reader. It starts no runtime and loads no model. Exit 0
means all three goals passed on compatible evidence; exit 1 leaves work open.
Historical reports without source pins remain readable but cannot complete it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("campaign", "carrier", "content", "interventional", "reports", "state-log", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--null-sweep", action="append", default=[], help="NAME=JSON")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("the output already exists; choose a new path to preserve the evidence record")

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.battery import assemble
    from core.subject.bridge import solve
    from core.subject.completion import compatibility, completion
    from core.subject.isc_v5 import nulls_passing_the_rest
    from core.subject.lineage import HER_RECORD_BEGINS_AT, lineage_from_state_log
    from tools.score_isc_v5 import _deciding_sweep
    from tools.solve_for_j import state_rows

    sources = {}

    def read(name: str, path: Path) -> dict:
        raw = path.read_bytes()
        sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a JSON object")
        return value

    reports = {name: read(name, getattr(args, name)) for name in (
        "campaign", "carrier", "content", "interventional", "reports",
    )}
    blockers = compatibility(reports)
    campaign, carrier = reports["campaign"], reports["carrier"]
    campaign_authority = campaign.get("campaign") or {}
    if campaign_authority.get("authoritative") is not True or campaign_authority.get("authority_blockers"):
        blockers.append("the campaign has no clean authority")
    scope = str(carrier.get("scope") or "")
    if not scope or reports["content"].get("scope") != scope:
        blockers.append("carrier and content did not record the same scope")
    sweep, missing = _deciding_sweep(carrier)
    null_sweeps = {}
    for item in args.null_sweep:
        name, separator, path = item.partition("=")
        if not separator or not name or name in null_sweeps:
            raise ValueError("each null sweep needs a distinct NAME=JSON")
        null_sweeps[name] = read(f"null:{name}", Path(path))
    evidence = {key: value for key, value in campaign.items() if key != "verdict"}
    evidence["interventional_closure"] = reports["interventional"]
    if sweep is None:
        blockers.append(missing)
    else:
        evidence["interventional_cut"] = {
            "sweep": sweep, "null_sweeps": null_sweeps,
            "nulls_that_pass": nulls_passing_the_rest((campaign.get("nulls") or {}).get("detail") or {}),
        }
    verdict = assemble(evidence)
    rows = state_rows(args.state_log)
    sources["state_log"] = {
        "path": str(args.state_log.resolve()),
        "rows": len(rows),
        "person_rows_sha256": hashlib.sha256(json.dumps(
            sorted(rows, key=lambda row: str(row["state_id"])),
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest(),
    }
    lineage = lineage_from_state_log(rows, record_begins_at=HER_RECORD_BEGINS_AT)
    j = solve(
        carrier, reports["content"], lineage,
        battery={line.key: line.passed for line in verdict.v6_lines()},
        reports=reports["reports"], interventional=reports["interventional"],
    )
    result = completion(verdict, j, blockers=blockers, scope=scope)
    output = {
        "completion": result, "verdict": verdict.as_dict(), "j_star": j.as_dict(),
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps(output, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
