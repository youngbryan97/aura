#!/usr/bin/env python3
"""Compare a replication of seed 101's substrate campaign with the reference runs, line by line.

    python3 tools/replicate/compare_with_reference.py <run directory, or its subject_core_report.json> [--out DIR]

Writes replication_report.md and replication_report.json (beside the run, or in --out): each of the 22 lines the
campaign decides, its verdict here and on each reference run, whether the tape was the reference tape, and the machine
and package versions. Standard library only, so it runs with any Python 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "seed101.json"
VERSIONS = ("criteria", "v2_criteria", "v3_criteria", "v5_criteria", "v6_criteria", "v7_criteria")

#: Decided before any replication ran: the line that differed between the reference runs, and why. A flip here
#: is the outcome the README says to expect; a flip elsewhere is news.
BORDERLINE = {
    "synergy": "its weakest triple (perception and memory onto the world model) passed its one-direction test at "
    "p = 0.0052 on the first machine against a bar of 0.01, read p = 0.60 on the MacBook's first run and "
    "p = 0.0029 on its second",
}


#: Lines whose reading here is not the one the 24 of 24 used.
NOTES = {
    "causal_closure_of_the_core": "the campaign's observational reading, which fails on all three reference runs; the "
    "24 of 24 decides closure by intervention, in the carrier sweep, which this kit does not run",
}


def _when(stamp: object) -> str:
    import datetime

    if isinstance(stamp, (int, float)):
        return datetime.datetime.fromtimestamp(float(stamp)).strftime("%Y-%m-%d %H:%M")
    return str(stamp)


def campaign_lines(report: dict) -> dict[str, dict]:
    """Each line as its newest version decided it, without the two the carrier sweep decides (ISC-v5)."""
    body = report.get("verdict", report)
    out: dict[str, dict] = {}
    for key in VERSIONS:
        for row in body.get(key) or []:
            name = row.get("criterion")
            if name:
                out[name] = {"passed": bool(row.get("passed")), "version": key, "value": str(row.get("value"))[:160]}
    return {name: line for name, line in out.items() if line["version"] != "v5_criteria"}


def compare(report: dict, reference: dict) -> dict:
    mine = campaign_lines(report)
    runs = reference["runs"]
    names = list(runs["first_machine"]["lines"])
    rows = []
    for name in names:
        row = {"line": name, "here": mine.get(name, {}).get("passed")}
        for run, data in runs.items():
            row[run] = data["lines"][name]["passed"]
        row["same_as_first_machine"] = row["here"] == row["first_machine"]
        row["borderline"] = name in BORDERLINE
        rows.append(row)
    tape = (report.get("conversation") or {}).get("digest")
    campaign = report.get("campaign") or {}
    return {
        "commit": campaign.get("commit"),
        "seed": reference["seed"],
        "tape_is_the_reference_tape": tape == reference["tape_digest"],
        "tape_digest": tape,
        "lines": len(names),
        "same_as_first_machine": sum(1 for r in rows if r["same_as_first_machine"]),
        "different": [r["line"] for r in rows if not r["same_as_first_machine"]],
        "missing": [r["line"] for r in rows if r["here"] is None],
        "rows": rows,
        "environment": report.get("environment") or campaign.get("environment") or {},
        "started_at": campaign.get("started_at"),
        "finished_at": campaign.get("finished_at"),
    }


def markdown(result: dict) -> str:
    def mark(value: object) -> str:
        return {True: "PASS", False: "FAIL", None: "missing"}[value]  # type: ignore[index]

    env = result["environment"]
    lines = [
        "# Replication of seed 101's substrate campaign",
        "",
        f"- Commit: {result['commit']}; seed {result['seed']}.",
        f"- The tape was the reference tape: {'yes' if result['tape_is_the_reference_tape'] else 'NO'} "
        f"(digest {str(result['tape_digest'])[:12]}).",
        f"- Same verdict as the first machine on {result['same_as_first_machine']} of {result['lines']} lines.",
        f"- Machine: {json.dumps(env.get('hardware', {}))}; {env.get('os', '?')}; Python {env.get('python', '?')}.",
        f"- Packages: {json.dumps(env.get('packages', {}))}.",
        f"- Started {_when(result['started_at'])}, finished {_when(result['finished_at'])}.",
        "",
        "| Line | Here | First machine | MacBook run 1 | MacBook run 2 | |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        note = "differs" if not row["same_as_first_machine"] else ""
        if row["borderline"]:
            note = (note + "; " if note else "") + "known to be borderline"
        if row["line"] in NOTES:
            note = (note + "; " if note else "") + "see the note below"
        lines.append(f"| {row['line']} | {mark(row['here'])} | {mark(row['first_machine'])} | "
                     f"{mark(row.get('macbook_run_1'))} | {mark(row.get('macbook_run_2'))} | {note} |")
    lines += ["", "Known borderline before this run:"] + [f"- {name}: {why}." for name, why in BORDERLINE.items()]
    lines += ["", "Notes:"] + [f"- {name}: {why}." for name, why in NOTES.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="a campaign run directory, or its subject_core_report.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    path = args.run / "subject_core_report.json" if args.run.is_dir() else args.run
    if not path.exists():
        print(f"no report at {path}", file=sys.stderr)
        return 2
    result = compare(json.loads(path.read_text()), json.loads(REFERENCE.read_text()))
    out = args.out or path.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication_report.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "replication_report.md").write_text(markdown(result))
    print(markdown(result))
    print(f"wrote {out / 'replication_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
