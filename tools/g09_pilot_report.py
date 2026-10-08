#!/usr/bin/env python3
"""What G09's pilot found, per domain: her drafts, her organ, and where they differ.

Both arms are regraded here with the harness's current graders, so a grader
fixed after a row was written applies to both arms alike. For each domain:
her accuracy alone and after her organ, the discordant pairs, how often her
gate admitted the request, which authority adopted a replacement, and what
the organ cost in seconds and in candidates cut at the turn's deadline.

Usage:
    g09_pilot_report.py --stem DIR_PREFIX [--domains d1,d2] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def domain_report(directory: Path, domain: str) -> dict[str, Any]:
    from tools.run_g09_organ import GRADERS, LOADERS

    truths = {task["id"]: task["truth"] for task in LOADERS[domain]()}
    grade = GRADERS[domain]
    ordinary = {p.name: json.loads(p.read_text()) for p in (directory / "rows" / "ordinary").glob("*.json")}
    organ = {p.name: json.loads(p.read_text()) for p in (directory / "rows" / "organ").glob("*.json")}
    regraded: dict[str, tuple[bool, bool | None]] = {}
    for name, row in ordinary.items():
        draft_ok = grade(row["text"], truths[row["id"]])[0]
        organ_ok = None
        if name in organ:
            delivered = organ[name].get("text")
            organ_ok = draft_ok if delivered is None else grade(delivered, truths[row["id"]])[0]
        regraded[name] = (draft_ok, organ_ok)
    paired = {name: value for name, value in regraded.items() if value[1] is not None}
    seconds = [row["seconds"] for row in organ.values() if "seconds" in row]
    generations = [g for row in organ.values() for g in row.get("generations", [])]
    return {
        "drafts": len(ordinary),
        "drafts_correct": sum(value[0] for value in regraded.values()),
        "paired": len(paired),
        "ordinary_correct": sum(value[0] for value in paired.values()),
        "organ_correct": sum(bool(value[1]) for value in paired.values()),
        "organ_only": sum(bool(value[1]) and not value[0] for value in paired.values()),
        "ordinary_only": sum(value[0] and not value[1] for value in paired.values()),
        "admitted": sum(bool(row.get("admitted")) for row in organ.values()),
        "authorities": dict(Counter(str(row.get("authority")) for row in organ.values())),
        "adopted": sum(bool(row.get("adopted")) for row in organ.values()),
        "errors": dict(Counter(str(row.get("error", ""))[:40] for row in organ.values() if row.get("error"))),
        "stood_down": sum(bool(row.get("stood_down")) for row in organ.values()),
        "organ_median_seconds": round(statistics.median(seconds), 1) if seconds else None,
        "generations": len(generations),
        "generations_cut_at_deadline": sum(bool(g.get("cut_at_turn_deadline")) for g in generations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True, help="e.g. ~/.aura/rlc-evidence/g09-pilot4")
    parser.add_argument("--date", default="20261007")
    parser.add_argument("--domains", default="math,code,planning,knowledge,transfer,bbeh,trip,aime")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for domain in args.domains.split(","):
        directory = Path(f"{args.stem}-{domain}-{args.date}").expanduser()
        if (directory / "rows" / "ordinary").exists():
            report[domain] = domain_report(directory, domain)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
