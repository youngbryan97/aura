#!/usr/bin/env python3
"""G09's numbers: her drafts against the same requests after her kept procedures, per domain and pooled.

Each row of ``rows/kept`` (tools/g09_kept_procedure_arm.py) carries both
outcomes for one request, so pairing is exact. Per domain, and per group
within it: both accuracies with 95% Wilson intervals, how often a kept
procedure answered, the discordant pairs, and an exact two-sided McNemar
test on them. The pooled line sums the domains' discordant pairs; it is
printed beside them, never instead of them.

Usage:
    g09_analysis.py --domain NAME=DIR [--domain NAME=DIR ...] --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.g12_analysis import two_sided_mcnemar, wilson

    n = len(rows)
    draft = sum(bool(r["draft_correct"]) for r in rows)
    kept = sum(bool(r["correct"]) for r in rows)
    gained = sum(bool(r["correct"]) and not r["draft_correct"] for r in rows)
    lost = sum(bool(r["draft_correct"]) and not r["correct"] for r in rows)
    return {"requests": n, "answered_by_a_kept_procedure": sum(bool(r["adopted"]) for r in rows),
            "draft_correct": draft, "draft_interval": wilson(draft, n),
            "kept_correct": kept, "kept_interval": wilson(kept, n),
            "gained": gained, "lost": lost, "two_sided_p": two_sided_mcnemar(gained, lost)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", required=True, help="NAME=DIR, DIR holding rows/kept")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"domains": {}}
    pooled: list[dict[str, Any]] = []
    for spec in args.domain:
        name, directory = spec.split("=", 1)
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((Path(directory).expanduser() / "rows" / "kept").glob("*.json"))]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get("group"))].append(row)
        report["domains"][name] = {**summary(rows),
                                   "by_group": {g: summary(r) for g, r in sorted(groups.items())}}
        pooled.extend(rows)
    report["pooled"] = summary(pooled)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"pooled": report["pooled"],
                      **{name: {k: v for k, v in d.items() if k != "by_group"}
                         for name, d in report["domains"].items()}}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
