#!/usr/bin/env python3
"""What an induction run produced, kind by kind, and what writing it cost.

Reads ``candidates.jsonl`` and ``requests/round*.jsonl`` from each run
directory (tools/induce_g09_procedures.py): per kind, the functions written,
how many were kept, the round each kept one came in, the most known answers a
function agreed with, and why the rest failed (an answer that disagreed, a
raise, a timeout, nothing returned); per run, the requests decoded, the
tokens and the seconds.

Usage:
    g09_induction_summary.py DIR [DIR ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def failure_kind(outcome: dict[str, Any]) -> str:
    error = str(outcome.get("error") or "")
    if not error:
        return "disagreed"
    if error.startswith("timeout"):
        return "timed_out"
    if "returned nothing" in error or "no result" in error:
        return "returned_nothing"
    return "raised"


def summarize(directory: Path) -> dict[str, Any]:
    kinds: dict[str, dict[str, Any]] = defaultdict(lambda: {"written": 0, "kept": 0, "kept_rounds": [],
                                                            "best": None, "failures": Counter()})
    path = directory / "candidates.jsonl"
    for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else []):
        record = json.loads(line)
        kind = kinds[record["family"]]
        kind["written"] += 1
        known = record["pool"][-1] + record["sealed"][-1]
        agreed = record["pool"][0] + record["sealed"][0]
        if kind["best"] is None or agreed > kind["best"][0]:
            kind["best"] = [agreed, known]
        if record["admitted"]:
            kind["kept"] += 1
            kind["kept_rounds"].append(record["round"])
        for failure in record.get("failures", []):
            kind["failures"][failure_kind(failure)] += 1
    decoded = {"requests": 0, "tokens": 0, "seconds": 0.0, "cut_at_budget": 0}
    for log in sorted((directory / "requests").glob("round*.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            decoded["requests"] += 1
            decoded["tokens"] += int(record.get("generated_tokens") or 0)
            decoded["seconds"] += float(record.get("seconds") or 0.0)
            decoded["cut_at_budget"] += record.get("termination") != "stop"
    decoded["seconds"] = round(decoded["seconds"], 1)
    return {"kinds": {name: {**value, "failures": dict(value["failures"])} for name, value in sorted(kinds.items())},
            "kinds_kept": sum(1 for value in kinds.values() if value["kept"]), "decoded": decoded}


def main() -> int:
    report = {str(Path(arg).expanduser()): summarize(Path(arg).expanduser()) for arg in sys.argv[1:]}
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
