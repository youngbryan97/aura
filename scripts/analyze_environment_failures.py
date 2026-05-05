#!/usr/bin/env python
"""Summarize failure classes from a trace replay postmortem."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment.failure_taxonomy import FAILURE_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()
    counts: Counter[str] = Counter()
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lesson = str(row.get("outcome_assessment", {}).get("lesson", ""))
        if "prediction" in lesson:
            counts["prediction_error"] += 1
        if row.get("execution_result", {}).get("error"):
            counts["execution_error"] += 1
    print(json.dumps({k: counts.get(k, 0) for k in sorted(FAILURE_CLASSES) if counts.get(k, 0)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
