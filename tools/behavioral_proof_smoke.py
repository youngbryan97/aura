#!/usr/bin/env python3
"""Run the behavioral proof smoke gate and write a JSON artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.behavioral_proof import run_behavioral_proof_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/behavioral_proof/latest.json")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--task-count", type=int, default=50)
    args = parser.parse_args()

    report = run_behavioral_proof_smoke(
        output_path=args.output,
        seed=args.seed,
        task_count=args.task_count,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
