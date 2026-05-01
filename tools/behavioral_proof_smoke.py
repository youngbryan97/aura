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

from core.evaluation.behavioral_proof import run_behavioral_proof_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/behavioral_proof/latest.json")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--live-loop-seed", type=int, default=20260502)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--live-loop-task-count", type=int, default=8)
    parser.add_argument("--receipt-root", default="artifacts/behavioral_proof/receipts")
    args = parser.parse_args()

    report = run_behavioral_proof_bundle(
        output_path=args.output,
        smoke_seed=args.seed,
        live_loop_seed=args.live_loop_seed,
        smoke_task_count=args.task_count,
        live_loop_task_count=args.live_loop_task_count,
        receipt_root=args.receipt_root,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
