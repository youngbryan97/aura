#!/usr/bin/env python3
"""Run the behavioral proof and append its REAL results to the L3 output ledger.

The ledger (core/evaluation/behavioral_ledger.py) is an append-only, hash-chained
record of every behavioral-proof run on the sealed held-out pack. Over time it
becomes the longitudinal L3 evidence: real outputs and scores you can audit and
verify weeks later, tamper-evident by construction.

    python tools/behavioral_proof_ledger.py            # run + record + summarize
    python tools/behavioral_proof_ledger.py --summary  # just print the ledger
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.behavioral_ledger import (  # noqa: E402
    BehavioralLedger,
    record_bundle_to_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="artifacts/behavioral_proof/output_ledger.jsonl")
    parser.add_argument("--output", default="artifacts/behavioral_proof/latest.json")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--summary", action="store_true", help="print the ledger summary and exit")
    args = parser.parse_args()

    ledger = BehavioralLedger(args.ledger)

    if not args.summary:
        from core.evaluation.behavioral_proof import run_behavioral_proof_bundle

        bundle = run_behavioral_proof_bundle(
            output_path=args.output,
            smoke_seed=args.seed,
            live_loop_seed=args.seed + 1,
            smoke_task_count=args.task_count,
            live_loop_task_count=8,
            receipt_root="artifacts/behavioral_proof/receipts",
        )
        recorded = record_bundle_to_ledger(bundle, ledger_path=args.ledger)
        print(f"Recorded {len(recorded)} entries (pack {bundle.smoke.pack_id[:12]}…).")

    summary = ledger.summary()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    if not summary["chain_ok"]:
        print(f"\nLedger chain INVALID: {summary['chain_detail']}")
        return 1
    if not summary["held_out_integrity_ok"]:
        print("\nHeld-out integrity broken: a sealed pack changed manifest.")
        return 1
    print(f"\nLedger verified: {summary['total_runs']} runs, chain intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
