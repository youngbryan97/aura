#!/usr/bin/env python3
"""Run the AGI proof gauntlet and write the receipts.

    python tools/run_agi_gauntlet.py                 # everything runnable
    python tools/run_agi_gauntlet.py --gate 4        # one of them
    python tools/run_agi_gauntlet.py --quick         # small, for a check
    python tools/run_agi_gauntlet.py --into somewhere

A single benchmark is not the claim; the intersection is. Twelve of the
eighteen run here. The other six need a private holdout, a sealed image, a
post-cutoff repository, hours of wall clock, multimodal assets or people
playing colleagues, and each of those prints the protocol for running it
rather than a number, because a harness that substitutes a proxy for the
thing it names is how a system gets credited with a capability nobody
measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.agi_gauntlet.bundle import write_the_bundle  # noqa: E402
from tools.agi_gauntlet.gates import THE_GATES, run_a_gate  # noqa: E402
from tools.agi_gauntlet.protocol import take_the_freeze  # noqa: E402

QUICK = {
    "instances": 12,
    "worlds": 10,
    "trajectories": 8,
    "episodes": 6,
    "pairs": 40,
    "questions": 12,
}
FULL = {
    # Two hundred, because forty is not enough to decide an 0.85 bar.
    #
    # The standard error at that share on forty instances is about 0.056, so
    # an ordinary run swings by a tenth either way — and it did: across six
    # freezes the same solver measured 0.65, 0.77, 0.80, 0.825, 0.85 and
    # 0.925. Any one of those read as a verdict is a coin landing. At two
    # hundred the interval is about 0.05 wide and the number means something.
    "instances": 200,
    "worlds": 30,
    "trajectories": 30,
    "episodes": 12,
    "pairs": 220,
    "questions": 40,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=int, action="append", default=[])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--into", default="artifacts/agi_gauntlet")
    parser.add_argument("--json", action="store_true", help="the report, and nothing else")
    args = parser.parse_args()

    freeze = take_the_freeze()
    options = dict(QUICK if args.quick else FULL)
    into = Path(args.into) / time.strftime("%Y%m%d-%H%M%S")
    wanted = set(args.gate) if args.gate else None

    receipts = []
    for gate in THE_GATES:
        if wanted is not None and gate.number not in wanted:
            continue
        receipt = run_a_gate(gate, freeze, options)
        receipt.write(into)
        receipts.append((gate, receipt))
        if not args.json:
            print(_a_line(gate, receipt), flush=True)

    ran = [one for _g, one in receipts if one.ran]
    passed = [one for one in ran if one.passed]
    report = {
        "freeze": freeze.to_dict(),
        "gates": len(receipts),
        "ran": len(ran),
        "passed": len(passed),
        "not_run": [
            {"gate": g.name, "needs": r.why_not}
            for g, r in receipts
            if not r.ran
        ],
        "results": [
            {"gate": g.name, "ran": r.ran, "passed": r.passed, **r.measurements}
            for g, r in receipts
        ],
    }
    (into / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_the_bundle(into, freeze)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print()
        print(f"{len(passed)} of {len(ran)} runnable gates passed; "
              f"{len(receipts) - len(ran)} need an evaluator this harness is not.")
        if not freeze.trustworthy:
            print(
                "The freeze is not trustworthy: the tree is dirty, so the commit "
                "names something other than what ran and every environment here "
                "was derived from a description of the system rather than the "
                "system."
            )
        print(f"receipts: {into}")
    return 0


def _a_line(gate, receipt) -> str:
    if not receipt.ran:
        return f"{gate.number:2}  ―  {gate.name}\n      needs: {receipt.why_not}"
    mark = "PASS" if receipt.passed else "FAIL"
    said = ", ".join(
        f"{key}={value}"
        for key, value in receipt.measurements.items()
        if isinstance(value, (int, float, str, bool)) and key != "passed"
    )
    return f"{gate.number:2} {mark}  {gate.name}\n      {said[:160]}"


if __name__ == "__main__":
    raise SystemExit(main())
