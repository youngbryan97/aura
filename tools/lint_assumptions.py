#!/usr/bin/env python3
"""Gate: no assumption may claim a checker that does not exist.

A DISCHARGED assumption pointing at a renamed test is worse than an
UNDISCHARGED one. It occupies the slot that would otherwise show up as debt and
reports the system as covered — the exact "absence of a check reported as a
passed check" failure this repository keeps finding in other forms.

This gate is deliberately narrow. It does not judge whether the assumption set
is complete, whether an assumption is wise, or whether the debt is acceptable;
none of that is machine-checkable. It checks the one property that always rots
on its own: that named things exist.

Run: ``python tools/lint_assumptions.py`` / ``--json`` / ``--list``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.verify import system_assumptions  # noqa: E402,F401  (import populates)
from core.verify.assumptions import (  # noqa: E402
    AssumptionStatus,
    assumption_report,
    get_assumption_registry,
    verify_dischargers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full ledger as JSON")
    parser.add_argument("--list", action="store_true", help="print the ledger for humans")
    args = parser.parse_args()

    registry = get_assumption_registry()
    failures = verify_dischargers()

    if args.json:
        print(json.dumps(assumption_report(), indent=2))
        return 1 if failures else 0

    if args.list:
        for status in AssumptionStatus:
            group = registry.by_status(status)
            if not group:
                continue
            print(f"\n{status.value.upper()} ({len(group)})")
            for item in group:
                print(f"  [{item.scope}] {item.id}")
                print(f"      {item.statement}")
                if item.discharged_by:
                    print(f"      checked by: {item.discharged_by}")
        print()

    counts = {s.value: len(registry.by_status(s)) for s in AssumptionStatus}
    print(
        f"{len(registry)} assumptions — "
        f"{counts['discharged']} discharged, "
        f"{counts['undischarged']} undischarged, "
        f"{counts['outside_the_system']} outside the system"
    )

    if failures:
        print("\nassumptions naming a checker that does not exist:")
        for failure in failures:
            print(f"   {failure}")
        print(
            "\nEither point the assumption at the checker's new name, or change its "
            "status to undischarged. A discharge that cannot be located is not one."
        )
        return 1

    print("every discharged assumption names a checker that exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
