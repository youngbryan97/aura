#!/usr/bin/env python3
"""Print the three proofs the interiority layer makes about itself, and fail
the build if any of them stops holding.

Run by ``make interiority``. The output is meant to be read: it names the
faculties, the interventions, and the downstream quantities that move,
so a reviewer can check the claims rather than trust a green tick.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.interiority.faculties import load_all  # noqa: E402
from core.interiority.longitudinal import summary as longitudinal_summary  # noqa: E402
from core.interiority.proving import summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_all()
    report = summary()

    counterfactuals = report["counterfactuals"]
    nulls = report["nulls"]
    ablation = report["ablation"]

    print(f"faculties declared            {report['faculties']}")
    print(
        f"declared refutations run      {counterfactuals['held']}/{counterfactuals['run']}"
    )
    print(f"silent in their null world    {nulls['held']}/{nulls['run']}")
    print(
        f"reach a measured behaviour    {ablation['reach_behaviour']}/{ablation['run']}"
    )
    longitudinal = longitudinal_summary()
    print(
        f"long-running properties       {longitudinal['held']}/{longitudinal['episodes']}"
    )

    if args.verbose:
        print("\nwhat each faculty moves when it is removed:")
        for faculty, result in sorted(ablation["by_faculty"].items()):
            moved = ", ".join(
                f"{k}={v:+.3f}" for k, v in sorted(result["deltas"].items())
            )
            extra = ""
            if result["unblocked"]:
                extra += f" unblocks={result['unblocked']}"
            if result["unheld"]:
                extra += f" releases={result['unheld']}"
            print(f"  {faculty:<32} {moved}{extra}")

    report["longitudinal"] = longitudinal

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nfull report written to {args.json}")

    problems: list[str] = []
    for failure in counterfactuals["failed"]:
        problems.append(
            f"{failure['faculty']} :: {failure['counterfactual']} expected "
            f"{failure['expect']} but got {failure['detail']}"
        )
    for failure in nulls["failed"]:
        problems.append(
            f"{failure['faculty']} fired at {failure['intensity']:.4f} with "
            "nothing present"
        )
    for faculty in ablation["decorative"]:
        problems.append(
            f"{faculty} changes nothing any subsystem reads when removed"
        )
    for failure in longitudinal["failed"]:
        problems.append(f"{failure['episode']}: {failure['detail']}")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\n✅ every declared refutation holds, nothing fires on nothing, "
          "no faculty is decorative, and every long-running property keeps "
          "its shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
