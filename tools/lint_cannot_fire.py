"""Report parameters every production caller hands an empty collection.

A gate, not a wall. It fails on a NEW finding and on a reviewed entry whose
reason no longer applies; it does not fail on the recorded ones, because
turning an inherited list into a blocker is how a gate stops a runtime instead
of improving it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core.verify.can_this_ever_fire import report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    result = report(arguments.root)
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"   {len(result['found'])} parameter(s) every caller passes empty")
    for identity in result["outstanding"]:
        print(f"   • outstanding, no reason recorded yet: {identity}")
    for identity in result["new"]:
        print(f"   ✗ NEW: {identity}")
    for identity in result["reviewed_gone"]:
        print(f"   ✗ reviewed entry no longer found, remove it: {identity}")

    if result["new"]:
        print(
            "\n   A caller that always passes an empty collection has written down "
            "that this mechanism does not run. Give it something, or record why "
            "empty is right in config/switched_off_arguments.json."
        )
        return 1
    if result["reviewed_gone"]:
        print("\n   A reason survives its finding. Remove the stale entries.")
        return 1
    print("   ✅ nothing new can be switched off from a call site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
