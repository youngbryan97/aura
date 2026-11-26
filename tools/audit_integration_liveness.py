#!/usr/bin/env python3
"""Report which optional integrations are actually alive.

Fails only on BROKEN integrations — a module that is installed but cannot be
imported. An absent optional dependency is legitimate and is reported, not
failed, because that distinction is exactly what silent decay erases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.integration_liveness import (  # noqa: E402
    IMPORT_TIMEOUT_S,
    probe_all,
)

_MARK = {"live": "ok", "broken": "failed", "absent": "○"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    parser.add_argument("--out", default="", help="write the report to this path")
    parser.add_argument("--timeout", type=float, default=IMPORT_TIMEOUT_S)
    parser.add_argument(
        "--strict-absent",
        action="store_true",
        help="also fail when an optional integration is not installed",
    )
    args = parser.parse_args()

    report = probe_all(timeout_s=args.timeout)
    payload = report.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Integration liveness (real imports, not find_spec)")
        print("=" * 62)
        for result in report.results:
            mark = _MARK.get(result.state, "?")
            facing = " [user-facing]" if result.integration.user_facing else ""
            print(f"  {mark} {result.integration.name:<16} {result.state:<7} "
                  f"{result.duration_s:>6.2f}s{facing}")
            if result.state != "live":
                print(f"      powers: {result.integration.powers}")
                if result.detail:
                    print(f"      detail: {result.detail}")
        print("-" * 62)
        print(f"  live {len(report.live)} · broken {len(report.broken)} · "
              f"absent {len(report.absent)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failed = list(report.broken)
    if args.strict_absent:
        failed += report.absent

    if failed:
        print()
        for result in failed:
            print(
                f"{result.integration.name} is {result.state}: "
                f"{result.integration.powers}"
            )
        print(
            "\nA module that is installed but cannot be imported is the silent-decay "
            "defect this gate exists to catch."
        )
        return 1

    print("\nNo broken integrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
