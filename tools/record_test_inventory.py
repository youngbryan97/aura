#!/usr/bin/env python3
"""How many tests there are, recorded once instead of typed into eight documents.

README, TESTING, HOW_IT_WORKS, AGENTS, CLAUDE, OPERATOR_GUIDE, REVIEWER_PACKET
and ROADMAP each stated the size of the test suite. All eight said "34,382
tests across 2,373 test files" long after the tree had grown past 40,000,
because the number was typed rather than derived, and a typed number goes stale
in eight places at once.

This writes the count to config/test_inventory.json. `make doc-drift` reads
that file and fails any document whose stated count disagrees, so the numbers
move together or the gate stops the commit.

Collection is the slow part — around 25 seconds on an idle host — which is why
this is its own target rather than something the lint gate runs.

    python tools/record_test_inventory.py           # show the current count
    python tools/record_test_inventory.py --write   # record it
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "config" / "test_inventory.json"

#: The marker `make test` runs under. Tests outside it need hardware or a
#: network, so they are counted but not run.
MARKER = "not live and not network and not external"

SUMMARY = re.compile(r"^(\d+)/(\d+) tests collected|^(\d+) tests collected", re.M)


def collect() -> tuple[int, int, int]:
    """Return (collected_under_marker, total, distinct_files)."""
    env = dict(os.environ, AURA_TESTING="1")
    env.setdefault("AURA_LOG_DIR", "/tmp/aura_test_inventory_logs")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-m", MARKER],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    out = proc.stdout
    m = SUMMARY.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise SystemExit("collection produced no summary line")
    if m.group(1):
        selected, total = int(m.group(1)), int(m.group(2))
    else:
        selected = total = int(m.group(3))
    files = len({line.split("::", 1)[0] for line in out.splitlines() if "::" in line})
    return selected, total, files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="record the count")
    args = ap.parse_args()

    selected, total, files = collect()
    record = {
        "collected": total,
        "run_by_make_test": selected,
        "files": files,
        "marker": MARKER,
        "recorded_on": date.today().isoformat(),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.strip(),
    }
    print(f"{total:,} tests collect across {files:,} files; "
          f"{selected:,} run under the offline marker")

    if args.write:
        INVENTORY.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY.write_text(json.dumps(record, indent=1) + "\n")
        print(f"recorded to {INVENTORY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
