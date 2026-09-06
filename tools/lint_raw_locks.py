#!/usr/bin/env python3
"""Count the locks lockdep cannot see, and refuse a rise.

Lockdep finds an ABBA deadlock without the deadlock happening, and it only
sees locks it wraps. A raw ``threading.Lock()`` is therefore not a smaller
version of a checked one — it is invisible to the thing that would have found
the deadlock, and the guide says so.

There are hundreds. This does not convert them; it stops there being more.
The baseline in ``config/raw_lock_baseline.json`` only goes down.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "raw_lock_baseline.json"

#: The package that implements checked locks, and the package that must not
#: depend on it. lockdep cannot wrap itself, and the writer gateway is below
#: it in the layering.
ALLOWED = {
    "core/runtime/lockdep.py",
}

#: What counts. `threading.Condition` and `Semaphore` are not lock-ordering
#: primitives in the sense lockdep checks, so they are out of scope rather
#: than quietly excused.
_RAW = {"Lock", "RLock"}


def raw_locks(root: Path | None = None) -> list[str]:
    """Every `threading.Lock()` / `RLock()` under core, as path:line."""
    here = root or ROOT
    found: list[str] = []
    for path in sorted((here / "core").rglob("*.py")):
        rel = str(path.relative_to(here))
        if rel in ALLOWED or "__pycache__" in rel:
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"))
        except (SyntaxError, OSError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = node.func
            if (
                isinstance(call, ast.Attribute)
                and call.attr in _RAW
                and isinstance(call.value, ast.Name)
                and call.value.id == "threading"
            ):
                found.append(f"{rel}:{node.lineno}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="write the current count as the new ceiling")
    args = parser.parse_args()

    found = raw_locks()
    if args.baseline:
        BASELINE.write_text(
            json.dumps(
                {
                    "note": "Locks lockdep cannot see. This number only goes "
                            "down. Convert with checked_lock, or adopt an "
                            "existing lock with instrument().",
                    "count": len(found),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {len(found)}")
        return 0

    try:
        allowed = int(json.loads(BASELINE.read_text("utf-8"))["count"])
    except (OSError, ValueError, KeyError):
        print(f"no baseline; {len(found)} raw locks under core")
        return 1
    if len(found) > allowed:
        print(f"❌ {len(found)} raw locks under core, baseline {allowed}")
        for one in found[-10:]:
            print(f"   • {one}")
        print("   use checked_lock(name) so lockdep can see it")
        return 1
    print(f"✅ {len(found)} raw locks under core (baseline {allowed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
