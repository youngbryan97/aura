#!/usr/bin/env python3
"""Every world she has learned, read again by the code in this checkout: what no longer holds.

Each part of a world's record is read by the module that keeps it, which registers how that part is read again
(core/runtime/what_she_learned_still_holds.py). A change to how she learns or reads any world is replayed here against
what she has actually learned, which tests that start learning from nothing never see.

Usage:
    tools/what_she_learned_still_holds.py [--root DIR] [--world NAME ...]

Exit 0 when everything read still holds, 1 when something does not, 2 when no world was read.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The modules that keep a part of a world's record, imported so each registers how it reads its part again.
KEEPERS = ("core.skills.screen_pursuit", "core.skills.screen_pursuit_board_of_pieces")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None,
                        help="the state root whose state/worlds is read (default: hers, ~/.aura)")
    parser.add_argument("--world", action="append", default=None, help="read only this world (may repeat)")
    args = parser.parse_args(argv)
    from core.runtime import what_she_learned
    from core.runtime.what_she_learned_still_holds import readers, what_no_longer_holds

    what_she_learned._KEPT_IN = (args.root or Path.home() / ".aura") / "state" / "worlds"
    for keeper in KEEPERS:
        importlib.import_module(keeper)
    found, read = what_no_longer_holds(args.world)
    print(f"read {read} world(s) through the readers of: {', '.join(readers())}")
    for world, lines in sorted(found.items()):
        for line in lines:
            print(f"FAIL {world}: {line}")
    if not read:
        return 2
    print("everything she learned still holds" if not found else f"{len(found)} world(s) with something that does not")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
