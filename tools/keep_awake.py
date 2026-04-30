#!/usr/bin/env python3
"""Start, stop, or inspect Aura's macOS keep-awake assertion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.keep_awake import get_keep_awake_controller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--display", action="store_true", help="Also prevent display sleep.")
    parser.add_argument("--no-ac-required", action="store_true", help="Do not add caffeinate -s AC-power assertion.")
    args = parser.parse_args()

    controller = get_keep_awake_controller()
    if args.action == "start":
        status = controller.start(keep_display_awake=args.display, require_ac_power=not args.no_ac_required)
    elif args.action == "stop":
        status = controller.stop()
    else:
        status = controller.status()
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
