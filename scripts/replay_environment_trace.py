#!/usr/bin/env python
"""Replay an environment black-box trace."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment.replay import EnvironmentTraceReplay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path")
    parser.add_argument("--postmortem-out", default="")
    args = parser.parse_args()
    result = EnvironmentTraceReplay().load(args.trace_path)
    if args.postmortem_out:
        Path(args.postmortem_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.postmortem_out).write_text(json.dumps(result.postmortem, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": result.ok, "rows": len(result.rows), "corrupt_rows": result.corrupt_rows}, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
