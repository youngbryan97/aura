#!/usr/bin/env python
"""Run one environment-family preflight."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", default="fixture")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.env == "terminal_grid:nethack":
        return subprocess.call([sys.executable, "-m", "pytest", "tests/environments/terminal_grid", "-q"], cwd=str(root))
    print(f"No preflight tests registered for {args.env} in mode {args.mode}; passing empty fixture preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
