#!/usr/bin/env python3
"""Run Aura's deterministic RSI validation gauntlet."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.rsi_gauntlet import run_rsi_gauntlet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Aura repository root")
    parser.add_argument("--artifact-dir", default=None, help="Directory for gauntlet artifacts")
    parser.add_argument("--max-source-files", type=int, default=2500)
    args = parser.parse_args()

    result = asyncio.run(
        run_rsi_gauntlet(
            Path(args.root).resolve(),
            artifact_dir=Path(args.artifact_dir).resolve() if args.artifact_dir else None,
            max_source_files=args.max_source_files,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
