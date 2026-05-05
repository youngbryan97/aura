#!/usr/bin/env python
"""Run the general environment architecture preflight gate."""
from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


TEST_TARGETS = [
    "tests/architecture/preflight",
    "tests/architecture/contracts",
    "tests/architecture/perception",
    "tests/architecture/modal",
    "tests/architecture/command",
    "tests/architecture/action_gateway",
    "tests/architecture/belief",
    "tests/architecture/planning",
    "tests/architecture/simulation",
    "tests/architecture/outcome",
    "tests/architecture/learning",
    "tests/architecture/governance",
    "tests/architecture/trace",
    "tests/architecture/resilience",
    "tests/architecture/generalization",
    "tests/architecture/benchmarks",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ok = compileall.compile_dir(root / "core", quiet=1) and compileall.compile_dir(root / "scripts", quiet=1) and compileall.compile_dir(root / "tests", quiet=1)
    if not ok:
        return 1
    existing = [target for target in TEST_TARGETS if (root / target).exists()]
    if not existing:
        return 0
    return subprocess.call([sys.executable, "-m", "pytest", *existing, "-q"], cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
