#!/usr/bin/env python3
"""Run Aura's autonomous successor RSI trial.

This is the wall-clock runner for the final proof shape. It repeatedly invokes
the autonomous successor engine, writes every cycle as frozen artifacts, and
emits a compact report suitable for external reproduction. Defaults are short
so the command is safe around live LoRA training; use --duration-s for 24h/72h
trials.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.autonomous_rsi import AutonomousSuccessorEngine
from core.runtime.atomic_writer import atomic_write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura autonomous RSI successor trial")
    parser.add_argument("--artifact-dir", default="data/autonomous_rsi_trial", help="Directory for frozen generations and reports")
    parser.add_argument("--seed", type=int, default=4401, help="Base external-custody seed")
    parser.add_argument("--tasks-per-generation", type=int, default=40, help="Hidden tasks per generation")
    parser.add_argument("--generations", type=int, default=4, help="Generations per cycle")
    parser.add_argument("--cycles", type=int, default=1, help="Maximum cycles to run")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional wall-clock duration cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    cycle = 0

    while cycle < max(1, args.cycles):
        if args.duration_s > 0 and time.time() - started >= args.duration_s:
            break
        cycle += 1
        cycle_dir = root / f"cycle_{cycle:04d}"
        engine = AutonomousSuccessorEngine(
            cycle_dir,
            seed=args.seed + cycle - 1,
            tasks_per_generation=args.tasks_per_generation,
        )
        result = engine.run(generations=args.generations)
        results.append(result.to_dict())
        atomic_write_text(
            root / "latest_cycle_result.json",
            json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    report = {
        "cycles": len(results),
        "duration_s": round(time.time() - started, 3),
        "verdicts": [result["verdict"]["verdict"] for result in results],
        "all_reproduced": all(result["independently_reproduced"] for result in results),
        "all_mirrors_ok": all(result["mirror_ok"] for result in results),
        "results": results,
    }
    atomic_write_text(
        root / "autonomous_rsi_trial_report.json",
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, sort_keys=True))
    return 0 if results and report["all_reproduced"] and report["all_mirrors_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
