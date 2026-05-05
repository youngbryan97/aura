#!/usr/bin/env python
"""Run a safe headless canary through the environment kernel."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment import ActionIntent, EnvironmentKernel
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler, NetHackTerminalGridAdapter


async def run_nethack(steps: int, trace_path: Path) -> int:
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    kernel = EnvironmentKernel(
        adapter=adapter,
        state_compiler=NetHackStateCompiler(),
        command_compiler=NetHackCommandCompiler(),
        trace_path=trace_path,
    )
    await kernel.start(run_id="canary-safe", seed=1)
    try:
        for idx in range(steps):
            intent = ActionIntent(name="observe", expected_effect="information_gain")
            if idx % 5 == 1:
                intent = ActionIntent(name="inventory", expected_effect="inventory_modal", tags={"inspect"})
            await kernel.step(intent)
    finally:
        await kernel.close()
    return 0 if trace_path.exists() and trace_path.stat().st_size > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--trace", default="artifacts/latest_canary_trace.jsonl")
    args = parser.parse_args()
    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()
    if args.env != "terminal_grid:nethack":
        raise SystemExit(f"canary not registered for {args.env}")
    return asyncio.run(run_nethack(args.steps, trace_path))


if __name__ == "__main__":
    raise SystemExit(main())
