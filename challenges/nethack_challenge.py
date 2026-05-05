#!/usr/bin/env python
"""Run the canonical environment kernel against NetHack as a stress test.

NetHack is only the adapter here. The loop is Aura's general environment OS:
observe -> compile typed state -> update belief/map -> plan/select intent ->
simulate -> govern/gate -> compile command -> execute -> semantic diff ->
learn -> trace/postmortem.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment import EnvironmentCapabilityMatrix, EnvironmentKernel
from core.environments.terminal_grid import (
    NetHackCommandCompiler,
    NetHackStateCompiler,
    NetHackTerminalGridAdapter,
)
from core.environments.terminal_grid.nethack_adapter import EnvironmentMode

LOG = logging.getLogger("Aura.NetHackChallenge")


def _mode(value: str) -> EnvironmentMode:
    return {
        "auto": EnvironmentMode.AUTO,
        "strict_real": EnvironmentMode.STRICT_REAL,
        "simulated": EnvironmentMode.SIMULATED,
    }[value]


async def run_challenge(*, steps: int, trace_path: Path, mode: EnvironmentMode) -> int:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    adapter = NetHackTerminalGridAdapter(mode=mode)
    kernel = EnvironmentKernel(
        adapter=adapter,
        state_compiler=NetHackStateCompiler(),
        command_compiler=NetHackCommandCompiler(),
        trace_path=trace_path,
    )

    run_id = f"nethack-{int(time.time())}"
    await kernel.start(run_id=run_id, seed=None)
    try:
        EnvironmentCapabilityMatrix().audit(kernel).require_clean()
        for idx in range(steps):
            frame = await kernel.step(intent=None)
            outcome = frame.outcome_assessment
            parsed = frame.post_parsed_state or frame.parsed_state
            LOG.info(
                "step=%s context=%s phase=%s intent=%s score=%s events=%s",
                idx + 1,
                parsed.context_id,
                kernel.episode.state.phase if kernel.episode else "unknown",
                frame.action_intent.name if frame.action_intent else "none",
                round(outcome.success_score, 3) if outcome else None,
                outcome.observed_events[:5] if outcome else [],
            )
            if kernel.run_manager.current_record is None:
                break
    finally:
        await kernel.close()

    latest = kernel.run_manager.records[-1] if kernel.run_manager.records else None
    if latest:
        LOG.info(
            "run=%s reason=%s steps=%s score=%.3f trace=%s",
            latest.run_id,
            latest.terminal_reason,
            latest.total_steps,
            latest.final_score,
            trace_path,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--mode", choices=["auto", "strict_real", "simulated"], default="strict_real")
    parser.add_argument("--trace", default=str(Path.home() / ".aura" / "logs" / "nethack" / "kernel_trace.jsonl"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(run_challenge(steps=args.steps, trace_path=Path(args.trace), mode=_mode(args.mode)))


if __name__ == "__main__":
    raise SystemExit(main())
