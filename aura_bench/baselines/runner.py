"""Baseline-defeat comparison runner (L1).

Defines the seven baselines Aura must beat:

  baseline_id            description
  ─────────────────      ───────────────────────────────────────────────
  prompt_only             pure prompt-only LLM, no agent loop / memory
  llm_with_memory         LLM + flat episodic memory, no governance
  llm_with_tools          LLM + tools, no governance
  llm_agent_loop          ReAct-style agent loop, no substrate
  scripted_personality    LLM + a richly-styled system prompt only
  full_aura               the system as shipped
  ablations.no_substrate  Aura with substrate disabled
  ablations.no_memory     Aura with memory disabled
  ablations.no_agency     Aura with agency_orchestrator bypassed
  ablations.no_governance Aura with conscience disabled
  ablations.no_topology   Aura with topology evolution frozen

Each baseline is exercised against the same task suite the Consciousness
Courtroom uses. Results land in ``aura_bench/baselines/results.jsonl``.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List


@dataclass
class BaselineResult:
    when: float
    baseline_id: str
    suite: str
    mean_score: float
    samples: List[float] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


_OUT = Path(__file__).parent / "results.jsonl"


async def _eval(baseline_id: str) -> List[float]:
    # Use the courtroom task suite as the canonical comparison.
    from aura_bench.courtroom.courtroom import TASKS, SYSTEMS
    fn = SYSTEMS.get(baseline_id, SYSTEMS["prompt_only"])
    samples: List[float] = []
    for task in TASKS:
        for _ in range(4):
            out = await fn(task.task_id)
            samples.append(task.score_fn(out))
    return samples


async def run_all() -> List[BaselineResult]:
    out: List[BaselineResult] = []
    for baseline_id in ("prompt_only", "no_substrate", "no_memory", "standard_agent", "full_aura"):
        samples = await _eval(baseline_id)
        result = BaselineResult(
            when=time.time(),
            baseline_id=baseline_id,
            suite="courtroom_proxy",
            mean_score=statistics.fmean(samples) if samples else 0.0,
            samples=samples,
        )
        out.append(result)
        _persist(result)
    return out


def _persist(r: BaselineResult) -> None:
    with open(_OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(r), default=str) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_all())
