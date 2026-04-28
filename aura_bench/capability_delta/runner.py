"""Capability-delta runner — score every task under every profile.

The runner is intentionally synchronous and CPU-only.  Adapters that
need GPU/network do their own work; the harness just orchestrates and
computes deltas.

The capability delta for a task is::

    delta(task) = score[full] - score[base_llm_only]

aggregated as the mean across the suite, with per-profile means
exposed for ablation analysis.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from aura_bench.capability_delta.adapter import (
    BenchAdapter,
    BenchTask,
    LLMCallable,
    TaskOutcome,
)
from aura_bench.capability_delta.profiles import (
    ABLATION_PROFILES,
    AblationProfile,
)


@dataclass
class DeltaResult:
    adapter_name: str
    profile_name: str
    n_tasks: int
    mean_score: float
    pass_rate: float
    outcomes: List[TaskOutcome] = field(default_factory=list)


@dataclass
class DeltaReport:
    """Top-level report from a capability-delta run."""

    adapter_name: str
    started_at: float
    finished_at: float
    by_profile: Dict[str, DeltaResult]
    capability_delta: float = 0.0  # full - base_llm_only

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "adapter_name": self.adapter_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "capability_delta": self.capability_delta,
            "by_profile": {
                name: {
                    "adapter_name": r.adapter_name,
                    "profile_name": r.profile_name,
                    "n_tasks": r.n_tasks,
                    "mean_score": r.mean_score,
                    "pass_rate": r.pass_rate,
                    "outcomes": [asdict(o) for o in r.outcomes],
                }
                for name, r in self.by_profile.items()
            },
        }
        return out


def run_capability_delta(
    adapter: BenchAdapter,
    *,
    llm: LLMCallable,
    profiles: Iterable[AblationProfile] = ABLATION_PROFILES,
    max_tasks: int = 0,
) -> DeltaReport:
    """Run every task in ``adapter.tasks()`` under every profile.

    ``max_tasks`` (when > 0) caps the suite size — useful for smoke
    tests so the harness can be exercised without running the whole
    real benchmark.
    """
    started = time.time()
    profiles_list = list(profiles)
    by_profile: Dict[str, DeltaResult] = {}

    all_tasks: List[BenchTask] = list(adapter.tasks())
    if max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]

    for profile in profiles_list:
        outcomes: List[TaskOutcome] = []
        for task in all_tasks:
            outcome = adapter.run(task, profile.name, llm)
            outcomes.append(outcome)
        scores = [o.score for o in outcomes] or [0.0]
        passes = sum(1 for o in outcomes if o.success)
        by_profile[profile.name] = DeltaResult(
            adapter_name=adapter.name,
            profile_name=profile.name,
            n_tasks=len(outcomes),
            mean_score=statistics.fmean(scores),
            pass_rate=passes / max(1, len(outcomes)),
            outcomes=outcomes,
        )

    delta = 0.0
    full = by_profile.get("full")
    base = by_profile.get("base_llm_only")
    if full is not None and base is not None:
        delta = float(full.mean_score - base.mean_score)

    return DeltaReport(
        adapter_name=adapter.name,
        started_at=started,
        finished_at=time.time(),
        by_profile=by_profile,
        capability_delta=delta,
    )
