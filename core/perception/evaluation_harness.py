"""Evaluation harness for embodied cognition stress tests.

The harness is generic: NetHack can supply depth/score/death metrics, a
browser task can supply task-completion metrics, and robotics can supply
collision/energy metrics. The important thing is held-out repeatability,
not one lucky trace.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbodiedRunMetrics:
    run_id: str
    domain: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    progress: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    death_or_failure_cause: str = ""
    held_out: bool = False
    seed: Optional[str] = None

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(amount)

    def set_progress(self, key: str, value: float) -> None:
        self.progress[key] = float(value)

    def finish(self, cause: str = "") -> None:
        self.finished_at = time.time()
        self.death_or_failure_cause = cause

    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)


class EmbodiedEvaluationHarness:
    """Aggregates repeated runs and guards against single-run self-deception."""

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.runs: List[EmbodiedRunMetrics] = []

    def start_run(self, run_id: str, *, held_out: bool = False, seed: Optional[str] = None) -> EmbodiedRunMetrics:
        run = EmbodiedRunMetrics(run_id=run_id, domain=self.domain, held_out=held_out, seed=seed)
        self.runs.append(run)
        return run

    def summarize(self, progress_key: str = "progress") -> Dict[str, Any]:
        values = [run.progress.get(progress_key, 0.0) for run in self.runs]
        held_out_values = [
            run.progress.get(progress_key, 0.0) for run in self.runs if run.held_out
        ]
        causes: Dict[str, int] = {}
        for run in self.runs:
            if run.death_or_failure_cause:
                causes[run.death_or_failure_cause] = causes.get(run.death_or_failure_cause, 0) + 1
        return {
            "domain": self.domain,
            "runs": len(self.runs),
            "median": statistics.median(values) if values else 0.0,
            "best": max(values) if values else 0.0,
            "held_out_median": statistics.median(held_out_values) if held_out_values else 0.0,
            "failure_causes": causes,
            "total_duration_s": round(sum(run.duration_s for run in self.runs), 3),
        }
