"""Independent validation loops, ablations, and long-run receipts."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .schemas import stable_hash


@dataclass
class BenchmarkTask:
    task_id: str
    domain: str
    prompt: dict[str, Any]
    hidden_checker: Callable[[Any], bool] | None = None
    baseline_score: float = 0.0
    tags: tuple[str, ...] = ()


@dataclass
class BenchmarkResult:
    result_id: str
    task_id: str
    domain: str
    score: float
    passed: bool
    baseline_score: float
    ablation: str
    receipts: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndependentValidationLoop:
    """Runs hidden checks and ablations without trusting self-reports."""

    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def evaluate(
        self,
        task: BenchmarkTask,
        runner: Callable[[dict[str, Any]], Any],
        *,
        scorer: Callable[[Any], float] | None = None,
        ablation: str = "full_system",
    ) -> BenchmarkResult:
        output = runner(dict(task.prompt))
        passed = bool(task.hidden_checker(output)) if task.hidden_checker else True
        score = float(scorer(output)) if scorer else (1.0 if passed else 0.0)
        result = BenchmarkResult(
            result_id=stable_hash(
                {
                    "task": task.task_id,
                    "domain": task.domain,
                    "score": score,
                    "passed": passed,
                    "ablation": ablation,
                    "ts": round(time.time(), 3),
                },
                prefix="bench_",
            ),
            task_id=task.task_id,
            domain=task.domain,
            score=score,
            passed=passed,
            baseline_score=task.baseline_score,
            ablation=ablation,
            notes="improved_over_baseline" if score > task.baseline_score else "no_improvement",
        )
        self.results.append(result)
        return result

    def compare_ablations(self, task: BenchmarkTask, runners: Mapping[str, Callable[[dict[str, Any]], Any]]) -> dict[str, Any]:
        results = [self.evaluate(task, runner, ablation=name) for name, runner in runners.items()]
        best = max(results, key=lambda r: r.score)
        return {
            "task_id": task.task_id,
            "results": [r.to_dict() for r in results],
            "best_ablation": best.ablation,
            "causal_impact": best.score - min(r.score for r in results),
            "receipt_id": stable_hash([r.to_dict() for r in results], prefix="abl_"),
        }

    def summary(self, *, recent: int = 100) -> dict[str, Any]:
        items = self.results[-recent:]
        if not items:
            return {"count": 0, "pass_rate": 0.0, "avg_score": 0.0}
        return {
            "count": len(items),
            "pass_rate": sum(1 for r in items if r.passed) / len(items),
            "avg_score": sum(r.score for r in items) / len(items),
            "domains": sorted({r.domain for r in items}),
            "receipt_id": stable_hash([r.result_id for r in items], prefix="val_"),
        }
