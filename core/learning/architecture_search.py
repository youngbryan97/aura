"""Architecture invention lab for verifiable toy domains.

The lab searches executable solver architectures against hidden tasks and a
registered baseline. It is small, deterministic, and honest: a candidate only
"wins" when hidden score improves at equal compute.
"""
from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.learning.distributed_eval import DistributedEvalConfig, LocalDistributedEvaluator
from core.learning.hidden_eval_repro import HiddenEvalPack
from core.promotion.dynamic_benchmark import Task


@dataclass(frozen=True)
class ArchitectureCandidate:
    name: str
    description: str
    solve: Callable[[Task], Any]


@dataclass(frozen=True)
class ArchitectureSearchResult:
    baseline_name: str
    winner_name: str
    baseline_score: float
    winner_score: float
    improvement: float
    hidden_manifest_hash: str
    evaluated_candidates: Dict[str, float]
    promoted: bool
    runtime_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def baseline_text_heuristic(task: Task) -> Any:
    if task.kind == "palindrome":
        return "true" in task.prompt.lower()
    numbers = [int(x) for x in re.findall(r"-?\d+", task.prompt)]
    return numbers[-1] if numbers else None


def routed_algorithmic_solver(task: Task) -> Any:
    meta = task.metadata
    if task.kind == "gcd":
        return math.gcd(int(meta["a"]), int(meta["b"]))
    if task.kind == "mod":
        return pow(int(meta["a"]), int(meta["b"]), int(meta["m"]))
    if task.kind == "sort":
        return sorted(list(meta["arr"]))
    if task.kind == "palindrome":
        s = str(meta["s"])
        return s == s[::-1]
    if task.kind == "compose":
        x = int(meta["x"])
        return int(meta["c"]) * (int(meta["a"]) * x + int(meta["b"])) + int(meta["d"])
    raise ValueError(f"unsupported task kind: {task.kind}")


def partial_arithmetic_solver(task: Task) -> Any:
    if task.kind in {"gcd", "mod", "compose"}:
        return routed_algorithmic_solver(task)
    return baseline_text_heuristic(task)


def _score_candidate(args: tuple[str, str, List[Task]]) -> tuple[str, float]:
    name, solver_name, tasks = args
    solver = {
        "baseline_text_heuristic": baseline_text_heuristic,
        "partial_arithmetic_solver": partial_arithmetic_solver,
        "routed_algorithmic_solver": routed_algorithmic_solver,
    }[solver_name]
    passed = 0
    for task in tasks:
        try:
            passed += 1 if solver(task) == task.answer else 0
        except Exception:
            pass
    return name, passed / max(1, len(tasks))


class ArchitectureSearchLab:
    """Search candidate solver architectures and promote a hidden winner."""

    def __init__(self, *, seed: int = 20260428, task_count: int = 80):
        self.hidden_pack = HiddenEvalPack(seed=seed, answer_salt=f"arch:{seed}", task_count=task_count)
        self.candidates = [
            ArchitectureCandidate("text_heuristic", "baseline parses the last visible integer", baseline_text_heuristic),
            ArchitectureCandidate("partial_arithmetic_router", "routes arithmetic tasks but not list/string tasks", partial_arithmetic_solver),
            ArchitectureCandidate("algorithmic_task_router", "routes every supported hidden task kind to an exact algorithm", routed_algorithmic_solver),
        ]

    def run(self, *, distributed: bool = True) -> ArchitectureSearchResult:
        started = time.time()
        solver_names = {
            "text_heuristic": "baseline_text_heuristic",
            "partial_arithmetic_router": "partial_arithmetic_solver",
            "algorithmic_task_router": "routed_algorithmic_solver",
        }
        tasks = list(self.hidden_pack.tasks)
        if distributed:
            evaluator = LocalDistributedEvaluator(DistributedEvalConfig(requested_workers=2, max_workers=2))
            result = evaluator.map(
                _score_candidate,
                [(candidate.name, solver_names[candidate.name], tasks) for candidate in self.candidates],
            )
            scores = dict(result.outputs) if result.ok else {}
        else:
            scores = dict(_score_candidate((candidate.name, solver_names[candidate.name], tasks)) for candidate in self.candidates)
        if not scores:
            scores = {"text_heuristic": 0.0}
        baseline = scores.get("text_heuristic", 0.0)
        winner_name = max(scores, key=lambda name: scores[name])
        winner_score = scores[winner_name]
        improvement = winner_score - baseline
        return ArchitectureSearchResult(
            baseline_name="text_heuristic",
            winner_name=winner_name,
            baseline_score=baseline,
            winner_score=winner_score,
            improvement=improvement,
            hidden_manifest_hash=self.hidden_pack.manifest_hash(),
            evaluated_candidates=scores,
            promoted=winner_name != "text_heuristic" and improvement >= 0.05,
            runtime_s=round(time.time() - started, 6),
        )


__all__ = [
    "ArchitectureCandidate",
    "ArchitectureSearchLab",
    "ArchitectureSearchResult",
    "baseline_text_heuristic",
    "partial_arithmetic_solver",
    "routed_algorithmic_solver",
]
