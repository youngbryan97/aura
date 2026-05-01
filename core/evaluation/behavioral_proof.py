"""Behavioral proof smoke gate.

This is intentionally modest: it does not claim Aura has reached
superhuman-scale research output. It verifies that the evaluation machinery can
score sealed tasks, reject a weak baseline, accept a competent solver through
the statistical promotion gate, and emit an artifact that future live Aura runs
can replace with a real solver lane.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.learning.hidden_eval_repro import HiddenEvalPack, HiddenEvalResult
from core.promotion.dynamic_benchmark import Task
from core.promotion.gate import PromotionDecision, PromotionGate, ScoreEstimate
from core.runtime.atomic_writer import atomic_write_text


Solver = Callable[[Task], Any]


@dataclass(frozen=True)
class SolverScore:
    name: str
    result: HiddenEvalResult
    stderr: float

    @property
    def score(self) -> float:
        return self.result.score

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = self.score
        payload["result"] = self.result.to_dict()
        return payload


@dataclass(frozen=True)
class BehavioralProofReport:
    generated_at: float
    pack_id: str
    manifest_hash: str
    task_count: int
    baseline: SolverScore
    candidate: SolverScore
    promotion: PromotionDecision
    leakage_controls: dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "pack_id": self.pack_id,
            "manifest_hash": self.manifest_hash,
            "task_count": self.task_count,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "promotion": self.promotion.to_dict(),
            "leakage_controls": dict(self.leakage_controls),
            "passed": self.passed,
        }


def solve_from_public_metadata(task: Task) -> Any:
    """Reference solver that uses public task parameters, not sealed answers."""
    data = task.metadata
    if task.kind == "gcd":
        return math.gcd(data["a"], data["b"])
    if task.kind == "mod":
        return pow(data["a"], data["b"], data["m"])
    if task.kind == "sort":
        return sorted(data["arr"])
    if task.kind == "palindrome":
        s = data["s"]
        return s == s[::-1]
    if task.kind == "compose":
        return data["c"] * (data["a"] * data["x"] + data["b"]) + data["d"]
    raise ValueError(f"unsupported task kind: {task.kind}")


def weak_prompt_only_baseline(task: Task) -> Any:
    """A deliberately weak baseline that has the prompt but no task handler."""
    if task.kind == "palindrome":
        return False
    if task.kind == "sort":
        return list(task.metadata.get("arr", []))
    return 0


def run_behavioral_proof_smoke(
    *,
    output_path: str | Path | None = None,
    seed: int = 20260501,
    answer_salt: str = "behavioral-proof-smoke",
    task_count: int = 50,
    baseline_solver: Solver = weak_prompt_only_baseline,
    candidate_solver: Solver = solve_from_public_metadata,
) -> BehavioralProofReport:
    pack = HiddenEvalPack(seed=seed, answer_salt=answer_salt, task_count=task_count)
    manifest = pack.manifest()
    baseline = _score("weak_prompt_only_baseline", pack, baseline_solver)
    candidate = _score("metadata_reference_solver", pack, candidate_solver)

    gate = PromotionGate(
        critical_metrics=["hidden_accuracy"],
        delta=0.20,
        z=1.96,
        emit_receipts=False,
    )
    gate.compare(
        {
            "hidden_accuracy": ScoreEstimate(
                mean=baseline.score,
                stderr=baseline.stderr,
                n=baseline.result.total,
            )
        },
        metadata={"solver": baseline.name, "pack_id": pack.pack_id},
    )
    promotion = gate.compare(
        {
            "hidden_accuracy": ScoreEstimate(
                mean=candidate.score,
                stderr=candidate.stderr,
                n=candidate.result.total,
            )
        },
        metadata={"solver": candidate.name, "pack_id": pack.pack_id},
    )

    public_payload = json.dumps(manifest.public_tasks, sort_keys=True, default=str)
    answer_payload = json.dumps(
        [task.answer for task in pack.tasks],
        sort_keys=True,
        default=str,
    )
    leakage_controls = {
        "answer_hash_ok": candidate.result.answer_hash_ok and baseline.result.answer_hash_ok,
        "public_manifest_excludes_answer_fields": not _contains_key(
            manifest.public_tasks,
            "answer",
        ),
        "answer_key_not_serialized_in_public_tasks": answer_payload not in public_payload,
        "baseline_score_below_candidate": baseline.score < candidate.score,
    }
    passed = bool(
        promotion.accepted
        and candidate.result.score == 1.0
        and baseline.result.score <= 0.60
        and all(leakage_controls.values())
    )
    report = BehavioralProofReport(
        generated_at=time.time(),
        pack_id=pack.pack_id,
        manifest_hash=pack.manifest_hash(),
        task_count=task_count,
        baseline=baseline,
        candidate=candidate,
        promotion=promotion,
        leakage_controls=leakage_controls,
        passed=passed,
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path,
            json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return report


def _score(name: str, pack: HiddenEvalPack, solver: Solver) -> SolverScore:
    result = pack.evaluate(solver)
    stderr = _binomial_stderr(result.score, result.total)
    return SolverScore(name=name, result=result, stderr=stderr)


def _binomial_stderr(p: float, n: int) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(max(0.0, p * (1.0 - p)) / n)


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(
            key == target or _contains_key(item, target)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


__all__ = [
    "BehavioralProofReport",
    "SolverScore",
    "run_behavioral_proof_smoke",
    "solve_from_public_metadata",
    "weak_prompt_only_baseline",
]
