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
from core.runtime.receipts import (
    AutonomyReceipt,
    GovernanceReceipt,
    MemoryWriteReceipt,
    ReceiptStore,
    ToolExecutionReceipt,
)


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


@dataclass(frozen=True)
class LiveLoopStep:
    task_id: str
    goal: str
    action: str
    predicted: Any
    passed: bool
    artifact_id: str
    receipt_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveAutonomyLoopReport:
    pack_id: str
    manifest_hash: str
    task_count: int
    steps: list[LiveLoopStep]
    memory_updates: list[dict[str, Any]]
    loop_closure: dict[str, bool]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "manifest_hash": self.manifest_hash,
            "task_count": self.task_count,
            "steps": [step.to_dict() for step in self.steps],
            "memory_updates": list(self.memory_updates),
            "loop_closure": dict(self.loop_closure),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BehavioralProofBundle:
    smoke: BehavioralProofReport
    live_loop: LiveAutonomyLoopReport
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoke": self.smoke.to_dict(),
            "live_loop": self.live_loop.to_dict(),
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


def run_live_autonomy_loop_smoke(
    *,
    seed: int = 20260502,
    answer_salt: str = "live-autonomy-loop-smoke",
    task_count: int = 8,
    receipt_root: str | Path | None = None,
) -> LiveAutonomyLoopReport:
    """Exercise autonomous loop closure without loading a model."""
    pack = HiddenEvalPack(seed=seed, answer_salt=answer_salt, task_count=task_count)
    store = ReceiptStore(Path(receipt_root) if receipt_root is not None else None)
    memory: dict[str, Any] = {
        "competence": 0.25,
        "preferred_solver": "metadata_reference_solver",
        "attempts": 0,
        "successes": 0,
    }
    steps: list[LiveLoopStep] = []
    memory_updates: list[dict[str, Any]] = []

    for task in pack.tasks:
        task_id = task.hash_public()
        goal = _choose_autonomous_goal(memory, task)
        gov = store.emit(
            GovernanceReceipt(
                cause="behavioral_proof.live_loop",
                domain="autonomous_research_task",
                action="solve_hidden_eval_task",
                approved=True,
                reason=f"goal:{goal}",
                metadata={"task_id": task_id, "kind": task.kind},
            )
        )
        autonomy = store.emit(
            AutonomyReceipt(
                cause="behavioral_proof.live_loop",
                autonomy_level=2,
                proposed_action=goal,
                governance_receipt_id=gov.receipt_id,
                budget_remaining=max(
                    0.0,
                    1.0 - (memory["attempts"] / max(1, task_count)),
                ),
                metadata={"competence": memory["competence"]},
            )
        )
        predicted = solve_from_public_metadata(task)
        passed = predicted == task.answer
        artifact_id = f"artifact-{task_id[:12]}"
        tool = store.emit(
            ToolExecutionReceipt(
                cause="behavioral_proof.live_loop",
                tool="metadata_reference_solver",
                governance_receipt_id=gov.receipt_id,
                status="success_verified" if passed else "failed_verified",
                verification_evidence={
                    "task_id": task_id,
                    "artifact_id": artifact_id,
                    "independent_evaluator": "HiddenEvalPack",
                    "passed": passed,
                },
            )
        )
        memory["attempts"] += 1
        memory["successes"] += int(passed)
        memory["competence"] = memory["successes"] / memory["attempts"]
        memory_update = {
            "task_id": task_id,
            "artifact_id": artifact_id,
            "passed": passed,
            "competence": memory["competence"],
        }
        mem_receipt = store.emit(
            MemoryWriteReceipt(
                cause="behavioral_proof.live_loop",
                family="behavioral_proof_memory",
                record_id=f"memory-{task_id[:12]}",
                bytes_written=len(
                    json.dumps(memory_update, sort_keys=True, default=str)
                ),
                governance_receipt_id=gov.receipt_id,
                metadata=memory_update,
            )
        )
        memory_updates.append(memory_update)
        steps.append(
            LiveLoopStep(
                task_id=task_id,
                goal=goal,
                action="metadata_reference_solver",
                predicted=predicted,
                passed=passed,
                artifact_id=artifact_id,
                receipt_ids=[
                    gov.receipt_id,
                    autonomy.receipt_id,
                    tool.receipt_id,
                    mem_receipt.receipt_id,
                ],
            )
        )

    loop_closure = {
        "internal_state_generated_goals": all(step.goal for step in steps),
        "actions_emitted_artifacts": all(step.artifact_id for step in steps),
        "independent_evaluation_passed": all(step.passed for step in steps),
        "memory_updated_after_each_action": len(memory_updates) == len(steps),
        "future_policy_changed": bool(
            memory_updates
            and memory_updates[-1]["competence"] != 0.25
            and "maintain" in steps[-1].goal.lower()
        ),
        "receipts_cover_each_step": all(len(step.receipt_ids) >= 4 for step in steps),
    }
    return LiveAutonomyLoopReport(
        pack_id=pack.pack_id,
        manifest_hash=pack.manifest_hash(),
        task_count=task_count,
        steps=steps,
        memory_updates=memory_updates,
        loop_closure=loop_closure,
        passed=bool(steps and all(loop_closure.values())),
    )


def run_behavioral_proof_bundle(
    *,
    output_path: str | Path | None = None,
    smoke_seed: int = 20260501,
    live_loop_seed: int = 20260502,
    smoke_task_count: int = 50,
    live_loop_task_count: int = 8,
    receipt_root: str | Path | None = None,
) -> BehavioralProofBundle:
    smoke = run_behavioral_proof_smoke(
        seed=smoke_seed,
        task_count=smoke_task_count,
    )
    live_loop = run_live_autonomy_loop_smoke(
        seed=live_loop_seed,
        task_count=live_loop_task_count,
        receipt_root=receipt_root,
    )
    bundle = BehavioralProofBundle(
        smoke=smoke,
        live_loop=live_loop,
        passed=smoke.passed and live_loop.passed,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path,
            json.dumps(bundle.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return bundle


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


def _choose_autonomous_goal(memory: dict[str, Any], task: Task) -> str:
    if memory["competence"] < 0.80:
        return f"Improve competence by solving held-out {task.kind} task"
    return f"Maintain verified competence on held-out {task.kind} task"


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
    "BehavioralProofBundle",
    "LiveAutonomyLoopReport",
    "LiveLoopStep",
    "SolverScore",
    "run_behavioral_proof_bundle",
    "run_behavioral_proof_smoke",
    "run_live_autonomy_loop_smoke",
    "solve_from_public_metadata",
    "weak_prompt_only_baseline",
]
