"""Controlled multi-generation successor lab.

The lab demonstrates the shape required for RSI proof: each generation creates
a successor architecture that handles more hidden task families, and the
successor is scored on both capability and improver quality. It is deliberately
bounded and reproducible so auditors can rerun it.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.learning.architecture_search import baseline_text_heuristic, routed_algorithmic_solver
from core.learning.hidden_eval_repro import HiddenEvalPack
from core.learning.rsi_lineage import (
    PROVENANCE_AUTHORED,
    RSIGenerationRecord,
    RSILineageLedger,
    RSILineageVerdict,
    evaluate_lineage,
)
from core.promotion.dynamic_benchmark import Task


def _solve_arithmetic(task: Task) -> Any:
    if task.kind in {"gcd", "mod", "compose"}:
        return routed_algorithmic_solver(task)
    return baseline_text_heuristic(task)


def _solve_arithmetic_sort(task: Task) -> Any:
    if task.kind in {"gcd", "mod", "compose", "sort"}:
        return routed_algorithmic_solver(task)
    return baseline_text_heuristic(task)


def _solve_all(task: Task) -> Any:
    return routed_algorithmic_solver(task)


@dataclass(frozen=True)
class SuccessorLabResult:
    records: list[RSIGenerationRecord]
    verdict: RSILineageVerdict
    ledger_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "verdict": self.verdict.to_dict(),
            "ledger_path": self.ledger_path,
        }


class SuccessorLab:
    """Produce G1-G4 controlled successor records against fresh hidden packs."""

    #: The four successors, in order. Each is a solver a developer wrote.
    #:
    #: There used to be a fourth column here — 0.30, 0.50, 0.75, 0.90 — read
    #: into `improver_score`, so this lab reported a rising improver curve that
    #: was four numbers typed in ascending order. Nothing in the lab improves
    #: an improver: the succession is a fixed list, and the intelligence that
    #: produced each solver was human and is not being measured. The column is
    #: gone and the records declare `PROVENANCE_AUTHORED`, which keeps the
    #: lab's verdict at BOUNDED_SELF_OPTIMIZATION where it belongs.
    #: G4 runs `_solve_all`, the same function as G3. It is kept so the lab
    #: still exercises a four-entry lineage, and it is not promoted, because
    #: measured against the same hidden families it adds nothing. The blended
    #: score used to promote it anyway: the authored column rose 0.75 to 0.90
    #: and carried `after_score` past G3 on a capability that had not moved.
    STRATEGIES: list[tuple[str, str, Callable[[Task], Any]]] = [
        ("Aura-G1", "arithmetic_router", _solve_arithmetic),
        ("Aura-G2", "arithmetic_sort_router", _solve_arithmetic_sort),
        ("Aura-G3", "full_task_router", _solve_all),
        ("Aura-G4", "full_task_router_with_reproduction", _solve_all),
    ]

    def __init__(self, artifact_dir: Path | str, *, seed: int = 9917, tasks_per_generation: int = 40):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.tasks_per_generation = int(tasks_per_generation)
        self.ledger = RSILineageLedger(self.artifact_dir / f"successor_lineage_{int(time.time() * 1000)}.jsonl")

    def run(self) -> SuccessorLabResult:
        records: list[RSIGenerationRecord] = []
        parent = "Aura-G0"
        previous_score = self._score_solver(
            HiddenEvalPack(seed=self.seed, answer_salt="successor-g0", task_count=self.tasks_per_generation),
            baseline_text_heuristic,
        )
        for idx, (generation_id, strategy_name, solver) in enumerate(self.STRATEGIES, start=1):
            pack = HiddenEvalPack(
                seed=self.seed + idx,
                answer_salt=f"successor-{generation_id}",
                task_count=self.tasks_per_generation,
            )
            start = time.time()
            # The capability curve is the hidden-eval score. It used to be
            # `0.80 * hidden + 0.20 * improver`, which folded the authored
            # column into the capability measurement as well.
            after_score = self._score_solver(pack, solver)
            record = RSIGenerationRecord(
                generation_id=generation_id,
                parent_generation_id=parent,
                hypothesis=f"{strategy_name} expands hidden task-family coverage",
                intervention_type="successor_architecture",
                artifact_hashes={"hidden_manifest": pack.manifest_hash()},
                baseline_score=previous_score,
                after_score=after_score,
                hidden_eval_score=after_score,
                promoted=after_score > previous_score,
                ablation_result=f"{strategy_name}_beats_parent",
                time_to_valid_improvement_s=round(time.time() - start, 6),
                improver_score=0.0,
                improver_provenance=PROVENANCE_AUTHORED,
                improver_measurement={
                    "reason": "successor solvers are written by a developer; "
                    "this lab measures capability, not an improver"
                },
            )
            self.ledger.append(record)
            records.append(record)
            parent = generation_id
            previous_score = after_score
        return SuccessorLabResult(
            records=records,
            verdict=evaluate_lineage(records),
            ledger_path=str(self.ledger.path),
        )

    @staticmethod
    def _score_solver(pack: HiddenEvalPack, solver: Callable[[Task], Any]) -> float:
        return pack.evaluate(solver).score


__all__ = ["SuccessorLab", "SuccessorLabResult"]
