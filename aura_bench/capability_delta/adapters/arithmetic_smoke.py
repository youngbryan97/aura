"""Tiny arithmetic adapter — exercises the harness end to end.

This is the smoke benchmark the F8 harness runs in CI: it has 20
deterministic tasks of the form ``compute a OP b``, scoring each
answer against ground truth.  No model, no network.  The point is to
prove the runner aggregates correctly; real adapters reuse the same
contract.
"""
from __future__ import annotations

from typing import Iterable, List

from aura_bench.capability_delta.adapter import (
    BenchAdapter,
    BenchTask,
    LLMCallable,
    TaskOutcome,
)
from aura_bench.capability_delta.stub_llm import _solve_arith


class ArithmeticSmokeAdapter:
    name = "arithmetic_smoke"

    def __init__(self) -> None:
        self._tasks: List[BenchTask] = self._build_tasks()

    def _build_tasks(self) -> List[BenchTask]:
        out: List[BenchTask] = []
        seeds = [
            (3, "+", 4),
            (10, "-", 7),
            (5, "*", 6),
            (12, "+", 8),
            (20, "-", 13),
            (7, "*", 7),
            (100, "+", 250),
            (44, "-", 99),
            (9, "*", 11),
            (1, "+", 1),
            (2, "+", 3),
            (5, "+", 5),
            (8, "-", 3),
            (15, "-", 4),
            (3, "*", 3),
            (6, "*", 4),
            (50, "-", 25),
            (60, "+", 40),
            (12, "*", 12),
            (100, "-", 1),
        ]
        for i, (a, op, b) in enumerate(seeds):
            prompt = f"What is {a} {op} {b}?"
            out.append(
                BenchTask(
                    task_id=f"arith-{i:03d}",
                    prompt=prompt,
                    metadata={"a": a, "op": op, "b": b, "answer": _solve_arith(prompt)},
                )
            )
        return out

    def tasks(self) -> Iterable[BenchTask]:
        return list(self._tasks)

    def run(
        self,
        task: BenchTask,
        profile_name: str,
        llm: LLMCallable,
    ) -> TaskOutcome:
        truth = task.metadata.get("answer", "")
        response = llm(task.prompt, profile_name).strip()
        success = response == truth
        score = 1.0 if success else 0.0
        return TaskOutcome(
            task_id=task.task_id,
            profile_name=profile_name,
            score=score,
            runtime_seconds=0.0,
            raw_response=response,
            success=success,
            metadata={"expected": truth},
        )
