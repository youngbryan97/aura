"""Bridge: feed UnknownUnknownGenerator output into the F9 curriculum loop.

The F9 ``CurriculumLoop`` accepts ``LearningTask`` objects as
seed_tasks.  This bridge converts the F18 ``Task`` (procedural
benchmark task with ground truth) plus the F21 mutated unknowns into
LearningTasks that drive curriculum iterations.

Why this matters for autonomy: when the research core finds tasks
the model fails on, it doesn't store them as static evals — it feeds
them into the curriculum so Aura learns to handle them.  That's the
"failure-finding tests become regression bank" loop the F21 docs
promised.
"""
from __future__ import annotations

import uuid
from typing import List, Sequence

from core.curriculum.task_generator import LearningTask
from core.promotion.dynamic_benchmark import Task


def task_to_learning_task(
    task: Task, *, strategy: str = "default", iteration: int = 0
) -> LearningTask:
    """Convert a procedural Task → LearningTask for the curriculum loop."""
    return LearningTask(
        task_id=f"learn-{uuid.uuid4().hex[:12]}",
        belief=f"task:{task.kind}",
        modality="symbolic",
        prompt=task.prompt,
        expected={"answer": task.answer},
        strategy=strategy,
        iteration=iteration,
        metadata={
            "source_kind": task.kind,
            "source_metadata": dict(task.metadata),
            "source_hash": task.hash_public(),
        },
    )


def tasks_to_learning_tasks(
    tasks: Sequence[Task], *, strategy: str = "default", iteration: int = 0
) -> List[LearningTask]:
    return [task_to_learning_task(t, strategy=strategy, iteration=iteration) for t in tasks]
