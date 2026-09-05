"""The closed curriculum loop.

Wires gap detection -> task generation -> attempt -> measurement (via
F2 prediction ledger) -> lesson store -> strategy update ->
improvement snapshot.  An ``Attempter`` callable is injected so tests
can drop in a deterministic stub LLM; the real wiring uses Aura's
LLM router.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.curriculum.gap_detector import GapDetector, GapReport
from core.curriculum.improvement import ImprovementRecorder, ImprovementSnapshot
from core.curriculum.lesson_store import Lesson, LessonStore
from core.curriculum.strategy import Strategy, StrategyController
from core.curriculum.task_generator import LearningTask, TaskGenerator
from core.runtime.prediction_ledger import PredictionLedger


# Attempter takes (task) and returns (predicted_truth, prior_prob, raw_response).
Attempter = Callable[[LearningTask], Tuple[bool, float, str]]

# Oracle resolves the ground truth for a task.
Oracle = Callable[[LearningTask], bool]


@dataclass
class LoopOutcome:
    iterations: int
    final_strategy: Strategy
    final_snapshot: ImprovementSnapshot
    snapshots: List[ImprovementSnapshot]
    lessons: List[Lesson]
    converged: bool
    trend: str


class CurriculumLoop:
    def __init__(
        self,
        *,
        ledger: PredictionLedger,
        lessons: LessonStore,
        attempter: Attempter,
        oracle: Oracle,
        gap_detector: Optional[GapDetector] = None,
        task_generator: Optional[TaskGenerator] = None,
        strategy: Optional[StrategyController] = None,
        improvement: Optional[ImprovementRecorder] = None,
        seed_tasks: Optional[List[LearningTask]] = None,
    ):
        self.ledger = ledger
        self.lessons = lessons
        self.attempter = attempter
        self.oracle = oracle
        self.gap_detector = gap_detector or GapDetector(ledger)
        self.task_generator = task_generator or TaskGenerator()
        self.strategy = strategy or StrategyController()
        self.improvement = improvement or ImprovementRecorder(ledger)
        self.seed_tasks = list(seed_tasks or [])

    def _attempt_and_resolve(
        self,
        task: LearningTask,
    ) -> Tuple[Lesson, ImprovementSnapshot]:
        # 1. attempt — predicted truth + prior probability
        predicted_truth, prior_prob, raw_response = self.attempter(task)

        # 2. register the prediction
        prediction_id = self.ledger.register(
            belief=task.belief,
            modality=task.modality,
            action=task.strategy,
            expected=task.expected,
            prior_prob=prior_prob,
        )

        # 3. resolve via oracle
        observed_truth = self.oracle(task)
        ledger_record = self.ledger.resolve(
            prediction_id,
            observed={"truth": observed_truth, "raw_response": raw_response},
            observed_truth=observed_truth,
        )

        # 4. derive success & lesson
        success = predicted_truth == observed_truth
        lesson = Lesson(
            lesson_id=f"lesson-{uuid.uuid4().hex[:12]}",
            task_id=task.task_id,
            iteration=task.iteration,
            belief=task.belief,
            modality=task.modality,
            strategy=task.strategy,
            success=success,
            brier=ledger_record.brier,
            summary=(
                f"strategy={task.strategy} predicted={predicted_truth} "
                f"observed={observed_truth} brier={ledger_record.brier:.3f}"
            ),
            metadata={"raw_response": raw_response, "prediction_id": prediction_id},
        )
        self.lessons.append(lesson)

        # 5. update strategy on next iteration's belief
        self.strategy.record_outcome(task.belief, success=success)

        # 6. snapshot improvement
        snapshot = self.improvement.snapshot(task.iteration)
        return lesson, snapshot

    def run(
        self,
        *,
        max_iterations: int = 10,
        success_run_to_converge: int = 3,
    ) -> LoopOutcome:
        """Run until either max_iterations or ``success_run_to_converge``
        consecutive successes on the same belief."""
        consecutive_success = 0
        last_belief: Optional[str] = None
        all_lessons: List[Lesson] = []
        all_snapshots: List[ImprovementSnapshot] = []
        converged = False

        for i in range(max_iterations):
            gap = self.gap_detector.detect()
            seed = None
            if i < len(self.seed_tasks):
                seed = self.seed_tasks[i]
            if seed is not None:
                task = LearningTask(
                    task_id=seed.task_id,
                    belief=seed.belief,
                    modality=seed.modality,
                    prompt=seed.prompt,
                    expected=seed.expected,
                    strategy=self.strategy.current(seed.belief).value,
                    iteration=i,
                    metadata=dict(seed.metadata),
                )
            elif gap.has_gap:
                strategy_for_gap = self.strategy.current(gap.belief or "")
                task = self.task_generator.generate(
                    gap=gap,
                    strategy=strategy_for_gap.value,
                    iteration=i,
                )
            else:
                # No work to do: stop.
                break

            lesson, snapshot = self._attempt_and_resolve(task)
            all_lessons.append(lesson)
            all_snapshots.append(snapshot)

            if last_belief == task.belief and lesson.success:
                consecutive_success += 1
            elif lesson.success:
                consecutive_success = 1
            else:
                consecutive_success = 0
            last_belief = task.belief

            if consecutive_success >= success_run_to_converge:
                converged = True
                break

        final_strategy = (
            self.strategy.current(last_belief or "")
            if last_belief
            else Strategy.DEFAULT
        )
        return LoopOutcome(
            iterations=len(all_lessons),
            final_strategy=final_strategy,
            final_snapshot=(all_snapshots[-1] if all_snapshots else self.improvement.snapshot(0)),
            snapshots=all_snapshots,
            lessons=all_lessons,
            converged=converged,
            trend=self.improvement.trend(),
        )
