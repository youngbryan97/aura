"""Closed-loop curriculum learning over the prediction ledger.

The pipeline is:

    detect_gap  -> generate_task -> attempt -> measure
        ^                                         |
        +-------- update_strategy <- store_lesson +

``measure`` resolves the prediction in the F2 ledger so the loop's
reward signal is forensically auditable.  ``ImprovementRecorder``
captures Brier loss and accuracy at every iteration so external
observers can see whether the loop is *actually* improving over time.
"""
from core.curriculum.gap_detector import GapDetector, GapReport
from core.curriculum.task_generator import LearningTask, TaskGenerator
from core.curriculum.lesson_store import Lesson, LessonStore
from core.curriculum.strategy import Strategy, StrategyController
from core.curriculum.improvement import (
    ImprovementRecorder,
    ImprovementSnapshot,
)
from core.curriculum.loop import CurriculumLoop, LoopOutcome

__all__ = [
    "GapDetector",
    "GapReport",
    "TaskGenerator",
    "LearningTask",
    "Lesson",
    "LessonStore",
    "Strategy",
    "StrategyController",
    "ImprovementRecorder",
    "ImprovementSnapshot",
    "CurriculumLoop",
    "LoopOutcome",
]
