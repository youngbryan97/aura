"""General staged environment curriculum."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurriculumTask:
    task_id: str
    environment_family: str
    difficulty: int
    objective: str
    allowed_capabilities: set[str]
    success_metrics: dict[str, float]
    failure_metrics: dict[str, float]
    seed: int | None = None


class CurriculumEngine:
    STAGES = [
        "observe-only",
        "reversible no-op",
        "simple navigation",
        "modal recovery",
        "resource stabilization",
        "object/affordance use",
        "long-horizon goal",
        "crisis recovery",
        "hidden-state reasoning",
        "full benchmark",
    ]

    def next_stage_allowed(self, completed_stage: int, requested_stage: int) -> bool:
        return requested_stage <= completed_stage + 1


__all__ = ["CurriculumTask", "CurriculumEngine"]
