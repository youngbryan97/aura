"""General staged environment curriculum."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "environment_family": self.environment_family,
            "difficulty": self.difficulty,
            "objective": self.objective,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "success_metrics": dict(self.success_metrics),
            "failure_metrics": dict(self.failure_metrics),
            "seed": self.seed,
        }


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

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.mastery: dict[str, float] = {}

    def next_stage_allowed(self, completed_stage: int, requested_stage: int) -> bool:
        return requested_stage <= completed_stage + 1

    def record_result(
        self,
        task: CurriculumTask | None = None,
        *,
        environment_family: str | None = None,
        objective: str | None = None,
        outcome_score: float | None = None,
        bottleneck: str = "",
        success: bool | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        metrics = dict(metrics or {})
        if task is None:
            family = environment_family or "general"
            objective_value = objective or "unknown_objective"
            stage = self.completed_stage(family)
            task = CurriculumTask(
                task_id=f"runtime:{family}:{stage}:{objective_value}",
                environment_family=family,
                difficulty=stage,
                objective=objective_value,
                allowed_capabilities=self._capabilities_for_stage(stage),
                success_metrics={"success_rate": 0.8},
                failure_metrics={"loop_rate": 0.2},
            )
        if outcome_score is not None:
            metrics.setdefault("outcome_score", float(outcome_score))
            if success is None:
                success = float(outcome_score) >= 0.6
        if bottleneck:
            metrics.setdefault("bottleneck", bottleneck)
        if success is None:
            success = False
        key = f"{task.environment_family}:{task.difficulty}:{task.objective}"
        old = self.mastery.get(key, 0.0)
        target = 1.0 if success else 0.0
        self.mastery[key] = round((old * 0.75) + (target * 0.25), 4)
        self.history.append({
            "task_id": task.task_id,
            "environment_family": task.environment_family,
            "difficulty": task.difficulty,
            "objective": task.objective,
            "success": bool(success),
            "metrics": metrics,
        })
        self.history = self.history[-1000:]

    def propose_next_task(self, *, environment_family: str, bottlenecks: dict[str, float] | None = None) -> CurriculumTask:
        bottlenecks = dict(bottlenecks or {})
        if bottlenecks:
            objective = max(bottlenecks.items(), key=lambda item: item[1])[0]
        elif self.history:
            failed = [item for item in self.history if not item["success"] and item["environment_family"] == environment_family]
            objective = failed[-1]["objective"] if failed else "transfer_general_strategy"
        else:
            objective = "observe_and_build_belief"
        completed = self.completed_stage(environment_family)
        difficulty = min(len(self.STAGES) - 1, completed + 1)
        return CurriculumTask(
            task_id=f"curriculum:{environment_family}:{difficulty}:{objective}",
            environment_family=environment_family,
            difficulty=difficulty,
            objective=objective,
            allowed_capabilities=self._capabilities_for_stage(difficulty),
            success_metrics={"success_rate": 0.8, "recovery_rate": 0.6},
            failure_metrics={"loop_rate": 0.2, "irreversible_failure_rate": 0.0},
        )

    def completed_stage(self, environment_family: str) -> int:
        best = 0
        for key, mastery in self.mastery.items():
            family, difficulty, _ = key.split(":", 2)
            if family == environment_family and mastery >= 0.75:
                best = max(best, int(difficulty))
        return best

    @staticmethod
    def _capabilities_for_stage(stage: int) -> set[str]:
        base = {"observe", "trace", "belief_update"}
        staged = [
            {"wait"},
            {"move_reversible"},
            {"modal_resolve"},
            {"resource_stabilize"},
            {"object_use_reversible"},
            {"long_horizon_goal"},
            {"crisis_recovery"},
            {"hidden_state_probe"},
            {"strict_benchmark"},
        ]
        allowed = set(base)
        for idx in range(min(stage, len(staged))):
            allowed.update(staged[idx])
        return allowed


__all__ = ["CurriculumTask", "CurriculumEngine"]
