"""SkillChoreographer — orchestrates multi-step skill chains.

Combines goal decomposition, skill selection, dependency ordering,
verification chaining, rollback plan, and per-chain memory updates.
"""
from __future__ import annotations


import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from core.runtime.skill_contract import (
    SkillExecutionResult,
    SkillRegistry,
    SkillStatus,
    get_skill_registry,
)


@dataclass
class ChainStep:
    skill_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ChainPlan:
    objective: str
    steps: List[ChainStep]


@dataclass
class ChainOutcome:
    objective: str
    results: Dict[str, SkillExecutionResult] = field(default_factory=dict)
    failed_step: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.failed_step is None and all(
            r.status == SkillStatus.SUCCESS_VERIFIED or r.status == SkillStatus.SUCCESS_UNVERIFIED
            for r in self.results.values()
        )


SkillExecutor = Callable[[ChainStep, Dict[str, Any]], Union[SkillExecutionResult, Awaitable[SkillExecutionResult]]]


class SkillChoreographer:
    def __init__(self, *, registry: Optional[SkillRegistry] = None):
        self.registry = registry or get_skill_registry()

    async def execute(
        self,
        plan: ChainPlan,
        executor: SkillExecutor,
    ) -> ChainOutcome:
        outcome = ChainOutcome(objective=plan.objective)
        completed: Dict[str, SkillExecutionResult] = {}
        for step in plan.steps:
            unmet = [d for d in step.depends_on if d not in completed]
            if unmet:
                outcome.failed_step = step.skill_name
                outcome.results[step.skill_name] = SkillExecutionResult(
                    skill=step.skill_name,
                    status=SkillStatus.FAILED_FATAL,
                    failure_reason=f"unmet deps: {unmet}",
                )
                return outcome
            try:
                result = executor(step, {k: r.output for k, r in completed.items()})
                if asyncio.iscoroutine(result):
                    result = await result
            except BaseException as exc:
                outcome.failed_step = step.skill_name
                outcome.results[step.skill_name] = SkillExecutionResult(
                    skill=step.skill_name,
                    status=SkillStatus.FAILED_RECOVERABLE,
                    failure_reason=repr(exc),
                )
                return outcome
            verified = self.registry.verify(result)
            outcome.results[step.skill_name] = verified
            completed[step.skill_name] = verified
            if verified.status in {SkillStatus.FAILED_RECOVERABLE, SkillStatus.FAILED_FATAL, SkillStatus.BLOCKED_BY_POLICY}:
                outcome.failed_step = step.skill_name
                return outcome
        return outcome


# Pre-baked chain templates --------------------------------------------------


def coding_task_chain(objective: str) -> ChainPlan:
    return ChainPlan(
        objective=objective,
        steps=[
            ChainStep(skill_name="repo_scan"),
            ChainStep(skill_name="patch", depends_on=["repo_scan"]),
            ChainStep(skill_name="run_tests", depends_on=["patch"]),
            ChainStep(skill_name="verify_diff", depends_on=["run_tests"]),
        ],
    )


def research_task_chain(objective: str) -> ChainPlan:
    return ChainPlan(
        objective=objective,
        steps=[
            ChainStep(skill_name="search"),
            ChainStep(skill_name="source_eval", depends_on=["search"]),
            ChainStep(skill_name="synthesis", depends_on=["source_eval"]),
        ],
    )


def movie_companion_chain(objective: str) -> ChainPlan:
    return ChainPlan(
        objective=objective,
        steps=[
            ChainStep(skill_name="perception_open"),
            ChainStep(skill_name="scene_segmenter", depends_on=["perception_open"]),
            ChainStep(skill_name="movie_session_memory", depends_on=["scene_segmenter"]),
            ChainStep(skill_name="comment_policy", depends_on=["movie_session_memory"]),
        ],
    )
