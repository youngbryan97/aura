"""Honest ablation harness — does the architecture beat a stateless wrapper?

This replaces a previously FABRICATED benchmark (hardcoded baseline scores +
an assert-victory that never ran the tasks). The keystone honesty test for
Aura's emergence claims must measure real behaviour, so this harness:

  - runs each task under named conditions through an INJECTED responder
    (``responder(condition, task, turn_index, history) -> str``), so the
    orchestration is deterministic and unit-testable offline and the live tool
    plugs in real model/architecture calls;
  - scores outputs with OBJECTIVE graders against an answer key (no model is
    asked to grade itself, no hand-set numbers);
  - reports per-condition means with bootstrap confidence intervals and an
    HONEST verdict: the architecture "beats" a baseline only when its lower CI
    clears the baseline's upper CI on the real scores. The verdict can be
    False — that is a valid, informative result, not a failure of the harness.

The canonical task family is multi-turn recall/continuity, where a stateless
wrapper structurally cannot succeed (it never sees the earlier turn) while a
memory/context architecture can. That makes the architectural variable causal
and the answer objectively checkable.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.evaluation.matched_budget import ConditionBudget, check_budget_parity
from core.evaluation.statistics import bootstrap_ci as _bootstrap_ci

# Condition names. RAW = bare model call, current turn only. PROMPTED = bare
# model + a fixed system/identity prompt, still no history. FULL = the
# architecture's assembled context (conversation history / recalled memory).
RAW = "raw_model"
PROMPTED = "prompted_model"
FULL = "full_architecture"
STATELESS_CONDITIONS = (RAW, PROMPTED)


@dataclass(frozen=True)
class AblationTask:
    """A multi-turn task with an objectively checkable final answer.

    ``turns`` are user messages in order; the responder answers the LAST turn.
    Earlier turns carry the information a stateful architecture can use and a
    stateless wrapper cannot. ``answer_key`` + ``grader`` define correctness.
    """

    task_id: str
    family: str
    turns: Sequence[str]
    answer_key: str
    grader: str = "recall_substring"

    @property
    def final_prompt(self) -> str:
        return self.turns[-1] if self.turns else ""


# ---- graders: output, answer_key -> score in [0,1]. Objective only. ----------

def _grade_recall_substring(output: str, answer_key: str) -> float:
    return 1.0 if answer_key.strip().lower() in (output or "").lower() else 0.0


def _grade_exact(output: str, answer_key: str) -> float:
    return 1.0 if (output or "").strip().lower() == answer_key.strip().lower() else 0.0


def _grade_token_overlap(output: str, answer_key: str) -> float:
    key_tokens = {t for t in re.findall(r"\w+", answer_key.lower()) if t}
    if not key_tokens:
        return 0.0
    out_tokens = {t for t in re.findall(r"\w+", (output or "").lower()) if t}
    return len(key_tokens & out_tokens) / len(key_tokens)


_GRADERS: dict[str, Callable[[str, str], float]] = {
    "recall_substring": _grade_recall_substring,
    "exact": _grade_exact,
    "token_overlap": _grade_token_overlap,
}


def grade(output: str, task: AblationTask) -> float:
    grader = _GRADERS.get(task.grader, _grade_recall_substring)
    return float(grader(output, task.answer_key))


# ---- statistics --------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 1234,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for the mean (reuses the canonical
    core.evaluation.statistics implementation)."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (round(float(values[0]), 4), round(float(values[0]), 4))
    lo, hi = _bootstrap_ci(list(values), n_resamples=max(1, iterations), confidence=0.95, seed=seed)
    return (round(lo, 4), round(hi, 4))


@dataclass
class ConditionResult:
    condition: str
    per_task: dict[str, float] = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return list(self.per_task.values())

    @property
    def mean_score(self) -> float:
        return round(_mean(self.scores), 4)

    def ci(self, *, iterations: int = 2000) -> tuple[float, float]:
        return bootstrap_ci(self.scores, iterations=iterations)

    def as_dict(self, *, iterations: int = 2000) -> dict[str, Any]:
        lo, hi = self.ci(iterations=iterations)
        return {
            "condition": self.condition,
            "mean_score": self.mean_score,
            "lower_ci": lo,
            "upper_ci": hi,
            "n": len(self.scores),
            "per_task": dict(self.per_task),
        }


class AblationHarness:
    """Runs conditions over tasks with an injected responder; scores honestly."""

    def __init__(
        self,
        *,
        conditions: Sequence[str] = (RAW, PROMPTED, FULL),
        bootstrap_iterations: int = 2000,
        budgets: Sequence[ConditionBudget] | None = None,
    ):
        """``budgets`` declares what each arm was allowed to spend.

        Optional, because the unit tests here drive an injected responder where
        the question does not arise. Required in practice for anything that
        becomes evidence: this harness previously had no concept of what an arm
        had been allowed to do, which is how a 160-token baseline was compared
        against an effectively unbounded, solver-assisted treatment and the
        result was published as 100% versus 16.67%.

        When budgets are declared and do not match, `verdict()` returns void
        rather than a number.
        """
        self.conditions = tuple(conditions)
        self.bootstrap_iterations = bootstrap_iterations
        self.budgets = tuple(budgets) if budgets else ()

    def run(
        self,
        responder: Callable[[str, AblationTask, int, list[str]], str],
        tasks: Sequence[AblationTask],
    ) -> dict[str, ConditionResult]:
        results = {c: ConditionResult(condition=c) for c in self.conditions}
        for condition in self.conditions:
            for task in tasks:
                history: list[str] = []
                output = ""
                for turn_index, turn in enumerate(task.turns):
                    output = str(responder(condition, task, turn_index, list(history)))
                    history.append(turn)
                    history.append(output)
                results[condition].per_task[task.task_id] = grade(output, task)
        return results

    def verdict(self, results: dict[str, ConditionResult]) -> dict[str, Any]:
        """Honest verdict: architecture beats a stateless baseline only when its
        lower CI clears that baseline's upper CI on the real per-task scores."""
        # Parity first. A verdict computed from arms that were not allowed
        # the same resources is not a weak result, it is not a result.
        if self.budgets:
            parity = check_budget_parity(self.budgets)
            if not parity.matched:
                return {
                    "architecture_beats_stateless": False,
                    "verdict": "void",
                    "reason": parity.refusal_reason(),
                    "budget_parity": parity.to_dict(),
                    "comparisons": {},
                }

        full = results.get(FULL)
        comparisons: dict[str, Any] = {}
        beats_all_stateless = full is not None and bool(
            [c for c in STATELESS_CONDITIONS if c in results]
        )
        for cond in STATELESS_CONDITIONS:
            base = results.get(cond)
            if base is None or full is None:
                continue
            full_lo, _ = full.ci(iterations=self.bootstrap_iterations)
            _, base_hi = base.ci(iterations=self.bootstrap_iterations)
            ci_separated = full_lo > base_hi
            delta = round(full.mean_score - base.mean_score, 4)
            comparisons[cond] = {
                "delta_mean": delta,
                "full_lower_ci": full_lo,
                "baseline_upper_ci": base_hi,
                "ci_separated": ci_separated,
            }
            if not ci_separated:
                beats_all_stateless = False
        return {
            "architecture_beats_stateless": bool(beats_all_stateless),
            "comparisons": comparisons,
        }

    def report_from_results(
        self, results: dict[str, ConditionResult], *, tasks_evaluated: int | None = None
    ) -> dict[str, Any]:
        """Build the honest report from already-computed per-task scores.

        Lets a live runner drive an async model on a single event loop and feed
        the real scores back through the same verdict/formatting path.
        """
        verdict = self.verdict(results)
        n = (
            tasks_evaluated
            if tasks_evaluated is not None
            else max((len(r.per_task) for r in results.values()), default=0)
        )
        return {
            "methodology": (
                "Real per-task scores graded objectively against answer keys; "
                "per-condition bootstrap CIs; verdict requires CI separation. "
                "No hardcoded scores."
            ),
            "tasks_evaluated": n,
            "conditions": {
                c: results[c].as_dict(iterations=self.bootstrap_iterations)
                for c in self.conditions
                if c in results
            },
            "verdict": verdict,
        }

    def report(
        self,
        responder: Callable[[str, AblationTask, int, list[str]], str],
        tasks: Sequence[AblationTask],
    ) -> dict[str, Any]:
        results = self.run(responder, tasks)
        verdict = self.verdict(results)
        return {
            "methodology": (
                "Real per-task scores from an injected responder, graded "
                "objectively against answer keys; per-condition bootstrap CIs; "
                "verdict requires CI separation. No hardcoded scores."
            ),
            "tasks_evaluated": len(tasks),
            "conditions": {
                c: results[c].as_dict(iterations=self.bootstrap_iterations)
                for c in self.conditions
            },
            "verdict": verdict,
        }
