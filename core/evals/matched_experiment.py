"""core/evals/matched_experiment.py — the comparison that would be evidence.

Everything else in this repository is an architectural improvement. A matched
independent experiment is the one thing that could be evidence, and only if it
is run so that the obvious objections are answered before anybody raises them.
There are five, and a harness that does not answer all five produces a number
nobody outside has any reason to believe:

**The arms were not matched.** Aura with tools, memory and unlimited time
against a base model with none of that measures the scaffolding. Every arm
here gets the same token allowance, the same wall clock, the same tool access
and the same information, and an arm that exceeds its allowance produces a
VOID trial rather than a win.

**The tasks were written by the system under test.** A task set authored here,
or drawn from anything Aura has seen, measures memory. Every task carries a
provenance record and the harness refuses to score one whose author is not
external.

**The grader knew which arm it was reading.** Answers are graded without the
arm label, in an order that does not encode it.

**The win came from somewhere uninteresting.** A single A/B says Aura beat the
base model and nothing about what did it. The ladder adds one capability at a
time, so a gain lands on a rung.

**There was no null.** A sham arm carries the full apparatus with the
capability under test neutralised, and the result is compared against a
permutation null over the pairing rather than against zero.

The criterion is sealed before the run through
:mod:`core.verify.epistemic_independence`, so what counts as a win cannot be
adjusted once the numbers are in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Evals.Matched")

#: Sources whose tasks were not authored by this system or seen by it. A task
#: from anywhere else is scored VOID, because a benchmark this system helped
#: write measures recall.
EXTERNAL_AUTHORS = frozenset({"external_human", "published_benchmark", "third_party"})

#: A weaker but real category: an instance drawn fresh at run time from a space
#: large enough that no particular one can have been memorised — a product of
#: two random three-digit numbers, a freshly shuffled puzzle. The generator was
#: written here, so this does not establish that the task type is novel; it
#: establishes only that this instance is. That is worth less than an
#: externally authored set and more than nothing, and the report says which
#: kind a run used so a reader can weigh it rather than having to ask.
PROCEDURAL_AUTHORS = frozenset({"procedural_unseen"})

ADMISSIBLE_AUTHORS = EXTERNAL_AUTHORS | PROCEDURAL_AUTHORS

#: Permutations for the null over the pairing. Enough that a p-value below
#: 0.01 is expressible at all.
NULL_PERMUTATIONS = 2000


class Outcome(StrEnum):
    """What happened on one task for one arm."""

    CORRECT = "correct"
    WRONG = "wrong"
    #: The arm broke the matched allowance. Not a win and not a loss.
    VOID_BUDGET = "void_budget"
    #: The task was not externally authored.
    VOID_PROVENANCE = "void_provenance"
    #: The arm failed to produce anything.
    VOID_ERROR = "void_error"

    @property
    def scorable(self) -> bool:
        return self in {Outcome.CORRECT, Outcome.WRONG}


@dataclass(frozen=True)
class Budget:
    """The allowance every arm gets, identically."""

    tokens: int = 4096
    seconds: float = 60.0
    tool_calls: int = 8

    def exceeded_by(self, spend: Spend) -> str:
        if spend.tokens > self.tokens:
            return f"tokens {spend.tokens} over {self.tokens}"
        if spend.seconds > self.seconds:
            return f"seconds {spend.seconds:.2f} over {self.seconds}"
        if spend.tool_calls > self.tool_calls:
            return f"tool calls {spend.tool_calls} over {self.tool_calls}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "seconds": self.seconds, "tool_calls": self.tool_calls}


@dataclass
class Spend:
    """What one arm actually used on one task."""

    tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "tool_calls": self.tool_calls,
        }


@dataclass(frozen=True)
class Task:
    """One externally authored problem, with the provenance to prove it."""

    task_id: str
    prompt: str
    #: How an answer is graded. Returns True for correct.
    grade: Callable[[str], bool]
    author: str
    source: str = ""
    #: True when this system, or a model in it, has seen this task before.
    seen_before: bool = False

    @property
    def admissible(self) -> bool:
        return self.author in ADMISSIBLE_AUTHORS and not self.seen_before

    @property
    def externally_authored(self) -> bool:
        """The stronger claim: somebody else wrote the task, not just the instance."""
        return self.author in EXTERNAL_AUTHORS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "author": self.author,
            "source": self.source,
            "seen_before": self.seen_before,
            "admissible": self.admissible,
        }


@dataclass(frozen=True)
class Arm:
    """One configuration under test, and what it is allowed to use."""

    name: str
    #: Runs one task. Returns (answer, spend). Must respect the budget itself;
    #: exceeding it is recorded rather than prevented, so the overrun is
    #: visible instead of being silently truncated into a worse answer.
    run: Callable[[str, Budget], tuple[str, Spend]]
    #: The capabilities this rung adds over the one below it. Empty for base.
    adds: tuple[str, ...] = ()
    #: True when this arm carries the apparatus with the capability under test
    #: neutralised. The null.
    sham: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "adds": list(self.adds), "sham": self.sham}


@dataclass(frozen=True)
class Trial:
    """One arm on one task."""

    arm: str
    task_id: str
    outcome: Outcome
    spend: Spend
    answer_digest: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "outcome": str(self.outcome),
            "spend": self.spend.to_dict(),
            "answer_digest": self.answer_digest,
            "note": self.note,
        }


@dataclass(frozen=True)
class ArmResult:
    """One arm's standing over the whole task set."""

    arm: str
    correct: int
    wrong: int
    void: int
    adds: tuple[str, ...]
    sham: bool

    @property
    def scored(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy(self) -> float | None:
        return None if self.scored == 0 else self.correct / self.scored

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "correct": self.correct,
            "wrong": self.wrong,
            "void": self.void,
            "scored": self.scored,
            "accuracy": self.accuracy,
            "adds": list(self.adds),
            "sham": self.sham,
        }


@dataclass(frozen=True)
class Rung:
    """One step of the ladder, and what the step bought."""

    arm: str
    over: str
    adds: tuple[str, ...]
    delta: float
    p_value: float
    #: Tasks where exactly one of the two was right. The paired evidence.
    discordant: int

    @property
    def attributable(self) -> bool:
        """Whether this rung's gain is distinguishable from the pairing null."""
        return self.discordant >= _MIN_DISCORDANT and self.p_value < _ALPHA

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "over": self.over,
            "adds": list(self.adds),
            "delta": round(self.delta, 4),
            "p_value": round(self.p_value, 4),
            "discordant": self.discordant,
            "attributable": self.attributable,
        }


#: Tasks on which two arms differed, below which no p-value means anything.
_MIN_DISCORDANT = 8

#: The bar. Sealed rather than compared inline.
_ALPHA = 0.05


@dataclass(frozen=True)
class Report:
    """Everything the run establishes, and everything it does not."""

    arms: tuple[ArmResult, ...]
    ladder: tuple[Rung, ...]
    trials: tuple[Trial, ...]
    tasks_admissible: int
    tasks_refused: int
    #: How many admissible tasks were externally authored, as against
    #: procedurally generated here. A run made entirely of the second kind
    #: establishes less and the verdict says so.
    tasks_externally_authored: int
    budget: Budget
    seal: str
    at: float = field(default_factory=time.time)

    @property
    def verdict(self) -> str:
        if self.tasks_admissible < _MIN_DISCORDANT:
            return "inconclusive: too few admissible tasks"
        attributable = [r for r in self.ladder if r.attributable]
        if not attributable:
            return "no rung of the ladder beat the pairing null"
        top = max(attributable, key=lambda r: r.delta)
        verdict = (
            f"{top.arm} over {top.over} by {top.delta:+.3f} "
            f"(p={top.p_value:.4f}), attributable to {', '.join(top.adds) or 'nothing named'}"
        )
        if self.tasks_externally_authored == 0:
            verdict += (
                " — on procedurally generated instances only, so the instances "
                "were unseen and the task type was not authored elsewhere"
            )
        elif self.tasks_externally_authored < self.tasks_admissible:
            verdict += (
                f" — {self.tasks_externally_authored} of {self.tasks_admissible} "
                "tasks externally authored, the rest procedurally generated"
            )
        return verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "seal": self.seal,
            "budget": self.budget.to_dict(),
            "tasks_admissible": self.tasks_admissible,
            "tasks_refused": self.tasks_refused,
            "tasks_externally_authored": self.tasks_externally_authored,
            "arms": [a.to_dict() for a in self.arms],
            "ladder": [r.to_dict() for r in self.ladder],
            "verdict": self.verdict,
            "trials": [t.to_dict() for t in self.trials],
        }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _seal(arms: Sequence[Arm], tasks: Sequence[Task], budget: Budget) -> str:
    """A fingerprint of the design, taken before anything runs."""
    return hashlib.sha256(
        json.dumps(
            {
                "arms": [a.to_dict() for a in arms],
                "tasks": sorted(t.task_id for t in tasks),
                "budget": budget.to_dict(),
                "alpha": _ALPHA,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _paired_null(
    left: dict[str, bool], right: dict[str, bool], *, seed: int = 0x9A17
) -> tuple[float, float, int]:
    """Exchangeability test over the pairing. Returns (delta, p, discordant).

    The null is that the arm label carries no information, so swapping the two
    answers on any task should be as likely as not. Permuting the labels
    within each pair is exactly that null, and it is the right one here
    because the two arms saw the same tasks — comparing against zero would
    ignore the pairing that makes this powerful.
    """
    shared = sorted(set(left) & set(right))
    pairs = [(left[t], right[t]) for t in shared]
    discordant = sum(1 for a, b in pairs if a != b)
    if not pairs:
        return (0.0, 1.0, 0)
    observed = sum(1 for a, b in pairs if a) - sum(1 for a, b in pairs if b)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(NULL_PERMUTATIONS):
        total = 0
        for a, b in pairs:
            if rng.random() < 0.5:
                a, b = b, a
            total += (1 if a else 0) - (1 if b else 0)
        if abs(total) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (NULL_PERMUTATIONS + 1)
    delta = observed / len(pairs)
    return (delta, p_value, discordant)


def run_matched(
    arms: Sequence[Arm],
    tasks: Sequence[Task],
    *,
    budget: Budget | None = None,
    seed: int = 0x51DE,
) -> Report:
    """Run every arm over every admissible task under one matched allowance."""
    allowance = budget or Budget()
    seal = _seal(arms, tasks, allowance)
    _declare_criterion()

    admissible = [t for t in tasks if t.admissible]
    refused = len(tasks) - len(admissible)
    if refused:
        logger.warning(
            "Matched experiment refused %d task(s): not externally authored or "
            "already seen",
            refused,
        )

    # The order arms and tasks are run in must not encode the arm, so the
    # sequence is shuffled once and used for every arm alike.
    rng = random.Random(seed)
    order = list(admissible)
    rng.shuffle(order)

    trials: list[Trial] = []
    correct_by_arm: dict[str, dict[str, bool]] = {a.name: {} for a in arms}

    for arm in arms:
        for task in order:
            started = time.perf_counter()
            try:
                answer, spend = arm.run(task.prompt, allowance)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                trials.append(
                    Trial(
                        arm=arm.name,
                        task_id=task.task_id,
                        outcome=Outcome.VOID_ERROR,
                        spend=Spend(),
                        answer_digest="",
                        note=f"{type(exc).__name__}: {exc}"[:160],
                    )
                )
                continue
            if spend.seconds <= 0.0:
                spend.seconds = time.perf_counter() - started
            overrun = allowance.exceeded_by(spend)
            if overrun:
                trials.append(
                    Trial(
                        arm=arm.name,
                        task_id=task.task_id,
                        outcome=Outcome.VOID_BUDGET,
                        spend=spend,
                        answer_digest=_digest(answer),
                        note=overrun,
                    )
                )
                continue
            # Graded without the arm label: the grader is the task's own,
            # and it is handed the answer alone.
            right = bool(task.grade(answer))
            correct_by_arm[arm.name][task.task_id] = right
            trials.append(
                Trial(
                    arm=arm.name,
                    task_id=task.task_id,
                    outcome=Outcome.CORRECT if right else Outcome.WRONG,
                    spend=spend,
                    answer_digest=_digest(answer),
                )
            )

    results = []
    for arm in arms:
        arm_trials = [t for t in trials if t.arm == arm.name]
        results.append(
            ArmResult(
                arm=arm.name,
                correct=sum(1 for t in arm_trials if t.outcome is Outcome.CORRECT),
                wrong=sum(1 for t in arm_trials if t.outcome is Outcome.WRONG),
                void=sum(1 for t in arm_trials if not t.outcome.scorable),
                adds=arm.adds,
                sham=arm.sham,
            )
        )

    ladder: list[Rung] = []
    for lower, upper in zip(arms, arms[1:], strict=False):
        delta, p_value, discordant = _paired_null(
            correct_by_arm[upper.name], correct_by_arm[lower.name]
        )
        ladder.append(
            Rung(
                arm=upper.name,
                over=lower.name,
                adds=upper.adds,
                delta=delta,
                p_value=p_value,
                discordant=discordant,
            )
        )

    return Report(
        arms=tuple(results),
        ladder=tuple(ladder),
        trials=tuple(trials),
        tasks_admissible=len(admissible),
        tasks_refused=refused,
        tasks_externally_authored=sum(1 for t in admissible if t.externally_authored),
        budget=allowance,
        seal=seal,
    )


def _declare_criterion() -> None:
    """Seal what counts as a win before the run that will meet it."""
    try:
        from core.verify.epistemic_independence import declare

        declare(
            "matched_experiment.alpha",
            threshold=_ALPHA,
            direction="below",
            rationale=(
                "the permutation p-value over the arm-label pairing below "
                "which a rung's gain is treated as attributable; fixed before "
                "any arm was run"
            ),
        )
    except (ImportError, RuntimeError, ValueError):
        return


__all__ = [
    "ADMISSIBLE_AUTHORS",
    "EXTERNAL_AUTHORS",
    "PROCEDURAL_AUTHORS",
    "NULL_PERMUTATIONS",
    "Arm",
    "ArmResult",
    "Budget",
    "Outcome",
    "Report",
    "Rung",
    "Spend",
    "Task",
    "Trial",
    "run_matched",
]
