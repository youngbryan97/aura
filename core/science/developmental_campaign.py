"""core/science/developmental_campaign.py — did living through the earlier tasks help.

This is the experiment the architecture exists to pass. Two instances, same
cortex, same tools, same compute. GROWN keeps what it acquires between task
blocks; RESET loses it. Run both over many sealed heterogeneous tasks and watch
the gap.

    Delta(n) = P(GROWN, T_n) - P(RESET, T_n)

The claim is not that Delta is positive. A positive Delta on task ten proves
almost nothing: the grown instance may simply be remembering task three's
answer, and on a suite where the tasks resemble each other it will. The claim
is that **Delta grows with experience on tasks that cannot be solved by
remembering**, and that **lesioning the acquired artifacts erases it**.

So this refuses four ways of being wrong, each of which has produced a
convincing-looking developmental result somewhere:

* **Answer leakage.** A task whose answer appeared in an earlier block is
  excluded from the trend, and if too many are, the campaign is void. Growth
  that is recall is recall.
* **Context, not structure.** The grown arm must not be given more context
  than the reset arm. A wider window is a confound wearing development's
  clothes, and it is the single easiest way to fake this result.
* **Compute drift.** Both arms declare budgets, and parity is delegated to
  ``matched_budget``.
* **A slope that is one lucky block.** The trend is fitted across blocks with
  its own bootstrap interval, and a campaign whose interval crosses zero
  reports NO EFFECT rather than a direction.

The lesion arm is what makes it causal
--------------------------------------
A third arm, GROWN_LESIONED, keeps the history and has the acquired artifacts
removed. If its scores fall back to RESET, the advantage ran through the
artifacts. If it stays high, the advantage came from something else — and that
something else is worth finding, because it is not the thing the architecture
claims.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import random
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Arm",
    "TaskResult",
    "Block",
    "CampaignVerdict",
    "DevelopmentalCampaign",
]


class Arm(StrEnum):
    #: Keeps acquired representations, procedures, world models, habits.
    GROWN = "grown"
    #: Loses acquired artifacts between blocks. Same cortex, same tools.
    RESET = "reset"
    #: Grown, with the acquired artifacts lesioned at evaluation time.
    GROWN_LESIONED = "grown_lesioned"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task, one arm, one score."""

    task_id: str
    arm: Arm
    score: float
    #: Tasks whose answers appeared in an earlier block cannot support a
    #: developmental claim, whatever they score.
    answer_seen_before: bool = False
    #: Tokens of context the arm was given. Compared across arms, not summed.
    context_tokens: int = 0
    seed: int = 0


@dataclass
class Block:
    """One block of tasks, run across every arm."""

    index: int
    results: list[TaskResult] = field(default_factory=list)

    def mean(self, arm: Arm, *, clean_only: bool = True) -> float | None:
        scores = [
            r.score for r in self.results
            if r.arm is arm and (not clean_only or not r.answer_seen_before)
        ]
        return sum(scores) / len(scores) if scores else None

    def context_mean(self, arm: Arm) -> float:
        rows = [r.context_tokens for r in self.results if r.arm is arm]
        return sum(rows) / len(rows) if rows else 0.0

    def delta(self) -> float | None:
        grown, reset = self.mean(Arm.GROWN), self.mean(Arm.RESET)
        return None if grown is None or reset is None else grown - reset


@dataclass(frozen=True, slots=True)
class CampaignVerdict:
    """What the campaign is entitled to say."""

    blocks: int
    deltas: tuple[float, ...]
    slope: float
    slope_ci: tuple[float, float]
    contaminated_fraction: float
    context_parity: bool
    lesion_restores_baseline: bool | None
    void_because: tuple[str, ...]

    @property
    def void(self) -> bool:
        return bool(self.void_because)

    @property
    def effect(self) -> str:
        if self.void:
            return "void"
        if self.slope_ci[0] <= 0.0 <= self.slope_ci[1]:
            return "no effect"
        return "compounding" if self.slope > 0 else "degrading"

    @property
    def statement(self) -> str:
        if self.void:
            return "no claim: " + "; ".join(self.void_because)
        if self.effect == "no effect":
            return (
                f"no compounding: the slope interval {self.slope_ci} crosses zero over "
                f"{self.blocks} blocks"
            )
        if self.effect == "degrading":
            return f"the grown instance is getting relatively worse (slope {self.slope:+.4g})"
        if self.lesion_restores_baseline is None:
            return (
                f"the gap grows with experience (slope {self.slope:+.4g}), and no lesion "
                "arm was run, so nothing says the acquired artifacts caused it"
            )
        if not self.lesion_restores_baseline:
            return (
                f"the gap grows with experience (slope {self.slope:+.4g}) and survives "
                "lesioning the acquired artifacts, so something else is causing it"
            )
        return (
            f"the gap grows with experience (slope {self.slope:+.4g}) and lesioning the "
            "acquired artifacts erases it"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "deltas": list(self.deltas),
            "slope": self.slope,
            "slope_ci": list(self.slope_ci),
            "effect": self.effect,
            "contaminated_fraction": self.contaminated_fraction,
            "context_parity": self.context_parity,
            "lesion_restores_baseline": self.lesion_restores_baseline,
            "void": self.void,
            "void_because": list(self.void_because),
            "statement": self.statement,
        }


#: Above this fraction of tasks whose answers were seen before, the suite is
#: measuring recall and the campaign says nothing about development.
MAX_CONTAMINATION = 0.2

#: Context tokens may differ between arms by this fraction before the
#: comparison is a context experiment rather than a developmental one.
CONTEXT_TOLERANCE = 0.1

#: Blocks needed before a slope means anything. Three points fit any line.
MIN_BLOCKS = 5


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mx) ** 2 for x in xs)
    return 0.0 if denominator == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denominator


class DevelopmentalCampaign:
    """Grown against reset, over blocks, with the confounds refused."""

    def __init__(self, *, seed: int = 0) -> None:
        self._lock = checked_lock("core.science.developmental_campaign.DevelopmentalCampaign", reentrant=True)
        self._blocks: dict[int, Block] = {}
        self._rng = random.Random(seed)
        #: answer key -> the first block it appeared in. Contamination is
        #: appearing in an EARLIER block; the three arms running the same task
        #: in the same block is the design, not a leak.
        self._first_seen: dict[str, int] = {}

    def record(
        self,
        block_index: int,
        task_id: str,
        arm: Arm,
        score: float,
        *,
        answer_key: str = "",
        context_tokens: int = 0,
        seed: int = 0,
    ) -> TaskResult:
        """Record one task result, flagging an answer that appeared before."""
        with self._lock:
            first = self._first_seen.get(answer_key) if answer_key else None
            seen = first is not None and first < block_index
            if answer_key and first is None:
                self._first_seen[answer_key] = block_index
            result = TaskResult(
                task_id=task_id, arm=arm, score=float(score),
                answer_seen_before=seen, context_tokens=int(context_tokens), seed=seed,
            )
            self._blocks.setdefault(block_index, Block(index=block_index)).results.append(result)
            return result

    def verdict(self, *, bootstrap: int = 2000) -> CampaignVerdict:
        with self._lock:
            blocks = [self._blocks[i] for i in sorted(self._blocks)]
            all_results = [r for b in blocks for r in b.results]

        void: list[str] = []
        if len(blocks) < MIN_BLOCKS:
            void.append(f"{len(blocks)} blocks; a slope needs at least {MIN_BLOCKS}")

        contaminated = (
            sum(1 for r in all_results if r.answer_seen_before) / len(all_results)
            if all_results else 0.0
        )
        if contaminated > MAX_CONTAMINATION:
            void.append(
                f"{contaminated:.0%} of tasks had answers seen in an earlier block; "
                "growth that is recall is recall"
            )

        grown_context = sum(b.context_mean(Arm.GROWN) for b in blocks)
        reset_context = sum(b.context_mean(Arm.RESET) for b in blocks)
        scale = max(grown_context, reset_context, 1.0)
        context_parity = abs(grown_context - reset_context) / scale <= CONTEXT_TOLERANCE
        if not context_parity:
            void.append(
                "the grown arm was given a different amount of context; a wider window "
                "is a confound wearing development's clothes"
            )

        deltas = [d for d in (b.delta() for b in blocks) if d is not None]
        indices = list(range(len(deltas)))
        slope = _slope([float(i) for i in indices], deltas) if len(deltas) >= 2 else 0.0

        low, high = 0.0, 0.0
        if len(deltas) >= 2:
            samples = []
            for _ in range(bootstrap):
                picked = [self._rng.randrange(len(deltas)) for _ in deltas]
                samples.append(_slope([float(i) for i in picked], [deltas[i] for i in picked]))
            samples.sort()
            low = samples[int(0.025 * len(samples))]
            high = samples[int(0.975 * len(samples)) - 1]

        lesion_restores: bool | None = None
        lesioned = [b.mean(Arm.GROWN_LESIONED) for b in blocks]
        if any(v is not None for v in lesioned):
            pairs = [
                (b.mean(Arm.GROWN_LESIONED), b.mean(Arm.RESET), b.mean(Arm.GROWN))
                for b in blocks
            ]
            usable = [(a, r, g) for a, r, g in pairs if a is not None and r is not None and g is not None]
            if usable:
                lesion_gap = sum(a - r for a, r, _ in usable) / len(usable)
                grown_gap = sum(g - r for _, r, g in usable) / len(usable)
                lesion_restores = abs(lesion_gap) <= abs(grown_gap) * 0.5

        return CampaignVerdict(
            blocks=len(blocks),
            deltas=tuple(deltas),
            slope=slope,
            slope_ci=(low, high),
            contaminated_fraction=contaminated,
            context_parity=context_parity,
            lesion_restores_baseline=lesion_restores,
            void_because=tuple(void),
        )
