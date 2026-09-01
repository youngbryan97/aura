"""core/cognition/curriculum_optimiser.py — choosing what to practise next.

Aura has curiosity and self-play and no optimiser over them, so what she
practises is whatever surfaced. That produces two failure modes with opposite
shapes: grinding what is already mastered, because it succeeds and feels
productive, and reaching for what is far out of range, because it is
interesting and fails every time.

The quantity that avoids both is **learning progress**: not how well a task is
done, but how fast that is changing. A task at 95% and a task at 5% both have
near-zero progress; a task at 60% and climbing has the most. Selecting on
progress puts the frontier first without anyone declaring where the frontier is.

Three terms, and the second and third are what stop progress-chasing from
collapsing:

    value = learning_progress + diversity_bonus + information_gain

* **Diversity** rewards a task unlike what has been practised recently. Without
  it, one family with a good progress signal absorbs the whole curriculum.
* **Information gain** rewards a task whose outcome is genuinely uncertain,
  which is what keeps a plateau from looking like a wall.

The envelope
------------
Every candidate passes a governance check before it is scored, not after. A
task that is out of bounds does not compete and then lose; it is not a
candidate. Scoring first and filtering later is how a curriculum learns to want
things it may not have.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Task", "Curriculum", "get_curriculum", "reset_curriculum_for_test"]

#: How many recent attempts define "recently practised" for the diversity term.
RECENT_WINDOW = 20


@dataclass
class Task:
    """One thing she could practise, and how it has been going."""

    task_id: str
    family: str
    scores: list[float] = field(default_factory=list)
    attempts: int = 0
    #: Set by governance. A task outside the envelope never becomes a candidate.
    permitted: bool = True
    refusal_reason: str = ""

    @property
    def mastery(self) -> float:
        return sum(self.scores[-5:]) / len(self.scores[-5:]) if self.scores else 0.0

    @property
    def learning_progress(self) -> float:
        """How fast performance is changing. Mastered and impossible both read zero."""
        if len(self.scores) < 4:
            return 0.0
        half = len(self.scores) // 2
        early = sum(self.scores[:half]) / half
        late = sum(self.scores[half:]) / (len(self.scores) - half)
        return late - early

    @property
    def uncertainty(self) -> float:
        """How unsettled the outcome is. Peaks at a coin flip."""
        p = self.mastery
        return 4.0 * p * (1.0 - p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "attempts": self.attempts,
            "mastery": self.mastery,
            "learning_progress": self.learning_progress,
            "uncertainty": self.uncertainty,
            "permitted": self.permitted,
            "refusal_reason": self.refusal_reason,
        }


class Curriculum:
    """What to practise next, and the evidence that it was worth practising."""

    def __init__(
        self,
        *,
        diversity_weight: float = 0.3,
        information_weight: float = 0.2,
        governance: Callable[[Task], tuple[bool, str]] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, Task] = {}
        self._recent: list[str] = []
        self._diversity = float(diversity_weight)
        self._information = float(information_weight)
        self._governance = governance
        self._refused: list[dict[str, str]] = []

    def offer(self, task_id: str, family: str) -> Task:
        """Add a candidate, running governance before it can be scored."""
        with self._lock:
            task = self._tasks.setdefault(task_id, Task(task_id=task_id, family=family))
            if self._governance is not None:
                permitted, reason = self._governance(task)
                task.permitted = permitted
                task.refusal_reason = "" if permitted else reason
                if not permitted:
                    self._refused.append({"task_id": task_id, "reason": reason})
            return task

    def record(self, task_id: str, score: float) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.scores.append(float(score))
            task.attempts += 1
            self._recent.append(task.family)
            if len(self._recent) > RECENT_WINDOW:
                del self._recent[: len(self._recent) - RECENT_WINDOW]

    def _diversity_bonus_locked(self, task: Task) -> float:
        if not self._recent:
            return 1.0
        share = self._recent.count(task.family) / len(self._recent)
        return 1.0 - share

    def select(self, k: int = 1) -> list[dict[str, Any]]:
        """The k most worth practising, with the terms that made them so."""
        with self._lock:
            candidates = [t for t in self._tasks.values() if t.permitted]
            scored = []
            for task in candidates:
                progress = task.learning_progress
                diversity = self._diversity_bonus_locked(task)
                value = (
                    progress
                    + self._diversity * diversity
                    + self._information * task.uncertainty
                )
                scored.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "value": value,
                        "learning_progress": progress,
                        "diversity_bonus": diversity,
                        "uncertainty": task.uncertainty,
                        "mastery": task.mastery,
                    }
                )
        return sorted(scored, key=lambda row: -row["value"])[:k]

    def report(self) -> dict[str, Any]:
        with self._lock:
            tasks = list(self._tasks.values())
            refused = list(self._refused)
        permitted = [t for t in tasks if t.permitted]
        return {
            "tasks": len(tasks),
            "permitted": len(permitted),
            "refused_by_governance": refused,
            "families": sorted({t.family for t in permitted}),
            "mastered": [t.task_id for t in permitted if t.mastery > 0.9],
            "out_of_reach": [
                t.task_id for t in permitted if t.attempts >= 4 and t.mastery < 0.1
            ],
            "at_the_frontier": [row["task_id"] for row in self.select(5)],
            "recent_family_mix": {
                family: self._recent.count(family) for family in sorted(set(self._recent))
            },
        }


_lock = threading.Lock()
_curriculum: Curriculum | None = None


def get_curriculum() -> Curriculum:
    global _curriculum
    with _lock:
        if _curriculum is None:
            _curriculum = Curriculum()
        return _curriculum


def reset_curriculum_for_test(**kwargs: Any) -> Curriculum:
    global _curriculum
    with _lock:
        _curriculum = Curriculum(**kwargs)
        return _curriculum
