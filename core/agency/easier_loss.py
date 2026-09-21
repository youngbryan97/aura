"""Choosing an easier loss to avoid a harder task.

"Unsweetened Lemonade" has somebody take the loss they could manage over the
task they could not face, and do it again. Her arbiter can do the same thing
without anything noticing. `resource_cost` is one of its eight dimensions and a
cheaper initiative scores higher on it, which is right; but when cost is the
only thing that decided it, a task worth more to her loses to a cheaper one,
stays pending, and loses the same way at the next arbitration, for as long as
it stays harder.

Kept, per pending initiative:

    avoided    how many different cheaper tasks were chosen over it that it
               would have beaten if the two had cost the same: worth more to
               her on expected value, dearer on cost, and ahead with cost set
               equal for both. Different tasks, not arbitrations: four readers
               ask the arbiter the same question each turn, and asking again
               is not choosing again.

`lift` is what the arbiter applies to a task's cost score before it weighs it.
A task avoided `m` times has its cost score moved toward free by the share
`m / (m + 1)` of the room left: halfway after one avoidance, three quarters
after three. Nothing else about it changes, so it wins only if it was losing
on cost alone. Taking it clears the count, and a task that is dropped from
pending takes its count with it.

`easier_losses` counts the occasions, which is the reading.

The loop closes through the arbiter's next choice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = [
    "AvoidanceLedger",
    "get_avoidance_ledger",
    "reset_for_test",
]


def _key(initiative: Any) -> str:
    if not isinstance(initiative, dict):
        return ""
    return str(initiative.get("goal") or initiative.get("description") or "").strip()[:200]


class AvoidanceLedger:
    """Which harder tasks lost to easier ones on cost alone, and how often."""

    def __init__(self) -> None:
        self._avoided: dict[str, set[str]] = {}
        self.easier_losses = 0

    def avoided(self, initiative: Any) -> int:
        return len(self._avoided.get(_key(initiative), ()))

    def lift(self, initiative: Any, cost_score: float) -> float:
        """The cost score after what avoiding this task has taught her."""
        score = max(0.0, min(1.0, float(cost_score)))
        times = self.avoided(initiative)
        if times <= 0:
            return score
        return score + (1.0 - score) * (times / (times + 1.0))

    def note_choice(
        self,
        chosen: Any,
        passed_over: Iterable[Any],
        weigh: Callable[[dict[str, float]], float],
    ) -> int:
        """One arbitration. Returns how many tasks it passed over on cost alone.

        `chosen` and each of `passed_over` carry `.initiative` and `.scores`;
        `weigh` is the arbiter's own composite, asked again with cost set equal.
        """
        chosen_scores = dict(getattr(chosen, "scores", {}) or {})
        chosen_key = _key(getattr(chosen, "initiative", None))
        chosen_even = weigh({**chosen_scores, "resource_cost": 1.0})
        still_pending = {chosen_key}
        count = 0
        for other in passed_over:
            key = _key(getattr(other, "initiative", None))
            if not key:
                continue
            still_pending.add(key)
            scores = dict(getattr(other, "scores", {}) or {})
            worth_more = float(scores.get("expected_value", 0.0)) > float(chosen_scores.get("expected_value", 0.0))
            dearer = float(scores.get("resource_cost", 1.0)) < float(chosen_scores.get("resource_cost", 1.0))
            ahead_even = weigh({**scores, "resource_cost": 1.0}) > chosen_even
            if worth_more and dearer and ahead_even:
                lost_to = self._avoided.setdefault(key, set())
                if chosen_key not in lost_to:
                    lost_to.add(chosen_key)
                    count += 1
        self._avoided.pop(chosen_key, None)
        for key in list(self._avoided):
            if key not in still_pending:
                del self._avoided[key]
        self.easier_losses += count
        return count

    def status(self) -> dict[str, Any]:
        return {
            "easier_losses": self.easier_losses,
            "avoided": {key: len(lost_to) for key, lost_to in sorted(self._avoided.items())},
        }


_LEDGER: AvoidanceLedger | None = None


def get_avoidance_ledger() -> AvoidanceLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = AvoidanceLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
