"""Choosing the class she measures herself against.

"Man of the Year" is partly about which comparison to accept: whose record a
performance is read against decides what it says. Her belief in a capability
had one comparison available. `core/agency/capacity.py` takes its prior from
everything else she has tried, so a new kind of writing was judged against her
record at tool calls, and nothing could say whether that was the right company
for it.

This is the reference class problem, and the standard answer is to let the
classes compete on prediction. Two classes are offered for a capability:

    everything   every other capability she has tried (the incumbent)
    its kind     the other capabilities that share its verb, the part of the
                 name before the first separator: `write_notes` is judged
                 beside `write_plan`, not beside `append_log`

Each of her attempts is first predicted under both, from the counts as they
stood before it, and the outcome is scored by log loss, the proper scoring rule
for a probability. `chosen` is the class with the lower mean loss once both
have scored enough attempts; until then it is the incumbent. Her confidence in
every capability then takes its prior from the chosen class.

The loop: which class she adopts sets her `can:` beliefs, those decide what
she attempts, and each attempt is scored against both classes again.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.agency.capacity import Capacity, capacity_of, confidence_with_capacity

__all__ = [
    "CLASSES",
    "MIN_SCORED",
    "ReferenceChoice",
    "ReferenceLedger",
    "family",
    "get_reference_ledger",
    "reset_for_test",
]

#: The classes that compete, the incumbent first.
CLASSES: tuple[str, ...] = ("everything", "its kind")

#: Attempts each class must have predicted before the losses are compared.
#: Eight: below that one surprising outcome decides which company she keeps.
MIN_SCORED: int = 8

_SEPARATOR = re.compile(r"[_:.\-\s]")


def family(what: str) -> str:
    """The verb of a capability: its name up to the first separator."""
    name = str(what or "").strip().lower()
    return _SEPARATOR.split(name, maxsplit=1)[0] if name else ""


def _capacity(by_capability: Mapping[str, Sequence[int]], what: str, reference: str) -> Capacity:
    if reference == "its kind":
        kin = family(what)
        rows = {name: row for name, row in by_capability.items() if name != what and family(name) == kin}
        return capacity_of(rows)
    return capacity_of(by_capability, excluding=what)


@dataclass
class ReferenceChoice:
    chosen: str = CLASSES[0]
    losses: dict[str, float] | None = None
    scored: dict[str, int] | None = None
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "losses": {k: round(v, 4) for k, v in (self.losses or {}).items()},
            "scored": dict(self.scored or {}),
            "measured": self.measured,
        }


class ReferenceLedger:
    """How well each class has predicted her own attempts."""

    def __init__(self) -> None:
        self._loss: dict[str, float] = dict.fromkeys(CLASSES, 0.0)
        self._scored: dict[str, int] = dict.fromkeys(CLASSES, 0)

    def score(self, by_capability: Mapping[str, Sequence[int]], what: str, succeeded: bool) -> None:
        """Before an attempt is counted: what each class predicted, against what happened."""
        attempts, successes = (list(by_capability.get(what, [0, 0])) + [0, 0])[:2]
        for reference in CLASSES:
            capacity = _capacity(by_capability, what, reference)
            if not capacity.measured:
                continue
            p = confidence_with_capacity(int(attempts), int(successes), capacity)
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            self._loss[reference] += -math.log(p if succeeded else 1.0 - p)
            self._scored[reference] += 1

    def read(self) -> ReferenceChoice:
        means = {
            reference: self._loss[reference] / self._scored[reference]
            for reference in CLASSES
            if self._scored[reference] > 0
        }
        measured = all(self._scored[reference] >= MIN_SCORED for reference in CLASSES)
        chosen = min(means, key=means.get) if measured else CLASSES[0]
        return ReferenceChoice(chosen=chosen, losses=means, scored=dict(self._scored), measured=measured)

    def capacity_for(self, by_capability: Mapping[str, Sequence[int]], what: str) -> Capacity:
        """The prior for this capability, from the class she has chosen.

        A chosen class that holds nothing for this capability, a verb she has
        never used before, falls back to everything she has done.
        """
        chosen = self.read().chosen
        capacity = _capacity(by_capability, what, chosen)
        if not capacity.measured and chosen != CLASSES[0]:
            capacity = _capacity(by_capability, what, CLASSES[0])
        return capacity


_LEDGER: ReferenceLedger | None = None


def get_reference_ledger() -> ReferenceLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ReferenceLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
