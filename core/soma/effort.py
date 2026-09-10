"""What she is spending on thinking, as distinct from what the machine is doing.

Interoception, until now, was proprioception of the host: cpu, memory,
temperature, all read from psutil. Those are facts about the computer she runs
on, and most of the load on them is not hers. Nothing anywhere reported how
hard *she* was working — how wide a recall she asked for, how many integration
steps the substrate took, how much text she produced, how many gradient steps
the world model spent on what she showed it.

That distinction matters twice over. A body sense that cannot separate its own
exertion from the room's temperature cannot notice that thinking harder cost
something, so nothing she chooses can ever come back to her as a felt cost. And
an experiment that holds the host still to keep two arms comparable holds still
the only body channel she has, which leaves deliberation with no route back
into the body at all.

Effort is reported by the subsystems that spend it, in their own units, and
drained once per cognitive cycle by the organ that senses the body. Nothing
here decides anything; it is a ledger.
"""

from __future__ import annotations

import threading

__all__ = ["EffortLedger", "get_effort_ledger", "note_effort", "reset_effort_for_test"]

#: What each kind of exertion costs, in units of one cycle's ordinary work, so
#: that quantities in different units can be added into one felt total. Each is
#: the amount of that kind an unremarkable turn produces, taken from the
#: recorded runs rather than chosen: a turn retrieves a handful of memories,
#: the substrate integrates a few dozen steps, a reply is a few hundred
#: characters, and training is a step or two behind.
UNIT_COST: dict[str, float] = {
    "recall": 8.0,
    "substrate_steps": 40.0,
    "response_chars": 400.0,
    "train_steps": 2.0,
    "phases": 30.0,
    "tool_calls": 1.0,
    # How many things were competing for the workspace. Weighing eleven
    # candidates is more work than weighing two, and it is work she causes:
    # the count is what the rest of her had to say this cycle.
    "candidates": 8.0,
}


class EffortLedger:
    """Exertion reported since the last time the body looked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}
        self._lifetime: dict[str, float] = {}

    def note(self, kind: str, amount: float = 1.0) -> None:
        try:
            value = float(amount)
        except (TypeError, ValueError):
            return
        if value <= 0.0 or value != value:
            return
        with self._lock:
            self._pending[kind] = self._pending.get(kind, 0.0) + value
            self._lifetime[kind] = self._lifetime.get(kind, 0.0) + value

    def drain(self) -> dict[str, float]:
        """Everything reported since the last drain, and clear it."""
        with self._lock:
            out = dict(self._pending)
            self._pending.clear()
        return out

    def peek(self) -> dict[str, float]:
        with self._lock:
            return dict(self._pending)

    def lifetime(self) -> dict[str, float]:
        with self._lock:
            return dict(self._lifetime)

    @staticmethod
    def exertion(spent: dict[str, float]) -> float:
        """One number for how hard the last cycle was, bounded at one.

        A sum of ratios rather than of raw counts: a hundred characters and a
        hundred integration steps are not the same amount of work, and adding
        them would make the total a fact about which subsystem happens to count
        in smaller units.
        """
        total = 0.0
        for kind, amount in spent.items():
            unit = UNIT_COST.get(kind)
            if not unit:
                continue
            total += max(0.0, float(amount)) / unit
        return max(0.0, min(1.0, total / max(1, len(UNIT_COST))))


_ledger: EffortLedger | None = None
_ledger_lock = threading.Lock()


def get_effort_ledger() -> EffortLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = EffortLedger()
    return _ledger


def note_effort(kind: str, amount: float = 1.0) -> None:
    """Report exertion. Never raises into the path that was doing the work."""
    try:
        get_effort_ledger().note(kind, amount)
    except (RuntimeError, TypeError, ValueError):
        return


def reset_effort_for_test() -> None:
    global _ledger
    with _ledger_lock:
        _ledger = None
