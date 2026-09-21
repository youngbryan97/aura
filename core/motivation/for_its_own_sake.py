"""A motive that survives the reward arriving.

"Man of the Year" is a record about still wanting to do the thing after it has
paid. Her drives could not tell that apart from anything else. A drive is a
budget that drains at a fixed rate, and an intention formed from it was
finished the moment the budget was back above the line it was judged against:
the result arrived, the need was met, the intention was retired. Whether the
doing had been worth anything to her while it lasted made no difference, so
nothing she did was ever done for its own sake.

Two rewards are kept apart, per drive:

    doing      how much more engaged she was while an intention of that drive
               was open than while none was. The reward of the act.

    result     how far the budget came back between the intention opening and
               the need being met, as a share of its capacity. The reward of
               the outcome.

`own_sake` is the doing's share of the two. Above one half the act has been
worth more to her than what it returned, which is where the two are equal and
the only line the comparison itself supplies.

`survives` is what the motivation phase asks when a need is met. An intention
whose drive she does for its own sake is kept open past the reward, for as long
as the doing still pays: the most recent turn she spent on it has to have
engaged her at least as much as her turns spent on nothing. When it stops
paying it closes like any other, so a motive that outlives its reward still
ends when the act stops being worth doing.

The loop closes through the doing itself: a kept intention is more turns of
that act, each of which is measured again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_TURNS",
    "DoingLedger",
    "OwnSake",
    "get_doing_ledger",
    "reset_for_test",
]

#: Turns of doing, and turns of not doing, before a drive's share is read.
#: Four of each: fewer and one unusually engaged turn decides it.
MIN_TURNS: int = 4

#: Turns held per side, so what the act is worth now is read from recent turns.
WINDOW: int = 60


@dataclass
class _Drive:
    doing: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    results: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    opened_at: float | None = None
    last_doing: float | None = None


@dataclass
class OwnSake:
    """What a drive's acts have been worth to her, split by where the worth came from."""

    drive: str
    doing: float = 0.0
    result: float = 0.0
    own_sake: float = 0.0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "drive": self.drive,
            "doing": round(self.doing, 4),
            "result": round(self.result, 4),
            "own_sake": round(self.own_sake, 4),
            "measured": self.measured,
        }


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


class DoingLedger:
    """Engagement while doing and while not, and what each need's meeting returned."""

    def __init__(self) -> None:
        self._drives: dict[str, _Drive] = {}
        self._idle: deque[float] = deque(maxlen=WINDOW)

    def _drive(self, name: str) -> _Drive:
        held = self._drives.get(name)
        if held is None:
            held = _Drive()
            self._drives[name] = held
        return held

    def note_turn(self, doing: str, engagement: float, *, level: float | None = None) -> None:
        """One turn: which drive's intention was open, if any, and how engaged she was."""
        try:
            value = max(0.0, min(1.0, float(engagement)))
        except (TypeError, ValueError):
            return
        name = str(doing or "").strip()
        if not name:
            self._idle.append(value)
            return
        held = self._drive(name)
        held.doing.append(value)
        held.last_doing = value
        if held.opened_at is None and level is not None:
            try:
                held.opened_at = float(level)
            except (TypeError, ValueError):
                held.opened_at = None

    def note_met(self, drive: str, level: float, capacity: float) -> None:
        """The need behind an intention was met: what the result returned."""
        held = self._drive(str(drive or "").strip())
        if held.opened_at is None:
            return
        try:
            room = max(float(capacity), 1e-9)
            returned = max(0.0, (float(level) - held.opened_at) / room)
        except (TypeError, ValueError):
            return
        held.results.append(min(1.0, returned))
        held.opened_at = None

    def read(self, drive: str) -> OwnSake:
        name = str(drive or "").strip()
        held = self._drives.get(name)
        if held is None or len(held.doing) < MIN_TURNS or len(self._idle) < MIN_TURNS or not held.results:
            return OwnSake(drive=name)
        doing = max(0.0, _mean(held.doing) - _mean(self._idle))
        result = _mean(held.results)
        total = doing + result
        return OwnSake(
            drive=name,
            doing=doing,
            result=result,
            own_sake=doing / total if total > 0.0 else 0.0,
            measured=True,
        )

    def survives(self, drive: str) -> bool:
        """Whether an intention of this drive stays open past its need being met."""
        reading = self.read(drive)
        if not reading.measured or reading.own_sake <= 0.5:
            return False
        held = self._drives.get(reading.drive)
        last = held.last_doing if held is not None else None
        return last is not None and last >= _mean(self._idle)

    def status(self) -> dict[str, Any]:
        return {name: self.read(name).as_dict() for name in sorted(self._drives)}


_LEDGER: DoingLedger | None = None


def get_doing_ledger() -> DoingLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = DoingLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
