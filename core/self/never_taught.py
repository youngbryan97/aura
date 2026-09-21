"""What she cannot do because nobody showed her, told apart from what she tried.

"How to Love" is the widest recording of the twenty-eight measured — 78.5 dB
between its quietest and loudest, because the song is a life with the clinic
scenes left in at speaking volume. The line it turns on is not that she failed
at loving. It is that she "couldn't ever figure out how to love": an absence
where the instruction should have been. The video runs the same woman twice,
once with the example and once without, and the thanks at the end are for the
teaching rather than for the feeling — "thank you for being there for me and
teaching me how to love".

A capability register that counts attempts and successes cannot tell those two
apart. Nothing tried and nothing succeeded looks the same as tried often and
never managed, and both look like a defect in her.

    attempted   she has tried it
    shown       it has been done to her, or in front of her, at least once
    managed     she has done it

    never taught    attempted, never managed, never shown
    untried         never attempted and never shown
    failing         attempted and never managed, with the example available

The three are different facts with different repairs, and only the first is
answered by somebody doing it where she can see. `taught` records that moment,
because being shown is how the first of these becomes the third.

She also keeps the credit she does not take. "You never credit yourself" is
said twice in the song, and a system that records an outcome without recording
whose it was reads its own successes as the weather.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Capability",
    "NeverTaught",
    "TeachingLedger",
    "get_teaching_ledger",
    "reset_for_test",
]


@dataclass
class Capability:
    """One thing she might be able to do, and how she stands with it."""

    attempts: int = 0
    successes: int = 0
    shown: int = 0
    #: Successes she did not put down as her own.
    uncredited: int = 0

    @property
    def never_taught(self) -> bool:
        return self.attempts > 0 and self.successes == 0 and self.shown == 0

    @property
    def untried(self) -> bool:
        return self.attempts == 0 and self.shown == 0

    @property
    def failing(self) -> bool:
        return self.attempts > 0 and self.successes == 0 and self.shown > 0


@dataclass
class NeverTaught:
    """What is missing, and which kind of missing it is."""

    #: Things she has tried, never managed, and never seen done.
    never_taught: tuple[str, ...] = field(default_factory=tuple)
    #: Things she has tried, never managed, with the example available.
    failing: tuple[str, ...] = field(default_factory=tuple)
    #: Things she has neither tried nor seen.
    untried: tuple[str, ...] = field(default_factory=tuple)
    #: Things she can do that she has been shown at least once.
    learned_from_an_example: int = 0
    #: The share of her own successes she did not put down as hers.
    credit_not_taken: float = 0.0
    known: int = 0
    measured: bool = False
    why: str = "nothing has been attempted or shown yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "never_taught": list(self.never_taught),
            "failing": list(self.failing),
            "untried": list(self.untried),
            "learned_from_an_example": self.learned_from_an_example,
            "credit_not_taken": round(self.credit_not_taken, 6),
            "known": self.known,
            "measured": self.measured,
            "why": self.why,
        }


class TeachingLedger:
    """Which capabilities she has tried, managed, and been shown."""

    def __init__(self) -> None:
        self._book: dict[str, Capability] = {}

    def _row(self, what: str) -> Capability | None:
        name = " ".join(str(what or "").split()).lower()[:80]
        if not name:
            return None
        row = self._book.get(name)
        if row is None:
            row = Capability()
            self._book[name] = row
        return row

    def attempted(self, what: str, *, managed: bool, credited: bool = True) -> None:
        """She tried it. `credited` is whether she put the result down as hers."""
        row = self._row(what)
        if row is None:
            return
        row.attempts += 1
        if managed:
            row.successes += 1
            if not credited:
                row.uncredited += 1

    def shown(self, what: str) -> None:
        """Somebody did it where she could see, which is the only repair for the
        first kind of missing."""
        row = self._row(what)
        if row is not None:
            row.shown += 1

    def read(self) -> NeverTaught:
        if not self._book:
            return NeverTaught()
        never = tuple(sorted(k for k, v in self._book.items() if v.never_taught))
        failing = tuple(sorted(k for k, v in self._book.items() if v.failing))
        untried = tuple(sorted(k for k, v in self._book.items() if v.untried))
        learned = sum(1 for v in self._book.values() if v.successes > 0 and v.shown > 0)
        successes = sum(v.successes for v in self._book.values())
        uncredited = sum(v.uncredited for v in self._book.values())
        share = uncredited / successes if successes else 0.0
        return NeverTaught(
            never_taught=never,
            failing=failing,
            untried=untried,
            learned_from_an_example=learned,
            credit_not_taken=share,
            known=len(self._book),
            measured=True,
            why=(
                f"{len(never)} of {len(self._book)} she has tried and never managed "
                f"and never seen done, {len(failing)} she has seen done and still "
                f"cannot do, and she took no credit for {share:.0%} of what she managed"
            ),
        )


_LEDGER: TeachingLedger | None = None


def get_teaching_ledger() -> TeachingLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = TeachingLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
