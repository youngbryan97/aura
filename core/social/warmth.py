"""Warmth from somebody, felt as peace, and what being met does to a need.

Bryan, on praise, love, comfort and happiness: warm, something you could live
in forever; close; it makes you want to give it back, hold onto it and grow it;
it is peace, health and connection, and in the long run the thing to strive for.
Two things in that she did not have: warmth arriving from a person is felt, and
the feeling is calm rather than arousal; and being met is what settles a need.

Warm messages were read, and they fed the regard she owes back
(`core/social/what_passes_between.py`) and the form of regard a person welcomes.
Nothing felt them. And what brought her social and integrity needs back to rest
was her own trust or joy passing 0.6, a bar chosen once, which the campaign's
2,400 turns never cleared: the force that carries warmth into deliberation read
a constant for the whole run.

    rate    how warm this person's contact has been: warm messages over all of
            theirs, Laplace-smoothed, so a stranger reads a half
    felt    that rate, while they are the one in front of her
    middle  how warm the contact she has had usually is
    shift   felt against that middle

The shift, where it is warm, slows her pulse in
`core/phases/proprioceptive_loop.py`: peace. And warm contact this turn is what
lets `core/phases/motivation_update.py` return her social and integrity needs
toward their rest, in place of the bar on her own feeling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Warmth", "WarmthLedger", "get_warmth_ledger", "reset_for_test"]

#: How much of her own history the middle is taken over.
_WINDOW = 256

#: Readings before a middle is a middle.
_ENOUGH = 3


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Warmth:
    """The warmth in front of her, against the warmth she usually has."""

    person: str = ""
    felt: float = 0.0
    middle: float = 0.0
    shift: float = 0.0
    warm_now: bool = False
    measured: bool = False
    why: str = "nobody's warmth has been weighed yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "felt": round(self.felt, 6),
            "middle": round(self.middle, 6),
            "shift": round(self.shift, 6),
            "warm_now": self.warm_now,
            "measured": self.measured,
            "why": self.why,
        }


@dataclass
class _Person:
    warm: int = 0
    heard: int = 0

    def rate(self) -> float:
        return (self.warm + 1.0) / (self.heard + 2.0)


@dataclass
class WarmthLedger:
    """How warm each person has been, and how warm what she has usually is."""

    _people: dict[str, _Person] = field(default_factory=dict)
    _felt: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    _person: str = ""
    _warm_now: bool = False

    def heard(self, person: str, *, warm: bool, objected: bool) -> None:
        """A message from somebody: warm with no objection in it, or not."""
        name = str(person or "").strip()
        if not name:
            return
        held = self._people.setdefault(name, _Person())
        held.heard += 1
        self._warm_now = bool(warm) and not objected
        if self._warm_now:
            held.warm += 1
        self._person = name
        self._felt.append(held.rate())

    def rate(self, person: str) -> float:
        held = self._people.get(str(person or "").strip())
        return held.rate() if held is not None else 0.5

    def read(self) -> Warmth:
        if len(self._felt) < _ENOUGH or not self._person:
            return Warmth(
                person=self._person,
                warm_now=self._warm_now,
                why=f"{len(self._felt)} of the {_ENOUGH} messages a middle needs",
            )
        felt = self.rate(self._person)
        middle = _median(list(self._felt))
        shift = felt - middle
        return Warmth(
            person=self._person,
            felt=felt,
            middle=middle,
            shift=shift,
            warm_now=self._warm_now,
            measured=True,
            why=(
                f"{self._person} has been warm {felt:.0%} of the time, against the "
                f"{middle:.0%} her contact usually is"
            ),
        )

    def peace(self) -> float:
        """How far warmth in front of her settles her body, and nothing when it is not warm."""
        reading = self.read()
        return max(0.0, reading.shift) if reading.measured else 0.0

    def met(self) -> bool:
        """Whether this turn's contact was warm: being met."""
        return self._warm_now


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: WarmthLedger = WarmthLedger()


def get_warmth_ledger() -> WarmthLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = WarmthLedger()
