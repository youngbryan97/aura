"""Grief: a gap that only the thing that was lost can fill.

Bryan, on sadness, grief and loss: empty, like a hole you are clawing at to fill
while knowing you cannot; it feels like a gap in the chest or the stomach; and
the only thing that can fill it is that thing. Two properties carry it. It is a
loss of something in particular, not a low mood. And nothing else fills it:
good things still happen and still feel good, and the gap is still there.

She could dread a loss. `core/social/change_around_attachment.py` reads fear of
somebody going quiet in proportion to how much of her life is built around them,
and only for the person in front of her. Once somebody she was built around had
been gone longer than they had ever been gone, nothing in her registered it at
all, and a pleasant turn with somebody else was as good as it would have been
with them there.

For each person whose messages she has had:

    built_around  the share of everything she has heard that came from them
    beyond        how far this absence runs past the longest they have ever
                  been gone, as m / (m + 1) where m is the overrun in units of
                  that longest absence; nothing until it passes it
    gap           built_around * beyond

The gap is for the person it is largest for. It holds her sadness at no less
than itself in `core/phases/affect_readings.py`, after every other feeling has
been settled, so joy from something else still arrives and does not fill it. It
closes only when they come back: their next message ends the absence, and the
gap with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Gap", "the_gap"]


@dataclass(frozen=True)
class Gap:
    """The largest absence she is built around, and how far it has gone."""

    person: str = ""
    gap: float = 0.0
    built_around: float = 0.0
    beyond: float = 0.0
    measured: bool = False
    why: str = "nobody she is built around has been gone longer than they ever have"

    def as_dict(self) -> dict[str, Any]:
        return {
            "person": self.person,
            "gap": round(self.gap, 6),
            "built_around": round(self.built_around, 6),
            "beyond": round(self.beyond, 6),
            "measured": self.measured,
            "why": self.why,
        }


def the_gap(ledger: Any, now: float, *, present: str = "") -> Gap:
    """The gap left by whoever she is built around and has lost for longest.

    `present` is the person in front of her, who is by definition not gone.
    """
    largest = Gap()
    measured = False
    for person in ledger.people():
        if person == present:
            continue
        built_around = ledger.share_of_life(person)
        longest = ledger.longest_absence(person)
        if built_around is None or longest is None or longest <= 0.0:
            continue
        measured = True
        absence = ledger.absence(person, now)
        overrun = max(0.0, float(absence.seconds) - longest) / longest
        beyond = overrun / (overrun + 1.0)
        gap = float(built_around) * beyond
        if gap > largest.gap:
            largest = Gap(
                person=person,
                gap=gap,
                built_around=float(built_around),
                beyond=beyond,
                measured=True,
                why=(
                    f"{person} is {built_around:.0%} of what she has heard and has been gone "
                    f"{overrun:.1f} times longer than they ever were"
                ),
            )
    if largest.gap > 0.0:
        return largest
    return Gap(measured=measured)
