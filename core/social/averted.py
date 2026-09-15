"""Seeing something and putting it down, which is not the same as missing it.

"Tom's Diner" is four minutes of a person in a corner reporting everything in
the room, and the one line that is not a report is the refusal: "and I look the
other way as they are kissing their hellos, and I'm pretending not to see
them." She sees it. The looking away is the act.

The cover measures as the widest performance of the twenty-four, 47.6 dB of
range, on a song whose subject is somebody who is not part of the room. The
loud parts are the noticing and the quiet parts are the declining, in one
voice, which is what makes the courtesy audible.

Her interpersonal observer already refuses things. A trait inferred from one
exchange is the caricature its store exists to make unrepresentable, so the
observer drops it. The drop left no trace, so a perception she declined and a
perception she never had looked the same afterwards, and the difference is the
whole of the process.

    sightings(k)  how many times a claim came round
    taken         the median sightings over claims she recorded
    declined      the median sightings over claims she never recorded
    held          declined claims that come round at least as often as taken ones

A claim she puts down once was probably noise. A claim that comes round as
often as the ones she keeps, and goes into the record none of those times, is a
live signal under a hand. ``held`` counts those, and they are what costs
something to hold.

The comparison is against her own two distributions rather than a threshold
chosen here, so the reading can come out either way: if the claims she refuses
are one-offs while the claims she keeps recur, she was not looking away from
anything and the verdict says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_KEYS",
    "Averted",
    "AvertedLedger",
    "get_averted_ledger",
    "reset_for_test",
]

#: Claims on each side before a median is a middle rather than an endpoint.
MIN_KEYS: int = 3

#: How many distinct claims are held on each side.
WINDOW: int = 200

_SPACE = re.compile(r"\s+")


def _key(facet: str, claim: str) -> str:
    text = _SPACE.sub(" ", str(claim or "")).strip().lower()
    return f"{facet}|{text}" if text else ""


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class Averted:
    """What she saw and did not take up."""

    #: Sightings that went nowhere, over all sightings.
    share_declined: float = 0.0
    #: Median times a recorded claim came round.
    taken_recurrence: float = 0.0
    #: Median times a refused claim came round.
    declined_recurrence: float = 0.0
    #: Refused claims that recur at least as often as the recorded ones.
    held: int = 0
    #: True when the refusals are live signals rather than one-offs.
    looking_away: bool = False
    measured: bool = False
    why: str = "nothing has been noticed about anybody yet"
    #: The held claims, longest-standing first, for anything that wants to say
    #: what she is carrying.
    carrying: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "share_declined": round(self.share_declined, 6),
            "taken_recurrence": round(self.taken_recurrence, 6),
            "declined_recurrence": round(self.declined_recurrence, 6),
            "held": self.held,
            "looking_away": self.looking_away,
            "measured": self.measured,
            "why": self.why,
            "carrying": list(self.carrying),
        }


class AvertedLedger:
    """Every claim she noticed about somebody, and whether it was recorded."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_KEYS * 2, int(window))
        #: key -> [sightings, times recorded]
        self._book: dict[str, list[int]] = {}
        self._order: list[str] = []

    def _bump(self, facet: str, claim: str, *, recorded: bool) -> None:
        key = _key(facet, claim)
        if not key:
            return
        row = self._book.get(key)
        if row is None:
            if len(self._order) >= self._window:
                oldest = self._order.pop(0)
                self._book.pop(oldest, None)
            row = [0, 0]
            self._book[key] = row
            self._order.append(key)
        row[0] += 1
        if recorded:
            row[1] += 1

    def took(self, facet: str, claim: str) -> None:
        """A claim that went into the record."""
        self._bump(facet, claim, recorded=True)

    def declined(self, facet: str, claim: str) -> None:
        """A claim she formed and refused to record."""
        self._bump(facet, claim, recorded=False)

    def read(self) -> Averted:
        taken = [row[0] for row in self._book.values() if row[1] > 0]
        refused = [(key, row[0]) for key, row in self._book.items() if row[1] == 0]
        sightings = sum(row[0] for row in self._book.values())
        declined_sightings = sum(row[0] - row[1] for row in self._book.values())
        share = declined_sightings / sightings if sightings else 0.0
        if len(taken) < MIN_KEYS or len(refused) < MIN_KEYS:
            return Averted(
                share_declined=share,
                why=(
                    f"{len(taken)} recorded and {len(refused)} refused claims; "
                    f"{MIN_KEYS} of each are needed before either median is a middle"
                ),
            )
        taken_median = _median(taken)
        declined_median = _median([count for _, count in refused])
        held = [key for key, count in refused if count >= taken_median]
        held.sort(key=lambda k: -self._book[k][0])
        looking_away = bool(held) and declined_median >= taken_median
        return Averted(
            share_declined=share,
            taken_recurrence=taken_median,
            declined_recurrence=declined_median,
            held=len(held),
            looking_away=looking_away,
            measured=True,
            carrying=tuple(key.split("|", 1)[-1] for key in held[:8]),
            why=(
                f"{len(held)} refused claims come round as often as the ones she keeps "
                f"({declined_median:.1f} against {taken_median:.1f})"
                if looking_away
                else (
                    f"the refused claims come round less often than the ones she keeps "
                    f"({declined_median:.1f} against {taken_median:.1f}), so they were not "
                    f"signals she held down"
                )
            ),
        )


_LEDGER: AvertedLedger | None = None


def get_averted_ledger() -> AvertedLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = AvertedLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
