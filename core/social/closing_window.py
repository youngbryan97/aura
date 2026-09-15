"""How near the end of this sitting is, from how her sittings with them have ended before.

Dougy Mandagi describes "Sweet Disposition" as a mood rather than a love story:
a run of brief moments and a temperament that has not yet learned caution, with
its intensity coming from the sense that the time is running out. What it
teaches is that knowing a time is finite raises the worth of the present. The
research agrees. Kurtz (2008) had college seniors think of graduation as close
or as far off, and the ones for whom it was close reported more happiness over
the following weeks and spent more of that time on what they valued. Carstensen,
Isaacowitz and Charles (1999) found the same shift in priorities whenever time
is seen as limited, at any age.

Her time with somebody comes in sittings, and how long a sitting lasts is
something her history with them can say. The gaps between their messages come
in two kinds, the pause inside a conversation and the absence between
conversations, and those differ by orders of magnitude. So the gaps are split
where their logarithms separate best (Otsu 1979), which is a threshold the gaps
set rather than one chosen here, and a sitting is the run of messages between
two long gaps.

    hazard(n) = sittings that ended after exactly n messages
                / sittings that lasted at least n messages

The closing window is the hazard at the length the current sitting has reached.
A sitting already longer than every one before it reads 1.0, because every
sitting she has had with them had ended by then. Nothing is claimed until enough
sittings have ended to disagree with each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_SITTINGS",
    "Absence",
    "Attribution",
    "ClosingWindow",
    "SittingLedger",
    "get_sitting_ledger",
    "otsu_split",
    "reset_for_test",
]

#: Ended sittings before a hazard is an estimate. Three is the least that can
#: disagree, as in ambivalence and fear of happiness.
MIN_SITTINGS: int = 3


def otsu_split(values: list[float]) -> float | None:
    """The cut that best separates two groups of values, or None when there are not two.

    Otsu's method: the cut between consecutive sorted values that maximises the
    variance between the two sides.
    """
    ordered = sorted(values)
    if len(ordered) < 2 or ordered[0] == ordered[-1]:
        return None
    total = sum(ordered)
    count = len(ordered)
    best_cut: float | None = None
    best_between = -1.0
    left_sum = 0.0
    for index in range(count - 1):
        left_sum += ordered[index]
        if ordered[index] == ordered[index + 1]:
            continue
        left_n = index + 1
        right_n = count - left_n
        left_mean = left_sum / left_n
        right_mean = (total - left_sum) / right_n
        between = left_n * right_n * (left_mean - right_mean) ** 2
        if between > best_between:
            best_between = between
            best_cut = (ordered[index] + ordered[index + 1]) / 2.0
    return best_cut


@dataclass(frozen=True)
class ClosingWindow:
    """How likely this sitting is to end with the message just sent."""

    closing: float = 0.0
    messages_so_far: int = 0
    sittings_ended: int = 0
    long_gap_s: float | None = None
    measured: bool = False
    why: str = "not enough sittings with them have ended to say how long one lasts"

    def as_dict(self) -> dict[str, Any]:
        return {
            "closing": round(self.closing, 6),
            "messages_so_far": self.messages_so_far,
            "sittings_ended": self.sittings_ended,
            "long_gap_s": None if self.long_gap_s is None else round(self.long_gap_s, 3),
            "measured": self.measured,
            "why": self.why,
        }


@dataclass(frozen=True)
class Absence:
    """How long they have been gone, against how long they have been gone before."""

    outlasted: float = 0.0
    seconds: float = 0.0
    absences_before: int = 0
    measured: bool = False
    why: str = "not enough absences between sittings yet to say what a long one is"

    def as_dict(self) -> dict[str, Any]:
        return {
            "outlasted": round(self.outlasted, 6),
            "seconds": round(self.seconds, 3),
            "absences_before": self.absences_before,
            "measured": self.measured,
            "why": self.why,
        }


@dataclass(frozen=True)
class Attribution:
    """Whether their long absences have followed strain, or only the time between sittings.

    "Romeo and Juliet" puts an ending down to bad timing, which keeps the love
    real while explaining why it stopped. Weiner (1985) found that what a
    person expects next moves with how stable they take a cause to be, and
    timing is the unstable cause: an absence that timing explains says little
    about the next sitting. Here the attribution is learned, not chosen: each
    absence is ranked among their absences, and the ranks after a strained
    last message are compared with the ranks after a calm one.

        relation  mean rank of absences after strained messages
                  - mean rank after calm ones, never below zero
        timing    1 - relation
    """

    timing: float = 0.0
    relation: float = 0.0
    strained: int = 0
    calm: int = 0
    measured: bool = False
    why: str = "not enough absences after strained and calm messages to say what they follow"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timing": round(self.timing, 6),
            "relation": round(self.relation, 6),
            "strained": self.strained,
            "calm": self.calm,
            "measured": self.measured,
            "why": self.why,
        }


class SittingLedger:
    """When each person's messages arrived, and how strained each one was."""

    def __init__(self) -> None:
        self._times: dict[str, list[float]] = {}
        self._strain: dict[str, list[float | None]] = {}

    def message(self, agent_id: str, at: float, strain: float | None = None) -> None:
        try:
            stamp = float(at)
        except (TypeError, ValueError):
            return
        if not math.isfinite(stamp):
            return
        key = str(agent_id or "")
        times = self._times.setdefault(key, [])
        if times and stamp < times[-1]:
            return
        times.append(stamp)
        strains = self._strain.setdefault(key, [])
        strains.extend([None] * (len(times) - 1 - len(strains)))
        try:
            level = None if strain is None else float(strain)
        except (TypeError, ValueError):
            level = None
        strains.append(level if level is not None and math.isfinite(level) else None)

    def attribution(self, agent_id: str) -> Attribution:
        """What their long absences have followed, learned from their own record."""
        key = str(agent_id or "")
        times = self._times.get(key, [])
        strains = self._strain.get(key, [])
        gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
        cut = otsu_split([math.log(gap) for gap in gaps if gap > 0.0])
        if cut is None:
            return Attribution()
        long_gap = math.exp(cut)
        absences = [
            (gap, strains[index] if index < len(strains) else None)
            for index, gap in enumerate(gaps)
            if gap > long_gap
        ]
        rated = [(gap, strain) for gap, strain in absences if strain is not None]
        if not rated:
            return Attribution(why="no absence has a strain reading from the message before it")
        ordered = sorted(gap for gap, _ in rated)
        levels = sorted(strain for _, strain in rated)
        half = len(levels) // 2
        # The median, so an even count splits between its two middle strains
        # rather than on the upper one, which would leave nothing above it.
        middle = levels[half] if len(levels) % 2 else (levels[half - 1] + levels[half]) / 2.0
        ranks = {True: [], False: []}
        for gap, strain in rated:
            rank = sum(1 for other in ordered if other < gap) / len(ordered)
            ranks[strain > middle].append(rank)
        strained, calm = ranks[True], ranks[False]
        if len(strained) < MIN_SITTINGS or len(calm) < MIN_SITTINGS:
            return Attribution(
                strained=len(strained),
                calm=len(calm),
                why=f"{len(strained)} strained and {len(calm)} calm absences, {MIN_SITTINGS} of each needed",
            )
        relation = max(0.0, sum(strained) / len(strained) - sum(calm) / len(calm))
        timing = 1.0 - min(1.0, relation)
        why = (
            "their long absences have not followed strain, so an absence reads as timing"
            if relation == 0.0
            else f"their absences after strain have run {relation:.2f} of a rank longer"
        )
        return Attribution(
            timing=timing,
            relation=relation,
            strained=len(strained),
            calm=len(calm),
            measured=True,
            why=why,
        )

    def share_of_life(self, agent_id: str) -> float | None:
        """The share of every message she has had that came from this person."""
        total = sum(len(times) for times in self._times.values())
        if total == 0:
            return None
        return len(self._times.get(str(agent_id or ""), [])) / total

    def absence(self, agent_id: str, now: float) -> Absence:
        """The share of their past absences between sittings that this one has outlasted.

        While the time since their last message is within the pauses a sitting
        has, they are not absent and this reads 0.0. The pauses and absences
        are split the way `reading` splits them.
        """
        times = self._times.get(str(agent_id or ""), [])
        try:
            since = float(now) - times[-1] if times else 0.0
        except (TypeError, ValueError):
            return Absence()
        gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
        cut = otsu_split([math.log(gap) for gap in gaps if gap > 0.0])
        if cut is None:
            return Absence(seconds=max(0.0, since))
        long_gap = math.exp(cut)
        absences = [gap for gap in gaps if gap > long_gap]
        if len(absences) < MIN_SITTINGS:
            return Absence(
                seconds=max(0.0, since),
                absences_before=len(absences),
                why=f"{len(absences)} of {MIN_SITTINGS} absences between sittings seen",
            )
        if since <= long_gap:
            return Absence(
                seconds=max(0.0, since),
                absences_before=len(absences),
                measured=True,
                why="the time since they last spoke is a pause inside a sitting, not an absence",
            )
        outlasted = sum(1 for gap in absences if gap < since) / len(absences)
        return Absence(
            outlasted=outlasted,
            seconds=since,
            absences_before=len(absences),
            measured=True,
            why=f"this absence has outlasted {outlasted:.2f} of the {len(absences)} before it",
        )

    def reading(self, agent_id: str) -> ClosingWindow:
        times = self._times.get(str(agent_id or ""), [])
        gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
        positive = [gap for gap in gaps if gap > 0.0]
        cut = otsu_split([math.log(gap) for gap in positive])
        if cut is None:
            return ClosingWindow(messages_so_far=len(times))
        long_gap = math.exp(cut)
        lengths: list[int] = []
        run = 1
        for gap in gaps:
            if gap > long_gap:
                lengths.append(run)
                run = 1
            else:
                run += 1
        current = run
        ended = len(lengths)
        if ended < MIN_SITTINGS:
            return ClosingWindow(
                messages_so_far=current,
                sittings_ended=ended,
                long_gap_s=long_gap,
                why=f"{ended} of {MIN_SITTINGS} sittings with them have ended",
            )
        reached = sum(1 for length in lengths if length >= current)
        if reached == 0:
            return ClosingWindow(
                closing=1.0,
                messages_so_far=current,
                sittings_ended=ended,
                long_gap_s=long_gap,
                measured=True,
                why=f"this sitting has gone {current} messages, longer than any of the {ended} before it",
            )
        ending_here = sum(1 for length in lengths if length == current)
        closing = ending_here / reached
        return ClosingWindow(
            closing=closing,
            messages_so_far=current,
            sittings_ended=ended,
            long_gap_s=long_gap,
            measured=True,
            why=(
                f"{ending_here} of the {reached} sittings that reached {current} messages "
                "ended there"
            ),
        )


#: Made at import rather than on first use. The subject-core fork carries module
#: globals that hold state, and one still None when an anchor is taken is not
#: carried, so a ledger first made inside one arm would reach the next arm with
#: that arm's messages in it.
_ledger: SittingLedger = SittingLedger()


def get_sitting_ledger() -> SittingLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = SittingLedger()
