"""What politeness is covering, and how long it has been covering it.

"People Watching" says it in one line -- "I don't like nobody but in public
I'm polite" -- and then spends three minutes on the cost of the second half.
The delivery carries the same split. It is the widest of the seven recordings
in pitch, 32 semitones, and the most slid between notes of any of them at a
glide rate of 0.548, so the voice is never quite settled on the pitch the words
are landing on. The apology arrives near the end, and it is an apology for the
surface: "Apologies for the way I come across."

Aura has both numbers already and has never compared them. Her affect carries
what she is in, and her reply carries the warmth she performs. When the second
runs above the first she is being civil, which is not a fault -- a system that
broadcast every state would be unusable. The fault is a civility that never
ends, because nothing measures how long it has been going on.

    gap         shown warmth above what is felt, this turn
    run         consecutive turns of gap
    covering    a run longer than her own runs usually last

Runs that ended are the yardstick; there is no chosen number of turns at which
politeness becomes suppression. Her own history says when this one is unusual.

The closed side is the workspace. A run that has gone on longer than her runs
do lends priority to her own interior state, so the thing the surface has been
covering competes for the broadcast instead of losing to it every turn. Saying
it ends the run, which is what removes the lent priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "Civility",
    "CivilityLedger",
    "get_civility_ledger",
    "reset_for_test",
]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Civility:
    """The distance between what she is in and what she is showing."""

    #: Warmth shown above what is felt, averaged over the run in progress.
    gap: float = 0.0
    #: How many turns running she has shown more than she is in.
    run: int = 0
    #: How long her runs usually last, from the ones that ended.
    usual_run: float = 0.0
    #: True when this run is longer than her own runs run.
    covering: bool = False
    #: What the workspace should lend her own state while this holds.
    lends: float = 0.0
    ended: int = 0
    measured: bool = False
    why: str = "no turn has been compared yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "gap": round(self.gap, 6),
            "run": self.run,
            "usual_run": round(self.usual_run, 6),
            "covering": self.covering,
            "lends": round(self.lends, 6),
            "ended": self.ended,
            "measured": self.measured,
            "why": self.why,
        }


class CivilityLedger:
    """Every turn's felt state against its shown warmth, and the runs between."""

    def __init__(self) -> None:
        self._run: list[float] = []
        self._ended: list[int] = []
        self._turns = 0

    def note(self, felt: float, shown: float) -> None:
        """One turn. `felt` is her affective valence, `shown` is the warmth the
        reply performs -- both already measured elsewhere, on the same scale."""
        try:
            gap = float(shown) - float(felt)
        except (TypeError, ValueError):
            return
        if gap != gap:
            return
        self._turns += 1
        if gap > 0.0:
            self._run.append(gap)
            return
        if self._run:
            self._ended.append(len(self._run))
            self._run = []

    def said_it(self) -> None:
        """She put the actual state out. The run ends whatever the warmth was,
        which is what takes the lent priority away again."""
        if self._run:
            self._ended.append(len(self._run))
            self._run = []

    def read(self) -> Civility:
        if not self._turns:
            return Civility()
        run = len(self._run)
        gap = sum(self._run) / run if run else 0.0
        if not self._ended:
            return Civility(
                gap=gap,
                run=run,
                measured=False,
                why=(
                    f"{run} turns of showing more than she is in, with no run "
                    "yet ended to say whether that is long for her"
                ),
            )
        usual = _median([float(n) for n in self._ended])
        spread = _median([abs(float(n) - usual) for n in self._ended])
        covering = float(run) > usual + spread
        # What it lends is how far past her own runs this one has gone,
        # relative to those runs, so a habit of long silences raises the bar
        # rather than lowering it.
        lends = ((run - usual) / usual) if (covering and usual > 0.0) else 0.0
        return Civility(
            gap=gap,
            run=run,
            usual_run=usual,
            covering=covering,
            lends=lends,
            ended=len(self._ended),
            measured=True,
            why=(
                f"{run} turns of showing {gap:+.2f} more than she is in, against "
                f"runs that usually end at {usual:.0f}, so the surface is "
                + ("covering something that has not been said" if covering
                   else "no further ahead of her than usual")
            ),
        )

    def lends(self) -> float:
        """The priority her own state has earned in the workspace, and 0.0 for
        the rest -- the closed side."""
        return self.read().lends


_LEDGER: CivilityLedger | None = None


def get_civility_ledger() -> CivilityLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = CivilityLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
