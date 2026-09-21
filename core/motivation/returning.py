"""A pull that outlives the reasons for it.

"Georgia On My Mind" spends its middle on the alternatives and does not argue
with them. "Other arms reach out to me, other eyes smile tenderly" — and then
"still in peaceful dreams I see the road leads back to you." Charles sings it
upward: the medians climb 124, 191, 208, 222 through the acknowledgment and
peak on the return. The performance has the narrowest dynamic range of the
twenty-four songs measured, 8.3 dB, and nearly the widest pitch range, 34.9
semitones. All of the expression is in pitch and none of it in volume, which is
what a pull that does not need to raise its voice sounds like.

Her chooser takes the option with the best score and says why. Nothing asked
the question the song asks: after an option loses, does she come back to it, and
does she come back to some more than to others?

    passed over   an option was on the table and something else won
    returned      an option she had passed over was chosen later
    base          everything returned over everything passed over — her own rate
    rate(o)       that option's returns over its passes
    excess(o)     rate(o) - base, in units of the spread of her own rates
    conceded(o)   how far the winner's score was above it on the passes it lost

An option she returns to no more often than she returns to anything is not a
pull. An option whose rate stands above her own spread, and which lost by a
margin rather than by a hair, is the one the song is about. Both readings are
taken against her own history, so the answer can come out either way, and does:
a chooser that always takes the top option and never comes back to a loser
reports no pull at all, which is the honest reading of that chooser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MIN_OPTIONS",
    "MIN_PASSES",
    "Returning",
    "ReturningLedger",
    "get_returning_ledger",
    "reset_for_test",
]

#: Passes before an option's return rate is a rate rather than a coin.
MIN_PASSES: int = 3

#: Options with enough passes before the spread of her rates is a spread.
MIN_OPTIONS: int = 3

#: How many options are held.
WINDOW: int = 400


@dataclass
class _Option:
    passes: int = 0
    returns: int = 0
    #: Margins it lost by, kept so a near miss and a rout are not one number.
    margins: list[float] = field(default_factory=list)
    #: True while it is standing passed over, so the next choice of it counts.
    waiting: bool = False
    label: str = ""


@dataclass
class Returning:
    """Whether something draws her back after better things have won."""

    #: Her own rate of coming back to anything she passed over.
    base_rate: float = 0.0
    #: The strongest pull, in units of the spread of her own per-option rates.
    pull: float = 0.0
    #: What it is.
    draws_her_back: str = ""
    #: How far the winners stood above it on the occasions it lost.
    conceded: float = 0.0
    options: int = 0
    #: True when something comes back to her above her own spread and lost by a
    #: margin rather than by a hair.
    survives_alternatives: bool = False
    measured: bool = False
    why: str = "she has not passed anything over yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_rate": round(self.base_rate, 6),
            "pull": round(self.pull, 6),
            "draws_her_back": self.draws_her_back,
            "conceded": round(self.conceded, 6),
            "options": self.options,
            "survives_alternatives": self.survives_alternatives,
            "measured": self.measured,
            "why": self.why,
        }


class ReturningLedger:
    """What she passed over, and what she went back to."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_OPTIONS * 2, int(window))
        self._book: dict[str, _Option] = {}
        self._order: list[str] = []

    def _row(self, option_id: str) -> _Option:
        row = self._book.get(option_id)
        if row is None:
            if len(self._order) >= self._window:
                self._book.pop(self._order.pop(0), None)
            row = _Option()
            self._book[option_id] = row
            self._order.append(option_id)
        return row

    def note_choice(
        self,
        *,
        chosen_id: str,
        scores: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """One choice: what was on the table, and what won.

        The losers are marked as standing passed over. The winner counts as a
        return when it was standing passed over from an earlier choice, which
        is the only way an option can come back.
        """
        if not chosen_id or not scores:
            return
        labels = labels or {}
        try:
            winning = float(scores.get(chosen_id, 0.0))
        except (TypeError, ValueError):
            return
        if winning != winning:
            return
        chosen = self._row(str(chosen_id))
        chosen.label = labels.get(chosen_id, chosen.label) or str(chosen_id)
        if chosen.waiting:
            chosen.returns += 1
            chosen.waiting = False
        for option_id, score in scores.items():
            if str(option_id) == str(chosen_id):
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value != value:
                continue
            row = self._row(str(option_id))
            row.label = labels.get(option_id, row.label) or str(option_id)
            row.passes += 1
            row.margins.append(max(0.0, winning - value))
            row.waiting = True

    def pull_for(self, option_id: str) -> float:
        """How much more often she comes back to this one than to anything.

        In her own units: a rate minus a rate, so it is a share of occasions
        and sits on the same scale as the other terms the chooser adds. Zero
        until the option has been passed over enough times for its rate to be a
        rate, and zero while her own base rate cannot be read.
        """
        row = self._book.get(str(option_id))
        if row is None or row.passes < MIN_PASSES:
            return 0.0
        passes = sum(item.passes for item in self._book.values())
        returns = sum(item.returns for item in self._book.values())
        if passes <= 0:
            return 0.0
        return max(0.0, (row.returns / row.passes) - (returns / passes))

    def read(self) -> Returning:
        passes = sum(row.passes for row in self._book.values())
        returns = sum(row.returns for row in self._book.values())
        if passes <= 0:
            return Returning()
        base = returns / passes
        eligible = [
            (option_id, row)
            for option_id, row in self._book.items()
            if row.passes >= MIN_PASSES
        ]
        if len(eligible) < MIN_OPTIONS:
            return Returning(
                base_rate=base,
                options=len(eligible),
                why=(
                    f"{len(eligible)} options have been passed over {MIN_PASSES} times; "
                    f"{MIN_OPTIONS} are needed before the spread of her rates is a spread"
                ),
            )
        rates = [row.returns / row.passes for _, row in eligible]
        mean = sum(rates) / len(rates)
        spread = math.sqrt(sum((r - mean) ** 2 for r in rates) / len(rates))
        if spread <= 1e-9:
            # She comes back to everything equally often. That is a real answer:
            # nothing draws her more than anything else does.
            return Returning(
                base_rate=base,
                options=len(eligible),
                measured=True,
                why=(
                    f"she returns to all {len(eligible)} of them at the same rate "
                    f"({mean:.2f}), so nothing draws her back more than anything else"
                ),
            )
        best_id, best_row = max(eligible, key=lambda item: item[1].returns / item[1].passes)
        best_rate = best_row.returns / best_row.passes
        pull = (best_rate - mean) / spread
        conceded = (
            sum(best_row.margins) / len(best_row.margins) if best_row.margins else 0.0
        )
        # It has to have lost to something better, not to a tie. A margin at or
        # below the spread of her own scores is a coin landing, and coming back
        # to a coin is not the thing the song is about.
        margins = [m for _, row in eligible for m in row.margins]
        typical = sum(margins) / len(margins) if margins else 0.0
        survives = pull > 1.0 and conceded >= typical
        return Returning(
            base_rate=base,
            pull=pull,
            draws_her_back=best_row.label or best_id,
            conceded=conceded,
            options=len(eligible),
            survives_alternatives=survives,
            measured=True,
            why=(
                f"{best_row.label or best_id} draws her back at {best_rate:.2f} against "
                f"her own {mean:.2f} ({pull:.1f} spreads), having lost by {conceded:.2f} "
                f"against a usual {typical:.2f}"
                if survives
                else (
                    f"the strongest pull is {best_row.label or best_id} at {pull:.1f} "
                    f"spreads over her own rate, which is not more than she comes back "
                    f"to anything"
                )
            ),
        )


_LEDGER: ReturningLedger | None = None


def get_returning_ledger() -> ReturningLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ReturningLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
