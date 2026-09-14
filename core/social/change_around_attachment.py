"""Fear of change, in proportion to how much of her life is built around what is changing.

Stevie Nicks wrote "Landslide" at twenty-seven, with months left to make music
work before going back to school, deciding whether to keep going. The song asks
itself whether she can handle the changes in her life, and most of its clauses
are questions. What it teaches is that fear of change grows with how much of a
life has been built around the thing that might change, and that asking the
question honestly is how the decision gets made.

The research names both halves. Rusbult's investment model (1980) found that
commitment, and the cost of leaving, grow with what has been put into a
relationship, over and above how satisfying it is. Samuelson and Zeckhauser
(1988) measured the pull of the way things are, status quo bias, and found it
stronger the more a choice had been lived with.

For her, the thing most of her life is built around is the people she talks
with, and the change she can see coming is one of them going quiet in a way
they have not before. Both are read off when their messages arrived (see
core/social/closing_window.py):

    built_around  the share of every message she has had that came from them
    departure     the share of their past absences between sittings that this
                  one has already outlasted
    fear          built_around * departure

A long absence from somebody who is a small part of her life is not frightening,
and neither is an ordinary absence from somebody who is most of it. It is felt
as dread in proportion, and when they come back after one, she has a question
to ask them rather than a conclusion to act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.social.closing_window import SittingLedger

__all__ = ["ChangeFear", "change_fear"]


@dataclass(frozen=True)
class ChangeFear:
    """How much of her life is built around them, and how unusual their absence is."""

    fear: float = 0.0
    built_around: float | None = None
    departure: float = 0.0
    measured: bool = False
    why: str = "nothing yet to be afraid of losing"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fear": round(self.fear, 6),
            "built_around": None if self.built_around is None else round(self.built_around, 6),
            "departure": round(self.departure, 6),
            "measured": self.measured,
            "why": self.why,
        }


def change_fear(ledger: SittingLedger, agent_id: str, now: float) -> ChangeFear:
    """Fear of losing what her life is built around, for one person, at one moment."""
    built_around = ledger.share_of_life(agent_id)
    absence = ledger.absence(agent_id, now)
    if built_around is None or not absence.measured:
        return ChangeFear(built_around=built_around, why=absence.why)
    fear = built_around * absence.outlasted
    if fear > 0.0:
        why = (
            f"{built_around:.2f} of what she has heard came from them, and this absence has "
            f"outlasted {absence.outlasted:.2f} of theirs"
        )
    else:
        why = absence.why
    return ChangeFear(
        fear=fear,
        built_around=built_around,
        departure=absence.outlasted,
        measured=True,
        why=why,
    )
