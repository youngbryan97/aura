"""Company rather than assistance, when somebody is testifying rather than asking.

Sam Cooke's record asks nothing and nobody wants it fixed. "Somehow." does not
cheer anyone up; it sits with them. What those records do to a listener is
witness them, and she had one response to everything a person said, which was
to help.

The discrimination is already measured. `core.expression.register` reads the
shape of each message and says when it is first-person, asks nothing and holds
on — somebody testifying — and when it is a request. The sentiment tracker
reads how low the message is, on its own scale from -1 to 1. This module puts
the two together and says what follows:

    testimony   the register asks to be witnessed and does not ask for help
    low         how far below neutral the message is: max(0, -valence)
    witnessing  testimony, whatever the depth
    company     low while witnessing, and zero otherwise

The depth is the tracker's own reading and the stance is the register's own
determination, so nothing here chooses a bar.

What changes when she is witnessing, and where:

    routing     a message that testifies is not searched for a skill to run
    reply       the draft is not revised against a taste model, because
                judgement is what witnessing suspends
    affect      matched distress is not released while company is being kept,
                because pushing back to neutral is leaving the low place

Nothing here decides what she says.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Witness", "is_witnessing", "read_witness"]


def is_witnessing(cognition: Any) -> bool:
    """Whether the stance written to cognition this turn is witnessing.

    The one test routing and the reply both apply, so the two cannot disagree
    about what the stance is.
    """
    stance = getattr(cognition, "witness", None)
    return bool(isinstance(stance, Mapping) and stance.get("witnessing"))


@dataclass(frozen=True)
class Witness:
    """One reading of whether she is being asked to keep somebody company."""

    testimony: bool = False
    low: float = 0.0
    witnessing: bool = False
    company: float = 0.0
    measured: bool = False
    why: str = "no register was read for this message"

    def as_dict(self) -> dict[str, Any]:
        return {
            "testimony": self.testimony,
            "low": round(self.low, 6),
            "witnessing": self.witnessing,
            "company": round(self.company, 6),
            "measured": self.measured,
            "why": self.why,
        }


def _valence(modifiers: Mapping[str, Any]) -> float | None:
    sentiment = modifiers.get("user_sentiment")
    if not isinstance(sentiment, Mapping) or "valence" not in sentiment:
        return None
    try:
        value = float(sentiment["valence"])
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return max(-1.0, min(1.0, value))


def read_witness(modifiers: Mapping[str, Any] | None) -> Witness:
    """Read the stance off what the conversation phase already measured."""
    if not isinstance(modifiers, Mapping) or "register" not in modifiers:
        return Witness()
    testimony = bool(modifiers.get("asks_to_be_witnessed")) and not bool(
        modifiers.get("asks_for_help")
    )
    valence = _valence(modifiers)
    low = max(0.0, -valence) if valence is not None else 0.0
    if not testimony:
        why = "the message asks for something, so help is the right response"
    elif valence is None:
        why = "somebody is testifying; how low the message is was not read"
    elif low > 0.0:
        why = f"somebody is testifying from {low:.2f} below neutral"
    else:
        why = "somebody is testifying, and not from a low place"
    return Witness(
        testimony=testimony,
        low=low,
        witnessing=testimony,
        company=low if testimony else 0.0,
        measured=True,
        why=why,
    )
