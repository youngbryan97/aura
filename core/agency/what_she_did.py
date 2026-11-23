"""What she did, as the world took it: each act proposed, attempted, and accepted or refused, kept apart.

An intention said aloud is not an act, and an act attempted is not an act the world accepted. LIVE 23 November 2026
her account of a game named moves the game had refused: what she said as she played held "I'll move ..." before each
attempt, and nothing kept the attempts the game would not take apart from the moves it made. An account built on
that can describe a game that did not happen.

So each act goes through its stages here, in order: proposed (what she decided), attempted (what she did with her
hands), and settled, accepted when the world showed it done and refused when the world would not take it. What she
reports is built from the settled acts: what was done is what was accepted, and what she tried and could not do is
said as that.

Nothing here knows what the acts are: an act is named as its doer names it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ACCEPTED", "REFUSED", "WhatSheDid"]

ACCEPTED = "accepted"
REFUSED = "refused"


@dataclass
class WhatSheDid:
    """Her acts in one undertaking, each at the stage it reached."""

    acts: list[dict[str, Any]] = field(default_factory=list)

    def proposed(self, what: str) -> int:
        """An act decided on; its number, for the stages after."""
        self.acts.append({"what": str(what), "stage": "proposed", "at": time.time()})
        return len(self.acts) - 1

    def attempted(self, act: int) -> None:
        if 0 <= act < len(self.acts) and self.acts[act]["stage"] == "proposed":
            self.acts[act]["stage"] = "attempted"

    def settled(self, act: int, outcome: str, *, known_by: str = "") -> None:
        """What the world did with the act: ACCEPTED, shown done, or REFUSED; and how that is known."""
        if 0 <= act < len(self.acts):
            self.acts[act].update(stage=outcome, known_by=known_by, settled_at=time.time())

    def done(self) -> list[str]:
        return [act["what"] for act in self.acts if act["stage"] == ACCEPTED]

    def refused(self) -> list[str]:
        return [act["what"] for act in self.acts if act["stage"] == REFUSED]

    def unsettled(self) -> list[str]:
        """Acts proposed or attempted whose outcome the world never showed: neither done nor refused, as far as she
        knows."""
        return [act["what"] for act in self.acts if act["stage"] in ("proposed", "attempted")]

    def as_record(self) -> dict[str, Any]:
        return {"done": len(self.done()), "refused": self.refused(), "unsettled": self.unsettled()}
