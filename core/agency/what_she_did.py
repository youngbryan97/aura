"""What she did, as the world took it: each act proposed, attempted, and accepted or refused, kept apart.

An intention said aloud is not an act, and an act attempted is not an act the world accepted. LIVE 23 November 2026
her account of a game named moves the game had refused: what she said as she played held "I'll move ..." before each
attempt, and nothing kept the attempts the game would not take apart from the moves it made. An account built on
that can describe a game that did not happen.

Each act is proposed, attempted, and dispatched once the input adapter acknowledges sending it. Only a dispatched
act can be settled from the world's response. Reports can distinguish accepted actions and explicit refusals from
attempts whose outcome is still unknown.

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

    def dispatched(self, act: int) -> None:
        """Mark an act whose input adapter acknowledged sending the input."""
        if 0 <= act < len(self.acts) and self.acts[act]["stage"] == "attempted":
            self.acts[act].update(stage="dispatched", dispatched_at=time.time())

    def settled(self, act: int, outcome: str, *, known_by: str = "") -> bool:
        """Settle a delivered input when its effect can be attributed to that attempt."""
        if not (0 <= act < len(self.acts)) or not known_by or self.acts[act]["stage"] != "dispatched":
            return False
        self.acts[act].update(stage=outcome, known_by=known_by, settled_at=time.time())
        return True

    def done(self) -> list[str]:
        return [act["what"] for act in self.acts if act["stage"] == ACCEPTED]

    def refused(self) -> list[str]:
        return [act["what"] for act in self.acts if act["stage"] == REFUSED]

    def unsettled(self) -> list[str]:
        """Proposals and inputs without a confirmed outcome."""
        return [act["what"] for act in self.acts if act["stage"] in ("proposed", "attempted", "dispatched")]

    def as_record(self) -> dict[str, Any]:
        return {"done": len(self.done()), "refused": self.refused(), "unsettled": self.unsettled()}
