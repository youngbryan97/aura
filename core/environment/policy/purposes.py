"""What an action is for, apart from what any one world calls it: getting on, restoring what runs low, getting away, waiting.

The ranker scores an action by its purpose. Which purpose an action of a given
world serves is that world's to say: the layer that proposes its actions tags
them (a stair taken advances, food eaten restores). Without a tag, the generic
words for moving on, getting away and holding still say it; a name that is
neither tagged nor generic serves no purpose the ranker knows, and is scored
on what its simulation predicts alone.
"""
from __future__ import annotations

from typing import Any, Final

__all__ = ["ADVANCE", "EVADE", "IDLE", "PURPOSES", "RESTORE", "purpose_of"]

ADVANCE: Final = "advance"
RESTORE: Final = "restore"
EVADE: Final = "evade"
IDLE: Final = "idle"
PURPOSES: Final = (ADVANCE, RESTORE, EVADE, IDLE)

#: The generic actions every world has, by what they are for.
_GENERIC: Final = {
    "move": ADVANCE, "explore_frontier": ADVANCE, "navigate": ADVANCE,
    "stabilize_resource": RESTORE,
    "retreat": EVADE, "retreat_to_safety": EVADE,
    "wait": IDLE, "observe": IDLE,
}


def purpose_of(intent: Any) -> str:
    """The purpose an action serves: its own tag, else what its generic name says, else ""."""
    tags = set(getattr(intent, "tags", ()) or ())
    for purpose in PURPOSES:
        if purpose in tags:
            return purpose
    if "threat_response" in tags:
        return EVADE
    return _GENERIC.get(str(getattr(intent, "name", "") or ""), "")
