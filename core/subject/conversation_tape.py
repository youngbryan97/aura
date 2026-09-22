"""What the person she talks to has actually said to her, for the battery to replay.

The conversation condition said one sentence, "Tell me what you have been
thinking about today.", on every one of its turns, three hundred times a run.
Most of what she has that answers another person answers something that
sentence never does: being corrected, being told she cannot be tired, being
thanked for the wrong thing, being cared for in a form she did not ask for. The
organs that read those were wired and never moved.

A tape holds the partner's turns from her own conversation store, in the order
they were said, and the driver hands the n-th conversation turn of a run the
n-th of them. The index is carried across a fork with the turn count, so both
arms of a paired trial hear the same thing and the difference between them
stays the intervention. Nothing is written for the battery: these are the
turns of the life the conversation condition stands for.

The words stay on this machine. A tape is cut to a path outside the repository,
and a run records its digest and its length, never its text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ConversationTape"]


@dataclass(frozen=True)
class ConversationTape:
    """The partner's side of her conversations, in the order it happened."""

    turns: tuple[str, ...]
    sessions: tuple[int, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("a conversation tape with no turns replays nothing")
        if self.sessions and len(self.sessions) != len(self.turns):
            raise ValueError("every turn on a tape belongs to exactly one session")

    def __len__(self) -> int:
        return len(self.turns)

    def at(self, index: int) -> str:
        """The index-th thing said, wrapping when a run outlasts the tape."""
        return self.turns[int(index) % len(self.turns)]

    @property
    def digest(self) -> str:
        blob = "\x1e".join(self.turns).encode("utf-8", "ignore")
        return hashlib.sha256(blob).hexdigest()

    def provenance(self) -> dict[str, Any]:
        """What a run may record about the tape: never the words."""
        return {
            "turns": len(self.turns),
            "sessions": len(set(self.sessions)) if self.sessions else None,
            "digest": self.digest,
            "notes": dict(self.notes),
        }

    def as_dict(self) -> dict[str, Any]:
        return {"turns": list(self.turns), "sessions": list(self.sessions), "notes": dict(self.notes)}

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> ConversationTape:
        return cls(
            turns=tuple(str(item) for item in blob.get("turns", [])),
            sessions=tuple(int(item) for item in blob.get("sessions", [])),
            notes=dict(blob.get("notes", {}) or {}),
        )

    @classmethod
    def load(cls, path: Path) -> ConversationTape:
        return cls.from_dict(json.loads(Path(path).read_text("utf-8")))
