"""What she notices as she goes, most of it let go, some of it grown into what she goes by.

A person at work on something notices things without being asked to: "that's
odd", "I wonder if", "every time I do this, that happens", "that wasn't here
before". Most of it comes to nothing and is forgotten. Some of it is seen
again, and again, until it is a thing they know: hitting that one makes it
stop for a moment; this button is what scores; the count goes up by itself.
And what they know changes what they do.

So does she. A note is one such noticing, kept with how often it held and how
often it did not, and with how much it holds her interest, which fades unless
it is seen again or bears on what she is there to do. A note seen often enough
and holding well enough becomes a theory: said once, kept with its evidence,
given to her reasoning and to the check on what she does
(core/cognition/checking_the_debate.py), and let go again if the evidence turns
against it. Notes that are only notes stay in her log, not in what she says.

Nothing here knows what is noticed: it is told so by whatever is watching
(core/agency/noticing_in_play.py for play).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Note", "Notebook"]

logger = logging.getLogger("Aura.Noticing")

#: Seen this often, holding this much of the time, a note is a theory; holding less than this after this many, it is
#: let go again.
A_THEORY_AFTER = 3
HOLDS = 0.6
LET_GO_BELOW = 0.35
JUDGED_AFTER = 6

#: The kinds of note that can grow into a theory: patterns (what follows what, what a thing does when met, what goes on by
#: itself). A sighting ("that's new"), a question or a remark stays a note, however often it comes.
PATTERNS = frozenset({"after", "when met", "by itself", "together", "noticed"})

#: How long interest in a note lasts, in seconds, by half; and below how much a note that is no theory is forgotten.
HALF_LIFE_S = 90.0
FORGOTTEN_BELOW = 0.15
MOST_NOTES = 60


@dataclass
class Note:
    """One noticing: what it says, how often it held and did not, and how much it holds her interest."""

    key: str
    said: str
    kind: str
    first: float
    last: float
    seen: int = 0
    against: int = 0
    interest: float = 1.0
    theory: bool = False
    #: What it bears on: an act ("space", 'click "Next"'), a thing, a count; and whether it pays (+1) or costs (-1).
    about: str = ""
    pays: int = 0

    @property
    def holds(self) -> float:
        return self.seen / max(1, self.seen + self.against)


@dataclass
class Notebook:
    """Her notes in one place, and the theories they grew into."""

    notes: dict[str, Note] = field(default_factory=dict)

    def notice(self, key: str, said: str, at: float, *, kind: str = "noticed", held: bool = True, about: str = "",
               pays: int = 0, bears: bool = False) -> Note | None:
        """Something noticed (``held``) or not borne out this time; what became a theory now, else None. ``bears``: it
        bears on what she is there to do, and holds her interest more."""
        note = self.notes.get(key)
        if note is None:
            if not held:
                return None
            note = self.notes[key] = Note(key, said, kind, at, at, about=about, pays=pays)
            logger.info("noticing: %s", said)
        note.interest = note.interest * 0.5 ** ((at - note.last) / HALF_LIFE_S) + (1.5 if bears else 1.0) * (1 if held else 0.3)
        note.last, note.said = at, said or note.said
        if held:
            note.seen += 1
        else:
            note.against += 1
        became = None
        if not note.theory and note.kind in PATTERNS and note.seen >= A_THEORY_AFTER and note.holds >= HOLDS:
            note.theory = True
            became = note
            logger.info("noticing became a theory (%d for, %d against): %s", note.seen, note.against, note.said)
        elif note.theory and note.seen + note.against >= JUDGED_AFTER and note.holds < LET_GO_BELOW:
            note.theory = False
            logger.info("a theory let go (%d for, %d against): %s", note.seen, note.against, note.said)
        self._forget(at)
        return became

    def wonder(self, key: str, said: str, at: float) -> None:
        """A question or a passing remark ("that's odd", "I wonder if"), kept only while it holds her interest."""
        self.notice(key, said, at, kind="wondered")

    def theories(self) -> list[Note]:
        return sorted((n for n in self.notes.values() if n.theory), key=lambda n: (-n.holds, -n.seen))

    def paying(self, about: str) -> int:
        """Whether her theories say what ``about`` names pays (+1) or costs (-1), or say nothing of it (0)."""
        found = [n.pays for n in self.theories() if n.about and n.about == about and n.pays]
        return max(found, key=abs) if found else 0

    def lately(self, at: float, most: int = 3) -> list[Note]:
        """The notes that hold her interest most now, theories or not."""
        def now(note: Note) -> float:
            return note.interest * 0.5 ** ((at - note.last) / HALF_LIFE_S)
        return sorted(self.notes.values(), key=now, reverse=True)[:most]

    def for_thinking(self, at: float) -> str:
        """What she has come to think, and what she has noticed lately, for reasoning with."""
        theories = [f"{n.said} (seen {n.seen}, against {n.against})" for n in self.theories()[:4]]
        lately = [n.said for n in self.lately(at) if not n.theory][:2]
        lines = (["I've come to think: " + " | ".join(theories)] if theories else []) + (
            ["I've noticed: " + " | ".join(lately)] if lately else [])
        return "\n".join(lines)

    def _forget(self, at: float) -> None:
        for key, note in list(self.notes.items()):
            if not note.theory and note.interest * 0.5 ** ((at - note.last) / HALF_LIFE_S) < FORGOTTEN_BELOW:
                del self.notes[key]
        if len(self.notes) > MOST_NOTES:
            keep = sorted(self.notes.values(), key=lambda n: (n.theory, n.interest), reverse=True)[:MOST_NOTES]
            self.notes = {n.key: n for n in keep}

    def as_memory(self) -> dict[str, Any]:
        return {"theories": [{"key": n.key, "said": n.said, "kind": n.kind, "seen": n.seen, "against": n.against,
                              "about": n.about, "pays": n.pays} for n in self.theories()]}

    @classmethod
    def from_memory(cls, held: Any) -> Notebook:
        book = cls()
        for one in (held or {}).get("theories") or [] if isinstance(held, dict) else []:
            note = Note(str(one.get("key")), str(one.get("said")), str(one.get("kind") or "noticed"), 0.0, 0.0,
                        seen=int(one.get("seen") or 0), against=int(one.get("against") or 0),
                        about=str(one.get("about") or ""), pays=int(one.get("pays") or 0), theory=True,
                        interest=math.inf)
            book.notes[note.key] = note
        return book
