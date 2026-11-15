"""Ways that would not open yet, remembered with what they want, and gone back to when she has it.

A person who meets a locked door, a level that needs thirty stars, a gap they cannot jump or a feature "available in
Pro" does not forget it. They note where it was and what it seemed to want, go on, and when a key turns up, the stars are
counted or the double jump is learned, they go back. Games are built of these (the lock and key of a dungeon, the ability
that opens an area of a Metroidvania: Dormans' missions and spaces, the cycles of his generated dungeons), and so are
programs and sites.

A screen that says a way is closed to her (locked, needs, requires, not yet, not enough) is a lock: kept with what she
was trying when it said so, or the thing it names, and what it asks for, in its own words. A screen that says she has
something now (found, got, acquired, learned, unlocked) is a gain. A gain that names what a lock wanted makes that lock
one to go back to: her choice of move takes the acts that lead, by the screens she knows, toward where it was
(core/agency/where_things_lead.py), and she says so.

Whether a sentence says a way is closed, or that she has gained something, is learned over her model's own
representation of sentences (core/language/learned_matcher.py) from examples drawn from games, programs and sites, and
from every lock and gain she meets; a small word floor stands until it decides.

Nothing here knows a game.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Lock", "Locks", "SAYS_IT_IS_CLOSED", "SAYS_SHE_GAINED", "heard_on_a_screen", "locks_as_memory",
           "locks_from_memory"]

logger = logging.getLogger("Aura.LocksSheMet")

#: The most locks kept for one place.
MOST_LOCKS = 24

_WORD = re.compile(r"[a-z]{3,}")
_PLAIN = frozenset("""the and you your for with from that this into onto any each other another one all its yet now
    need needs require requires required have has had get got must first only before can cannot not more enough some
    unlock unlocked unlocks locked open opens closed found find acquired learned new available use""".split())

_CLOSED = re.compile(r"\b(?:locked|you need|need(?:s)? (?:a|an|the|\d)|requires?|required|not (?:yet|enough)|"
                     r"can'?t (?:go|enter|open|use|reach)|come back (?:later|when)|only (?:with|for)|"
                     r"available (?:in|with|after)|unlock(?:s)? (?:at|with|after|when))\b", re.IGNORECASE)
_GAINED = re.compile(r"\b(?:you (?:found|got|have|learned|obtained|received|earned)|got the|acquired|obtained|"
                     r"unlocked|new (?:ability|item|skill|weapon|power)|learned)\b", re.IGNORECASE)
#: What a closed way asks for: the words after what says it is wanted.
_WANTS = re.compile(r"\b(?:need(?:s)?|requires?|required|with|collect|have|unlock(?:s)? (?:at|with|after|when)|"
                    r"available (?:in|with|after)|until you (?:have|get|find))\s+(.{2,48}?)(?:[.!?,;]|$)", re.IGNORECASE)


def _surface(name: str, positives: tuple[str, ...], negatives: tuple[str, ...]) -> Any:
    from core.language.learned_matcher import LearnedMatcher
    from core.language.model_features import model_hidden_features

    return LearnedMatcher(name=name, positives=positives, negatives=negatives, features=model_hidden_features)


SAYS_IT_IS_CLOSED = _surface(
    "says_it_is_closed",
    ("This door is locked.", "You need a key to open this.", "Requires level 10.", "Collect 30 stars to enter.",
     "Upgrade to Pro to use this feature.", "You can't go this way yet.", "Not enough coins.",
     "This area is closed until you finish the tutorial.", "Sign in to continue."),
    ("Press space to jump.", "You found a key!", "Level complete!", "Click Start to play.", "Score: 300",
     "Welcome back!", "Drag the piece into place.", "The door opens."),
)
SAYS_SHE_GAINED = _surface(
    "says_she_gained",
    ("You found a key!", "Got the hookshot!", "New ability unlocked: double jump.", "You now have 30 stars.",
     "Item acquired: rusty sword.", "You learned Fireball.", "Pro features unlocked.", "You received a map."),
    ("You need a key.", "Press X to jump.", "Game over.", "Level 1", "This door is locked.", "Click to continue.",
     "Choose a character."),
)


def _decides(surface: Any, floor: re.Pattern[str], sentence: str) -> bool:
    try:
        decided = surface.decide_without_waiting(sentence)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("The sentence is judged by the floor: her surface could not decide it (%s).", exc)
        decided = None
    return bool(floor.search(sentence)) if decided is None else bool(decided)


def _named(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(str(text or "").lower()) if w not in _PLAIN}


@dataclass
class Lock:
    """A way that would not open: what it said, where it was (the label she tried, or the thing it names), what it wants,
    and whether what it wants has turned up."""

    said: str
    at: str
    wants: str
    open_to_try: bool = False
    #: Whether it was seen rather than said: a place she made for and got no nearer to (a ledge too high, a gap too
    #: wide). It names nothing it wants, so whatever she gains next may open it.
    seen: bool = False


@dataclass
class Locks:
    """The locks met in one place, and what she has gained there."""

    met: list[Lock] = field(default_factory=list)
    gained: list[str] = field(default_factory=list)

    def heard(self, sentences: Sequence[str], labels: Sequence[str], last_act: str = "") -> list[str]:
        """A screen's sentences: a lock kept, a gain matched to the locks it opens. What to say of it."""
        news: list[str] = []
        closed: list[str] = []
        for sentence in (" ".join(str(s).split()) for s in sentences):
            if len(sentence.split()) < 2:
                continue
            if _decides(SAYS_SHE_GAINED, _GAINED, sentence):
                news += self._gained(sentence)
            elif _decides(SAYS_IT_IS_CLOSED, _CLOSED, sentence):
                closed.append(sentence)
        # What one screen says of a way that will not open ("This door is locked. You need the brass key.") is one lock.
        return news + (self._closed(" ".join(closed), labels, last_act) if closed else [])

    def _closed(self, sentence: str, labels: Sequence[str], last_act: str) -> list[str]:
        wanted = _WANTS.search(sentence)
        wants = wanted.group(1).strip() if wanted else ""
        named = [label for label in labels if label and _named(label) & _named(sentence) - _named(wants)]
        at = named[0] if named else _label(last_act)
        if not at or any(lock.at == at and lock.wants == wants for lock in self.met):
            return []
        self.met.append(Lock(said=sentence, at=at, wants=wants))
        del self.met[:-MOST_LOCKS]
        logger.info("a way that would not open: %r at %r, wanting %r", sentence, at, wants)
        return [f"{at} won't open yet" + (f": it wants {_lower(wants)}." if wants else ".") + " I'll come back to it."]

    def _gained(self, sentence: str) -> list[str]:
        self.gained.append(sentence)
        del self.gained[:-MOST_LOCKS]
        news = []
        for lock in self.met:
            if not lock.open_to_try and lock.wants and _named(lock.wants) & _named(sentence):
                lock.open_to_try = True
                news.append(f"That's what {lock.at} wanted. Going back to it.")
            elif not lock.open_to_try and lock.seen:
                lock.open_to_try = True
                news.append(f"With that, {lock.at} may be in reach now. Trying it again.")
        return news

    def saw_out_of_reach(self, at: str) -> list[str]:
        """A place she made for and got no nearer to, kept as a lock that names nothing it wants; what to say of it."""
        if any(lock.at == at for lock in self.met):
            return []
        # Said of the first such place, not of each: a place is a position on the screen, and in a busy game every
        # few seconds brings another (LIVE 2026-10-11 "I can't get to the place at 96% across and 54% down yet").
        first = not any(lock.seen for lock in self.met)
        self.met.append(Lock(said="I made for it and got no nearer", at=at, wants="", seen=True))
        del self.met[:-MOST_LOCKS]
        return ["There are places I can't get to yet. I'll come back to them when something changes."] if first else []

    def reopened_seen(self) -> list[Lock]:
        """The seen locks a gain has opened, taken off the list as they are handed back to be tried."""
        opened = [lock for lock in self.met if lock.seen and lock.open_to_try]
        self.met = [lock for lock in self.met if lock not in opened]
        return opened

    def open_to_try(self) -> list[Lock]:
        return [lock for lock in self.met if lock.open_to_try]

    def tried(self, act: str, changed: bool) -> None:
        """An act on a lock that was open to try: gone back to, and done with once it answered."""
        if changed:
            self.met = [lock for lock in self.met if not (lock.open_to_try and lock.at == _label(act))]

    def for_thinking(self) -> str:
        if not self.met:
            return ""
        return "Ways that would not open yet: " + "; ".join(
            f"{lock.at} (wants {lock.wants})" if lock.wants else lock.at for lock in self.met[-6:]) + (
            ". Open to try now: " + ", ".join(lock.at for lock in self.open_to_try()) if self.open_to_try() else "")


def heard_on_a_screen(guide: Any, says: str, labels: Sequence[str], last_act: str = "") -> list[str]:
    """A screen's words taken into the locks of the place the guide is to; what to say of them."""
    locks = getattr(guide, "locks", None)
    if locks is None:
        return []
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", str(says or "")) if s.strip()]
    return locks.heard(sentences, list(labels), last_act)


def _label(act: str) -> str:
    found = re.match(r'^click "(.*)"$', str(act or ""))
    return found.group(1) if found else str(act or "")


def _lower(text: str) -> str:
    from core.language.words_of_the_language import as_said_inside

    return as_said_inside(str(text).rstrip("."))


def locks_as_memory(locks: Any) -> dict[str, Any]:
    """The ways that would not open and what each wanted, and what she gained, kept with the guide to the place."""
    if locks is None or not (locks.met or locks.gained):
        return {}
    return {"met": [[lock.said, lock.at, lock.wants, lock.open_to_try, lock.seen] for lock in locks.met[-12:]],
            "gained": list(locks.gained[-12:])}


def locks_from_memory(held: Any) -> Locks:
    locks = Locks()
    if not isinstance(held, dict):
        return locks
    for kept in held.get("met") or []:
        if isinstance(kept, list) and len(kept) >= 3:
            locks.met.append(Lock(str(kept[0]), str(kept[1]), str(kept[2]), bool(kept[3]) if len(kept) > 3 else False,
                                  bool(kept[4]) if len(kept) > 4 else False))
    locks.gained = [str(g) for g in held.get("gained") or []]
    return locks

