"""A map of the screens of a thing she uses, and where each act on each of them led.

A person who has been through a game's menus once goes through them the second
time without reading them: Start, then the first level, then play. Lost, they
press Play Again, not Main Menu. What they remember is not the game's rules but
its screens and what each control on them did: this one led on, that one led
back, this one did nothing. The same memory takes a person through any program
or site they have used before.

Each screen is known by the words on it, so a screen read a little differently
the next time is the same screen (two readings sharing most of their words).
After every act she records where it led from where she was: further in, by
the order she first came to the screens this sitting, or to where she has
played (on); back to a screen she came to earlier (back); the same screen
changed (moved); or nothing at all. Kept with the rest of what
she learns about a place, it is there the next time, and it orders what she
tries on a screen she knows: what led on first, what did nothing last.

It is also how she explores a place she does not know, as the explorers that do best at games no one has seen before
do it (state-graph exploration, ARC-AGI-3, 2025): each screen a node, each act on it an edge to the screen it led
to. On a screen she tries what she has not tried there before what she has, and what did nothing there last. Where
everything on a screen has been tried, she takes the act that leads, by the edges she knows, nearest to a screen
that still has something untried. LIVE 2026-10-10 she clicked and carried on one game's screens "to see what it
does", the same acts again and again, while a tab of its device library was never opened.

Nothing here knows what a menu, a level or a game is.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhereThingsLead", "screen_words"]

#: How much two readings must share, of the words either has, to be one screen.
SAME_SCREEN = 0.5

#: The fewest words a screen is told apart by. Fewer, and it is "a screen with
#: nothing to read", one place however many such screens there are.
FEWEST_WORDS = 2

#: The screens kept for one place, the most visited first.
MOST_SCREENS = 60

#: Where a screen with nothing to read is kept.
WORDLESS = -1


def screen_words(labels: Sequence[str], says: str = "") -> frozenset[str]:
    """The words a screen is known by: those of its labels and of what it says, without numbers or readouts. A screen
    with too few words to tell it by is known by where the shapes on it stand ("at 90 across 90 down"), as the explorers
    of wordless games tell one state from another by how it looks."""
    words: set[str] = set()
    places: set[str] = set()
    for text in [*labels, says]:
        text = str(text or "")
        if text.startswith(('click "the shape at', 'click "the one that stands out at')):
            # Named by where it is, which is not a word on the screen; kept, to the nearest tenth, for a wordless one.
            at = re.findall(r"(\d+)% (?:across|down)", text)
            if len(at) == 2:
                places.add(f"at {round(int(at[0]), -1)} across {round(int(at[1]), -1)} down")
            continue
        words.update(w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w != "click")
    return frozenset(words) if len(words) >= FEWEST_WORDS or len(places) < FEWEST_WORDS else frozenset(places)


@dataclass
class WhereThingsLead:
    """The screens of one place and where each act on them led, across sittings."""

    #: The words of each screen, by its number.
    screens: list[frozenset[str]] = field(default_factory=list)
    #: "screen|act" -> counts of how the act went from that screen: on, back, moved, nothing.
    went: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Screens where her keys moved things: where the playing is.
    played: set[int] = field(default_factory=set)
    #: "screen|act" -> the screen it last led to; and what each screen offered to be done on it.
    to: dict[str, int] = field(default_factory=dict)
    offered: dict[int, frozenset[str]] = field(default_factory=dict)
    #: This sitting: the screens in the order she first came to them, and where she is.
    _been: list[int] = field(default_factory=list)
    _here: int | None = None
    _before: int | None = None

    def which(self, words: frozenset[str]) -> int:
        """The number of the screen these words are, a new one if no screen she knows is it."""
        if len(words) < FEWEST_WORDS:
            return WORDLESS
        best, share = None, 0.0
        for number, known in enumerate(self.screens):
            shared = len(words & known) / max(1, len(words | known))
            if shared > share:
                best, share = number, shared
        if best is not None and share >= SAME_SCREEN:
            # Read again, it is known by the words both readings had and any new ones.
            self.screens[best] = frozenset(self.screens[best] | words) if len(self.screens[best]) < 40 else self.screens[best]
            return best
        self.screens.append(words)
        return len(self.screens) - 1

    def looked(self, labels: Sequence[str], says: str = "") -> None:
        """She looked at a screen: where she is now, and where she was before it."""
        here = self.which(screen_words(labels, says))
        self._before, self._here = self._here, here
        if here not in self._been:
            self._been.append(here)

    def acted(self, act: str, changed: bool) -> None:
        """The act she took on the screen before this look, and whether anything answered it."""
        if self._before is None or self._here is None or not act:
            return
        before, after = self._before, self._here
        counts = self.went.setdefault(f"{before}|{act}", {})
        if changed and after != before:
            self.to[f"{before}|{act}"] = after
        if not changed:
            how = "nothing"
        elif after == before:
            how = "moved"
            if not str(act).startswith("click "):
                self.played.add(before)
        elif after in self.played or self._been.index(after) > self._been.index(before):
            # Further in than where she was, by the order she first came to them.
            how = "on"
        else:
            how = "back"
        counts[how] = counts.get(how, 0) + 1
        self._before = None  # one act per look: a second report of the same act is not a second act

    def how_it_led(self, act: str) -> float:
        """How much more ``act`` is worth trying on this screen for where it led before: above one led on, below one did nothing."""
        if self._here is None:
            return 1.0
        counts = self.went.get(f"{self._here}|{act}")
        toward = self._toward_the_untried(act)
        if counts is None:
            # Not done on this screen yet: what it did on the others. The arrow
            # in a comic's corner that turned the first page turns the second
            # (LIVE-like 2026-10-05 it was taken for tried and never pressed
            # again, and she pressed keys at the last page for six minutes).
            elsewhere = [c for key, c in self.went.items() if key.split("|", 1)[1] == act]
            on = sum(c.get("on", 0) for c in elsewhere)
            back = sum(c.get("back", 0) for c in elsewhere)
            return 1.0 + 0.75 * on / (on + back + 1) if on > back else 1.0
        on, back, nothing, moved = (counts.get(k, 0) for k in ("on", "back", "nothing", "moved"))
        if on > back:
            return 1.0 + 1.5 * on / (on + back + nothing + 1)
        if toward:
            # Everything here has been tried: the way to where something has not been is worth taking.
            return toward
        if back > on:
            # What took her back the way she came is not the way on: LIVE 2026-10-07 a level select's corner
            # icon took her to the title three times, and the level beside it was never clicked.
            return 0.6 ** (back - on)
        if nothing >= 2 and not (on or moved):
            return 0.5
        # Tried here and nothing came of it, or only this screen changed: below what has not been tried here.
        return 0.8 if nothing and not (on or moved) else 0.9

    def taken_here_to_no_end(self, act: str) -> int:
        """How many times ``act`` has been taken on this screen and led nowhere: nothing answered, or the screen stayed itself."""
        if self._here is None:
            return 0
        counts = self.went.get(f"{self._here}|{act}") or {}
        return 0 if counts.get("on", 0) > counts.get("back", 0) else counts.get("nothing", 0) + counts.get("moved", 0)

    def in_order(self, acts: Sequence[str]) -> tuple[str, ...]:
        """``acts``, what led on from this screen first, then what has not been tried here, and what did nothing here
        last, otherwise as they were (the order they were offered in says which stand out)."""
        if self._here is not None and acts:
            self.offered[self._here] = frozenset(acts) | self.offered.get(self._here, frozenset())
        return tuple(sorted(acts, key=lambda act: -self.how_it_led(act)))

    def untried_on(self, screen: int) -> set[str]:
        """What a screen offered that has not been done there."""
        return {act for act in self.offered.get(screen, ()) if f"{screen}|{act}" not in self.went}

    def _toward_the_untried(self, act: str) -> float:
        """Where nothing on this screen is left untried, how much ``act`` leads toward a screen with something untried
        on it, by the screens it is known to lead to: above one the nearer that is; 0 where it leads nowhere such."""
        here = self._here
        if here is None or here == WORDLESS or self.untried_on(here):
            return 0.0
        first = self.to.get(f"{here}|{act}")
        if first is None or first == here:
            return 0.0
        seen, frontier, steps = {here, first}, [first], 0
        while frontier and steps < 6:
            if any(self.untried_on(screen) for screen in frontier):
                return 1.0 + 0.6 / (1 + steps)
            steps += 1
            frontier = [nxt for screen in frontier for key, nxt in self.to.items()
                        if key.split("|", 1)[0] == str(screen) and nxt not in seen and not seen.add(nxt)]
        return 0.0

    def says(self) -> str:
        """What the map holds, for whoever has to answer for it."""
        led_on = sum(1 for counts in self.went.values() if counts.get("on", 0) > counts.get("back", 0))
        return f"{len(self.screens)} screens known here, {led_on} ways on found"

    def as_memory(self) -> dict[str, Any]:
        kept = sorted(range(len(self.screens)), key=lambda n: -sum(
            sum(c.values()) for key, c in self.went.items() if key.startswith(f"{n}|")))[:MOST_SCREENS]
        renumber = {old: new for new, old in enumerate(kept)}
        renumber[WORDLESS] = WORDLESS
        went = {}
        for key, counts in self.went.items():
            screen, act = key.split("|", 1)
            if int(screen) in renumber:
                went[f"{renumber[int(screen)]}|{act}"] = dict(counts)
        to = {f"{renumber[int(key.split('|', 1)[0])]}|{key.split('|', 1)[1]}": renumber[n] for key, n in self.to.items()
              if int(key.split("|", 1)[0]) in renumber and n in renumber}
        return {
            "screens": [sorted(self.screens[n]) for n in kept],
            "went": went,
            "played": sorted(renumber[n] for n in self.played if n in renumber),
            "to": to,
            "offered": {str(renumber[n]): sorted(acts)[:40] for n, acts in self.offered.items() if n in renumber},
        }

    @classmethod
    def from_memory(cls, held: Any) -> WhereThingsLead:
        if not isinstance(held, dict):
            return cls()
        try:
            return cls(
                screens=[frozenset(str(w) for w in words) for words in held.get("screens") or []],
                went={str(k): {str(h): int(n) for h, n in v.items()} for k, v in (held.get("went") or {}).items()},
                played={int(n) for n in held.get("played") or []},
                to={str(k): int(v) for k, v in (held.get("to") or {}).items()},
                offered={int(k): frozenset(str(a) for a in v) for k, v in (held.get("offered") or {}).items()},
            )
        except (TypeError, ValueError, AttributeError):
            return cls()
