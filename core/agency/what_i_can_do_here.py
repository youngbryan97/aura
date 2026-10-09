"""Which of her actions do anything here, found out rather than declared.

A solver written for one thing is handed its action set: four moves, named in
the source. She was handed hers the same way — a tuple of four arrow keys in
core/runtime/watched_goal.py — and that is the last large thing about an
unfamiliar world that somebody else was still establishing for her. Everything
else she now works out: which window she is in, which part of it answers, how
it moves when she pushes it, what changes on its own, and what a good situation
looks like.

So the action set is a hypothesis like every other one here. She starts with
whatever she was told, tries the rest when what she was told is not working,
and keeps what answers. A key that never changes anything is not one of her
actions in this world, whoever wrote it down.

One line is not crossed. Only inputs that commit to nothing are tried without
knowing what they will do. An arrow moves a view or a piece and can be undone
by moving back; Return and Space activate whatever has focus, which may be a
Send, a Buy or a Delete. She may find out what moves things. She may not find
out what a button does by pressing it.

Except inside the thing she was sent to play. A game drawn on a page is a world
of its own, and clicking in it is how it is played: nothing in there is a Send
or a Buy. So what can be clicked there joins her moves, but only there — the
caller hands over only what lies inside the drawing the page said it is making
— and only when what she was told is not working, the same as any other move
nobody named.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from core.agency.taking_and_using import TakingAndUsing, what_is_used
from core.agency.where_things_lead import WhereThingsLead

__all__ = [
    "COMMITS_TO_NOTHING",
    "THE_PICTURE",
    "ENOUGH_TO_JUDGE",
    "WhatWorksHere",
    "a_click_on",
    "what_is_clicked",
    "worth_trying",
]

logger = logging.getLogger("Aura.WhatICanDoHere")

#: Inputs she may try without knowing what they will do.
#:
#: Every one of these moves something — a view, a selection, a piece — and
#: every one is undone by moving back. Nothing here activates a control.
COMMITS_TO_NOTHING: tuple[str, ...] = ("up", "down", "left", "right")

#: Inputs that act on whatever has focus. Never tried to find out what they
#: do: what they do is press whatever button is under them, and that is a
#: decision rather than an experiment.
COMMITS_TO_SOMETHING: tuple[str, ...] = ("return", "enter", "space", "tab", "delete")

#: How many times an input has to have done nothing before it is not one of
#: her actions here. Once is a bad moment — a board can refuse a direction it
#: will accept two moves later. Several times running is a fact about the
#: world rather than about the moment.
ENOUGH_TO_JUDGE = 4


def a_click_on(label: str) -> str:
    """The move that clicks a thing, named by what is written on it."""
    return f'click "{" ".join(str(label or "").split())}"'


def _goes_on(move: str) -> bool:
    from core.language.a_way_on import how_much_it_leads_on

    label = what_is_clicked(move)
    return bool(label) and how_much_it_leads_on(label) > 1.0


def _not_read_only(move: str) -> bool:
    """Whether a click may do something: not writing that is there to be read (a score, a caption, a measure)."""
    from core.language.a_way_on import how_much_it_leads_on

    label = what_is_clicked(move)
    return label is None or how_much_it_leads_on(label) > 0.7


def what_is_clicked(move: str) -> str | None:
    """What a click move clicks, or None when the move is not a click."""
    name = str(move or "").strip()
    if name.startswith('click "') and name.endswith('"') and len(name) > len('click ""'):
        return name[len('click "'):-1]
    return None


#: How a screen asks for a key: "press space", "hit the space bar to start",
#: "press any key", "press enter to continue", "press P to play".
_ASKING = re.compile(
    r"\b(?:press|hit|push|tap)\s+(?:the\s+)?(space\s*bar|spacebar|space|enter|return|any\s+key|[a-z])\b"
    r"(?:\s+(?:key\s+)?to\s+(start|play|begin|continue|go|launch|serve|fire|jump|shoot))?",
    re.IGNORECASE,
)


def keys_a_screen_asks_for(words: str) -> tuple[str, ...]:
    """The keys a screen's words ask to be pressed, in the order it asks.

    A single letter counts only with what it is for beside it ("press P to
    play"), because "press a button" asks for no key called A.
    """
    import unicodedata

    # Read as letters: text recognition gives "SPAÇE" for SPACE on a game's
    # title (offline 2026-10-03, Pong), and the accent is noise, not a word.
    plain = unicodedata.normalize("NFKD", str(words or "")).encode("ascii", "ignore").decode("ascii")
    asked: list[str] = []
    for found in _ASKING.finditer(" ".join(plain.split())):
        named, purpose = found.group(1).lower(), found.group(2)
        if named.replace(" ", "") in ("space", "spacebar") or named.startswith("any"):
            key = "space"
        elif named in ("enter", "return"):
            key = "return"
        elif purpose:
            key = named
        else:
            continue
        if key not in asked:
            asked.append(key)
    return tuple(asked)


#: A click on the picture itself, for a screen that asks for a click and names nothing to click.
THE_PICTURE = "the middle of the picture"

#: How a screen's words ask for a click on nothing in particular: "Click your mouse button to start the hamster in
#: motion", "click anywhere to continue", "click to start". Clicked in the middle of what she is looking at.
_CLICK_ANYWHERE = re.compile(
    r"\b(?:click|tap)\s+(?:(?:your|the|a|any|left)\s+)*(?:mouse(?:\s+button)?|anywhere|screen)\b"
    r"|\b(?:click|tap)\s+to\s+(?:start|begin|play|continue|launch|go|shoot|throw|jump|fire)\b",
    re.IGNORECASE,
)

#: How a screen's words ask for a click: "click on the GENERATE button", "press PLAY to start", "select Easy".
_ASKING_TO_CLICK = re.compile(r"\b(?:click|press|hit|tap|select|choose)\s+(?:on\s+)?(?:the\s+)?([a-z0-9]+)")


def clicks_a_screen_asks_for(words: str, clickable: Sequence[str]) -> tuple[str, ...]:
    """The labels on a screen that its own words ask to be clicked, in the order they are on it.

    A person reads "click on the GENERATE button to see your character" and
    clicks GENERATE. A label cut off at the edge of the picture, or read a
    letter or two short, is the word it begins ("GENE" for GENERATE): four
    letters at least, so "on" or "go" is never taken for another word.
    """
    said = " ".join(re.findall(r"[a-z0-9]+", str(words or "").lower()))
    named = {found.group(1) for found in _ASKING_TO_CLICK.finditer(said)}
    asked = []
    for move in clickable:
        first = re.findall(r"[a-z0-9]+", (what_is_clicked(move) or "").lower())[:1]
        if not first or len(first[0]) < 3:
            continue
        word = first[0]
        if any(word == name or (min(len(word), len(name)) >= 4 and (name.startswith(word) or word.startswith(name))) for name in named):
            asked.append(move)
    if not asked and _CLICK_ANYWHERE.search(said):
        asked.append(a_click_on(THE_PICTURE))
    return tuple(asked)


def worth_trying(told: Sequence[str] = ()) -> tuple[str, ...]:
    """Everything she could try here, hers first and the rest after.

    What she was told comes first because somebody usually knows, and the
    others follow because sometimes nobody does.
    """
    named = [str(key or "").strip().lower() for key in told]
    ordered = [key for key in named if key in COMMITS_TO_NOTHING]
    ordered += [key for key in COMMITS_TO_NOTHING if key not in ordered]
    return tuple(ordered)


@dataclass
class WhatWorksHere(TakingAndUsing):
    """Which inputs have done anything, and which have never done anything; and what she has taken to use
    (core/agency/taking_and_using.py)."""

    #: What she was told her actions were, if anything.
    told: tuple[str, ...] = ()
    did_something: dict[str, int] = field(default_factory=dict)
    did_nothing: dict[str, int] = field(default_factory=dict)
    #: Said once, when what she was told turns out to be wrong.
    said_it_differs: bool = False
    #: What can be clicked on the screen in front of her now, inside the thing
    #: she was sent to play. Set by `looked_at` each time she looks.
    on_screen: tuple[str, ...] = ()
    #: Keys the screen in front of her asks for, in its own words ("Press
    #: SPACE to play"). Pressing one is doing what it says, not finding out.
    asked_for: tuple[str, ...] = ()
    #: Labels on it that its words ask to be clicked ("click on the GENERATE button").
    clicks_asked_for: tuple[str, ...] = ()
    #: Whether its words name the mouse and no key ("Click your mouse button to start"): it is played by clicking.
    pointer_only: bool = False
    #: What the look before this one could click, to hold the next look to.
    seen_before: tuple[str, ...] = ()
    #: What she has done since the screen last answered anything, all of it to
    #: no effect: the screen in front of her has not been moved by any of it.
    quiet_since: set[str] = field(default_factory=set)
    #: The screens of this place and where each act on them led, across sittings.
    leads: WhereThingsLead = field(default_factory=WhereThingsLead)

    def asked_for_by(self, words: str, clickable: Sequence[str] = (), drawn: Sequence[str] = ()) -> None:
        """Keys the screen asks her to press, in its words or drawn as keys on it, and labels its words ask her to click."""
        from core.agency.playing_as_it_happens import controls_named_in

        self.asked_for = tuple(dict.fromkeys([*keys_a_screen_asks_for(words), *drawn]))
        self.clicks_asked_for = clicks_a_screen_asks_for(words, clickable)
        keys, pointer = controls_named_in(words, keys_without_words=())
        # A screen that names no controls keeps what the game's own screens said before it: LIVE 2026-10-08 the rules
        # said "click your mouse button to start", and on the game's wordless screen after them she pressed arrows.
        if keys or pointer:
            self.pointer_only = pointer and not keys

    def looked_at(self, clickable: Sequence[str], says: str = "") -> None:
        """What she can click now: the writing that was there at the last look too.

        A control stays where it is; a readout changes. LIVE 2026-10-03 04:49,
        with the game's clock running, every look read it differently ("TImE
        01:50", "TTE 0152", "TITE 040"), each reading was a control she had
        never tried, and she clicked the clock again and again.
        """
        now = tuple(clickable)
        self.noticed_taking(now)
        self.on_screen = tuple(label for label in now if label in self.seen_before)
        self.seen_before = now
        self.leads.looked(now, says)

    # ── finding out ──────────────────────────────────────────────────────

    def tried(self, key: str, changed: bool) -> None:
        """One input, and whether the world answered it."""
        name = str(key or "").strip()
        name = name if what_is_clicked(name) is not None or what_is_used(name) is not None else name.lower()
        if not name:
            return
        if what_is_clicked(name) is not None:
            self.clicked_on(name, self.seen_before, changed)
        self.leads.acted(name, changed)
        if changed:
            self.quiet_since.clear()
            self.did_something[name] = self.did_something.get(name, 0) + 1
            self.did_nothing.pop(name, None)
            # A click that changed the screen may have changed what works:
            # keys that did nothing on a title screen are how the game itself
            # is played (LIVE 2026-10-03 04:48, and never pressed again).
            if what_is_clicked(name) is not None:
                self.did_nothing.clear()
        else:
            self.quiet_since.add(name)
            self.did_nothing[name] = self.did_nothing.get(name, 0) + 1

    def keys_do_nothing_here(self) -> bool:
        """Whether every key she was told has been pressed at this screen since it last answered, and none moved it.

        A key that worked on another screen is not dead, and while it is not
        the labels that do not read as a way on were never offered. LIVE-like
        2026-10-05, a game's level select ("Strike Em Out", "Backyard
        Beatdown"): she pressed up twenty times, because up had worked in the
        game before, and never clicked a level.
        """
        pressed = self.quiet_since & set(self.told)
        return bool(self.told) and len(pressed) >= min(len(self.told), 3)

    # ── using it ─────────────────────────────────────────────────────────

    def dead(self) -> tuple[str, ...]:
        """Inputs that have done nothing, every time, enough times to say so."""
        return tuple(
            key
            for key, times in sorted(self.did_nothing.items())
            if times >= ENOUGH_TO_JUDGE and key not in self.did_something
        )

    def works(self) -> tuple[str, ...]:
        """Inputs that have done something at least once."""
        return tuple(sorted(self.did_something))

    def untried(self) -> tuple[str, ...]:
        """Inputs she could try and has not."""
        seen = set(self.did_something) | set(self.did_nothing)
        return tuple(key for key in worth_trying(self.told) if key not in seen)

    def available(self) -> tuple[str, ...]:
        """What to offer her now, what led on from this screen before first (`core.agency.where_things_lead`)."""
        return self.leads.in_order(self._available()) + self.uses(self.on_screen)

    def _available(self) -> tuple[str, ...]:
        """What to offer her now.

        What she was told, minus anything that has proved inert, plus
        anything else worth trying while some of what she was told is not
        working. A world where the named keys do the job never widens.
        """
        dead = set(self.dead())
        # What the screen asks for comes first and is never written off: the
        # screen is still asking, so it has not been done yet. LIVE 2026-10-03
        # 20:23, offline on a Pong whose title said "Press SPACE to play", she
        # made 199 moves with the arrow keys and never pressed space.
        asked = tuple(key for key in self.asked_for if key not in self.told)
        asked += tuple(click for click in self.clicks_asked_for
                       if (click in self.on_screen or what_is_clicked(click) == THE_PICTURE) and click not in dead)
        told = tuple(key for key in self.told if key not in dead)
        # A label that goes on (Play, Next, Easy) is offered whatever the keys
        # are doing: on a menu that moves on its own, keys look as if they work
        # and the labels never joined. LIVE-like 2026-10-05 she pressed left
        # forty times on a game's "Easy / Hard" screen.
        # Where her keys do nothing, the other labels are tried too, but not what is there to be read: LIVE
        # 2026-10-07 she clicked "score" and "Meter:" off a game's scoreboard to see what they did.
        goes_on = tuple(click for click in self.on_screen if click not in dead
                        and (_goes_on(click) or self.keys_do_nothing_here() and _not_read_only(click)))
        if asked:
            # Named things before a click on nothing in particular, and no keys where the screen names only the mouse.
            anywhere = tuple(a for a in asked if what_is_clicked(a) == THE_PICTURE)
            named = tuple(a for a in asked if a not in anywhere)
            return tuple(dict.fromkeys(named + (() if self.pointer_only else told) + goes_on + anywhere))
        clicks = tuple(click for click in self.on_screen if click not in dead and _not_read_only(click))
        if self.pointer_only:
            # The picture itself too: a game played with the mouse is mostly clicked where nothing is written.
            clicks += tuple(c for c in (a_click_on(THE_PICTURE),) if c not in dead and c not in clicks)
            # A screen whose words name the mouse and no key is done by clicking: LIVE 2026-10-07 "Click your mouse
            # button to start the hamster in motion" and she pressed each arrow to see what it did.
            return clicks
        # Only what she was told can be shown wrong about what she was told.
        # A key nobody named that never did anything says nothing about the
        # ones they did: a remembered dead "down" took the caller's Tab and
        # Return away before either had been pressed once.
        if told and len(told) == len(self.told):
            return told + goes_on
        wider = list(told) + [
            key for key in worth_trying(self.told) if key not in dead and key not in told
        ]
        wider += [click for click in self.on_screen if click not in dead and click not in wider and _not_read_only(click)]
        if wider and set(wider) != set(self.told) and not self.said_it_differs:
            self.said_it_differs = True
            logger.info(
                "what she was told her moves were (%s) is not what works here (%s)",
                ", ".join(self.told) or "nothing",
                ", ".join(wider),
            )
        return tuple(wider)

    def still_finding_out(self) -> bool:
        """Whether anything is left to try."""
        return bool(self.untried())

    def says(self) -> str:
        """What she has found out, for whoever has to answer for it."""
        works = self.works()
        dead = self.dead()
        if not works and not dead:
            return "which of my moves do anything here is not worked out yet"
        said = f"these do something here: {', '.join(works) or 'none so far'}"
        if dead:
            said = f"{said}; these never do: {', '.join(dead)}"
        return said

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, object]:
        return {
            "told": list(self.told),
            "did_something": dict(self.did_something),
            "did_nothing": dict(self.did_nothing),
            "leads": self.leads.as_memory(),
        }

    @classmethod
    def from_memory(cls, held: object, told: Sequence[str] = ()) -> "WhatWorksHere":
        """What worked here last time, as a starting point rather than a fact."""
        named = tuple(str(key or "").strip().lower() for key in told)
        if not isinstance(held, dict):
            return cls(told=named)

        # A label works on its own screen: "Play" does nothing on the
        # instructions page and everything on the title. Kept across sessions,
        # LIVE-like 2026-10-05, "click Play never does anything" was the first
        # thing she knew on a game's title screen, and she never pressed it.
        def counts(value: object) -> dict[str, int]:
            if not isinstance(value, dict):
                return {}
            return {
                str(key): int(times)
                for key, times in value.items()
                if isinstance(times, (int, float)) and what_is_clicked(str(key)) is None
            }

        return cls(
            told=named or tuple(str(key) for key in (held.get("told") or ())),
            did_something=counts(held.get("did_something")),
            did_nothing=counts(held.get("did_nothing")),
            leads=WhereThingsLead.from_memory(held.get("leads")),
        )
