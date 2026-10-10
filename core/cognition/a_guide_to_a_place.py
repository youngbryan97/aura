"""A guide to the place she is in: how it is worked, what it wants, what is in it, kept up to date as she goes.

A person new to a game, a program or a site does not work it out from nothing.
They read what it says, what the box says, what a friend or the web says, and
they bring what they know of games and programs in general; and they keep a
picture in their head of how this one works that changes as it does: the third
level adds lasers, the shield wore off, the arrows now steer a boat. The
picture tells them how to operate; it does not play for them.

This is that picture. It is built from every source there is, each kept with
where it came from:

- what the place shows and tells her (its instructions, hints, labels, the
  words between stages): the most trusted, and what keeps the guide current;
- what its page says of it;
- its program, where it can be had (core/perception/reading_a_program.py): only
  how it is worked (keys, pointer, controls, its instructions), never what
  play would show her later (answers, secrets, what is ahead, cheats);
- counsel from her own model, her Wikipedia and the web
  (core/cognition/taking_stock.py);
- the mechanics she knows of every place (core/agency/mechanics_she_knows.py),
  which say what each thing she finds means and how it is played.

Words outrank code: a key a program listens for that no word names is offered
only where no word names any. What the place says now outranks what it said:
a mechanic it says is gone is gone, one it says is new is added, and each such
change is kept with when it came, so she can say what changed and reason with
the place as it is now. It is informational: it says how; she decides what.

Nothing here knows a place.
"""
from __future__ import annotations

import contextvars
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.cognition.what_she_notices import Notebook
from core.cognition.what_things_are import WhatSheSees
from core.cognition.what_this_place_is import WhatThisPlaceIs

__all__ = ["MODEL", "THE_GUIDE", "Guide", "carry_over", "confirmed_by", "guide_for", "guide_of", "heard_in_play", "the_guide"]

#: Sources, most trusted first. What her own model knows of a place is the least sure of what is said of it: it may be
#: of places like it, and it is never taken for how the place is worked (its controls), only for what it is for.
SCREEN, TOLD, PAGE, COUNSEL, MODEL, PROGRAM = ("the screen", "its instructions", "its page", "what I looked up",
                                               "what I know of it", "its program")
_TRUST = {SCREEN: 5, TOLD: 4, PAGE: 3, COUNSEL: 2, MODEL: 1.5, PROGRAM: 1}

#: Sentences that say how a round is won, how it is lost, and how to do well: a manual's other sections.
_A_WIN = re.compile(r"\b(?:to win|you win|wins? (?:the|a|by|if|when)|beat (?:the|each|every|all)|complete (?:the|each|all)|"
                    r"clear (?:the|each|all)|land (?:safely|softly|gently)|safe landing|victory|until you)\b", re.I)
_A_LOSS = re.compile(r"\b(?:you lose|lose (?:a|one|all|the|your)|game over|crash\w*|you die|dies|killed|fail\w*|run out|"
                     r"runs out|don'?t let|if you (?:touch|hit|fall|miss|crash|run)|too (?:fast|hard|slow))\b", re.I)
_A_TIP = re.compile(r"\b(?:tip|trick|best (?:way|to)|make sure|remember|be careful|slowly|gently|first|strategy|it helps|"
                    r"the key is|instead of|rather than|don'?t (?:waste|rush))\b", re.I)

#: A sentence that sets a goal: what to do to win, or what not to let happen.
_A_GOAL = re.compile(r"\b(?:your (?:goal|mission|job|task) is|the (?:goal|object|aim) (?:of the game )?is|try to|"
                     r"you (?:need|have|must) to|help \w+(?: \w+)? (?:to )?(?:find|get|escape|save|rescue|stop|reach|"
                     r"collect|enjoy|win)|get (?:all|as many|to the)|reach|collect|"
                     r"guide|rescue|save|escape|survive|don'?t let|before (?:time|the timer)|as (?:many|far|long) as)\b",
                     re.I)
#: A heading a sentence was read run into ("INSTRUCTIONS Use the arrow keys", "Winning Objective: Land safely"): the
#: section's name, not what it says.
_A_HEADING = re.compile(r"^\s*(?:(?:[A-Z]{3,}\s+){1,3}(?=[A-Z][a-z])|(?:[A-Z][A-Za-z]*\s+){0,2}(?i:objective|goal|controls?|"
                        r"instructions|how to play|rules|tips?|strategy|hint)\s*:\s*)")
#: A short label shown with a value: what is read off the screen, not pressed.
_A_READOUT = re.compile(r"^\W*([A-Za-z][A-Za-z .'-]{1,24}?)\s*[:=]?\s*(?:[\d.,/%]+|\.\.\.|…)?\W*$")
#: Writing that says which stage she is on.
_A_STAGE = re.compile(r"\b(level|stage|wave|round|mission|world|chapter)\s*:?\s*(\d{1,3})\b", re.I)
#: A power that lasts a while: "invincible for 10 seconds".
_FOR_A_WHILE = re.compile(r"\bfor (\d{1,3}) (?:seconds|secs?)\b", re.I)
#: Buttons and keys that are not part of play: pressed in play, they stop it, silence it, or leave it.
_NOT_FOR_PLAY = ("pause", "mute", "unmute", "sound", "music", "menu", "quit", "exit", "credits", "more games", "options",
                 "settings", "help", "fullscreen", "level select", "home")
#: Keys a program's handlers listen for that work its menus, fields and windows, not play.
_NOT_PLAYED_BY = ("escape", "tab", "control", "shift", "alt", "home", "end", "pageup", "pagedown", "delete", "backspace",
                  "insert", "return")
#: The clusters of letters keyboards steer by, and the acts a key may be tested for in play.
_CLUSTERS = ("wasd", "ijkl", "esdf", "zqsd")
_PLAY_ACTS = ("jump", "fire", "shoot", "throw", "attack", "punch", "kick", "stomp", "thrust", "boost", "brake",
              "accelerat", "rotate", "duck", "crouch", "block", "dash", "switch", "swap")
#: What a control may be for that stops, silences or leaves play: not part of the task while it goes on.
_INTERRUPTS = ("pause", "mute", "sound", "quit", "menu", "help")
#: What the pointer may be named for where keys move her: aiming and what is done at what is aimed at.
_AIMS = ("aim", "shoot", "fire", "throw", "aim and shoot", "aim and fire", "aim and throw", "click", "attack", "launch",
         "select", "use")
#: The keys a body is moved by.
_MOVING = frozenset({"up", "down", "left", "right", *"wasdijklefzq"})
#: What a key kept for something other than play is for: the menu's keys, not play's.
_KEYS_NOT_FOR_PLAY = ("pause", "mute", "sound", "quit", "menu", "help", "restart", "play", "start", "begin", "continue")


@dataclass
class Control:
    """One control: a key ("left", "space", "w") or the pointer, what it does, and where that was learned."""

    key: str
    act: str
    source: str
    since: float = 0.0


@dataclass
class Known:
    """A mechanic as this place has it: the evidence, from where, since when, and whether it is in play now."""

    name: str
    evidence: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    since: float = 0.0
    in_play: bool = True
    until: float = 0.0


@dataclass
class Guide:
    """What she knows of how one place is worked, from every source, as it is now."""

    place: str = ""
    controls: dict[str, Control] = field(default_factory=dict)
    pointer: dict[str, str] = field(default_factory=dict)
    mechanics: dict[str, Known] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    #: How a round is won, how it is lost, and how to do well, as the sources say it.
    win: list[str] = field(default_factory=list)
    lose: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)
    #: A thing's words, and what to do about it: "meet", "avoid", "shoot".
    things: dict[str, str] = field(default_factory=dict)
    #: A label read off the screen, and the mechanic it measures.
    readouts: dict[str, str] = field(default_factory=dict)
    #: Controls that are not part of play, by what they are for.
    not_for_play: list[str] = field(default_factory=list)
    strategy: list[str] = field(default_factory=list)
    #: (when, what changed), oldest first.
    changes: list[tuple[float, str]] = field(default_factory=list)
    stage: str = ""
    sources: set[str] = field(default_factory=set)
    #: Sentences already taken in, so a screen read again is not news.
    _read: set[str] = field(default_factory=set)
    began: float = field(default_factory=time.monotonic)
    #: Whether what it holds has been said to whoever is watching, and whether play has begun.
    said: bool = False
    playing: bool = False
    #: Whether her own model has been asked what it knows of the place (core/cognition/what_i_know_of_a_place.py).
    asked_what_i_know: bool = False
    asking_what_i_know: Any = None
    #: How her model is asked, as whoever began the work here gave it (core/cognition/what_i_know_of_a_place.py).
    ask_her_model: Any = None
    #: The sentence each control was named in, for what play confirms of it.
    named_in: dict[str, str] = field(default_factory=dict)
    #: The place's own names for things, by the part they play ("me", "goal", "get", "avoid", "shoot"), and which
    #: name each thing she sees has been given, by the part it plays and what it looks like.
    names: dict[str, list[str]] = field(default_factory=dict)
    given: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Things her model supposed the place has, not the place's own words: never names for what she sees.
    _supposed: set[str] = field(default_factory=set)
    #: What she notices there, and the theories it grew into (core/cognition/what_she_notices.py).
    notes: Notebook = field(default_factory=Notebook)
    #: What the things she sees there are, as they look, and who its characters are (core/cognition/what_things_are.py).
    seen: WhatSheSees = field(default_factory=WhatSheSees)
    #: What the place is: what it is about, who she is in it, what she works it with and by what, what she wants out of
    #: it, and what makes sense there (core/cognition/what_this_place_is.py).
    reading: WhatThisPlaceIs = field(default_factory=WhatThisPlaceIs)

    # -- taking things in ---------------------------------------------------------------------------------------

    def take_in(self, source: str, said: str | Iterable[str], at: float | None = None) -> list[str]:
        """Writing from one source: its controls, goals, things, readouts and mechanics; what it changed, said."""
        from core.agency.mechanics_she_knows import changes_told, mechanics_in

        at = time.monotonic() if at is None else at
        lines = [said] if isinstance(said, str) else list(said)
        sentences = [s.strip() for line in lines for s in re.split(r"(?<=[.!?])\s+|\n|•", str(line or "")) if s and s.strip()]
        new = [s for s in sentences if s.lower() not in self._read]
        if not new:
            return []
        self._read |= {s.lower() for s in new}
        self.sources.add(source)
        text = " ".join(new)
        news: list[str] = []
        if source != MODEL:
            self._controls_from(new, source, at)
        self._goals_and_things_from(new)
        self._sections_from(new)
        if source != MODEL:
            # Names come from the place's own words and what is said of it, never from what her model supposes.
            self._names_from(new)
        if source == SCREEN:
            # A label read off the screen with a value beside it; an instruction ("Watch your fuel") is not one.
            for label in new:
                self._readout_from(label)
        news += self._stage_from(text, at)
        # New to play, not merely read first: what the place shows after play has begun that it had not shown before.
        late = self.playing and source == SCREEN and bool(self.mechanics)
        # What the place says is new or gone first; then what it only mentions.
        for how, name, sentence in changes_told(text):
            news += self._changed(how, name, sentence, at)
        for name, evidence in mechanics_in(text).items():
            news += self._know(name, evidence, source, at, late=late)
        return news

    def take_in_program(self, read: Any, at: float | None = None) -> None:
        """What a program says of how it is worked, as the least trusted source; its instructions as its words."""
        from core.agency.mechanics_she_knows import mechanics_in

        if not read:
            return
        at = time.monotonic() if at is None else at
        # The instructions a program holds are what it tells its player, as a manual does: its instructions.
        self.take_in(TOLD, getattr(read, "words", []) or [], at)
        for key, purpose in (getattr(read, "keys", {}) or {}).items():
            if key not in self.controls:
                self.controls[key] = Control(key, purpose or "", PROGRAM, at)
            if purpose in _INTERRUPTS and purpose not in self.not_for_play:
                self.not_for_play.append(f"{purpose} ({key})")
        for act in getattr(read, "keyed", []) or []:
            self.pointer.setdefault(f"keys kept for {act}", PROGRAM)
        for use in getattr(read, "pointer", []) or []:
            self.pointer.setdefault(use, PROGRAM)
        for button in getattr(read, "buttons", []) or []:
            if button in _NOT_FOR_PLAY and button not in self.not_for_play:
                self.not_for_play.append(button)
        for name, evidence in mechanics_in(program=getattr(read, "text", "")).items():
            self._know(name, evidence, PROGRAM, at)
        for name in getattr(read, "mechanics_found", []) or []:
            self._know(name, ["its program shows it"], PROGRAM, at)
        self.sources.add(PROGRAM)

    def take_in_counsel(self, told: str, at: float | None = None) -> None:
        if told:
            self.strategy += [s for s in re.split(r"(?<=[.!?])\s+", told) if s and s not in self.strategy][:6]
            self.take_in(COUNSEL, told, at)

    # -- what it holds ---------------------------------------------------------------------------------------------

    def keys_for_play(self) -> list[str]:
        """The keys play is worked by: those words name; a program's only where no word names a key."""
        worded = [c.key for c in self.controls.values() if c.source != PROGRAM and c.key != "the pointer"
                  and not self._not_for_play_act(c.act)]
        if worded or any(c.source != PROGRAM for c in self.controls.values()):
            # Words that name only what keys do ("space to attack") say nothing of how she moves: the keys its
            # program moves her by are kept beside them.
            if worded and not set(worded) & _MOVING:
                worded += [c.key for c in self.controls.values() if c.source == PROGRAM and c.key in _MOVING
                           and c.key not in worded]
            return worded
        # A program's handlers listen for keys its menus and its debugging use too: only the keys play is usually
        # worked by are taken from it, digits only where it switches between things.
        coded = [c for c in self.controls.values() if c.source == PROGRAM and c.key != "the pointer"
                 and not self._not_for_play_act(c.act) and c.key not in _NOT_PLAYED_BY]
        letters = {c.key for c in coded if len(c.key) == 1 and c.key.isalpha()}
        clusters = {k for cluster in _CLUSTERS if set(cluster) <= letters for k in cluster}
        # A letter a program tests is a key of play only in a cluster keyboards steer by, or tested for something play
        # does: letters tested one after another spell a code (a cheat, a name), not controls.
        return [c.key for c in coded
                if (not c.key.isdigit() or "switching" in self.mechanics)
                and (not (len(c.key) == 1 and c.key.isalpha()) or c.key in clusters or c.act in _PLAY_ACTS)]

    def pointer_named(self) -> bool:
        """Whether the place is worked with the pointer, by words; or by its program where no word names any control.
        Where keys move her and the pointer aims or shoots ("WASD to move, mouse to aim"), it is worked by the keys,
        the pointer only aiming: LIVE 2026-10-09 the reticle that followed the mouse was taken for her, and the body
        her keys moved was never found."""
        pointer = self.controls.get("the pointer")
        if self.pointer_aims():
            return False
        if pointer is not None:
            return True
        worded = any(c.source != PROGRAM for c in self.controls.values())
        return not worded and any(use in ("steers", "aims", "drags") for use in self.pointer)

    def pointer_aims(self) -> bool:
        """Whether keys move her and the pointer only aims or shoots: her body is found by the keys, not the pointer."""
        pointer = self.controls.get("the pointer")
        return (pointer is not None and pointer.act in _AIMS
                and any(key in _MOVING for key in self.keys_for_play()))

    def in_play(self, name: str) -> bool:
        known = self.mechanics.get(name)
        return bool(known and known.in_play and (not known.until or time.monotonic() < known.until))

    def measures(self, label: str) -> str:
        """The mechanic a label on the screen measures ("health", "charging", "a clock"), or ""."""
        from core.agency.mechanics_she_knows import what_a_label_measures

        said = " ".join(str(label or "").lower().split())
        return self.readouts.get(said) or what_a_label_measures(said)

    def is_not_for_play(self, label: str) -> bool:
        said = " ".join(str(label or "").lower().split())
        return any(re.search(rf"\b{re.escape(c.split(' (')[0])}\b", said) for c in self.not_for_play) or said in _NOT_FOR_PLAY

    def is_to_read(self, label: str) -> bool:
        """Whether a label on the screen is information to read, not something to press."""
        said = " ".join(str(label or "").lower().split())
        return said in self.readouts or bool(self.measures(said)) and len(said.split()) <= 3

    def changed_since(self, since: float) -> list[str]:
        return [what for when, what in self.changes if when > since]

    # -- saying it -------------------------------------------------------------------------------------------------

    def says(self) -> str:
        """The guide as a person would give it before handing over: what it is for, how a round is won and lost, how it
        is worked, how its mechanics behave, what to get and keep clear of, what to watch, and a plan; each part from
        where it came. "" where it holds nothing worth saying."""
        lines: list[str] = []
        goal = self.goals[0] if self.goals else ""
        if goal:
            lines.append(f"• Goal: {_short(goal)}")
        win = next((w for w in self.win if w != goal), "")
        if win:
            lines.append(f"• To win: {_short(win)}")
        if self.lose:
            lines.append(f"• What loses: {_short(self.lose[0])}")
        controls = self._controls_said()
        if controls:
            lines.append(f"• {controls}")
        for mechanic in self._the_mechanics_that_matter()[:2]:
            lines.append(f"• How it works ({mechanic.name}): {mechanic.what}; {mechanic.means}.")
        get = [w for w, st in self.things.items() if st == "meet"][:3]
        avoid = [w for w, st in self.things.items() if st == "avoid"][:3]
        if get or avoid:
            lines.append("• " + "; ".join(([f"worth getting: {_listed(get)}"] if get else [])
                                          + ([f"keep clear of: {_listed(avoid)}"] if avoid else [])).capitalize() + ".")
        watch = list(dict.fromkeys([*self.readouts, *(n for n in ("fuel", "health", "lives", "a clock") if self.in_play(n))]))
        if watch:
            lines.append(f"• Watch: {_listed(watch[:4])}.")
        # A plan from what helped or was told, else from how its controls behave (a control's mechanic before the
        # world's).
        mattering = sorted(self._the_mechanics_that_matter(), key=lambda m: m.kind != "control")
        plan = self.tips[0] if self.tips else (mattering[0].play if mattering else "")
        if plan:
            lines.append(f"• Plan: {_short(re.sub(r'^(?:tips?|hint|strategy)\s*:\s*', '', plan, flags=re.I))}")
        if self.not_for_play:
            names = list(dict.fromkeys(c.split(" (")[0] for c in self.not_for_play))[:3]
            lines.append(f"• Its {_listed(names)} controls aren't part of play.")
        if len(lines) < 2:
            return ""
        sources = [x for x in (SCREEN, TOLD, PAGE, COUNSEL, MODEL, PROGRAM) if x in self.sources]
        place = f" to {self.place}" if self.place and len(self.place) <= 60 else ""
        return f"My guide{place}, from {_listed(sources)}:\n" + "\n".join(lines)

    def _the_mechanics_that_matter(self) -> list[Any]:
        """The mechanics in play that say most about how this place behaves, the ones its words speak of first."""
        from core.agency.mechanics_she_knows import known

        found = []
        for name, kept in self.mechanics.items():
            mechanic = known(name)
            if mechanic is None or not self.in_play(name) or name in _SAID_ELSEWHERE or mechanic.kind not in _BEHAVES:
                continue
            worded = any(source != PROGRAM for source in kept.sources)
            found.append((0 if worded else 1, _BEHAVES.index(mechanic.kind), -len(kept.evidence), mechanic))
        return [mechanic for *_rank, mechanic in sorted(found, key=lambda row: row[:3])]

    def in_brief(self) -> str:
        """What a move is chosen with: what it is for, how it is worked, what to get and keep clear of, what is not part
        of the task, and what changed lately; short, as it is read at every move."""
        lines = [f"Place: {self.place}" if self.place else "", self.reading.for_thinking(),
                 "Goal: " + _short(self.goals[0]) if self.goals else "",
                 "What loses: " + _short(self.lose[0]) if self.lose else "",
                 self._controls_said(),
                 "Things: " + ", ".join(f"{w} ({s})" for w, s in list(self.things.items())[:6]) if self.things else "",
                 "Not part of the task: " + ", ".join(self.not_for_play[:4]) if self.not_for_play else "",
                 "Lately changed: " + " | ".join(what for _when, what in self.changes[-2:]) if self.changes else ""]
        noted = self.notes.for_thinking(time.monotonic()) if self.notes is not None else ""
        return "\n".join(line for line in [*lines, self.seen.for_thinking(), noted] if line)[:1400]

    def for_thinking(self) -> str:
        """Everything it holds, plainly, for reasoning with: what she is told, never what she must do."""
        lines = [f"Place: {self.place}" if self.place else "", self.reading.for_thinking(), self._controls_said(),
                 "Goals: " + " | ".join(self.goals[:4]) if self.goals else "",
                 "Mechanics in play: " + ", ".join(n for n in self.mechanics if self.in_play(n))
                 if any(self.in_play(n) for n in self.mechanics) else "",
                 "Gone from play: " + ", ".join(n for n in self.mechanics if not self.in_play(n))
                 if any(not self.in_play(n) for n in self.mechanics) else "",
                 "Things: " + ", ".join(f"{w} ({s})" for w, s in list(self.things.items())[:8]) if self.things else "",
                 "Read off the screen: " + ", ".join(f"{w} ({m})" for w, m in list(self.readouts.items())[:8])
                 if self.readouts else "",
                 "Not part of the task: " + ", ".join(self.not_for_play[:6]) if self.not_for_play else "",
                 f"Now: {self.stage}" if self.stage else "",
                 "Lately changed: " + " | ".join(what for _when, what in self.changes[-3:]) if self.changes else "",
                 "To win: " + " | ".join(self.win[:2]) if self.win else "",
                 "What loses: " + " | ".join(self.lose[:2]) if self.lose else "",
                 "Tips: " + " | ".join(self.tips[:3]) if self.tips else "",
                 "Counsel: " + " ".join(self.strategy[:3]) if self.strategy else ""]
        from core.agency.mechanics_she_knows import known

        lines.append(self.seen.for_thinking())
        if self.notes is not None and self.notes.for_thinking(time.monotonic()):
            lines.append(self.notes.for_thinking(time.monotonic()))
        for name in [n for n in self.mechanics if self.in_play(n)][:6]:
            mechanic = known(name)
            if mechanic is not None:
                lines.append(f"{name}: {mechanic.means}; {mechanic.play}")
        return "\n".join(line for line in lines if line)

    # -- inside -----------------------------------------------------------------------------------------------------

    def _controls_from(self, sentences: list[str], source: str, at: float) -> None:
        from core.agency.the_controls_a_game_names import controls_named_in

        for sentence in sentences:
            keys, pointer = controls_named_in(sentence, keys_without_words=(), during_play=True)
            act = _the_act(sentence)
            for key in keys:
                known = self.controls.get(key)
                if known is None or _TRUST.get(source, 0) >= _TRUST.get(known.source, 0):
                    self.controls[key] = Control(key, act or (known.act if known else ""), source, at)
                    self.named_in[key] = sentence[:200]
            if pointer:
                known = self.controls.get("the pointer")
                if known is None or _TRUST.get(source, 0) >= _TRUST.get(known.source, 0):
                    self.controls["the pointer"] = Control("the pointer", act or (known.act if known else ""), source, at)

    def _goals_and_things_from(self, sentences: list[str]) -> None:
        from core.perception.what_a_legend_shows import stance_of

        for sentence in sentences:
            sentence = _A_HEADING.sub("", sentence)
            if _A_GOAL.search(sentence) and len(sentence.split()) >= 3 and sentence not in self.goals:
                self.goals.append(sentence[:200])
            # Each thing by what its own clause says to do about it ("steer the cart and collect the coins": the coins
            # are to get), else by what a caption beside it says ("Candy: gives you invincibility").
            for verb, thing in _things_spoken_of(sentence):
                stance = _STANCE_OF_A_VERB.get(verb.lower(), "") or stance_of(sentence)
                if stance:
                    self.things[thing] = stance
        del self.goals[:-8]

    def _names_from(self, sentences: list[str]) -> None:
        """The place's own names for what plays each part: what its controls move, where it is to be brought, what to
        get, keep clear of and shoot."""
        for sentence in sentences:
            for part, says in _NAMING:
                for found in says.finditer(sentence):
                    name = _a_name(found.group(1))
                    if name and name not in self.names.setdefault(part, []):
                        self.names[part].append(name)
        # Whose the place is, by its title ("Buddy's Big Adventure"): who she likely plays, after what its words say.
        for found in _A_TITLE_OWNER.finditer(self.place or ""):
            name = _a_name(found.group(1))
            if name and name not in self.names.setdefault("me", []):
                self.names["me"].append(name)
        for thing, stance in self.things.items():
            if thing in self._supposed:
                continue
            part = {"meet": "get", "avoid": "avoid", "shoot": "shoot"}.get(stance, "")
            name = _a_name(thing)
            if part and name and name not in self.names.setdefault(part, []):
                self.names[part].append(name)

    def name_for(self, part: str, looks: str, *, alternatives: tuple[str, ...] = ()) -> str:
        """What the place calls the thing she sees that plays ``part``, looking like ``looks`` (a colour); "" where its
        words name nothing for it. The same thing keeps its name; two things are not given one name."""
        if (part, looks) in self.given:
            return self.given[(part, looks)]
        taken = set(self.given.values())
        candidates = [n for p in (part, *alternatives) for n in self.names.get(p, []) if n not in taken]
        if not candidates:
            return ""
        coloured = [n for n in candidates if looks and looks in n.split()]
        name = (coloured or candidates)[0]
        self.given[(part, looks)] = name
        return name

    def _sections_from(self, sentences: list[str]) -> None:
        """Each sentence that says how a round is won or lost, or how to do well, kept under that."""
        for sentence in (_A_HEADING.sub("", s) for s in sentences):
            if len(sentence.split()) < 3:
                continue
            for kept, says in ((self.win, _A_WIN), (self.lose, _A_LOSS), (self.tips, _A_TIP)):
                if says.search(sentence) and sentence not in kept:
                    kept.append(sentence[:220])
                    del kept[:-6]

    def take_in_what_i_know(self, manual: dict[str, Any], at: float | None = None) -> list[str]:
        """What her own model knows of the place, as a manual's sections: kept as the least sure source, never as its
        controls. The sections it filled that were empty, said."""
        filled: list[str] = []
        for part, kept in (("goal", self.goals), ("win", self.win), ("lose", self.lose)):
            said = " ".join(str(manual.get(part) or "").split())
            if said and said.lower() not in ("unknown", "none", "n/a"):
                filled += [part] if not kept else []
                kept.append(said[:220])
        for part, stance in (("get", "meet"), ("avoid", "avoid")):
            for thing in manual.get(part) or []:
                thing = " ".join(str(thing).lower().split())[:40]
                if thing and thing not in self.things and thing not in ("unknown", "none"):
                    self.things[thing] = stance
                    self._supposed.add(thing)
                    filled.append(part)
        tips = [" ".join(str(t).split())[:220] for t in manual.get("tips") or [] if str(t).strip()]
        if tips:
            filled += ["tips"] if not self.tips else []
            self.tips += [t for t in tips if t not in self.tips]
            del self.tips[:-8]
        if self.seen.take_in_cast(manual.get("who") or []):
            filled.append("who")
        self.sources.add(MODEL)
        how = " ".join(str(manual.get("how") or "").split())
        if how:
            self.take_in(MODEL, how, at)
        return list(dict.fromkeys(filled))

    def _readout_from(self, label: str) -> None:
        match = _A_READOUT.match(label or "")
        if not match or len(label.split()) > 4:
            return
        words = " ".join(match.group(1).lower().split())
        measured = self.measures(words)
        if measured and words not in self.readouts:
            self.readouts[words] = measured

    def _stage_from(self, text: str, at: float) -> list[str]:
        found = _A_STAGE.search(text)
        if not found:
            return []
        stage = f"{found.group(1).lower()} {found.group(2)}"
        if stage == self.stage:
            return []
        before, self.stage = self.stage, stage
        if not before:
            return []
        self.changes.append((at, f"now on {stage}"))
        return [f"Now on {stage}."]

    def _know(self, name: str, evidence: list[str], source: str, at: float, *, late: bool = False) -> list[str]:
        known = self.mechanics.get(name)
        if known is None:
            known = self.mechanics[name] = Known(name, since=at)
            if late:
                # First read once play is under way: a note of hers, not news. What the place says is new is said
                # (``_changed``); a screen that only mentions a thing is not.
                self.changes.append((at, f"{name} first read of"))
                self.notes.notice(f"read of {name}", f"this place speaks of {name} now", at, kind="read of")
        known.evidence += [e for e in evidence if e not in known.evidence][:6]
        known.sources.add(source)
        return []

    def _changed(self, how: str, name: str, sentence: str, at: float) -> list[str]:
        new = name not in self.mechanics
        known = self.mechanics.setdefault(name, Known(name, since=at))
        was = known.in_play and not new
        known.in_play = how == "added"
        lasts = _FOR_A_WHILE.search(sentence)
        known.until = at + float(lasts.group(1)) if lasts and how == "added" else 0.0
        if was == known.in_play and how == "added" and not known.until:
            return []
        said = f"{name} {'came into play' if how == 'added' else 'went out of play'}: “{sentence[:120]}”"
        self.changes.append((at, said))
        del self.changes[:-40]
        return [f"Something changed: {said}."]

    def _controls_said(self) -> str:
        by_act: dict[str, list[str]] = {}
        playing = set(self.keys_for_play()) | {"the pointer"}
        for control in self.controls.values():
            if control.source == PROGRAM and control.key not in playing:
                continue
            by_act.setdefault(control.act or "?", []).append(control.key)
        said = [f"{_listed(keys)} {'to ' + act if act != '?' else ''}".strip() for act, keys in by_act.items()]
        uses = [use for use in self.pointer if not use.startswith("keys kept")]
        if uses and "the pointer" not in self.controls:
            said.append(f"the mouse {_listed(uses)}")
        return ("Controls: " + "; ".join(said) + ".") if said else ""

    @staticmethod
    def _not_for_play_act(act: str) -> bool:
        return act in _KEYS_NOT_FOR_PLAY

    # -- kept between sittings ------------------------------------------------------------------------------------

    def as_memory(self) -> dict[str, Any]:
        return {"place": self.place,
                "controls": {k: [c.act, c.source] for k, c in self.controls.items()},
                "pointer": dict(self.pointer),
                "mechanics": {n: {"evidence": k.evidence[:6], "sources": sorted(k.sources), "in_play": k.in_play}
                              for n, k in self.mechanics.items()},
                "goals": self.goals[:8], "things": dict(list(self.things.items())[:40]),
                "readouts": dict(list(self.readouts.items())[:40]), "not_for_play": self.not_for_play[:20],
                "strategy": self.strategy[:6], "stage": self.stage, "sources": sorted(self.sources),
                "names": {part: names[:6] for part, names in self.names.items()},
                "notes": self.notes.as_memory() if self.notes is not None else {},
                "reading": self.reading.as_memory()}

    @classmethod
    def from_memory(cls, held: Any, place: str = "") -> Guide:
        guide = cls(place=place)
        if not isinstance(held, dict):
            return guide
        guide.place = str(held.get("place") or place)
        for key, (act, source) in (held.get("controls") or {}).items():
            guide.controls[key] = Control(key, act, source)
        guide.pointer = dict(held.get("pointer") or {})
        for name, known in (held.get("mechanics") or {}).items():
            guide.mechanics[name] = Known(name, list(known.get("evidence") or []), set(known.get("sources") or []),
                                          in_play=bool(known.get("in_play", True)))
        guide.goals, guide.things = list(held.get("goals") or []), dict(held.get("things") or {})
        guide.readouts, guide.not_for_play = dict(held.get("readouts") or {}), list(held.get("not_for_play") or [])
        guide.strategy, guide.stage = list(held.get("strategy") or []), str(held.get("stage") or "")
        guide.sources = set(held.get("sources") or [])
        guide.names = {part: list(names) for part, names in (held.get("names") or {}).items()}
        guide.notes = Notebook.from_memory(held.get("notes"))
        guide.reading = WhatThisPlaceIs.from_memory(held.get("reading"))
        return guide


#: Where a place's words name what plays each part: what its controls move ("guide the lander", "help Bloo"), where it
#: is to be brought ("onto the landing platform", "reach the exit"), and what to get, keep clear of and shoot.
_A_NOUN = r"((?:[A-Za-z]+'s\s+)?(?:the\s+|your\s+|a\s+|an\s+)?[A-Za-z][A-Za-z -]{1,32}?)"
_ENDS = (r"(?=\s+(?:onto|to|into|through|around|across|safely|slowly|and|with|by|using|past|so|while|before|without|from|"
         r"in|on|at|as|escape|find|get|reach|save|stop|enjoy|catch|collect|win|land|fly|jump|run)\b|[.,!;:)]|$)")
_NAMING = (
    ("me", re.compile(r"\b(?:guide|control|steer|drive|park|ride|fly|pilot|move|help|lead|play as|you are|you're|you play)\s+"
                      + _A_NOUN + _ENDS, re.I)),
    ("goal", re.compile(r"\b(?:onto|land on|reach|get to|into|bring (?:it|them) to)\s+" + _A_NOUN + _ENDS, re.I)),
    ("get", re.compile(r"\b(?:collect|grab|catch|pick up|gather)\s+(?:all\s+|every\s+|as many\s+)?" + _A_NOUN + _ENDS, re.I)),
    ("avoid", re.compile(r"\b(?:avoid|dodge|watch out for|beware of|keep away from|don'?t (?:touch|hit)|escape(?: from)?|"
                         r"flee(?: from)?|run from|get away from|hide from)\s+" + _A_NOUN + _ENDS, re.I)),
    ("friend", re.compile(r"\b(?:rescue|save|protect|free|defend)\s+(?:all\s+|every\s+)?" + _A_NOUN + _ENDS, re.I)),
    ("shoot", re.compile(r"\b(?:shoot|destroy|defeat|blast|zap|stop)\s+(?:all\s+|every\s+)?" + _A_NOUN + _ENDS, re.I)),
)
#: A title that says whose the place is: "Buddy's Big Adventure", "Dexter's Laboratory: Robot Rampage".
_A_TITLE_OWNER = re.compile(r"(?:^|[:\-–]\s*)(?:The\s+)?([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})['’]s\s+[A-Z]")
#: Words that are no name for a thing.
_NOT_A_NAME = frozenset("""the a an your you it its them they this that these those what which keys key arrow arrows mouse
    button buttons space spacebar time way ground game screen level points score as many all every each other own
    is are was were be been being will can could would should may might must do does did has have had he she his her
    him we us our my me i who whom whose there here then than so too very just also not no yes up down left right
    in on at of to by for from with into onto out off over under again more most less""".split())


def _a_name(said: str) -> str:
    """A thing's name as a place says it, kept to its last few words and lowered: "Tommy's Lunar Lander" is the lunar
    lander; "" where it is no name ("the arrow keys", "it")."""
    words = re.sub(r"^[A-Za-z]+'s\s+", "", str(said or "").strip()).split()
    words = [w for w in words if w.lower() not in ("the", "your", "a", "an")]
    if (not words or len(words) > 4 or words[-1].lower() in _NOT_A_NAME or len(words[-1]) < 3
            or not any(w.lower() not in _NOT_A_NAME for w in words)):
        return ""
    return " ".join(w.lower() for w in words[-3:])


#: Words that say nothing of a mechanic, left out of what is learned as a sign of one.
_PLAIN = frozenset("""the and with your you use press hold click keys key arrow arrows mouse button buttons move moves to
    of in on at for from into onto this that then when while each every all any can will get make them they their
    his her its it's""".split())


def _plain_words(text: str) -> list[str]:
    return [w for w in dict.fromkeys(re.findall(r"[a-z][a-z']{3,}", str(text or "").lower())) if w not in _PLAIN]


#: The kinds of mechanic that say how a place behaves, in the order they matter to say; and those said in other parts of
#: the guide, or too general to tell a place by.
_BEHAVES = ("world", "control", "hazard", "resource", "structure", "goal")
_SAID_ELSEWHERE = frozenset({"steering", "menus", "information shown", "decoration", "winning and losing", "score",
                             "levels", "a clock", "clicking things", "choosing from options", "dialogs", "sound controls",
                             "pausing", "menus of a program", "links and pages", "focus and selection", "story",
                             "collecting", "reaching a place", "hazards", "lives", "health", "fuel", "typing"})


def _short(text: str, most: int = 150) -> str:
    said = " ".join(str(text or "").split()).rstrip(".")
    return (said if len(said) <= most else said[: most - 1].rsplit(" ", 1)[0] + "…") + ("" if said.endswith(("!", "?")) else ".")


def _listed(items: list[str]) -> str:
    items = [str(i) for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


#: The act a sentence names its controls for: "to move", "to jump", "to aim and shoot".
_TO_DO = re.compile(r"\bto ((?:move|walk|run|jump|shoot|fire|throw|aim|steer|guide|drive|fly|glide|attack|stomp|punch|kick|"
                    r"switch|swap|pause|start|launch|dodge|duck|crouch|block|boost|brake|thrust|turn|rotate|land|catch|"
                    r"pick up|use|select|play)(?: and (?:move|jump|shoot|fire|throw|aim|attack))?)\b", re.I)


def _the_act(sentence: str) -> str:
    found = _TO_DO.search(sentence or "")
    return found.group(1).lower() if found else ""


#: The thing a caption speaks of: "Avoid the spikes", "Collect coins", "Candy: gives you invincibility".
_SPOKEN_OF = re.compile(r"^\W*([A-Za-z][A-Za-z ]{1,24}?)\s*[:\-–]\s|\b(avoid|collect|catch|grab|get|destroy|shoot|"
                        r"dodge|stop|beware of|watch out for|keep away from|rescue|save|free)\s+(?:the |all |all the |"
                        r"any |every |your )?([a-z][a-z ]{1,24}?)(?=[.,!;]| to | and | or | before | while |$)", re.I)


#: Words that are not things: what a caption's verb is followed by when it speaks of no thing.
_NOT_THINGS = frozenset("""to it its you your them him her this that there here away past apart up down out off over
    in on at as by for from with into onto back again more all each every some any the a an no not one ones way time
    while when hint hints tip tips instructions""".split())


#: What a verb says to do about the thing it is said of.
_STANCE_OF_A_VERB = {"avoid": "avoid", "dodge": "avoid", "beware of": "avoid", "watch out for": "avoid",
                     "keep away from": "avoid", "collect": "meet", "catch": "meet", "grab": "meet", "get": "meet",
                     "rescue": "meet", "save": "meet", "free": "meet", "destroy": "shoot", "shoot": "shoot", "stop": "shoot"}


def _things_spoken_of(sentence: str) -> list[tuple[str, str]]:
    """Each thing a sentence speaks of, with the verb said of it ("" for a caption's "Thing: what it does")."""
    out: list[tuple[str, str]] = []
    for found in _SPOKEN_OF.finditer(sentence or ""):
        words = (found.group(1) or found.group(3) or "").strip().lower().split()
        if not words or len(words) > 3 or words[0] in _NOT_THINGS or words[-1] in _NOT_THINGS or words[-1].endswith("ing"):
            continue
        out.append((found.group(2) or "", " ".join(words)))
    return out


def guide_of(keep: dict[str, Any], place: str = "") -> Guide:
    """The guide kept for a place in what she keeps of it, made if there is none."""
    guide = keep.get("guide")
    if not isinstance(guide, Guide):
        guide = Guide.from_memory(guide, place) if guide else Guide(place=place)
        keep["guide"] = guide
    return guide


#: The guide to the place she is working in now, set by whoever begins the work there (a hand-over of a thing a page
#: draws sets the one it built); None until then.
THE_GUIDE: contextvars.ContextVar[Guide | None] = contextvars.ContextVar("aura_the_guide", default=None)


def the_guide(run: dict[str, Any] | None = None, place: str = "") -> Guide:
    """The guide to where she is: the one set for this work, else the one this run keeps, made where there is none."""
    guide = THE_GUIDE.get()
    if guide is not None:
        return guide
    if isinstance(run, dict):
        kept = run.get("guide")
        if not isinstance(kept, Guide):
            kept = run["guide"] = Guide(place=place)
        return kept
    return Guide(place=place)


def heard_in_play(texts: Iterable[str], at: float | None = None) -> list[str]:
    """Writing read while play goes on, taken into the guide to where she is, if there is one: what it changed."""
    guide = THE_GUIDE.get()
    if guide is None:
        return []
    guide.playing = True
    return guide.take_in(SCREEN, [str(t) for t in texts if t], at)


#: The guides to the places she has been lately, by place: a site, a program, a thing a page draws.
_GUIDES: dict[str, Guide] = {}
MOST_GUIDES = 32


def guide_for(place: str) -> Guide:
    """The guide to a place she is working in, kept while she works there and across her visits this sitting."""
    key = " ".join(str(place or "").lower().split())
    guide = _GUIDES.pop(key, None) or Guide(place=place)
    _GUIDES[key] = guide
    while len(_GUIDES) > MOST_GUIDES:
        del _GUIDES[next(iter(_GUIDES))]
    return guide


def carry_over(guide: Guide, before: Guide) -> None:
    """What play taught her of the place on an earlier visit: what she came to think, its names, and the tips that
    held; never how it is worked, which the place itself says again."""
    for note in before.notes.theories():
        guide.notes.notes.setdefault(note.key, note)
    for part, names in before.names.items():
        guide.names.setdefault(part, [])
        guide.names[part] += [n for n in names if n not in guide.names[part]]
    guide.tips += [t for t in before.tips if t not in guide.tips][:6]


def confirmed_by(guide: Guide, stretch: dict[str, Any]) -> list[str]:
    """What a stretch of play confirmed of the mechanics the guide holds, and the words this place used for them,
    taught to what she knows of every place (core/agency/mechanics_she_knows.py ``LearnedSigns``): the words that
    became signs anywhere."""
    from core.agency.mechanics_she_knows import learned_signs

    if not guide.place:
        return []
    learned = learned_signs()
    became: list[str] = []
    moved_by = list(stretch.get("keys_that_move_her") or [])
    if stretch.get("hers") and moved_by:
        said = " ".join(guide.named_in.get(key, "") for key in moved_by)
        became += learned.confirmed("steering", _plain_words(said), guide.place)
    counters = [str(name) for name in (stretch.get("counters") or {})]
    if int(stretch.get("gains") or 0) > 0 and counters:
        became += learned.confirmed("score", _plain_words(" ".join(counters)), guide.place)
    return became
