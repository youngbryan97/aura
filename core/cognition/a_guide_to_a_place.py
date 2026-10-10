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

__all__ = ["THE_GUIDE", "Guide", "guide_for", "guide_of", "heard_in_play", "the_guide"]

#: Sources, most trusted first.
SCREEN, TOLD, PAGE, COUNSEL, PROGRAM = "the screen", "its instructions", "its page", "what I looked up", "its program"
_TRUST = {SCREEN: 5, TOLD: 4, PAGE: 3, COUNSEL: 2, PROGRAM: 1}

#: A sentence that sets a goal: what to do to win, or what not to let happen.
_A_GOAL = re.compile(r"\b(?:your (?:goal|mission|job|task) is|the (?:goal|object|aim) (?:of the game )?is|try to|"
                     r"you (?:need|have|must) to|help \w+(?: \w+)? (?:to )?(?:find|get|escape|save|rescue|stop|reach|"
                     r"collect|enjoy|win)|get (?:all|as many|to the)|reach|collect|"
                     r"guide|rescue|save|escape|survive|don'?t let|before (?:time|the timer)|as (?:many|far|long) as)\b",
                     re.I)
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
    #: Whether what it holds has been said to whoever is watching.
    said: bool = False
    #: The sentence each control was named in, for what play confirms of it.
    named_in: dict[str, str] = field(default_factory=dict)

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
        self._controls_from(new, source, at)
        self._goals_and_things_from(new)
        for label in new:
            self._readout_from(label)
        news += self._stage_from(text, at)
        late = at - self.began > 2.0 and source in (SCREEN, TOLD) and bool(self.mechanics)
        for name, evidence in mechanics_in(text).items():
            news += self._know(name, evidence, source, at, late=late)
        for how, name, sentence in changes_told(text):
            news += self._changed(how, name, sentence, at)
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

    def confirmed_by(self, stretch: dict[str, Any]) -> list[str]:
        """What a stretch of play confirmed of the mechanics the guide holds, and the words this place used for them,
        taught to what she knows of every place (core/agency/mechanics_she_knows.py ``LearnedSigns``): the words that
        became signs anywhere."""
        from core.agency.mechanics_she_knows import learned_signs

        if not self.place:
            return []
        learned = learned_signs()
        became: list[str] = []
        moved_by = list(stretch.get("keys_that_move_her") or [])
        if stretch.get("hers") and moved_by:
            said = " ".join(self.named_in.get(key, "") for key in moved_by)
            became += learned.confirmed("steering", _plain_words(said), self.place)
        counters = [str(name) for name in (stretch.get("counters") or {})]
        if int(stretch.get("gains") or 0) > 0 and counters:
            became += learned.confirmed("score", _plain_words(" ".join(counters)), self.place)
        return became

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
        """Whether the place is worked with the pointer, by words; or by its program where no word names any control."""
        if any(c.key == "the pointer" for c in self.controls.values()):
            return True
        worded = any(c.source != PROGRAM for c in self.controls.values())
        return not worded and any(use in ("steers", "aims", "drags") for use in self.pointer)

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
        """In a few sentences, for whoever is watching: how it is worked, what it counts, what to get and keep clear of."""
        parts: list[str] = []
        controls = self._controls_said()
        if controls:
            parts.append(controls)
        counted = [n for n in ("score", "lives", "health", "fuel", "a clock", "levels") if self.in_play(n)]
        if counted:
            parts.append("It keeps " + _listed(counted) + ".")
        get = [w for w, s in self.things.items() if s == "meet"][:3]
        avoid = [w for w, s in self.things.items() if s == "avoid"][:3]
        if get:
            parts.append(f"Worth getting: {_listed(get)}.")
        if avoid:
            parts.append(f"To keep clear of: {_listed(avoid)}.")
        if self.not_for_play:
            names = list(dict.fromkeys(c.split(" (")[0] for c in self.not_for_play))[:3]
            parts.append(f"Its {_listed(names)} controls aren't part of play.")
        sources = [s for s in (SCREEN, TOLD, PAGE, COUNSEL, PROGRAM) if s in self.sources]
        lead = f"From {_listed(sources)}: " if sources else ""
        return (lead + " ".join(parts)).strip() if parts else ""

    def for_thinking(self) -> str:
        """Everything it holds, plainly, for reasoning with: what she is told, never what she must do."""
        lines = [f"Place: {self.place}" if self.place else "", self._controls_said(),
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
                 "Counsel: " + " ".join(self.strategy[:3]) if self.strategy else ""]
        from core.agency.mechanics_she_knows import known

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
            if _A_GOAL.search(sentence) and len(sentence.split()) >= 3 and sentence not in self.goals:
                self.goals.append(sentence[:200])
            # Each thing by what its own clause says to do about it ("steer the cart and collect the coins": the coins
            # are to get), else by what a caption beside it says ("Candy: gives you invincibility").
            for verb, thing in _things_spoken_of(sentence):
                stance = _STANCE_OF_A_VERB.get(verb.lower(), "") or stance_of(sentence)
                if stance:
                    self.things[thing] = stance
        del self.goals[:-8]

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
                self.changes.append((at, f"{name} came into play"))
                return [f"Something new here: {name}."]
        known.evidence += [e for e in evidence if e not in known.evidence][:6]
        known.sources.add(source)
        return []

    def _changed(self, how: str, name: str, sentence: str, at: float) -> list[str]:
        known = self.mechanics.setdefault(name, Known(name, since=at))
        was = known.in_play
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
                "strategy": self.strategy[:6], "stage": self.stage, "sources": sorted(self.sources)}

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
        return guide


#: Words that say nothing of a mechanic, left out of what is learned as a sign of one.
_PLAIN = frozenset("""the and with your you use press hold click keys key arrow arrows mouse button buttons move moves to
    of in on at for from into onto this that then when while each every all any can will get make them they their
    his her its it's""".split())


def _plain_words(text: str) -> list[str]:
    return [w for w in dict.fromkeys(re.findall(r"[a-z][a-z']{3,}", str(text or "").lower())) if w not in _PLAIN]


def _listed(items: list[str]) -> str:
    items = [str(i) for i in items if i]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


#: The act a sentence names its controls for: "to move", "to jump", "to aim and shoot".
_TO_DO = re.compile(r"\bto ((?:move|walk|run|jump|shoot|fire|throw|aim|steer|drive|fly|glide|attack|stomp|punch|kick|"
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
    return guide.take_in(SCREEN, [str(t) for t in texts if t], at) if guide is not None else []


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
