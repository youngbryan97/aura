"""What the things she sees are, from what she knows of such things anywhere: a zombie is trouble, a coin is worth
getting, a bat is swung, a platform is stood on, a door is a way through.

A person who sees a zombie in a game they have never played already knows a good deal: zombies are usually hostile,
so it is best kept clear of, or fought. They know a bat is something you swing, perhaps to defend yourself; that a
platform is something you stand on; that a cursor is where the mouse is, not a character. None of it is sure (this
zombie may be a friendly one), and all of it is where a person starts.

So does she. What a kind of thing usually is, how it usually bears on whoever is there (a danger, a thing to get, to
use, to stand on, to reach, a friend, a control to press, something to read, scenery), a thought a person might have
on seeing one, and how one usually looks, are asked of her own model once for each name, beside her work, and kept for
every place after (``what_things_are``). Before her model answers, the mechanics she knows of every place
(core/agency/mechanics_she_knows.py) give a first guess where a name speaks of one ("spikes" are a hazard).

What she sees in one place (``WhatSheSees``, held by the guide to it, core/cognition/a_guide_to_a_place.py) is each
kind of thing by what it was seen as, what about one looked unlike its kind, and who the place's characters are, as
her model knows them: for her reasoning, her names for things and what she makes of them.

Nothing here knows a game, a site or a program.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BEARS", "WhatAThingIs", "WhatSheSees", "WhatThingsAre", "first_guess", "what_things_are", "words_shared"]

logger = logging.getLogger("Aura.WhatThingsAre")

#: How a kind of thing may bear on whoever is there.
BEARS = ("danger", "to get", "to use", "to stand on", "to reach", "a friend", "to press", "information", "scenery",
         "unsure")

#: How many names her model is asked of at once, and the most it writes.
ASKED_AT_ONCE = 8
MOST_TOKENS = 700
#: The most names kept, across every place.
MOST_KEPT = 600


@dataclass
class WhatAThingIs:
    """What a kind of thing usually is, how it bears on whoever is there, a thought on seeing one, how one looks."""

    name: str
    usually: str = ""
    bears: str = "unsure"
    thought: str = ""
    looks: str = ""
    source: str = ""

    def as_memory(self) -> dict[str, str]:
        return {"usually": self.usually, "bears": self.bears, "thought": self.thought, "looks": self.looks,
                "source": self.source}


#: The mechanics whose words, in a thing's name, say how it bears; in the order they are believed.
_BY_MECHANIC = (("hazards", "danger"), ("enemies", "danger"), ("being seen", "danger"), ("power-ups", "to get"),
                ("collecting", "to get"), ("rescue and allies", "a friend"), ("reaching a place", "to reach"),
                ("information shown", "information"), ("decoration", "scenery"))


def first_guess(name: str) -> WhatAThingIs | None:
    """How a thing bears, from the mechanics she knows of every place, where its name speaks of one; else None."""
    from core.agency.mechanics_she_knows import mechanics_in

    found = mechanics_in(str(name or ""))
    for mechanic, bears in _BY_MECHANIC:
        if mechanic in found:
            return WhatAThingIs(name=_key(name), bears=bears, source=f"what I know of {mechanic}")
    return None


def _key(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' -]", " ", str(name or "").lower()).split())


@dataclass
class WhatThingsAre:
    """What she knows of kinds of things anywhere, by name: asked of her model once a name, kept for every place."""

    known: dict[str, WhatAThingIs] = field(default_factory=dict)
    asking: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def of(self, name: str) -> WhatAThingIs | None:
        """What she knows of a thing by this name: from her model where it has said, else a first guess, else None.
        A name of several words is known by its longest ending that is known ("old baseball bat" by "baseball bat")."""
        key = _key(name)
        if not key:
            return None
        words = key.split()
        found = next((self.known[k] for k in (" ".join(words[i:]) for i in range(len(words))) if k in self.known), None)
        return found or first_guess(key)

    def ask_about(self, names: Iterable[str], ask: Callable[..., Awaitable[Any]], *,
                  then: Callable[[list[WhatAThingIs]], Any] | None = None) -> bool:
        """Ask her model of the names not yet known, beside her work; ``then`` told of what it said. Whether asked."""
        wanted = [k for k in dict.fromkeys(_key(n) for n in names) if k and k not in self.known and k not in self.asking]
        wanted = wanted[:ASKED_AT_ONCE]
        if not wanted:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self.asking |= set(wanted)

        async def asked() -> None:
            try:
                learned = await _asked_of_her_model(wanted, ask)
            finally:
                self.asking -= set(wanted)
            if not learned:
                return
            with self.lock:
                for one in learned:
                    self.known[one.name] = one
                while len(self.known) > MOST_KEPT:
                    del self.known[next(iter(self.known))]
            self.keep()
            logger.info("what things are, from her model: %s", {o.name: o.bears for o in learned})
            if then is not None:
                then(learned)

        loop.create_task(asked())
        return True

    def keep(self) -> None:
        from core.runtime.what_she_learned import named, remember

        try:
            remember(named("what holds everywhere", "what things are"), self.as_memory())
        except (RuntimeError, OSError, ValueError, TypeError) as why:
            logger.info("what things are could not be kept: %s", why)

    def as_memory(self) -> dict[str, Any]:
        return {"things": {name: one.as_memory() for name, one in self.known.items()}}

    def take_in(self, held: Any) -> None:
        for name, one in ((held or {}).get("things") or {}).items() if isinstance(held, dict) else ():
            if isinstance(one, dict):
                self.known[_key(name)] = WhatAThingIs(name=_key(name), **{k: str(one.get(k) or "") for k in
                                                                        ("usually", "bears", "thought", "looks", "source")})


async def _asked_of_her_model(names: list[str], ask: Callable[..., Awaitable[Any]]) -> list[WhatAThingIs]:
    from typing import Literal

    from pydantic import BaseModel, Field

    class _One(BaseModel):
        name: str = Field(description="the name, as asked")
        usually: str = Field(default="", max_length=140, description="what such a thing usually is or does")
        bears: Literal[BEARS] = Field(default="unsure", description="how such a thing usually bears on whoever is there")
        thought: str = Field(default="", max_length=180,
                             description="one short, tentative sentence a person might think on seeing one")
        looks: str = Field(default="", max_length=120, description="how one usually looks")

    class _All(BaseModel):
        things: list[_One] = Field(default_factory=list, max_length=ASKED_AT_ONCE)

    prompt = ("Someone in a game, on a website or in a program sees these things: " + "; ".join(names) + ".\nFor each, "
              "from what you know of such things in general: what such a thing usually is or does; how it usually "
              "bears on whoever is there (danger: to keep clear of or fight; to get: worth collecting; to use: picked "
              "up, wielded or ridden; to stand on; to reach: a place to go or a way through; a friend: on their side; "
              "to press: a button or control; information: something to read; scenery: there for the look of it; "
              "unsure); one short sentence a person might think on seeing one, tentatively (\"probably\", \"maybe\"); "
              "and how one usually looks. Leave a part empty where you are not fairly sure.")
    try:
        got = await ask(prompt, _All, MOST_TOKENS)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("what things are could not be asked: %s", str(why)[:160])
        return []
    out: list[WhatAThingIs] = []
    for one in getattr(got, "things", None) or []:
        key = _key(one.name)
        if key in names:
            out.append(WhatAThingIs(name=key, usually=one.usually.strip(), bears=one.bears, thought=one.thought.strip(),
                                    looks=one.looks.strip(), source="her model"))
    return out


_ALL: dict[str, WhatThingsAre] = {}
_LOADING = threading.Lock()


def what_things_are() -> WhatThingsAre:
    """What she knows of kinds of things, for every place: one store for her, kept between sittings (one per place her
    learning is kept in)."""
    from core.runtime.what_she_learned import _kept_in, named, recall

    where = str(_kept_in())
    with _LOADING:
        if where not in _ALL:
            _ALL[where] = WhatThingsAre()
            try:
                _ALL[where].take_in(recall(named("what holds everywhere", "what things are")))
            except (RuntimeError, OSError, ValueError, TypeError) as why:
                logger.info("what things are, kept before, could not be read: %s", why)
        return _ALL[where]


#: Words two names may share without being of one thing.
_COMMON = frozenset("""the a an of and with in on at to my your his her its their big small little old new red orange
    yellow green blue purple pink brown black white grey gray dark light thing one man woman""".split())


def words_shared(a: str, b: str) -> set[str]:
    """The words two names share that say what a thing is ("Cow" and "a purple cow": cow)."""
    def words(name: str) -> set[str]:
        return {w.rstrip("s") for w in _key(name).split() if w not in _COMMON and len(w) >= 3}
    return words(a) & words(b)


@dataclass
class WhatSheSees:
    """What she sees in one place: each kind of thing as it was seen (by the number play gave its kind), what about one
    looked unlike its kind, and who the place's characters are, as her model knows them."""

    by_kind: dict[int, str] = field(default_factory=dict)
    odd: dict[int, str] = field(default_factory=dict)
    #: The place's characters and named things, as her model knows them: name, what they are, how they usually look,
    #: whose side they are on ("you", "with you", "against you", "neither").
    cast: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Kinds seen as one of the cast, by the name.
    who: dict[int, str] = field(default_factory=dict)

    def saw(self, kind: int, what: str, odd: str = "") -> str:
        """A kind seen as ``what``: the cast member it is, by name, where it is one; else ""."""
        self.by_kind[kind] = what
        if odd:
            self.odd[kind] = odd
        for name in self.cast:
            if words_shared(name, what) or words_shared(name, self.cast[name].get("what", "")) & set(_key(what).split()):
                self.who[kind] = name
                return name
        return ""

    def take_in_cast(self, who: Iterable[Any]) -> list[str]:
        """Her model's knowledge of who and what the place has in it: what was added."""
        added: list[str] = []
        for one in who or []:
            said = one if isinstance(one, dict) else getattr(one, "model_dump", lambda: {})()
            name = " ".join(str(said.get("name") or "").split())
            if not name or len(name.split()) > 4 or _key(name) in {_key(n) for n in self.cast}:
                continue
            self.cast[name] = {k: " ".join(str(said.get(k) or "").split())[:160] for k in ("what", "looks", "side")}
            added.append(name)
        return added

    def as_seen(self, kind: int) -> str:
        """What a kind is called by what it was seen as: the cast member it is, else what it looked like; "" if neither."""
        return self.who.get(kind) or self.by_kind.get(kind, "")

    def for_thinking(self) -> str:
        """What she sees and who is who, for reasoning with."""
        lines: list[str] = []
        store = what_things_are()
        seen = []
        for kind, what in list(self.by_kind.items())[:8]:
            known = store.of(what)
            bears = f" ({known.bears}{': ' + known.usually if known.usually else ''})" if known and known.bears != "unsure" else ""
            odd = f" [looks odd: {self.odd[kind]}]" if kind in self.odd else ""
            seen.append(f"{self.as_seen(kind)}{bears}{odd}")
        if seen:
            lines.append("What I see here: " + "; ".join(seen))
        cast = [f"{name} — {c.get('what') or '?'}" + (f", {c['side']}" if c.get("side") else "")
                for name, c in list(self.cast.items())[:6]]
        if cast:
            lines.append("Who's who, from what I know: " + "; ".join(cast))
        return "\n".join(lines)
