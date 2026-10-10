"""What a place is: what it is about, what one does there and with what, who she is in it, what she wants out of it,
and what makes sense to try there; read from its name, what she sees in it, what it says, and what play shows.

A person opening something new takes in the whole of it at once, and what they take it for decides what they try. A
game named for a wrestling league that opens on wrestlers and a ring is a wrestling match: they are probably one of the
wrestlers, and win by beating the other. A basketball court is for playing basketball. A program named for a garage
band, all tracks and instruments, is for making music: what makes sense there is playing, recording and arranging, and
nobody tries to shoot a hoop in it. A name that suggests a quiet tale, beside monsters, swords and demons, is more
likely an adventure with fighting in it. A game named for hamsters in flight, showing hamsters and a giant slingshot, is
about launching the hamsters with the slingshot to make them fly: the slingshot is what to work, and pulling it back and
letting go is how.

And what is known outranks what is supposed. A program named for making music whose own instructions are all about
building robots is about building robots; a game whose keys turn out to move the cart, not the car, has her in the cart.
So each reading rests on the strongest kind of evidence it was made from: the name alone; the name and what she sees;
what the place itself says (its instructions, its page, what was found out about it); what play showed. A reading on
stronger evidence replaces one on weaker, never the other way round; what play showed replaces the part it showed, from
whatever source; and where a new reading disagrees with the last, she says so and goes with the stronger.

What a reading says is her own model's knowledge, asked beside her work (core/rebuilding/her_model.py), with its
answers held to what is there: which of the things she has seen she is, which she works with, how each bears on her,
and which of her own ways of playing (core/agency/ways_of_playing.py) the place is done by, makes sense in, or makes
none in. Then it is used: the way it is done by is offered after any the place's words ask for
(core/agency/the_way_it_is_played.py); what she works with is where a send is tried from first
(core/agency/playing_by_shots.py); what each thing is to her there is what she supposes of it until meeting it says
(core/agency/naming_what_she_sees.py); and the reading is in what she reasons with at every move
(core/cognition/a_guide_to_a_place.py).

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NAME", "PLAYED", "SAID", "SEEN", "Reading", "WhatThisPlaceIs", "ask_for_a_reading", "how_an_act_fits",
           "not_steered", "played_shows_hers", "rests_on_for", "where_seen", "ways_it_is_played_by"]

logger = logging.getLogger("Aura.WhatThisPlaceIs")

#: What a reading rests on, from the least sure to the surest.
NAME, SEEN, SAID, PLAYED = "its name", "its name and what I see", "what it says of itself", "what play showed"
_TRUST = {NAME: 1, SEEN: 2, SAID: 3, PLAYED: 4}

#: What a thing seen there may be to her.
ROLES = ("me", "what I work with", "danger", "a rival", "to get", "to reach", "a friend", "to use", "information",
         "scenery")
NONE_OF_THESE = "none of these"

#: Her ways that are acts in a place, by what each asks; the rest of core/agency/ways_of_playing.py are how she reads
#: a place (a guide, its bars, its counters) and are no way a place is done by.
_ACTS = ("steer", "shoot", "strike", "charge", "click things", "send", "type", "jump", "time a press", "use things",
         "carry", "remember what was shown", "switch", "keys shown", "copy a sequence", "a board in turns",
         "a grid's rule", "make", "stack", "a chain", "serve")
#: How each is said in a line a watcher hears.
_SAID_AS = {"steer": "moving about", "shoot": "aiming and firing", "strike": "hitting what's close", "charge":
            "holding and letting go", "click things": "clicking things", "send": "pulling back and letting go",
            "type": "typing", "jump": "jumping", "time a press": "pressing at the right moment", "use things":
            "picking things up and using them", "carry": "dragging things into place", "remember what was shown":
            "remembering what each showed", "switch": "switching between my own", "keys shown": "pressing the keys "
            "it shows", "copy a sequence": "doing again what it shows", "a board in turns": "taking turns on a board",
            "a grid's rule": "working out the grid", "make": "making something", "stack": "stacking things up",
            "a chain": "building a chain from start to end", "serve": "giving each what it asks for"}
#: Which of the ways her reflexes play each act is (core/agency/the_way_it_is_played.py); the rest are her moves on a
#: screen she reads and acts on (core/skills/screen_pursuit.py).
_REFLEX = {"send": "send", "type": "type", "copy a sequence": "copy a sequence", "keys shown": "copy a sequence",
           **{act: "as it happens" for act in ("steer", "shoot", "strike", "charge", "click things", "jump",
                                               "time a press", "switch")}}

#: How much two sayings of what a place is about must share to be one reading of it.
ALIKE = 0.25
#: How many sightings make a reading by what she sees worth asking again: a first few, and a fuller picture.
SEEN_AFTER = 2
SEEN_FULLER = 5
#: The most readings one place is asked for.
MOST_READINGS = 5
#: The most her model writes for a reading, and how many readings may go unanswered before none is asked again.
MOST_TOKENS = 600
UNANSWERED_AT_MOST = 3

_WORD = re.compile(r"[a-z]{3,}")
_COMMON = frozenset("""the and with for that this you your are its it's their there what who into from about game play
    playing played place most likely probably maybe person people one some thing things using use make made where
    which while them they then than have has had can will would could being been also just more get getting""".split())


@dataclass
class Reading:
    """One reading of a place, and what it rests on."""

    about: str = ""
    doing: str = ""
    hers: str = ""
    worked_with: str = ""
    by: str = ""
    want: str = ""
    makes_sense: tuple[str, ...] = ()
    no_sense: tuple[str, ...] = ()
    #: What each thing seen there is to her, by its seen name.
    roles: dict[str, str] = field(default_factory=dict)
    rests_on: str = NAME
    at: float = 0.0

    def as_memory(self) -> dict[str, Any]:
        return {"about": self.about, "doing": self.doing, "hers": self.hers, "worked_with": self.worked_with,
                "by": self.by, "want": self.want, "makes_sense": list(self.makes_sense), "no_sense": list(self.no_sense),
                "roles": dict(self.roles), "rests_on": self.rests_on}

    @classmethod
    def from_memory(cls, held: Any) -> Reading | None:
        if not isinstance(held, dict) or not held.get("about"):
            return None
        rests_on = str(held.get("rests_on") or NAME)
        return cls(**{k: str(held.get(k) or "") for k in ("about", "doing", "hers", "worked_with", "by", "want")},
                   makes_sense=tuple(held.get("makes_sense") or ()), no_sense=tuple(held.get("no_sense") or ()),
                   roles={str(k): str(v) for k, v in (held.get("roles") or {}).items()},
                   rests_on=rests_on if rests_on in _TRUST else NAME)


def _words(said: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(str(said or "").lower()) if w not in _COMMON}


def _alike_said(a: str, b: str) -> bool:
    """Whether two sayings of what a place is about are one reading: enough of their words shared."""
    one, other = _words(a), _words(b)
    if not one or not other:
        return True
    return len(one & other) / len(one | other) >= ALIKE


def _a(name: str) -> str:
    name = str(name or "").strip()
    return name if not name or name.lower().startswith(("the ", "a ", "an ")) else f"the {name}"


@dataclass
class WhatThisPlaceIs:
    """Her reading of one place as it stands, the readings before it, and what play has shown of it."""

    now: Reading | None = None
    before: list[Reading] = field(default_factory=list)
    #: What play showed, by the part of a reading it settles ("hers", "by"): outranks every reading.
    shown: dict[str, str] = field(default_factory=dict)
    #: How each seen name looks to her play: (colour, size) of the kinds seen as it.
    looks: dict[str, list[tuple[tuple[int, int, int], float]]] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)
    asking: Any = None
    #: Lines already said, so a reading taken again is not said again.
    said: set[str] = field(default_factory=set)
    #: What her eyes took the screens in as at a glance (core/perception/what_a_scene_is.py): settings, and what stood
    #: out in them; and the screens glanced at, by their words.
    scenes: list[str] = field(default_factory=list)
    stood_out: list[str] = field(default_factory=list)
    glanced: list[str] = field(default_factory=list)
    glancing: Any = None
    #: Readings asked for and not answered.
    unanswered: int = 0
    #: What each part of the reading rests on now, by part ("about", "hers", "role <name>").
    rests: dict[str, str] = field(default_factory=dict)
    #: What play has measured of the place, by what each fact is about ("crowd", "view", "control", "counted").
    measured: dict[str, str] = field(default_factory=dict)

    # -- taking a reading in -------------------------------------------------------------------------------------

    def take(self, reading: Reading) -> str:
        """A new reading, part by part: each part it has is kept where it rests on at least what that part rested on
        before, never on less; what play showed laid over all. What to say of it, "" for nothing new."""
        reading.at = reading.at or time.monotonic()
        last = self.now
        if last is not None:
            self.before.append(Reading(**{**last.__dict__, "roles": dict(last.roles)}))
            del self.before[:-8]
        now = self.now = self.now or Reading(rests_on=reading.rests_on, at=reading.at)
        trust = _TRUST[reading.rests_on]
        was = (now.about, self.rests.get("about", NAME))
        for part in _PARTS:
            value = getattr(reading, part)
            if value in ("", (), None) or trust < _TRUST[self.rests.get(part, NAME)] and getattr(now, part):
                continue
            setattr(now, part, value)
            self.rests[part] = reading.rests_on
        for name, role in reading.roles.items():
            if trust >= _TRUST[self.rests.get(f"role {name}", NAME)] or name not in now.roles:
                now.roles[name] = role
                self.rests[f"role {name}"] = reading.rests_on
        now.rests_on, now.at = self.rests.get("about", reading.rests_on), reading.at
        self._lay_shown_over()
        logger.info("what this place is (%s): %s", now.rests_on, now.as_memory())
        if not was[0] and now.about:
            return self._once(self._first_said())
        if was[0] and now.about != was[0] and not _alike_said(was[0], now.about):
            return self._once(f"From {was[1]} I'd taken it to be about {_lower(was[0])}; from {now.rests_on} it's more "
                              f"likely about {_lower(now.about)}, so I'm going by that.")
        return ""

    def measure(self, facts: dict[str, str]) -> None:
        """What play measures of the place (core/agency/what_play_measures_of_a_place.py), as it stands now."""
        self.measured = dict(facts)

    def show(self, part: str, value: str) -> str:
        """Play showed one part of what the place is (``hers``: the thing her keys move, by its seen name; ``by``: the
        way that paid): kept over every reading. What to say where it is not what she had taken it for."""
        value = " ".join(str(value or "").split())
        if part not in ("hers", "by") or not value or self.shown.get(part) == value:
            return ""
        self.shown[part] = value
        was = getattr(self.now, part, "") if self.now is not None else ""
        self._lay_shown_over()
        if was and was != value and not _alike_said(was, value):
            said = (f"I'd taken myself for {_a(was)}, but what my keys move is {_a(value)}." if part == "hers" else
                    f"I'd taken it to be done by {_SAID_AS.get(was, was)}, but {_SAID_AS.get(value, value)} is what pays.")
            return self._once(said)
        return ""

    def saw(self, name: str, colour: Sequence[int], size: float) -> None:
        """A kind of thing her play tracks was seen as ``name``: how it looks, for finding it again (``where_seen``)."""
        kept = self.looks.setdefault(" ".join(str(name).lower().split()), [])
        entry = (tuple(int(c) for c in colour[:3]), float(size))
        if entry not in kept:
            kept.append(entry)  # type: ignore[arg-type]
            del kept[:-4]

    def glimpsed(self, scene: str, things: Sequence[str]) -> None:
        """A screen taken in at a glance: its setting, and what stood out in it."""
        if scene and scene not in self.scenes:
            self.scenes.append(scene)
        self.stood_out += [t for t in things if t and t not in self.stood_out]
        del self.stood_out[:-12]

    def _lay_shown_over(self) -> None:
        if self.now is None:
            return
        for part, value in self.shown.items():
            setattr(self.now, part, value)

    def _once(self, line: str) -> str:
        if not line or line in self.said:
            return ""
        self.said.add(line)
        return line

    def _first_said(self) -> str:
        reading = self.now
        if reading is None or not reading.about:
            return ""
        parts = [f"From {reading.rests_on}, it looks like it's about {_lower(reading.about)}"]
        if reading.hers and reading.hers != NONE_OF_THESE:
            parts.append(f"I'm probably {_a(reading.hers)}")
        how = _SAID_AS.get(reading.by, "")
        if reading.worked_with and reading.worked_with not in (NONE_OF_THESE, reading.hers):
            parts.append(f"working {_a(reading.worked_with)}" + (f" by {how}" if how else ""))
        elif how:
            parts.append(f"mostly by {how}")
        line = ": ".join(parts[:1]) + (": " + ", ".join(parts[1:]) if len(parts) > 1 else "")
        return line + (f". What I want out of it: {_lower(reading.want).rstrip('.')}." if reading.want else ".")

    # -- using it ------------------------------------------------------------------------------------------------

    def role_of(self, name: str) -> str:
        """What a thing seen as ``name`` is to her there, where the reading says; "" where it does not."""
        if self.now is None:
            return ""
        from core.cognition.what_things_are import words_shared

        key = " ".join(str(name or "").lower().split())
        if not key:
            return ""
        # A reading made before anything was seen names things as its words do ("a wrestler"), not as she saw them.
        if key == self.now.hers.lower() or words_shared(key, self.now.hers):
            return "me"
        if key == self.now.worked_with.lower() or words_shared(key, self.now.worked_with):
            return "what I work with"
        return self.now.roles.get(key) or next((r for n, r in self.now.roles.items() if words_shared(key, n)), "")

    def acts(self) -> list[str]:
        """The ways the place is done by, most likely first, and none the reading says make no sense there."""
        if self.now is None:
            return []
        ways = [self.now.by, *self.now.makes_sense]
        return [w for w in dict.fromkeys(ways) if w in _ACTS and w not in self.now.no_sense]

    def for_thinking(self) -> str:
        """The reading, for reasoning with, with what it rests on."""
        reading = self.now
        if reading is None or not reading.about:
            return ""
        parts = [f"What this place is ({reading.rests_on}): about {_lower(reading.about)}"]
        if reading.doing:
            parts.append(f"one {_lower(reading.doing)}")
        if reading.hers and reading.hers != NONE_OF_THESE:
            parts.append(f"I'm {_a(reading.hers)}" + (" (play showed it)" if "hers" in self.shown else " (supposed)"))
        if reading.worked_with and reading.worked_with != NONE_OF_THESE:
            parts.append(f"worked with {_a(reading.worked_with)}")
        if self.acts():
            parts.append("done by " + ", ".join(self.acts()[:3]))
        if reading.no_sense:
            parts.append("makes no sense here: " + ", ".join(reading.no_sense[:4]))
        if reading.want:
            parts.append(f"what I want out of it: {_lower(reading.want)}")
        others = [f"{name} ({role})" for name, role in list(reading.roles.items())[:6] if role not in ("me", "scenery")]
        if others:
            parts.append("what things are to me here: " + ", ".join(others))
        if self.measured:
            parts.append("what play measures: " + "; ".join(self.measured.values()))
        return "; ".join(parts) + "."

    def as_memory(self) -> dict[str, Any]:
        return {"now": self.now.as_memory() if self.now is not None else None, "shown": dict(self.shown),
                "rests": dict(self.rests)}

    @classmethod
    def from_memory(cls, held: Any) -> WhatThisPlaceIs:
        place = cls()
        if isinstance(held, dict):
            place.now = Reading.from_memory(held.get("now"))
            place.shown = {str(k): str(v) for k, v in (held.get("shown") or {}).items()}
            place.rests = {str(k): str(v) for k, v in (held.get("rests") or {}).items() if str(v) in _TRUST}
            place._lay_shown_over()
        return place


#: The parts of a reading, each kept with what it rests on.
_PARTS = ("about", "doing", "hers", "worked_with", "by", "want", "makes_sense", "no_sense")


def _lower(text: str) -> str:
    from core.language.words_of_the_language import as_said_inside

    return as_said_inside(text)


def ways_it_is_played_by(guide: Any) -> list[str]:
    """The ways her reflexes play that the reading of the place says it is done by, most likely first
    (core/agency/the_way_it_is_played.py)."""
    reading = getattr(guide, "reading", None)
    acts = reading.acts() if isinstance(reading, WhatThisPlaceIs) else []
    return list(dict.fromkeys(_REFLEX[act] for act in acts if act in _REFLEX))


def not_steered(guide: Any) -> str:
    """Where the reading of the place says moving something about makes no sense there and it is done another way
    (making music, putting parts in place), that way; else "". A playhead running or a scene moving in such a place is
    no world to steer."""
    reading = getattr(getattr(guide, "reading", None), "now", None)
    if reading is None or "steer" not in reading.no_sense or reading.by in ("", *_REFLEX):
        return ""
    return reading.by


def done_by_its_lesson(guide: Any) -> str:
    """Where the place's lesson, as read (core/cognition/reading_the_rules.py), asks only for acts that are not played
    as things happen (carrying parts into place, building a chain, making), the first of them; else "". Clicking is how
    any screen is worked and says nothing either way. LIVE 2026-10-10 a game of devices dragged into a room was played
    as it happened, because its title moved and a key was named: "The traps cost me. Keeping clear of them."."""
    rules = getattr(guide, "rules", None)
    steps = rules.steps() if rules is not None else []
    acts = [f.act for f in steps if f.act != "click things"]
    if not acts or any(act in _REFLEX for act in acts):
        return ""
    return acts[0]


#: Keys that move a body about: steering.
_STEERING_KEYS = frozenset({"up", "down", "left", "right", *"wasd"})


def how_an_act_fits(guide: Any, label: str, key: str) -> tuple[bool, bool]:
    """Whether an act (a click on ``label``, or the key ``key``) fits what the reading of the place says it is, and
    whether it makes no sense there. A click fits where its words are of what the place is about, what one does there,
    or what she works with; steering keys fit where it is steered and make no sense where steering makes none."""
    reading = getattr(getattr(guide, "reading", None), "now", None)
    if reading is None:
        return False, False
    if key:
        steering = key in _STEERING_KEYS
        return steering and "steer" in (reading.by, *reading.makes_sense), steering and "steer" in reading.no_sense
    said = _words(" ".join((reading.about, reading.doing, reading.want, reading.worked_with)))
    return bool(said & _words(label)), False


def where_seen(guide: Any, moves: Any, *names: str) -> list[tuple[float, float]]:
    """Where things looking like what was seen as any of ``names`` are on the picture ``moves`` tracks, as shares,
    largest first."""
    from core.perception.what_moves_in_the_picture import KIND_ALIKE

    reading = getattr(guide, "reading", None)
    if not isinstance(reading, WhatThisPlaceIs) or not getattr(moves, "things", None):
        return []
    from core.cognition.what_things_are import words_shared

    # By the name it was seen as, or one sharing what it is ("giant slingshot" at a glance, "slingshot" in play).
    looks = [look for name in names if name for seen_as, kept in reading.looks.items()
             if seen_as == " ".join(name.lower().split()) or words_shared(seen_as, name) for look in kept]
    kinds = getattr(moves, "kinds", [])
    found = []
    for thing in moves.things.values():
        kind = kinds[thing.kind] if 0 <= thing.kind < len(kinds) else None
        if kind is not None and any(kind.like(colour, size) <= KIND_ALIKE for colour, size in looks):
            found.append((-thing.w * thing.h, moves.share(thing.x, thing.y)))
    return [place for _size, place in sorted(found)]


def rests_on_for(guide: Any) -> str:
    """What a reading of the place could rest on now: what it says of itself where its instructions, its page, its
    program or what was found out about it are in the guide; else what she sees, where she has seen things; else its
    name. A title screen's words alone say little of what a place is."""
    from core.cognition.a_guide_to_a_place import COUNSEL, PAGE, PROGRAM, TOLD

    if set(getattr(guide, "sources", ()) or ()) & {TOLD, PAGE, COUNSEL, PROGRAM}:
        return SAID
    reading = getattr(guide, "reading", None)
    if getattr(getattr(guide, "seen", None), "by_kind", None) or getattr(reading, "scenes", None):
        return SEEN
    return NAME


def _what_it_says(guide: Any) -> str:
    lines = [*(guide.goals[:2]), *(guide.win[:1]), *(guide.lose[:1])]
    controls = guide._controls_said() if hasattr(guide, "_controls_said") else ""
    return " ".join(" ".join([*lines, controls]).split())[:600]


def _seen(guide: Any) -> list[str]:
    """The things she has seen there: what her play's things were seen as, then what stood out at a glance."""
    seen = getattr(guide, "seen", None)
    names = [seen.as_seen(kind) for kind in list(getattr(seen, "by_kind", {}))[:12]] if seen is not None else []
    names += list(getattr(getattr(guide, "reading", None), "stood_out", []) or [])
    return [n for n in dict.fromkeys(" ".join(str(n).lower().split()) for n in names) if n][:14]


def ask_for_a_reading(guide: Any, ask: Callable[..., Awaitable[Any]] | None, *, task: str = "",
                      tell: Callable[[str], Any] | None = None) -> bool:
    """Ask her model for a reading of the place on what it can rest on now, beside her work, where one on that has not
    been asked; taken into the guide's reading when it comes, and what it says said. Whether asked."""
    reading = getattr(guide, "reading", None)
    if ask is None or not isinstance(reading, WhatThisPlaceIs) or not getattr(guide, "place", ""):
        return False
    if reading.asking is not None and not reading.asking.done():
        return False
    rests_on = rests_on_for(guide)
    seen = _seen(guide)
    # Asked again only on more to go on: stronger evidence, or what is seen grown from nothing to a few to a fuller
    # picture; never on weaker evidence than a reading already asked on.
    due = f"{rests_on}:{0 if len(seen) < SEEN_AFTER else 1 if len(seen) < SEEN_FULLER else 2}"
    stronger = all(_TRUST[rests_on] >= _TRUST[r.split(":")[0]] for r in reading.asked)
    if due in reading.asked or not stronger or len(reading.asked) >= MOST_READINGS or due == f"{SEEN}:0":
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    reading.asked.append(due)

    async def asked() -> None:
        got = await _a_reading(guide.place, task, rests_on, seen if len(seen) >= SEEN_AFTER else [],
                               _what_it_says(guide) if rests_on == SAID else "",
                               _what_play_showed(reading), ask, scenes=reading.scenes[-3:])
        if got is None:
            # Not answered: asked again when there is next more to go on, a few times at most.
            reading.unanswered += 1
            if reading.unanswered < UNANSWERED_AT_MOST and due in reading.asked:
                reading.asked.remove(due)
            return
        line = reading.take(got)
        if line and tell is not None:
            tell(line)
        notes = getattr(guide, "notes", None)
        if notes is not None and reading.before and line:
            notes.wonder("what this place is", line, time.monotonic())

    reading.asking = loop.create_task(asked())
    return True


def _what_play_showed(reading: WhatThisPlaceIs) -> str:
    parts = []
    if reading.shown.get("hers"):
        parts.append(f"the thing the person's keys move is the {reading.shown['hers']}")
    if reading.shown.get("by"):
        parts.append(f"what has worked is {_SAID_AS.get(reading.shown['by'], reading.shown['by'])}")
    # And what its screen does, as play measured it: the reading must fit these.
    parts += list(reading.measured.values())
    return "; ".join(parts)


async def _a_reading(place: str, task: str, rests_on: str, seen: list[str], says: str, showed: str,
                     ask: Callable[..., Awaitable[Any]], *, scenes: Sequence[str] = ()) -> Reading | None:
    """Her model's reading of a place, its choices held to what was seen and to her own ways; None where it gave none."""
    from typing import Literal

    from pydantic import BaseModel, Field, create_model

    from core.agency.ways_of_playing import WAYS

    acts = Literal[tuple(_ACTS)]  # type: ignore[valid-type]
    if seen:
        thing: Any = Literal[(*seen, NONE_OF_THESE)]  # type: ignore[valid-type]
        role = create_model("_Role", name=(Literal[tuple(seen)], ...), role=(Literal[ROLES], ...))  # type: ignore[valid-type]
    else:
        thing = str
        role = create_model("_Role", name=(str, Field(max_length=40)), role=(Literal[ROLES], ...))  # type: ignore[valid-type]
    schema: type[BaseModel] = create_model(
        "_Reading",
        about=(str, Field(default="", max_length=160, description="what it is most likely about")),
        doing=(str, Field(default="", max_length=160, description="what a person most likely does there")),
        hers=(thing, Field(default=NONE_OF_THESE if seen else "", description="which thing the person most likely is or controls")),
        worked_with=(thing, Field(default=NONE_OF_THESE if seen else "", description="which thing they most likely act with")),
        by=(Literal[(*_ACTS, NONE_OF_THESE)], Field(default=NONE_OF_THESE, description="the way it is mostly done by")),  # type: ignore[valid-type]
        want=(str, Field(default="", max_length=160, description="what a person wants to get out of it there")),
        makes_sense=(list[acts], Field(default_factory=list, max_length=4, description="ways that make sense there")),
        no_sense=(list[acts], Field(default_factory=list, max_length=4, description="ways that make no sense there")),
        roles=(list[role], Field(default_factory=list, max_length=8, description="what each thing is to the person")),
    )
    ways = "; ".join(f"{w.name}: {w.asks}" for w in WAYS if w.name in _ACTS)
    prompt = (f"Someone has just opened “{place}”" + (f" to {_the_task(task)}" if task else "") + ".\n"
              + (f"What it says of itself: {says}\n" if says else "")
              + (f"Its screens, at a glance: {'; '.join(scenes)}.\n" if scenes else "")
              + (f"What they see in it: {'; '.join(seen)}.\n" if seen else "")
              + (f"What playing it has shown: {showed}.\n" if showed else "")
              + "From what you know of places like it and of what its name and what is in it usually mean, read it as "
              "a person would on first opening it: what it is most likely about; what a person does there; which "
              "thing they are or control and which they act with; which of these ways it is mostly done by, which "
              "make sense there and which make none (" + ways + "); what they want out of it; and what each thing seen "
              "is to them. Where what it says of itself and what its name suggests differ, what it says is so. Be "
              "tentative where you are unsure.")
    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently

    try:
        # Her thinking at each move holds her model; a question beside her work waits its turn and asks again.
        got = await asked_patiently(ask, prompt, schema, MOST_TOKENS, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("a reading of %r could not be asked: %s", place, str(why)[:160])
        return None
    if got is None:
        return None
    said = got.model_dump() if hasattr(got, "model_dump") else dict(got)
    def pick(value: Any) -> str:
        return "" if str(value or "") == NONE_OF_THESE else " ".join(str(value or "").split())[:60]

    return Reading(about=" ".join(str(said.get("about") or "").split()), doing=" ".join(str(said.get("doing") or "").split()),
                   hers=pick(said.get("hers")), worked_with=pick(said.get("worked_with")), by=pick(said.get("by")),
                   want=" ".join(str(said.get("want") or "").split()),
                   makes_sense=tuple(dict.fromkeys(str(w) for w in said.get("makes_sense") or ())),
                   no_sense=tuple(dict.fromkeys(str(w) for w in said.get("no_sense") or ())),
                   roles={" ".join(str(r.get("name") or "").lower().split()): str(r.get("role") or "")
                          for r in said.get("roles") or () if isinstance(r, dict) and r.get("name")},
                   rests_on=rests_on, at=time.monotonic())


def _the_task(task: str) -> str:
    said = " ".join(str(task or "").split())
    said = re.sub(r"^(?:please\s+)?(?:go to\s+\S+\s+and\s+)?", "", said, flags=re.I).rstrip(".")
    return said[:120]


def played_shows_hers(guide: Any, kind: Any) -> str:
    """Play found her thing, of kind ``kind``: where it was seen as something, that is who she is there. What to say."""
    reading = getattr(guide, "reading", None)
    seen = getattr(guide, "seen", None)
    if not isinstance(reading, WhatThisPlaceIs) or seen is None or kind is None:
        return ""
    name = seen.as_seen(kind)
    return reading.show("hers", name) if name else ""
