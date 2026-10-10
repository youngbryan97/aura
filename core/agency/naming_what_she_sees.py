"""What she calls the things she plays among, and what she makes of them by what they are.

A thing is called what the place calls it for the part it plays ("the lunar lander", "the robot dogs"), where the
place's words name it; else what it looks like to her eyes ("the red car", "the zombie",
core/perception/what_things_look_like.py); else its colour and shape ("the green thing").

And what a thing looks like tells her something before she has met it, as it tells anyone: a zombie is probably
trouble, so she keeps clear of it until meeting it says otherwise; a coin is probably worth getting; a bat is probably
swung, maybe to defend herself; a platform is stood on (core/cognition/what_things_are.py). What looks unlike its
kind (a face on an orange, a weapon on a rabbit) she wonders about, in her own notes, and her model is asked what it
might suggest. None of it is sure, and what meeting a thing shows outranks all of it
(core/agency/what_meeting_things_does.py ``supposed``).

Nothing here knows a game.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.what_meeting_things_does import AVOID, MEET, SHOOT

__all__ = ["SeeingInPlay", "a_pointer_seen", "a_stance_said", "colour_name", "describe", "i_go_where_the_mouse_goes", "named_for_its_part", "plural", "seen_in_play", "shape_name", "still_looking_at",
           "where_on_screen"]

logger = logging.getLogger("Aura.NamingWhatSheSees")

_COLOURS = {
    "black": (20, 20, 20), "white": (240, 240, 240), "grey": (128, 128, 128),
    "red": (210, 40, 40), "orange": (240, 140, 30), "yellow": (240, 210, 50),
    "green": (60, 180, 70), "blue": (50, 110, 230), "cyan": (60, 210, 230),
    "purple": (150, 60, 200), "pink": (240, 110, 200), "brown": (140, 90, 45),
}


def colour_name(rgb: Sequence[int]) -> str:
    return min(_COLOURS, key=lambda name: sum((a - b) ** 2 for a, b in zip(rgb, _COLOURS[name], strict=True)))


def shape_name(w: float, h: float) -> str:
    long, short = max(w, h), max(1.0, min(w, h))
    if long / short >= 2.5:
        return "bar"
    return "thing"


def _the_guide() -> Any:
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    return THE_GUIDE.get()


def describe(moves: Any, kind: int, thing: Any = None) -> str:
    """A kind as it looks: what her eyes saw one as, with its colour ("red car"), or the one of the place's characters
    it is; else its colour and shape ("green thing")."""
    alike = [thing] if thing is not None else [t for t in moves.things.values() if t.kind == kind]
    if not alike or kind >= len(moves.kinds):
        return "other"
    sample = alike[0]
    colour = colour_name(moves.kinds[kind].colour)
    guide = _the_guide()
    seen = guide.seen.as_seen(kind) if guide is not None else ""
    if seen:
        return seen if kind in guide.seen.who or colour in seen.split() else f"{colour} {seen}"
    return f"{colour} {shape_name(sample.w, sample.h)}"


def named_for_its_part(moves: Any, kind: int, thing: Any = None, part: str = "") -> str:
    """A thing as the place calls it, for the part it plays ("me", "goal", "get", "avoid", "shoot"), where the guide to
    the place (core/cognition/a_guide_to_a_place.py) has a name for it; else as it looks."""
    looks = describe(moves, kind, thing)
    guide = _the_guide()
    if guide is None or not part or looks == "other":
        return looks
    # What follows the pointer is the place's character only where the mouse is said to move it; else a cursor.
    pointer = guide.controls.get("the pointer")
    if part == "me" and pointer is not None and pointer.act not in ("move", "steer", "guide", "drive", "fly", "walk", "run"):
        return looks
    # Still, a thing to get to is where she is to be brought; going about, a thing to get.
    still = thing is not None and not getattr(thing, "moved", True)
    alternatives = {"get": ("goal",) if still else (), "goal": ("get",)}.get(part, ())
    return guide.name_for("goal" if part == "get" and still else part, looks.split()[0], alternatives=alternatives) or looks


#: What a thing that goes where the mouse goes is called when it is the pointer itself, not a character it moves.
_A_POINTER = re.compile(r"\b(?:cursor|pointer|mouse|arrow|crosshairs?|cross-hairs?|reticle|hand)\b", re.I)


def a_pointer_seen(kind: int) -> str:
    """What her eyes saw a kind as, where that is the pointer itself (a cursor, a crosshair, a hand); else ""."""
    guide = _the_guide()
    seen = guide.seen.by_kind.get(kind, "") if guide is not None else ""
    return seen if _A_POINTER.search(seen) else ""


def still_looking_at(run: Any, kind: int | None, at: float) -> bool:
    """Whether her eyes have been asked what a kind is a moment ago and have not said yet: what she is is said once,
    so it waits a little for them."""
    looking = getattr(getattr(run, "seeing", None), "looking", None)
    if looking is None or kind is None or kind in looking.by_kind:
        return False
    return kind in looking.asked and at - looking.asked_at < WAIT_FOR_EYES_S and not looking.resting()


def i_go_where_the_mouse_goes(moves: Any, kind: int, mine: Any) -> str:
    """What she says of the thing that goes where the mouse goes: her pointer, where her eyes see one; else herself."""
    where = where_on_screen(moves, mine.x, mine.y)
    pointer = a_pointer_seen(kind)
    if pointer:
        return f"The {pointer} at the {where} is my pointer: it goes where the mouse goes."
    return f"That's me: the {named_for_its_part(moves, kind, mine, 'me')} at the {where}. It goes where the mouse goes."


def where_on_screen(moves: Any, x: float, y: float) -> str:
    sx, sy = moves.share(x, y)
    across = "left" if sx < 0.33 else "right" if sx > 0.67 else ""
    down = "top" if sy < 0.33 else "bottom" if sy > 0.67 else ""
    return " ".join(part for part in (down, across) if part) or "middle"


def plural(name: str) -> str:
    if name.endswith("s"):
        return name
    if name.endswith(("x", "z", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in tuple("aeiou"):
        return name[:-1] + "ies"
    return name + "s"


#: How long between two lines on what a kind of thing is worth, and how many of them a game.
STANCE_EVERY_S = 6.0
MOST_STANCES_SAID = 5


def a_stance_said(run: Any, moves: Any, kind: int, stance: str, at: float) -> str:
    """What to say of what meeting a kind has shown it is worth, or "" where it is not to be said now: a line a while
    after the last, a few a game, and never the opposite of what was said of a thing by the same name. LIVE 2026-10-10
    sixteen of them came in a minute of a crowded game, and "the red things cost me" was followed by "the red things are
    worth getting to" (two kinds, one colour)."""
    seeing: SeeingInPlay | None = getattr(run, "seeing", None)
    if seeing is None:
        seeing = run.seeing = SeeingInPlay()
    if at - seeing.stance_said_at < STANCE_EVERY_S or len(seeing.stances_said) >= MOST_STANCES_SAID:
        return ""
    sample = next((t for t in moves.things.values() if t.kind == kind), None)
    name = named_for_its_part(moves, kind, sample, {MEET: "get", AVOID: "avoid", SHOOT: "shoot"}.get(stance, ""))
    if name in seeing.stances_said:
        return ""
    proper = name[:1].isupper()
    line = ({MEET: f"{name} is worth getting to.", AVOID: f"{name} costs me. Keeping clear.",
             SHOOT: f"{name} is worth shooting."} if proper else
            {MEET: f"The {plural(name)} are worth getting to.", AVOID: f"The {plural(name)} cost me. Keeping clear of them.",
             SHOOT: f"The {plural(name)} are worth shooting."}).get(stance, "")
    if line:
        seeing.stances_said[name] = stance
        seeing.stance_said_at = at
    return line


def _a(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name if name[:1].islower() else name


# -- what she makes of what she sees ----------------------------------------------------------------------------------

#: What a kind of thing's bearing makes her do about it, until meeting it says.
_STANCE_OF = {"danger": AVOID, "to get": MEET, "to use": MEET, "to reach": MEET}
#: What a thing is to her in the reading of the place (core/cognition/what_this_place_is.py), as a bearing. A rival is
#: beaten by whatever the place is done by (passed in a race, fought in a ring), so it sets no stance of its own.
_BEARS_AS = {"danger": "danger", "to get": "to get", "to use": "to use", "what I work with": "to use",
             "to reach": "to reach", "a friend": "a friend", "information": "information", "scenery": "scenery"}
#: Bearings worth saying out loud on first seeing a thing: what she would act on.
_WORTH_SAYING = frozenset({"danger", "to get", "to use", "to stand on", "to reach", "a friend"})
#: How long between two things said of what she sees, and how many in one game; and how many odd looks she asks her
#: model about.
SAY_EVERY_S = 6.0
MOST_SAID = 5
MOST_WONDERED = 3
#: How long what a thing usually is waits on her model before a first guess is gone with; and how long what she is
#: waits on her eyes.
WAIT_FOR_HER_MODEL_S = 30.0
WAIT_FOR_EYES_S = 6.0
#: What she thinks of a thing by how it bears, where her model gave no thought of its own.
_THOUGHT_OF = {"danger": "Probably trouble; I'll keep clear of it until I know better.",
               "to get": "Probably worth getting.", "to use": "Maybe something I can use.",
               "to stand on": "Probably something to stand on.", "to reach": "Maybe where I'm meant to go.",
               "a friend": "Probably on my side.", "to press": "Probably something to press."}


@dataclass
class SeeingInPlay:
    """What she has asked her eyes and her model of the things in one game, and what she has made of their answers."""

    looking: Any = None
    #: Kinds seen whose bearing is still to be made something of, with when each was seen; and those made something of.
    pending: dict[int, float] = field(default_factory=dict)
    done: set[int] = field(default_factory=set)
    asked_names: bool = False
    said_at: float = -1e9
    said: int = 0
    wondered: int = 0
    #: Her kind as play last had it and since when; and whether who she is has been put right aloud this game.
    hers_kind: Any = None
    hers_since: float = 0.0
    hers_shown: bool = False
    #: What each thing, by the name it is said by, has been said to be worth, and when the last such line was said.
    stances_said: dict[str, str] = field(default_factory=dict)
    stance_said_at: float = -1e9


def seen_in_play(run: Any, moves: Any, hers: Any, meeting: Any, picture: Any, at: float,
                 tell: Callable[[str, str], Any], *, ask: Any = None) -> None:
    """One picture: her eyes asked what the kinds not yet named are (beside her play), and what each turned out to be
    made something of: its name in the guide, a stance until meeting it says, a thought said, an odd look wondered at."""
    from core.cognition.what_things_are import what_things_are
    from core.perception.what_things_look_like import LookingAtThings

    guide = _the_guide()
    seeing: SeeingInPlay | None = getattr(run, "seeing", None)
    if guide is None or seeing is None or picture is None:
        return
    if seeing.looking is None:
        seeing.looking = LookingAtThings()
    # Her model, as whoever began the work here gave it to the guide (core/cognition/what_i_know_of_a_place.py).
    ask = ask or getattr(guide, "ask_her_model", None)
    store = what_things_are()
    if not seeing.asked_names:
        # What the place's own words name (its robot dogs, its gems) says something too, before any is seen.
        seeing.asked_names = True
        named = [n for part in ("avoid", "get", "shoot", "goal", "friend") for n in guide.names.get(part, [])]
        # Not the place's characters: who they are is the cast's (its what and its side), not what such things are.
        if ask is not None:
            store.ask_about(named, ask)
    seeing.looking.look(picture, moves, at, mine=hers.number, met=list(meeting.evidence))
    for kind, sighting in seeing.looking.newly():
        who = guide.seen.saw(kind, sighting.what, sighting.odd, getattr(sighting, "who", ""))
        colour = colour_name(moves.kinds[kind].colour) if kind < len(moves.kinds) else ""
        if kind < len(moves.kinds):
            # How it looks to her play, so where it is can be found by other ways of looking (a send's).
            looked = moves.kinds[kind]
            guide.reading.saw(who or sighting.what, looked.colour, float(getattr(looked, "size", 0.0) or 0.0))
        guide.notes.notice(f"looks like {kind}", f"the {colour} one looks like {_a(who or sighting.what)}", at,
                           kind="seen", about=str(kind))
        if ask is not None:
            store.ask_about([sighting.what], ask)
        seeing.pending[kind] = at
        if sighting.odd:
            _wonder_at(guide, seeing, who or sighting.what, sighting.odd, at, ask)
    _read_the_place(guide, seeing, hers, at, ask, tell)
    for kind, since in sorted(seeing.pending.items()):
        what = guide.seen.as_seen(kind)
        cast = guide.seen.cast.get(guide.seen.who.get(kind, ""), {})
        seen_as = guide.seen.by_kind.get(kind, "")
        known = store.of(seen_as)
        # Her model is waited for a while, where it is being asked: a first guess has no thought to say.
        if not cast and (known is None or known.source != "her model") and seen_as in store.asking \
                and at - since < WAIT_FOR_HER_MODEL_S:
            continue
        del seeing.pending[kind]
        seeing.done.add(kind)
        _make_of(run, seeing, guide, moves, hers, meeting, kind, what, known, cast, at, tell)


def _make_of(run: Any, seeing: SeeingInPlay, guide: Any, moves: Any, hers: Any, meeting: Any, kind: int, what: str,
             known: Any, cast: dict[str, str], at: float, tell: Callable[[str, str], Any]) -> None:
    """What a kind turned out to be, made something of: a stance until meeting it says, and a thought said once."""
    side = cast.get("side", "")
    # What it is to her in this place, as the reading of the place has it, before what such things are anywhere.
    role = guide.reading.role_of(guide.seen.by_kind.get(kind, "") or what)
    bears = (_BEARS_AS.get(role) or ("danger" if side == "against you" else "a friend" if side == "with you" else "")
             or (known.bears if known else ""))
    mine = kind == hers.kind or side == "you" or role == "me"
    stance = _STANCE_OF.get(bears)
    if stance is not None and not mine and kind not in meeting.told:
        meeting.supposed[kind] = stance
        logger.info("what she supposes of kind %s (%s, %s): %s", kind, what, bears, stance)
    if mine or bears not in _WORTH_SAYING or seeing.said >= MOST_SAID or at - seeing.said_at < SAY_EVERY_S:
        return
    if not any(t.kind == kind for t in moves.things.values()):
        return
    thought = (cast.get("what") and f"{what}: {cast['what']}") or (known.thought if known else "") or _THOUGHT_OF[bears]
    line = f"That looks like {_a(what)}." + (f" {thought[:1].upper()}{thought[1:].rstrip('.')}." if thought else "")
    seeing.said += 1
    seeing.said_at = at
    tell(line, f"seen {what}")


#: How long play must have held the same thing to be hers before that is who she is, over any reading.
HERS_HELD_S = 8.0


def _read_the_place(guide: Any, seeing: SeeingInPlay, hers: Any, at: float, ask: Any,
                    tell: Callable[[str, str], Any]) -> None:
    """What she sees, put to the reading of what the place is (core/cognition/what_this_place_is.py): read again, beside
    her play, where there is more to go on; and the thing her keys move, once play has held it for a while and it has
    been seen as something, kept as who she is there over any reading. Who she is is put right aloud once a game: LIVE
    2026-10-10 play's guess went from one thing to another and back, and each change was said ("I'd taken myself for the
    letter, but what my keys move is the barrel", then the other way). A cursor she points with is not who she is."""
    from core.cognition.what_this_place_is import ask_for_a_reading, played_shows_hers

    kind = getattr(hers, "kind", None) if getattr(hers, "number", None) is not None else None
    if kind != seeing.hers_kind:
        seeing.hers_kind, seeing.hers_since = kind, at
    held = kind is not None and at - seeing.hers_since >= HERS_HELD_S
    pointer = bool(getattr(hers, "follows_pointer", False) and _A_POINTER.search(guide.seen.by_kind.get(kind, "")))
    said = played_shows_hers(guide, kind) if held and not pointer else ""
    if said and not seeing.hers_shown:
        seeing.hers_shown = True
        tell(said, f"who I am: {said[:80]}")
    ask_for_a_reading(guide, ask, task=" ".join(guide.goals[:1]), tell=lambda line: tell(line, f"what this place is: {line[:80]}"))


def _wonder_at(guide: Any, seeing: SeeingInPlay, what: str, odd: str, at: float, ask: Any) -> None:
    """A thing that looks unlike its kind, wondered at in her notes; and her model asked, beside her play, what that
    might suggest, a few times a game."""
    import asyncio

    guide.notes.wonder(f"looks odd {what}", f"that {what} looks unusual ({odd}); I wonder why", at)
    if ask is None or seeing.wondered >= MOST_WONDERED:
        return
    seeing.wondered += 1
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    from pydantic import BaseModel, Field

    class _Maybe(BaseModel):
        maybe: str = Field(default="", max_length=200, description="one short, tentative sentence; empty if nothing")

    from core.cognition.what_things_are import asked_patiently, what_things_are

    known = what_things_are().of(what)
    prompt = (f"In “{guide.place or 'a place on a screen'}”, something looks like {_a(what)}, but: {odd}."
              + (f" Such a thing usually: {known.usually}." if known is not None and known.usually else "")
              + " In one short, tentative sentence, what might that suggest about it there (what it does, whose side "
              "it is on, what to watch for)?")

    async def asked() -> None:
        try:
            got = await asked_patiently(ask, prompt, _Maybe, 120)
        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
            logger.info("what an odd look suggests could not be asked: %s", str(why)[:120])
            return
        maybe = " ".join(str(getattr(got, "maybe", "") or "").split())
        if maybe:
            guide.notes.wonder(f"looks odd {what}", f"that {what} looks unusual ({odd}): {maybe}", at)
            logger.info("what an odd look suggests: %s — %s", what, maybe)

    loop.create_task(asked())
