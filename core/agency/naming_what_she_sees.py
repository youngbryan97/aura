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

import numpy as np

from core.agency.what_meeting_things_does import AVOID, MEET, SHOOT
from core.perception.reading_again_after_loss import ReadingAgain
from core.perception.what_things_look_like import EVERY_S, LookingAtThings

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
    scoped = guide is not None and guide.seen.instances.active(moves)
    instance = guide.seen.instances.of(moves, sample) if scoped and thing is not None else None
    seen = ((instance.who or instance.what) if instance is not None else '' if scoped and thing is not None
            else guide.seen.as_seen(kind) if guide is not None else '')
    if seen:
        return seen if (instance is not None and instance.who) or kind in guide.seen.who or colour in seen.split() else f"{colour} {seen}"
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


def still_looking_at(run: Any, kind: int | None, at: float, mine: int | None = None) -> bool:
    """Whether her eyes have been asked what a kind is a moment ago and have not said yet, or are yet to look at it
    where she is (``mine``) after looking at another thing like her: what she is is said once, so it waits a little."""
    looking = getattr(getattr(run, "seeing", None), "looking", None)
    if looking is None or kind is None or looking.resting() or at - looking.asked_at >= WAIT_FOR_EYES_S + EVERY_S:
        return False
    if isinstance(looking, LookingAtThings) and mine is not None:
        queries = [key for key in looking.asked_instances if key[2] == mine and key[4] == kind]
        if queries:
            latest = max(queries, key=lambda key: key[3])
            return latest not in looking.by_instance and at - looking.asked_at < WAIT_FOR_EYES_S
    if mine is not None and kind in looking.looked_on and looking.looked_on[kind] != mine:
        return True
    return kind not in looking.by_kind and kind in looking.asked and at - looking.asked_at < WAIT_FOR_EYES_S


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
MOST_STANCES_SAID = 3


def a_stance_said(run: Any, moves: Any, kind: int, stance: str, at: float, *, met: bool = True) -> str:
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
    # A listener knows a thing by its colour: two kinds of one colour said to be one worth getting and the other costly
    # are heard as one thing said both ways (offline 2026-10-10 "I'll keep clear of the cyan things", then "The cyan bars
    # are worth getting to"). The second is kept to herself.
    colour = _the_colour_in(name)
    if colour and any(_the_colour_in(said) == colour and earlier != stance for said, earlier in seeing.stances_said.items()):
        return ""
    proper = name[:1].isupper()
    them = name if proper else f"the {plural(name)}"
    if stance == AVOID and not met:
        # What she was told or supposes, not what has cost her, is said as what she means to do: LIVE 2026-10-10 "The
        # traps cost me" a second into a game's first screen, of traps its words had named and nothing had touched.
        line = f"I'll keep clear of {them}."
    elif proper:
        line = {MEET: f"{name} is worth getting to.", AVOID: f"{name} costs me. Keeping clear.",
                SHOOT: f"{name} is worth shooting."}.get(stance, "")
    else:
        line = {MEET: f"The {plural(name)} are worth getting to.", AVOID: f"The {plural(name)} cost me. Keeping clear of them.",
                SHOOT: f"The {plural(name)} are worth shooting."}.get(stance, "")
    if line:
        seeing.stances_said[name] = stance
        seeing.stance_said_at = at
    return line


def _the_colour_in(name: str) -> str:
    """The colour a name says its thing is, where it says one ("the cyan bars", "small red things"); else ""."""
    return next((word for word in str(name or "").lower().split() if word in _COLOURS), "")


#: What a thing can be only to a body moved about: somewhere to go, something to stand on, something to keep clear of.
_TO_A_BODY = frozenset({"to stand on", "to reach", "danger"})


def _not_moved_about(guide: Any) -> bool:
    """Whether the place is done by a way other than moving a body about, by what it says or what it is read to be."""
    from core.cognition.what_this_place_is import not_steered, said_not_steered

    try:
        return bool(said_not_steered(guide) or not_steered(guide))
    except (AttributeError, TypeError, ValueError):
        return False


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
    #: Which thing her eyes took for the player's own (core/perception/where_i_am_on_screen.py): the look, its answer,
    #: when play began, and whether it has been said.
    avatar_asking: Any = None
    avatar: Any = None
    avatar_observation: Any = None
    avatar_renewal: ReadingAgain = field(default_factory=ReadingAgain)
    began: float | None = None
    avatar_said: bool = False


def seen_in_play(run: Any, moves: Any, hers: Any, meeting: Any, picture: Any, at: float,
                 tell: Callable[[str, str], Any], *, ask: Any = None) -> None:
    """One picture: her eyes asked what the kinds not yet named are (beside her play), and what each turned out to be
    made something of: its name in the guide, a stance until meeting it says, a thought said, an odd look wondered at."""
    from core.cognition.what_things_are import what_things_are

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
    if isinstance(seeing.looking, LookingAtThings):
        guide.seen.instances.begin(moves)
    seeing.looking.look(picture, moves, at, mine=hers.number, met=list(meeting.evidence))
    # What the place's code says of what she sees, before her eyes' own guesses (core/agency/what_its_code_says_she_sees.py).
    from core.agency.what_its_code_says_she_sees import what_its_code_says_she_sees

    what_its_code_says_she_sees(run, moves, hers, meeting, at, tell, guide)
    _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, picture, at, tell)
    for kind, sighting in seeing.looking.newly():
        # A name her eyes give a thing is its name where the place's words have it: LIVE 2026-10-11 her eyes called
        # things on a game's title "Big Shot Checkers", the game's own name, and "Littlest Pet Shop", which nothing
        # there says.
        named = getattr(sighting, "who", "")
        if sighting.query is not None:
            who = guide.seen.saw_instance(moves, sighting.query, sighting.what, sighting.odd,
                                         named if guide.has_said(named) else '')
            if who is None:
                continue
        else:
            who = guide.seen.saw(kind, sighting.what, sighting.odd, named if guide.has_said(named) else "")
        colour = colour_name(moves.kinds[kind].colour) if kind < len(moves.kinds) else ""
        common = guide.seen.instances.consensus(moves, kind) if sighting.query is not None else None
        if kind < len(moves.kinds) and (sighting.query is None or common is not None):
            # How it looks to her play, so where it is can be found by other ways of looking (a send's).
            looked = moves.kinds[kind]
            guide.reading.saw(who or sighting.what, looked.colour, float(getattr(looked, "size", 0.0) or 0.0))
        instance_key = f'{kind}:{sighting.query.number}:{sighting.query.born}' if sighting.query is not None else str(kind)
        guide.notes.notice(f"looks like {instance_key}", f"the {colour} one looks like {_a(who or sighting.what)}", at,
                           kind="seen", about=str(kind))
        if ask is not None:
            store.ask_about([sighting.what], ask)
        seeing.pending[kind] = at
        if sighting.odd:
            _wonder_at(guide, seeing, who or sighting.what, sighting.odd, at, ask)
    for kind in guide.seen.refresh_instances(moves):
        meeting.supposed.pop(kind, None)
        seeing.pending.pop(kind, None)
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


#: Words that name only a shape, not what a thing is.
_ONLY_A_SHAPE = frozenset({"square", "squares", "circle", "circles", "rectangle", "rectangles", "triangle", "triangles",
                           "oval", "shape", "shapes", "object", "objects", "thing", "things", "blob", "dot", "dots",
                           "line", "lines", "bar", "bars", "block", "blocks", "tile", "tiles"})


def names_only_a_shape(what: str) -> bool:
    """Whether a name says only what shape a thing is ("a square", "red bars"), not what it is."""
    words = str(what or "").lower().split()
    return bool(words) and words[-1] in _ONLY_A_SHAPE


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
    # Where the place is done another way than by moving a body (building, making, a board), what a thing would be to
    # a body (a door to go through, a platform to stand on) bears on nothing she does: LIVE 2026-10-11 "That looks like a
    # door. Maybe I can go through it, or perhaps it's locked" over a game of building a machine.
    if _not_moved_about(guide) and bears in _TO_A_BODY:
        return
    if not any(t.kind == kind for t in moves.things.values()):
        return
    # A shape and nothing more ("a square", "a red bar") names nothing a watcher would want remarked on: offline
    # 2026-10-12 "That looks like a square. Maybe it's a platform I can walk on or a tile to avoid" in fourteen games.
    if names_only_a_shape(what):
        return
    thought = (cast.get("what") and f"{what}: {cast['what']}") or (known.thought if known else "") or _THOUGHT_OF[bears]
    line = f"That looks like {_a(what)}." + (f" {thought[:1].upper()}{thought[1:].rstrip('.')}." if thought else "")
    seeing.said += 1
    seeing.said_at = at
    tell(line, f"seen {what}")


#: How long into play her eyes are asked which thing is hers, and how long her keys are tried before what her eyes
#: took for her is taken for her where nothing has answered.
AVATAR_ASKED_AFTER_S = 2.0
AVATAR_TAKEN_AFTER_S = 8.0
#: How far, in working pixels, a press must move something the same way each time for the screen to answer her.
ANSWERED_BY_PX = 4.0


def _who_i_am_to_my_eyes(run: Any, seeing: SeeingInPlay, guide: Any, moves: Any, hers: Any, meeting: Any, picture: Any,
                         at: float, tell: Callable[[str, str], Any]) -> None:
    """Her eyes asked which thing is the player's own, told what the place calls her; what they point at
    said, kept for her keys to be tried on, and taken for her where her trials have found nothing that answers."""
    import asyncio

    from core.agency.a_body_seen_then import ABodySeenThen
    from core.perception.where_i_am_on_screen import where_the_avatar_is

    seeing.began = at if seeing.began is None else seeing.began
    if _no_body_of_hers(run, guide):
        return
    observation = seeing.avatar_observation
    measured = bool(getattr(moves, "seeing", False) and getattr(moves, "_last_at", None) == at)
    if observation is not None:
        observation.follow(moves, at, measured=measured)
    pending = seeing.avatar_asking is not None and not seeing.avatar_asking.done()
    usable = observation is not None and observation.thing(moves, at) is not None
    needed = getattr(hers, "number", None) is None and not getattr(hers, "follows_pointer", False)
    refresh = seeing.avatar_renewal.due(at, pending=pending, measured=measured, usable=usable, needed=needed)
    if refresh and at - seeing.began >= AVATAR_ASKED_AFTER_S and getattr(moves, "things", None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        names = [*guide.names.get("me", [])[:2],
                 *[name for name, said in guide.seen.cast.items() if said.get("side") == "you"][:1]]
        frame = np.array(picture)[..., :3]
        seeing.avatar_renewal.began(at)
        seeing.avatar = None
        observation = seeing.avatar_observation = ABodySeenThen(moves, at)

        async def looked() -> None:
            from core.cognition.guidance_lifetime import question_work

            with question_work() as stamp:
                avatar = await where_the_avatar_is(frame, names)
                stamp.check()
                seeing.avatar = avatar
                observation.bind(avatar)

        from core.cognition.guidance_lifetime import create_guidance_task, retire_when_closed

        seeing.avatar_asking = task = create_guidance_task(looked())

        def release() -> None:
            if seeing.avatar_asking is task:
                seeing.avatar_asking = None
                seeing.avatar_observation = None
                seeing.avatar_renewal = ReadingAgain()

        retire_when_closed(task, release)
        return
    if seeing.avatar is None:
        return
    observation = seeing.avatar_observation
    thing = observation.thing(moves, at) if observation is not None else None
    if thing is None:
        return
    what = seeing.avatar.what
    # What costs her when met is no body of hers, whatever it looks like.
    if meeting.stance(thing.kind) == AVOID and meeting.known(thing.kind):
        return
    known = getattr(hers, "number", None) is not None
    if not seeing.avatar_said:
        seeing.avatar_said = True
        guide.seen.mine = (thing.kind, what)
        # A first guess is said only while it is one: where play has found her already, what her eyes took is kept, not
        # said over what play showed.
        if not known and not _not_moved_about(guide):
            tell(f"That looks like me: {_the(what)}. I'll see if it answers to me.", "looks like me")
        else:
            logger.info("her eyes took %s for hers; play has her as %s", what, hers.number)
    keys = list(getattr(run, "keys", []) or [])
    tried = getattr(run, "trying", 0) >= 2 * max(1, len(keys))
    # Taken for her only where her presses are seen to move something: in a story playing out, or a screen of its
    # lesson, nothing answers her keys at all, and a figure in it is no body of hers (LIVE 2026-10-10, over a game's
    # opening scene: "That looks like me: Tom", then "Tom is worth getting to", "the traps cost me").
    presses = getattr(run, "presses", None)
    answered = presses is not None and presses.moves_anything(ANSWERED_BY_PX)
    if not known and tried and answered and at - seeing.began >= AVATAR_TAKEN_AFTER_S and not hers.follows_pointer:
        hers.seen_to_be_her(thing, at, what)


#: The keys that move a body about.
_MOVING_KEYS = frozenset({"up", "down", "left", "right", "w", "a", "s", "d"})


def _no_body_of_hers(run: Any, guide: Any) -> bool:
    """Whether the place puts no body of hers in the picture: a click is her shot, no key she has moves anything, and
    nothing is said to follow the pointer. Her eyes are not asked which one is her there: offline 2026-10-11 a food
    fight seen over a table had its opponent said to be her ("That looks like me: the muscular man")."""
    if guide is not None and hasattr(guide, "controls"):
        # As she understands the place (core/agency/how_a_game_works.py), where she understands what she is in it.
        from core.agency.how_a_game_works import understood

        body = understood(guide).has_a_body()
        if body is not None:
            return not body
    keys = set(getattr(run, "keys", ()) or ())
    moves = (getattr(guide, "pointer", None) or {}).get("moves")
    return bool(getattr(run, "pointer_trigger", False)) and not keys & _MOVING_KEYS and not moves


def _the(what: str) -> str:
    return what if what[:1].isupper() else f"the {re.sub(r'^(?:a|an|the)\s+', '', what)}"


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

    from core.cognition.asking_in_turn import A_LOOK
    from core.cognition.what_things_are import asked_patiently, what_things_are

    known = what_things_are().of(what)
    prompt = (f"In “{guide.place or 'a place on a screen'}”, something looks like {_a(what)}, but: {odd}."
              + (f" Such a thing usually: {known.usually}." if known is not None and known.usually else "")
              + " In one short, tentative sentence, what might that suggest about it there (what it does, whose side "
              "it is on, what to watch for)?")

    async def asked() -> None:
        try:
            got = await asked_patiently(ask, prompt, _Maybe, 120, matters=A_LOOK)
        except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
            logger.info("what an odd look suggests could not be asked: %s", str(why)[:120])
            return
        maybe = " ".join(str(getattr(got, "maybe", "") or "").split())
        if maybe:
            guide.notes.wonder(f"looks odd {what}", f"that {what} looks unusual ({odd}): {maybe}", at)
            logger.info("what an odd look suggests: %s — %s", what, maybe)

    loop.create_task(asked())
