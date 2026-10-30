"""Playing by shots: pulling or holding a press and letting go, watching where it sends something, and doing better.

A world that waits until she sends something into it (a putt, a toss, a
launch, a top let go into a ring) is not played by steering. Each turn is one
act with a measure (core/agency/how_hard_and_which_way.py): she presses at a
place, pulls some way or holds some time, lets go, and watches what sets off,
where it goes and where it stops, and whether anything counted went up. Then
the next setting, from what the shots so far say.

What to send from: the place her own thing is, where she knows it, else the
shapes on the picture that stand out from what is around them, each tried in
turn until one sends something. What to send it to: the thing she has learned
is worth meeting, where there is one; else no place, and what pays is found by
searching the settings for what is counted.

Nothing here names a game, a key or a kind of thing.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from core.agency.how_hard_and_which_way import HOLD, PULL, Setting, Shot, Shots, first_settings
from core.runtime.wall_clock import monotonic_time

logger = logging.getLogger("Aura.PlayingByShots")

__all__ = ["hers_is_in_sight", "play_by_shots", "sends_by_letting_go"]

#: What a press that changed the whole screen is said to have sent: nothing, it was a button.
NAVIGATED: Any = object()

#: How long after letting go something must have set off for it to have been sent, and the longest a shot is
#: watched before what it sent is taken to have stopped where it is.
SETS_OFF_WITHIN_S = 1.0
WATCHED_FOR_S = 6.0

#: How slow, in working pixels a second, a thing must go for this long to have stopped.
STOPPED_BELOW = 6.0
STOPPED_FOR_S = 0.4

#: How many steps a pull is made in, so the page sees a drag and not a jump, and how long it is held at its end.
PULL_STEPS = 8
HELD_AT_FULL_PULL_S = 0.07

#: How long a thing sent may be out of sight before it is taken to have gone for good: a ball rolls behind a post, or
#: over something of its own colour, and comes out the other side.
BACK_IN_SIGHT_WITHIN_S = 0.8

#: How near its start, as a share of the picture's width, a thing sets off that is the thing she pressed.
SETS_OFF_FROM_THERE = 0.08

#: Words that say a thing is sent by a press pulled or held and let go. Not "drag": a thing dragged is carried to a
#: place and stays there (core/agency/putting_things_in_place.py); LIVE 2026-10-09 "click and drag it into place"
#: had her letting go of shots at a title screen for a minute. Not "aim" or "power" alone: a shooter is aimed with
#: the mouse and a platformer has power-ups, and "Mouse to aim" sent her into shots on a game's menu the same day.
_SENDING_WORDS = ("pull", "release", "let go", "launch", "fling", "sling", "putt", "power meter", "power bar")

#: Words for a press held: a send only where what it is held for is aiming or gathering strength to let go of. Held to
#: use something for as long as it is held (glide, fly, speed up), it is no send: LIVE 2026-10-09 a game said "while
#: flying, click and hold your mouse button to use Glide Power", and she pulled and let go two hundred times.
_HELD = re.compile(r"\b(?:hold the mouse|click and hold|hold down the mouse|hold the button)\b")
_HELD_TO_SEND = re.compile(r"\b(?:aim\w*|release|let go|power up|shot|charge|strength|how hard|shoot|throw|putt\w*|swing)\b")

#: Words for sending that a click alone may do: "click to throw" is a shot aimed by the pointer, made at once
#: (core/agency/playing_as_it_happens.py); a throw held, pulled or let go is sent. LIVE 2026-10-09 a food fight's
#: "aim with your mouse and click to throw" was read as a world that waits for something sent.
_THROWN = re.compile(r"\b(?:throw|toss)\w*")
_A_CLICK_THROWS = re.compile(r"\bclick\w*(?:\s+\w+){0,3}\s+to\s+(?:throw|toss|launch|fling)", re.I)


def _surface() -> Any:
    """The learned surface over the floor below: sentences that tell a player to send something by letting go of a press,
    declared with the phrasings the floor gets wrong both ways (core/language/learned_matcher.py)."""
    from core.language.learned_matcher import LearnedMatcher
    from core.language.model_features import model_hidden_features

    return LearnedMatcher(
        name="sends_by_letting_go",
        positives=("Pull back and release to launch the ball.", "Click and hold to aim, then release to shoot.",
                   "Drag the slingshot back, then let go.", "Hold the mouse button to build up power, then let go to putt.",
                   "The farther you pull back, the harder it flies.", "Drag back to set the angle and power, release to fire."),
        negatives=("The game was released in 2002 by Cartoon Network.", "The rocket launches when the timer hits zero.",
                   "Click to throw a snowball at the other team.", "While flying, click and hold to use Glide Power.",
                   "Use the arrow keys to move and space to jump.", "Drag the stickers onto the scene, then take a photo.",
                   "Drag each part to where it attaches.", "Hold the space bar to charge, then release for a super shot."),
        features=model_hidden_features)


SENDS_BY_LETTING_GO = _surface()


def sends_by_letting_go(told: str) -> bool:
    """Whether words a thing says of itself speak of sending by a press pulled or held and let go.

    The floor below decides the sentences it is sure of, and teaches the learned surface with each; a sentence the floor
    passes over is decided by the surface where it has decided one like it, and abstains otherwise.
    """
    lowered = "\n".join(" ".join(line.lower().split()) for line in told.splitlines())
    # A press of the pointer held, in a passage that says what it is held for: offline 2026-10-11 "HOW TO USE THE PUTTER:
    # 1. Click on your ball and hold down the mouse button." was read as a hold that sends nothing, and on the course
    # she clicked the words painted on it instead of putting.
    held_to_send = bool(_HELD_TO_SEND.search(lowered))
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", lowered):
        if held_to_send and _HELD.search(sentence) and not _a_key_held(sentence):
            SENDS_BY_LETTING_GO.observe(sentence, holds=True)
            return True
        if _a_key_held(sentence):
            SENDS_BY_LETTING_GO.observe(sentence, holds=False)
            continue
        if _the_floor_says_it_sends(sentence):
            SENDS_BY_LETTING_GO.observe(sentence, holds=True)
            return True
        if len(sentence.split()) >= 4 and _learned_it_sends(sentence):
            return True
    return False


#: A key held and let go, and the pointer: what a key held does is the key's (a charge, a jump held long), and a send is
#: a press of the pointer pulled or held. LIVE 2026-10-11 "Hold space bar to charge, then release for super shot" had
#: her pulling and letting go of the mouse at a flying shooter for the whole of a game.
_A_KEY_HELD = re.compile(r"\b(?:hold|holding|press and hold|keep)\s+(?:down\s+)?(?:the\s+)?(?:space\s*bar|space|"
                         r"(?:[a-z]|\w+)\s+key|keys?|arrows?(?:\s+keys?)?|shift|enter|ctrl)\b")
_THE_POINTER = re.compile(r"\b(?:mouse|click\w*|pointer|cursor|drag\w*|pull\w*|slingshot)\b")


def _a_key_held(sentence: str) -> bool:
    """Whether a sentence tells of a key held and let go, and not of the pointer."""
    return bool(_A_KEY_HELD.search(sentence)) and not _THE_POINTER.search(sentence)


def _learned_it_sends(sentence: str) -> bool:
    try:
        return bool(SENDS_BY_LETTING_GO.decide_without_waiting(sentence))
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return False


def _the_floor_says_it_sends(sentence: str) -> bool:
    """The floor: a sentence that names, in the forms a player is told them, a way of letting go that sends."""
    # A word on its own is a label, not a way of playing: LIVE 2026-10-10 a tunnel racer's start button said
    # "LAUNCH", and she pulled and let go at a game steered with the arrows.
    if len(sentence.split()) < 3:
        return False
    # "click again to launch the hamster" is a click that sends, made at once; "pull back and release to launch"
    # is a press let go.
    clicked = _A_CLICK_THROWS.search(sentence)
    # In the forms a player is told to do it: "release", "releasing", "pull back"; not a word that tells of
    # something done ("released in 2002", "the rocket launches", "pulled by gravity"): offline 2026-10-10 what was
    # looked up of a photo game said it was released, and she pulled and let go at it forty times.
    if any(_told_to(word, sentence) for word in _SENDING_WORDS if not (clicked and word in ("launch", "fling"))):
        return True
    if _HELD.search(sentence) and _HELD_TO_SEND.search(sentence):
        return True
    # A throw told ("throw the ball", "then throw it"), not a throw told of: LIVE 2026-10-11 what fans wrote of a
    # bowling game, that its hero throws bowling balls at pins, had her pulling and letting go at its level select.
    return bool(not clicked and any(_told_to(word, sentence) for word in ("throw", "toss")))


def _told_to(word: str, sentence: str) -> bool:
    """Whether ``sentence`` tells a player to do ``word``: the word where an instruction puts it, at the head of the
    sentence or of a clause, or after "then", "and", "or", "to" ("Pull back and release to launch", "then let go"), or
    "doing" it at the head of a sentence ("Pulling back further gives more power"). A thing named by it is not told:
    "the new extreme sport of hamster launching" (offline 2026-10-10, twenty-nine pulls at a game played by clicks)."""
    if " " in word and not word.startswith("let "):
        return bool(re.search(rf"\b{re.escape(word)}\b", sentence))
    doing = f"{word[:-1]}ing" if word.endswith("e") else f"{word}ing"
    head = r"(?:^|[,;:]\s*|\b(?:then|and|or|to|just|simply|now|you)\s+)"
    return bool(re.search(rf"{head}{re.escape(word)}\b", sentence) or re.search(rf"^{re.escape(doing)}\b", sentence))


async def _let_go(hands: Any, start: tuple[float, float], setting: Setting, way: str) -> None:
    """The act itself: press at ``start``, pull or hold, and let go."""
    import asyncio

    await hands.press(*start)
    try:
        if way == PULL:
            for step in range(1, PULL_STEPS + 1):
                share = step / PULL_STEPS
                await hands.point(start[0] + setting.across * share, start[1] + setting.down * share)
                await asyncio.sleep(0.015)
            # Held a moment at its full length: a place that reads the pull once a frame has to see it there.
            await asyncio.sleep(HELD_AT_FULL_PULL_S)
        else:
            await asyncio.sleep(setting.held_s)
    finally:
        await hands.release()


def _went_on_after(thing: Any, since: float) -> bool:
    """Whether a thing went on going after the button came up, half its own size at least: a thing carried under the
    pointer stops where it is let go, and is not sent. LIVE 2026-10-10 a device dragged across a puzzle's board was
    taken for a shot, and she set about finding how hard to throw it."""
    after = [(x, y) for at, x, y in getattr(thing, "path", ()) if at >= since]
    if len(after) < 3:
        return False
    return math.dist(after[0], after[-1]) >= max(3.0, 0.5 * max(float(thing.w), float(thing.h)))


async def _where_it_went(look: Callable[[], Awaitable[Any]], moves: Any, let_go_at: float,
                         read_words: Callable[[Any], list[dict[str, Any]]] | None, counters: Any,
                         going_already: frozenset[int] = frozenset(),
                         start: tuple[float, float] | None = None) -> tuple[Any, int, tuple[float, float] | None]:
    """Watch after a shot until what it set going has come to rest: where the last thing still going was last seen, as
    shares of the picture; what was counted meanwhile; and, where that thing went out of sight still going, where the
    watch had it by then.

    A thing already going before she pressed was not sent by her: LIVE 2026-10-09 a title's word bobbing on its
    own was taken for what her press sent, and she went on shooting at it. What a letting go sets going is not always
    the thing sent, nor at once: offline 2026-10-12 a putter swung for a second and more before it struck the ball,
    she had stopped watching after one, and she was pressing the next thing while her ball rolled away. So the watch
    lasts while anything the act set going is still going, and what it follows is whichever of those is going now:
    the club, and then the ball.
    """
    followed: int | None = None
    last: tuple[float, float] | None = None
    went_out_at: tuple[float, float] | None = None
    stirred_at = let_go_at
    lost_at: float | None = None
    gained = 0
    read_at = 0.0
    while True:
        seen = await look()
        if seen is None:
            break
        picture, at = seen
        happened = moves.see(picture, at)
        if any(event.get("what") == "new screen" for event in happened):
            # The whole screen changed under the press: a button was pressed, not a thing sent. After something was
            # sent it is what came of the shot (the next hole, the round's end), and the shot ended where it was seen.
            return (last if followed is not None else NAVIGATED), gained, went_out_at
        tall, wide = moves.shape
        going = _set_going_by_it(moves, let_go_at, going_already, start, followed)
        if going:
            thing = max(going, key=lambda one: math.hypot(one.vx, one.vy))
            now_at = (thing.x / max(1, wide), thing.y / max(1, tall))
            if thing.number != followed:
                logger.debug("after the letting go, a thing sets off at %s", now_at)
            followed, lost_at, stirred_at, went_out_at, last = thing.number, None, at, None, now_at
        elif followed is not None and followed not in moves.things and lost_at is None:
            # Out of sight is not gone: a ball rolls behind a post and comes out the other side. Where the watch had
            # it going by then is kept apart from where it was last seen whole, for whoever can tell which to go by.
            gone = next((h for h in happened if h.get("what") == "gone" and h.get("thing") == followed), None)
            if gone is not None:
                went_out_at = last
                last = (float(gone["last_visible"][0]) / max(1, wide), float(gone["last_visible"][1]) / max(1, tall))
            lost_at = at
        if lost_at is not None:
            if at - lost_at > BACK_IN_SIGHT_WITHIN_S:
                break
        elif at - stirred_at > (STOPPED_FOR_S if followed is not None else SETS_OFF_WITHIN_S):
            break
        if read_words is not None and counters is not None and at - read_at >= 0.5:
            read_at = at
            for verdict in counters.read(read_words(picture), at, None):
                gained += int(verdict["by"]) if verdict["what"] == "gain" else 0
        if at - let_go_at > WATCHED_FOR_S:
            break
    return last, gained, went_out_at


def _set_going_by_it(moves: Any, let_go_at: float, going_already: frozenset[int],
                     start: tuple[float, float] | None, followed: int | None) -> list[Any]:
    """The things going now that the letting go set going: not going before it, and gone on going after it. Those
    that set off from where she pressed, where any did: a club swung beside a ball is faster than the ball."""
    going = [thing for thing in moves.things.values() if thing.number not in going_already
             and math.hypot(thing.vx, thing.vy) > (STOPPED_BELOW if thing.number == followed else 2 * STOPPED_BELOW)
             and _went_on_after(thing, let_go_at)]
    if not going or start is None:
        return going
    tall, wide = moves.shape
    pressed = (start[0] * wide, start[1] * tall)
    from_there = [thing for thing in going
                  if math.dist(next(((x, y) for at, x, y in thing.path if at >= let_go_at), (thing.x, thing.y)),
                               pressed) <= SETS_OFF_FROM_THERE * wide]
    return from_there or going


def _a_way_on_shown(picture: Any, read_words: Callable[[Any], list[dict[str, Any]]] | None) -> str:
    """A label on the picture that goes on (Play Again, Next, Continue), or '': a screen that offers one is past the play.

    LIVE 2026-10-09 a game's clock ran out three shots into a stretch, and she let go of forty more at its "Game Over,
    Play Again" screen.
    """
    if read_words is None or picture is None:
        return ""
    from core.language.a_way_on import how_much_it_leads_on

    for piece in read_words(picture):
        label = " ".join(str(piece.get("text") or "").split())
        if 0 < len(label) <= 40 and how_much_it_leads_on(label) > 1.0:
            return label
    return ""


#: How many places are tried for what sends, the likeliest first.
PLACES_TRIED = 24

#: How near two places are, as a share of the picture, that are one place.
ONE_PLACE = 0.03

#: How like in colour (0 to 1) a thing must be to what she sent before to be that thing again, and to the marker by
#: her name to be marked by it.
THE_SAME_PAINT = 0.7
MARKED_BY = 0.5

#: How many shots in a row may send nothing before what sends is looked for afresh.
MISSES_BEFORE_LOOKING_AGAIN = 3

#: How many screens of play in a row may show nothing of what she sends before it is found afresh.
OUT_OF_SIGHT_AT_MOST = 3

#: How long after a thing of hers went from sight it, or another, is waited for before the place is left to say what
#: comes next. A thing taken up and carried off is set down again: offline 2026-10-12 a saucer over a golf course
#: lifted her ball, and three seconds on she was pressing the veins of an eyeball that were its colour.
ANOTHER_WITHIN_S = 8.0

#: How far apart in width a thing may be from the one she sent and still be it: the finder's sizes come in steps of
#: about a third, and a thing is found a step either side of itself from one look to the next.
THE_SAME_SIZE = 1.7

#: The ways a stretch ends after which a thing struck where it lies is somewhere of the next screen's choosing.
_HERS_IS_ELSEWHERE_AFTER = ("what she sent got there", "what she sent has gone from sight", "the screen changed",
                            "nothing of hers is in sight")

#: What sight says became of the thing she pressed.
RESTS, GONE, THERE, UNSEEN = "it lies somewhere new", "it is gone from sight", "one is where she pressed", "sight cannot say"

#: How often her thing is looked for while it comes to rest, how near sightings are that are one resting place, and
#: how many in a row must agree. Offline 2026-10-12 a ball still rolling the last of its way, a hundredth of the picture
#: between two looks, was taken to lie where it was, and the next press was at where it had been.
RESTS_LOOKED_EVERY_S = 0.3
AT_REST_WITHIN = 0.004
SIGHTINGS_AT_REST = 3

#: How near where she pressed a thing still is that has not moved at all. A short putt moves a ball less than the
#: breadth of one place, and it has moved.
NOT_MOVED_WITHIN = 0.006


def _things_on(picture: Any, marker: Any = None) -> list[Any]:
    """The small things that stand out on the picture (core/perception/things_that_stand_out.py): what is sent is
    nearly always one (a ball, a top, a bird, a ball of laundry). LIVE 2026-10-09 a putt game's ball, sat on its tee,
    was never among the places she pressed; offline 2026-10-11 it was still not, as wide as the window it was looked
    for through, and she pressed sixty other things. Not what is drawn on the marker by her name: that is a legend,
    and no thing of the field's."""
    from core.perception.things_that_stand_out import things_that_stand_out

    try:
        things = things_that_stand_out(picture)
    except (ValueError, TypeError, IndexError, MemoryError) as why:
        logger.info("the small things on the picture could not be looked for: %s", why)
        return []
    return [thing for thing in things if not marker.holds(thing.place)] if marker else things


def _her_marker(picture: Any, her_name: str, read_words: Callable[[Any], list[dict[str, Any]]] | None) -> Any:
    """The marker drawn beside the name she gave (core/perception/the_marker_by_a_name.py), or None."""
    if not her_name or read_words is None:
        return None
    from core.perception.the_marker_by_a_name import the_marker_by

    try:
        marker = the_marker_by(her_name, read_words(picture), picture)
    except (ValueError, TypeError, IndexError, KeyError) as why:
        logger.info("the marker by her name could not be read: %s", why)
        return None
    if marker:
        logger.info("beside her name (%s) the place draws %s", her_name, marker.colours)
    return marker or None


def _what_the_place_is_worked_with(moves: Any) -> list[tuple[float, float]]:
    """Where the things the reading of the place says she works with, or is, are on the picture: where a send is tried
    from before anything that only stands out (core/cognition/what_this_place_is.py). A slingshot beside hamsters is
    pulled; a ball by a club is struck."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.what_this_place_is import where_seen

    guide = THE_GUIDE.get()
    reading = getattr(getattr(guide, "reading", None), "now", None)
    if reading is None:
        return []
    found: list[tuple[float, float]] = []
    for place in where_seen(guide, moves, reading.worked_with, reading.hers):
        if all(math.dist(place, other) > 0.05 for other in found):
            found.append(place)
    return found[:3]


def _places_to_send_from(picture: Any, keep: dict[str, Any], *, things: Sequence[Any] | None = None,
                         marker: Any = None, worked_with: Sequence[tuple[float, float]] = (),
                         not_from: Sequence[tuple[float, float]] = (),
                         chosen: Sequence[int] | None = None) -> list[tuple[float, float]]:
    """Where a press might send something from, the likeliest first: where it did before; a thing painted like what
    she sent before; a thing painted like the one she chose when the place asked her to choose; a thing painted like
    the marker by her name; what the reading of the place says she works with; then whatever stands out, what stands
    out most first. Never where a press changed the whole screen, which is a button, nor where shots have just sent
    nothing, nor the marker itself, which is a legend."""
    from core.perception.shapes_that_look_pressable import pressable_shapes
    from core.perception.the_marker_by_a_name import how_like_the_marker
    from core.perception.things_that_stand_out import alike_in_colour

    things = _things_on(picture, marker) if things is None else things
    sent_before = keep.get("sends_what")
    known = keep.get("sends_from")
    buttons = [tuple(place) for place in keep.get("navigates_from") or ()]

    def hers(thing: Any) -> float:
        if sent_before is not None and alike_in_colour(thing.colour, tuple(sent_before)) >= THE_SAME_PAINT:
            return 3.0
        picked = alike_in_colour(thing.colour, tuple(chosen)) if chosen is not None else 0.0
        if picked >= MARKED_BY:
            return 1.0 + picked
        marked = how_like_the_marker(thing.colour, marker) if marker else 0.0
        return marked if marked >= MARKED_BY else 0.0

    likely = sorted((thing for thing in things if hers(thing) > 0.0), key=lambda thing: -hers(thing))
    others = [thing for thing in things if hers(thing) == 0.0]
    shapes = [(float(shape.get("center_x", 0.5)), float(shape.get("center_y", 0.5))) for shape in pressable_shapes(picture)]
    places: list[tuple[float, float]] = []
    for place in [*([tuple(known)] if known else []), *(thing.place for thing in likely), *worked_with,
                  *(thing.place for thing in others), *shapes, (0.5, 0.5)]:
        if all(math.dist(place, other) > ONE_PLACE for other in places):
            places.append(place)
    return [place for place in places if all(math.dist(place, shut) > 0.05 for shut in buttons)
            and all(math.dist(place, shut) > ONE_PLACE for shut in not_from)][:PLACES_TRIED]


def _tries_to_find_what_sends(places: Sequence[tuple[float, float]], ways: Sequence[str],
                              aim: tuple[float, float] | None) -> list[tuple[tuple[float, float], str, Setting]]:
    """The tries that find what sends, in order: every place once before any place twice. One pull at each, then one
    long hold at each, then the other pull and the short hold. Offline 2026-10-11 four tries at a place before the next
    took a minute and a half to come round fifteen places, and the game's clock is the same clock."""
    tries = []
    for way, which in ((PULL, 0), (HOLD, 1), (PULL, 1), (HOLD, 0)):
        if way not in ways:
            continue
        for place in places:
            numbers = first_settings(way, aim, place)
            tries.append((place, way, Setting.of(way, numbers[min(which, len(numbers) - 1)])))
    return tries


def _what_is_at(things: Sequence[Any], place: tuple[float, float]) -> Any:
    """The thing that stands out at ``place``, or None where what is there is no such thing."""
    there = [thing for thing in things if math.dist(thing.place, place) <= max(ONE_PLACE, thing.wide)]
    return min(there, key=lambda thing: math.dist(thing.place, place)) if there else None


def _in(box: Sequence[float], place: tuple[float, float]) -> bool:
    left, top, right, bottom = box
    return left <= place[0] <= right and top <= place[1] <= bottom


def _painted_like(things: Sequence[Any], painted: Sequence[int], wide: float | None = None) -> list[Any]:
    """The things painted like ``painted`` and, where the width of the thing meant is known, about as wide."""
    from core.perception.things_that_stand_out import alike_in_colour

    return [thing for thing in things if alike_in_colour(thing.colour, tuple(painted)) >= THE_SAME_PAINT
            and (not wide or max(thing.wide, wide) <= THE_SAME_SIZE * min(thing.wide, wide))]


def _where_hers_is(things: Sequence[Any], keep: dict[str, Any],
                   not_from: Sequence[tuple[float, float]] = ()) -> tuple[float, float] | None:
    """Where to press this turn, or None where that is not known. A thing sent from a place that stays put (a sling,
    a tee a new ball is set on) is pressed there again. A thing struck where it lies is one thing, and is pressed
    where it lies now: the thing painted like it nearest where the last shot left it, on whatever screen it is.
    Offline 2026-10-11 she would have putted a ball once and pressed its empty tee ever after."""
    known, painted = keep.get("sends_from"), keep.get("sends_what")
    if painted is None or not keep.get("lies_where_struck"):
        return tuple(known) if known is not None else None
    like = [thing for thing in _painted_like(things, painted, keep.get("sends_wide"))
            if all(math.dist(thing.place, shut) > ONE_PLACE for shut in not_from)]
    if not like:
        return None
    near = keep.get("rests_at") or known
    return (min(like, key=lambda thing: math.dist(thing.place, tuple(near))) if near is not None else like[0]).place


def hers_is_in_sight(picture: Any, keep: dict[str, Any], her_name: str = "",
                     read_words: Callable[[Any], list[dict[str, Any]]] | None = None) -> bool:
    """Whether the one thing she has been sending in this place, struck where it lies, is on ``picture``: a new hole, a
    new table, another go. Not its picture on the marker by her name."""
    painted = (keep or {}).get("sends_what")
    if painted is None or not keep.get("lies_where_struck"):
        return False
    return bool(_painted_like(_things_on(picture, _her_marker(picture, her_name, read_words)), painted,
                              keep.get("sends_wide")))


async def _what_became_of_it(look: Callable[[], Awaitable[Any]], before: Sequence[Any],
                             painted: Sequence[int] | None, start: tuple[float, float],
                             marked: Callable[[Any], Any] | None = None,
                             as_wide: float | None = None) -> tuple[str, tuple[float, float] | None]:
    """What became of the thing she pressed, by sight, and where it is where sight can say: RESTS at a place, where it
    is no longer where she pressed it and one thing of its paint lies where none was; GONE, where nothing of its paint
    is anywhere new and none is where she pressed; THERE, where one is where she pressed it, the same or another put
    in its place; UNSEEN where sight cannot say. A ball a few pixels wide is too small for the watch to follow and
    plain to find lying still, and a ball that stopped behind a post is where it is found, not where it was lost."""
    if painted is None:
        return UNSEEN, None
    # The others of its paint, where they were: what she pressed is not one of them.
    others = [thing for thing in _painted_like(before, painted, as_wide) if math.dist(thing.place, start) > ONE_PLACE]
    sightings: list[tuple[float, float]] = []
    began: float | None = None
    looked_at = -math.inf
    while True:
        seen = await look()
        if seen is None:
            return UNSEEN, None
        began = seen[1] if began is None else began
        if seen[1] - began > WATCHED_FOR_S:
            return UNSEEN, None
        if seen[1] - looked_at < RESTS_LOOKED_EVERY_S:
            continue    # the pictures' own clock, not hers: sightings a moment apart that agree are a thing at rest
        looked_at = seen[1]
        like = _painted_like(_things_on(seen[0], marked(seen[0]) if marked is not None else None), painted, as_wide)
        if any(math.dist(thing.place, start) <= NOT_MOVED_WITHIN for thing in like):
            return THERE, None
        new = [thing for thing in like if all(math.dist(thing.place, other.place) > ONE_PLACE for other in others)]
        if len(new) != 1:
            return (UNSEEN if new else GONE), None
        sightings.append(new[0].place)
        lately = sightings[-SIGHTINGS_AT_REST:]
        if len(lately) == SIGHTINGS_AT_REST and all(math.dist(place, lately[-1]) <= AT_REST_WITHIN for place in lately):
            return RESTS, lately[-1]


def _where_it_ended(became: tuple[str, tuple[float, float] | None], last: tuple[float, float] | None,
                    went_out_at: tuple[float, float] | None) -> tuple[float, float] | None:
    """Where a shot ended, from what the watch followed and what sight found afterwards. Where the thing lies in
    sight, there. Where it is gone from sight for good, where the watch had it going as it went: a ball lost at a
    hole's edge ended in the hole. Else where the watch last saw it whole: a thing lost beside something and then put
    back where it began went no further than it was seen to."""
    how, rests = became
    if how == RESTS:
        return rests
    if how == GONE and last is not None:
        return went_out_at or last
    return last


async def _hers_shows_again(look: Callable[[], Awaitable[Any]], painted: Sequence[int],
                            marked: Callable[[Any], Any] | None, as_wide: float | None = None) -> bool:
    """Whether a thing painted like hers is on the picture, now or within a few seconds: another to send."""
    began: float | None = None
    looked_at = -math.inf
    while True:
        seen = await look()
        if seen is None:
            return False
        began = seen[1] if began is None else began
        if seen[1] - looked_at >= RESTS_LOOKED_EVERY_S:
            looked_at = seen[1]
            if _painted_like(_things_on(seen[0], marked(seen[0]) if marked is not None else None), painted, as_wide):
                return True
        if seen[1] - began > ANOTHER_WITHIN_S:
            return False


async def _a_look_with_the_first_over(look: Callable[[], Awaitable[Any]], moves: Any) -> Any:
    """A picture, taken once the watch's first look at this screen is over and what stays put on it is known."""
    seen = await look()
    while seen is not None:
        moves.see(seen[0], seen[1])
        if moves.has_looked:
            break
        seen = await look()
    return seen


def _the_way_that_sends(keep: dict[str, Any], shots_by_way: dict[str, Shots], ways: Sequence[str]) -> str:
    """The way that sent from where she presses: the one kept, else the one that has sent lately, else the first."""
    kept = keep.get("sends_by")
    if kept in ways:
        return str(kept)
    return next((way for way in ways if way in shots_by_way and shots_by_way[way].sends_anything), ways[0])


async def play_by_shots(
    look: Callable[[], Awaitable[Any]],
    hands: Any,
    *,
    seconds: float,
    keep: dict[str, Any] | None = None,
    say: Callable[[str], Any] | None = None,
    read_words: Callable[[Any], list[dict[str, Any]]] | None = None,
    aim_at: Callable[[Any, tuple[float, float] | None], tuple[float, float] | None] | None = None,
    ways: Sequence[str] = (PULL, HOLD),
    her_name: str = "",
    chosen: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Send things by letting go of a press, shot after shot, for ``seconds``; doing better each time.

    ``hands`` has ``press(x, y)``, ``point(x, y)`` and ``release()``, places as shares of the picture. ``aim_at``
    says, of what ``look`` shows and of where her thing is on it (or None), where a thing sent ought to end up, or None
    where nothing says. ``her_name`` is the
    name she gave the place, where she gave one: what it draws beside that name marks which thing is hers. ``chosen``
    is the paint of the thing she took where the place asked her to choose one: the thing painted like it is hers. ``keep``
    carries what she found out (where a press sends from, what the thing sent is painted like, which way of letting go
    sends, the shots) into the next stretch.
    """
    from core.agency.what_meeting_things_does import Readouts
    from core.perception.what_moves_in_the_picture import WhatMoves

    keep = keep if keep is not None else {}
    began = monotonic_time()
    counters = Readouts()
    shots_by_way: dict[str, Shots] = keep.setdefault("shots", {})
    said: set[str] = set()

    def tell(line: str) -> None:
        if say is not None and line not in said:
            said.add(line)
            say(line)

    seen = await look()
    if seen is None:
        return {"shots": 0, "ended": "the picture could not be taken"}
    # One watch of the picture for all the shots, so what stays put is known before anything is sent.
    moves = WhatMoves(kinds=keep.get("kinds"))
    moves.see(seen[0], seen[1])
    marker: Any = None

    def marked(picture: Any) -> Any:
        return _her_marker(picture, her_name, read_words)

    to_try: list[tuple[tuple[float, float], str, Setting]] | None = None
    sent_nothing: list[tuple[float, float]] = []
    taken = gains = misses = sent = got_there = 0
    ended = "out of time"
    while monotonic_time() - began < seconds:
        seen = await _a_look_with_the_first_over(look, moves)
        if seen is None:
            ended = "the picture could not be taken"
            break
        # The marker by her name is read on every screen: it is on the course's plate and on the card of scores, in two
        # places. Offline 2026-10-12 the ball drawn by her name on a scorecard was taken for her ball, and pressed.
        marker = _her_marker(seen[0], her_name, read_words)
        things = [thing for thing in _things_on(seen[0], marker)
                  if not any(_in(box, thing.place) for box in keep.get("markers_at") or ())]
        hers = _where_hers_is(things, keep, sent_nothing)
        aim = aim_at(seen[0], hers) if aim_at is not None else None
        if hers is None and keep.get("lies_where_struck") and keep.get("sends_what") is not None:
            # She knows what she sends here, and it is not on this screen: nothing else is pressed to find out. But not
            # look after look: what she sends may look otherwise here (another hole's light, another ball), and after
            # a few screens of play without it, it is found afresh. Offline 2026-10-12 a golf ball known by its paint
            # was not found on the next hole, and shots were begun and given up forty times without one being taken.
            keep["out_of_sight"] = int(keep.get("out_of_sight") or 0) + 1
            if keep["out_of_sight"] < OUT_OF_SIGHT_AT_MOST:
                ended = "nothing of hers is in sight"
                break
            keep["out_of_sight"] = 0
            for known in ("sends_what", "sends_wide", "lies_where_struck", "sends_from", "rests_at", "first_sent_from"):
                keep.pop(known, None)
            to_try = None
            continue
        if hers is not None:
            keep["out_of_sight"] = 0
        if hers is None:
            if to_try is None:
                places = _places_to_send_from(seen[0], keep, things=things, marker=marker, not_from=sent_nothing,
                                              worked_with=_what_the_place_is_worked_with(moves), chosen=chosen)
                to_try = _tries_to_find_what_sends(places, ways, aim)
            if not to_try:
                ended = "nothing sends anything"
                break
            start, way, setting = to_try.pop(0)
        else:
            start, way = hers, _the_way_that_sends(keep, shots_by_way, ways)
            setting = shots_by_way.setdefault(way, Shots(way=way)).next_setting(aim, start)
        pressed = _what_is_at(things, start)
        painted, as_wide = (pressed.colour, pressed.wide) if pressed is not None else (None, None)
        going = frozenset(n for n, thing in moves.things.items() if math.hypot(thing.vx, thing.vy) > STOPPED_BELOW)
        await _let_go(hands, start, setting, way)
        # When it was let go, on the pictures' own clock: the time of the last picture before it, and the act's length.
        let_go_at = seen[1] + (setting.held_s if way == HOLD else PULL_STEPS * 0.015 + HELD_AT_FULL_PULL_S)
        ended_at, gained, went_out_at = await _where_it_went(look, moves, let_go_at, read_words, counters,
                                                             going_already=going, start=start)
        if ended_at is NAVIGATED:
            # LIVE 2026-10-09 a press on a game's rules screen went back to its title, the title's moving into place was
            # taken for a thing sent, and every shot after it pressed START.
            # Not where what she pressed was her own thing, known by its paint: a putt from the lip drops at once, and
            # the screen that follows is the next hole's.
            if not (hers is not None and painted is not None and keep.get("lies_where_struck")):
                keep.setdefault("navigates_from", []).append(start)
                if keep.get("sends_from") is not None and math.dist(tuple(keep["sends_from"]), start) <= 0.05:
                    keep["sends_from"] = None
            ended = "the screen changed"
            break
        keep["kinds"] = moves.kinds
        became = await _what_became_of_it(look, things, painted, start, marked, as_wide)
        ended_at = _where_it_ended(became, ended_at, went_out_at)
        shot = Shot(setting=setting, ended_at=ended_at, gained=gained, aimed_at=aim, began_at=start,
                    lies_in_sight=became[0] == RESTS, gone_from_sight=became[0] == GONE)
        taken += 1
        gains += max(0, gained)
        logger.info("shot %d (%s from %s, %s): ended at %s (%s), gained %d, aimed at %s", taken, way, start, setting,
                    ended_at, became[0], gained, aim)
        after = await look()
        way_on = _a_way_on_shown(after[0] if after else None, read_words)
        if way_on:
            ended = f"the screen offers a way on ({way_on})"
            break
        if became[0] == GONE and painted is not None and not await _hers_shows_again(look, painted, marked, as_wide):
            # What she sent is gone and nothing of hers is left to send: the place has the next word (the next hole,
            # a count, a way on), and whoever handed this over reads it.
            if ended_at is not None:
                sent += 1
                shots_by_way.setdefault(way, Shots(way=way)).took(shot)
            got_there += int(shot.got_there)
            ended = "what she sent got there" if shot.got_there else "what she sent has gone from sight"
            break
        if ended_at is None:
            if hers is None:
                continue
            if keep.get("lies_where_struck") and keep.get("first_sent_from") is not None and not keep.get("tried_first_place"):
                # Pressed where what she sent lies, and nothing went: it may be one of many let go from a place that
                # stays put, come to lie where the place shows them (tops spinning in a ring). The place the first was
                # sent from is tried once. Offline 2026-10-12 tops let go into a ring were taken for the one thing she
                # sends, and she pressed them where they lay for a whole game.
                keep["tried_first_place"], keep["lies_where_struck"] = True, False
                keep["sends_from"] = keep["first_sent_from"]
                continue
            shots_by_way[way].took(shot)
            misses += 1
            if misses >= MISSES_BEFORE_LOOKING_AGAIN:
                # What sent before sends nothing now: where it is, is found again. What a pull does to it is kept.
                logger.info("%d shots from %s sent nothing: finding what sends afresh", misses, start)
                sent_nothing.append(start)
                keep["sends_from"], to_try, misses = None, None, 0
            continue
        misses, sent = 0, sent + 1
        if marker and all(math.dist(marker.where[:2], box[:2]) > ONE_PLACE for box in keep.setdefault("markers_at", [])):
            # A screen she sent something on is a screen of play, and the plate her name is on there stays where it is.
            keep["markers_at"].append(tuple(marker.where))
        if hers is None:
            to_try = None
            tell("Pressing there and letting go sends it; now finding how hard and which way.")
        keep["sends_from"], keep["sends_by"] = start, way
        if keep.get("first_sent_from") is None:
            keep["first_sent_from"] = start
        if painted is not None:
            keep["sends_what"], keep["sends_wide"] = painted, as_wide
        if became[0] == THERE or (keep.get("tried_first_place") and keep.get("lies_where_struck") is False
                                  and math.dist(start, tuple(keep["first_sent_from"])) <= ONE_PLACE):
            # One is where she pressed it (a tee a new ball is set on), or the place the first was sent from has sent
            # again: things are sent from it one after another, wherever the ones already sent have come to lie.
            keep["lies_where_struck"] = False
        elif became[0] == RESTS:
            # Seen lying somewhere new, it is one thing struck where it lies.
            keep["lies_where_struck"] = True
        keep["rests_at"] = ended_at if became[0] == RESTS else None
        shots_by_way.setdefault(way, Shots(way=way)).took(shot)
        if shot.got_there:
            got_there += 1
            tell("That's the measure of it.")
    if keep.get("lies_where_struck") and ended in _HERS_IS_ELSEWHERE_AFTER:
        # The next screen puts her thing somewhere of its own: where it was pressed on this one says nothing of that.
        keep["sends_from"] = keep["rests_at"] = keep["first_sent_from"] = None
    # A thing of hers got where it was to go: that is getting somewhere, whether or not the place counts it in sight.
    return {"shots": taken, "sent": sent, "gains": gains + got_there, "got_there": got_there, "ended": ended,
            "sends_from": keep.get("sends_from"),
            "how_it_goes": {way: shots.how_it_goes() for way, shots in shots_by_way.items()}}
