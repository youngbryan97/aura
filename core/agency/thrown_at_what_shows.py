"""Where a click is a shot and the pointer aims it, and nothing in the place is her own body: what the shot goes at,
and when she gets out of the way.

Some places put a player nowhere in the picture. She stands this side of it, and what she does goes into it: a
food fight seen over a table, a gallery of targets, a snowball thrown at whoever shows. A person told "take aim with
your mouse and click to throw" does not look for herself there. She throws at whoever has just shown, since they
will not be up for long; failing that, at the biggest thing going across. And where the words give her a key for
getting out of the way, she presses it when something comes at her, which in such a place means it grows.

Nothing here knows a game. A thing is whatever her eyes follow in the picture (core/perception/what_moves_in_the_picture.py).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Collection, Iterable
from typing import Any

logger = logging.getLogger("Aura.ThrownAtWhatShows")

__all__ = ["A_FIGURE", "HOLD_FOR_A_STOP_S", "JUST_SHOWN_S", "HowLongAThrowTakes", "WhereHarmLands", "how_long_throws_take", "led", "comes_at_her",
           "comes_at_her_where_harm_lands", "what_the_shot_goes_at", "where_the_one_thrown_at_is"]

#: How long after it comes into sight a thing is one that has just shown, in seconds: about as long as one stays up.
JUST_SHOWN_S = 1.5
#: How fast a thing must go to be going somewhere and not standing where it showed, in pixels a second.
GOING = 8.0
#: How much bigger a thing must have got, over the pictures her eyes keep of it, to be coming at her.
GROWN = 1.3
#: The fewest pictures of a thing its growing is judged over.
OVER = 4


#: The least a thing that has just shown is, as a share of the biggest thing that has moved, to be someone showing and
#: not a bit of the picture redrawn (a bar going down, a sparkle).
A_FIGURE = 0.25


def what_the_shot_goes_at(things: Iterable[Any], at: float, *, not_numbers: Collection[int] = (),
                          not_kinds: Collection[int] = (), against_someone: bool = False,
                          shots_at: Collection[float] = ()) -> Any | None:
    """The thing to send the shot at: the biggest of those that have just shown and stand where they showed; where
    none has, the biggest of those going somewhere; and where nothing goes, the biggest that has moved, standing for a
    moment; None where there is none of these. What has just shown is someone only where it is near the size of what
    moves: offline 2026-10-11 a food fight's opponent ran to and fro the whole round, and each throw went at a speck of
    the counter redrawn as her bar went down. ``not_numbers`` and ``not_kinds`` are left out: her own thing, and the
    kinds her own presses bring out."""
    # What grows is coming at her, to be got out of the way of, not thrown at: offline 2026-10-11 her throws went at the
    # food flying at her face.
    # And what came out just after one of her own shots (``shots_at``, when she made them) is her shot in flight:
    # offline 2026-10-11 she threw after her own ball as it rose, and not at the eggs coming down.
    things = list(things)
    taken_afresh = _taken_in_afresh(things)
    # And a strip is no one: offline 2026-10-12 a food fight's two health bars, redrawn at each splat, drew throw after
    # throw to the top of the screen while the opponent stood in the middle of it.
    mine = [thing for thing in things if thing.number not in not_numbers and thing.kind not in not_kinds
            and not _growing(thing) and not _a_strip(thing) and (thing in taken_afresh or not _her_shot(thing, shots_at))]
    # Someone gets about; a thing that only sways where it hangs (a lamp, a flag, a sign) is part of the room: offline
    # 2026-10-11 throws went at a swinging lamp while the opponent ran to and fro below it.
    about = [thing for thing in mine if _gets_about(thing)]
    mine = about or mine
    figures = [thing for thing in mine if getattr(thing, "moved", False) or math.hypot(thing.vx, thing.vy) > GOING]
    least = A_FIGURE * max((float(thing.size) for thing in figures), default=0.0)
    shown = [thing for thing in mine if at - float(getattr(thing, "born", at)) <= JUST_SHOWN_S
             and math.hypot(thing.vx, thing.vy) <= GOING and float(thing.size) >= least]
    going = [thing for thing in mine if math.hypot(thing.vx, thing.vy) > GOING]
    # Where the place is played against someone, the one who has been there all along and gets about is who is thrown
    # at, before whatever has just shown (a splat on her own screen, her own throw in flight): offline 2026-10-11 a food
    # fight's opponent ran to and fro the whole round and was hardly thrown at.
    there = [thing for thing in figures if at - float(getattr(thing, "born", at)) >= THERE_ALL_ALONG_S
             and at - float(getattr(thing, "seen", at)) <= SEEN_JUST_NOW_S] if against_someone else []
    for lot in (there, shown, going, figures):
        if lot:
            return max(lot, key=lambda thing: float(thing.size))
    return None


#: How long the one she last threw at is looked for again where it was, and how far from there, as a share of the
#: picture's width, a figure near its size may be and still be it.
KEPT_IN_MIND_S = 1.0
STILL_IT_WITHIN = 0.2


def where_the_one_thrown_at_is(things: Iterable[Any], last: tuple[float, float, float, float] | None, at: float,
                               wide: float) -> tuple[float, float, Any] | None:
    """Where the one she was last throwing at is now, and the thing it is: the figure nearest where it was and near its
    size, though the eye lost it and found it again under another number (it turned, it crouched, food went past it).
    None where no such figure is in sight, or it has been out of sight too long to be there still: a throw at where it
    was goes where it is not (offline 2026-10-12, measured on the pictures, half again as far off as a throw at
    whoever was in sight)."""
    if last is None or at - last[2] > KEPT_IN_MIND_S:
        return None
    x, y, _seen, size = last
    near = [thing for thing in things if not _growing(thing) and not _a_strip(thing)
            and math.dist((thing.x, thing.y), (x, y)) <= STILL_IT_WITHIN * wide and 0.4 * size <= float(thing.size) <= 2.5 * size]
    if not near:
        return None
    seen = min(near, key=lambda thing: math.dist((thing.x, thing.y), (x, y)))
    return seen.x, seen.y, seen


#: How long a thing has been in sight to have been there all along, and how lately it must have been seen to be there now.
THERE_ALL_ALONG_S = 2.0
SEEN_JUST_NOW_S = 0.3


#: How soon after a shot of hers a thing that comes into sight is that shot.
SHOT_SHOWS_WITHIN_S = 0.8


#: How many things coming into sight at one moment are the picture taken in afresh, not something sent; and how near
#: together in time that moment is.
TAKEN_IN_TOGETHER = 3
ONE_MOMENT_S = 0.05


def _taken_in_afresh(things: list[Any]) -> list[Any]:
    """The things that came into sight together with several others: the picture taken in afresh (a flash across the
    screen, a new look begun), not a shot of hers come into sight. Offline 2026-10-12 everything on a food fight's
    screen was found anew after each splat, all of it just after one of her throws, all of it taken for her throw,
    and nineteen asks in twenty found no one to throw at."""
    born = sorted(float(getattr(thing, "born", -1e9)) for thing in things)
    together = {when for when in born if sum(abs(other - when) <= ONE_MOMENT_S for other in born) >= TAKEN_IN_TOGETHER}
    return [thing for thing in things if any(abs(float(getattr(thing, "born", -1e9)) - when) <= ONE_MOMENT_S for when in together)]


#: How many times longer than it is wide a thing is that is a strip (a bar, a rule, an edge redrawn), not anyone.
A_STRIP = 3.5


def _a_strip(thing: Any) -> bool:
    wide, tall = float(getattr(thing, "w", 0) or 0), float(getattr(thing, "h", 0) or 0)
    return wide > 0 and tall > 0 and max(wide, tall) >= A_STRIP * min(wide, tall)


def _her_shot(thing: Any, shots_at: Collection[float]) -> bool:
    born = float(getattr(thing, "born", -1e9))
    return any(0.0 <= born - shot <= SHOT_SHOWS_WITHIN_S for shot in shots_at)


def _growing(thing: Any) -> bool:
    return comes_at_her([thing])


#: How far, in its own sizes, a thing's path must reach for it to get about and not sway where it is.
GETS_ABOUT = 0.5


def _gets_about(thing: Any) -> bool:
    """Whether a thing's path, over the pictures her eyes keep of it, reaches half its own size or more; a thing not yet
    followed long enough to tell is taken to."""
    path = [point for point in (getattr(thing, "path", ()) or ()) if len(point) >= 3]
    if len(path) < OVER:
        return True
    xs, ys = [point[1] for point in path], [point[2] for point in path]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) >= GETS_ABOUT * math.sqrt(max(1.0, float(thing.size)))


#: How soon a thing must be arriving, by how fast the marks on it spread, to be coming at her.
ARRIVES_WITHIN_S = 1.0


def comes_at_her(things: Iterable[Any], within_s: float = ARRIVES_WITHIN_S) -> bool:
    """Whether something is coming at her out of the picture. Where the marks on it were followed, by when it will
    arrive at the rate they spread apart (core/perception/points_followed.py), as an eye judges a thing looming; else
    by its outline: it has grown by a third over the last pictures of it, and went on growing to the last. Offline
    2026-10-12 the outlines of a food fight's flying food and turning opponent grew and shrank picture by picture, and
    something was taken to be coming at her nearly all the time."""
    for thing in things:
        closing = getattr(thing, "closing_s", None)
        if closing is not None:
            if closing <= within_s:
                return True
            continue
        sizes = [float(size) for size in getattr(thing, "sizes", ())]
        if len(sizes) >= OVER and sizes[-OVER] > 0.0 and sizes[-1] >= GROWN * sizes[-OVER] and sizes[-1] >= sizes[-2]:
            return True
    return False


#: How long before a loss is read a thing that went may be what landed on her, in seconds: a counter is read a picture
#: or two after the hit that changed it.
LANDED_WITHIN_S = 0.8
#: How many landings tell where harm lands, and how many are kept.
LANDINGS_TELL = 2
LANDINGS_REMEMBERED = 12
#: How soon a thing must be coming to where harm lands for the key that gets her out of the way to be pressed, in
#: seconds: a duck or a dodge lasts a moment, and pressed long before it lands is over by then.
DUCK_AHEAD_S = 0.45


class WhereHarmLands:
    """Where what costs her lands, in a place where nothing of hers is seen: the place a thing that went was last seen
    the moment she lost something. A food fight seen over a table puts her this side of it; the food thrown at her is
    gone where it hits her, and that is where she is. Offline 2026-10-13 a food fight's thrower hit her with five throws
    of six, every game, and she ducked by what grew in the picture, which food going down a table does not.

    Only a thing that was going lands: a counter's figure or a life's mark goes when she loses one, and stays where it
    was. A thing whose going had taken it out of the picture by the time it was missed left; it did not land. Her own
    throws went out of the top of that food fight's picture as losses were read, and put where harm lands up there.
    """

    def __init__(self) -> None:
        self.last: dict[int, tuple[float, float, float, float, float, float, float]] = {}
        self.gone: list[tuple[float, float, float, float]] = []
        self.landings: list[tuple[float, float, float]] = []
        self.losses = 0

    def saw(self, things: Iterable[Any], losses: int, at: float, shape: tuple[int, int] | None = None) -> None:
        """One picture: the things in it, how many losses have been read so far, and the picture's height and width."""
        seen = {int(t.number): t for t in things}
        for number, (when, x, y, w, h, vx, vy) in list(self.last.items()):
            if number in seen:
                continue
            del self.last[number]
            ahead_x, ahead_y = x + vx * (at - when), y + vy * (at - when)
            left = shape is not None and not (0.0 <= ahead_x <= shape[1] and 0.0 <= ahead_y <= shape[0])
            if math.hypot(vx, vy) >= GOING and not left:
                self.gone.append((when, x, y, max(w, h)))
        for number, thing in seen.items():
            across = math.sqrt(max(0.0, float(getattr(thing, "size", 0.0) or 0.0)))
            self.last[number] = (at, float(thing.x), float(thing.y), float(getattr(thing, "w", across)),
                                 float(getattr(thing, "h", across)), float(getattr(thing, "vx", 0.0)),
                                 float(getattr(thing, "vy", 0.0)))
        self.gone = [g for g in self.gone if at - g[0] <= LANDED_WITHIN_S]
        self.losses = min(self.losses, losses)           # a new stretch counts its losses from none
        while self.losses < losses:
            self.losses += 1
            if self.gone:
                landed = self._the_one_that_landed()
                self.landings.append(landed[1:])
                self.gone.remove(landed)
                del self.landings[:-LANDINGS_REMEMBERED]

    def _the_one_that_landed(self) -> tuple[float, float, float, float]:
        """Of the things gone a moment before a loss, the one gone where harm has landed before; else the last gone."""
        known = self.where()
        if known is not None:
            near = [g for g in self.gone if math.hypot(g[1] - known[0], g[2] - known[1]) <= known[2]]
            if near:
                return max(near, key=lambda g: g[0])
        return max(self.gone, key=lambda g: g[0])

    def where(self) -> tuple[float, float, float] | None:
        """Where harm has landed, and how far about it: None until it has landed in one place more than once."""
        if len(self.landings) < LANDINGS_TELL:
            return None
        xs, ys = sorted(p[0] for p in self.landings), sorted(p[1] for p in self.landings)
        x, y = xs[len(xs) // 2], ys[len(ys) // 2]
        near = sorted(math.hypot(p[0] - x, p[1] - y) for p in self.landings)[len(self.landings) // 2]
        return x, y, max(near, max(p[2] for p in self.landings))

    def coming(self, things: Iterable[Any], within_s: float = DUCK_AHEAD_S) -> bool:
        """Whether something going is on its way to where harm lands, and will be there within ``within_s``."""
        landing = self.where()
        if landing is None:
            return False
        x, y, about = landing
        for thing in things:
            vx, vy = float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0))
            speed = vx * vx + vy * vy
            if speed < GOING * GOING:
                continue
            dx, dy = x - float(thing.x), y - float(thing.y)
            when = (dx * vx + dy * vy) / speed
            if 0.0 < when <= within_s and math.hypot(dx - vx * when, dy - vy * when) <= about:
                return True
        return False


def comes_at_her_where_harm_lands(run: Any, things: Iterable[Any], at: float, within_s: float = ARRIVES_WITHIN_S,
                                  shape: tuple[int, int] | None = None) -> bool:
    """Whether something is coming at her: growing as it comes out of the picture (``comes_at_her``), or, where nothing
    of hers is seen, on its way to where what costs her has landed (``WhereHarmLands``), kept on ``run``."""
    things = list(things)
    lands = getattr(run, "harm_lands", None)
    if lands is None:
        lands = run.harm_lands = WhereHarmLands()
    lands.saw(things, int(getattr(run, "losses", 0) or 0), at, shape)
    return comes_at_her(things, within_s) or lands.coming(things, min(within_s, DUCK_AHEAD_S))


#: The fewest landings seen before how long a throw takes is known, the most kept, and the longest a throw is waited
#: for; how near where it was aimed a thing must come into sight to be the throw landing, in the picture's own pixels;
#: what a throw is taken to take until one is seen.
LANDINGS_KNOWN = 2
LANDINGS_KEPT = 8
LANDS_WITHIN_S = 1.5
LANDS_NEAR_PX = 14.0
TAKES_AT_FIRST_S = 0.08
#: How long after it came into sight a thing where she aimed is judged, in seconds, and how fast it may be going and
#: still be her throw landing there, in the picture's pixels a second: a splat stays where it was made.
LANDED_SEEN_FOR_S = 0.1
STAYS_PX_S = 20.0
#: How soon after a throw the thing that is the throw itself comes into sight, in seconds, and how nearly straight at
#: where she aimed it must be going, as the cosine of the angle off.
SHOT_SEEN_WITHIN_S = 0.4
SHOT_HEADED = 0.95


class HowLongAThrowTakes:
    """How long a throw of hers takes to land where it was aimed, learned by watching where she aimed: what comes into
    sight there, a moment after the throw, and stays there is the throw landing (a splat, a puff, a hole); or by
    watching the throw itself, come into sight after it and going at where she aimed, as far as it has yet to go.
    Offline 2026-10-12 a food fight's opponent ran to and fro; each throw landed where he had been when she threw, a
    splat a moment later where she clicked, and his bar never went down.

    On 14 October in a food fight that draws no splat, what came into sight where she aimed was the other side's food
    leaving his hand. Her throws were taken to land in 0.25 to 0.67 s where they took 0.83 s, she led and held them by
    that, and threw about 1.2 times a second where she had thrown 2.7."""

    def __init__(self) -> None:
        self.waiting: list[tuple[float, float, float]] = []
        self.landings: list[float] = []

    def thrown(self, at: float, x: float, y: float) -> None:
        self.waiting = [*self.waiting[-5:], (at, x, y)]

    def saw(self, things: Iterable[Any], at: float) -> None:
        """What came into sight, for the throws still waited for."""
        things = list(things)
        kept = []
        for index, (thrown_at, x, y) in enumerate(self.waiting):
            next_throw = self.waiting[index + 1][0] if index + 1 < len(self.waiting) else math.inf
            landed = self._landed(things, thrown_at, x, y, at) + self._on_its_way(things, thrown_at, next_throw, x, y, at)
            if landed:
                self.landings = [*self.landings[-(LANDINGS_KEPT - 1):], min(landed)]
            elif at - thrown_at <= LANDS_WITHIN_S:
                kept.append((thrown_at, x, y))
        self.waiting = kept

    @staticmethod
    def _landed(things: list[Any], thrown_at: float, x: float, y: float, at: float) -> list[float]:
        """How long the throw took, by what came into sight where it was aimed and has stayed there."""
        out = []
        for thing in things:
            born = float(getattr(thing, "born", -1e9))
            if not 0.0 < born - thrown_at <= LANDS_WITHIN_S or at - born < LANDED_SEEN_FOR_S:
                continue
            going = math.hypot(float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0)))
            if going <= STAYS_PX_S and math.hypot(thing.x - x, thing.y - y) <= LANDS_NEAR_PX:
                out.append(born - thrown_at)
        return out

    @staticmethod
    def _on_its_way(things: list[Any], thrown_at: float, next_throw: float, x: float, y: float,
                    at: float) -> list[float]:
        """How long the throw will have taken, by the throw itself on its way to where it was aimed: come into sight
        after it and before the next, going straight at that place, and how far it has yet to go."""
        out = []
        for thing in things:
            born = float(getattr(thing, "born", -1e9))
            if not (thrown_at < born <= min(next_throw, thrown_at + SHOT_SEEN_WITHIN_S)):
                continue
            vx, vy = float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0))
            going, dx, dy = math.hypot(vx, vy), x - thing.x, y - thing.y
            away = math.hypot(dx, dy)
            if going >= GOING and away > LANDS_NEAR_PX and (dx * vx + dy * vy) / (going * away) >= SHOT_HEADED:
                out.append(at - thrown_at + away / going)
        return out

    def seconds(self) -> float:
        """How long a throw takes, as seen: the middle of the landings seen, once there are enough."""
        if len(self.landings) < LANDINGS_KNOWN:
            return TAKES_AT_FIRST_S
        ordered = sorted(self.landings)
        middle = ordered[len(ordered) // 2]
        if getattr(self, "_said", None) != round(middle, 1):
            self._said = round(middle, 1)
            logger.info("her throws take %.2fs to land, from %d seen", middle, len(ordered))
        return middle


def how_long_throws_take(run: Any, moves: Any, at: float) -> Any:
    """What she has seen of how long her throws take to land, brought up to the picture before her."""
    takes = getattr(run, "throw_takes", None)
    if takes is None:
        takes = HowLongAThrowTakes()
        try:
            run.throw_takes = takes
        # not a failure: a run that cannot hold them is given fresh ones each time, as just made
        except AttributeError:
            pass
    takes.saw(moves.things.values(), at)
    return takes


#: The longest a throw is held for someone running to stop, in seconds: briefly, since standing still is not always
#: when someone can be hit (offline 2026-10-12 a throw landed square on an opponent standing with his arms up, and his
#: bar did not move), and throws not thrown are chances not taken.
HOLD_FOR_A_STOP_S = 0.5
#: How far, in its own sizes, a target may go before a throw lands for the throw to go at it as it stands.
STILL_ENOUGH = 0.5


def led(run: Any, target: Any, at: float) -> tuple[float, float] | None:
    """Where to throw at ``target`` so the throw lands where it will be: led by how long her throws take to land (seen,
    else a moment). Where it would go further than its own size before the throw lands, the throw is held a moment for
    it to stop, as someone running to and fro does to throw; held long enough, it is thrown led. None while it is held.
    Offline 2026-10-12 every throw landed where a running opponent had been."""
    takes = getattr(run, "throw_takes", None)
    lead = min(1.0, takes.seconds()) if takes is not None else TAKES_AT_FIRST_S
    speed = math.hypot(target.vx, target.vy)
    if speed * lead <= STILL_ENOUGH * math.sqrt(max(1.0, float(target.size))):
        run.held_since = None
        return target.where_at(lead) if speed > 8.0 else (target.x, target.y)
    if speed * lead > math.sqrt(max(1.0, float(target.size))):
        held = getattr(run, "held_since", None)
        if held is None:
            run.held_since = at
            return None
        if at - held < HOLD_FOR_A_STOP_S:
            return None
    run.held_since = None
    return target.where_at(lead)
