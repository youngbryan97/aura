"""Choose a place and an input from the motion and consequences measured here.

Target interception, reachable ground and danger along a prospective path share
one observed scene. This component evaluates those choices; the realtime loop
owns delivery, observation and the lifetime of play.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from core.agency.a_choice_and_its_reasons import how_soon_she_reaches, reachable_arrivals
from core.agency.how_far_her_blow_reaches import BLOW_EVERY_S
from core.agency.what_meeting_things_does import AVOID, CLICK, IGNORE, MEET, SHOOT, WhatMeetingDoes
from core.agency.what_she_has_left import CAREFUL_BELOW
from core.agency.whether_it_goes_by_itself import goes_somewhere
from core.agency.which_one_answers_to_her import WhichIsHers, ways_carried
from core.agency.who_is_who import IN_HER_WAY, NEVER_SHOT
from core.perception.how_things_move_here import HowThingsMoveHere
from core.perception.watching_someone_play import stance_watched
from core.perception.what_moves_in_the_picture import WhatMoves

#: How far ahead she looks for anything she might walk into.
DANGER_AHEAD_S = (0.08, 0.16, 0.28, 0.42, 0.6)


def _times_to(lasts: float) -> tuple[float, ...]:
    """The moments a press's path is looked at: the near ones, then a tenth of a second apart to its end."""
    later = []
    step = DANGER_AHEAD_S[-1] + 0.1
    while step <= lasts + 1e-9:
        later.append(round(step, 3))
        step += 0.1
    return (*DANGER_AHEAD_S, *later)


#: How much less a press's path must meet than holding her course does, for the press to be made: half, so a press
#: is made for what it clears and not for a near tie.
CLEARER = 0.5


#: Whether she imagines the moment ahead to choose what to hold where things that matter are moving
#: (core/agency/imagining_ahead.py).
IMAGINES = True


#: How fast a thing that follows the pointer can be taken somewhere, in working
#: pixels a second: the width of a picture in a few frames.
POINTER_SPEED = 2000.0


#: How far ahead, in seconds, she weighs each key for a thing of hers carried on by its own going; and, carried onto a
#: still thing, the share of her going she comes onto it at where how fast is too fast is not yet known.
CARRIED_AHEAD_S = 0.6


GENTLE_SHARE = 0.25


def _ahead(thing: Any, after: float, low: float, high: float, axis: int) -> float:
    """Where a thing will be along ``axis`` after ``after`` seconds, bouncing between low and high."""
    start = (thing.x, thing.y)[axis]
    speed = (thing.vx, thing.vy)[axis]
    span = max(1e-6, high - low)
    travelled = (start - low) + speed * after
    folded = travelled % (2 * span)
    return low + (folded if folded <= span else 2 * span - folded)


def _range_of(moves: WhatMoves, kind: int, axis: int, physics: HowThingsMoveHere | None = None) -> tuple[float, float]:
    """The picture's bounds, replaced by walls whose rebounds have been measured.

    A short path's minimum and maximum are places already visited, and give
    no evidence that the next step will turn there.
    """
    tall, wide = moves.shape
    low, high = 0.0, float((wide, tall)[axis])
    if physics is not None:
        names = ("left", "right") if axis == 0 else ("top", "bottom")
        lower, upper = (physics.edge(kind, name) for name in names)
        if lower[0] == "bounce" and lower[1] is not None:
            low = lower[1]
        if upper[0] == "bounce" and upper[1] is not None:
            high = upper[1]
    return low, high


def goes_with(thing: Any, mine: Any) -> bool:
    """Whether a thing beside hers has gone where hers went, picture for picture: its shadow, what it carries.

    LIVE 2026-10-07 a hero's shadow, a thing of its own to her eyes, was
    always just below it; she made for it, and held the arrow down against the
    bottom wall. Over their last pictures together the two keep one distance
    apart while hers goes somewhere.
    """
    size = max(float(mine.w), float(mine.h), 1.0)
    if math.dist((thing.x, thing.y), (mine.x, mine.y)) > 2.0 * size:
        return False
    theirs = {at: (x, y) for at, x, y in list(thing.path)[-8:]}
    together = [(at, x, y) for at, x, y in list(mine.path)[-8:] if at in theirs]
    if len(together) < 4:
        return False
    apart = [(x - theirs[at][0], y - theirs[at][1]) for at, x, y in together]
    went = math.dist(together[0][1:], together[-1][1:])
    return went > 0.5 * size and max(math.dist(gap, apart[0]) for gap in apart) < 0.25 * went + 2.0


def _boxes_meet(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _moved_box(box: tuple[float, float, float, float], dx: float, dy: float, grow: float = 0.0) -> tuple[float, float, float, float]:
    return (box[0] + dx - grow, box[1] + dy - grow, box[2] + dx + grow, box[3] + dy + grow)


def _contact_line(mine: Any, coming: Any, axis: int) -> float:
    """The incoming thing's centre when the two visible bodies first touch."""
    here, there = (mine.x, mine.y)[axis], (coming.x, coming.y)[axis]
    reach = ((mine.w, mine.h)[axis] + (coming.w, coming.h)[axis]) / 2
    return here + math.copysign(reach, there - here)


class _Choosing:
    """The arithmetic of one decision, from what she has measured so far."""

    #: The tells she has learned and the time of the picture it is choosing on (core/agency/warning_signs.py).
    signs: Any = None
    at: float | None = None
    #: The situations the place's words put her in (core/agency/situations_she_knows.py).
    situations: tuple[Any, ...] = ()
    #: What a recording of someone playing it showed (core/perception/watching_someone_play.py).
    watched: Any = None
    #: How long her presses take to show, and the arrival she goes to meet (when, where, how soon she can be there).
    response_s: float = 0.0
    arrival: Any = None
    #: Whether her thing goes where the pointer goes; the plan she imagined best, the key she holds as it was made, and
    #: the path a press of a key takes her on (core/agency/imagining_ahead.py).
    pointing: bool = False
    planned: Any = None
    held_now: str = ""
    press_paths: Any = None

    #: How much wider a berth what costs her is given than usual: more as what she has left runs low.
    caution: float = 1.0
    #: How far her blow reaches, in her own sizes, where she has one worth standing her ground for; else None.
    strikes: float | None = None
    #: Carried on by her own going: the pace no key gives her, and how far ahead each key is weighed (0: the next picture).
    rest: tuple[float, float] = (0.0, 0.0)
    ahead_s: float = 0.0
    #: Where to go when nothing more pressing is chosen: a thing standing still that a key she was told of is for
    #: (core/agency/using_a_control_on_a_thing.py).
    going_to: tuple[float, float] | None = None
    #: The thing she is going for, kept to (core/agency/a_thing_gone_for.py), and how pressing time is, nought to one
    #: (core/agency/what_the_place_does_to_her.py).
    gone_for: Any = None
    urgency: float = 0.0
    #: Where she faces, where the place turns her (core/agency/which_way_she_faces.py), and what comes for her
    #: (core/agency/what_comes_for_her.py).
    faces: Any = None
    comes: Any = None
    #: The walls met on this screen and the way round them (core/agency/a_way_through.py).
    way_through: Any = None
    #: What she plays for (core/agency/what_she_plays_for.py), who is who (core/agency/who_is_who.py), and the seconds
    #: left on the clock, where there is one.
    plays_for: Any = None
    who: Any = None
    seconds_left: float | None = None
    #: Spots a warning heard lately marks, to keep clear of until its blow is due and gone: (x, y, how near counts, from,
    #: until), the times in seconds from now (core/agency/what_a_sound_comes_with.py).
    warned_spots: tuple = ()
    #: What comes for her allies, and her allies on the screen (core/agency/keeping_allies.py).
    allies: Any = None
    allies_here: tuple = ()
    #: What she pushes and where each goes (core/agency/pushing_a_thing.py).
    pushing: Any = None
    #: The ground its code says is to be gone over, laid on her picture, with what she has marked of it
    #: (core/agency/ground_her_code_names.py): where it is known, the ground is the code's, not cells of her own size.
    ground: Any = None
    #: What the place's code says of how fast a kind of thing wanders, by kind, in working pixels a second.
    swerves_told: Any = None
    #: The kinds that are the marks of ground gone over, and whether its code says nothing inside where she is kept
    #: stops her (core/agency/what_its_code_says_she_sees.py).
    ground_marks: Any = frozenset()
    nothing_stops_her: bool = False

    def __init__(self, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, keys: list[str],
                 physics: HowThingsMoveHere | None = None, *, next_picture_s: float = 1 / 30,
                 response_s: float = 0.0, covered: tuple[set[tuple[int, int]], float] | None = None,
                 barred: tuple[set[tuple[int, int]], float] | None = None, presses: Any = None,
                 reach: Any = None) -> None:
        self.moves, self.hers, self.meeting = moves, hers, meeting
        #: Where the rules ask for a place to be gone over, the cells of it her thing has been over; else None.
        self.covered = covered
        #: The cells she has found out of her reach, and their size.
        self.barred = barred
        self.physics = physics
        self.next_picture_s = next_picture_s
        self.response_s = response_s
        self.mine = hers.thing(moves)
        self.ways = hers.keys_that_move_her(keys)
        # Carried on by her own going, a key is the pace it gives her a moment from now, and no key is where the pull on
        # her takes her: she brakes before she gets there, not when she is there.
        if getattr(hers, "carried", False) and self.mine is not None:
            self.ahead_s = CARRIED_AHEAD_S
            self.ways, self.rest = ways_carried(hers, keys, self.mine, self.ahead_s)
        self.reach = reach
        #: When the thing she goes to meet arrives, where, and how soon she can be there (None where her speed is not known).
        self.arrival: tuple[float, float, float | None] | None = None
        # A key that lifts her and lets her fall is pressed for what its press does, not held as a way to go.
        self.lifts = (presses.lifts(self.mine.kind, max(float(self.mine.w), float(self.mine.h)), self.mine.number)
                      if presses is not None and self.mine is not None else [])
        self.press_paths = ((lambda key: presses.path(key, self.mine.kind, self.mine.number))
                            if presses is not None and self.mine is not None else None)
        for lift in self.lifts:
            self.ways.pop(lift, None)
        self.across, self.updown = hers.follows_along if hers.follows_pointer else hers.axes(keys)
        self.pointing = hers.follows_pointer
        self._what_its_code_says_of_her_keys(keys)
        shots = hers.makes
        self.shot = next(iter(shots.values()), None)
        # Low on what she has left, what costs her is given a wider berth.
        self.caution = 1.0 + 2.0 * max(0.0, CAREFUL_BELOW - getattr(meeting, "vitals", 1.0)) / CAREFUL_BELOW

    def _what_its_code_says_of_her_keys(self, keys: list[str]) -> None:
        """The paces its code says her keys give her, where her trials have not measured them, and the ways they open
        (core/agency/what_its_code_says_she_sees.py). What she measured stands."""
        if self.pointing or getattr(self.hers, "carried", False) or self.mine is None:
            return
        from core.agency.what_its_code_says_she_sees import (
            ground_gone_over,
            nothing_inside_stops_her,
            paces_its_code_says,
        )

        self.ground_marks = ground_gone_over(self.moves)
        if nothing_inside_stops_her(self.moves, self.mine):
            # Its code keeps her in by its edges alone: no cell inside is out of her reach, and no wall is there to go round.
            self.barred, self.nothing_stops_her = None, True

        for key, way in paces_its_code_says(self.moves, self.mine, keys).items():
            if key in getattr(self, "lifts", []):
                # Its code sets her pace by it, and lifts her by nothing: what play took for a lift was her walking.
                self.lifts = [lift for lift in self.lifts if lift != key]
            elif key in self.ways:
                continue
            self.ways[key] = way
            self.across = True if abs(way[0]) > 0 else self.across
            self.updown = True if abs(way[1]) > 0 else self.updown

    def line(self) -> int | None:
        """The axis she cannot move along, when she moves along only one."""
        if self.across is True and self.updown is False:
            return 1
        if self.updown is True and self.across is False:
            return 0
        return None

    def others(self) -> list[Any]:
        shot_kinds = {made.kind for made in self.hers.makes.values()}
        # Only her own thing is left out, and still copies of it (a row of
        # lives), and what goes wherever it goes (its shadow, what it holds).
        # Another thing that looks like hers may be the other player.
        return [
            t for t in self.moves.things.values()
            if self.mine is not None and t.number != self.mine.number
            and not (t.kind == self.hers.kind and not t.moved)
            and t.kind not in shot_kinds and t.number not in self.meeting.writing
            and not goes_with(t, self.mine)
        ]

    def stance(self, thing: Any) -> str:
        if thing.kind in self.ground_marks:
            return IGNORE
        from core.agency.an_effect_contact_cannot_avoid import needs_a_countermeasure
        if needs_a_countermeasure(self.meeting, thing.kind, self.shot):
            return SHOOT
        # What she has found to be in her way is nothing to go and meet (core/agency/who_is_who.py): a wall she was
        # stopped by is not worth a detour to touch again. Not what is only taken for the place, never met (a door is
        # big and still, and is what she is to go to), nor what has not come near her yet (a foe high above is on its
        # way down, and her beam was held back from it: offline 2026-10-12).
        if self.who is not None:
            found = self.who.of(thing)
            if found.sure and found.name == IN_HER_WAY and self.meeting.stance(thing.kind) == MEET:
                return IGNORE
        # A kind that has cost her from a distance is kept clear of, whatever meeting it has done.
        if self.reach is not None and self.reach.reach(int(thing.kind)) is not None:
            return AVOID
        # A thing showing its kind's tell, or one not yet known closing faster than she answers (core/agency/warning_signs.py).
        if self.signs is not None and self.at is not None and self.mine is not None and (self.signs.showing(thing, self.at) or (
                not self.meeting.known(thing.kind)
                and self.signs.closing_in(thing, self.mine, max(2 * self.response_s, 3 * self.next_picture_s)))):
            return AVOID
        # A kind not yet met nor told of is what her situation says such a thing is, by how it moves: what falls is to be
        # caught or kept clear of, what heads for her kept clear of or shot (core/agency/situations_she_knows.py).
        # What a recording of someone playing it showed the player do with such a thing, first
        # (core/perception/watching_someone_play.py).
        if self.watched and self.meeting.evidence.get(thing.kind) is None and thing.kind not in self.meeting.told:
            seen = stance_watched(self.watched, self.moves, thing.kind)
            if seen:
                return seen
        if self.situations and self.meeting.evidence.get(thing.kind) is None and thing.kind not in self.meeting.told:
            from core.agency.situations_she_knows import prior_stance

            supposed = prior_stance(self.situations, thing, self.mine)
            if supposed:
                return supposed
        # What only moves in place (a flag waving, stars twinkling, a figure idling) is judged as what stands still is:
        # touched and found to count for nothing, it is decoration, left alone.
        return self.meeting.stance(thing.kind, fixture=not thing.moved or not goes_somewhere(thing))

    def swerve(self, thing: Any) -> float:
        """How much a thing's going could change before she can answer it, in working pixels a second
        (core/agency/how_far_a_thing_could_swerve.py): the most its kind has been seen to, or the place's code says."""
        if not hasattr(self, "_swerves"):
            from core.agency.how_far_a_thing_could_swerve import swerves_of
            from core.agency.what_its_code_says_she_sees import swerves_its_code_says

            told = self.swerves_told if self.swerves_told is not None else swerves_its_code_says(self.moves)
            self._swerves = swerves_of(getattr(self.moves, "things", {}), told)
        return float(self._swerves.get(int(getattr(thing, "kind", -1)), 0.0))

    def answer_s(self) -> float:
        """How long before what she does about something reaches it: a picture and the delay of her hands."""
        return float(self.next_picture_s or 0.0) + float(self.response_s or 0.0)

    def speed(self, axis: int) -> float:
        if self.pointing:
            # A thing under the pointer is wherever the pointer is put.
            return POINTER_SPEED if (self.across, self.updown)[axis] else 1.0
        return max([abs(way[axis]) for way in self.ways.values()] + [1.0])

    def target(self) -> tuple[tuple[float | None, float | None], str, Any]:
        """Where she wants to be, why, and for which thing."""
        line = self.line()
        target = self._target_on_a_line(line) if line is not None else self._target_anywhere()
        self.last_target = target
        return target

    def _target_on_a_line(self, line: int) -> tuple[tuple[float | None, float | None], str, Any]:
        free = 1 - line
        mine = self.mine
        her_line, her_free = (mine.x, mine.y)[line], (mine.x, mine.y)[free]
        candidates = []
        for thing in self.others():
            if self.stance(thing) not in (MEET, CLICK) or not thing.moved:
                continue
            gap = her_line - (thing.x, thing.y)[line]
            closing = (thing.vx, thing.vy)[line]
            if gap * closing <= 0 or abs(closing) < 5.0:
                continue
            contact = _contact_line(mine, thing, line)
            when = max(0.0, (contact - (thing.x, thing.y)[line]) / closing)
            low, high = _range_of(self.moves, thing.kind, free, self.physics)
            there = _ahead(thing, when, low, high, free)
            # Her own physics of this world, once she has some: gravity, where
            # the walls really are, how much a bounce keeps.
            imagined = self.physics.when_it_reaches(thing, line, contact) if self.physics is not None else None
            if imagined is not None:
                when, there = imagined
            candidates.append((when, there, thing))
        # Of what arrives, what she can get to in time; a catcher lets go of what cannot be reached, and keeps to the
        # one gone for among the rest (core/agency/a_choice_and_its_reasons.py).
        best = self._kept_to(reachable_arrivals(self, candidates, free, her_free) or candidates)
        if best is not None:
            self.arrival = (best[0], best[1], how_soon_she_reaches(self, best, free, her_free))
            goal = [None, None]
            goal[free] = best[1]
            return (goal[0], goal[1]), "meet", best[2]
        # With nothing coming, a thing standing on her own level (a star by her feet, a coin down the hall) is walked
        # to, as a walker does: offline 2026-10-10 the star behind a door she opened was never gone to.
        level = [t for t in self.others() if not t.moved and self.stance(t) in (MEET, CLICK) and not self._out_of_reach(t)
                 and abs((t.x, t.y)[line] - her_line) <= max((mine.w, mine.h)[line], (t.w, t.h)[line])
                 and not self._already_there(t)]
        if level:
            nearest = min(level, key=lambda t: abs((t.x, t.y)[free] - her_free))
            goal = [None, None]
            goal[free] = (nearest.x, nearest.y)[free]
            return (goal[0], goal[1]), "meet", nearest
        aimed = self._aim_a_shot(line)
        if aimed is not None:
            return aimed
        if self.going_to is not None:
            goal = [None, None]
            goal[free] = self.going_to[free]
            return (goal[0], goal[1]), "use", None
        # A walker in a world that waits for her walks on to where she has not been along her own way, not up a wall.
        ground = self._ground_not_yet_covered(line)
        if ground is not None:
            goal = [None, None]
            goal[free] = ground[free]
            return (goal[0], goal[1]), "cover", None
        low, high = self.hers.lowest[free], self.hers.highest[free]
        goal = [None, None]
        goal[free] = (low + high) / 2 if high > low else her_free
        return (goal[0], goal[1]), "wait", None

    def _aim_a_shot(self, line: int | None) -> tuple[tuple[float | None, float | None], str, Any] | None:
        if self.shot is None:
            return None
        along = 1 if abs(self.shot.vy) >= abs(self.shot.vx) else 0
        across = 1 - along
        flight = abs((self.shot.vx, self.shot.vy)[along])
        mine = self.mine
        best = None
        for thing in self.others():
            if self.stance(thing) == IGNORE or self.stance(thing) == MEET and self.meeting.known(thing.kind):
                continue
            # What is on her side, racing her fairly, doing nothing, or part of the place is not shot at, unless shooting
            # its kind is what pays (core/agency/who_is_who.py).
            if (self.who is not None and self.stance(thing) != SHOOT
                    and self.who.of(thing).name in NEVER_SHOT):
                continue
            ahead = ((thing.x, thing.y)[along] - (mine.x, mine.y)[along]) * (self.shot.vx, self.shot.vy)[along]
            if ahead <= 0:
                continue
            when = abs((thing.x, thing.y)[along] - (mine.x, mine.y)[along]) / max(1.0, flight)
            low, high = _range_of(self.moves, thing.kind, across, self.physics)
            there = _ahead(thing, when, low, high, across)
            # What comes for one of her allies is aimed at before what does not (core/agency/keeping_allies.py).
            first = 0 if self.allies is not None and self.allies.threatened([thing], self.allies_here) else 1
            if best is None or (first, when) < (best[3], best[0]):
                best = (when, there, thing, first)
        if best is None:
            return None
        goal: list[float | None] = [None, None]
        goal[across] = best[1]
        return (goal[0], goal[1]), "shoot", best[2]

    def _target_anywhere(self) -> tuple[tuple[float | None, float | None], str, Any]:
        mine = self.mine
        speed = max(self.speed(0), self.speed(1))
        # Where walls are known on a floor, a still thing is as far as the way round them to it
        # (core/agency/a_way_through.py), not as far as a straight line through them.
        through = self.way_through if self.way_through is not None and self.way_through.floor and self.way_through.walls \
            else None
        # A thing she pushes, whose place is on the screen, is got there from behind (core/agency/pushing_a_thing.py).
        pushed = self._a_thing_to_push()
        if pushed is not None:
            return pushed
        candidates = []
        for thing in self.others():
            stance = self.stance(thing)
            # A still thing she is on already is not somewhere to go: what being on it does, it is doing.
            if stance not in (MEET, CLICK) or self._out_of_reach(thing) or not thing.moved and self._already_there(thing):
                continue
            when = math.hypot(thing.x - mine.x, thing.y - mine.y) / speed
            if through is not None and not thing.moved:
                tall, wide = self.moves.shape
                when = through.way_length(float(mine.x), float(mine.y), float(thing.x), float(thing.y), wide, tall) / speed
                if math.isinf(when):
                    continue
            for _ in range(3):
                px, py = thing.where_at(when)
                when = math.hypot(px - mine.x, py - mine.y) / speed
            curious = (not thing.moved or not goes_somewhere(thing)) and not self.meeting.known(thing.kind)
            # A thing not yet known is worth a detour to find out what it is, by what knowing it is worth
            # (core/agency/what_she_plays_for.py); less so as the time runs out.
            rank = when * (self._detour(thing) * (1.0 + self.urgency) if curious else 1.0)
            candidates.append((rank, thing.where_at(when), thing))
        best = self._kept_to(candidates)
        aimed = self._aim_a_shot(None)
        if best is not None and aimed is not None and self._worth_taking_out(aimed, best[0]):
            return aimed
        if best is not None:
            return best[1], "meet", best[2]
        if aimed is not None:
            return aimed
        if self.going_to is not None:
            return self.going_to, "use", None
        ground = self._ground_not_yet_covered()
        if ground is not None:
            return ground, "cover", None
        return (None, None), "wait", None

    def _already_there(self, thing: Any) -> bool:
        """Crossing a place at speed is still an approach, rather than a settled arrival."""
        if not _boxes_meet(self.mine.box(), thing.box()):
            return False
        if not self.ahead_s:
            return True
        limit = self.meeting.gentle_below(thing.kind) if hasattr(self.meeting, "gentle_below") else None
        limit = limit if limit is not None else GENTLE_SHARE * max(self.speed(0), self.speed(1))
        return math.hypot(self.mine.vx - thing.vx, self.mine.vy - thing.vy) <= limit

    def _a_thing_to_push(self) -> tuple[tuple[float | None, float | None], str, Any] | None:
        """Where to go to push a thing she pushes onto its place, where both are on the screen."""
        if self.pushing is None or not self.pushing.places:
            return None
        others = list(self.others())
        for thing in others:
            if not self.pushing.pushable(thing.kind):
                continue
            places = [p for p in others if p.kind in self.pushing.places.get(thing.kind, ()) and p is not thing]
            if not places:
                continue
            place = min(places, key=lambda p: math.hypot(p.x - thing.x, p.y - thing.y))
            goal = self.pushing.where_to_go(self.mine, thing, place)
            if goal is not None:
                return goal, "meet", thing
        return None

    def _detour(self, thing: Any) -> float:
        """How far out of her way a thing not yet known is worth going, as a multiple of the time to it."""
        if self.plays_for is None or self.at is None:
            return 2.0
        alike = sum(1 for other in self.others() if other.kind == thing.kind and other is not thing)
        return self.plays_for.detour_for_the_unknown(self.at, alike_on_screen=alike, seconds_left=self.seconds_left)

    def _worth_taking_out(self, aimed: tuple[tuple[float | None, float | None], str, Any], meet_rank: float) -> bool:
        """Whether taking out the threat she can line up a shot on pays more, over the time it takes, than the thing she
        would go for pays over the time to it (core/agency/what_she_plays_for.py): a threat's worth is the blows it
        would deal while it stays, by what a blow is worth."""
        if self.plays_for is None or self.who is None or self.at is None:
            return False
        (gx, gy), _why, thing = aimed
        from core.agency.who_is_who import THREAT

        if thing is None or self.who.of(thing).name != THREAT:
            return False
        mine = self.mine
        lining_up = max(abs(gx - mine.x) / self.speed(0) if gx is not None else 0.0,
                        abs(gy - mine.y) / self.speed(1) if gy is not None else 0.0)
        comes = self.comes is not None and self.comes.comes_for_her(thing, max(float(mine.w), float(mine.h)))
        worth_s = self.plays_for.worth_taking_out_s(thing.kind, comes_for_her=comes, at=self.at,
                                                    seconds_left=self.seconds_left)
        # Losing an ally it comes for costs her as a blow does.
        if self.allies is not None and self.allies_here:
            worth_s += self.allies.worth_taking_out_s(thing, self.allies_here,
                                                      self.plays_for.worth_of_a_blow_s(self.at, self.seconds_left) or 0.0)
        # Both in gains a second: the threat's worth over the time to take it out, against one gain over the time to it.
        return worth_s * self.plays_for.rate(self.at) / max(lining_up, self.next_picture_s) > 1.0 / max(meet_rank, 1e-3)

    def _kept_to(self, candidates: list[tuple[float, Any, Any]]) -> tuple[float, Any, Any] | None:
        """The candidate gone for: the one she is keeping to, unless there is a reason to let it go
        (core/agency/a_thing_gone_for.py); the soonest where she keeps to none."""
        if not candidates:
            if self.gone_for is not None and self.at is not None:
                self.gone_for.choose([], self.at)
            return None
        if self.gone_for is None or self.at is None:
            return min(candidates, key=lambda candidate: candidate[0])
        return self.gone_for.choose(candidates, self.at, urgency=self.urgency)

    def _out_of_reach(self, thing: Any) -> bool:
        """A thing standing where she has found she cannot go: behind a wall, up on the scenery."""
        if self.barred is None or thing.moved:
            return False
        cells, cell = self.barred
        return (int(thing.x // cell), int(thing.y // cell)) in cells

    def _ground_not_yet_covered(self, line: int | None = None) -> tuple[float, float] | None:
        """Where the rules ask for a place to be gone over, the middle of the nearest part of it her thing has not been over.

        Waiting is right for a paddle and wrong where the ground is the point:
        LIVE 2026-10-07 a painter told to give "the entire campground a fresh
        coat of paint" stood waiting for something to go to, and its paint ran
        out where it stood. Measured in cells of her own thing's size, over
        the picture; ties go to the way she is already going, so she sweeps.
        Where she goes along one ``line`` only, the cells along it at her own.
        """
        mine = self.mine
        if self.ground is not None and mine is not None:
            return self._the_code_s_ground_not_yet_covered(line)
        if self.covered is None or mine is None:
            return None
        covered, cell = self.covered
        tall, wide = self.moves.shape
        columns, rows = max(1, int(wide // cell)), max(1, int(tall // cell))
        heading = (self.hers.last_velocity[0], self.hers.last_velocity[1])
        best = None
        for row in range(rows):
            for column in range(columns):
                if (column, row) in covered or line is not None and (column, row)[line] != int((mine.x, mine.y)[line] // cell):
                    continue
                x, y = (column + 0.5) * cell, (row + 0.5) * cell
                far = math.hypot(x - mine.x, y - mine.y)
                ahead = (x - mine.x) * heading[0] + (y - mine.y) * heading[1]
                rank = far - (0.25 * cell if ahead > 0 else 0.0)
                if best is None or rank < best[0]:
                    best = (rank, x, y)
        return (best[1], best[2]) if best is not None else None

    def _the_code_s_ground_not_yet_covered(self, line: int | None = None) -> tuple[float, float] | None:
        """Where her body is to be for the point of her that marks to be over the nearest cell of the code's ground she
        has not marked: nearest by where that point is now, ties to the way she is going so that she sweeps; along
        her own line where she goes along one only. None where every cell she can reach is marked."""
        ground, mine = self.ground, self.mine
        mx, my = ground.mark_of(float(mine.x), float(mine.y))
        heading = (self.hers.last_velocity[0], self.hers.last_velocity[1])
        size = min(ground.cell)
        best = None
        for column, row in ground.left_to_do():
            x, y = ground.where_to_be_for(column, row)
            if line is not None and abs((x, y)[line] - (mine.x, mine.y)[line]) > size / 2:
                continue
            cx, cy = ground.middle_of(column, row)
            far = math.hypot(cx - mx, cy - my)
            ahead = (cx - mx) * heading[0] + (cy - my) * heading[1]
            rank = far - (0.25 * size if ahead > 0 else 0.0)
            if best is None or rank < best[0]:
                best = (rank, x, y, (column, row), far)
        if best is None:
            return None
        if hasattr(ground, "going_to"):
            ground.going_to(best[3], near=best[4] <= 2.5 * max(ground.cell))
        return best[1], best[2]

    def danger(self, way: tuple[float, float]) -> float:
        return self.danger_along(lambda after: (way[0] * after, way[1] * after))

    def a_press_that_clears(self, presses: Any, held: str) -> str | None:
        """A key whose press takes her clear of what is coming where holding ``held`` on runs into it; else None. Where
        she has imagined the moment ahead and it decided, its plan says whether to press now."""
        if self.planned is not None and self.planned.decides and self.lifts:
            return self.planned.plan.press_now()
        way = self.ways.get(held, (0.0, 0.0))
        staying = self.danger(way)
        if staying <= 0.0 or presses is None:
            return None
        best: tuple[float, str] | None = None
        for key in self.lifts:
            path = presses.path(key, self.mine.kind, self.mine.number)
            if key == held or path is None:
                continue
            # Judged over all of what the press does, not only the next moment: a jump that clears what is near and
            # lands on it is no way clear (offline 2026-10-09: jumps made 0.7 s early landed on the block each time).
            lasts = max(DANGER_AHEAD_S[-1], presses.lasts(key, self.mine.kind, self.mine.number))
            risk = self.danger_along(lambda after, p=path: (way[0] * after + p(after)[0], way[1] * after + p(after)[1]),
                                     times=_times_to(lasts))
            if risk < CLEARER * staying and (best is None or risk < best[0]):
                best = (risk, key)
        return best[1] if best is not None else None

    def _warned_along(self, path: Callable[[float], tuple[float, float]]) -> float:
        """What the spots a warning marks cost along ``path``: where she will be when each one's blow is due, by how
        deep inside its reach that is, so that every step further out is a step better
        (core/agency/what_a_sound_comes_with.py)."""
        if not self.warned_spots or self.mine is None:
            return 0.0
        tall, wide = self.moves.shape
        total = 0.0
        for sx, sy, near, start, end in self.warned_spots:
            due = (start + end) / 2
            dx, dy = path(due)
            x = min(max(self.mine.x + dx, 0.0), float(wide))
            y = min(max(self.mine.y + dy, 0.0), float(tall))
            total += max(0.0, 1.0 - math.hypot(x - sx, y - sy) / max(near, 1e-6)) / (due + 0.1)
        return total

    def danger_along(self, path: Callable[[float], tuple[float, float]], times: Sequence[float] = DANGER_AHEAD_S) -> float:
        """How soon and how often her thing, taken along ``path`` (seconds to displacement), meets what to keep clear of,
        looked at ``times`` seconds ahead."""
        mine = self.mine
        warned = self._warned_along(path)
        threats = [(t, 0.0) for t in self.others()
                   if (self.stance(t) in (AVOID,) or self.stance(t) == SHOOT) and not self._struck_first(t)
                   and int(t.kind) not in getattr(self.meeting, "lands_anyway", ())]
        # What to keep clear of that went out of sight is still where its going was taking it (what_moves_in_the_picture).
        threats += [(t, self.at - when) for t, when in getattr(self.moves, "out_of_sight", {}).values()
                    if self.at is not None and self.meeting.stance(t.kind) == AVOID]
        if not threats:
            return warned
        tall, wide = self.moves.shape
        low_x, high_x = mine.w / 2, wide - mine.w / 2
        low_y, high_y = mine.h / 2, tall - mine.h / 2
        total = 0.0
        for after in times:
            dx, dy = path(after)
            x = min(max(mine.x + dx, low_x), high_x) if high_x >= low_x else mine.x
            y = min(max(mine.y + dy, low_y), high_y) if high_y >= low_y else mine.y
            box = _moved_box(mine.box(), x - mine.x, y - mine.y, 2.0)
            for thing, away in threats:
                # What comes for her is where she will be, not where its line was taking it.
                if self.comes is not None and self.comes.comes_for_her(thing, max(float(mine.w), float(mine.h))):
                    tx, ty = self.comes.where_at(thing, after + away, x, y)
                else:
                    tx, ty = thing.where_at(after + away)
                if (_boxes_meet(box, _moved_box(thing.box(), tx - thing.x, ty - thing.y, 2.0))
                        or (self.reach is not None and self.reach.within(thing, x, y, (tx, ty)))):
                    total += 1.0 / (after + 0.1)
        return total + warned

    def _struck_first(self, thing: Any) -> bool:
        """Whether a thing coming at her comes on slowly enough for her blow to reach it before it reaches her.

        A person with a punch stands their ground against what walks up to them and hits it; one who backs away from
        everything never lands a blow. What stands still, or comes too fast to be struck, is kept clear of as before.
        """
        mine = self.mine
        if not self.strikes or mine is None or not thing.moved:
            return False
        dx, dy = float(mine.x) - float(thing.x), float(mine.y) - float(thing.y)
        apart = math.hypot(dx, dy) or 1e-6
        closing = (float(thing.vx) * dx + float(thing.vy) * dy) / apart
        size = max(float(mine.w), float(mine.h), 1.0)
        return 0.0 < closing <= self.strikes * size / BLOW_EVERY_S

    def key(self, held: str, *, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[str, str, Any]:
        """The key to hold now ("" for none), why, and the thing it is for.

        ``offset`` moves where she wants to be by that much: meeting a thing
        with the end of her paddle instead of its middle.
        """
        (gx, gy), why, aim = self.target()
        if why == "meet":
            gx = gx - offset[0] if gx is not None else None
            gy = gy - offset[1] if gy is not None else None
        mine = self.mine
        # Round the walls she has met to where she is going, on a floor seen from above: the next place on the way.
        through = self.way_through if self.way_through is not None and self.way_through.a_floor(self.ways) else None
        through = None if self.nothing_stops_her else through
        if through is not None and why in ("meet", "cover") and gx is not None and gy is not None and not self.pointing:
            tall, wide = self.moves.shape
            step = through.next_step(float(mine.x), float(mine.y), gx, gy, wide, tall)
            if step is not None:
                gx, gy = step
        # A key that runs her straight into a wall she knows is no choice at all.
        walled = set()
        if through is not None and through.walls:
            size = (float(mine.w), float(mine.h))
            walled = {key for key, way in self.ways.items()
                      if through.ahead_is_a_wall(float(mine.x), float(mine.y), way, size)}
        # Where her keys turn her and take her the way she faces, she is turned toward where she is going and then taken
        # there (core/agency/which_way_she_faces.py): a key's way on the screen says nothing in such a place.
        if self.faces is not None and not self.pointing and self.faces.turns_her() and (gx is not None or gy is not None):
            turned = self.faces.key_toward(float(mine.x), float(mine.y), mine.x if gx is None else gx,
                                           mine.y if gy is None else gy)
            if turned is not None:
                return turned, why, aim
        # Where things that matter are on the move, she imagines the next moment under each plan she could follow and
        # does the first part of the best (core/agency/imagining_ahead.py), a body carried on by its own going by what
        # each key adds to it; where no plan meets anything different, she goes for where she means to be, as below.
        if IMAGINES and not self.pointing:
            from core.agency.imagining_ahead import best_plan

            self.held_now = held
            self.planned = best_plan(self, held, (gx, gy),
                                     arrive_at=aim if self.ahead_s and aim is not None and not aim.moved else None)
            if self.planned is not None and self.planned.decides and self.planned.plan.first not in walled:
                return self.planned.plan.first, why, aim
        choices = {"": self.rest, **{key: way for key, way in self.ways.items() if key not in walled}}
        moving = choices.get(held, (0.0, 0.0))
        # The old command keeps moving her while the next one is delivered.
        # Its measured delay belongs in the prediction before the new command.
        here_x, here_y = mine.x + moving[0] * self.response_s, mine.y + moving[1] * self.response_s
        best_key, best_cost = held if held in choices else "", math.inf
        for key, way in choices.items():
            x, y = here_x + way[0] * (self.ahead_s or self.next_picture_s), here_y + way[1] * (self.ahead_s or self.next_picture_s)
            cost = 0.0
            if gx is not None:
                cost += abs(gx - x) / self.speed(0)
            if gy is not None:
                cost += abs(gy - y) / self.speed(1)
            if self.ahead_s and aim is not None and not aim.moved:
                # Carried onto a still thing, she comes to it slowly: what lands hard on a thing breaks on it; and
                # under the speed meeting it has been found to cost above, where that is known.
                near = max(0.0, 1.0 - math.hypot(aim.x - x, aim.y - y) / (3 * max(float(mine.w), float(mine.h))))
                arriving = math.hypot(2 * way[0] - mine.vx, 2 * way[1] - mine.vy)
                limit = self.meeting.gentle_below(aim.kind) if hasattr(self.meeting, "gentle_below") else None
                # Under the speed meeting it has cost above, or, before that is known, a gentle share of her going: to
                # come on slowly is to arrive, not to hover over it.
                limit = limit if limit is not None else GENTLE_SHARE * max(self.speed(0), self.speed(1))
                cost += near * 3.0 * max(0.0, arriving - limit) / max(1.0, limit)
            cost += 3.0 * self.caution * self.danger(way)
            cost += 0.0 if key == held else self.next_picture_s * 0.1
            if cost < best_cost:
                best_key, best_cost = key, cost
        return best_key, why, aim

    def fire(self, aim: Any, why: str) -> str | None:
        """The key that fires, when what she is lined up on is worth a shot."""
        if self.shot is None or aim is None or why != "shoot":
            return None
        along = 1 if abs(self.shot.vy) >= abs(self.shot.vx) else 0
        across = 1 - along
        flight = abs((self.shot.vx, self.shot.vy)[along])
        when = abs((aim.x, aim.y)[along] - (self.mine.x, self.mine.y)[along]) / max(1.0, flight)
        low, high = _range_of(self.moves, aim.kind, across, self.physics)
        there = _ahead(aim, when, low, high, across)
        if abs(there - (self.mine.x, self.mine.y)[across]) <= (aim.w, aim.h)[across] / 2 + 2:
            return self.shot.key
        return None
