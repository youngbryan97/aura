"""Which of the things on screen is hers, found by what answers when she acts.

In a game that runs on its own, everything moves. The ball moves, the other
paddle moves, the clouds move. One thing moves because she pressed a key, and
it is the only thing whose movement depends on which key she is holding.

That is a measurement, and an old one in animal learning: contingency. For
each thing, its speed is collected under each key she was holding at the time.
If the speeds differ more between keys than they vary under any one key, the
thing answers to her. The ball's speed is the same whatever she holds; her
paddle's is up under one key and down under another. The ratio of those two
spreads is the F statistic of a one-way analysis of variance, and the thing
with the largest one, well clear of chance, is hers.

The test holds only while the keys are chosen without looking. Once she plays,
she presses up because the ball is going up, and then the ball's speed depends
on her keys too: her own choices make the ball look like hers. So only the
pictures taken while she is trying keys, in a fixed order whatever is on the
screen, go into the test. Every picture of her own thing still says what each
key does to it, in pixels a second, which is what a plan needs. A key that moves nothing may still make something: a shot
that appears beside her and flies off. That is found the same way, by what
turns up near her just after the key and not otherwise.

Nothing here is told which thing is hers, or which keys do what.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from core.agency.causal_identification import CausalIdentification, CausalWitness, within_observed_reach
from core.verify import invariant

logger = logging.getLogger(__name__)

__all__ = ["Makes", "WhichIsHers"]

#: How long after a key is pressed its effect shows in a picture: one frame of
#: the game and one of the picture being taken.
RESPONSE_S = 0.05

#: The F ratio a thing's speeds must reach to count as answering to her. At
#: the sample sizes here, chance stays under 5 for three keys with fifty
#: samples (the 1% point of F(2, 50) is 5.06); this is four times that.
ANSWERS = 20.0

#: And the difference has to be a movement, not a jitter: the fastest and
#: slowest key must differ by this many working pixels a second.
REALLY_MOVES = 20.0

#: Samples a key needs before it says anything about a thing.
ENOUGH = 4

#: The share of a key's presses that must each bring a thing beside her for
#: the key to be one that makes it.
MADE_EVERY = 0.3

#: Pictures of holding a key over which her thing must mostly not go the
#: key's way before it is taken not to be hers: about a second.
ANSWERING_OVER = 40

@dataclass
class Makes:
    """What pressing a key brings into the picture beside her."""

    key: str
    kind: int
    vx: float
    vy: float
    times: int


#: The speed of a thing is a line through its last 0.15 s of places, so for
#: that long after a key changes (and the game's frame to answer) it still
#: shows the key before.
SETTLE_S = 0.15 + RESPONSE_S


@dataclass
class _Speeds:
    by_key: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    #: The same speeds by the press of a key they were taken in, once settled.
    by_press: dict[tuple[str, float], list[tuple[float, float]]] = field(default_factory=dict)
    #: Speeds taken while the thing was not held against the end of its way.
    free: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    lowest: list[float] = field(default_factory=lambda: [math.inf, math.inf])
    highest: list[float] = field(default_factory=lambda: [-math.inf, -math.inf])

    def add(self, key: str, vx: float, vy: float, *, press: float | None = None, pinned: bool = False,
            settled: bool = True, position: tuple[float, float] | None = None) -> None:
        if position is not None:
            for axis, (place, speed) in enumerate(zip(position, (vx, vy), strict=True)):
                self.lowest[axis] = min(self.lowest[axis], place)
                self.highest[axis] = max(self.highest[axis], place)
                if (self.highest[axis] - self.lowest[axis] > 10.0
                        and min(place - self.lowest[axis], self.highest[axis] - place) < 1.5
                        and abs(speed) < REALLY_MOVES):
                    pinned = True
        for kept in (self.by_key[key], *(() if pinned or not settled else (self.free[key],))):
            kept.append((vx, vy))
            if len(kept) > 200:
                del kept[:100]
        if press is not None:
            self.by_press.setdefault((key, press), []).append((vx, vy))
            if len(self.by_press) > 80:
                del self.by_press[next(iter(self.by_press))]

    def ratio(self) -> tuple[float, float]:
        """F over presses, and the largest difference between two keys' mean speeds.

        One press of a key is one sample. The pictures taken while a key was
        held are not separate trials of it: a ball that bounced between two
        presses moves one way through every picture of the first and the
        other way through every picture of the second, and counted a picture
        at a time it answered to the keys better than her paddle did (offline
        2026-10-04, one run in three). Counted a press at a time, the ball's
        presses of one key disagree with each other and her paddle's do not.
        """
        presses: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (key, _began), values in self.by_press.items():
            if len(values) >= 2:
                presses[key].append(_mean(values))
        # Each key needs repeated trials. One transient object seen during
        # one press can otherwise produce an enormous ratio by chance.
        groups = {key: values for key, values in presses.items() if len(values) >= 2}
        if len(groups) < 2:
            return 0.0, 0.0
        means = {key: _mean(values) for key, values in groups.items()}
        everyone = [value for values in groups.values() for value in values]
        grand = _mean(everyone)
        between = sum(len(groups[key]) * _apart2(means[key], grand) for key in groups)
        within = sum(_apart2(value, means[key]) for key, values in groups.items() for value in values)
        k, n = len(groups), len(everyone)
        if n <= k:
            return 0.0, 0.0
        f = (between / (k - 1)) / max(1e-6, within / (n - k))
        widest = max(math.dist(a, b) for a in means.values() for b in means.values())
        return f, widest

    def typical(self, key: str) -> tuple[float, float] | None:
        """The middle speed under a key: a picture matched to the wrong thing does not move it.

        Taken from the pictures in which the thing was free to move, where
        there are enough: held against the top of its way, the up key moves a
        paddle nowhere, and a plan that holds up into the top for a while
        should not conclude that up does nothing.
        """
        # A blocked or unsettled observation cannot establish what a key does.
        # Keep it in the history, but leave the control unknown for another try.
        values = self.free.get(key) or []
        if len(values) < ENOUGH:
            return None
        return statistics.median(v[0] for v in values), statistics.median(v[1] for v in values)


def _ambiguous_controls_remain_unknown() -> bool:
    speeds = _Speeds()
    for _ in range(20):
        speeds.add("a", 0.0, 0.0, pinned=True)
        speeds.add("b", 0.0, 80.0, settled=False)
    unknown = speeds.typical("a") is None and speeds.typical("b") is None
    for _ in range(ENOUGH):
        speeds.add("a", 80.0, 0.0)
    return unknown and speeds.typical("a") == (80.0, 0.0)


@invariant("agency.ambiguous_controls_remain_unknown", scope="agency",
           owner="core/agency/which_one_answers_to_her.py", observational=False)
def _control_measurement_invariant() -> tuple:
    assert _ambiguous_controls_remain_unknown(), "blocked or unsettled motion established a control"
    return ()


#: How far, in working pixels, the pointer must have gone each way along an
#: axis (and at least a third as far back as there) before what followed it
#: can be told from what was going that way anyway.
THERE_AND_BACK = 10.0


def _measured_along(pairs: list[tuple[float, float, float, float]], extent: tuple[float, float] | None = None) -> tuple[bool | None, bool | None]:
    """Along which axes a thing's place goes with the pointer's; None where it cannot be told.

    Correlation above 0.9 and slope near one, over a stretch in which the
    pointer went both ways. A pointer swept one way while a ball crosses the
    same way correlates with the ball perfectly: offline 2026-10-04 the other
    side's paddle, chasing the ball, was taken for hers, and three games were
    lost steering the mouse. Only something that turns when the pointer turns
    is following it.
    """
    import numpy as np

    data = np.asarray(pairs[-60:], dtype=np.float64)
    along: list[bool | None] = []
    for axis in (0, 1):
        pointer, thing = data[:, axis], data[:, 2 + axis]
        steps = np.diff(pointer)
        there, back = float(steps[steps > 0].sum()), float(-steps[steps < 0].sum())
        if min(there, back) < max(THERE_AND_BACK, max(there, back) / 3):
            along.append(None)
            continue
        if pointer.std() < 5.0 or thing.std() < 5.0:
            along.append(False)
            continue
        r = float(np.corrcoef(pointer, thing)[0, 1])
        slope = float(np.polyfit(pointer, thing, 1)[0])
        # And it is where the pointer is, along that axis. A thing the mouse
        # moves is put under it; a target sliding to and fro on its own can
        # keep pace with a sweep for a while, offline 2026-10-04, but it is
        # wherever it happens to be.
        near = extent is None or float(np.median(np.abs(thing - pointer))) < UNDER_THE_POINTER * extent[axis]
        along.append(r > 0.9 and 0.5 < slope < 2.0 and near)
    return along[0], along[1]


#: Delays at which a page may answer the pointer, in seconds.
POINTER_LAGS_S = (0.0, 0.05, 0.1, 0.2)

#: How far a thing that follows the pointer may stand from it, along the axis
#: it follows, as a share of the picture.
UNDER_THE_POINTER = 0.1


def _extent(moves: Any) -> tuple[float, float] | None:
    """The picture's width and height in working pixels, where the picture says."""
    shape = getattr(moves, "shape", None)
    return (float(shape[1]), float(shape[0])) if shape else None


def _mean(values: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(v[0] for v in values) / len(values), sum(v[1] for v in values) / len(values)


def _apart2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class WhichIsHers:
    """Keeps, for each thing and each kind of thing, how it moved under each key."""

    def __init__(self) -> None:
        self.identification = CausalIdentification(statistic_over=ANSWERS, effect_over=REALLY_MOVES)
        self._held: deque = deque(maxlen=400)
        self._by_thing: dict[int, _Speeds] = defaultdict(_Speeds)
        self._by_kind: dict[int, _Speeds] = defaultdict(_Speeds)
        #: How whichever thing was hers at the time moved under each key. Not
        #: by kind: the other player's paddle looks just like hers.
        self._hers = _Speeds()
        self.number: int | None = None
        self.kind: int | None = None
        self.last_seen: tuple[float, float] | None = None
        self.last_seen_at = -math.inf
        self.last_velocity = (0.0, 0.0)
        self.last_size = 0.0
        self.lowest: list[float] = [math.inf, math.inf]
        self.highest: list[float] = [-math.inf, -math.inf]
        self._taps: list[tuple[str, float, tuple[float, float]]] = []
        self._made: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        self._pressed: dict[str, int] = defaultdict(int)
        self.makes: dict[str, Makes] = {}
        self._pointer: list[tuple[float, float, float]] = []
        self._followed: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        self._looked = 0
        self._sighted: int | None = None
        self.follows_pointer = False
        self._new_screen_at = -math.inf
        #: The axes along which it follows: a paddle under the pointer may follow only across.
        self.follows_along: tuple[bool, bool] = (False, False)
        self._since_believed: list[tuple[float, float, float]] = []
        #: Things found not to be hers after all, and when the last was found out.
        self.not_mine: set[int] = set()
        self.lost_at = -math.inf
        self.last_shape: tuple[float, float] | None = None
        self._answered: dict[str, list[float]] = {}
        self._answered_by: int | None = None
        self._expecting: dict[str, tuple[float, float]] = {}

    # -- what she did ------------------------------------------------------

    def holding(self, key: str, at: float, *, trying: bool = False) -> None:
        """She is holding ``key`` from ``at`` on ("" for nothing); ``trying`` when it was not chosen by looking."""
        if not self._held or self._held[-1][1:] != (key, trying):
            self._held.append((at, key, trying))

    def tapped(self, key: str, at: float) -> None:
        if self.last_seen is not None:
            self._taps.append((key, at, self.last_seen))
            self._pressed[key] += 1

    def pointed(self, x: float, y: float, at: float) -> None:
        """The pointer was taken to (x, y), in working pixels."""
        self._pointer.append((at, x, y))
        del self._pointer[:-400]

    def _pointer_at(self, at: float) -> tuple[float, float] | None:
        import bisect

        index = bisect.bisect_right(self._pointer, (at, math.inf, math.inf))
        return self._pointer[index - 1][1:] if index else None

    def _with_the_pointer(self, seen: list[tuple[float, float, float]], extent: tuple[float, float] | None) -> tuple[bool | None, bool | None]:
        """Along which axes places seen go with where the pointer was, at whichever delay fits best.

        How long a page takes to answer the pointer is the page's own: one
        follows within a picture, another a few later. Matched at the wrong
        delay, a paddle that goes exactly where the mouse goes lags its
        pointer at every turn and falls short of the test (offline
        2026-10-04, bounce). Each delay is a separate try; the best stands.
        """
        best: list[bool | None] = [None, None]
        for lag in POINTER_LAGS_S:
            pairs = [(p[0], p[1], x, y) for at, x, y in seen if (p := self._pointer_at(at - lag)) is not None]
            if len(pairs) < 20:
                continue
            for axis, value in enumerate(_measured_along(pairs, extent)):
                if value or (value is False and best[axis] is None):
                    best[axis] = value
        return best[0], best[1]

    def _what_follows_the_pointer(self, moves: Any, at: float) -> None:
        """A thing that goes where the pointer went is hers, the way a key's thing is."""
        if not self._pointer or self._pointer_at(at) is None:
            return
        if self.follows_pointer:
            self._still_follows(moves, at)
            return
        pointer = self._pointer_at(at)
        # By kind, the nearest of each to the pointer: a thing under a pointer
        # that moved too far in one picture comes back as a new thing. Two of
        # a kind take turns being nearest a pointer swept across between
        # them, and together keep pace with it (offline 2026-10-04, two
        # paddles); neither is under it, which is what _follows asks.
        nearest: dict[int, Any] = {}
        for thing in moves.things.values():
            # Only what moves can follow anything. Of a column of identical
            # still things (the dashes of a net), the one nearest a pointer
            # swept up and down rises and falls with it (LIVE 2026-10-04).
            if thing.seen != at or not thing.moved:
                continue
            best = nearest.get(thing.kind)
            if best is None or math.dist((thing.x, thing.y), pointer) < math.dist((best.x, best.y), pointer):
                nearest[thing.kind] = thing
        for kind, thing in nearest.items():
            self._followed[kind].append((at, thing.x, thing.y))
            del self._followed[kind][:-60]
        self._looked += 1
        if self._looked % 4:
            return
        for kind, seen in self._followed.items():
            if kind not in nearest or len(seen) < 20:
                continue
            along = tuple(bool(v) for v in self._with_the_pointer(seen, _extent(moves)))
            if not any(along):
                continue
            # Once is a sighting; the same again on pictures taken since is a
            # finding. Tested every few pictures at a few delays, chance
            # agreement turns up now and then (offline 2026-10-04, the other
            # paddle during a long Pong game), and seldom twice running.
            if self._sighted != kind:
                self._sighted = kind
                seen.clear()
                continue
            self.number, self.kind, self.follows_pointer = nearest[kind].number, kind, True
            self.follows_along = along
            thing = nearest[kind]
            logger.info(
                "follows the pointer: thing %s of kind %s at (%.0f, %.0f), %dx%d, along %s, over %d pictures",
                thing.number, kind, thing.x, thing.y, thing.w, thing.h, along, len(seen),
            )
            return

    def _still_follows(self, moves: Any, at: float) -> None:
        """Whether her thing still goes where she points, measured as she plays; if not, it was never hers.

        A belief that a thing follows the pointer is tested by every move she
        makes with it. Where she has since taken the pointer both ways along an
        axis and her thing did not go with it, the belief is dropped and which
        thing is hers is found again.
        """
        mine = moves.things.get(self.number) if self.number is not None else None
        if mine is None or mine.seen != at:
            return
        pairs = self._since_believed
        pairs.append((at, mine.x, mine.y))
        del pairs[:-60]
        self._looked += 1
        if len(pairs) < 30 or self._looked % 4:
            return
        measured = self._with_the_pointer(pairs, _extent(moves))
        believed = [m for m, b in zip(measured, self.follows_along, strict=True) if b and m is not None]
        if believed and not any(believed):
            self.follows_pointer, self.follows_along = False, (False, False)
            self._followed.clear()
            pairs.clear()
            self.number = None

    def _under_the_pointer(self, moves: Any, at: float) -> int | None:
        """The thing of her kind nearest the pointer: a thing that follows it is re-made when it jumps."""
        pointer = self._pointer_at(at)
        mine = [t for t in moves.things.values() if t.kind == self.kind]
        if not mine or pointer is None:
            return None
        return min(mine, key=lambda t: math.dist((t.x, t.y), pointer)).number

    def _held_at(self, at: float) -> tuple[float, str, bool] | None:
        for began, key, trying in reversed(self._held):
            if began <= at:
                return began, key, trying
        return None

    def _answering(self, thing: Any, key: str, held_for: float, at: float) -> None:
        """Whether the thing she takes for hers goes the way her key sends it; if it keeps not, it is not hers.

        What a key does is known; while she holds it, her thing should go
        that way, unless it is against the end of its way. A thing that
        does not, for a second of holding, was taken for hers by mistake:
        offline 2026-10-04, after a game began again she took the score's
        digit for her paddle and steered it for a whole game. She says it was
        not hers and finds hers again.
        """
        if self.follows_pointer or not key or held_for < SETTLE_S:
            return
        # What the key was known to do before this thing was taken for hers:
        # its own pictures go into what she knows of her keys, and a thing
        # that never moves would soon teach her that no key moves her.
        if self._answered_by != thing.number:
            self._answered, self._answered_by, self._expecting = {}, thing.number, {}
        way = self._expecting.get(key) or self._hers.typical(key)
        if way is None:
            return
        self._expecting[key] = way
        if math.hypot(*way) <= REALLY_MOVES or self._pinned_toward(thing, way):
            return
        heard = self._answered.setdefault(key, [])
        heard.append((thing.vx * way[0] + thing.vy * way[1]) / (way[0] ** 2 + way[1] ** 2))
        del heard[:-ANSWERING_OVER]
        # Judged key by key: a thing none of her keys moves is not hers, but one
        # key taken wrongly for hers is not every key. LIVE 2026-10-05 "left"
        # was believed to move her paddle; holding it moved nothing, and she
        # disowned her own paddle and played the computer's.
        moving_keys = {k for k in self._hers.free
                       if (v := self._expecting.get(k) or self._hers.typical(k)) is not None
                       and math.hypot(*v) > REALLY_MOVES}
        judged = [statistics.median(self._answered[k]) for k in moving_keys
                  if len(self._answered.get(k, [])) >= 3]
        if (sum(map(len, self._answered.values())) >= ANSWERING_OVER and moving_keys
                and len(judged) == len(moving_keys) and all(m < 0.2 for m in judged)):
            logger.info("control identity %s contradicted by keys %s", thing.number, sorted(moving_keys))
            self.identification.reset("observed control contradiction")
            self.not_mine.add(thing.number)
            self.number, self.kind, self.lost_at = None, None, at
            # Contradiction invalidates the experiment's attribution. An old
            # rival's correlation cannot inherit her controls or identify it.
            self._by_thing.clear()
            self._by_kind.clear()
            self._hers = _Speeds()
            self._answered = {}
            self._answered_by, self._expecting = None, {}
            self.lowest, self.highest = [math.inf, math.inf], [-math.inf, -math.inf]

    def _pinned_toward(self, thing: Any, way: tuple[float, float]) -> bool:
        """Whether a thing stands at the end of its way in the direction ``way`` sends it."""
        for axis, (place, push) in enumerate(((thing.x, way[0]), (thing.y, way[1]))):
            low, high = self.lowest[axis], self.highest[axis]
            if abs(push) <= REALLY_MOVES or high - low <= 10.0:
                continue
            if (push < 0 and place - low < 1.5) or (push > 0 and high - place < 1.5):
                return True
        return False

    def _pinned(self, thing: Any) -> bool:
        """Whether a thing stands at the end of the way it has been seen to go, not moving off it."""
        for axis, (place, speed) in enumerate(((thing.x, thing.vx), (thing.y, thing.vy))):
            low, high = self.lowest[axis], self.highest[axis]
            if high - low > 10.0 and min(place - low, high - place) < 1.5 and abs(speed) < REALLY_MOVES:
                return True
        return False

    # -- what she saw ------------------------------------------------------

    def recheck_controls(self) -> None:
        """Start an independent experiment after inconclusive or contradicted control trials.

        A rejection belongs to the experiment that measured it. New trials
        can establish a response even when an earlier reset or occlusion
        made that same object's response appear absent.
        """
        self.identification.reset("new independent control experiment")
        self.number, self.kind, self._sighted = None, None, None
        for kept in (self._by_thing, self._by_kind, self._followed, self.not_mine):
            kept.clear()
        self._hers = _Speeds()
        self._since_believed, self._answered, self._answered_by = [], {}, None
        self._expecting = {}
        self._pointer = []
        self.follows_pointer, self.follows_along = False, (False, False)
        self.lowest, self.highest = [math.inf, math.inf], [-math.inf, -math.inf]

    def numbered_afresh(self) -> None:
        """The picture's things are numbered from one again: forget what was kept of each by its number.

        Her kind, her shape, where she was and what her keys do stay; which
        number she was does not. Offline 2026-10-04, kept across a game
        beginning again, her paddle's number belonged to the ball in the new
        game, and she steered the ball for a game.
        """
        self.identification.reset("observation identities renumbered")
        self.number, self._sighted = None, None
        for kept in (self._by_thing, self._followed, self.not_mine):
            kept.clear()
        self._since_believed, self._answered, self._answered_by = [], {}, None

    def saw(self, moves: Any, happened: list[dict[str, Any]], at: float) -> None:
        if any(h.get("what") == "new screen" for h in happened):
            self._new_screen_at = at
        if self.number is not None and self.number not in moves.things and not self.follows_pointer:
            # A missing track cannot hand its old trial scores to a rival.
            # Nearby visual continuity may retain the measured controls;
            # otherwise a new independent experiment must establish them.
            nearby = self._one_of_her_kind(moves, at)
            self.identification.reset("established track absent")
            self._by_thing.clear()
            self._by_kind.clear()
            if nearby is None:
                self.recheck_controls()
                self.lost_at = at
            else:
                self.number = nearby
        # A renderer may answer immediately. Delaying the control label can
        # assign the first movement after a change to the previous key.
        # Use the delivered control and exclude its unsettled motion window.
        held = self._held_at(at)
        # A screen being drawn afresh moves everything at once, whatever she held.
        if held is not None and at - self._new_screen_at > 0.5:
            began, key, trying = held
            press = began if at - began >= SETTLE_S else None
            for thing in moves.things.values():
                if thing.seen != at or thing.born == at:
                    continue
                if trying:
                    self._by_thing[thing.number].add(key, thing.vx, thing.vy, press=press,
                                                   settled=press is not None, position=(thing.x, thing.y))
                    self._by_kind[thing.kind].add(key, thing.vx, thing.vy, press=press,
                                                 settled=press is not None)
                if thing.number == self.number:
                    self._hers.add(key, thing.vx, thing.vy, pinned=self._pinned(thing),
                                   settled=press is not None, position=(thing.x, thing.y))
                    self._answering(thing, key, at - began, at)
        self._what_follows_the_pointer(moves, at)
        if not self.follows_pointer:
            self._decide(moves, at)
        elif self.number not in moves.things:
            self.number = self._under_the_pointer(moves, at)
        self._what_keys_make(moves, happened, at)

    def _decide(self, moves: Any, at: float | None = None) -> None:
        witnesses = []
        for number, speeds in self._by_thing.items():
            if number not in moves.things or not moves.things[number].moved or number in self.not_mine:
                continue
            f, widest = speeds.ratio()
            trials: dict[str, int] = defaultdict(int)
            for (key, _press), values in speeds.by_press.items():
                if len(values) >= 2:
                    trials[key] += 1
            witnesses.append(CausalWitness(number, self.identification.epoch, tuple(sorted(trials.items())), f, widest))
        # Keep an established control identity until its own response disproves
        # it. A competing object's old trial score cannot revoke a response
        # she is still measuring on her own object.
        best = self.identification.choose(witnesses, visible=set(moves.things),
                                          established=self.number, excluded=self.not_mine)
        if best is not None:
            if best != self.number:
                for key, values in self._by_thing[best].by_key.items():
                    self._hers.by_key[key].extend(values[-50:])
                for key, values in self._by_thing[best].free.items():
                    self._hers.free[key].extend(values[-50:])
            self.number = best
            self.kind = moves.things[best].kind
        elif self.number not in moves.things and self.kind is not None:
            self.number = self._one_of_her_kind(moves, at)
        mine = moves.things.get(self.number) if self.number is not None else None
        if mine is not None:
            self.last_seen = (mine.x, mine.y)
            if at is not None:
                self.last_seen_at = mine.seen
            self.last_velocity = (mine.vx, mine.vy)
            self.last_size = mine.size
            self.last_shape = (mine.w, mine.h)
            self.kind = mine.kind
            for axis, value in enumerate((mine.x, mine.y)):
                self.lowest[axis] = min(self.lowest[axis], value)
                self.highest[axis] = max(self.highest[axis], value)

    def _one_of_her_kind(self, moves: Any, at: float | None = None) -> int | None:
        """After she was lost, the thing of her kind nearest where she was.

        Or, where nothing of her kind is left, a thing her size close to where
        she was last seen: a kind can be looked at again and changed under her.
        """
        candidates = [t for t in moves.things.values() if t.kind == self.kind and t.number not in self.not_mine
                      and self._her_shape(t) and self.last_seen is not None
                      and self._within_reach(t, at)]
        if not candidates and self.last_seen is not None and self.last_size:
            candidates = [
                t for t in moves.things.values()
                if self._within_reach(t, at) and 0.5 < t.size / self.last_size < 2.0
                and t.number not in self.not_mine and self._her_shape(t)
            ]
        if not candidates:
            return None
        if self.last_seen is None:
            return candidates[-1].number
        x, y = self.last_seen
        return min(candidates, key=lambda t: math.hypot(t.x - x, t.y - y)).number

    def _within_reach(self, thing: Any, at: float | None) -> bool:
        if self.last_seen is None or self.last_shape is None:
            return False
        gap = max(0.0, at - self.last_seen_at) if at is not None else math.inf
        return within_observed_reach(self.last_seen, (thing.x, thing.y), extent=self.last_shape,
                                     velocity=self.last_velocity, gap=gap)

    def _her_shape(self, thing: Any) -> bool:
        """Whether a thing has her shape: a kind is a colour and a size, and a digit can have a paddle's."""
        if self.last_shape is None:
            return True
        w, h = self.last_shape
        return abs(math.log(max(1.0, thing.w) / max(1.0, w))) < math.log(1.6) and abs(math.log(max(1.0, thing.h) / max(1.0, h))) < math.log(1.6)

    def _what_keys_make(self, moves: Any, happened: list[dict[str, Any]], at: float) -> None:
        """A thing that turns up beside her just after a key, and then flies off."""
        fresh = [h for h in happened if h.get("what") == "appeared"]
        self._taps = [tap for tap in self._taps if at - tap[1] < 0.4]
        for key, when, (x, y) in self._taps:
            for event in fresh:
                if event["at"] - when < 0.0 or math.hypot(event["x"] - x, event["y"] - y) > 30.0:
                    continue
                if event["kind"] == self.kind:
                    continue
                self._made[(key, event["kind"])].append((when, event["thing"]))
        for (key, kind), births in self._made.items():
            speeds = [
                (moves.things[number].vx, moves.things[number].vy)
                for _when, number in births[-6:]
                if number in moves.things and moves.things[number].moved
            ]
            # A key that fires makes its shot most times it is pressed. Things
            # also turn up beside her by themselves now and then, and pressed
            # a hundred times while dodging, offline 2026-10-04, the arrow
            # keys each "made" a falling rock twice.
            made_often = len(births) >= MADE_EVERY * self._pressed[key]
            if len(births) >= 2 and speeds and made_often:
                vx, vy = _mean(speeds)
                if math.hypot(vx, vy) > REALLY_MOVES:
                    self.makes[key] = Makes(key, kind, vx, vy, len(births))
            elif key in self.makes and self.makes[key].kind == kind and not made_often:
                del self.makes[key]

    # -- what she knows ----------------------------------------------------

    def thing(self, moves: Any) -> Any:
        return moves.things.get(self.number) if self.number is not None else None

    def way_of(self, key: str) -> tuple[float, float] | None:
        """How fast her thing goes while ``key`` is held, in working pixels a second."""
        if self.kind is None:
            return None
        return self._hers.typical(key)

    def keys_that_move_her(self, keys: list[str]) -> dict[str, tuple[float, float]]:
        moving = {}
        for key in keys:
            way = self.way_of(key)
            if way is not None and math.hypot(*way) > REALLY_MOVES:
                moving[key] = way
        return moving

    def tried(self, key: str) -> int:
        if self.kind is None:
            return max((len(s.by_key.get(key) or []) for s in self._by_thing.values()), default=0)
        return len(self._hers.free.get(key) or [])

    def keys_known(self, keys: list[str]) -> bool:
        """Whether every key has been held long enough on her thing to say what it does."""
        return self.kind is not None and all(self.tried(key) >= ENOUGH for key in keys)

    def axes(self, keys: list[str]) -> tuple[bool, bool]:
        """Whether her keys move her across, and up and down."""
        ways = self.keys_that_move_her(keys).values()
        return (
            any(abs(vx) > REALLY_MOVES for vx, _vy in ways),
            any(abs(vy) > REALLY_MOVES for _vx, vy in ways),
        )


def _fresh_control_experiment_forgets_old_rejections() -> bool:
    hers = WhichIsHers()
    hers.number, hers.kind = 3, 0
    hers.not_mine.add(3)
    for _ in range(ENOUGH):
        hers._hers.add("a", 90.0, 0.0)
        hers._by_thing[3].add("a", 90.0, 0.0)
    hers.recheck_controls()
    return (hers.number is None and hers.kind is None and not hers.not_mine
            and not hers._by_thing and hers._hers.typical("a") is None)


@invariant("agency.fresh_control_experiment_releases_old_evidence", scope="agency",
           owner="core/agency/which_one_answers_to_her.py", observational=False)
def _control_recheck_invariant() -> tuple:
    assert _fresh_control_experiment_forgets_old_rejections(), "a fresh trial retained a prior control rejection"
    return ()
