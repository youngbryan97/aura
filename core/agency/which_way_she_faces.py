"""Which way she faces, where her keys turn her and take her the way she faces: a car, a tank, a ship that turns.

Most games move what is hers the same way on the screen for a key whatever it looks like: right is right. Some do not.
"Forward, Reverse, Turn left, Turn right": the up key takes her the way she is facing, which is any way at all, and left
and right turn her on the spot and move her nowhere. Learned the usual way, a key and the way it moves her on the
screen, the up key moves her nowhere in particular and the turning keys move her not at all, and she is left with no
way of getting anywhere.

So her keys are also read against where she faces. Where she faces is the way she goes while a key that moves her is
held. A key that moves her along it (ahead or back) is measured by its speed along her facing; a key that moves her
nowhere is measured by how far her facing had turned the next time she moved, over how long it was held. A game is
taken to turn her only on that evidence: one key that has moved her in ways far apart on the screen, and another that
turned her. Then she is steered as anything that turns is steered: turned toward where she is going until she faces
near enough to it, then taken there. Nothing here knows a car or a key.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.agency.direction_from_a_measured_pose_path import DirectionFromPosePath
from core.agency.direction_linked_to_observed_pose import DirectionFromPose
from core.agency.turning_while_travelling import TurningWhileTravelling
from core.verify import invariant

__all__ = ["WhichWaySheFaces"]

#: How fast, in working pixels a second, she must go for her going to say which way she faces.
MOVING = 20.0

#: How near the way most of a key's presses went another press must go to be counted with them; twice as far off, it
#: went another way.
SPREAD = math.radians(30)

#: The longest a run of turning presses between two goings may be held in all to be read: turned further than half
#: round, how far it turned cannot be told from the going after it.
LONGEST_TURN_S = 1.0

#: The least turning, in radians a second, a key that moves her nowhere must do to be a turning key.
TURNS = 0.5

#: How far off where she is going she may face and still be taken there rather than turned.
NEAR_ENOUGH = math.radians(25)

#: How many presses of a key are read before what it does is believed.
PRESSES = 2

#: How many presses of a key must have taken her somewhere for how widely they spread to be read: fewer, and one press
#: read wrong is half of them.
SPREAD_PRESSES = 3

#: How long after a key is pressed her going is still the last key's: the game, and the picture of it, answer late.
SETTLE_S = 0.2

#: The longest interval between observations that can establish a continuous turn.
OBSERVATION_GAP_S = .35


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _spread(angles: list[float]) -> bool:
    """Whether a key's presses took her ways far apart: two presses or more at least twice ``SPREAD`` from the way most
    of them went. One press read wrong does not make a key take her every way."""
    if len(angles) < SPREAD_PRESSES:
        return False
    # The way most presses went: the press with the most others near it.
    most = max(angles, key=lambda a: sum(abs(_wrap(b - a)) <= SPREAD for b in angles))
    return sum(abs(_wrap(a - most)) >= 2 * SPREAD for a in angles) >= 2


@invariant("agency.turns_require_settled_observations", scope="agency",
           owner="core/agency/which_way_she_faces.py", observational=False)
def _turn_observation_invariant() -> tuple:
    faces = WhichWaySheFaces()
    for at in (0, .25, .3, .35, .4):
        faces.saw(80, 0, "a", at)
    faces.saw(0, 0, "b", .5)
    for at in (.55, .8, .85, .9):
        faces.saw(0, 80, "a", at)
    assert not faces.turns, "an unmeasured press was assigned a turn"
    return ()


@invariant('agency.observed_travel_is_not_erased_by_a_later_missing_picture', scope='agency',
           owner='core/agency/which_way_she_faces.py', observational=False)
def _past_travel_invariant() -> tuple:
    faces = WhichWaySheFaces()
    for at in (0., .25, .4):
        faces.saw(80., 0., 'opaque', at, began=0.)
    faces.saw(0., 0., 'opaque', .55, began=0., measured=False)
    assert not faces.going and not faces.turns_measured and not faces.along
    faces.saw(0., 0., '', .7, began=.7, measured=False)
    assert faces.along['opaque'] == [80.] and not faces.turns
    assert faces.ahead_key() is None, 'one segment became independent repeated presses'
    return ()


@dataclass
class WhichWaySheFaces:
    """What her keys do to where she faces, learned from her going; and the key that takes her toward a place."""

    facing: float | None = None
    body_epoch: tuple | None = None
    heading_dependent: bool = False
    #: Each key that moves her: the ways on the screen it took her, press by press, and its speed along her facing.
    ways: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    along: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    #: Between two goings: how long each key that moved her nowhere was held, and how far her facing turned.
    turns: list[tuple[dict[str, float], float]] = field(default_factory=list)
    #: The key held now, since when, where she faced when it was pressed, and how she went under it.
    held: str = ""
    held_since: float = 0.0
    #: Whether the key held takes her ahead (1) or back (-1), as read while it is held.
    sign: float | None = None
    going: list[tuple[float, float]] = field(default_factory=list)
    #: Best observed segment of this delivered press; a later missing picture cannot erase it.
    observed_going: tuple[float, float, float | None, int] | None = None
    #: Pictures taken after the held input had time to answer, including those in which she stood still.
    settled_pictures: int = 0
    #: Turning keys pressed since she last went somewhere, each with how long it was held: what turned her.
    turned_by: list[tuple[str, float]] = field(default_factory=list)
    #: Whether every stationary press since the last sighting was observed long enough to teach a turn.
    turns_measured: bool = True
    #: Every key pressed, the last key a trial asked for, and the time of the latest picture.
    pressed: set[str] = field(default_factory=set)
    #: Earlier owned responses can request new direction experiments; they supply no heading measurements.
    heading_hints: set[str] = field(default_factory=set)
    heading_requests: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_tried: str = ""
    now: float = 0.0
    last_picture_at: float | None = None
    travel_turns: TurningWhileTravelling = field(default_factory=lambda: TurningWhileTravelling(
        moving=MOVING, settle_s=SETTLE_S, gap_s=OBSERVATION_GAP_S, turns=TURNS))
    steering_model: Any = None
    steering_basis: bool | None = None
    body_birth: float | None = None
    pose_direction: DirectionFromPose = field(default_factory=DirectionFromPose)
    pose_path: DirectionFromPosePath = field(default_factory=DirectionFromPosePath)

    def saw(self, vx: float, vy: float, held: str, at: float, *, measured: bool = True,
            began: float | None = None) -> None:
        """Read observed motion under its delivered input; missing motion advances time without teaching a response."""
        if not math.isfinite(at) or (self.last_picture_at is not None and at <= self.last_picture_at):
            return
        if began is not None and (not math.isfinite(began) or began > at):
            return
        gap = self.last_picture_at is not None and at - self.last_picture_at > OBSERVATION_GAP_S
        self.last_picture_at = at
        self.now = at
        if held != self.held or (began is not None and began != self.held_since):
            since = at if began is None else began
            self._let_go(since)
            self.held, self.held_since, self.going, self.sign = held, since, [], None
            self.observed_going = None
            self.settled_pictures = 0
            if held:
                self.pressed.add(held)
        self.travel_turns.saw(held, self.held_since, at, vx, vy, measured=measured)
        if gap or not measured or not all(math.isfinite(v) for v in (vx, vy)):
            self._keep_observed_going()
            self.going = []
            self.settled_pictures = 0
            self.turns_measured = False
            if not measured or not all(math.isfinite(v) for v in (vx, vy)):
                return
        # Just after a press she still goes as the last key had her going: in rooms with walls (offline 2026-10-11) a
        # press into a wall was read as a key that took her the way the key before it had.
        if not held or at - self.held_since < SETTLE_S:
            return
        self.settled_pictures += 1
        if math.hypot(vx, vy) < MOVING:
            return
        self.going.append((vx, vy))
        self._saw_direction(vx, vy, held)

    def _saw_direction(self, vx: float, vy: float, held: str) -> None:
        """Anchor the facing to observed travel and close any observed turning interval."""
        # Whether this key takes her ahead or back: as it has before; else, for the first key she ever goes by, ahead,
        # which sets which way is ahead; else whichever of its two readings (the way she goes, or its opposite) is
        # nearer where the turns since she last went should have brought her facing: by the rates learned so far, and
        # before any are, a press taken to turn her less than a quarter round.
        sign = self._sign(held)
        if sign is None:
            if self.facing is None:
                sign = 1.0
            else:
                expected = self.facing + self._turned_since()
                sign = 1.0 if abs(_wrap(math.atan2(vy, vx) - expected)) <= math.pi / 2 else -1.0
        self.sign = sign
        way = _wrap(math.atan2(vy, vx) + (math.pi if sign < 0 else 0.0))
        if self.facing is None:
            # Turns made before she had ever gone anywhere turned her from a facing nobody saw.
            self.turned_by = []
            self.turns_measured = True
        if self.facing is not None and self.turned_by:
            # Her facing moved by what the keys pressed since she last went did together: kept as one sample, and what
            # each key does is found from all the samples at once.
            held_for: dict[str, float] = defaultdict(float)
            for key, seconds in self.turned_by:
                held_for[key] += seconds
            if sum(held_for.values()) <= LONGEST_TURN_S and self.turns_measured:
                self.turns.append((dict(held_for), _wrap(way - self.facing)))
                del self.turns[:-60]
            self.turned_by = []
        self.facing = way
        self.turns_measured = True

    def _turned_since(self) -> float:
        """How far the turning keys pressed since she last went are known to have turned her."""
        rates = self._stationary_turning_keys()
        return sum(rates.get(key, 0.0) * seconds for key, seconds in self.turned_by)

    def _sign(self, key: str) -> float | None:
        speeds = self.along.get(key)
        return (1.0 if statistics.median(speeds) >= 0 else -1.0) if speeds else None

    def _let_go(self, at: float) -> None:
        key, held_for = self.held, at - self.held_since
        if not key or held_for <= 0:
            return
        self._keep_observed_going()
        if self.observed_going is not None:
            vx, vy, sign, _count = self.observed_going
            self.ways[key].append(math.atan2(vy, vx))
            self.heading_dependent |= self.spread()
            del self.ways[key][:-20]
            if sign is not None:
                self.along[key].append(sign * math.hypot(vx, vy))
                del self.along[key][:-20]
        else:
            self.turned_by.append((key, held_for))
            # Offline 2026-10-12 presses lasting 0.06 s had no settled picture,
            # and were read as stationary turns. Known rates still predict
            # where a short press takes her; it teaches no new rate.
            if self.going or self.settled_pictures < 2:
                self.turns_measured = False

    def _keep_observed_going(self) -> None:
        """Retain one qualified segment per press, without bridging a missing interval."""
        if len(self.going) >= 2 and (self.observed_going is None or len(self.going) >= self.observed_going[3]):
            self.observed_going = (statistics.fmean(v[0] for v in self.going),
                                   statistics.fmean(v[1] for v in self.going), self.sign, len(self.going))

    # -- what it comes to --------------------------------------------------------

    def spread(self) -> bool:
        """Whether a key has taken her ways far apart on the screen: what a place that turns her shows first."""
        return any(_spread(ways) for ways in self.ways.values())

    def turns_her(self) -> bool:
        """Whether this place turns her: a key that has taken her ways far apart, and a key that turned her."""
        return bool(self.turning_keys()) and (self.heading_dependent or self.spread()
                                             or bool(self.travel_turns.rates()) and self.ahead_key() is not None)

    def turning_keys(self) -> dict[str, float]:
        return {**self._stationary_turning_keys(), **self.travel_turns.rates()}

    def _stationary_turning_keys(self) -> dict[str, float]:
        """Each key that turns her, and how fast: radians a second, positive the way angles grow (down the screen from
        right, so clockwise as the screen is seen). Found from every run of presses between two goings together, by
        least squares: how far each run turned her is each key's rate times how long it was held."""
        import numpy as np

        keys = sorted({key for held_for, _turned in self.turns for key in held_for})
        seen = {key: sum(1 for held_for, _ in self.turns if key in held_for) for key in keys}
        # A key that has taken her somewhere on its own, press after press, moves her: pressed into a wall it moved
        # her nowhere, and that says nothing of turning.
        keys = [key for key in keys if seen[key] >= PRESSES and len(self.ways.get(key, ())) < PRESSES]
        if not keys:
            return {}
        rows = np.array([[held_for.get(key, 0.0) for key in keys] for held_for, _ in self.turns])
        turned = np.array([turned for _, turned in self.turns])
        rates, *_ = np.linalg.lstsq(rows, turned, rcond=None)
        return {key: float(rate) for key, rate in zip(keys, rates, strict=True) if abs(rate) >= TURNS}

    def ahead_key(self, least: int = PRESSES) -> str | None:
        """The key that takes her along where she faces fastest, ahead or back, from ``least`` presses of it or more."""
        along = {key: statistics.median(speeds) for key, speeds in self.along.items()
                 if len(speeds) >= least and not self.travel_turns.curved(key) and not self.travel_turns.needs_reprobe(key)}
        best = max(along.items(), key=lambda item: abs(item[1]), default=None)
        return best[0] if best is not None and abs(best[1]) >= MOVING else None

    def facing_now(self) -> float | None:
        """Where she faces now: where she last went, turned on by what the keys pressed since are known to turn her, the
        one held now for as long as it has been. Seen only while she goes, her facing is otherwise worked out, as a
        sailor's is between sightings: steered by where she last went, she turned and turned (offline 2026-10-11)."""
        if self.pose_direction.at == self.now and self.pose_direction.current is not None:
            return self.pose_direction.current
        if self.facing is None:
            return None
        rates = self._stationary_turning_keys()
        held_now = rates.get(self.held, 0.0) * max(0.0, self.now - self.held_since) if not self.going else 0.0
        return _wrap(self.facing + self._turned_since() + held_now)

    def key_toward(self, x: float, y: float, to_x: float, to_y: float) -> str | None:
        """The key to hold to get from (x, y) to (to_x, to_y): a turning key while she faces too far off it, else the key
        that takes her ahead. None where what she faces, or her keys, are not yet known."""
        ahead, turners = self.ahead_key(), self.turning_keys()
        if self.facing is None or ahead is None or not turners:
            return None
        # Where the fastest key takes her back, not ahead, it is driven backwards: she turns her back to the place.
        goes = (self.facing_now() or 0.0) + (math.pi if statistics.median(self.along[ahead]) < 0 else 0.0)
        off = _wrap(math.atan2(to_y - y, to_x - x) - goes)
        if abs(off) <= NEAR_ENOUGH:
            return ahead
        key = max(turners.items(), key=lambda item: item[1] * off)[0]
        return ahead if key in self.travel_turns.rates() and not self.travel_turns.travelling else key

    def trial(self, keys: list[str]) -> str | None:
        """The key to try next to learn what turns her, where trying keys in turn cannot tell: two keys that move her
        nowhere tried one after the other, for as long each, cancel or add up, and which did what is lost (her keys are
        tried in a fixed order, left beside right). So each key that moved her nowhere is tried on its own, between two
        goings ahead, until it has been seen so twice; None where there is nothing left to learn this way."""
        ahead = self.ahead_key(least=1)
        if ahead is None or self.facing is None:
            return self._first_direction_trial(keys)
        # Only where a key has taken her ways far apart, or where two keys or more moved her nowhere while another moved
        # her (turn left, turn right): in a place whose keys move her one way, one key that moves her nowhere is a shot
        # or a jump, and needs no trial. Two that turn her, tried in turn, cancel, and she never turns to show it.
        nowhere = {key for key in self.pressed if key not in self.along and key not in self.ways}
        curved = {key for key in keys if self.travel_turns.curved(key) or self.travel_turns.needs_reprobe(key)}
        if ahead is None or self.facing is None or not (self.spread() or len(nowhere) >= 2 or curved):
            return None
        if curved and self.travel_turns.needs_neutral_travel():
            # Build motion, then observe it released. A stationary rest cannot
            # tell whether the travel would have curved without the input.
            self.last_tried = '' if self.last_tried == ahead else ahead
            return self.last_tried
        alone = defaultdict(int)
        for held_for, _turned in self.turns:
            if len(held_for) == 1:
                alone[next(iter(held_for))] += 1
        observed = {key: self.travel_turns.repeats(key) if key in curved else alone[key] for key in keys}
        still = [key for key in keys if key and (key not in self.along or key in curved)
                 and key in self.pressed and observed[key] < PRESSES]
        if not still:
            if not self.turns_her() and (rates := self.turning_keys()):
                # A new body can have measured turns before enough separate
                # goings establish its driver and heading dependence.
                turn = max(rates, key=lambda key: abs(rates[key]))
                self.last_tried = turn if self.last_tried == ahead else ahead
                return self.last_tried
            return None
        # Ahead, then a still key on its own, then ahead again: her facing read before and after each.
        self.last_tried = min(still, key=lambda key: observed[key]) if self.last_tried == ahead else ahead
        return self.last_tried

    def _first_direction_trial(self, keys: list[str]) -> str | None:
        """A new measured body needs a direction before isolated turning trials can answer."""
        if self.held in self.heading_hints and (self.last_tried == self.held or len(self.going) >= 2):
            self.last_tried = ''
            return ''
        candidates = [key for key in keys if key in self.heading_hints and self.heading_requests[key] < PRESSES]
        if not candidates:
            return None
        self.last_tried = min(candidates, key=lambda key: self.heading_requests[key])
        self.heading_requests[self.last_tried] += 1
        return self.last_tried

    def way_of(self, key: str) -> tuple[float, float] | None:
        """How a key takes her now, where it takes her along her facing; None for a key that turns her or is not known."""
        heading = self.facing_now()
        if heading is None or key in self.turning_keys() or key not in self.along or len(self.along[key]) < PRESSES:
            return None
        speed = statistics.median(self.along[key])
        return speed * math.cos(heading), speed * math.sin(heading)
