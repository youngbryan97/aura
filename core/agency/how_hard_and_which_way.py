"""How hard and which way: an act whose strength and direction she sets, learned from where what it sends goes.

Some things are not steered but sent. A golf ball is struck once, a ball is
tossed at a hoop, a top is let go into a ring, a stone is pulled back on a
sling: the act is a pull and a letting go, or a press held and let go, and all
of it is in how far, which way and how long. Nobody knows the right pull for a
putt before trying one. They try, see where the ball stops, and pull a little
more or less, a little left or right, and after a few they have the measure of
it.

So does she. A setting is a few numbers: the pull across and down, as shares
of the picture, and how long the press is held. A shot is a setting tried and
what came of it: where what it sent ended up, and whether anything counted
went up. From the shots so far she fits how a change in the setting moves
where the thing ends up, near the settings tried (a linear model by least
squares, which with two numbers and two shots is the secant method), and
solves it for the place she wants the thing to go. Where there is no place to
aim at, only a count that went up or did not, she searches the settings the
way the cross-entropy method does (Rubinstein, 1999): settings drawn around the
best so far, the spread narrowing round those that paid.

Nothing here knows what is being sent, or what a game is. A slider dragged to
a value, a page flung to a place, a window thrown to a corner are settings
found the same way: an act with a measure, and a result to hold it against.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["HOLD", "PULL", "Setting", "Shot", "Shots", "first_settings"]

#: The two ways of sending: a press pulled some way and let go, and a press held some time and let go.
PULL = "pull"
HOLD = "hold"

#: The furthest a pull goes, as a share of the picture, and the longest a press is held, in seconds.
FURTHEST_PULL = 0.4
LONGEST_HOLD_S = 2.5

#: How many of the latest shots a way is judged by, for whether it sends anything now.
SENT_LATELY = 8

#: The first settings tried, to see which way and how far a pull sends: a little each way, then a little more.
FIRST_PULLS = ((0.15, 0.0), (0.0, 0.15), (-0.15, 0.0), (0.0, -0.15))
FIRST_HOLDS = (0.3, 0.9)

#: How long the first pull is where there is a place to send the thing to: straight back from that place, as a sling,
#: a cue and a putter are drawn, and then straight at it. Shorter than a pull made blind: one that goes the right way
#: at once ought not to go clean past.
FIRST_PULL_WITH_AN_AIM = 0.1

#: How near, as a share of the picture, a thing must end to the place aimed at for the shot to have got there.
NEAR_ENOUGH = 0.04

#: How many shots the cross-entropy search draws before it narrows, how many of the best it narrows round, and the
#: least spread it keeps, as a share of the picture (or of the longest hold).
DRAWN_EACH_TIME = 4
NARROWED_ROUND = 2
LEAST_SPREAD = 0.02

#: How many times in a row a setting that got there is tried with nothing counted before it is doubted.
SAME_TO_NO_END = 3

#: How nearly along the line of its pull (the cosine between them) a shot must have gone to count in full for what a
#: pull does, and how much one counts that went wider.
ALONG_ITS_LINE = 0.9
WIDE_OF_ITS_LINE_COUNTS = 0.2

#: How much the plain rule (a pull carries a thing along its own line, the same every way) counts against the shots
#: themselves, as a share of what they weigh.
THE_PLAIN_RULE_COUNTS = 0.25

#: How much the fit leans toward no change at all, as a share of how much the shots themselves weigh: enough that two
#: shots close together do not make a line that sends the next to the edge of the picture.
LEANING = 1e-4


@dataclass(frozen=True)
class Setting:
    """A pull across and down (shares of the picture) and how long the press is held (seconds)."""

    across: float = 0.0
    down: float = 0.0
    held_s: float = 0.0

    def numbers(self, way: str) -> np.ndarray:
        return np.array([self.across, self.down]) if way == PULL else np.array([self.held_s])

    @staticmethod
    def of(way: str, numbers: Sequence[float]) -> Setting:
        if way == PULL:
            across, down = float(numbers[0]), float(numbers[1])
            length = math.hypot(across, down)
            if length > FURTHEST_PULL:
                across, down = across * FURTHEST_PULL / length, down * FURTHEST_PULL / length
            return Setting(across=across, down=down)
        return Setting(held_s=min(LONGEST_HOLD_S, max(0.05, float(numbers[0]))))


@dataclass
class Shot:
    """A setting tried, and what came of it: where what it sent ended (shares of the picture), and what was counted."""

    setting: Setting
    ended_at: tuple[float, float] | None
    gained: int = 0
    aimed_at: tuple[float, float] | None = None
    #: Where what was sent set off from: a ball struck where it lies sets off from a new place each time.
    began_at: tuple[float, float] | None = None
    #: Whether what was sent was afterwards seen lying where it stopped, or was nowhere to be seen.
    lies_in_sight: bool = False
    gone_from_sight: bool = False

    @property
    def got_there(self) -> bool:
        """Whether the shot did what it was for: something was counted, or what was sent ended where it was aimed and
        is not lying there in plain sight (it went in, or under, or was taken). A ball lying beside a hole has not gone in: offline 2026-10-12 one stopped a
        fortieth of the picture short, and she said she had the measure of it."""
        if self.gained > 0:
            return True
        if self.ended_at is None or self.aimed_at is None or self.lies_in_sight:
            return False
        # A thing gone from sight is placed by where it was last followed, short of where it went in: it got there
        # from twice as far.
        return math.dist(self.ended_at, self.aimed_at) <= (2 if self.gone_from_sight else 1) * NEAR_ENOUGH


def first_settings(way: str, aim: tuple[float, float] | None = None,
                   start: tuple[float, float] | None = None) -> list[tuple[float, ...]]:
    """The first settings of a way, as numbers: where there is a place to send the thing to and a place it is, a pull
    straight back from there and one straight at it come before the pulls made blind."""
    if way != PULL:
        return [(held,) for held in FIRST_HOLDS]
    pulls = list(FIRST_PULLS)
    if aim is not None and start is not None and math.dist(aim, start) > 1e-6:
        far = math.dist(aim, start)
        back = ((start[0] - aim[0]) / far * FIRST_PULL_WITH_AN_AIM, (start[1] - aim[1]) / far * FIRST_PULL_WITH_AN_AIM)
        pulls = [back, (-back[0], -back[1]), *pulls]
    return pulls


@dataclass
class Shots:
    """The shots of one way of sending, and the next setting to try."""

    way: str = PULL
    tried: list[Shot] = field(default_factory=list)
    _mean: np.ndarray | None = None
    _spread: np.ndarray | None = None
    _drawn: list[Setting] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    # -- what she has found out ------------------------------------------------

    def took(self, shot: Shot) -> None:
        self.tried.append(shot)

    @property
    def sends_anything(self) -> bool:
        """Whether any of the latest settings sent something anywhere.

        Lately, not ever: LIVE 2026-10-09 one shot of a putt game sent something early on, and every one of the next
        four hundred, held where she stood without a pull, sent nothing, and she went on holding.
        """
        return any(shot.ended_at is not None for shot in self.tried[-SENT_LATELY:])

    @property
    def the_measure(self) -> Setting | None:
        """The setting that last got there, to be tried again while the place aimed at stays where it is."""
        for shot in reversed(self.tried):
            if shot.got_there:
                return shot.setting
        return None

    def how_it_goes(self) -> str:
        """In words: how far she has got with it."""
        landed = [shot for shot in self.tried if shot.ended_at is not None]
        if not landed:
            return f"{len(self.tried)} tries, nothing sent yet"
        good = sum(shot.got_there for shot in self.tried)
        if good:
            return f"{len(self.tried)} tries, {good} got there"
        if len(landed) >= 2:
            near = min(math.dist(shot.ended_at, shot.aimed_at) for shot in landed if shot.aimed_at is not None) if any(
                shot.aimed_at is not None for shot in landed) else None
            if near is not None:
                return f"{len(self.tried)} tries, the nearest {near:.0%} of the picture away"
        return f"{len(self.tried)} tries"

    # -- the next setting -------------------------------------------------------

    def next_setting(self, aim: tuple[float, float] | None, start: tuple[float, float] | None = None) -> Setting:
        """The setting to try next: toward ``aim`` where there is a place to aim at, else the search for what pays."""
        firsts = [Setting.of(self.way, numbers) for numbers in first_settings(self.way, aim, start)]
        untried = [setting for setting in firsts if setting not in {shot.setting for shot in self.tried}]
        landed = [shot for shot in self.tried if shot.ended_at is not None]
        measure = self.the_measure
        if measure is not None and not self._tried_to_no_end(measure) and (aim is None or self._still_aimed(aim, start)):
            return measure
        if aim is not None and start is not None and self.way == PULL and landed:
            solved = self._by_what_a_pull_does(aim, landed, start)
            if solved is not None:
                return solved
        if aim is not None and len(landed) >= 2:
            solved = self._toward(aim, landed, start)
            if solved is not None:
                return solved
        if untried and len(landed) < 2:
            return untried[0]
        return self._searched()

    def _tried_to_no_end(self, measure: Setting) -> bool:
        """Whether the setting that got there has since been tried again and again with nothing counted for it: it
        ends near enough to look right, and is not right. An invented field scored only a ball that stopped in its
        mark, and thirty shots in a row stopped just short of it, each one "there"."""
        lately = self.tried[-SAME_TO_NO_END:]
        return len(lately) == SAME_TO_NO_END and all(shot.setting == measure and shot.gained <= 0 for shot in lately)

    def _still_aimed(self, aim: tuple[float, float], start: tuple[float, float] | None = None) -> bool:
        """Whether the shot that last got there was the same shot as this one: to the same place, and from it."""
        last = next((shot for shot in reversed(self.tried) if shot.got_there), None)
        if last is None or (last.aimed_at is not None and math.dist(last.aimed_at, aim) > NEAR_ENOUGH):
            return False
        return start is None or last.began_at is None or math.dist(last.began_at, start) <= NEAR_ENOUGH

    def _by_what_a_pull_does(self, aim: tuple[float, float], landed: list[Shot], start: tuple[float, float]) -> Setting | None:
        """The pull that carries the thing from ``start`` to ``aim``, from how far and which way each pull so far
        carried it from where it then lay.

        What anyone takes for granted of a thing pulled back and let go is where this begins: it goes along the line of
        the pull, one way or the other, as far in proportion, and the same whichever way it is pulled. One shot gives
        that one number. More shots, pulled other ways, show where the place departs from it (a slope seen from the
        side, where a pull up the picture carries half as far as one across), and the fit follows them as far as they
        go and keeps to the plain rule where they say nothing. Fitted the plain way, two shots along one line said
        nothing of the line across it, and the least noise in where they ended sent the third the wrong way entirely:
        an invented field, 2026-10-12, a ball pulled away from its mark fifteen times running.

        Shots that went wide of their own line met something on the way, and count for little. Shots that carried the
        thing nearest to as far as it now has to go count for most, for the rule is only straight near where it was
        measured.
        """
        pulls = np.array([shot.setting.numbers(PULL) for shot in landed], dtype=float)
        carried = np.array([np.subtract(shot.ended_at, shot.began_at or start) for shot in landed], dtype=float)
        want = np.subtract(aim, start).astype(float)
        long, far = np.linalg.norm(pulls, axis=1), np.linalg.norm(carried, axis=1)
        told = (long > 1e-6) & (far > 1e-6)
        if not told.any():
            return None
        pulls, carried, long, far = pulls[told], carried[told], long[told], far[told]
        on_its_line = np.abs((pulls * carried).sum(axis=1)) / (long * far)
        weights = np.where(on_its_line >= ALONG_ITS_LINE, 1.0, WIDE_OF_ITS_LINE_COUNTS) / (0.05 + np.linalg.norm(carried - want, axis=1))
        weights = weights / weights.sum()
        for_each = float((weights * (pulls * carried).sum(axis=1)).sum() / (weights * long * long).sum())
        if abs(for_each) < 1e-3:
            return None
        weighed = pulls * weights[:, None]
        gram = weighed.T @ pulls
        lean = THE_PLAIN_RULE_COUNTS * float(np.trace(gram)) / 2.0
        try:
            does = np.linalg.solve(gram + lean * np.eye(2), weighed.T @ carried + lean * for_each * np.eye(2))
            pull = np.linalg.solve(does.T, want)
        except np.linalg.LinAlgError:
            pull = want / for_each
        if not np.all(np.isfinite(pull)):
            pull = want / for_each
        return Setting.of(PULL, pull)

    def _toward(self, aim: tuple[float, float], landed: list[Shot], start: tuple[float, float] | None = None) -> Setting | None:
        """The setting a straight-line fit of where shots ended, against their settings, says ends at ``aim``.

        The fit is weighted to the shots that ended nearest the aim, for the line
        is only straight near where it was measured, and leans a little toward
        the settings already tried (ridge regression) so two shots close together
        do not send the next to the edge of the picture.
        """
        settings = np.array([shot.setting.numbers(self.way) for shot in landed])
        ended = np.array([shot.ended_at for shot in landed], dtype=float)
        target = np.array(aim, dtype=float)
        if start is not None:
            # How far each shot carried the thing from where it set off, and how far this one has to: a ball struck
            # where it lies is somewhere new each time, and what a pull does to it is the same from anywhere. Letting
            # go with no pull, or no hold, carries it nowhere.
            ended = ended - np.array([shot.began_at or start for shot in landed], dtype=float)
            target = target - np.array(start, dtype=float)
            settings = np.vstack([settings, np.zeros((1, settings.shape[1]))])
            ended = np.vstack([ended, np.zeros((1, 2))])
        if len({tuple(row) for row in settings}) < 2:
            return None
        distance = np.linalg.norm(ended - target, axis=1)
        weights = 1.0 / (0.05 + distance)
        weights = weights / weights.sum()
        # ended ~ base + settings @ slope, by weighted least squares.
        design = np.hstack([np.ones((len(ended), 1)), settings])
        root = np.sqrt(weights)[:, None]
        lhs = design * root
        rhs = ended * root
        gram = lhs.T @ lhs
        reg = LEANING * float(np.trace(gram)) / gram.shape[0] * np.eye(design.shape[1])
        reg[0, 0] = 0.0
        try:
            fitted = np.linalg.solve(gram + reg, lhs.T @ rhs)
        # not a failure: a singular system has no fit
        except np.linalg.LinAlgError:
            return None
        base, slope = fitted[0], fitted[1:]
        if not np.all(np.isfinite(slope)) or float(np.abs(slope).sum()) < 1e-6:
            return None
        # settings @ slope = target - base, the least change from the nearest shot's setting that does it.
        nearest = settings[int(np.argmin(distance))]
        want = target - base - nearest @ slope
        change, *_ = np.linalg.lstsq(slope.T, want, rcond=None)
        proposed = nearest + change
        # Not further than twice the spread of the settings tried in one go: the line may not hold out there.
        reach = 2.0 * max(float(np.ptp(settings, axis=0).max()), 0.05 if self.way == PULL else 0.3)
        step = proposed - nearest
        size = float(np.linalg.norm(step))
        if size > reach:
            proposed = nearest + step * reach / size
        return Setting.of(self.way, proposed)

    def _searched(self) -> Setting:
        """The next setting of a cross-entropy search over what paid: drawn round the best, the spread narrowing."""
        dims = 2 if self.way == PULL else 1
        scale = FURTHEST_PULL / 2 if self.way == PULL else LONGEST_HOLD_S / 3
        if self._mean is None:
            best = max(self.tried, key=lambda shot: shot.gained, default=None)
            self._mean = best.setting.numbers(self.way) if best is not None else np.zeros(dims) + (0.0 if self.way == PULL else 0.6)
            self._spread = np.full(dims, scale)
        recent = self.tried[-DRAWN_EACH_TIME:]
        if len(self._drawn) >= DRAWN_EACH_TIME and len(recent) >= DRAWN_EACH_TIME:
            ranked = sorted(recent, key=lambda shot: -shot.gained)[:NARROWED_ROUND]
            chosen = np.array([shot.setting.numbers(self.way) for shot in ranked])
            self._mean = chosen.mean(axis=0)
            floor = LEAST_SPREAD if self.way == PULL else LEAST_SPREAD * LONGEST_HOLD_S
            self._spread = np.maximum(floor, 0.5 * self._spread + 0.5 * chosen.std(axis=0))
            self._drawn = []
        numbers = [self._rng.gauss(float(m), float(s)) for m, s in zip(self._mean, self._spread, strict=True)]
        setting = Setting.of(self.way, numbers)
        self._drawn.append(setting)
        return setting
