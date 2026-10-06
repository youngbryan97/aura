"""What comes of touching a thing, or letting it by: learned from what the game counts.

A game says how she is doing in numbers it writes on the screen. A score goes
up, a count of lives goes down. Those are the only verdicts it gives, and they
arrive a moment after whatever earned them: the fruit was caught, then the
score changed; the ball got past, then the other side's number went up.

A changed number was earned between the last reading that still showed the old
one and the first that showed the new one, which is half a second. So she
keeps, for every kind of thing, whether its touches and its passings fell
inside such a stretch, and compares that with how much of the whole run those
stretches cover, which is how often anything would fall inside one by chance. A kind whose touch is followed
by losses more often than chance is one to keep away from. A kind whose
passing is followed by losses is one to get in the way of. A kind whose touch
is followed by gains is one to go and get. With nothing known, she meets
things, because that is how anything gets known.

Reading the counters is general knowledge about how games report: a number
labelled as a score is better higher, a number labelled as lives is worse
lower, a clock is neither. Two numbers with no labels on one line are two
sides' scores, and hers is the one on her side.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.verify.invariants import invariant

__all__ = ["AVOID", "CLICK", "IGNORE", "MEET", "SHOOT", "STANCES", "Readouts", "WhatMeetingDoes"]

MEET, AVOID, SHOOT, IGNORE, CLICK = "meet", "avoid", "shoot", "ignore", "click"
STANCES = (MEET, AVOID, SHOOT, IGNORE, CLICK)

#: How far meeting a kind must have gone worse (or better) than letting it by,
#: over three settled meetings and passes or more, to overrule what she was
#: told about it.
OVERRULES = 0.6

#: How long after a touch or a pass its verdict may still arrive: a reading of
#: the counters every half second, a second reading to confirm a change, and
#: the reading's own time.
VERDICT_WITHIN_S = 2.0

#: How long before the last reading that showed the old number the thing that
#: changed it may have happened: the game taking a moment to redraw, and her
#: own picture of it arriving a frame late.
REDRAWN_WITHIN_S = 0.4

#: Label words, by what a change in them means. Plain readings of the words.
_UP_IS_GOOD = (
    "score", "point", "pts", "coin", "gold", "money", "cash", "gem", "hit", "kill",
    "caught", "catch", "bonus", "star", "ring", "collect", "total", "win",
)
_DOWN_IS_BAD = (
    "live", "life", "lives", "health", "hp", "energy", "heart", "ship", "men",
    "ball", "tries", "try", "shield", "fuel", "power",
)
_NEITHER = (
    "time", "timer", "clock", "sec", "level", "lvl", "round", "stage", "wave",
    "high", "best", "hi", "record", "top", "speed", "distance", "lap",
)


@dataclass
class Readout:
    label: str
    value: int
    x: float
    y: float


def _meaning(label: str) -> str:
    words = re.findall(r"[a-z]+", label.lower())
    for word in words:
        if any(word.startswith(stem) for stem in _NEITHER):
            return "neither"
    for word in words:
        if any(word.startswith(stem) for stem in _DOWN_IS_BAD):
            return "down is bad"
        if any(word.startswith(stem) for stem in _UP_IS_GOOD):
            return "up is good"
    return ""


def readouts_in(regions: list[dict[str, Any]]) -> list[Readout]:
    """The numbers written on a screen, each with the words written beside it."""
    found: list[Readout] = []
    words: list[dict[str, Any]] = []
    for region in regions:
        said = str(region.get("text") or "")
        numbers = re.findall(r"\d+", said)
        label = " ".join(re.findall(r"[A-Za-z]+", said))
        if numbers:
            found.append(Readout(label, int(numbers[-1]), float(region.get("center_x", 0)), float(region.get("center_y", 0))))
        elif label:
            words.append(region)
    for readout in found:
        if readout.label:
            continue
        for word in words:
            same_line = abs(float(word.get("center_y", 0)) - readout.y) < max(0.02, float(word.get("height", 0)))
            just_left = 0.0 < readout.x - float(word.get("center_x", 0)) < 0.25
            if same_line and just_left:
                readout.label = str(word.get("text") or "")
    return found


@dataclass
class Readouts:
    """The counters on the screen, followed from one reading to the next."""

    values: dict[str, int] = field(default_factory=dict)
    waiting: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    where: dict[str, tuple[float, float]] = field(default_factory=dict)
    read_at: dict[str, float] = field(default_factory=dict)
    present: set[str] = field(default_factory=set)

    @property
    def current(self) -> dict[str, int]:
        """Confirmed values of counters visible in the latest reading."""
        return {key: self.values[key] for key in self.present if key in self.values}

    @staticmethod
    def _key(readout: Readout) -> str:
        if readout.label:
            return readout.label.lower()
        return f"number at {round(readout.x * 20) / 20:.2f},{round(readout.y * 10) / 10:.1f}"

    def read(self, regions: list[dict[str, Any]], at: float, her_x: float | None) -> list[dict[str, Any]]:
        """Take one reading in. Returns the verdicts it confirms: gains and losses."""
        verdicts = []
        readouts = readouts_in(regions)
        self.present = {self._key(readout) for readout in readouts}
        unlabelled = [r for r in readouts if not r.label]
        for readout in readouts:
            key = self._key(readout)
            self.where[key] = (readout.x, readout.y)
            meaning = _meaning(readout.label) if readout.label else self._unlabelled(readout, unlabelled, her_x)
            change = self._confirmed(key, readout.value, at)
            if change is None or meaning in ("", "neither"):
                continue
            delta, when, since = change
            good = delta > 0 if meaning in ("up is good", "down is bad") else delta < 0
            verdicts.append({"what": "gain" if good else "loss", "at": when, "since": since, "counter": key, "by": delta})
        return verdicts

    @staticmethod
    def _unlabelled(readout: Readout, all_unlabelled: list[Readout], her_x: float | None) -> str:
        row = [r for r in all_unlabelled if abs(r.y - readout.y) < 0.05]
        if len(row) < 2 or her_x is None:
            return "up is good"
        nearest = min(row, key=lambda r: abs(r.x - her_x))
        return "up is good" if nearest is readout else "up is bad"

    def _confirmed(self, key: str, value: int, at: float) -> tuple[int, float, float] | None:
        """A change in a counter once two readings in a row agree on it: by how much, when first seen, and when last not."""
        before = self.values.get(key)
        if before is None:
            self.values[key] = value
            self.read_at[key] = at
            return None
        if value == before:
            self.waiting.pop(key, None)
            self.read_at[key] = at
            return None
        pending = self.waiting.get(key)
        if pending is None or pending[0] != value:
            self.waiting[key] = (value, at, self.read_at.get(key, at))
            return None
        self.waiting.pop(key, None)
        self.values[key] = value
        self.read_at[key] = at
        return value - before, pending[1], pending[2]


def _absent_counters_are_not_current() -> bool:
    counters = Readouts()
    counters.read([{"text": "Score 12", "center_x": 0.2, "center_y": 0.1}], 1.0, 0.2)
    counters.read([{"text": "Lives 3", "center_x": 0.8, "center_y": 0.1}], 2.0, 0.2)
    return counters.current == {"lives": 3} and counters.values == {"score": 12, "lives": 3}


@invariant("agency.current_counters_exclude_absent_history", scope="agency",
           owner="core/agency/what_meeting_things_does.py", observational=False)
def _current_counter_invariant() -> tuple:
    assert _absent_counters_are_not_current(), "an absent counter was presented as current"
    return ()


@dataclass
class _Evidence:
    """What came of a kind meeting her, getting by her, and being shot: sums, and how many were settled."""

    touch_sum: float = 0.0
    shoot: float = 0.0
    touched: int = 0
    passed: int = 0
    shot: int = 0
    pass_sum: float = 0.0
    touches_settled: int = 0
    passes_settled: int = 0
    #: Touches followed at once by a loss: a cost that is the meeting's own.
    immediate: int = 0

    @property
    def meet(self) -> float:
        """How much better meeting it went than letting it by: the mean after each, one less the other.

        The contrast, not a sum. In a rally every point lost comes a moment
        after one of her own returns, so a sum over touches sinks the longer
        she rallies: offline 2026-10-04 it taught her to dodge the ball. Let
        by, the ball costs a point every time; met, only sometimes. Meeting
        it is the better of the two, and that is what a stance is for.
        """
        touch = self.touch_sum / self.touches_settled if self.touches_settled else 0.0
        passing = self.pass_sum / self.passes_settled if self.passes_settled else 0.0
        return touch - passing

    def settled(self) -> int:
        return self.touches_settled + self.passes_settled


class WhatMeetingDoes:
    """Keeps what came of each kind of thing meeting her, passing her, or being shot."""

    def __init__(self) -> None:
        self.readouts = Readouts()
        self.verdicts: list[dict[str, Any]] = []
        self.since: float | None = None
        self._open: list[dict[str, Any]] = []
        self._touching: set[int] = set()
        #: Things that met her lately, and when; and things beside her that
        #: have not, with how they were going when they came beside her.
        self._met: dict[int, float] = {}
        self._beside: dict[int, tuple[float, float]] = {}
        self._side: dict[int, float] = {}
        self.evidence: dict[int, _Evidence] = defaultdict(_Evidence)
        self.writing: set[int] = set()
        self._last_lost = -math.inf
        self.new_screen_at = -math.inf
        self._clicks: dict[int, tuple[float, int]] = {}
        #: What reading the situation said each kind is for, before any evidence.
        self.told: dict[int, str] = {}
        self._places: list[tuple[float, float, float]] = []
        self._settled_places: set[tuple[float, float]] = set()

    # -- verdicts ----------------------------------------------------------

    def read(self, regions: list[dict[str, Any]], at: float, her_x: float | None) -> list[dict[str, Any]]:
        verdicts = self.readouts.read(regions, at, her_x)
        # Counters going back to where they start, as a new screen comes up,
        # are a game beginning again, not something she earned or lost.
        verdicts = [v for v in verdicts if v["at"] - self.new_screen_at > VERDICT_WITHIN_S]
        for verdict in verdicts:
            self._verdict(verdict)
        return verdicts

    def _verdict(self, verdict: dict[str, Any]) -> None:
        verdict.setdefault("since", verdict["at"])
        verdict["from"] = verdict["since"] - REDRAWN_WITHIN_S
        if verdict["what"] == "loss":
            if verdict["at"] - self._last_lost < 1.0:
                return
            self._last_lost = verdict["at"]
        self.verdicts.append(verdict)

    def clicked(self, thing: Any, at: float) -> None:
        """A click aimed at a thing: if the thing goes at once, the click met it."""
        self._clicks[thing.number] = (at, thing.kind)

    def clicked_lately(self, number: int, at: float) -> bool:
        return number in self._clicks and at - self._clicks[number][0] < 0.4

    def _clicks_that_met(self, happened: list[dict[str, Any]], at: float) -> None:
        gone = {h["thing"] for h in happened if h.get("what") == "gone"}
        for number, (when, kind) in list(self._clicks.items()):
            if number in gone and at - when < 0.4:
                self._open.append({"what": "touched", "kind": kind, "at": when})
                self.evidence[kind].touched += 1
                del self._clicks[number]
            elif at - when >= 0.4:
                del self._clicks[number]

    def changed_in_place(self, when: float, x: float, y: float, her_x: float | None) -> None:
        """A counter that changed where text recognition could not read it.

        Read as two sides' scores when changes have been seen on both halves
        of one row: a change on her half is hers, on the other half the other
        side's. A change seen on only one half so far waits, and is settled
        once the row shows its other half. A place a reading already names is
        left to the reading.
        """
        if any(abs(x - rx) < 0.06 and abs(y - ry) < 0.06 for rx, ry in self.readouts.where.values()):
            return
        self._places.append((when, x, y))
        if her_x is None:
            return
        row = [(t, px, py) for t, px, py in self._places if abs(py - y) < 0.06]
        if len({px < 0.5 for _t, px, _py in row}) < 2:
            return
        for t, px, _py in row:
            if (t, px) in self._settled_places:
                continue
            self._settled_places.add((t, px))
            ours = (px < 0.5) == (her_x < 0.5)
            self._verdict({"what": "gain" if ours else "loss", "at": t, "since": t - 0.5, "counter": f"the score at {px:.2f}"})

    def she_was_lost(self, at: float) -> None:
        """Her own thing went: a loss whatever the counters say, earned just before."""
        self._verdict({"what": "loss", "at": at, "since": at - 0.1, "counter": "her own thing"})

    def _by_chance(self, what: str, now: float) -> float:
        """The share of the run covered by stretches in which a verdict of this kind was earned."""
        if self.since is None:
            return 0.0
        span = max(VERDICT_WITHIN_S, now - self.since)
        covered = sum(v["at"] - v["from"] for v in self.verdicts if v["what"] == what)
        return min(1.0, covered / span)

    def _earned_while(self, at: float) -> tuple[int, int]:
        inside = [v for v in self.verdicts if v["from"] <= at <= v["at"]]
        return sum(1 for v in inside if v["what"] == "gain"), sum(1 for v in inside if v["what"] == "loss")

    def _settle(self, at: float) -> None:
        """Close every touch and pass whose time for a verdict is over."""
        still_open = []
        for event in self._open:
            if at - event["at"] < VERDICT_WITHIN_S:
                still_open.append(event)
                continue
            gains, losses = self._earned_while(event["at"])
            better = (gains - self._by_chance("gain", at)) - (losses - self._by_chance("loss", at))
            kept = self.evidence[event["kind"]]
            if event["what"] == "touched":
                kept.touch_sum += better
                kept.touches_settled += 1
                if any(v["what"] == "loss" and 0.0 <= v["at"] - event["at"] <= AT_ONCE_S for v in self.verdicts):
                    kept.immediate += 1
            elif event["what"] == "passed":
                kept.pass_sum += better
                kept.passes_settled += 1
            elif event["what"] == "shot":
                kept.shoot += better
        self._open = still_open

    # -- what happened between her and the rest ---------------------------

    def numbered_afresh(self) -> None:
        """The picture's things are numbered from one again: forget what was kept of each by its number."""
        for kept in (self._touching, self._met, self._beside, self._side, self.writing, self._clicks):
            kept.clear()

    def begin_run(self) -> None:
        """Keep learned effects; start counters and pending events afresh."""
        self.numbered_afresh()
        self.readouts = Readouts()
        self.verdicts.clear()
        self._open.clear()
        self.since = None
        self._last_lost = -math.inf
        self.new_screen_at = -math.inf
        self._places.clear()
        self._settled_places.clear()

    def saw(self, moves: Any, hers: Any, happened: list[dict[str, Any]], at: float, line: int | None) -> None:
        """One picture's worth: touches, passes, shots, and her own loss."""
        if self.since is None:
            self.since = at
        if any(h.get("what") == "new screen" for h in happened):
            self.new_screen_at = at
        mine = hers.thing(moves)
        self._clicks_that_met(happened, at)
        self._touches(moves, mine, hers, at)
        self._gone(moves, mine, hers, happened, at)
        if mine is not None and line is not None:
            self._passes(moves, mine, hers, at, line)
        self._settle(at)

    def _note(self, what: str, thing: Any, at: float) -> None:
        self._open.append({"what": what, "kind": thing.kind, "at": at})
        kept = self.evidence[thing.kind]
        if what == "touched":
            kept.touched += 1
        elif what == "passed":
            kept.passed += 1
        else:
            kept.shot += 1

    def _touches(self, moves: Any, mine: Any, hers: Any, at: float) -> None:
        """Things that met her: that overlapped her, or that turned while beside her.

        Beside her is not met. A ball that slips past the end of a paddle
        comes within a pixel of it and goes on its way; counted as met,
        offline 2026-10-04, every point lost that way was laid on meeting the
        ball. Two pictures can miss the moment of overlap, so a thing that
        came close and left on another course was met too.
        """
        if mine is None:
            return
        now = set()
        for thing in moves.things.values():
            if thing.number == mine.number or thing.kind in self._shot_kinds(hers):
                continue
            if thing.kind == hers.kind and (not thing.moved or _inside(thing, mine)):
                # Her own kind standing where she stands is her, drawn again:
                # a ship that blinks after a hit comes back as a new thing on
                # top of the old one, and meeting herself taught her, offline
                # 2026-10-04, that her own colour costs a life.
                continue
            if not _close(mine.box(), thing.box(), 2.0):
                continue
            now.add(thing.number)
            if thing.number in self._met or thing.number in self.writing:
                continue
            if _close(mine.box(), thing.box(), 0.0) or _turned(self._beside.get(thing.number), thing):
                self._met[thing.number] = at
                self._note("touched", thing, at)
            else:
                self._beside.setdefault(thing.number, (thing.vx, thing.vy))
        for number in [n for n in self._beside if n not in now]:
            thing = moves.things.get(number)
            if thing is not None and number not in self._met and _turned(self._beside[number], thing):
                self._met[number] = at
                self._note("touched", thing, at)
            del self._beside[number]
        self._touching = now
        self._met = {n: when for n, when in self._met.items() if n in now or at - when < 1.0}

    @staticmethod
    def _shot_kinds(hers: Any) -> tuple[int, ...]:
        return tuple(made.kind for made in hers.makes.values())

    def _gone(self, moves: Any, mine: Any, hers: Any, happened: list[dict[str, Any]], at: float) -> None:
        gone = [h for h in happened if h.get("what") == "gone"]
        shots = [h for h in gone if h["kind"] in self._shot_kinds(hers)]
        for event in gone:
            if event["thing"] == hers.number:
                if not _at_an_edge(event, moves.shape):
                    self.she_was_lost(at)
                continue
            if event["kind"] in self._shot_kinds(hers):
                continue
            # A thing that goes the moment it reaches her was met, though no
            # picture showed the two together: the game took it away first.
            last = moves.last_box.get(event["thing"])
            if mine is not None and last is not None and event["thing"] not in self._met and _close(mine.box(), last, 4.0):
                # Unless it went off the edge of the picture behind her: then
                # it got by. Offline 2026-10-04, a ball a paddle missed at the
                # court's edge was filed as met, every point lost after it
                # was laid on meeting the ball, and she learned to dodge it.
                what = "passed" if _at_an_edge(event, moves.shape) and _behind(mine, event, moves.shape) else "touched"
                self._open.append({"what": what, "kind": event["kind"], "at": at})
                if what == "passed":
                    self.evidence[event["kind"]].passed += 1
                else:
                    self.evidence[event["kind"]].touched += 1
                continue
            hit = any(math.hypot(s["x"] - event["x"], s["y"] - event["y"]) < 14.0 for s in shots)
            if hit:
                self._open.append({"what": "shot", "kind": event["kind"], "at": at})
                self.evidence[event["kind"]].shot += 1

    def _passes(self, moves: Any, mine: Any, hers: Any, at: float, line: int) -> None:
        """A thing crossing the line she stands on without touching her: it got by."""
        mine_at = (mine.x, mine.y)[line]
        seen = set()
        for thing in moves.things.values():
            if thing.number == mine.number or not thing.moved:
                continue
            seen.add(thing.number)
            side = (thing.x, thing.y)[line] - mine_at
            before = self._side.get(thing.number)
            self._side[thing.number] = side
            crossed = before is not None and before * side < 0
            if crossed and thing.number not in self._met:
                self._note("passed", thing, at)
        self._side = {n: s for n, s in self._side.items() if n in seen}

    # -- what she makes of it ---------------------------------------------

    def stance(self, kind: int, *, fixture: bool = False) -> str:
        """What to do about a kind: what was measured, else what she was told, else meet it.

        What the situation reading said holds until the evidence against it is
        clear, so a rule read wrongly is corrected by play, and play does not
        have to pay a life to learn what the rules already said.
        """
        kept = self.evidence.get(kind)
        told = self.told.get(kind)
        if told is not None:
            measured = kept.meet if kept is not None else 0.0
            enough = kept is not None and kept.settled() >= 3
            against = enough and ((told in (MEET, CLICK) and measured <= -OVERRULES) or (told == AVOID and measured >= OVERRULES))
            if not against and not (kept is not None and kept.shoot >= 0.5 and told != AVOID):
                return SHOOT if told == SHOOT else told
        if kept is None:
            return MEET
        if kept.shoot >= 0.5:
            return SHOOT
        if kept.meet <= -0.6 and _a_cost_shown(kept):
            return AVOID
        if fixture and kept.touched >= 2 and kept.meet < 0.5:
            return IGNORE
        return MEET

    def known(self, kind: int) -> bool:
        """Whether what she makes of a kind rests on enough to say it out loud.

        A cost is believed at once, because a second one is expensive. A
        benefit is believed when several touches agree, because a gain that
        came while she happened to be touching something proves little.
        """
        kept = self.evidence.get(kind)
        if kept is None:
            return False
        return (kept.meet <= -0.6 and _a_cost_shown(kept)) or kept.shoot >= 0.5 or (kept.meet >= 0.5 and kept.touches_settled >= 3)


def _close(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float) -> bool:
    return a[0] - margin < b[2] and b[0] < a[2] + margin and a[1] - margin < b[3] and b[1] < a[3] + margin


def _at_an_edge(event: dict[str, Any], shape: tuple[int, int]) -> bool:
    tall, wide = shape
    return min(event["x"], event["y"], wide - event["x"], tall - event["y"]) < 6.0


def _behind(mine: Any, event: dict[str, Any], shape: tuple[int, int]) -> bool:
    """Whether a thing that went at the picture's edge went between her and that edge."""
    tall, wide = shape
    x, y = event["x"], event["y"]
    nearest = min((x, "left"), (wide - x, "right"), (y, "top"), (tall - y, "bottom"))[1]
    return {"left": x < mine.x, "right": x > mine.x, "top": y < mine.y, "bottom": y > mine.y}[nearest]


def _turned(before: tuple[float, float] | None, thing: Any) -> bool:
    """Whether a thing is now going a way more than sixty degrees from the way it was, or has stopped."""
    if before is None:
        return False
    was, now = math.hypot(*before), math.hypot(thing.vx, thing.vy)
    if was < 1.0:
        return False
    if now < 0.3 * was:
        return True
    return (before[0] * thing.vx + before[1] * thing.vy) < 0.5 * was * now


def _inside(thing: Any, mine: Any) -> bool:
    """Whether a thing's middle is within her box."""
    left, top, right, bottom = mine.box()
    return left <= thing.x <= right and top <= thing.y <= bottom


#: How soon after a touch a loss is the touch's own doing.
AT_ONCE_S = 0.5


def _a_cost_shown(kept: _Evidence) -> bool:
    """Whether meeting a kind is shown to cost: at once, or against letting it by.

    A cost is believed at once when it comes at once: a bomb takes a life as
    it is touched. A loss that comes later, after a ball met and sent away
    came back and got by, is only evidence against meeting it when letting it
    by has been seen to do better. LIVE 2026-10-04, six seconds into a game,
    one return followed by a lost point and nothing yet let by taught her to
    dodge the ball.
    """
    return kept.immediate > 0 or kept.passes_settled > 0
