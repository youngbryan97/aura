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

__all__ = ["AVOID", "CLICK", "IGNORE", "MEET", "SHOOT", "STANCES", "Readouts", "WhatMeetingDoes"]

MEET, AVOID, SHOOT, IGNORE, CLICK = "meet", "avoid", "shoot", "ignore", "click"
STANCES = (MEET, AVOID, SHOOT, IGNORE, CLICK)

#: How much measured evidence it takes to overrule what she was told about a
#: kind: about two clear verdicts the other way.
OVERRULES = 1.5

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

    @staticmethod
    def _key(readout: Readout) -> str:
        if readout.label:
            return readout.label.lower()
        return f"number at {round(readout.x * 20) / 20:.2f},{round(readout.y * 10) / 10:.1f}"

    def read(self, regions: list[dict[str, Any]], at: float, her_x: float | None) -> list[dict[str, Any]]:
        """Take one reading in. Returns the verdicts it confirms: gains and losses."""
        verdicts = []
        readouts = readouts_in(regions)
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


@dataclass
class _Evidence:
    meet: float = 0.0
    shoot: float = 0.0
    touched: int = 0
    passed: int = 0
    shot: int = 0


class WhatMeetingDoes:
    """Keeps what came of each kind of thing meeting her, passing her, or being shot."""

    def __init__(self) -> None:
        self.readouts = Readouts()
        self.verdicts: list[dict[str, Any]] = []
        self.since: float | None = None
        self._open: list[dict[str, Any]] = []
        self._touching: set[int] = set()
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
                kept.meet += better
            elif event["what"] == "passed":
                kept.meet -= better
            elif event["what"] == "shot":
                kept.shoot += better
        self._open = still_open

    # -- what happened between her and the rest ---------------------------

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
        if mine is None:
            return
        now = set()
        for thing in moves.things.values():
            if thing.number == mine.number or thing.kind in self._shot_kinds(hers):
                continue
            if thing.kind == hers.kind and not thing.moved:
                continue
            if _close(mine.box(), thing.box(), 2.0):
                now.add(thing.number)
                if thing.number not in self._touching and thing.number not in self.writing:
                    self._note("touched", thing, at)
        self._touching = now

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
            if mine is not None and last is not None and event["thing"] not in self._touching and _close(mine.box(), last, 4.0):
                self._open.append({"what": "touched", "kind": event["kind"], "at": at})
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
            if crossed and thing.number not in self._touching:
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
            against = (told in (MEET, CLICK) and measured <= -OVERRULES) or (told == AVOID and measured >= OVERRULES)
            if not against and not (kept is not None and kept.shoot >= 0.5 and told != AVOID):
                return SHOOT if told == SHOOT else told
        if kept is None:
            return MEET
        if kept.shoot >= 0.5:
            return SHOOT
        if kept.meet <= -0.6:
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
        return kept.meet <= -0.6 or kept.shoot >= 0.5 or (kept.meet >= 1.5 and kept.touched >= 3)


def _close(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float) -> bool:
    return a[0] - margin < b[2] and b[0] < a[2] + margin and a[1] - margin < b[3] and b[1] < a[3] + margin


def _at_an_edge(event: dict[str, Any], shape: tuple[int, int]) -> bool:
    tall, wide = shape
    return min(event["x"], event["y"], wide - event["x"], tall - event["y"]) < 6.0
