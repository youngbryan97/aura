"""The signs that come before harm, learned by play, and a thing coming at her faster than she can answer.

A person who fights the same enemy a few times stops watching where it is and starts watching what it does just before
it hurts them: it stops and draws back, it grows, it turns to face them, it rushes. Those are its tells, and the game
gives them so they can be read (an enemy's attack is shown before it lands: the telegraph of action games, the wind-up
of a boss). A falling block shakes before it drops, a platform flashes before it goes, a car speeds up before it pulls
out. Nobody learns these as a list; they learn that for this kind of thing this change came before getting hurt, more
often than not, and they get out of the way when they see it.

So, for every kind of thing she sees in play, the changes in it that a body can show are watched for: it starts moving
after standing still, it stops after moving, it grows, it speeds up, it turns toward her. Each change is held against
whether a loss followed within the time losses arrive, and against how often a loss would follow any moment by chance.
A change after which losses come more often than chance, and more often than not, is a tell of that kind; when a thing
of that kind shows it, she keeps clear of it until the moment has passed, and says once what she has learned.

And before anything is learned, a thing she does not yet know that is closing on her faster than she can answer is
kept clear of: what fills the eye fast is about to arrive, which animals and infants flinch from before they know what
it is (looming, and the time to contact it gives: Lee, 1976). Once she knows a kind is one to get, its coming is not a
threat.

Nothing here knows a game or an enemy.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SIGNS", "WarningSigns", "read_the_signs"]

#: The changes a thing can show, as she says them.
STARTS, STOPS, GROWS, SPEEDS_UP, TURNS_TOWARD = "starts moving", "stops", "grows", "speeds up", "turns toward me"
SIGNS = (STARTS, STOPS, GROWS, SPEEDS_UP, TURNS_TOWARD)
#: How long after a sign the loss it warns of arrives at most, and how far back a change is measured over.
WARNS_WITHIN_S = 2.0
OVER_S = 0.4
#: How much a thing must grow, or quicken, over that time for it to have; and how squarely it must face her.
GROWN = 1.3
QUICKER = 2.0
FACING = 0.8
#: The fewest times a sign must have come before a loss to be a tell.
FEWEST = 2


@dataclass
class _Seen:
    at: float
    area: float
    speed: float
    still: bool
    facing: float


@dataclass
class WarningSigns:
    """The tells of each kind of thing, learned from what followed each change; and what is showing one now."""

    #: thing number -> its recent looks.
    looks: dict[int, deque] = field(default_factory=dict)
    #: (kind, sign) -> [times followed by a loss, times not].
    counts: dict[tuple[int, str], list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    #: Signs shown and not yet settled: (kind, sign, at).
    pending: list[tuple[int, str, float]] = field(default_factory=list)
    #: thing number -> when it last showed a tell.
    warned: dict[int, float] = field(default_factory=dict)
    losses: list[float] = field(default_factory=list)
    #: thing number -> the changes it was showing at the last picture: a change counts once, when it begins.
    showing_now: dict[int, set[str]] = field(default_factory=dict)
    began: float | None = None
    said: set[tuple[int, str]] = field(default_factory=set)

    def saw(self, things: dict[int, Any], mine: Any, at: float, verdicts: list[dict[str, Any]] = ()) -> list[tuple[int, str]]:
        """One picture: each thing's changes noted, losses taken from the verdicts, and signs settled. The tells shown."""
        self.began = at if self.began is None else self.began
        self.losses = [v["at"] for v in verdicts if v.get("what") == "loss"][-200:]
        shown = []
        for number, thing in things.items():
            if mine is not None and number == getattr(mine, "number", None):
                continue
            changes = set(self._changes(number, thing, mine, at))
            begun, self.showing_now[number] = changes - self.showing_now.get(number, set()), changes
            for sign in sorted(begun):
                kind = int(getattr(thing, "kind", -1))
                self.pending.append((kind, sign, at))
                if self.is_a_tell(kind, sign):
                    self.warned[number] = at
                    shown.append((kind, sign))
        self._settle(at)
        for gone in set(self.looks) - set(things):
            del self.looks[gone]
            self.showing_now.pop(gone, None)
        return shown

    def _changes(self, number: int, thing: Any, mine: Any, at: float) -> list[str]:
        area = float(getattr(thing, "w", 0.0)) * float(getattr(thing, "h", 0.0))
        vx, vy = float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0))
        speed = math.hypot(vx, vy)
        facing = 0.0
        if mine is not None and speed > 0:
            dx, dy = float(mine.x) - float(thing.x), float(mine.y) - float(thing.y)
            apart = math.hypot(dx, dy)
            facing = (vx * dx + vy * dy) / (speed * apart) if apart > 0 else 0.0
        looks = self.looks.setdefault(number, deque(maxlen=24))
        now = _Seen(at, area, speed, bool(getattr(thing, "still", False)), facing)
        then = next((s for s in reversed(looks) if at - s.at >= OVER_S), None)
        looks.append(now)
        if then is None:
            return []
        out = []
        if then.still and not now.still:
            out.append(STARTS)
        if not then.still and now.still:
            out.append(STOPS)
        if then.area > 0 and now.area >= GROWN * then.area:
            out.append(GROWS)
        if then.speed > 0 and now.speed >= QUICKER * then.speed:
            out.append(SPEEDS_UP)
        if now.facing >= FACING > then.facing:
            out.append(TURNS_TOWARD)
        return out

    def _settle(self, at: float) -> None:
        waiting = []
        for kind, sign, when in self.pending:
            if at - when < WARNS_WITHIN_S:
                waiting.append((kind, sign, when))
                continue
            followed = any(when < lost <= when + WARNS_WITHIN_S for lost in self.losses)
            self.counts[(kind, sign)][0 if followed else 1] += 1
        self.pending = waiting[-400:]

    def _by_chance(self) -> float:
        """How often a loss follows any moment within the time a sign warns: the share of play such stretches cover."""
        if self.began is None or not self.losses:
            return 0.0
        span = max(WARNS_WITHIN_S, self.losses[-1] - self.began)
        return min(1.0, len(self.losses) * WARNS_WITHIN_S / span)

    def is_a_tell(self, kind: int, sign: str) -> bool:
        """Whether this change in this kind has come before losses often enough: at least twice, more often than not,
        and more often than a loss follows any moment."""
        hit, miss = self.counts.get((kind, sign), (0, 0))
        if hit < FEWEST:
            return False
        rate = hit / (hit + miss)
        return rate > 0.5 and rate > self._by_chance()

    def showing(self, thing: Any, at: float) -> bool:
        """Whether a thing showed one of its kind's tells lately enough that what it warns of may still come."""
        when = self.warned.get(int(getattr(thing, "number", -1)))
        return when is not None and at - when <= WARNS_WITHIN_S

    def closing_in(self, thing: Any, mine: Any, within_s: float) -> bool:
        """Whether a thing is coming at her fast enough to arrive within ``within_s``, by how fast the gap closes."""
        looks = self.looks.get(int(getattr(thing, "number", -1)))
        if mine is None or not looks or len(looks) < 2:
            return False
        vx, vy = float(getattr(thing, "vx", 0.0)), float(getattr(thing, "vy", 0.0))
        dx, dy = float(mine.x) - float(thing.x), float(mine.y) - float(thing.y)
        apart = math.hypot(dx, dy)
        closing = (vx * dx + vy * dy) / apart if apart > 0 else 0.0
        return closing > 0 and apart / closing < within_s

    def news(self, name_of: Any) -> list[str]:
        """What she has newly learned to read, said once each: "When the red ones stop, they hurt: I'll get clear"."""
        lines = []
        for (kind, sign) in list(self.counts):
            if (kind, sign) in self.said or not self.is_a_tell(kind, sign):
                continue
            self.said.add((kind, sign))
            lines.append(f"When {name_of(kind)} {_as_said(sign)}, harm follows: I'll get clear when I see it.")
        return lines


def read_the_signs(signs: WarningSigns | None, moves: Any, mine: Any, verdicts: list[dict[str, Any]], at: float,
                   say_once: Any) -> None:
    """One picture of play: each thing's changes held against the losses that followed, and a tell newly learned said
    once, the kind named as play names what to keep clear of. Only where a thing of hers is in play: with no body to get
    clear with, a change before a loss tells her nothing she can act on, and her own presses are what the losses follow
    (offline 2026-10-10 a game of timing presses to a needle learned "when the white bars stop, harm follows")."""
    if signs is None or mine is None:
        return
    signs.saw(moves.things, mine, at, verdicts)
    from core.agency.naming_what_she_sees import named_for_its_part, plural

    def name_of(kind: int) -> str:
        sample = next((t for t in moves.things.values() if t.kind == kind), None)
        name = named_for_its_part(moves, kind, sample, "avoid")
        return name if name[:1].isupper() else f"the {plural(name)}"

    for line in signs.news(name_of):
        say_once(line)


def _as_said(sign: str) -> str:
    return {STARTS: "start moving", STOPS: "stop", GROWS: "grow", SPEEDS_UP: "speed up",
            TURNS_TOWARD: "turn toward me"}.get(sign, sign)
