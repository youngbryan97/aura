"""Two parties doing something together: its talk, its common ground, and what both can see.

Bryan, 13 October 2026: "I can ask Aura 'hey, watch this youtube video for 10
minutes and comment on it as you watch.' and she'll do that. as a task. but
that's different from asking 'Hey do you want to watch this video with me?'
... all as a way of *being* you know? ... that's not a task or a job it's
real-time interaction."

A task is finished by producing what was asked for. Two people watching a
film are doing something nothing finishes: it is over when one of them
leaves. Goffman's name for the unit is a "with", and for its talk an open
state of talk: either may speak, neither has to, and a silence is not a
conversation that failed (Goffman 1963; 1981). Nothing in her held that
state. A message was a request and its reply closed it, and whatever she
said unprompted was weighed as an interruption of somebody alone
(core/agency/ambient_governor.py), which is right for somebody alone and
wrong for somebody who asked her to stay.

What a with holds:

    its talk            Who spoke and when, and so the silences between
                        turns. How long this pair usually leaves a silence
                        is read off the silences they have left, and is the
                        only clock the with has.
    its common ground   What each can take the other to know. A thing is
                        presented by one party and is grounded when the
                        other gives evidence of having taken it in: by
                        naming it back, or by going on with the next
                        relevant thing (Clark and Schaefer 1989; Clark
                        1996). What is in view of both is common ground
                        without being said (Clark and Brennan 1991).
    what is in view     The shared field over time, so "go back" and "that
                        thing a moment ago" have something to refer to.
    how she is taken    Whether her remarks are taken up. An infant who
                        points to share a thing is satisfied only when the
                        adult looks at it and answers with interest, and
                        stops pointing for an adult who does not
                        (Liszkowski, Carpenter and Tomasello 2007). Her
                        remarks are counted the same way.

Nothing here decides anything. core/together/when_to_say_something.py reads
it to decide whether to speak.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["HER", "IT", "THEM", "Grounded", "Shown", "Together", "Turn", "With", "get_together"]

HER, THEM = "her", "them"
#: The shared thing itself making sound: a film's dialogue, a video's voice.
IT = "it"

#: How many turns, silences and views a with keeps to read itself by; the
#: window her record reads her own history over.
_WINDOW = 256


@dataclass
class Turn:
    by: str
    at: float
    until: float | None = None
    about: tuple[str, ...] = ()


@dataclass
class Shown:
    """What the shared field showed from a moment on, and how much it had changed."""

    at: float
    what: str
    where: tuple[float, float, float, float] | None = None
    change: float | None = None


@dataclass
class Grounded:
    """One thing in common ground: who put it there, and what shows the other took it in."""

    by: str
    at: float
    #: "named" when the other said it back, "went on" when they took the next
    #: turn, "in view" when both could see it, "" while it is only presented.
    evidence: str = ""
    taken_at: float | None = None


def _share_below(values: Iterable[float], value: float) -> float | None:
    """The part of `values` that `value` is at least as large as; nothing to read from none."""
    held = list(values)
    if not held:
        return None
    return sum(1 for item in held if item <= value) / len(held)


@dataclass
class With:
    """One standing shared activity with one other party."""

    who: str
    what: str
    proposed_by: str = THEM
    began: float = field(default_factory=time.time)
    ended: float | None = None
    left_by: str = ""
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    silences: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    ground: dict[str, Grounded] = field(default_factory=dict)
    field_of_view: deque[Shown] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    #: Her remarks nobody had asked for, and how many the other took up.
    remarks: int = 0
    taken_up: int = 0
    went_by: int = 0
    #: How many remarks the other asked for when it began ("think out loud while we watch"): counted as taken up in
    #: how welcome a remark is, until her own record outweighs them (core/intent/asks_for_remarks.py).
    invited: int = 0
    #: What she has already said her piece about.
    remarked_on: set[str] = field(default_factory=set)
    _waiting: Turn | None = None
    #: The words of the last screen reading the with saw, to measure change by.
    _screen_words: set[str] | None = None

    @property
    def open(self) -> bool:
        return self.ended is None

    # ── talk ──────────────────────────────────────────────────────────

    def said(
        self,
        by: str,
        at: float,
        *,
        about: Iterable[str] = (),
        until: float | None = None,
        unprompted: bool = False,
    ) -> Turn:
        """A turn of talk began. `about` is what it presented; `until` is when it ended, once known.

        `unprompted` marks a remark of hers that answered nothing.
        """
        about = tuple(dict.fromkeys(str(key) for key in about if str(key).strip()))
        last = self.turns[-1] if self.turns else None
        if last is not None:
            if last.until is None:
                # Nobody said when it ended, so it ended when the next began,
                # and that is no silence to learn from.
                last.until = max(at, last.at)
            elif at >= last.until and by != IT and last.by != IT:
                self.silences.append(at - last.until)
        turn = Turn(by=by, at=at, until=until, about=about)
        self._settle_her_remark(turn)
        self._ground(turn)
        self.turns.append(turn)
        if by == HER:
            self.remarked_on.update(about)
            if unprompted:
                self.remarks += 1
                self._waiting = turn
        return turn

    def stopped_speaking(self, by: str, at: float) -> None:
        """The turn `by` was in the middle of ended."""
        for turn in reversed(self.turns):
            if turn.by == by:
                if turn.until is None or turn.until > at:
                    turn.until = at
                return

    def _settle_her_remark(self, turn: Turn) -> None:
        """Whether the remark she is waiting on was taken up, now that somebody has spoken."""
        waiting = self._waiting
        if waiting is None or turn.by == IT:
            return
        if turn.by == HER:
            # She spoke again first: the remark went by, neither taken up nor turned down. Counted as turned down,
            # her remarks while they watched in silence drove how welcome one is to 0.13, and an eventful video
            # outweighed every remark after six minutes (LIVE 20 October 2026). It leaves the count.
            self._waiting = None
            self.went_by += 1
            return
        named = bool(set(turn.about) & set(waiting.about))
        ended = waiting.until if waiting.until is not None else waiting.at
        usual = _share_below(self.silences, turn.at - ended)
        # Taken up when they name what she raised, or answer sooner than half
        # of the silences this pair leaves. With no silence yet to compare
        # with, answering at all is taking it up.
        if named or usual is None or usual <= 0.5:
            self.taken_up += 1
        self._waiting = None

    def _ground(self, turn: Turn) -> None:
        if turn.by == IT:
            return
        other = THEM if turn.by == HER else HER
        previous = next((t for t in reversed(self.turns) if t.by == other), None)
        if previous is not None:
            for key in previous.about:
                held = self.ground.get(key)
                if held is not None and held.by == other and not held.evidence:
                    held.evidence = "named" if key in turn.about else "went on"
                    held.taken_at = turn.at
        for key in turn.about:
            held = self.ground.get(key)
            if held is None:
                self.ground[key] = Grounded(by=turn.by, at=turn.at)
            elif held.by != turn.by and held.evidence != "named":
                held.evidence, held.taken_at = "named", turn.at

    def speaking(self, now: float) -> str:
        """Who has the floor at `now`, or nobody."""
        for turn in reversed(self.turns):
            if turn.at <= now and (turn.until is None or turn.until > now):
                # A turn with no end yet is still going only if it is the newest.
                if turn.until is None and turn is not self.turns[-1]:
                    continue
                return turn.by
            if turn.until is not None and turn.until <= now:
                break
        return ""

    def quiet(self, now: float) -> float:
        """How long nobody in the with has spoken; nothing while somebody is."""
        people = [turn for turn in self.turns if turn.by != IT]
        if not people:
            return max(0.0, now - self.began)
        last = people[-1]
        if last.until is None or last.until > now:
            return 0.0
        return max(0.0, now - last.until)

    def moment(self, now: float) -> float | None:
        """How far into one of this pair's usual silences `now` is, in [0, 1].

        The part of the silences they have left that were no longer than
        this one. Nothing to read before they have left a silence.
        """
        if self.speaking(now) in (HER, THEM):
            return 0.0
        if self.invited and not self.silences:
            return 1.0  # asked to talk as it goes, and no pause of theirs to break
        return _share_below(self.silences, self.quiet(now))

    def welcome(self) -> float:
        """How likely a remark of hers is to be taken up, from how her remarks have been.

        The mean of what the counts leave possible, starting from knowing
        nothing (Laplace 1814).
        """
        settled = self.remarks - self.went_by - (1 if self._waiting is not None else 0)
        return (self.taken_up + self.invited + 1) / (settled + self.invited + 2)

    # ── common ground ─────────────────────────────────────────────────

    def in_common(self, key: str) -> bool:
        """Whether both can be taken to know of it: grounded in talk, or in view of both."""
        held = self.ground.get(key)
        return bool(held is not None and held.evidence)

    def she_has_said_her_piece(self, key: str) -> bool:
        return key in self.remarked_on

    # ── what is in view ───────────────────────────────────────────────

    def shown(
        self,
        what: str,
        at: float,
        *,
        where: tuple[float, float, float, float] | None = None,
        change: float | None = None,
    ) -> Shown:
        """The shared field shows `what` from `at`; `change` is how different it is from before."""
        seen = Shown(at=at, what=str(what), where=where, change=change)
        self.field_of_view.append(seen)
        key = str(what).strip()
        if key and key not in self.ground:
            self.ground[key] = Grounded(by=IT, at=at, evidence="in view", taken_at=at)
        return seen

    def in_view(self, at: float) -> Shown | None:
        """What the shared field showed at `at`."""
        found = None
        for seen in self.field_of_view:
            if seen.at <= at:
                found = seen
            else:
                break
        return found

    def before(self, at: float, *, steps: int = 1) -> Shown | None:
        """What was in view `steps` changes of the field before what was showing at `at`."""
        earlier = [seen for seen in self.field_of_view if seen.at <= at]
        return earlier[-1 - steps] if len(earlier) > steps else None

    def eventful(self) -> float | None:
        """How much the field is changing now against how much it has changed, in [0, 1]."""
        changes = [seen.change for seen in self.field_of_view if seen.change is not None]
        if not changes:
            return None
        return _share_below(changes[:-1], changes[-1]) if len(changes) > 1 else None

    # ── ending ────────────────────────────────────────────────────────

    def leave(self, by: str, at: float) -> None:
        """One of them left. A with is ended by leaving and by nothing else."""
        if self.ended is None:
            self.ended, self.left_by = at, by

    def status(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        return {
            "who": self.who,
            "what": self.what,
            "proposed_by": self.proposed_by,
            "open": self.open,
            "for_s": round((self.ended or now) - self.began, 1),
            "turns": len(self.turns),
            "quiet_s": round(self.quiet(now), 1),
            "moment": self.moment(now),
            "welcome": round(self.welcome(), 3),
            "remarks": self.remarks,
            "taken_up": self.taken_up,
            "in_common": sorted(key for key in self.ground if self.in_common(key)),
            "in_view": getattr(self.in_view(now), "what", ""),
        }


class Together:
    """The withs she is in, by who they are with."""

    def __init__(self) -> None:
        self._lock = checked_lock("together.withs")
        self._withs: dict[str, With] = {}
        self._past: deque[With] = deque(maxlen=_WINDOW)

    def begin(self, who: str, what: str, *, proposed_by: str = THEM, at: float | None = None) -> With:
        """She and `who` are doing `what` together from now. One with per party."""
        at = time.time() if at is None else at
        with self._lock:
            held = self._withs.get(who)
            if held is not None and held.open:
                if held.what == what:
                    return held
                held.leave(HER, at)
                self._past.append(held)
            new = self._withs[who] = With(who=who, what=what, proposed_by=proposed_by, began=at)
            return new

    def current(self, who: str | None = None) -> With | None:
        """The open with she is in with `who`, or the one she is in if there is one."""
        with self._lock:
            if who is not None:
                held = self._withs.get(who)
                return held if held is not None and held.open else None
            open_now = [held for held in self._withs.values() if held.open]
            return max(open_now, key=lambda held: held.began) if open_now else None

    def end(self, who: str, *, by: str, at: float | None = None) -> With | None:
        at = time.time() if at is None else at
        with self._lock:
            held = self._withs.pop(who, None)
            if held is None:
                return None
            held.leave(by, at)
            self._past.append(held)
            return held

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open": [held.status() for held in self._withs.values() if held.open],
                "past": len(self._past),
            }


_TOGETHER: Together | None = None
_TOGETHER_LOCK = checked_lock("together.singleton")


def get_together() -> Together:
    global _TOGETHER
    with _TOGETHER_LOCK:
        if _TOGETHER is None:
            _TOGETHER = Together()
        return _TOGETHER
