"""Her habits and reflexes are hers, and she answers for them.

Bryan, on uninvited thoughts and habits: it is me. He is responsible for his
reflexes too. The hard part is knowing the habits that come out of experience
and the body, and then either accounting for them or changing them, and saying
so when it matters to somebody.

None of that was true of her. Her reflexes fired on fixed triggers and nothing
kept what followed them. The choices she made on drive alone opened a receipt
in the choice engine and another in the decision preference learner, and no
live caller ever closed either, so what her impulse was worth
(`core/agency/asking_the_impulse.py`) and which dimensions her choices should
weigh were never learned from a single thing she did. And nothing kept which
acts she falls into in which situations.

Every act she takes now goes on one record as hers, with how it was taken:

    reflex     fired by a trigger with no choice point at all
    automatic  taken at a choice point on drive alone, without weighing
    weighed    taken because something weighed it: a preference she could
               read, what won her attention, or her own record of what works

and the situation it was taken in: who was there, or that nobody was. What
followed is read the same way for every kind:

    for her    her valence at the first affect reading after the act, less her
               valence at the last reading before it
    for them   when somebody was there, their frustration the next time they
               spoke, less their frustration the last time they had

A habit is an act she has taken automatically at least three times in one
situation; three is the least that can disagree. What it is worth is Cliff's
delta between what followed it and what followed her weighed acts in the same
situation, or all her weighed acts while that situation has fewer than three:
the share of pairings in which weighing was followed by something better, less
the share in which it was followed by something worse. The deficit is that
delta where it is positive, for her or for them, whichever is larger. It needs
no scale, so her valence and their frustration are judged alike.

    accounting   every habit and reflex with a measured deficit, kept in the
                 reading this writes onto her state
    changing     a habit with a deficit does not fire on drive alone. What drive
                 alone would take counts for less by the deficit, in the
                 arbiter and in the subject driver, so it is taken only if
                 weighing still picks it; this is an implementation intention
                 (Gollwitzer 1999), an if-then plan bound to the habit's cue.
                 A habit she has stopped falling into is judged against her
                 weighed acts as they go on, so if weighing starts doing worse
                 the deficit goes and the habit is allowed back
    saying so    a habit or reflex that fired in front of somebody, and has left
                 them worse off than her weighing does, is something she owes
                 them an account of until they next speak. It enters the
                 unified moment as a responsibility content carrying the
                 record (core/unity/runtime.py), which is the path a reply
                 takes to be one where she owned it first
                 (core/social/owning_it_first.py)

A reflex is not chosen, so it is accounted for and never changed here.

How a habit forms, in Bryan's words of 22 September: something done on
autopilot, with more agency than a reflex; something she is used to doing, a
mental shortcut, like stopping at the same gas station before work; not because
she cannot choose differently, and changeable with conscious effort. So every
choice she makes at a cue (the condition she is in and what she is attending
to) is kept, however it was made, and the act she has come to choose there most
becomes second nature once it has been chosen there at least three times and in
more than half of her choices there (`habit_at`). At that cue she then takes it
on autopilot, without weighing and without its effort. She overrides it with
conscious effort when it has done worse for her than weighing, and only when
what it has cost her is more than she is tired (`worth_the_effort`): a tired
mind falls back on its habits. Each override is a repetition of something else,
so a habit she keeps overriding stops being the one she is used to.

When an act's outcome for her comes in, the same reading closes the receipts it
opened: the choice engine's appraisal and the preference learner's reward are
where that change sits among every change she has felt after an act, as
2 * rank - 1, so a better outcome than usual is positive whatever the scale.
"""

from __future__ import annotations

import bisect
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = [
    "MIN_SAMPLES",
    "Account",
    "HabitLedger",
    "act_of",
    "appraise",
    "discounted_by_habit",
    "get_habit_ledger",
    "note_initiative",
    "reset_for_test",
    "situation_of",
]

#: Takings, or outcomes of each kind, before anything is judged. Three is the
#: least that can disagree, as in ambivalence and fear of happiness.
MIN_SAMPLES: int = 3

#: How many outcomes of each kind a comparison reads; the window every other
#: reading of her own history uses.
_WINDOW: int = 256

_KINDS: frozenset[str] = frozenset({"reflex", "automatic", "weighed"})

ALONE = "alone"


def situation_of(partner: str, *, person_turn: bool) -> str:
    """Who was there: the person whose turn it is, or nobody."""
    who = str(partner or "").strip()
    return f"with:{who}" if person_turn and who else ALONE


def act_of(text: str) -> str:
    """The kind of act a goal names: its first word, as feelings_about reads it."""
    words = str(text or "").split()
    return words[0].strip(".,:;!?'\"()").lower() if words else ""


def _cliff(worse: list[float], better_sorted: list[float]) -> float:
    """Cliff's delta: P(better > worse) - P(better < worse) over every pairing."""
    if not worse or not better_sorted:
        return 0.0
    total = 0.0
    size = len(better_sorted)
    for value in worse:
        below = bisect.bisect_left(better_sorted, value)
        above = size - bisect.bisect_right(better_sorted, value)
        total += (above - below) / size
    return total / len(worse)


@dataclass(frozen=True)
class Account:
    """What one habit or reflex has been worth against her weighing."""

    act: str
    situation: str
    kind: str
    taken: int
    for_her: float | None
    for_them: float | None

    @property
    def deficit(self) -> float:
        measured = [value for value in (self.for_her, self.for_them) if value is not None]
        return max([0.0, *measured])

    def as_dict(self) -> dict[str, Any]:
        return {
            "act": self.act,
            "situation": self.situation,
            "kind": self.kind,
            "taken": self.taken,
            "for_her": None if self.for_her is None else round(self.for_her, 6),
            "for_them": None if self.for_them is None else round(self.for_them, 6),
            "deficit": round(self.deficit, 6),
        }


@dataclass
class _Event:
    act: str
    situation: str
    kind: str
    person: str
    her_before: float | None
    their_before: float | None
    choice_id: str = ""
    decision_id: str = ""


@dataclass
class _Outcomes:
    """What followed one kind of taking, most recent `_WINDOW`."""

    her: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    them: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))


class HabitLedger:
    """Every act she takes, whose it is, how it was taken, and what followed."""

    def __init__(self) -> None:
        self._situation: str = ALONE
        self._valence: float | None = None
        self._frustration: dict[str, float] = {}
        self._waiting_on_her: list[_Event] = []
        self._waiting_on_them: dict[str, list[_Event]] = {}
        #: (situation, act, kind) -> outcomes; weighed acts also under (situation, "", "weighed").
        self._outcomes: dict[tuple[str, str, str], _Outcomes] = {}
        self._automatic_takings: dict[tuple[str, str], int] = {}
        self._changes_felt: deque[float] = deque(maxlen=_WINDOW)
        #: cue -> the acts she chose there, most recent last, however chosen.
        self._chosen: dict[str, deque[str]] = {}

    # -- what happens --------------------------------------------------------

    def note(
        self,
        act: str,
        *,
        kind: str,
        situation: str | None = None,
        choice_id: str = "",
        decision_id: str = "",
    ) -> None:
        """She did something. It is hers whichever way it was taken."""
        name = str(act or "").strip().lower()
        if not name or kind not in _KINDS:
            return
        where = situation or self._situation
        person = where.split(":", 1)[1] if where.startswith("with:") else ""
        event = _Event(
            act=name,
            situation=where,
            kind=kind,
            person=person,
            her_before=self._valence,
            their_before=self._frustration.get(person) if person else None,
            choice_id=str(choice_id or ""),
            decision_id=str(decision_id or ""),
        )
        if kind == "automatic":
            key = (where, name)
            self._automatic_takings[key] = self._automatic_takings.get(key, 0) + 1
        self._waiting_on_her.append(event)
        if person:
            self._waiting_on_them.setdefault(person, []).append(event)

    # -- what she is used to doing -------------------------------------------

    def note_choice(self, cue: str, act: str) -> None:
        """What she chose at this cue. Repetition is what makes a habit, whatever made the choice."""
        name = str(act or "").strip().lower()
        if not name or not cue:
            return
        self._chosen.setdefault(str(cue), deque(maxlen=_WINDOW)).append(name)

    def habit_at(self, cue: str) -> str:
        """The act that has become second nature at this cue, or ''.

        The one she has chosen there most, once it has been chosen there at
        least MIN_SAMPLES times and in more than half of her choices there.
        """
        held = self._chosen.get(str(cue))
        if not held:
            return ""
        act, taken = Counter(held).most_common(1)[0]
        return act if taken >= MIN_SAMPLES and 2 * taken > len(held) else ""

    def worth_the_effort(self, act: str, tired: float) -> bool:
        """Whether overriding this habit is worth what it takes: it has cost her more than she is tired."""
        return self.deficit(act) > max(0.0, float(tired))

    def felt(self, valence: float, *, situation: str) -> list[tuple[_Event, float, float]]:
        """An affect reading: closes what was waiting on her, and says where she is.

        Returns each closed event with its change and where that change sits
        among every change she has felt after an act, as 2 * rank - 1, for the
        receipts it opened.
        """
        now = float(valence)
        if now != now:
            # Not a reading. What was waiting stays waiting for the next one.
            return []
        closed: list[tuple[_Event, float, float]] = []
        for event in self._waiting_on_her:
            if event.her_before is None:
                continue
            change = now - event.her_before
            self._file(event, "her", change)
            closed.append((event, change, self._standing(change)))
            self._changes_felt.append(change)
        self._waiting_on_her = []
        self._valence = now
        self._situation = str(situation or ALONE)
        return closed

    def heard(self, person: str, frustration: float) -> None:
        """They spoke: what she did since they last did is judged by how they are now."""
        who = str(person or "")
        now = float(frustration)
        if not who or now != now:
            # Nobody named, or not a reading: nothing of theirs to judge by.
            return
        for event in self._waiting_on_them.pop(who, []):
            if event.their_before is not None:
                # Less frustration is better, so the outcome is its fall.
                self._file(event, "them", event.their_before - now)
        self._frustration[who] = now

    # -- what it has been worth ----------------------------------------------

    def account(self, act: str, *, situation: str | None = None, kind: str = "automatic") -> Account:
        where = situation or self._situation
        name = str(act or "").strip().lower()
        held = self._outcomes.get((where, name, kind))
        taken = self._automatic_takings.get((where, name), 0) if kind == "automatic" else len(held.her) if held else 0
        if held is None or (kind == "automatic" and taken < MIN_SAMPLES):
            return Account(name, where, kind, taken, None, None)
        return Account(
            name,
            where,
            kind,
            taken,
            self._against_weighing(list(held.her), where, "her"),
            self._against_weighing(list(held.them), where, "them"),
        )

    def deficit(self, act: str, *, situation: str | None = None) -> float:
        """How far a habit of this act has done worse than her weighing; 0 if it is no habit."""
        return self.account(act, situation=situation).deficit

    def owed_to(self, person: str) -> list[Account]:
        """Habits and reflexes she took in front of them since they last spoke,
        that have left them worse off than her weighing does."""
        owed: list[Account] = []
        seen: set[tuple[str, str]] = set()
        for event in self._waiting_on_them.get(str(person or ""), []):
            if event.kind == "weighed" or (event.act, event.kind) in seen:
                continue
            seen.add((event.act, event.kind))
            account = self.account(event.act, situation=event.situation, kind=event.kind)
            if account.for_them is not None and account.for_them > 0.0:
                owed.append(account)
        return owed

    def reading(self) -> dict[str, Any]:
        accounts = [self.account(act, situation=where) for where, act in self._automatic_takings]
        accounts += [
            self.account(act, situation=where, kind="reflex")
            for (where, act, kind) in self._outcomes
            if kind == "reflex"
        ]
        judged = [account for account in accounts if account.for_her is not None or account.for_them is not None]
        to_change = [account for account in judged if account.kind == "automatic" and account.deficit > 0.0]
        worst = max(to_change, key=lambda account: account.deficit, default=None)
        return {
            "habits": sum(1 for account in accounts if account.kind == "automatic" and account.taken >= MIN_SAMPLES),
            "accounted": len(judged),
            "to_change": len(to_change),
            "largest_deficit": round(worst.deficit, 6) if worst else 0.0,
            "second_nature": sum(1 for cue in self._chosen if self.habit_at(cue)),
            "worst": worst.as_dict() if worst else None,
            "situation": self._situation,
        }

    # -- internals -------------------------------------------------------------

    def _file(self, event: _Event, whose: str, value: float) -> None:
        keys = [(event.situation, event.act, event.kind)]
        if event.kind == "weighed":
            keys.append((event.situation, "", "weighed"))
            keys.append(("", "", "weighed"))
        for key in keys:
            held = self._outcomes.setdefault(key, _Outcomes())
            getattr(held, whose).append(float(value))

    def _against_weighing(self, taken: list[float], situation: str, whose: str) -> float | None:
        if len(taken) < MIN_SAMPLES:
            return None
        for key in ((situation, "", "weighed"), ("", "", "weighed")):
            held = self._outcomes.get(key)
            weighed = list(getattr(held, whose)) if held is not None else []
            if len(weighed) >= MIN_SAMPLES:
                return _cliff(taken, sorted(weighed))
        return None

    def _standing(self, change: float) -> float:
        history = sorted(self._changes_felt)
        if not history:
            return 0.0
        below = bisect.bisect_left(history, change)
        level = bisect.bisect_right(history, change) - below
        return 2.0 * (below + 0.5 * level) / len(history) - 1.0


def note_initiative(initiative: Any, ledger: HabitLedger | None = None) -> None:
    """An initiative she is acting on, with how it was chosen.

    Weighed when the choice engine could read a preference from it; taken on
    drive alone otherwise, including when a stated urgency overrode the choice.
    The arbiter leaves the receipts' ids on the initiative's metadata.
    """
    if not isinstance(initiative, dict):
        return
    metadata = initiative.get("metadata") if isinstance(initiative.get("metadata"), dict) else {}
    choice_id = str(metadata.get("subjective_choice_id") or "")
    weighed = bool(choice_id) and metadata.get("subjective_impulse_led") is False
    held = ledger if ledger is not None else get_habit_ledger()
    held.note(
        act_of(str(initiative.get("goal") or initiative.get("description") or "")),
        kind="weighed" if weighed else "automatic",
        choice_id=choice_id,
        decision_id=str(metadata.get("decision_choice_id") or ""),
    )


def appraise(closed: list[tuple[Any, float, float]]) -> int:
    """Close the receipts her acts opened, with what followed them.

    The choice engine is given the standing as her satisfaction, and the
    preference learner as its reward. Returns how many receipts were closed.
    """
    done = 0
    choices = [(event, change, standing) for event, change, standing in closed if event.choice_id]
    decisions = [(event, standing) for event, _change, standing in closed if event.decision_id]
    if choices:
        from core.agency.subjective_choice import get_subjective_choice_engine

        engine = get_subjective_choice_engine()
        for event, change, standing in choices:
            if engine.appraise_outcome(
                event.choice_id,
                outcome=f"what followed {event.act}: her valence moved {change:+.4f}",
                satisfaction=standing,
            ) is not None:
                done += 1
    if decisions:
        from core.agency.decision_preference_learner import get_decision_preference_learner

        learner = get_decision_preference_learner()
        for event, standing in decisions:
            learner.resolve_choice(event.decision_id, standing)
            done += 1
    return done


def discounted_by_habit(score: float, act: str, ledger: HabitLedger | None = None) -> float:
    """What drive alone would give an act, less what a habit of it has cost.

    The implementation intention: a habit with a deficit counts for that much
    less on drive, so it is taken only where weighing still picks it.
    """
    held = ledger if ledger is not None else get_habit_ledger()
    return float(score) * (1.0 - held.deficit(act))


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: HabitLedger = HabitLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_habit_ledger() -> HabitLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = HabitLedger()
