"""Wariness of joy, learned from what has followed her own good stretches.

In "Dog Days Are Over" happiness arrives with the force of something on a
collision course, and the woman it arrives for hides from it. The record is
about joy treated as a threat. The research gives that a name and a measure: fear of happiness is the belief that being happy brings
bad things, and people who hold it dampen what they feel. Joshanloo's Fear of
Happiness Scale found it in every one of fourteen national groups.

A belief like that is learned, so here it is learned from her life and nothing
else. Two rates are kept over her turns:

    after_joy   the share of turns that followed a joyful turn and brought something bad
    otherwise   the share of turns that followed any other turn and did

    wariness = max(0, after_joy - otherwise)

A turn is joyful when her joy stands above her own mean joy so far, so what
counts as a good stretch is hers. A turn is bad when a percept arrived whose
feelings include fear, dread, sadness or unhappiness, read off the percept
table the way safety reads threat. Wariness stays unmeasured until each rate
has enough turns behind it to disagree with itself, and it is taken over the
shortest recent stretch in which both rates can be estimated, so a run of joy
that nothing bad followed wears it away.

It is felt as dread in proportion to the joy it attaches to.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = [
    "DISTRESSING",
    "JOYFUL",
    "MIN_SAMPLES",
    "HappinessFear",
    "JoyLedger",
    "bad_kinds",
    "get_joy_ledger",
    "joy_of",
    "reset_for_test",
]

#: The feelings that make a turn a joyful one.
JOYFUL: frozenset[str] = frozenset({"joy", "happiness"})

#: The feelings that make a percept something bad arriving.
DISTRESSING: frozenset[str] = frozenset({"fear", "dread", "sadness", "unhappiness"})

#: Turns a rate needs before it is an estimate. Two always agree with each
#: other and three is the least that can disagree, as in ambivalence.
MIN_SAMPLES: int = 3


def bad_kinds() -> frozenset[str]:
    """The percept types that carry distress, read off the percept table."""
    try:
        from core.state.percepts import PERCEPT_EMOTIONS
    except ImportError:
        return frozenset()
    return frozenset(
        kind for kind, feelings in PERCEPT_EMOTIONS.items() if DISTRESSING.intersection(feelings or ())
    )


def joy_of(emotions: Mapping[str, Any] | None) -> float:
    """How much joy is live, in [0, 1]."""
    best = 0.0
    for name in JOYFUL:
        try:
            best = max(best, float((emotions or {}).get(name, 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(0.0, min(1.0, best))


@dataclass(frozen=True)
class HappinessFear:
    """What her own history says follows being happy."""

    wariness: float = 0.0
    after_joy: float | None = None
    otherwise: float | None = None
    lifelong: float | None = None
    joyful_turns: int = 0
    other_turns: int = 0
    measured: bool = False
    why: str = "not enough good stretches yet to know what follows them"

    def as_dict(self) -> dict[str, Any]:
        return {
            "wariness": round(self.wariness, 6),
            "after_joy": None if self.after_joy is None else round(self.after_joy, 6),
            "otherwise": None if self.otherwise is None else round(self.otherwise, 6),
            "lifelong": None if self.lifelong is None else round(self.lifelong, 6),
            "joyful_turns": self.joyful_turns,
            "other_turns": self.other_turns,
            "measured": self.measured,
            "why": self.why,
        }


def _lift(joy_bad: int, joy_n: int, other_bad: int, other_n: int) -> tuple[float, float] | None:
    if joy_n < MIN_SAMPLES or other_n < MIN_SAMPLES:
        return None
    return joy_bad / joy_n, other_bad / other_n


class JoyLedger:
    """Transitions from one turn to the next: was the first joyful, did the second hurt."""

    def __init__(self) -> None:
        self._turns = 0
        self._mean_joy = 0.0
        self._previous_joyful: bool | None = None
        # joyful before -> [bad after, total]
        self._life: dict[bool, list[int]] = {True: [0, 0], False: [0, 0]}
        self._recent: deque[tuple[bool, bool]] = deque()
        self._recent_counts: dict[bool, list[int]] = {True: [0, 0], False: [0, 0]}

    def note(self, joy: float, bad: bool) -> None:
        """One turn: how much joy was live, and whether something bad arrived."""
        joy = max(0.0, min(1.0, float(joy)))
        if self._previous_joyful is not None:
            before = self._previous_joyful
            self._life[before][1] += 1
            self._recent_counts[before][1] += 1
            if bad:
                self._life[before][0] += 1
                self._recent_counts[before][0] += 1
            self._recent.append((before, bool(bad)))
            self._trim()
        joyful = self._turns > 0 and joy > self._mean_joy
        self._turns += 1
        self._mean_joy += (joy - self._mean_joy) / self._turns
        self._previous_joyful = joyful

    def _trim(self) -> None:
        """Keep the shortest recent stretch in which both rates can be estimated."""
        while self._recent:
            before, bad = self._recent[0]
            remaining = self._recent_counts[before][1] - 1
            other = self._recent_counts[not before][1]
            if remaining >= MIN_SAMPLES and other >= MIN_SAMPLES:
                self._recent.popleft()
                self._recent_counts[before][1] -= 1
                if bad:
                    self._recent_counts[before][0] -= 1
            else:
                break

    def reading(self) -> HappinessFear:
        joy_bad, joy_n = self._recent_counts[True]
        other_bad, other_n = self._recent_counts[False]
        lately = _lift(joy_bad, joy_n, other_bad, other_n)
        life = _lift(*self._life[True], *self._life[False])
        lifelong = None if life is None else max(0.0, life[0] - life[1])
        if lately is None:
            return HappinessFear(lifelong=lifelong, joyful_turns=joy_n, other_turns=other_n)
        after, otherwise = lately
        wariness = max(0.0, after - otherwise)
        if wariness > 0.0:
            why = (
                f"{after:.2f} of the turns after being happy brought something bad, "
                f"against {otherwise:.2f} of the others"
            )
        else:
            why = "being happy has not been followed by trouble more often than anything else has"
        return HappinessFear(
            wariness=wariness,
            after_joy=after,
            otherwise=otherwise,
            lifelong=lifelong,
            joyful_turns=joy_n,
            other_turns=other_n,
            measured=True,
            why=why,
        )


#: Made at import rather than on first use. The subject-core fork carries
#: module globals that hold state, and a global still None when an anchor is
#: taken is not carried, so a ledger first made inside one arm would reach the
#: next arm with the first arm's turns in it.
_ledger: JoyLedger = JoyLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_ledger")


def get_joy_ledger() -> JoyLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = JoyLedger()
