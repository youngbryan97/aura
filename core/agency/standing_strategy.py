"""Holding an approach, and knowing in advance what would end it.

Predicting one move and grading it afterwards is reactive: it assumes the
world sits still and corrects once the world says otherwise. That is not how
anyone plays a game that deals a new tile after every move, or does anything
in a world that keeps moving while they work.

What is missing between "one move at a time" and "a fixed plan" is a
standing approach: a line she is taking, the reason she is taking it, and the
condition that would tell her it has stopped being right — decided when she
adopts it rather than discovered when it fails. That is the difference
between shifting because something went wrong and shifting because she was
watching for the thing that makes it wrong.

Two properties make it general. The pivot condition is checked mechanically
against an ordinary reading, so it does not depend on her noticing. And the
alternatives are named at the same time, so when the condition fires she is
choosing between approaches she has already thought about rather than
starting over in the worst moment to start over.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.agency.deliberate_action import ActionOption, Expectation
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Strategy")

#: How many moves an approach runs before it is reconsidered even if nothing
#: has broken. A position drifts without ever tripping a condition, and an
#: approach nobody revisits becomes a habit.
RECONSIDER_AFTER = 12


@dataclass(frozen=True)
class Strategy:
    """A line she is taking, why, and what would end it."""

    approach: str
    because: str = ""
    #: What has to stay true for this to remain the right line. Checked
    #: against an ordinary reading, so it does not rely on her noticing.
    holds_while: Expectation = field(default_factory=Expectation)
    #: What she would do instead, named now rather than in the moment the
    #: condition fires.
    otherwise: tuple[str, ...] = ()
    adopted_at: float = field(default_factory=time.time)
    adopted_on_move: int = 0

    def as_evidence(self) -> list[str]:
        """The approach, as lines a decision can rest on."""
        lines = [f"The approach I am taking — {self.approach}"]
        if self.because:
            lines.append(f"Why — {self.because}")
        if self.holds_while.describes:
            lines.append(f"This holds while — {self.holds_while.describes}")
        if self.otherwise:
            lines.append(f"If it stops holding — {', '.join(self.otherwise)}")
        return lines

    def narrate(self) -> str:
        said = f"Plan: {self.approach}"
        if self.because:
            said = f"{said} — {self.because}"
        if self.holds_while.describes:
            said = f"{said}. Watching for: {self.holds_while.describes}"
        return said


def still_holds(strategy: Strategy | None, reading: str, moves_made: int = 0) -> tuple[bool, str]:
    """Whether the approach is still the right one, and why not if it is not.

    Two ways it ends. The condition it named stopped being true, which is the
    thing she was watching for. Or it has simply been running a long time
    without being revisited, because a position drifts without ever tripping
    a condition and an approach nobody reconsiders becomes a habit.
    """
    if strategy is None:
        return False, "no approach yet"
    if moves_made - strategy.adopted_on_move >= RECONSIDER_AFTER:
        return False, f"it has run {moves_made - strategy.adopted_on_move} moves without a fresh look"
    verdict = strategy.holds_while.check(reading, reading)
    if verdict.missing:
        return False, f"what it depends on is gone: {', '.join(verdict.missing)}"
    if verdict.lingering:
        return False, f"what it was avoiding has happened: {', '.join(verdict.lingering)}"
    return True, ""


def _watchable(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What a stated condition says must be present, and what must not be.

    Read from how the condition is phrased rather than from a schema, because
    the condition comes back in her own words: "while the 64 stays in the
    corner" names something to keep, "until the board fills" names something
    to avoid.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return (), ()
    keep: list[str] = []
    avoid: list[str] = []
    for match in re.finditer(r"\b(\d[\d,]{0,6})\b", body):
        value = match.group(1).replace(",", "")
        before = body[: match.start()].lower()
        if re.search(r"\b(?:no|not|without|avoid|never|unless)\b[^.]{0,30}$", before):
            avoid.append(value)
        else:
            keep.append(value)
    return tuple(dict.fromkeys(keep)), tuple(dict.fromkeys(avoid))


def read_strategy(reply: str, options: Sequence[ActionOption] = ()) -> Strategy | None:
    """An approach read out of a reply, when it names one.

    Deliberately forgiving about shape and strict about substance: an
    approach with nothing that could end it is not an approach, it is a
    preference, and the whole value here is knowing in advance what would
    change her mind.
    """
    text = " ".join(str(reply or "").split())
    if not text:
        return None
    approach = ""
    for pattern in (
        r"\bplan\s*[:\-]\s*(?P<said>[^.]{4,120})",
        r"\bapproach\s*[:\-]\s*(?P<said>[^.]{4,120})",
        r"\bI(?:'ll| will| am going to)\s+(?P<said>[^.]{4,120})",
        r"\bstrategy\s+is\s+(?P<said>[^.]{4,120})",
    ):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            approach = found.group("said").strip(" ,;")
            break
    if not approach:
        return None

    condition = ""
    for pattern in (
        r"\b(?:while|as long as|so long as|provided)\s+(?P<said>[^.]{4,120})",
        r"\bwatch(?:ing)? for\s+(?P<said>[^.]{4,120})",
        r"\buntil\s+(?P<said>[^.]{4,120})",
    ):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            condition = found.group("said").strip(" ,;")
            break
    if not condition:
        return None

    keep, avoid = _watchable(condition)
    because = ""
    reason = re.search(r"\bbecause\s+(?P<said>[^.]{4,140})", text, re.IGNORECASE)
    if reason:
        because = reason.group("said").strip(" ,;")

    otherwise: list[str] = []
    for pattern in (r"\botherwise\s+(?P<said>[^.]{3,80})", r"\bif not,?\s+(?P<said>[^.]{3,80})"):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            otherwise.append(found.group("said").strip(" ,;"))
    return Strategy(
        approach=approach,
        because=because,
        holds_while=Expectation(changed=False, contains=keep, absent=avoid, describes=condition),
        otherwise=tuple(otherwise),
    )


def _asking_for_an_approach(goal: str, situation: str, options: Sequence[ActionOption]) -> str:
    names = ", ".join(option.name for option in options)
    return (
        f"Decide how to play toward this goal, not just the next move: {goal}. "
        f"Say the approach you are taking, why, and what you are watching for that "
        f"would tell you to change it. The moves available are: {names}."
    )


async def settle_on_an_approach(
    goal: str,
    situation: str,
    options: Sequence[ActionOption],
    *,
    think: Any,
    knowledge: Sequence[str] = (),
    history: Sequence[Any] = (),
    previous: Strategy | None = None,
    moves_made: int = 0,
) -> Strategy | None:
    """Decide the line to take, and what would end it.

    Asked as its own question rather than folded into choosing a move: an
    approach that is re-derived every move is not an approach, and a move
    chosen without one is a reaction.
    """
    if think is None:
        return None
    evidence = [f"What is visible now: {situation}", *knowledge]
    if previous is not None:
        evidence.append(f"The approach I was taking — {previous.approach}")
        evidence.append("It stopped being right, which is why this is being decided again.")
    for attempt in list(history)[-3:]:
        if hasattr(attempt, "as_evidence"):
            evidence.append(attempt.as_evidence())
    try:
        reply = await think(_asking_for_an_approach(goal, situation, options), evidence)
    except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation(
            "standing_strategy", exc, severity="info", action="carried on without a stated approach"
        )
        return None
    settled = read_strategy(reply or "", options)
    if settled is None:
        return None
    return Strategy(
        approach=settled.approach,
        because=settled.because,
        holds_while=settled.holds_while,
        otherwise=settled.otherwise,
        adopted_on_move=int(moves_made),
    )
