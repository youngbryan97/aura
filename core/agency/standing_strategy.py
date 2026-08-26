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

import asyncio

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


#: Ways of naming a place in something laid out in rows.
_PLACES: tuple[tuple[str, str], ...] = (
    (r"\b(?:bottom[\s-]*left|lower[\s-]*left)\b", "bottom-left"),
    (r"\b(?:bottom[\s-]*right|lower[\s-]*right)\b", "bottom-right"),
    (r"\b(?:top[\s-]*left|upper[\s-]*left)\b", "top-left"),
    (r"\b(?:top[\s-]*right|upper[\s-]*right)\b", "top-right"),
    (r"\bcorner\b", "corner"),
    (r"\b(?:bottom|lower)\s+(?:row|edge)\b", "bottom"),
    (r"\b(?:top|upper)\s+(?:row|edge)\b", "top"),
    (r"\bleft\s+(?:column|edge|side)\b", "left"),
    (r"\bright\s+(?:column|edge|side)\b", "right"),
)


def place_named_in(text: str) -> str:
    """The place a condition names, if it names one."""
    said = " ".join(str(text or "").split()).lower()
    for pattern, place in _PLACES:
        if re.search(pattern, said):
            return place
    return ""


def where_it_sits(value: str, reading: str) -> set[str]:
    """Every place a value occupies in a reading laid out in rows.

    A reading that keeps its arrangement can be asked WHERE something is, and
    that is what makes a plan about a corner a plan that can be checked. A
    flattened reading cannot be asked at all.
    """
    wanted = str(value or "").strip()
    rows = [row.split() for row in str(reading or "").splitlines() if row.split()]
    if not wanted or not rows:
        return set()
    places: set[str] = set()
    last_row = len(rows) - 1
    for index, row in enumerate(rows):
        for column, cell in enumerate(row):
            if cell != wanted:
                continue
            last_column = len(row) - 1
            top, bottom = index == 0, index == last_row
            left, right = column == 0, column == last_column
            if top:
                places.add("top")
            if bottom:
                places.add("bottom")
            if left:
                places.add("left")
            if right:
                places.add("right")
            for vertical, horizontal in (("top", "left"), ("top", "right"),
                                         ("bottom", "left"), ("bottom", "right")):
                if {vertical, horizontal} <= places or (
                    (vertical == "top" and top or vertical == "bottom" and bottom)
                    and (horizontal == "left" and left or horizontal == "right" and right)
                ):
                    places.add(f"{vertical}-{horizontal}")
                    places.add("corner")
    return places


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
    # A plan about a place is checked against the place.
    #
    # "Keep the largest tile in the bottom-left corner" was held as "the 64 is
    # still somewhere on the board", which stays true while the 64 wanders
    # into the middle and the plan quietly stops being the plan. Checkable
    # only because the reading keeps its arrangement.
    place = place_named_in(strategy.holds_while.describes or strategy.approach)
    # Only where the reading actually has an arrangement to check against.
    #
    # A place is a fact about a layout. A reading that arrived as one line of
    # prose has no rows, no columns and no corners, and asking where
    # something sits in it produces an answer about nothing.
    laid_out = len([row for row in str(reading or "").splitlines() if row.split()]) > 1
    if place and laid_out:
        for value in strategy.holds_while.contains:
            if value not in str(reading or "").split():
                continue
            # Present, and at no edge, is the middle — which is not the
            # corner the plan is about. An empty set of places used to skip
            # the check, so a plan about a corner stayed "true" while the
            # thing it was about sat in the middle of the board.
            if place not in where_it_sits(value, reading):
                return False, f"the {value} is no longer in the {place}"
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


def _biggest_thing_in(situation: str) -> str:
    """The value a situation is most obviously about.

    An approach often refers to something without naming it: "keep the
    largest tile in the corner", "protect the big one", "build on what I
    have". None of that is checkable on its own, and refusing it means she
    holds no approach at all — measured live, she played whole games without
    one because her plans were phrased the way people phrase plans.

    What she is referring to is in front of her. Binding the approach to the
    largest value in the situation makes it checkable without putting words
    in her mouth: an approach built around the big tile stops being the right
    approach when the big tile is gone.
    """
    values = [
        int(found.replace(",", ""))
        for found in re.findall(r"\b(\d[\d,]{0,9})\b", str(situation or ""))
    ]
    return str(max(values)) if values else ""


#: The fewest words that can describe a way of going about something. Below
#: this it is a move, an answer, or encouragement.
ENOUGH_WORDS = 4


def _says_enough_to_be_an_approach(said: str, options: Sequence[ActionOption] = ()) -> bool:
    """Whether this describes a way of going about it at all."""
    words = re.findall(r"[\w'-]+", str(said or ""))
    if len(words) < ENOUGH_WORDS:
        return False
    names = {str(option.name or "").strip().lower() for option in options if option.name}
    return " ".join(words).lower() not in names


def read_strategy(
    reply: str, options: Sequence[ActionOption] = (), *, situation: str = ""
) -> Strategy | None:
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
        # No lead-in, so the answer itself is the approach.
        #
        # She was asked how she is going about this. Requiring her to open
        # with "Plan:" or "I will" before it counts made an approach depend
        # on a turn of phrase, and the phrasings people use do not end:
        # "I'm going to stack toward the left edge" says exactly what "I will
        # stack toward the left edge" says. What makes it an approach is that
        # it names something to hold to, which is checked below either way.
        approach = text[:200].strip(" ,;")
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
        # She said what she is doing without saying what would end it.
        #
        # Requiring both in one sentence made having an approach depend on
        # phrasing one — and a plan that only exists when the words come out
        # in a particular shape is a plan she mostly does not have. Measured
        # live: playing a whole game, she never once held an approach.
        #
        # What she named is what the approach depends on. An approach built
        # around a 64 in the corner stops being the right approach when the
        # 64 is gone, and that is checkable without her having said the word
        # "while". Only where she named something concrete — otherwise there
        # is genuinely nothing to watch, and a plan with nothing that could
        # end it is a preference.
        condition = approach

    keep, avoid = _watchable(condition)
    if not keep and not avoid:
        # She named no value, so bind it to what she is looking at — but only
        # if she actually described an approach. A bare option name is a
        # move, and two words of encouragement is a preference; neither is a
        # line to take, and anchoring them to the board would dress them up
        # as one.
        if not _says_enough_to_be_an_approach(approach, options):
            return None
        anchor = _biggest_thing_in(situation)
        if not anchor:
            return None
        keep, avoid = (anchor,), ()
        condition = f"{condition} (while the {anchor} is still there)"
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


#: Whether the last attempt to form an approach already went unrecorded, so a
#: question asked on nearly every cycle cannot fill the ledger on its own.
_said_it_could_not_plan: dict[str, bool] = {"value": False}


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
        logger.info("asked how to go about %r with no mind to ask", str(goal or "")[:60])
        return None
    logger.info("asking how to go about %r", str(goal or "")[:60])
    evidence = [f"What is visible now: {situation}", *knowledge]
    if previous is not None:
        evidence.append(f"The approach I was taking — {previous.approach}")
        evidence.append("It stopped being right, which is why this is being decided again.")
    for attempt in list(history)[-3:]:
        if hasattr(attempt, "as_evidence"):
            evidence.append(attempt.as_evidence())
    try:
        reply = await think(_asking_for_an_approach(goal, situation, options), evidence)
    except asyncio.CancelledError:
        # Her own deadline, or ours.
        #
        # A thought that ran out of time arrives here the same way a task
        # being torn down does, and the two must not be treated alike: the
        # first is an ordinary answer of "not this time", the second is a
        # shutdown that has to keep travelling. The current task knows which
        # it is, because only a real cancellation is recorded against it.
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        logger.info("her thinking ran out of time before it named an approach")
        return None
    except Exception as exc:  # noqa: BLE001 - see below
        # Not having a plan this time is an ordinary outcome, not damage.
        #
        # Recorded once and then left alone. Measured live: twenty of these
        # in half an hour opened a runtime integrity incident, because a
        # question she asks on almost every cycle was reporting each
        # unanswered attempt as a subsystem degradation. The condition worth
        # recording is that she keeps failing to form a plan, and one entry
        # says that as well as twenty do.
        # Caught wide on purpose. A narrow tuple here read as care and was
        # blindness: anything outside it left this function without touching
        # a single line of logging, and the caller — which must survive a
        # thought that fails — swallowed it. LIVE 2026-08-26: twenty-five
        # approach questions asked, not one answer, not one word about why.
        logger.info("could not settle on an approach: %s", exc.__class__.__name__)
        if not _said_it_could_not_plan["value"]:
            _said_it_could_not_plan["value"] = True
            record_degradation(
                "standing_strategy",
                exc,
                severity="info",
                action="carried on without a stated approach",
            )
        return None
    _said_it_could_not_plan["value"] = False
    logger.info(
        "she answered how to go about it (%d chars): %r",
        len(str(reply or "")),
        " ".join(str(reply or "").split())[:160],
    )
    settled = read_strategy(reply or "", options, situation=situation)
    if settled is None:
        # She answered, and what she said was not an approach.
        #
        # The only silent exit left in this organ. Everything else records
        # why, and this one — the one that actually happens — left nothing,
        # so "she never states a plan" could not be told apart from "she is
        # never asked". LIVE 2026-08-26: whole games with no approach held
        # and no trace of why.
        logger.info(
            "no approach in her answer (%d chars): %r",
            len(str(reply or "")),
            " ".join(str(reply or "").split())[:200],
        )
        return None
    return Strategy(
        approach=settled.approach,
        because=settled.because,
        holds_while=settled.holds_while,
        otherwise=settled.otherwise,
        adopted_on_move=int(moves_made),
    )
